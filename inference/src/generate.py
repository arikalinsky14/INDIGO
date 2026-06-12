"""
Batched constrained ensemble decoder for INDIGO inference.

  pool + InferenceSpec   ─►   N candidate structures (slots + thicknesses)

The single (lab target, material pool) input is broadcast across N model
replicas — one forward pass per decoding step covers all of them. At each
step we (a) read the model's per-position logits, (b) AND them with the
slot-validity mask the model already produces and any active constraint
decode-masks, and (c) sample from the masked distribution at the user's
temperature.

Determinism
-----------
Per-replica `torch.Generator`s seeded from `base_seed * 7919 + replica_idx`
make each replica reproducible. Same base seed + same pool + same model
checkpoint ⇒ bit-identical ensemble.

Memory budget (the user asked for this to be conservative)
----------------------------------------------------------
At the default N=500, batch_size=500, d_model=1024, M_MAX=32, MAX_LAYERS=10:
  - pool_features tile           ~16 MB    (500·32·2·128·4 bytes)
  - structure_matrix tile        ~0.6 MB   (500·32·10·4 bytes)
  - slot-encoder activations     ~200 MB peak (n_layers·B·M_MAX·d_model)
  - decoder activations          ~150 MB peak
  - Model parameters (frozen)    ~280 MB   (70 M params · float32)
                                   ─────
                                  ~650 MB
Comfortably under an L40s, leaves headroom for activation memory under
torch.no_grad. If you need to crank N higher than 1000, switch to the
`chunk_size` knob below so the model batch stays bounded.

The optical simulation downstream of decoding is the real cost driver
(sequential per-candidate due to the JLL/JIT issue documented in
simulate.py). Decoding itself completes in well under a second.

What this module deliberately does NOT do
-----------------------------------------
- No optical sim, no Lab computation, no ΔE — that's simulate.py.
- No selection / ranking — that's select.py.
- No refinement — that's refine.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.material_features import featurize_pool, pad_pool_features
from src.materials_vocab import (
    EOS_TOKEN, M_MAX, MAX_LAYERS, NUM_THICKNESSES, VOCAB_SIZE,
    build_output_mask_batch, decode_token, normalize_lab,
)
from src.model import ModelConfig, build_model

from inference.src.constraints import (
    ConstraintSet, FinishedStructure, PartialStructure,
)
from inference.src.schema import (
    Candidate, EnsembleStats, MaterialEntry, RobustnessReport,
)


# ----------------------------------------------------------------------------
# Lightweight value objects
# ----------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Knobs that affect *only* sampling — not the rest of inference."""
    ensemble_N: int = 500
    temperature: float = 1.0
    base_seed: int = 42
    # Cap on how many replicas the model sees at once. The default = N so the
    # whole ensemble rides one forward pass. Bump down (e.g. 128) to spread
    # memory if you scale ensemble_N past ~1000 or move to a smaller GPU.
    chunk_size: Optional[int] = None
    # If True, after decoding emit Candidate objects with normalised Lab and
    # placeholder reflectance/ΔE fields — the sim/select pass fills those in.
    emit_candidates: bool = True


# ----------------------------------------------------------------------------
# Pool packing helpers
# ----------------------------------------------------------------------------

def _materialentry_to_materialnk(entry: MaterialEntry):
    """Bridge MaterialEntry (inference schema) → MaterialNK (model side)."""
    from src.material_features import MaterialNK
    return MaterialNK(name=entry.canonical_name, n=entry.n, k=entry.k,
                      source=entry.source)


def pack_pool(pool: List[MaterialEntry]
              ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """List[MaterialEntry] → padded (pool_features, pool_mask, pool_size).

    Outputs match the model's input contract. Done once per inference call.
    """
    if len(pool) == 0:
        raise ValueError("pool is empty")
    if len(pool) > M_MAX:
        raise ValueError(f"pool of {len(pool)} exceeds M_MAX={M_MAX}")
    mnk = [_materialentry_to_materialnk(m) for m in pool]
    feats = featurize_pool(mnk, mode="raw_spectrum")
    feats_pad, mask = pad_pool_features(feats, m_max=M_MAX)
    return feats_pad, mask, len(pool)


# ----------------------------------------------------------------------------
# The decoder itself
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_ensemble(
    model: torch.nn.Module,
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
    constraint_set: ConstraintSet,
    cfg: GenerationConfig,
    device: Optional[torch.device] = None,
) -> Tuple[List[Candidate], EnsembleStats]:
    """Sample `cfg.ensemble_N` constrained structures for one (lab, pool).

    Returns
    -------
    candidates : list of Candidate
        Each carries `slot_indices`, `material_names`, `thicknesses_nm`,
        `multiplicity`, `sample_index`. Lab / reflectance / ΔE remain
        placeholders here — simulate.py / select.py fill them in.
    stats : EnsembleStats
        Bookkeeping for the GUI report (n_sampled, n_unique_after_dedup,
        and a per-constraint drop count, which we record at 0 here because
        decode-time masking means feasibility-by-construction; select.py
        adds post-hoc drops on top).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # Pack pool once.
    pool_feats_one, pool_mask_one, pool_size_int = pack_pool(pool)

    N = cfg.ensemble_N
    chunks = _plan_chunks(N, cfg.chunk_size)

    # Pre-compute the slot-validity mask (model-side) for this pool.
    pool_size_t = torch.full((1,), pool_size_int, dtype=torch.long, device=device)
    slot_validity = build_output_mask_batch(pool_size_t, device=device).cpu().numpy()[0]
    slot_validity_bool = np.isfinite(slot_validity)        # [VOCAB_SIZE] bool

    # Lab target in model's normalised convention.
    lab_norm = torch.tensor(
        list(normalize_lab(list(target_lab_raw))), dtype=torch.float32, device=device,
    )

    # Pre-build constant slices we'll broadcast across the ensemble batch.
    lab_one = lab_norm.unsqueeze(0)               # [1, 3]
    pool_feats_one = pool_feats_one.to(device)
    pool_mask_one = pool_mask_one.to(device)

    # Per-replica RNGs — deterministic, independent of host RNG.
    generators: List[torch.Generator] = []
    for r in range(N):
        g = torch.Generator(device=device)
        g.manual_seed(_replica_seed(cfg.base_seed, r))
        generators.append(g)

    # Per-replica running state, kept on CPU as plain Python objects so it
    # stays cheap for any N. The mid-decode structure_matrix tile *is* on
    # the device; we rebuild it each step from these lists.
    slots_by_replica: List[List[int]] = [[] for _ in range(N)]
    thick_by_replica: List[List[int]] = [[] for _ in range(N)]
    done = np.zeros(N, dtype=bool)

    # Decoding loop. The cross_attn model returns
    #   logits[B, MAX_LAYERS+1, VOCAB_SIZE]
    # and we read position `step` at each iteration. With the model's causal
    # mask + the seq-key padding mask derived from structure_matrix, position
    # k only sees layers 0..k-1 — i.e. the prefix actually placed so far.
    for step in range(MAX_LAYERS + 1):
        if done.all():
            break
        # Build the structure_matrix tile [N, M_MAX, MAX_LAYERS] from the
        # running lists. Cheap (N·MAX_LAYERS writes); avoids holding the full
        # tile on device when most positions are empty.
        struct_np = np.zeros((N, M_MAX, MAX_LAYERS), dtype=np.float32)
        for r in range(N):
            for layer_idx, (s, t) in enumerate(zip(slots_by_replica[r],
                                                   thick_by_replica[r])):
                struct_np[r, s, layer_idx] = t / 200.0  # normalize_thickness
        struct_t = torch.from_numpy(struct_np).to(device)

        # Run model in chunks if requested, otherwise the whole N at once.
        all_step_logits = torch.empty((N, VOCAB_SIZE),
                                      dtype=torch.float32, device=device)
        for chunk_lo, chunk_hi in chunks:
            cs = chunk_hi - chunk_lo
            logits = model(
                lab=lab_one.expand(cs, -1),
                pool_features=pool_feats_one.expand(cs, -1, -1, -1),
                pool_mask=pool_mask_one.expand(cs, -1),
                structure_matrix=struct_t[chunk_lo:chunk_hi],
                pool_size=pool_size_t.expand(cs),
            )
            # cross_attn → [cs, MAX_LAYERS+1, V]; mlp would be [cs, V]. The
            # ensemble decoder targets cross_attn (the head we trained for
            # production), but we keep the mlp path working for cheap.
            if logits.dim() == 3:
                logits = logits[:, step, :]
            all_step_logits[chunk_lo:chunk_hi] = logits

        # Convert to a numpy float32 work copy: applying per-replica
        # constraint masks is fastest on CPU because the masks vary per replica.
        step_logits = all_step_logits.detach().to(torch.float32).cpu().numpy()

        # Sample one token per replica.
        for r in range(N):
            if done[r]:
                continue
            partial = PartialStructure(
                slots_so_far=slots_by_replica[r],
                thicknesses_so_far=thick_by_replica[r],
                pool_size=pool_size_int,
            )
            constraint_mask = constraint_set.decode_mask(partial, pool)
            allowed = slot_validity_bool & constraint_mask

            # Universal: no two adjacent layers may share a slot. Two
            # consecutive layers of the same material would just be one
            # thicker layer of that material and would game the layer_count
            # constraints. Applied AFTER the constraint mask so Symmetry's
            # boost cannot accidentally suggest a same-slot repeat.
            if slots_by_replica[r]:
                prev_slot = slots_by_replica[r][-1]
                lo = prev_slot * NUM_THICKNESSES
                hi = lo + NUM_THICKNESSES
                allowed[lo:hi] = False

            row = step_logits[r].copy()

            # Apply any active logit boosts BEFORE masking out the disallowed
            # tokens — masking sets disallowed positions to -inf which is
            # idempotent under further additions, so boost order doesn't
            # matter for blocked positions.
            boost = constraint_set.decode_boost(partial, pool)
            if boost is not None:
                row = row + boost

            if not allowed.any():
                # Infeasible step — force EOS so this replica terminates
                # gracefully. The post-hoc constraint check will decide
                # whether the resulting (possibly invalid) structure gets
                # dropped. We never raise here — that's the contract.
                token_id = EOS_TOKEN
            else:
                row[~allowed] = -np.inf
                token_id = _sample_token(row, cfg.temperature, generators[r], device)

            if token_id == EOS_TOKEN:
                done[r] = True
                continue
            # Decode (slot, thickness) and append.
            _, thickness_nm, slot_idx, _, kind = decode_token(token_id, pool=None)
            assert kind == "LAYER"
            slots_by_replica[r].append(int(slot_idx))
            thick_by_replica[r].append(int(thickness_nm))
            # MAX_LAYERS reached without an EOS → also done (MAX_LEN).
            if len(slots_by_replica[r]) >= MAX_LAYERS:
                done[r] = True

    # Build Candidate objects and dedup by exact-equality on (slots, thicks).
    candidates, n_unique = _build_candidates(
        slots_by_replica, thick_by_replica, pool, target_lab_raw,
    )

    stats = EnsembleStats(
        n_sampled=N,
        n_unique_after_dedup=n_unique,
        n_feasible=len(candidates),       # decode-time masking ⇒ all feasible
        n_refined=0,
        n_returned=0,
        dropped_per_constraint={},
    )
    return candidates, stats


# ----------------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------------

def _sample_token(logits_row: np.ndarray, temperature: float,
                  generator: torch.Generator, device: torch.device) -> int:
    """Sample one token id given a masked logits row.

    Performed on a tiny [VOCAB_SIZE] vector; doing it on CPU via numpy after
    the mask AND saves a host↔device round-trip per replica per step.
    """
    if not np.any(np.isfinite(logits_row)):
        return EOS_TOKEN
    if temperature <= 0:
        # Greedy fallback — used when knobs.temperature is 0 (rare).
        return int(np.argmax(logits_row))
    scaled = logits_row / max(temperature, 1e-6)
    # Stable softmax with the -inf masked rows handled correctly.
    finite_max = np.max(scaled[np.isfinite(scaled)])
    exp = np.exp(scaled - finite_max)
    exp[~np.isfinite(scaled)] = 0.0
    Z = exp.sum()
    if Z <= 0 or not np.isfinite(Z):
        return EOS_TOKEN
    probs = exp / Z
    # Use the torch generator so we stay reproducible across runs even though
    # the actual sample is a numpy operation behind a torch RNG draw.
    u = torch.rand((), generator=generator, device=device).item()
    cum = np.cumsum(probs)
    return int(np.searchsorted(cum, u))


def _plan_chunks(N: int, chunk_size: Optional[int]
                 ) -> List[Tuple[int, int]]:
    """Build [(lo, hi), ...] over [0, N), capped by chunk_size if given."""
    if chunk_size is None or chunk_size >= N:
        return [(0, N)]
    chunks: List[Tuple[int, int]] = []
    for lo in range(0, N, chunk_size):
        chunks.append((lo, min(lo + chunk_size, N)))
    return chunks


def _replica_seed(base: int, replica_idx: int) -> int:
    # Multiplicative prime offset → no two replica seeds collide as long as
    # base seeds avoid pathological mod-7919 alignments.
    return (base * 7919 + replica_idx) & 0xFFFF_FFFF


# ----------------------------------------------------------------------------
# Candidate assembly & dedup
# ----------------------------------------------------------------------------

def _build_candidates(
    slots_by_replica: List[List[int]],
    thick_by_replica: List[List[int]],
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
) -> Tuple[List[Candidate], int]:
    """Collapse duplicate (slots, thicknesses) into one Candidate with
    multiplicity. Lab / reflectance / ΔE stay placeholder NaN — filled by
    simulate.py.
    """
    # Key on exact tuples. The plan suggests ±1-grid-step fuzzy dedup, but
    # for v1 we keep it strict — the sampler's temperature already produces
    # plenty of duplicates and strict dedup is least surprising.
    keep: dict = {}
    for r, (slots, thicks) in enumerate(zip(slots_by_replica, thick_by_replica)):
        if len(slots) == 0:
            continue  # zero-layer structures are uninteresting
        key = (tuple(slots), tuple(thicks))
        if key not in keep:
            keep[key] = {
                "slots": list(slots),
                "thicks": list(thicks),
                "sample_index": r,
                "multiplicity": 1,
            }
        else:
            keep[key]["multiplicity"] += 1
    out: List[Candidate] = []
    nan_lab = (float("nan"), float("nan"), float("nan"))
    for rec in keep.values():
        names = [pool[s].canonical_name for s in rec["slots"]]
        out.append(Candidate(
            slot_indices=rec["slots"],
            material_names=names,
            thicknesses_nm=[float(t) for t in rec["thicks"]],
            achieved_lab=nan_lab,
            reflectance=[float("nan")] * 128,
            delta_e=float("nan"),
            robustness=RobustnessReport(),
            objective=float("nan"),
            refined=False,
            refine_iters=0,
            sample_index=rec["sample_index"],
            multiplicity=rec["multiplicity"],
        ))
    return out, len(out)


# ----------------------------------------------------------------------------
# Model-loading helper (the orchestrator will use this; exposed for tests)
# ----------------------------------------------------------------------------

def load_inference_model(checkpoint_dir: Path,
                         device: Optional[torch.device] = None,
                         ) -> Tuple[torch.nn.Module, ModelConfig, str]:
    """Load model.pt + config.json from a saved checkpoint dir.

    Returns (model in eval, ModelConfig, sha256_first_16). The hash feeds
    `Provenance.model_sha256` so every Result names a specific weights file.
    """
    import hashlib
    import json
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    model = build_model(config)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    return model, config, sha


# ----------------------------------------------------------------------------
# Smoke test (no model required) — exercises plumbing only
# ----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Exercise pack_pool, _sample_token, _build_candidates, dedup, RNG.

    Run with `python -m inference.src.generate`.
    """
    pool = [
        MaterialEntry(canonical_name="Ag", n=np.ones(128, dtype=np.float32),
                      k=np.zeros(128, dtype=np.float32)),
        MaterialEntry(canonical_name="SiO2", n=np.ones(128, dtype=np.float32) * 1.45,
                      k=np.zeros(128, dtype=np.float32)),
    ]
    feats, mask, ps = pack_pool(pool)
    assert feats.shape == (M_MAX, 2, 128)
    assert mask.dtype == torch.bool and mask.shape == (M_MAX,)
    assert int(mask.sum().item()) == 2 and ps == 2

    # Sampling with a one-hot logits row → that token always wins.
    row = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
    row[10] = 0.0
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    tok = _sample_token(row, temperature=1.0, generator=g, device=torch.device("cpu"))
    assert tok == 10

    # All-inf row → falls back to EOS.
    row[:] = -np.inf
    tok = _sample_token(row, temperature=1.0, generator=g, device=torch.device("cpu"))
    assert tok == EOS_TOKEN

    # Dedup: two replicas land identical, one differs.
    slots = [[0, 1], [0, 1], [1, 0]]
    thicks = [[50, 100], [50, 100], [50, 100]]
    cands, n_unique = _build_candidates(slots, thicks, pool, (50, 0, 0))
    assert n_unique == 2
    m_count = sum(c.multiplicity for c in cands)
    assert m_count == 3
    print("[generate] smoke: pack_pool / _sample_token / dedup OK")


if __name__ == "__main__":
    _smoke_test()
