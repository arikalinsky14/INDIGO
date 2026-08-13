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
    EOS_TOKEN, M_MAX, MAX_LAYERS, MAX_THICKNESS_NM,
    MIN_THICKNESS_NM, NUM_THICKNESSES, VOCAB_SIZE,
    build_output_mask_batch, normalize_lab,
    normalize_thickness,
)
from src.model import ModelConfig, build_model

from inference.src.constraints import (
    ConstraintSet, FinishedStructure, PartialStructure,
)
from inference.src.schema import (
    Candidate, EnsembleStats, MaterialEntry, RobustnessReport,
)


# ---------------------------------------------------------------------------
# Slot mask projection.
#
# The constraint system in inference/src/constraints.py was designed for
# the old joint (slot × thickness) vocabulary of size M_MAX*NUM_THICKNESSES + 1.
# The two-head model votes only on the M_MAX + 1 slot vocab. Until the
# constraint layer is ported, we project joint-mask bytes to slot-mask
# bytes with the following contract:
#   - a SLOT is allowed if ANY of its thickness bins in the joint mask
#     were allowed (i.e. the user hasn't ruled the material out entirely).
#   - EOS carries over unchanged.
# Pure thickness constraints (thickness_range, total_thickness) become
# post-hoc filters — see select.py. Slot-only constraints (only Ag, no
# metals after position 3, symmetry, …) keep working losslessly.
# ---------------------------------------------------------------------------

def _project_joint_mask_to_slot_mask(joint_mask: np.ndarray) -> np.ndarray:
    """[M_MAX * NUM_THICKNESSES + 1] bool → [M_MAX + 1] bool."""
    slot_mask = np.zeros(M_MAX + 1, dtype=bool)
    for s in range(M_MAX):
        lo = s * NUM_THICKNESSES
        hi = lo + NUM_THICKNESSES
        slot_mask[s] = bool(joint_mask[lo:hi].any())
    # Old EOS lived at index M_MAX * NUM_THICKNESSES; new one at M_MAX.
    slot_mask[M_MAX] = bool(joint_mask[M_MAX * NUM_THICKNESSES])
    return slot_mask


def _project_joint_boost_to_slot_boost(joint_boost: np.ndarray) -> np.ndarray:
    """[M_MAX * NUM_THICKNESSES + 1] float → [M_MAX + 1] float (max over thicks)."""
    slot_boost = np.zeros(M_MAX + 1, dtype=np.float32)
    for s in range(M_MAX):
        lo = s * NUM_THICKNESSES
        hi = lo + NUM_THICKNESSES
        slot_boost[s] = float(joint_boost[lo:hi].max())
    slot_boost[M_MAX] = float(joint_boost[M_MAX * NUM_THICKNESSES])
    return slot_boost


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

    # Pre-compute the slot-validity mask (model-side) for this pool over
    # the NEW [M_MAX + 1] slot vocab.
    pool_size_t = torch.full((1,), pool_size_int, dtype=torch.long, device=device)
    slot_validity = build_output_mask_batch(
        pool_size_t, device=device,
    ).cpu().numpy()[0]                     # [M_MAX + 1]
    slot_validity_bool = np.isfinite(slot_validity)

    # Lab target in model's normalised convention.
    lab_norm = torch.tensor(
        list(normalize_lab(list(target_lab_raw))), dtype=torch.float32, device=device,
    )

    lab_one = lab_norm.unsqueeze(0)
    pool_feats_one = pool_feats_one.to(device)
    pool_mask_one = pool_mask_one.to(device)

    generators: List[torch.Generator] = []
    for r in range(N):
        g = torch.Generator(device=device)
        g.manual_seed(_replica_seed(cfg.base_seed, r))
        generators.append(g)

    slots_by_replica: List[List[int]] = [[] for _ in range(N)]
    thick_by_replica: List[List[float]] = [[] for _ in range(N)]
    done = np.zeros(N, dtype=bool)

    NEW_EOS = M_MAX  # slot-vocab EOS after the transition

    # Decoding loop over the two-head model.
    for step in range(MAX_LAYERS + 1):
        if done.all():
            break

        struct_np = np.zeros((N, M_MAX, MAX_LAYERS), dtype=np.float32)
        for r in range(N):
            for layer_idx, (s, t) in enumerate(zip(slots_by_replica[r],
                                                   thick_by_replica[r])):
                struct_np[r, s, layer_idx] = normalize_thickness(t)
        struct_t = torch.from_numpy(struct_np).to(device)

        all_slot_logits = torch.empty(
            (N, M_MAX + 1), dtype=torch.float32, device=device,
        )
        all_thickness_nm = torch.empty(
            (N, M_MAX), dtype=torch.float32, device=device,
        )
        for chunk_lo, chunk_hi in chunks:
            cs = chunk_hi - chunk_lo
            out = model(
                lab=lab_one.expand(cs, -1),
                pool_features=pool_feats_one.expand(cs, -1, -1, -1),
                pool_mask=pool_mask_one.expand(cs, -1),
                structure_matrix=struct_t[chunk_lo:chunk_hi],
                pool_size=pool_size_t.expand(cs),
            )
            slot_logits = out["slot_logits"]           # [cs, ..., M_MAX + 1]
            thick_nm = out["thickness_nm"]             # [cs, ..., M_MAX]
            if slot_logits.dim() == 3:
                slot_logits = slot_logits[:, step, :]
                thick_nm = thick_nm[:, step, :]
            all_slot_logits[chunk_lo:chunk_hi] = slot_logits
            all_thickness_nm[chunk_lo:chunk_hi] = thick_nm

        step_slot_logits = all_slot_logits.detach().to(torch.float32).cpu().numpy()
        step_thick_nm = all_thickness_nm.detach().to(torch.float32).cpu().numpy()

        for r in range(N):
            if done[r]:
                continue
            partial = PartialStructure(
                slots_so_far=slots_by_replica[r],
                thicknesses_so_far=thick_by_replica[r],
                pool_size=pool_size_int,
            )
            # Slot-mask projection from the joint-vocab constraint machinery.
            # Pure thickness constraints degrade to post-hoc filtering here.
            joint_mask = constraint_set.decode_mask(partial, pool)
            slot_mask = _project_joint_mask_to_slot_mask(joint_mask)
            allowed = slot_validity_bool & slot_mask

            # No two adjacent layers may share a slot.
            if slots_by_replica[r]:
                prev_slot = slots_by_replica[r][-1]
                allowed[prev_slot] = False

            row = step_slot_logits[r].copy()

            joint_boost = constraint_set.decode_boost(partial, pool)
            if joint_boost is not None:
                row = row + _project_joint_boost_to_slot_boost(joint_boost)

            if not allowed.any():
                slot_id = NEW_EOS
            else:
                row[~allowed] = -np.inf
                slot_id = _sample_slot(row, cfg.temperature, generators[r], device)

            if slot_id == NEW_EOS:
                done[r] = True
                continue

            thickness_nm = float(step_thick_nm[r, slot_id])
            thickness_nm = max(MIN_THICKNESS_NM,
                               min(MAX_THICKNESS_NM, thickness_nm))
            slots_by_replica[r].append(int(slot_id))
            thick_by_replica[r].append(thickness_nm)
            if len(slots_by_replica[r]) >= MAX_LAYERS:
                done[r] = True

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

def _sample_slot(logits_row: np.ndarray, temperature: float,
                 generator: torch.Generator, device: torch.device) -> int:
    """Sample one slot id ∈ [0, M_MAX] from a masked slot-logits row.

    The row is size M_MAX + 1 (slot vocab + EOS at index M_MAX). We fall
    back to EOS whenever every row is masked out or the softmax degenerates.
    """
    NEW_EOS = M_MAX
    if not np.any(np.isfinite(logits_row)):
        return NEW_EOS
    if temperature <= 0:
        return int(np.argmax(logits_row))
    scaled = logits_row / max(temperature, 1e-6)
    finite_max = np.max(scaled[np.isfinite(scaled)])
    exp = np.exp(scaled - finite_max)
    exp[~np.isfinite(scaled)] = 0.0
    Z = exp.sum()
    if Z <= 0 or not np.isfinite(Z):
        return NEW_EOS
    probs = exp / Z
    u = torch.rand((), generator=generator, device=device).item()
    cum = np.cumsum(probs)
    return int(np.searchsorted(cum, u))


# Backward-compat alias — external callers may still import _sample_token.
_sample_token = _sample_slot


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
    thick_by_replica: List[List[float]],
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

    # One-hot slot row → that slot always wins.
    row = np.full(M_MAX + 1, -np.inf, dtype=np.float32)
    row[5] = 0.0
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    tok = _sample_slot(row, temperature=1.0, generator=g, device=torch.device("cpu"))
    assert tok == 5

    # All-inf row → falls back to slot-vocab EOS (= M_MAX).
    row[:] = -np.inf
    tok = _sample_slot(row, temperature=1.0, generator=g, device=torch.device("cpu"))
    assert tok == M_MAX

    # Dedup: two replicas land identical, one differs. Continuous thicknesses.
    slots = [[0, 1], [0, 1], [1, 0]]
    thicks = [[50.5, 100.25], [50.5, 100.25], [50.5, 100.25]]
    cands, n_unique = _build_candidates(slots, thicks, pool, (50.0, 0.0, 0.0))
    assert n_unique == 2
    m_count = sum(c.multiplicity for c in cands)
    assert m_count == 3
    print("[generate] smoke: pack_pool / _sample_slot / dedup OK")


if __name__ == "__main__":
    _smoke_test()
