#!/usr/bin/env python3
"""
STE Projection Quality Test
===========================

The ΔE finetune's slot STE (see src/de_finetune.py:ste_pick) computes
gradients wrt every pool material's slot logit at position k by
*linearly projecting* the sim's local gradient at the argmax material's
n/k onto each other material's spectrum:

    grad_slot_logit[i] ≈ ⟨∂Lab/∂n_k, pool_n[i]⟩ + ⟨∂Lab/∂k_k, pool_k[i]⟩

This is a first-order Taylor expansion. It's cheap (one sim per
position, not 32), but only *locally* correct — for a material whose
n/k differs a lot from the base's, the projection can point in the
wrong direction.

The linearization anchor matters. Two modes to test:

    --base-mode gt (default)
        Base = GT slot at position k. sim(GT_prefix, GT_k, GT_suffix) =
        target Lab, so base ΔE = 0. Measures projection quality at the
        optimum.

    --base-mode model_argmax
        Loads the pretrained checkpoint, runs forward with GT
        structure_matrix, uses model's argmax slot + argmax thickness
        at position k. sim(GT_prefix, argmax_pick, GT_suffix) ≠ target
        in general, so base ΔE > 0. Measures projection quality at
        *training-time* anchors — where the finetune's STE gradient
        is actually evaluated on every step.

The first sweep of the finetune (Sept 8 2026) degraded val ΔE at every
non-trivial LR while slot_match_gt dropped simultaneously — evidence
that the training-time gradient direction is systematically wrong.
--base-mode model_argmax exists to test whether the linearization
accuracy is much worse at those non-optimal anchor points than the
first gt-mode test suggested.

Metrics per (example, layer)
----------------------------
    - Spearman ρ  : rank correlation of projected vs true post-swap ΔE
    - top1_hit    : did the projection's argmin match the true argmin?
    - top3_recall : is the true best material in the projection's top-3?
    - sign_agree  : per material, does sign(ΔE_proj − ΔE_base) match
                    sign(ΔE_true − ΔE_base)?  ← average across materials
    - proj_err_ΔE : |ΔE_proj_i − ΔE_true_i|      ← average across materials

Verdict (rule-of-thumb, spelled out in the summary)
---------------------------------------------------
    HOLDS   : median Spearman ≥ 0.70  AND  mean top1_hit ≥ 0.50
    MIXED   : anything in between
    NOISE   : median Spearman < 0.30  AND  mean top1_hit < 0.20

Outputs
-------
    <output-dir>/results.json          per-example raw + agg metrics
    <output-dir>/summary.txt           human-readable
    <output-dir>/projected_vs_true.png scatter over all (example, layer, i)
    <output-dir>/spearman_hist.png     per-(example, layer) ρ histogram
    <output-dir>/sign_agreement_hist.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Repo root on path
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import FlexThinFilmDataset, find_repo_root
from src.materials_vocab import (
    M_MAX,
    MAX_LAYERS,
    NUM_THICKNESSES,
    THICKNESSES,
    build_structure_matrix,
    denormalize_lab,
)
from src.optical_sim_diff import (
    _JAX_FORWARD,
    is_available as sim_is_available,
)

# ciede2000 lives in scripts/evaluate.py — import path-hackily.
sys.path.insert(0, str(_repo_root / "scripts"))
from evaluate import ciede2000  # noqa: E402


if not sim_is_available():
    raise RuntimeError(
        "jaxlayerlumos not available — cannot run projection quality test."
    )
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# ============================================================================
# Per-example measurement
# ============================================================================


@dataclass
class LayerResult:
    example_idx: int
    layer_k: int
    n_layers: int
    base_slot: int
    base_thickness_nm: float
    gt_slot: int
    gt_thickness_nm: float
    pool_size: int
    dE_base: float
    projected_dE: List[float]        # length pool_size
    true_dE: List[float]             # length pool_size
    spearman: float
    top1_hit: int                    # 0 / 1
    top3_recall: int                 # 0 / 1
    sign_agree_rate: float           # in [0, 1]
    mean_abs_proj_err: float


def _measure_one_layer(
    example_idx: int,
    layer_k: int,
    gt_slots: List[int],
    gt_thicknesses: List[int],
    pool_n_full: np.ndarray,          # [pool_size, NUM_LAMBDA]
    pool_k_full: np.ndarray,          # [pool_size, NUM_LAMBDA]
    target_lab: np.ndarray,           # [3] denormalised
    incidence_angle: float,
    base_slot: int,                   # anchor slot at position k
    base_thickness_nm: float,         # anchor thickness at position k
) -> LayerResult:
    """Compute projection metrics at one (example, k) pair.

    The stack fed to the sim is GT_prefix + (base_slot, base_thickness_nm)
    at position k + GT_suffix — exactly the shape training's rollout
    uses (STE-picked layer k, teacher-forced everything else). The
    linearization is anchored at (pool_n[base_slot], pool_k[base_slot]);
    the projection is evaluated against a swap of that layer's material
    to each candidate i in the pool while keeping base_thickness_nm
    fixed at position k.
    """
    pool_size = pool_n_full.shape[0]
    n_layers = len(gt_slots)

    # Assemble the stack: GT for every layer, then override layer k with
    # the STE anchor's (base_slot, base_thickness).
    n_stack_base = np.stack(
        [pool_n_full[s] for s in gt_slots], axis=0,
    )                                       # [n_layers, NUM_LAMBDA]
    k_stack_base = np.stack(
        [pool_k_full[s] for s in gt_slots], axis=0,
    )
    thicknesses_np = np.asarray(gt_thicknesses, dtype=np.float64)
    thicknesses_np[layer_k] = float(base_thickness_nm)

    n_jax = jnp.asarray(n_stack_base)
    k_jax = jnp.asarray(k_stack_base)
    t_jax = jnp.asarray(thicknesses_np)

    # Build a function of layer-k's n/k only. Other layers stay pinned.
    def sim_wrt_layer_k(n_k, k_k):
        n_full = n_jax.at[layer_k].set(n_k)
        k_full = k_jax.at[layer_k].set(k_k)
        return _JAX_FORWARD(n_full, k_full, t_jax, incidence_angle)

    n_k_base = jnp.asarray(pool_n_full[base_slot])
    k_k_base = jnp.asarray(pool_k_full[base_slot])

    # Jacobians ∂Lab/∂(n_k, k_k) at the base point, shape [3, NUM_LAMBDA].
    lab_base_jax = sim_wrt_layer_k(n_k_base, k_k_base)
    lab_base = np.asarray(lab_base_jax, dtype=np.float64)
    jac_fn = jax.jacrev(sim_wrt_layer_k, argnums=(0, 1))
    jac_n_jax, jac_k_jax = jac_fn(n_k_base, k_k_base)
    jac_n = np.asarray(jac_n_jax, dtype=np.float64)      # [3, NUM_LAMBDA]
    jac_k = np.asarray(jac_k_jax, dtype=np.float64)

    dE_base = float(ciede2000(target_lab.tolist(), lab_base.tolist()))

    projected_dE = np.zeros(pool_size, dtype=np.float64)
    true_dE = np.zeros(pool_size, dtype=np.float64)

    pool_n_base_np = pool_n_full[base_slot]
    pool_k_base_np = pool_k_full[base_slot]

    for i in range(pool_size):
        # Projected: linear extrapolation from base along (pool[i] - pool[base])
        delta_n = pool_n_full[i] - pool_n_base_np
        delta_k = pool_k_full[i] - pool_k_base_np
        delta_lab = jac_n @ delta_n + jac_k @ delta_k          # [3]
        lab_proj = lab_base + delta_lab
        projected_dE[i] = ciede2000(target_lab.tolist(), lab_proj.tolist())

        # True: actually swap material i in at layer k and re-sim.
        n_k_i = jnp.asarray(pool_n_full[i])
        k_k_i = jnp.asarray(pool_k_full[i])
        lab_true_jax = sim_wrt_layer_k(n_k_i, k_k_i)
        lab_true = np.asarray(lab_true_jax, dtype=np.float64)
        true_dE[i] = ciede2000(target_lab.tolist(), lab_true.tolist())

    # Metrics.
    if pool_size >= 2:
        try:
            spearman = float(
                stats.spearmanr(projected_dE, true_dE).correlation
            )
            if math.isnan(spearman):
                spearman = 0.0
        except Exception:
            spearman = 0.0
    else:
        spearman = float("nan")

    proj_best = int(np.argmin(projected_dE))
    true_best = int(np.argmin(true_dE))
    top1_hit = int(proj_best == true_best)
    top3 = np.argsort(projected_dE)[:3].tolist()
    top3_recall = int(true_best in top3)

    proj_dir = np.sign(projected_dE - dE_base)
    true_dir = np.sign(true_dE - dE_base)
    sign_agree_rate = float(np.mean(proj_dir == true_dir))

    mean_abs_proj_err = float(np.mean(np.abs(projected_dE - true_dE)))

    return LayerResult(
        example_idx=example_idx,
        layer_k=layer_k,
        n_layers=n_layers,
        base_slot=int(base_slot),
        base_thickness_nm=float(base_thickness_nm),
        gt_slot=int(gt_slots[layer_k]),
        gt_thickness_nm=float(gt_thicknesses[layer_k]),
        pool_size=pool_size,
        dE_base=dE_base,
        projected_dE=projected_dE.tolist(),
        true_dE=true_dE.tolist(),
        spearman=spearman,
        top1_hit=top1_hit,
        top3_recall=top3_recall,
        sign_agree_rate=sign_agree_rate,
        mean_abs_proj_err=mean_abs_proj_err,
    )


# ============================================================================
# Aggregation + verdict + plotting
# ============================================================================


def _summarize(results: List[LayerResult]) -> Dict[str, float]:
    if not results:
        return {}
    spearmans = np.array(
        [r.spearman for r in results if not math.isnan(r.spearman)]
    )
    top1 = np.array([r.top1_hit for r in results])
    top3 = np.array([r.top3_recall for r in results])
    signs = np.array([r.sign_agree_rate for r in results])
    errs = np.array([r.mean_abs_proj_err for r in results])
    dE_base = np.array([r.dE_base for r in results])
    pool_sizes = np.array([r.pool_size for r in results])
    return {
        "n_layer_pairs": int(len(results)),
        "spearman_median": float(np.median(spearmans)) if len(spearmans) else float("nan"),
        "spearman_mean":   float(np.mean(spearmans))   if len(spearmans) else float("nan"),
        "spearman_q1":     float(np.quantile(spearmans, 0.25)) if len(spearmans) else float("nan"),
        "spearman_q3":     float(np.quantile(spearmans, 0.75)) if len(spearmans) else float("nan"),
        "top1_hit_rate":   float(np.mean(top1)),
        "top3_recall":     float(np.mean(top3)),
        "sign_agree_mean": float(np.mean(signs)),
        "mean_abs_proj_err_mean":   float(np.mean(errs)),
        "mean_abs_proj_err_median": float(np.median(errs)),
        "dE_base_mean":    float(np.mean(dE_base)),
        "dE_base_median":  float(np.median(dE_base)),
        "pool_size_mean":  float(np.mean(pool_sizes)),
    }


def _verdict(summary: Dict[str, float]) -> Tuple[str, str]:
    sp = summary.get("spearman_median", float("nan"))
    t1 = summary.get("top1_hit_rate", float("nan"))
    if math.isnan(sp) or math.isnan(t1):
        return "UNKNOWN", "not enough data to render a verdict"
    if sp >= 0.70 and t1 >= 0.50:
        return "HOLDS", (
            f"median Spearman {sp:.2f} ≥ 0.70 and top-1 hit rate "
            f"{t1:.0%} ≥ 50% → linear projection is a useful "
            f"approximation. Keep the current STE."
        )
    if sp < 0.30 and t1 < 0.20:
        return "NOISE", (
            f"median Spearman {sp:.2f} < 0.30 and top-1 hit rate "
            f"{t1:.0%} < 20% → linear projection barely correlates "
            f"with truth. Switch to winning-material-only STE (see "
            f"README §alt approach)."
        )
    return "MIXED", (
        f"median Spearman {sp:.2f}, top-1 hit rate {t1:.0%} → partial "
        f"signal. Current STE may still help; also worth trying "
        f"winning-material-only for comparison."
    )


def _plot_scatter(results: List[LayerResult], out_path: Path) -> None:
    xs, ys = [], []
    for r in results:
        xs.extend(r.projected_dE)
        ys.extend(r.true_dE)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=6, alpha=0.15, rasterized=True)
    lim_max = float(np.percentile(np.concatenate([xs, ys]), 99))
    lim = [0, max(1.0, lim_max)]
    ax.plot(lim, lim, "r-", lw=1, label="y = x (perfect projection)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Projected ΔE₀₀ after swap")
    ax.set_ylabel("True ΔE₀₀ after swap")
    ax.set_title(
        f"STE projection vs true ΔE — {len(results)} (example, layer) pairs, "
        f"{len(xs)} material trials"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_hist(vals: np.ndarray, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vals, bins=30)
    ax.axvline(np.median(vals), color="red", ls="--",
               label=f"median={np.median(vals):.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count of (example, layer) pairs")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# CLI + main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Parquet dir (default: <repo>/data/finetune)")
    p.add_argument("--split", type=str, default="all",
                   choices=["train", "validation", "all"])
    p.add_argument("--n-examples", type=int, default=500,
                   help="Number of examples to evaluate")
    p.add_argument("--layers-per-example", type=str, default="all",
                   help="'all', 'random', or an integer (samples that many "
                        "layers per example, capped at n_layers)")
    p.add_argument("--incidence-angle", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)

    # Anchor selection.
    p.add_argument("--base-mode", type=str, default="gt",
                   choices=["gt", "model_argmax"],
                   help="Where to anchor the STE linearization. 'gt' uses "
                        "the true slot + thickness (ΔE_base=0, tests optimum-"
                        "region behavior). 'model_argmax' loads the pretrained "
                        "checkpoint and uses its argmax pick per (example, k) "
                        "— the anchor training's gradient is actually "
                        "evaluated at on every step.")

    # Only required if base-mode = model_argmax. Match finetune_de.py defaults.
    p.add_argument("--pretrained-checkpoint", type=Path, default=None,
                   help="Required if --base-mode=model_argmax")
    p.add_argument("--feature-mode", type=str, default="raw_spectrum")
    p.add_argument("--encoder-hidden", type=int, default=128)
    p.add_argument("--encoder-out", type=int, default=64)
    p.add_argument("--encoder-dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-mode", type=str, default="cross_attn",
                   choices=["cross_attn"])
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--slot-encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=1)

    return p.parse_args()


def _load_pretrained_model(args):
    """Import torch lazily so gt-mode runs don't need it. Returns
    (model_on_device, device)."""
    import torch  # lazy
    from src.model import ModelConfig, build_model

    if args.pretrained_checkpoint is None:
        raise SystemExit(
            "--base-mode=model_argmax requires --pretrained-checkpoint"
        )

    config = ModelConfig(
        feature_mode=args.feature_mode,
        encoder_hidden=args.encoder_hidden,
        encoder_out=args.encoder_out,
        encoder_dropout=args.encoder_dropout,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        head_mode=args.head_mode,
        n_heads=args.n_heads,
        slot_encoder_layers=args.slot_encoder_layers,
        decoder_layers=args.decoder_layers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    state = torch.load(
        args.pretrained_checkpoint / "model.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Model loaded on {device} from {args.pretrained_checkpoint}",
          flush=True)
    return model, device


def _model_argmax_for_example(model, device, example, gt_slots, gt_thicknesses):
    """Run one forward pass and return the model's argmax (slot_idx,
    thickness_nm) at every position k in 0..len(gt_slots)-1.

    Uses the same input semantics as the finetune's rollout: GT
    structure_matrix as teacher-forced context, cross-attn head produces
    logits per position, we split the vocab and pick slot argmax then
    thickness-bin argmax conditioned on that slot (mirrors
    src.de_finetune.ste_pick's forward pass).
    """
    import torch  # lazy
    from src.material_features import featurize_pool, pad_pool_features

    n_layers = len(gt_slots)

    pool_feats_unpadded = featurize_pool(example.pool, mode="raw_spectrum")
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
    pool_size = len(example.pool)

    gt_slots_use = list(gt_slots)[:MAX_LAYERS]
    gt_thick_use = list(gt_thicknesses)[:MAX_LAYERS]
    structure_matrix = build_structure_matrix(gt_slots_use, gt_thick_use)

    with torch.no_grad():
        logits = model(
            lab=example.lab.unsqueeze(0).to(device),
            pool_features=pool_feats.unsqueeze(0).to(device),
            pool_mask=pool_mask.unsqueeze(0).to(device),
            pool_size=torch.tensor([pool_size], dtype=torch.long).to(device),
            structure_matrix=structure_matrix.unsqueeze(0).to(device),
            apply_output_mask=True,
        )                                            # [1, MAX_LAYERS+1, VOCAB]
    logits = logits[0].detach().cpu().numpy()        # [MAX_LAYERS+1, VOCAB]
    # Sanitize -inf from output mask so argmax and max-over-row are stable.
    logits = np.nan_to_num(logits, neginf=-1e9, posinf=1e9)

    argmax_slot_per_k: List[int] = []
    argmax_thick_nm_per_k: List[float] = []
    for k in range(n_layers):
        layer_logits = logits[k, :M_MAX * NUM_THICKNESSES].reshape(
            M_MAX, NUM_THICKNESSES,
        )
        # Slot scored by its best thickness's logit (matches ste_pick).
        slot_scores = layer_logits.max(axis=-1)
        argmax_slot = int(np.argmax(slot_scores))
        argmax_thick_bin = int(np.argmax(layer_logits[argmax_slot]))
        argmax_slot_per_k.append(argmax_slot)
        argmax_thick_nm_per_k.append(float(THICKNESSES[argmax_thick_bin]))
    return argmax_slot_per_k, argmax_thick_nm_per_k


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is None:
        args.data_dir = find_repo_root(Path(__file__).parent) / "data" / "finetune"
    print(f"[INFO] Data dir: {args.data_dir}", flush=True)
    print(f"[INFO] Output dir: {args.output_dir}", flush=True)
    print(f"[INFO] N examples: {args.n_examples}", flush=True)
    print(f"[INFO] Layers per example: {args.layers_per_example}", flush=True)
    print(f"[INFO] Base mode: {args.base_mode}", flush=True)

    # If we're using model-argmax anchors, load the checkpoint up front.
    model = None
    device = None
    if args.base_mode == "model_argmax":
        model, device = _load_pretrained_model(args)

    # Load examples (streaming; take the first n_examples).
    ds = FlexThinFilmDataset(
        data_prompts_dir=args.data_dir,
        seed=args.seed,
        split=args.split,
        streaming=True,
        limit_examples=args.n_examples,
    )
    rng = random.Random(args.seed)

    all_results: List[LayerResult] = []
    t0 = time.time()

    for ex_idx, example in enumerate(ds):
        if ex_idx >= args.n_examples:
            break
        n_layers = len(example.target_slots)
        if n_layers == 0:
            continue

        # Materialise the pool as [pool_size, NUM_LAMBDA] arrays.
        pool_n = np.stack([m.n for m in example.pool], axis=0)  # [P, L]
        pool_k = np.stack([m.k for m in example.pool], axis=0)

        target_lab = np.asarray(
            denormalize_lab(example.lab), dtype=np.float64,
        )                                                       # [3]

        gt_slots_list = list(example.target_slots)
        gt_thicknesses_list = list(example.target_thicknesses)

        # Anchor per layer position: either GT or model argmax.
        if args.base_mode == "model_argmax":
            base_slots_per_k, base_thicks_per_k = _model_argmax_for_example(
                model, device, example, gt_slots_list, gt_thicknesses_list,
            )
            # Guard: if the pretrained model picks a padded slot (should not
            # happen because apply_output_mask=True, but defensive), clamp
            # to a valid slot from the pool.
            pool_size = pool_n.shape[0]
            for kk, s in enumerate(base_slots_per_k):
                if s >= pool_size:
                    base_slots_per_k[kk] = int(gt_slots_list[kk])
        else:
            # gt mode: base = GT.
            base_slots_per_k = list(gt_slots_list)
            base_thicks_per_k = [float(t) for t in gt_thicknesses_list]

        # Which layer positions to test?
        if args.layers_per_example == "all":
            layers_to_test = list(range(n_layers))
        elif args.layers_per_example == "random":
            layers_to_test = [rng.randrange(n_layers)]
        else:
            try:
                cap = int(args.layers_per_example)
            except ValueError:
                raise SystemExit(
                    f"--layers-per-example must be 'all', 'random', or int; "
                    f"got {args.layers_per_example!r}"
                )
            all_ks = list(range(n_layers))
            rng.shuffle(all_ks)
            layers_to_test = all_ks[:min(cap, n_layers)]

        for k in layers_to_test:
            result = _measure_one_layer(
                example_idx=ex_idx,
                layer_k=k,
                gt_slots=gt_slots_list,
                gt_thicknesses=gt_thicknesses_list,
                pool_n_full=pool_n,
                pool_k_full=pool_k,
                target_lab=target_lab,
                incidence_angle=args.incidence_angle,
                base_slot=base_slots_per_k[k],
                base_thickness_nm=base_thicks_per_k[k],
            )
            all_results.append(result)

        if (ex_idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            per_ex = elapsed / (ex_idx + 1)
            print(
                f"[progress] example {ex_idx + 1}/{args.n_examples}  "
                f"total layer pairs so far: {len(all_results)}  "
                f"elapsed {elapsed:.0f}s  per-example {per_ex:.2f}s",
                flush=True,
            )

    print(f"[INFO] Done. {len(all_results)} (example, layer) pairs.", flush=True)

    # ----- Aggregate + save -----
    summary = _summarize(all_results)
    verdict, verdict_text = _verdict(summary)
    summary["verdict"] = verdict

    # A useful side-metric in model_argmax mode: how often is the anchor
    # actually different from GT? (In gt mode it's 0 by construction.)
    anchor_matches_gt = float(np.mean([
        int(r.base_slot == r.gt_slot) for r in all_results
    ])) if all_results else float("nan")
    summary["anchor_matches_gt"] = anchor_matches_gt

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "split": args.split,
            "n_examples": args.n_examples,
            "layers_per_example": args.layers_per_example,
            "incidence_angle": args.incidence_angle,
            "seed": args.seed,
            "base_mode": args.base_mode,
            "pretrained_checkpoint": (
                str(args.pretrained_checkpoint)
                if args.pretrained_checkpoint else None
            ),
        },
        "summary": summary,
        "per_layer": [asdict(r) for r in all_results],
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # ----- Human-readable summary -----
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(
        f"STE PROJECTION QUALITY — summary  (base_mode={args.base_mode})"
    )
    lines.append("=" * 78)
    lines.append(f"  Verdict: {verdict}")
    lines.append(f"    {verdict_text}")
    lines.append("")
    lines.append(f"  N (example, layer) pairs      : {summary['n_layer_pairs']}")
    lines.append(f"  Mean pool size per example    : {summary['pool_size_mean']:.1f}")
    lines.append(f"  Anchor matches GT slot        : "
                 f"{summary['anchor_matches_gt']:.1%}  "
                 f"(gt mode = 100%, model_argmax mode = model's slot-match rate)")
    lines.append(f"  Mean base ΔE (before swap)    : {summary['dE_base_mean']:.2f}  "
                 f"(gt mode ~= 0; model_argmax mode = model's own greedy ΔE)")
    lines.append("")
    lines.append(f"  Spearman ρ (proj vs true)")
    lines.append(f"    median                       : {summary['spearman_median']:+.3f}")
    lines.append(f"    mean                         : {summary['spearman_mean']:+.3f}")
    lines.append(f"    IQR                          : "
                 f"[{summary['spearman_q1']:+.3f}, {summary['spearman_q3']:+.3f}]")
    lines.append("")
    lines.append(f"  Top-1 hit rate (proj's #1 == true #1)")
    lines.append(f"    fraction                     : {summary['top1_hit_rate']:.1%}")
    lines.append(f"  Top-3 recall (true best in proj top-3)")
    lines.append(f"    fraction                     : {summary['top3_recall']:.1%}")
    lines.append("")
    lines.append(f"  Sign-agreement per material    : "
                 f"mean {summary['sign_agree_mean']:.1%}")
    lines.append(f"  |ΔE_proj - ΔE_true|            : "
                 f"mean {summary['mean_abs_proj_err_mean']:.2f}, "
                 f"median {summary['mean_abs_proj_err_median']:.2f}")
    lines.append("=" * 78)
    lines.append("Decision guide:")
    lines.append("  HOLDS  → median ρ ≥ 0.70 AND top-1 ≥ 50%  → keep current STE")
    lines.append("  NOISE  → median ρ < 0.30 AND top-1 < 20%  → switch to winning-only")
    lines.append("  MIXED  → anything in between              → likely try both")
    lines.append("=" * 78)
    summary_text = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary_text)
    print("\n" + summary_text)

    # ----- Plots -----
    if all_results:
        _plot_scatter(all_results, args.output_dir / "projected_vs_true.png")
        _plot_hist(
            np.array([r.spearman for r in all_results
                      if not math.isnan(r.spearman)]),
            xlabel="Spearman ρ (per example, layer)",
            title="Rank agreement — projected vs true post-swap ΔE",
            out_path=args.output_dir / "spearman_hist.png",
        )
        _plot_hist(
            np.array([r.sign_agree_rate for r in all_results]),
            xlabel="Sign agreement rate (per example, layer)",
            title="Does projection get the right direction (better/worse)?",
            out_path=args.output_dir / "sign_agreement_hist.png",
        )
        print(f"[INFO] Plots written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
