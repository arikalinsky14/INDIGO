#!/usr/bin/env python3
"""
STE Projection Quality Test
===========================

The ΔE finetune's slot STE (see src/de_finetune.py:ste_pick) computes
gradients wrt every pool material's slot logit at position k by
*linearly projecting* the sim's local gradient at the argmax material's
n/k onto each other material's spectrum:

    grad_slot_logit[i] ≈ ⟨∂Lab/∂n_k, pool_n[i]⟩ + ⟨∂Lab/∂k_k, pool_k[i]⟩

This is a first-order Taylor expansion at the argmax point. It's cheap
(one sim per position, not 32), but only *locally* correct — for a
material whose n/k differs a lot from the argmax's, the projection can
point in the wrong direction.

This script quantifies how bad the approximation actually is by
comparing, for each candidate material i in the pool, the projected ΔE
after a swap against the true ΔE from an actual sim.

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

If HOLDS: keep the current STE. If NOISE: switch to a winning-material-
only approach (design in analyses/de_finetune/README.md if we get here).

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
from src.materials_vocab import denormalize_lab
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
    argmax_slot: int
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
) -> LayerResult:
    """Compute projection metrics at one (example, k) pair."""
    pool_size = pool_n_full.shape[0]
    n_layers = len(gt_slots)
    argmax_slot = int(gt_slots[layer_k])

    # Base stack from GT (all layers, all pool picks).
    n_stack_base = np.stack(
        [pool_n_full[s] for s in gt_slots], axis=0,
    )                                       # [n_layers, NUM_LAMBDA]
    k_stack_base = np.stack(
        [pool_k_full[s] for s in gt_slots], axis=0,
    )
    thicknesses_np = np.asarray(gt_thicknesses, dtype=np.float64)

    n_jax = jnp.asarray(n_stack_base)
    k_jax = jnp.asarray(k_stack_base)
    t_jax = jnp.asarray(thicknesses_np)

    # Build a function of layer-k's n/k only. Other layers stay pinned.
    def sim_wrt_layer_k(n_k, k_k):
        n_full = n_jax.at[layer_k].set(n_k)
        k_full = k_jax.at[layer_k].set(k_k)
        return _JAX_FORWARD(n_full, k_full, t_jax, incidence_angle)

    n_k_base = jnp.asarray(pool_n_full[argmax_slot])
    k_k_base = jnp.asarray(pool_k_full[argmax_slot])

    # Jacobians ∂Lab/∂(n_k, k_k) at the argmax point, shape [3, NUM_LAMBDA].
    lab_base_jax = sim_wrt_layer_k(n_k_base, k_k_base)
    lab_base = np.asarray(lab_base_jax, dtype=np.float64)
    jac_fn = jax.jacrev(sim_wrt_layer_k, argnums=(0, 1))
    jac_n_jax, jac_k_jax = jac_fn(n_k_base, k_k_base)
    jac_n = np.asarray(jac_n_jax, dtype=np.float64)      # [3, NUM_LAMBDA]
    jac_k = np.asarray(jac_k_jax, dtype=np.float64)

    dE_base = float(ciede2000(target_lab.tolist(), lab_base.tolist()))

    projected_dE = np.zeros(pool_size, dtype=np.float64)
    true_dE = np.zeros(pool_size, dtype=np.float64)

    pool_n_base_np = pool_n_full[argmax_slot]
    pool_k_base_np = pool_k_full[argmax_slot]

    for i in range(pool_size):
        # Projected: linear extrapolation from base along (pool[i] - pool[argmax])
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
        argmax_slot=argmax_slot,
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is None:
        args.data_dir = find_repo_root(Path(__file__).parent) / "data" / "finetune"
    print(f"[INFO] Data dir: {args.data_dir}", flush=True)
    print(f"[INFO] Output dir: {args.output_dir}", flush=True)
    print(f"[INFO] N examples: {args.n_examples}", flush=True)
    print(f"[INFO] Layers per example: {args.layers_per_example}", flush=True)

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
                gt_slots=list(example.target_slots),
                gt_thicknesses=list(example.target_thicknesses),
                pool_n_full=pool_n,
                pool_k_full=pool_k,
                target_lab=target_lab,
                incidence_angle=args.incidence_angle,
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

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "split": args.split,
            "n_examples": args.n_examples,
            "layers_per_example": args.layers_per_example,
            "incidence_angle": args.incidence_angle,
            "seed": args.seed,
        },
        "summary": summary,
        "per_layer": [asdict(r) for r in all_results],
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # ----- Human-readable summary -----
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("STE PROJECTION QUALITY — summary")
    lines.append("=" * 78)
    lines.append(f"  Verdict: {verdict}")
    lines.append(f"    {verdict_text}")
    lines.append("")
    lines.append(f"  N (example, layer) pairs      : {summary['n_layer_pairs']}")
    lines.append(f"  Mean pool size per example    : {summary['pool_size_mean']:.1f}")
    lines.append(f"  Mean base ΔE (before swap)    : {summary['dE_base_mean']:.2f}")
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
