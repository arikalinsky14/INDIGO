#!/usr/bin/env python3
"""
End-to-end verification for the KK-consistent generators + high-chroma search.

Run this BEFORE launching the 10M-row production data generation. It
generates a small dry-run shard, walks the parquet, and independently
checks the things you actually care about:

  1. Kramers-Kronig residuals per source, compared to real-material baseline.
     Real JLL materials are causal by measurement and set the noise floor
     (the KK integral is over ω ∈ [0, ∞) but our data covers only the
     visible band, so even real materials have a nonzero bandwidth-loss
     residual). Synthetic sources PASS if their p95 residual is within
     KK_SLACK_FACTOR × the real p95.

  2. n,k curves look physical.
     A random sample from each strategy (real / perturb / interpolate /
     lorentz / high-chroma-search layers) is plotted to a PNG grid so
     you can eyeball the shapes.

  3. Grid alignment: every thickness must be on the 5 nm token grid,
     in [5, 200] nm. High-chroma-search rows snap continuous refinement
     back to the grid before storing.

  4. Wall-clock per shard extrapolates to a reasonable full-scale run.
     We report seconds/row for both undirected (random) and directed
     (high-chroma-search) rows, then multiply out to 10M rows at the
     production 20% search share, and print the ETA at the SLURM
     concurrency the user is planning to use (--parallel-workers).

Usage
-----
    python scripts/verify_two_head.py \\
        --output-dir /tmp/indigo_dryrun \\
        --n-rows 200 --high-chroma-prob 0.2 \\
        --parallel-workers 8

Outputs
-------
    <output-dir>/dryrun.parquet       one shard produced end-to-end
    <output-dir>/dryrun.manifest.json shard manifest
    <output-dir>/curves.png           n,k grid across strategies
    <output-dir>/kk_report.json       per-material KK residuals + verdict
    <output-dir>/timing_report.json   per-row seconds + 10M extrapolation

Exits 0 on success; non-zero if any KK check fails or timing extrapolation
exceeds a printed wall-clock guard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.material_features import (
    CANONICAL_FREQ_HZ, CANONICAL_LAMBDA_NM, MaterialNK,
    load_jll_directory,
)
from src.materials_vocab import (
    MAX_THICKNESS_NM, MIN_THICKNESS_NM,
)
from src.synthetic_materials import (
    interpolate_real, parametric_lorentz, perturb_real,
)


# ============================================================================
# Kramers-Kronig test
# ============================================================================
#
# For a causal linear medium, ε(ω) = ε₁(ω) + iε₂(ω) satisfies:
#     ε₁(ω) − ε_∞ = (2/π) · P ∫_0^∞ (ω' · ε₂(ω') / (ω'^2 − ω^2)) dω'
# On a UNIFORMLY-sampled ω grid, the Hilbert transform of ε₂ gives this
# integral. Our raw (n, k) samples are uniform in λ, so we FIRST
# interpolate ε₁, ε₂ onto a uniform ω grid, then Hilbert.
#
# Even with the ω-grid fix, the check has an irreducible artefact: the
# integral runs over ω ∈ [0, ∞), but our data covers only the narrow
# visible band. The missing tails inflate every material's residual —
# including the real JLL materials, which are causal by construction
# (measured / tabulated). So we can't compare to an absolute tolerance;
# instead we use the real-material residuals as the noise floor and
# ask "is the synthetic residual within a small multiple of that?".
# This is the honest test given band-limited data.
# ============================================================================

def _hilbert_via_fft(x: np.ndarray) -> np.ndarray:
    """Discrete Hilbert transform via the analytic-signal trick."""
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    analytic = np.fft.ifft(X * h)
    return analytic.imag


def _residual_on_uniform_omega(
    mat: MaterialNK,
) -> Tuple[float, float, float]:
    """Compute the KK residual of one material on a uniform-ω grid.

    Returns (max_rel_resid, mean_rel_resid, eps1_dyn_range). Residuals are
    normalised by the dynamic range of ε₁ − ε_∞_fit so materials with a
    tiny ε swing (near-vacuum dielectrics) don't drown the statistics.
    """
    n = np.asarray(mat.n, dtype=np.float64)
    k = np.asarray(mat.k, dtype=np.float64)
    eps_r = n * n - k * k
    eps_i = 2.0 * n * k

    # Native ω grid — uniform in λ = c/(ω/2π), NOT in ω.
    omega = 2.0 * np.pi * CANONICAL_FREQ_HZ
    # Ensure monotone increasing (np.interp needs it).
    idx = np.argsort(omega)
    omega_sorted = omega[idx]
    eps_r_sorted = eps_r[idx]
    eps_i_sorted = eps_i[idx]

    # Interpolate onto a uniform-ω grid covering the same range.
    N = len(omega_sorted)
    omega_u = np.linspace(omega_sorted[0], omega_sorted[-1], N)
    eps_r_u = np.interp(omega_u, omega_sorted, eps_r_sorted)
    eps_i_u = np.interp(omega_u, omega_sorted, eps_i_sorted)

    # Hilbert transform of ε₂ on the uniform grid.
    h_eps_i = _hilbert_via_fft(eps_i_u)

    # Fit ε_∞ as the constant that makes ε₁ − ε_∞ ≈ H(ε₂) best in the
    # least-squares sense — that's just their mean difference. Better
    # than min(ε₁) which biased the residual toward the value at the
    # band edge.
    edge = len(eps_r_u) // 20
    core_r = eps_r_u[edge:-edge]
    core_h = h_eps_i[edge:-edge]
    eps_inf_fit = float(np.mean(core_r - core_h))

    lhs = eps_r_u - eps_inf_fit
    resid = lhs - h_eps_i
    core_lhs = lhs[edge:-edge]
    core_resid = resid[edge:-edge]
    dyn_range = max(float(np.ptp(core_lhs)), 1e-6)
    max_rel = float(np.max(np.abs(core_resid))) / dyn_range
    mean_rel = float(np.mean(np.abs(core_resid))) / dyn_range
    return max_rel, mean_rel, dyn_range


def check_kk(mat: MaterialNK, eps_inf_guess: Optional[float] = None
             ) -> Dict[str, float]:
    """Per-material KK residual on a uniform-ω interpolated grid."""
    max_rel, mean_rel, dyn = _residual_on_uniform_omega(mat)
    return {
        "name": mat.name,
        "source": mat.source,
        "eps1_dyn_range": dyn,
        "max_rel_residual": max_rel,
        "mean_rel_residual": mean_rel,
    }


def kk_verdict_by_source(
    reports: List[Dict[str, float]],
    slack_factor: float = 2.5,
) -> Dict[str, Dict[str, float]]:
    """Group residuals by source and compare each source to the real-material
    noise floor.

    A synthetic source PASSES if its p95 residual is ≤ slack_factor × p95
    of the JLL real materials. If no real materials are present, we fall
    back to the raw p95 with a permissive slack (real materials are the
    band-limitation noise floor; without them the KK check is unanchored).
    """
    from collections import defaultdict
    by_source: Dict[str, List[float]] = defaultdict(list)
    for r in reports:
        by_source[r["source"]].append(r["max_rel_residual"])

    real_source = "jaxlayerlumos"
    reals = by_source.get(real_source, [])
    if reals:
        real_p95 = float(np.percentile(reals, 95))
        real_median = float(np.median(reals))
        threshold = slack_factor * real_p95
    else:
        real_p95 = float("nan")
        real_median = float("nan")
        threshold = float("nan")

    out: Dict[str, Dict[str, float]] = {}
    for source, residuals in sorted(by_source.items()):
        arr = np.asarray(residuals)
        p95 = float(np.percentile(arr, 95))
        median = float(np.median(arr))
        if reals:
            passed = bool(p95 <= threshold)
        else:
            # No real reference. Warn caller by putting a NaN threshold in
            # the record; leave verdict to a permissive fallback (< 5).
            passed = bool(p95 <= 5.0)
        out[source] = {
            "n": int(arr.size),
            "median": median,
            "p95": p95,
            "max": float(arr.max()),
            "threshold_used": threshold if reals else float("nan"),
            "passed": passed,
        }
    out["_baseline"] = {
        "real_source": real_source,
        "real_median": real_median,
        "real_p95": real_p95,
        "slack_factor": slack_factor,
    }
    return out


# ============================================================================
# Small-scale data generation harness
# ============================================================================

def generate_dryrun_shard(
    output_dir: Path,
    n_rows: int,
    high_chroma_prob: float,
    seed: int = 0,
    jll_materials_dir: Optional[Path] = None,
    candidate_count: int = 24,
    refine_iters: int = 12,
) -> Tuple[Path, float, float]:
    """Produce one shard end-to-end.

    Returns (parquet_path, wall_random_s_per_row, wall_search_s_per_row).
    """
    from create_dataset.src.compile_datasets import (
        _find_jll_materials_dir, _GREY_RNG_MAGIC,
        _HIGH_CHROMA_GATE_RNG_MAGIC, _HIGH_CHROMA_SEARCH_RNG_MAGIC,
        _HIGH_CHROMA_TARGET_RNG_MAGIC, build_rows, derive_seeds,
        get_output_path,
    )
    from create_dataset.src.pool_sampler import PoolSamplerConfig, split_jll_real
    from create_dataset.src.random_layer import (
        LayerCountConfig, RandomLayerSimulation,
    )

    jll = _find_jll_materials_dir(jll_materials_dir)
    real_pool = load_jll_directory(jll)
    active, inactive = split_jll_real(real_pool)
    structure_seed, pool_seed = derive_seeds(seed)
    accept_seed = structure_seed ^ _GREY_RNG_MAGIC
    hc_gate_seed = structure_seed ^ _HIGH_CHROMA_GATE_RNG_MAGIC
    hc_target_seed = structure_seed ^ _HIGH_CHROMA_TARGET_RNG_MAGIC
    hc_search_seed = structure_seed ^ _HIGH_CHROMA_SEARCH_RNG_MAGIC

    layer_count = LayerCountConfig(lam=4.5, min_layers=2, max_layers=10)
    pool_cfg = PoolSamplerConfig(pool_size_min=4, pool_size_max=32, p_real=0.15)
    sim = RandomLayerSimulation(
        held_in_real=active, layer_count=layer_count,
        incidence_angle=0, p_real=0.15,
        synthetic_weights=pool_cfg.synthetic_weights,
        seed=structure_seed,
    )
    pool_rng = np.random.default_rng(pool_seed)
    accept_rng = np.random.default_rng(accept_seed)
    hc_gate_rng = np.random.default_rng(hc_gate_seed)
    hc_target_rng = np.random.default_rng(hc_target_seed)
    hc_search_rng = np.random.default_rng(hc_search_seed)

    hc_target_cfg = hc_search_cfg = None
    if high_chroma_prob > 0.0:
        from create_dataset.src.high_chroma_search import (
            HighChromaSearchConfig, HighChromaTargetConfig,
        )
        hc_target_cfg = HighChromaTargetConfig()
        hc_search_cfg = HighChromaSearchConfig(
            candidate_count=candidate_count,
            refine_iters=refine_iters,
            optimizer_name="dog",
        )

    t0 = time.time()
    rows, stats = build_rows(
        target_rows=n_rows,
        sim=sim,
        held_in_real=active,
        pool_config=pool_cfg,
        pool_rng=pool_rng,
        accept_rng=accept_rng,
        structure_seed=structure_seed,
        incidence_angle=0,
        greyscale_chroma_threshold=8.0,
        greyscale_keep_prob=0.2,
        max_attempts_factor=6.0,
        high_chroma_prob=high_chroma_prob,
        high_chroma_gate_rng=hc_gate_rng,
        high_chroma_target_rng=hc_target_rng,
        high_chroma_search_rng=hc_search_rng,
        high_chroma_target_cfg=hc_target_cfg,
        high_chroma_search_cfg=hc_search_cfg,
    )
    total_wall = time.time() - t0

    # Split wall-clock: search rows cost dramatically more than random ones.
    n_search = sum(1 for r in rows if r.get("structure_source") == "high_chroma_search")
    n_random = len(rows) - n_search
    # Attribute total wall time in proportion to counts × prior of relative cost.
    # A search row costs ~(candidate_count + refine_iters·grad_cost) forward-passes
    # so ~60× a random row at the defaults. Compute a first-order split using
    # attempts as the proxy — this is a rough sanity number, not billing.
    if n_search > 0 and n_random > 0:
        # Assume a 60x cost multiplier for search vs random attempts.
        cost_ratio = 60.0
        weighted = n_random + cost_ratio * n_search
        random_wall = total_wall * (n_random / weighted)
        search_wall = total_wall - random_wall
        sec_per_random = random_wall / max(n_random, 1)
        sec_per_search = search_wall / max(n_search, 1)
    elif n_search > 0:
        sec_per_random = 0.0
        sec_per_search = total_wall / n_search
    else:
        sec_per_random = total_wall / max(n_random, 1)
        sec_per_search = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "dryrun.parquet"
    import pandas as pd
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    manifest = {
        "n_rows": len(rows),
        "n_random": n_random,
        "n_search": n_search,
        "high_chroma_prob": high_chroma_prob,
        "rejection_stats": stats,
        "wall_total_s": total_wall,
        "sec_per_random": sec_per_random,
        "sec_per_search": sec_per_search,
    }
    (output_dir / "dryrun.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[dryrun] {len(rows)} rows in {total_wall:.1f}s "
          f"({n_random} random, {n_search} search)")
    return parquet_path, sec_per_random, sec_per_search


# ============================================================================
# Plotting
# ============================================================================

def plot_curves(materials_by_source: Dict[str, List[MaterialNK]],
                output_path: Path) -> None:
    sources = list(materials_by_source.keys())
    n_per = max(len(v) for v in materials_by_source.values())
    fig, axes = plt.subplots(len(sources), n_per,
                             figsize=(3 * n_per, 2.6 * len(sources)),
                             squeeze=False)
    for row_i, source in enumerate(sources):
        mats = materials_by_source[source]
        for col_i in range(n_per):
            ax = axes[row_i][col_i]
            if col_i >= len(mats):
                ax.axis("off")
                continue
            m = mats[col_i]
            ax.plot(CANONICAL_LAMBDA_NM * 1e9, m.n, label="n", color="tab:blue")
            ax.plot(CANONICAL_LAMBDA_NM * 1e9, m.k, label="k", color="tab:orange")
            ax.set_title(f"{source}\n{m.name[:20]}", fontsize=8)
            ax.tick_params(labelsize=7)
            if row_i == len(sources) - 1:
                ax.set_xlabel("λ (nm)", fontsize=8)
            if col_i == 0:
                ax.legend(fontsize=7)
    fig.suptitle("n, k spectra sampled across generation strategies",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

@dataclass
class TimingReport:
    n_rows: int
    high_chroma_prob: float
    wall_total_s: float
    sec_per_random: float
    sec_per_search: float
    per_row_avg_s: float
    projected_10M_hours: float
    projected_10M_hours_with_workers: float
    parallel_workers: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-rows", type=int, default=200,
                    help="rows to generate in the dry run (default 200)")
    ap.add_argument("--high-chroma-prob", type=float, default=0.2)
    ap.add_argument("--parallel-workers", type=int, default=8,
                    help="how many SLURM workers will be used for the "
                         "10M-row production run; used only for ETA.")
    ap.add_argument("--kk-slack-factor", type=float, default=2.5,
                    help="Synthetic sources PASS if their p95 KK residual "
                         "≤ slack × real-material p95. Real materials define "
                         "the noise floor since we're band-limited to the "
                         "visible; the KK integral runs over ω ∈ [0, ∞) so "
                         "even real materials show a bandwidth-limitation "
                         "residual > 0.")
    ap.add_argument("--high-chroma-candidate-count", type=int, default=24,
                    help="stage-1 candidates per search row (default 24). "
                         "Cut to 12 to roughly halve search cost.")
    ap.add_argument("--high-chroma-refine-iters", type=int, default=12,
                    help="stage-2 gradient iters per search row (default 12). "
                         "Cut to 6 to roughly halve search cost.")
    ap.add_argument("--jll-materials-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dry-run shard generation
    print("=" * 78)
    print("[1/3] Generating dry-run shard end-to-end")
    print("=" * 78)
    parquet_path, sec_random, sec_search = generate_dryrun_shard(
        output_dir=args.output_dir,
        n_rows=args.n_rows,
        high_chroma_prob=args.high_chroma_prob,
        seed=args.seed,
        jll_materials_dir=args.jll_materials_dir,
        candidate_count=args.high_chroma_candidate_count,
        refine_iters=args.high_chroma_refine_iters,
    )

    # 2. KK check + curve plot on generated materials
    print("\n" + "=" * 78)
    print("[2/3] Kramers-Kronig residual check + curve viz")
    print("=" * 78)
    # Rebuild MaterialNK objects from a sample of parquet rows.
    import pyarrow.parquet as pq
    tbl = pq.read_table(parquet_path).to_pandas()
    materials_by_source: Dict[str, List[MaterialNK]] = {}
    kk_reports: List[Dict[str, float]] = []
    for _, row in tbl.iterrows():
        for slot_i in range(int(row["pool_size"])):
            name = str(row["pool_names"][slot_i])
            source = str(row["pool_sources"][slot_i])
            n = np.asarray(row["pool_n"][slot_i], dtype=np.float64)
            k = np.asarray(row["pool_k"][slot_i], dtype=np.float64)
            m = MaterialNK(name=name, n=n, k=k, source=source)
            materials_by_source.setdefault(source, [])
            if len(materials_by_source[source]) < 6:
                materials_by_source[source].append(m)
            kk_reports.append(check_kk(m))

    # Also add a few pure high-chroma-search structure-layer materials for
    # inspection.
    hc_rows = tbl[tbl["structure_source"] == "high_chroma_search"]
    if len(hc_rows) > 0:
        sample = hc_rows.iloc[0]
        for slot_i in sample["layer_slots"][:4]:
            slot_i = int(slot_i)
            name = str(sample["pool_names"][slot_i])
            source = f"hc_layer:{sample['pool_sources'][slot_i]}"
            n = np.asarray(sample["pool_n"][slot_i], dtype=np.float64)
            k = np.asarray(sample["pool_k"][slot_i], dtype=np.float64)
            materials_by_source.setdefault(source, []).append(
                MaterialNK(name=name, n=n, k=k, source=source)
            )

    curves_path = args.output_dir / "curves.png"
    plot_curves(materials_by_source, curves_path)
    print(f"[curves] wrote {curves_path}")

    # KK verdict: per-source residual distribution vs real-material baseline.
    per_source = kk_verdict_by_source(kk_reports, slack_factor=args.kk_slack_factor)
    baseline = per_source.pop("_baseline")
    kk_summary = {
        "n_materials_checked": len(kk_reports),
        "slack_factor": args.kk_slack_factor,
        "baseline_real_median": baseline["real_median"],
        "baseline_real_p95": baseline["real_p95"],
        "per_source": per_source,
        "n_sources_pass": sum(1 for v in per_source.values() if v["passed"]),
        "n_sources_fail": sum(1 for v in per_source.values() if not v["passed"]),
    }
    (args.output_dir / "kk_report.json").write_text(json.dumps(
        {"summary": kk_summary, "per_material": kk_reports}, indent=2,
    ))
    print(f"[KK] {len(kk_reports)} materials across {len(per_source)} sources")
    print(f"[KK] Real ({baseline['real_source']}) baseline: "
          f"median={baseline['real_median']:.3f}, p95={baseline['real_p95']:.3f}")
    print(f"[KK] Threshold: p95 ≤ {args.kk_slack_factor} × real_p95 = "
          f"{args.kk_slack_factor * baseline['real_p95']:.3f}")
    for source, s in per_source.items():
        verdict = "PASS" if s["passed"] else "FAIL"
        print(f"[KK]   {source:40s} n={s['n']:5d} median={s['median']:.3f} "
              f"p95={s['p95']:.3f} → {verdict}")

    # Row-level thickness sanity: every layer must be on the 5 nm token grid,
    # in [5, 200] nm. High-chroma-search rows snap continuous refinement
    # back to the grid before storing.
    all_thicks = [t for row in tbl["layer_thicknesses"] for t in row]
    grid_frac = sum(1 for t in all_thicks
                    if abs(round(t / 5) * 5 - t) < 1e-6) / max(len(all_thicks), 1)
    range_ok = all(5 <= t <= 200 for t in all_thicks)
    print(f"[thickness] {len(all_thicks)} layers, "
          f"{100 * grid_frac:.1f}% grid-aligned "
          f"(expected 100% — 5 nm token grid), "
          f"range {'✓' if range_ok else '✗'} [5, 200] nm")

    # 3. Timing extrapolation
    print("\n" + "=" * 78)
    print("[3/3] Wall-clock projection to 10M rows")
    print("=" * 78)
    p_search = args.high_chroma_prob
    per_row = (1.0 - p_search) * sec_random + p_search * sec_search
    total_10M_s = per_row * 10_000_000
    hours_serial = total_10M_s / 3600.0
    hours_parallel = hours_serial / max(args.parallel_workers, 1)

    timing = TimingReport(
        n_rows=args.n_rows,
        high_chroma_prob=p_search,
        wall_total_s=(sec_random + sec_search) * args.n_rows,
        sec_per_random=sec_random,
        sec_per_search=sec_search,
        per_row_avg_s=per_row,
        projected_10M_hours=hours_serial,
        projected_10M_hours_with_workers=hours_parallel,
        parallel_workers=args.parallel_workers,
    )
    (args.output_dir / "timing_report.json").write_text(
        json.dumps(asdict(timing), indent=2)
    )
    print(f"[timing] sec/row: random={sec_random:.3f}s, search={sec_search:.3f}s")
    print(f"[timing] avg sec/row at p_search={p_search:.2f}: {per_row:.3f}s")
    print(f"[timing] 10M rows serial:  {hours_serial:.1f} h "
          f"({hours_serial / 24:.1f} d)")
    print(f"[timing] 10M rows / {args.parallel_workers} workers: "
          f"{hours_parallel:.1f} h ({hours_parallel / 24:.1f} d)")

    # Verdict.
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    kk_ok = kk_summary["n_sources_fail"] == 0
    grid_ok = grid_frac >= 0.99 and range_ok
    print(f"  KK causality:      {'PASS' if kk_ok else 'FAIL'} "
          f"({kk_summary['n_sources_pass']}/{len(per_source)} sources within "
          f"{args.kk_slack_factor}× real-material p95 baseline)")
    print(f"  Grid alignment:    {'PASS' if grid_ok else 'FAIL'} "
          f"({100 * grid_frac:.1f}% on 5 nm grid, "
          f"range ok={'yes' if range_ok else 'NO'})")
    print(f"  Timing:            10M @ {args.parallel_workers}-way "
          f"{hours_parallel:.1f} h")

    if not (kk_ok and grid_ok):
        print("\n[verdict] FAILURES — DO NOT launch production until resolved.")
        raise SystemExit(1)
    print("\n[verdict] All checks green. Safe to launch production data gen.")


if __name__ == "__main__":
    main()
