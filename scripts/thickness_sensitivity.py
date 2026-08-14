#!/usr/bin/env python3
"""
Thickness Sensitivity Study on High-Chroma Structures
=====================================================

Purpose: measure how ΔE_00 responds to thickness perturbations, so we can
decide the token grid's granularity (currently uniform 5 nm on [5, 200]).
Two questions we're trying to answer:

  1. What Δnm perturbation moves the achieved colour by ΔE ≈ 1 (~JND)?
     If the answer at some base thickness is < 5 nm, the current grid
     under-resolves that region and we're paying perceptual cost.

  2. Does sensitivity vary systematically with the layer's BASE thickness?
     Thin metal layers (Ag @ 15 nm) usually swing colour hard on ±1 nm,
     while thick dielectrics (SiO2 @ 150 nm) tolerate ±10 nm. If the
     effect is monotone in base thickness, a log-nm (or piecewise) grid
     with finer spacing at small t would spend tokens where they matter.

Method
------
* Generate N high-chroma structures via the existing directed-search path
  (create_dataset/src/high_chroma_search.py).
* For each structure, for each layer, sweep that layer's thickness in
  ±sweep_max_nm around its stored value at sweep_step_nm resolution,
  re-simulate the full stack, and record ΔE_00 vs the base achieved Lab.
* Aggregate per (base-thickness bin, material category) and report:
    - Local slope |dΔE/dnm| at Δnm ≈ 0 (via central finite difference).
    - Δnm needed to reach ΔE = 1, 2, 5 (the "resolvability").
    - Ranked-worst layers so we can eyeball the extreme cases.

Grid-comparison analysis
------------------------
Given a candidate grid, the worst-case snapping error for a layer at
base thickness t is half the local bin width, multiplied by the local
|dΔE/dnm|. We compute this for several grids and print a table:
    - linear 5 nm         (current default)
    - linear 2 nm         (2.5x finer, uniform)
    - linear 1 nm         (5x finer, uniform)
    - log-nm (ratio 1.10) (~50 bins, geometric)
    - log-nm (ratio 1.05) (~100 bins, geometric)

Outputs
-------
    <out>/sensitivity.json         raw sweep results + aggregates
    <out>/curves_examples.png      per-layer ΔE(Δnm) for a handful of layers
    <out>/sensitivity_by_bin.png   violin/box of |dΔE/dnm| by base-thickness bin
    <out>/delta_e_1_by_bin.png     Δnm needed for ΔE=1 by base-thickness bin
    <out>/grid_comparison.png      expected ΔE cost per grid candidate
    <out>/grid_comparison.txt      the same as a printable table

Wallclock: default N=30 structures, ±10 nm × 1 nm step ≈ 3000 evals
≈ 4-6 min on CPU. Bump --n-structures / --sweep-max-nm for more density.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.material_features import MaterialNK, load_jll_directory


# ============================================================================
# ΔE_00 (self-contained, no jax dep)
# ============================================================================

def _ciede2000(lab1, lab2) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0
    elif abs(dhp) > 180:
        dhp -= 360 if dhp > 180 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hbp = (h1p + h2p) / 2
    if C1p * C2p != 0 and abs(h1p - h2p) > 180:
        hbp += 180 if h1p + h2p < 360 else -180
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dTheta = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    R_C = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    S_L = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    S_C = 1 + 0.045 * Cbp
    S_H = 1 + 0.015 * Cbp * T
    R_T = -math.sin(math.radians(2 * dTheta)) * R_C
    return math.sqrt(
        (dLp / S_L) ** 2 + (dCp / S_C) ** 2 + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


# ============================================================================
# Material category classification (metal / dielectric / semi) — heuristic
# ============================================================================

_KNOWN_METALS = {"Ag", "Al", "Au", "Cu", "Ni", "Cr", "W", "Pt", "Ti"}


def _classify(mat: MaterialNK) -> str:
    """Cheap category tag from material name + k magnitude in the visible.

    We stratify by (metal / dielectric / other) since sensitivity trends
    differ sharply: a 20 nm Ag layer swings colour hard on ±1 nm, while
    a 100 nm SiO2 layer tolerates ±10 nm.
    """
    if mat.name in _KNOWN_METALS:
        return "metal"
    k_avg = float(np.mean(np.abs(mat.k)))
    if k_avg > 0.5:
        return "metal"
    if k_avg > 0.05:
        return "absorbing"
    return "dielectric"


# ============================================================================
# One-structure sweep
# ============================================================================

@dataclass
class LayerSweep:
    structure_id: int
    layer_idx: int
    material_name: str
    material_category: str
    base_thickness_nm: int
    achieved_lab: Tuple[float, float, float]
    sweep_delta_nm: List[float]
    sweep_delta_e: List[float]


def sweep_one_layer(
    sim, materials: List[MaterialNK], thicknesses_nm: List[int],
    layer_idx: int, base_lab: Tuple[float, float, float],
    sweep_max_nm: float, sweep_step_nm: float,
    min_nm: float = 1.0, max_nm: float = 300.0,
) -> Tuple[List[float], List[float]]:
    """Sweep one layer's thickness ± sweep_max_nm and record ΔE_00."""
    base = thicknesses_nm[layer_idx]
    deltas = np.arange(-sweep_max_nm, sweep_max_nm + 1e-9, sweep_step_nm)
    delta_es: List[float] = []
    valid_deltas: List[float] = []
    for d in deltas:
        t_new = base + float(d)
        if t_new < min_nm or t_new > max_nm:
            continue
        thicks_perturbed = list(thicknesses_nm)
        thicks_perturbed[layer_idx] = t_new
        lab = sim.compute_lab(materials, thicks_perturbed)
        delta_es.append(_ciede2000(base_lab, lab))
        valid_deltas.append(float(d))
    return valid_deltas, delta_es


# ============================================================================
# Sensitivity metrics
# ============================================================================

def local_slope(deltas: List[float], delta_es: List[float]) -> float:
    """|dΔE/dnm| at Δnm ≈ 0 via central finite difference on the two
    innermost points bracketing 0."""
    d = np.asarray(deltas, dtype=np.float64)
    de = np.asarray(delta_es, dtype=np.float64)
    below = np.where(d < 0)[0]
    above = np.where(d > 0)[0]
    if len(below) == 0 or len(above) == 0:
        return float("nan")
    b = below[-1]  # closest below 0
    a = above[0]   # closest above 0
    return float(abs((de[a] - de[b]) / (d[a] - d[b])))


def delta_nm_at_delta_e(
    deltas: List[float], delta_es: List[float], target_de: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Smallest |Δnm| (negative / positive side) at which ΔE crosses target.

    Returns (dnm_negative, dnm_positive). None on a side means the sweep
    range didn't reach target ΔE on that side (i.e. the layer tolerated
    at least sweep_max_nm without hitting the threshold).
    """
    d = np.asarray(deltas, dtype=np.float64)
    de = np.asarray(delta_es, dtype=np.float64)
    neg_mask = d < 0
    pos_mask = d > 0
    neg = None
    pos = None
    # Walk outward from 0 on each side and find the first crossing.
    for side_mask, out_side in ((neg_mask, "neg"), (pos_mask, "pos")):
        d_side = d[side_mask]
        de_side = de[side_mask]
        # Sort by |d| ascending so we walk out from 0.
        order = np.argsort(np.abs(d_side))
        for j in order:
            if de_side[j] >= target_de:
                if out_side == "neg":
                    neg = float(abs(d_side[j]))
                else:
                    pos = float(d_side[j])
                break
    return neg, pos


# ============================================================================
# Grid comparison
# ============================================================================
#
# For each candidate grid we compute the expected snapping error contribution:
# a layer at base t has local slope s = |dΔE/dnm|. Under grid G with local
# bin width w(t), the worst-case ΔE cost is s · w(t)/2 (uniform on bin) and
# the RMS cost is s · w(t)/√12. We report both p50 and p95 across layers,
# using the local bin width appropriate for each grid.


def linear_grid_bin_width(step_nm: float) -> callable:
    """Return a function w(t) = step_nm regardless of t."""
    return lambda t: step_nm


def log_grid_bin_width(ratio: float, t_min: float = 5.0,
                       t_max: float = 200.0) -> callable:
    """Return w(t) for a geometric grid where successive bins scale by
    `ratio` (e.g. 1.10 → each bin is 10 % wider than the previous)."""
    def w(t: float) -> float:
        # Local bin width for a geometric grid is roughly t · (ratio − 1).
        return max(0.5, float(t) * (ratio - 1.0))
    return w


def piecewise_grid_bin_width(pieces: List[Tuple[float, float, float]]) -> callable:
    """pieces = [(lo, hi, step), ...]. Overlaps use the last matching piece."""
    def w(t: float) -> float:
        step = pieces[-1][2]
        for lo, hi, s in pieces:
            if lo <= t <= hi:
                step = s
        return float(step)
    return w


def evaluate_grid(
    per_layer_stats: List[Dict],
    bin_width_fn: callable,
    name: str,
    max_thickness_nm: float = 200.0,
) -> Dict[str, float]:
    """Estimate p50/p95/max ΔE snapping cost per layer for a grid."""
    costs = []
    counts_used = 0
    for r in per_layer_stats:
        slope = r["local_slope_dE_per_nm"]
        base = r["base_thickness_nm"]
        if not math.isfinite(slope) or slope <= 0:
            continue
        w = bin_width_fn(base)
        # Worst-case snap error (per Nyquist-ish reasoning) ≈ w/2 · slope.
        cost_max = 0.5 * w * slope
        costs.append(cost_max)
        counts_used += 1
    if not costs:
        return {"grid": name, "n_layers": 0}
    arr = np.asarray(costs)
    return {
        "grid": name,
        "n_layers": counts_used,
        "median_snap_dE": float(np.median(arr)),
        "p95_snap_dE": float(np.percentile(arr, 95)),
        "max_snap_dE": float(arr.max()),
        "n_layers_over_1": int((arr > 1.0).sum()),
        "n_layers_over_2": int((arr > 2.0).sum()),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_example_curves(sweeps: List[LayerSweep], out_path: Path,
                        n_examples: int = 12) -> None:
    """A grid of per-layer ΔE(Δnm) curves for a random sample of layers."""
    rng = np.random.default_rng(0)
    if len(sweeps) > n_examples:
        chosen = rng.choice(len(sweeps), size=n_examples, replace=False).tolist()
    else:
        chosen = list(range(len(sweeps)))
    ncols = 4
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows),
                             squeeze=False)
    for ax_i, layer_i in enumerate(chosen):
        s = sweeps[layer_i]
        ax = axes[ax_i // ncols][ax_i % ncols]
        ax.plot(s.sweep_delta_nm, s.sweep_delta_e,
                color={"metal": "tab:red", "absorbing": "tab:orange",
                       "dielectric": "tab:blue"}.get(s.material_category, "gray"),
                lw=1.4)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
        ax.axhline(2.0, color="k", lw=0.5, alpha=0.2)
        ax.set_title(f"{s.material_name} @ {s.base_thickness_nm} nm "
                     f"({s.material_category})", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax_i // ncols == nrows - 1:
            ax.set_xlabel("Δt (nm)", fontsize=8)
        if ax_i % ncols == 0:
            ax.set_ylabel("ΔE_00", fontsize=8)
    # Hide unused axes.
    for j in range(len(chosen), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-layer ΔE vs Δthickness (dashed = ΔE=1 / ΔE=2)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_label(lo: float, hi: float) -> str:
    return f"{int(lo)}–{int(hi)} nm"


def plot_slope_by_bin(per_layer_stats: List[Dict], out_path: Path,
                      bin_edges=(5, 20, 40, 80, 120, 200)) -> None:
    """Violin/box of local slope |dΔE/dnm| grouped by base-thickness bin."""
    edges = list(bin_edges)
    groups: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    for r in per_layer_stats:
        s = r["local_slope_dE_per_nm"]
        t = r["base_thickness_nm"]
        if not math.isfinite(s):
            continue
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1] or (i == len(edges) - 2 and t == edges[-1]):
                groups[i].append(s)
                break
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(groups))
    # Use boxplot for cleaner tails than violin on small n.
    data = [g if g else [0.0] for g in groups]
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="#dfe7f5", edgecolor="#345"),
               medianprops=dict(color="#345"))
    for i, g in enumerate(groups):
        ax.scatter([positions[i]] * len(g), g, s=6, color="k", alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([_bin_label(edges[i], edges[i + 1])
                        for i in range(len(edges) - 1)])
    ax.set_ylabel("|dΔE/dnm|  (ΔE_00 per nm perturbation)")
    ax.set_xlabel("base thickness bin")
    ax.set_title("Local thickness sensitivity by base-thickness region")
    ax.axhline(0.2, color="tab:red", lw=0.7, linestyle=":",
               label="0.2 ΔE/nm (5 nm grid gives ~0.5 ΔE snap-error)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_nm_for_de1(per_layer_stats: List[Dict], out_path: Path,
                          bin_edges=(5, 20, 40, 80, 120, 200)) -> None:
    """For each base-thickness bin, boxplot of Δnm needed to reach ΔE=1."""
    edges = list(bin_edges)
    groups: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    for r in per_layer_stats:
        vals = []
        for side in ("dnm_de1_neg", "dnm_de1_pos"):
            v = r.get(side)
            if v is not None and math.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        t = r["base_thickness_nm"]
        # Use the smaller side — that's how quickly a small perturbation
        # crosses the JND.
        smallest = min(vals)
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1] or (i == len(edges) - 2 and t == edges[-1]):
                groups[i].append(smallest)
                break
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(groups))
    data = [g if g else [0.0] for g in groups]
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="#f5e2df", edgecolor="#653"),
               medianprops=dict(color="#653"))
    for i, g in enumerate(groups):
        ax.scatter([positions[i]] * len(g), g, s=6, color="k", alpha=0.4)
    ax.axhline(5.0, color="tab:blue", lw=0.7, linestyle=":",
               label="current grid spacing (5 nm)")
    ax.axhline(2.5, color="tab:green", lw=0.7, linestyle=":",
               label="candidate 2 nm grid (half-width)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_bin_label(edges[i], edges[i + 1])
                        for i in range(len(edges) - 1)])
    ax.set_ylabel("Δnm needed to reach ΔE = 1")
    ax.set_xlabel("base thickness bin")
    ax.set_title("Perceptibility distance: smallest Δnm that changes colour "
                 "by ΔE=1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_comparison(grid_results: List[Dict], out_path: Path) -> None:
    labels = [g["grid"] for g in grid_results]
    p50 = [g.get("median_snap_dE", 0) for g in grid_results]
    p95 = [g.get("p95_snap_dE", 0) for g in grid_results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, p50, width=0.4, label="median snap ΔE", color="#7ea3d9")
    ax.bar(x + 0.2, p95, width=0.4, label="p95 snap ΔE", color="#c86b6b")
    ax.axhline(1.0, color="k", lw=0.5, linestyle="--",
               label="JND (~ΔE 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("expected ΔE from snapping to grid")
    ax.set_title("Grid granularity vs perceptual snap cost")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-structures", type=int, default=30,
                    help="how many high-chroma structures to generate + probe")
    ap.add_argument("--sweep-max-nm", type=float, default=10.0,
                    help="sweep ± this many nm around each layer's base t")
    ap.add_argument("--sweep-step-nm", type=float, default=1.0,
                    help="Δnm resolution of the sweep")
    ap.add_argument("--high-chroma-candidate-count", type=int, default=24)
    ap.add_argument("--high-chroma-refine-iters", type=int, default=12)
    ap.add_argument("--jll-materials-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Late imports (jax pulls in slowly).
    from create_dataset.src.compile_datasets import _find_jll_materials_dir
    from create_dataset.src.pool_sampler import split_jll_real
    from create_dataset.src.random_layer import (
        LayerCountConfig, RandomLayerSimulation,
    )
    from create_dataset.src.high_chroma_search import (
        HighChromaSearchConfig, HighChromaTargetConfig,
        sample_high_chroma_target_lab, search_structure_for_target,
    )

    jll_dir = _find_jll_materials_dir(args.jll_materials_dir)
    real_pool = load_jll_directory(jll_dir)
    active, _ = split_jll_real(real_pool)
    print(f"[INFO] {len(active)} active real materials loaded from {jll_dir}")

    layer_count = LayerCountConfig(lam=4.5, min_layers=2, max_layers=10)
    sim = RandomLayerSimulation(
        held_in_real=active, layer_count=layer_count,
        incidence_angle=0, p_real=0.15, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    target_cfg = HighChromaTargetConfig()
    search_cfg = HighChromaSearchConfig(
        candidate_count=args.high_chroma_candidate_count,
        refine_iters=args.high_chroma_refine_iters,
    )

    # 1. Generate N high-chroma structures.
    print(f"[INFO] Generating {args.n_structures} high-chroma structures…")
    t0 = time.time()
    structures: List[Dict] = []
    for i in range(args.n_structures):
        target = sample_high_chroma_target_lab(rng, target_cfg)
        materials, thicks_snapped, achieved_lab, _raw = \
            search_structure_for_target(sim, target, search_cfg, rng)
        structures.append({
            "id": i,
            "target_lab": target,
            "materials": materials,
            "thicknesses_nm": thicks_snapped,
            "achieved_lab": tuple(achieved_lab),
        })
    print(f"[INFO] Generation done in {time.time() - t0:.1f}s")

    # 2. Sweep each layer of each structure.
    print(f"[INFO] Sweeping each layer ± {args.sweep_max_nm} nm at "
          f"{args.sweep_step_nm} nm resolution…")
    t0 = time.time()
    all_sweeps: List[LayerSweep] = []
    per_layer_stats: List[Dict] = []
    for s in structures:
        mats = s["materials"]
        thicks = s["thicknesses_nm"]
        base_lab = s["achieved_lab"]
        for li in range(len(thicks)):
            deltas, delta_es = sweep_one_layer(
                sim, mats, thicks, li, base_lab,
                args.sweep_max_nm, args.sweep_step_nm,
            )
            cat = _classify(mats[li])
            sw = LayerSweep(
                structure_id=s["id"], layer_idx=li,
                material_name=mats[li].name, material_category=cat,
                base_thickness_nm=int(thicks[li]),
                achieved_lab=tuple(base_lab),
                sweep_delta_nm=deltas, sweep_delta_e=delta_es,
            )
            all_sweeps.append(sw)
            slope = local_slope(deltas, delta_es)
            neg1, pos1 = delta_nm_at_delta_e(deltas, delta_es, 1.0)
            neg2, pos2 = delta_nm_at_delta_e(deltas, delta_es, 2.0)
            neg5, pos5 = delta_nm_at_delta_e(deltas, delta_es, 5.0)
            per_layer_stats.append({
                "structure_id": s["id"],
                "layer_idx": li,
                "material_name": mats[li].name,
                "material_category": cat,
                "base_thickness_nm": int(thicks[li]),
                "achieved_lab": list(base_lab),
                "local_slope_dE_per_nm": slope,
                "dnm_de1_neg": neg1, "dnm_de1_pos": pos1,
                "dnm_de2_neg": neg2, "dnm_de2_pos": pos2,
                "dnm_de5_neg": neg5, "dnm_de5_pos": pos5,
            })
    print(f"[INFO] Sweep done in {time.time() - t0:.1f}s "
          f"({len(all_sweeps)} layer-sweeps)")

    # 3. Grid comparison.
    grids = [
        ("linear 5 nm  (current)",     linear_grid_bin_width(5.0)),
        ("linear 2 nm",                linear_grid_bin_width(2.0)),
        ("linear 1 nm",                linear_grid_bin_width(1.0)),
        ("piecewise 1/2/5 nm",         piecewise_grid_bin_width([
            (5, 30, 1.0),
            (30, 80, 2.0),
            (80, 200, 5.0),
        ])),
        ("log-nm ratio 1.10",          log_grid_bin_width(1.10)),
        ("log-nm ratio 1.05",          log_grid_bin_width(1.05)),
    ]
    grid_results = [evaluate_grid(per_layer_stats, fn, name) for name, fn in grids]
    print("\n[grid comparison]")
    header = ("grid                       n     median ΔE   p95 ΔE    "
              "max ΔE   > 1 ΔE   > 2 ΔE")
    print(header)
    print("-" * len(header))
    rows = []
    for g in grid_results:
        if "median_snap_dE" not in g:
            continue
        line = (f"{g['grid']:26s}  {g['n_layers']:4d}   "
                f"{g['median_snap_dE']:7.3f}    "
                f"{g['p95_snap_dE']:7.3f}   "
                f"{g['max_snap_dE']:7.3f}   "
                f"{g['n_layers_over_1']:5d}    "
                f"{g['n_layers_over_2']:5d}")
        print(line)
        rows.append(line)
    (args.output_dir / "grid_comparison.txt").write_text(
        header + "\n" + "-" * len(header) + "\n" + "\n".join(rows) + "\n"
    )

    # 4. Plots.
    plot_example_curves(all_sweeps, args.output_dir / "curves_examples.png")
    plot_slope_by_bin(per_layer_stats, args.output_dir / "sensitivity_by_bin.png")
    plot_delta_nm_for_de1(per_layer_stats,
                          args.output_dir / "delta_e_1_by_bin.png")
    plot_grid_comparison(grid_results, args.output_dir / "grid_comparison.png")
    print(f"\n[plots] wrote 4 PNGs to {args.output_dir}")

    # 5. Persist raw + aggregate data as JSON for later re-plotting.
    payload = {
        "config": {
            "n_structures": args.n_structures,
            "sweep_max_nm": args.sweep_max_nm,
            "sweep_step_nm": args.sweep_step_nm,
            "high_chroma_candidate_count": args.high_chroma_candidate_count,
            "high_chroma_refine_iters": args.high_chroma_refine_iters,
            "seed": args.seed,
        },
        "grids": grid_results,
        "per_layer": per_layer_stats,
        "sweeps": [
            {
                "structure_id": s.structure_id,
                "layer_idx": s.layer_idx,
                "material_name": s.material_name,
                "material_category": s.material_category,
                "base_thickness_nm": s.base_thickness_nm,
                "achieved_lab": list(s.achieved_lab),
                "sweep_delta_nm": s.sweep_delta_nm,
                "sweep_delta_e": s.sweep_delta_e,
            } for s in all_sweeps
        ],
    }
    (args.output_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2))
    print(f"[json] wrote {args.output_dir / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
