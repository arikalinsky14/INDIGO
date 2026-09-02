#!/usr/bin/env python3
"""
Thickness Sensitivity Study — Most-Sensitive Layer, Adaptive Probe
==================================================================

Purpose: measure how ΔE_00 responds to thickness perturbations of the
LAYER THAT ACTUALLY DRIVES REFLECTED COLOUR, so we can decide the token
grid's granularity.

Method (one structure at a time)
--------------------------------

1. **Layer selection by finite-difference slope.** For each layer i,
   perturb its thickness by ±SLOPE_PROBE_NM (default 0.5 nm), recompute
   the achieved Lab, and record ΔE_00 on each side. Peak slope near
   zero is `max(|ΔE_+|, |ΔE_-|) / SLOPE_PROBE_NM`. Pick the layer with
   the highest peak slope — this is the layer whose thickness the model
   is most sensitive to, and therefore the one whose grid resolution
   matters most.

   Costs 2·N_layers simulator calls per structure. Replaces the previous
   "walk from air side inward until non-opaque" heuristic, which could
   pick a mildly-sensitive top layer while a highly-sensitive interior
   layer went unmeasured.

2. **Adaptive doubling probe on the chosen layer.** For each side
   (+ and −) independently:

     a. Doubling phase: probe outward through the schedule
        0.5 → 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 nm, stopping as soon
        as ΔE reaches the highest reported threshold (5.0) OR the
        perturbed thickness would fall outside [MIN_THICKNESS_NM,
        MAX_THICKNESS_NM].

     b. Bisection phase: for each threshold (ΔE = 2, 3, 5), find the
        smallest probe with ΔE ≥ threshold. Bisect [previous, that
        probe] until the interval is narrower than BISECT_TOL_NM
        (default 0.05 nm). That upper endpoint is the reported Δnm.

   If NO probe reached a threshold (e.g. a fully opaque metal layer),
   the crossing is marked **right-censored** at MAX_PROBE_NM (default
   128 nm). The structure is kept and counted — this is a key
   difference from the previous version, which dropped opaque
   structures entirely and biased the distribution of "min Δnm
   needed" toward sensitive layers.

   Costs ~7–9 doubling probes + ~4–6 bisection probes per threshold,
   with probes shared across thresholds where possible. Typical total:
   15–25 sims per side, ~30–50 per layer. Compare to a fixed ±15 nm /
   0.25 nm sweep grid at 121 sims per layer.

3. **Cross-structure parallelism.** The per-structure sweep is
   embarrassingly parallel; a `ProcessPoolExecutor` with per-worker
   simulator initialisation gives ~N_JOBS× speedup. The HC directed
   search stays serial (it has its own RNG state; parallelising it
   would change the sampled structures).

Grid-comparison analysis
------------------------

Given a candidate grid, the worst-case snapping error for a layer at
base thickness t is half the local bin width, multiplied by the local
|dΔE/dnm| (which we now measure precisely, via the ±0.5 nm probe used
for layer selection). We report p50 / p95 / max snap ΔE per grid:

    - linear 5 nm         (original default)
    - linear 2 nm         (2.5x finer, uniform)
    - linear 1 nm         (5x finer, uniform)
    - piecewise 1/2/5 nm  (finest at small t)
    - log-nm (ratio 1.10) (~50 bins, geometric)
    - log-nm (ratio 1.05) (~100 bins, geometric)

Outputs
-------
    <out>/sensitivity.json                     raw probes + crossings + aggregates
    <out>/per_layer.csv                        one row per chosen layer (summary
                                               stats: local slope, Δnm-to-ΔE₂/₃/₅,
                                               censoring flags)
    <out>/sweeps_long.csv                      one row per (structure, probe) point
    <out>/curves_examples.png                  ΔE(Δnm), sample of structures
    <out>/sensitivity_by_bin.png               |dΔE/dnm| by base-thickness bin
    <out>/delta_e_2_by_bin.png                 Δnm for ΔE=2, split by source,
                                               with per-bin censored fraction
    <out>/delta_e_3_by_bin.png                 Δnm for ΔE=3, same
    <out>/grid_comparison.png                  snap ΔE per grid, all sources
    <out>/grid_comparison_by_source.png        p95 snap ΔE, HC vs random
    <out>/grid_comparison.txt                  printable tables

Right-censoring: when a threshold isn't reached within MAX_PROBE_NM
(128 nm), the crossing is reported as MAX_PROBE_NM with cens_de*_*
= 1. The delta_nm plots exclude censored points from the box and
annotate the fraction censored above each box; the grid comparison
uses the local slope (always defined) so opaque layers correctly
contribute near-zero snap cost regardless of censoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# File lives at <repo>/analyses/thickness_sensitivity/thickness_sensitivity.py
# — walk up three parents to reach <repo>.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.material_features import MaterialNK, load_jll_directory


# ============================================================================
# Constants
# ============================================================================

DOUBLING_SCHEDULE_NM: Tuple[float, ...] = (
    0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
)
MAX_PROBE_NM: float = 128.0
THRESHOLDS_DE: Tuple[float, ...] = (2.0, 3.0, 5.0)
BISECT_TOL_NM: float = 0.05
SLOPE_PROBE_NM: float = 0.5
MIN_THICKNESS_NM: float = 1.0
MAX_THICKNESS_NM: float = 300.0


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
    if mat.name in _KNOWN_METALS:
        return "metal"
    k_avg = float(np.mean(np.abs(mat.k)))
    if k_avg > 0.5:
        return "metal"
    if k_avg > 0.05:
        return "absorbing"
    return "dielectric"


# ============================================================================
# Adaptive probe (doubling + bisection)
# ============================================================================

def _eval_probe(
    sim, mats: List[MaterialNK], thicks: List[float], layer_idx: int,
    sign: int, d: float, base_lab: Tuple[float, float, float],
    cache: Dict[float, float],
    min_nm: float = MIN_THICKNESS_NM, max_nm: float = MAX_THICKNESS_NM,
) -> float:
    """Evaluate ΔE at `thicks[layer_idx] += sign*d`. NaN if out of bounds.

    Cached by unsigned d so callers can seed the cache with a prior
    probe (e.g. the ±0.5 slope probe used for layer selection).
    """
    if d in cache:
        return cache[d]
    t_new = float(thicks[layer_idx]) + sign * d
    if t_new < min_nm or t_new > max_nm:
        cache[d] = float("nan")
        return float("nan")
    thicks_new = list(thicks)
    thicks_new[layer_idx] = t_new
    lab = sim.compute_lab(mats, thicks_new)
    de = _ciede2000(base_lab, lab)
    cache[d] = de
    return de


def probe_direction(
    sim, mats: List[MaterialNK], thicks: List[float], layer_idx: int,
    base_lab: Tuple[float, float, float], sign: int,
    seed_probes: Optional[Dict[float, float]] = None,
    schedule: Tuple[float, ...] = DOUBLING_SCHEDULE_NM,
    thresholds: Tuple[float, ...] = THRESHOLDS_DE,
    bisect_tol: float = BISECT_TOL_NM,
    cap_nm: float = MAX_PROBE_NM,
    min_nm: float = MIN_THICKNESS_NM, max_nm: float = MAX_THICKNESS_NM,
) -> Dict[str, Any]:
    """Adaptive one-sided probe of a layer's ΔE(Δnm) curve.

    Doubling stops at the first probe with ΔE ≥ max(thresholds), or at
    the schedule ceiling, or at a boundary hit. Bisection then refines
    each threshold's crossing to ±bisect_tol precision.

    Returns:
        probes:     dict{d: de} — every ΔE evaluated (unsigned d, NaN if
                    the perturbed thickness fell outside bounds)
        crossings:  dict{thresh: dnm} — dnm at which ΔE first reaches
                    threshold (== cap_nm if censored)
        censored:   dict{thresh: bool}
        max_de_seen: float — for diagnostics
    """
    cache: Dict[float, float] = dict(seed_probes) if seed_probes else {}
    highest = max(thresholds)

    def _eval(d: float) -> float:
        return _eval_probe(sim, mats, thicks, layer_idx, sign, d,
                           base_lab, cache, min_nm, max_nm)

    # Doubling phase.
    for d in schedule:
        de = _eval(d)
        if math.isnan(de):
            break
        if de >= highest:
            break

    def _valid_sorted() -> List[Tuple[float, float]]:
        return sorted(
            (d, v) for d, v in cache.items() if not math.isnan(v)
        )

    crossings: Dict[float, float] = {}
    censored: Dict[float, bool] = {}
    for thresh in thresholds:
        pts = _valid_sorted()
        lo, hi = 0.0, None
        for d, de in pts:
            if de < thresh:
                lo = d
            elif hi is None:
                hi = d
                break
        if hi is None:
            crossings[thresh] = cap_nm
            censored[thresh] = True
            continue
        while hi - lo > bisect_tol:
            mid = 0.5 * (lo + hi)
            mid_de = _eval(mid)
            if math.isnan(mid_de):
                break
            if mid_de < thresh:
                lo = mid
            else:
                hi = mid
        crossings[thresh] = hi
        censored[thresh] = False

    max_de = max(
        (v for v in cache.values() if not math.isnan(v)), default=0.0
    )
    return {
        "probes": dict(cache),
        "crossings": crossings,
        "censored": censored,
        "max_de_seen": max_de,
    }


def choose_layer_by_slope(
    sim, mats: List[MaterialNK], thicks: List[float],
    base_lab: Tuple[float, float, float],
    probe_d_nm: float = SLOPE_PROBE_NM,
    min_nm: float = MIN_THICKNESS_NM, max_nm: float = MAX_THICKNESS_NM,
) -> Tuple[Optional[int], List[float], List[Dict[int, float]]]:
    """Peak-slope layer selection via ±probe_d_nm finite difference.

    Returns:
        chosen_idx  — layer with the largest max(|de_+|, |de_-|) / probe_d,
                      or None if all layers hit thickness bounds on both sides
        slopes      — per-layer peak slope (NaN if no valid side)
        probes      — per-layer dict{sign: de} for the sides that were valid
    """
    n = len(thicks)
    slopes: List[float] = [float("nan")] * n
    probes: List[Dict[int, float]] = [{} for _ in range(n)]
    for i in range(n):
        base = float(thicks[i])
        for sign in (+1, -1):
            t_new = base + sign * probe_d_nm
            if not (min_nm <= t_new <= max_nm):
                continue
            thicks_new = list(thicks)
            thicks_new[i] = t_new
            lab = sim.compute_lab(mats, thicks_new)
            probes[i][sign] = _ciede2000(base_lab, lab)
        if probes[i]:
            slopes[i] = max(abs(v) for v in probes[i].values()) / probe_d_nm
    valid = [(s, i) for i, s in enumerate(slopes) if math.isfinite(s)]
    if not valid:
        return None, slopes, probes
    _, chosen = max(valid)
    return chosen, slopes, probes


# ============================================================================
# One-structure sweep (layer selection + adaptive probe)
# ============================================================================

@dataclass
class StructureSweep:
    structure_id: int
    structure_source: str
    chosen_layer_idx: int
    layer_depth_from_top: int
    material_name: str
    material_category: str
    base_thickness_nm: int
    achieved_lab: Tuple[float, float, float]
    local_slope_dE_per_nm: float
    all_layer_slopes: List[float]
    max_de_seen_neg: float
    max_de_seen_pos: float
    crossings_neg: Dict[float, float]
    crossings_pos: Dict[float, float]
    censored_neg: Dict[float, bool]
    censored_pos: Dict[float, bool]
    # Merged bi-directional probe list (signed dnm), sorted ascending.
    sweep_delta_nm: List[float]
    sweep_delta_e: List[float]


def sweep_structure(
    sim, structure: Dict,
    schedule: Tuple[float, ...] = DOUBLING_SCHEDULE_NM,
    thresholds: Tuple[float, ...] = THRESHOLDS_DE,
    bisect_tol: float = BISECT_TOL_NM,
    cap_nm: float = MAX_PROBE_NM,
    slope_probe_nm: float = SLOPE_PROBE_NM,
) -> Optional[StructureSweep]:
    """Layer-select by slope, then adaptive-probe the chosen layer both ways.

    Returns None only when the structure has ZERO layers where any
    thickness perturbation is in bounds — vanishingly rare (would
    require every layer at base = MIN_THICKNESS_NM = MAX_THICKNESS_NM).
    Opaque layers do NOT return None; they return finite slopes near
    zero and censored crossings.
    """
    mats = structure["materials"]
    thicks = structure["thicknesses_nm"]
    base_lab = tuple(structure["achieved_lab"])
    if not thicks:
        return None

    chosen, slopes, slope_probes = choose_layer_by_slope(
        sim, mats, thicks, base_lab, probe_d_nm=slope_probe_nm,
    )
    if chosen is None:
        return None

    seed_pos = ({slope_probe_nm: slope_probes[chosen][+1]}
                if +1 in slope_probes[chosen] else None)
    seed_neg = ({slope_probe_nm: slope_probes[chosen][-1]}
                if -1 in slope_probes[chosen] else None)

    pos = probe_direction(
        sim, mats, thicks, chosen, base_lab, sign=+1,
        seed_probes=seed_pos, schedule=schedule, thresholds=thresholds,
        bisect_tol=bisect_tol, cap_nm=cap_nm,
    )
    neg = probe_direction(
        sim, mats, thicks, chosen, base_lab, sign=-1,
        seed_probes=seed_neg, schedule=schedule, thresholds=thresholds,
        bisect_tol=bisect_tol, cap_nm=cap_nm,
    )

    sweep_delta_nm: List[float] = []
    sweep_delta_e: List[float] = []
    for d, de in sorted(neg["probes"].items(), reverse=True):
        if not math.isnan(de):
            sweep_delta_nm.append(-d)
            sweep_delta_e.append(de)
    sweep_delta_nm.append(0.0)
    sweep_delta_e.append(0.0)
    for d, de in sorted(pos["probes"].items()):
        if not math.isnan(de):
            sweep_delta_nm.append(d)
            sweep_delta_e.append(de)

    return StructureSweep(
        structure_id=structure["id"],
        structure_source=structure["structure_source"],
        chosen_layer_idx=chosen,
        layer_depth_from_top=len(thicks) - 1 - chosen,
        material_name=mats[chosen].name,
        material_category=_classify(mats[chosen]),
        base_thickness_nm=int(thicks[chosen]),
        achieved_lab=tuple(base_lab),
        local_slope_dE_per_nm=slopes[chosen],
        all_layer_slopes=list(slopes),
        max_de_seen_neg=neg["max_de_seen"],
        max_de_seen_pos=pos["max_de_seen"],
        crossings_neg=neg["crossings"],
        crossings_pos=pos["crossings"],
        censored_neg=neg["censored"],
        censored_pos=pos["censored"],
        sweep_delta_nm=sweep_delta_nm,
        sweep_delta_e=sweep_delta_e,
    )


# ============================================================================
# Grid comparison
# ============================================================================

def linear_grid_bin_width(step_nm: float):
    return lambda t: step_nm


def log_grid_bin_width(ratio: float, t_min: float = 5.0,
                       t_max: float = 200.0):
    def w(t: float) -> float:
        return max(0.5, float(t) * (ratio - 1.0))
    return w


def piecewise_grid_bin_width(pieces: List[Tuple[float, float, float]]):
    def w(t: float) -> float:
        step = pieces[-1][2]
        for lo, hi, s in pieces:
            if lo <= t <= hi:
                step = s
        return float(step)
    return w


def evaluate_grid(
    per_layer_stats: List[Dict],
    bin_width_fn,
    name: str,
    threshold_de: Tuple[float, ...] = (2.0, 3.0),
) -> Dict[str, float]:
    costs = []
    for r in per_layer_stats:
        slope = r["local_slope_dE_per_nm"]
        base = r["base_thickness_nm"]
        if not math.isfinite(slope) or slope <= 0:
            continue
        w = bin_width_fn(base)
        costs.append(0.5 * w * slope)
    if not costs:
        return {"grid": name, "n_layers": 0}
    arr = np.asarray(costs)
    out: Dict[str, float] = {
        "grid": name,
        "n_layers": int(len(costs)),
        "median_snap_dE": float(np.median(arr)),
        "p95_snap_dE": float(np.percentile(arr, 95)),
        "max_snap_dE": float(arr.max()),
    }
    for t in threshold_de:
        out[f"n_layers_over_{int(t)}"] = int((arr > t).sum())
    return out


# ============================================================================
# Plotting
# ============================================================================

def plot_example_curves(sweeps: List[StructureSweep], out_path: Path,
                        n_examples: int = 12) -> None:
    rng = np.random.default_rng(0)
    if len(sweeps) > n_examples:
        chosen = rng.choice(len(sweeps), size=n_examples,
                            replace=False).tolist()
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
                       "dielectric": "tab:blue"}.get(
                           s.material_category, "gray"),
                marker="o", markersize=3, lw=1.4)
        ax.axhline(2.0, color="k", lw=0.5, alpha=0.4)
        ax.axhline(3.0, color="k", lw=0.5, alpha=0.2)
        ax.set_title(f"{s.material_name} @ {s.base_thickness_nm} nm "
                     f"({s.material_category})", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax_i // ncols == nrows - 1:
            ax.set_xlabel("Δt (nm)", fontsize=8)
        if ax_i % ncols == 0:
            ax.set_ylabel("ΔE_00", fontsize=8)
    for j in range(len(chosen), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Adaptive probe of most-sensitive layer  "
                 "(dashed = ΔE=2 / ΔE=3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_label(lo: float, hi: float) -> str:
    return f"{int(lo)}–{int(hi)} nm"


def plot_slope_by_bin(per_layer_stats: List[Dict], out_path: Path,
                      bin_edges=(5, 20, 40, 80, 120, 200)) -> None:
    edges = list(bin_edges)
    groups: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    for r in per_layer_stats:
        s = r["local_slope_dE_per_nm"]
        t = r["base_thickness_nm"]
        if not math.isfinite(s):
            continue
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1] or (i == len(edges) - 2
                                                and t == edges[-1]):
                groups[i].append(s)
                break
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(groups))
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
    ax.set_ylabel("|dΔE/dnm|  (peak slope near Δnm=0)")
    ax.set_xlabel("base thickness bin")
    ax.set_title("Local thickness sensitivity of the most-sensitive layer "
                 "per structure")
    ax.axhline(0.2, color="tab:red", lw=0.7, linestyle=":",
               label="0.2 ΔE/nm (5 nm grid → ~0.5 ΔE snap-error)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_nm_for_de(
    per_layer_stats: List[Dict],
    target_de: float,
    out_path: Path,
    bin_edges=(5, 20, 40, 80, 120, 200),
    split_by_source: bool = True,
    cap_nm: float = MAX_PROBE_NM,
) -> None:
    """Δnm to reach ΔE=target_de, per base-thickness bin, split by source.

    Censored rows (crossing did not occur within cap_nm on either side)
    are excluded from the box body but their fraction is annotated
    above each box as `⌐N%` — the "N% of structures in this bin were
    still under ΔE=target at ±cap_nm" reading.
    """
    edges = list(bin_edges)
    de_key_neg = f"dnm_de{int(target_de)}_neg"
    de_key_pos = f"dnm_de{int(target_de)}_pos"
    cens_key_neg = f"cens_de{int(target_de)}_neg"
    cens_key_pos = f"cens_de{int(target_de)}_pos"
    sources = ["high_chroma_search", "random"] if split_by_source else [None]
    groups = {src: [[] for _ in range(len(edges) - 1)] for src in sources}
    counts = {src: [0 for _ in range(len(edges) - 1)] for src in sources}
    censored_counts = {src: [0 for _ in range(len(edges) - 1)] for src in sources}
    for r in per_layer_stats:
        t = r["base_thickness_nm"]
        src = r.get("structure_source") if split_by_source else None
        if src not in groups:
            continue
        bin_i = None
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1] or (i == len(edges) - 2
                                                and t == edges[-1]):
                bin_i = i
                break
        if bin_i is None:
            continue
        counts[src][bin_i] += 1
        cens_neg = bool(r.get(cens_key_neg))
        cens_pos = bool(r.get(cens_key_pos))
        # Take the smallest crossing across sides, only counting sides
        # that actually crossed. If both sides are censored, the whole
        # row is censored.
        cands = []
        if not cens_neg:
            v = r.get(de_key_neg)
            if v is not None and math.isfinite(v):
                cands.append(v)
        if not cens_pos:
            v = r.get(de_key_pos)
            if v is not None and math.isfinite(v):
                cands.append(v)
        if not cands:
            censored_counts[src][bin_i] += 1
        else:
            groups[src][bin_i].append(min(cands))

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    n_bins = len(edges) - 1
    positions_center = np.arange(n_bins)
    colors = {"high_chroma_search": "#c86b6b", "random": "#7ea3d9"}

    if split_by_source:
        widths = 0.35
        offsets = {"high_chroma_search": -widths / 2 - 0.02,
                   "random": +widths / 2 + 0.02}
        for src in sources:
            data = [g if g else [0.0] for g in groups[src]]
            ax.boxplot(
                data, positions=positions_center + offsets[src],
                widths=widths, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=colors[src], alpha=0.55,
                              edgecolor="#333"),
                medianprops=dict(color="#111"),
            )
            for i, g in enumerate(groups[src]):
                ax.scatter([positions_center[i] + offsets[src]] * len(g),
                           g, s=5, color="k", alpha=0.35)
            n_each = [len(g) for g in groups[src]]
            ax.plot([], [], color=colors[src], lw=6, alpha=0.7,
                    label=f"{src}  (n per bin = "
                          f"{','.join(str(x) for x in n_each)})")
        # Annotate % censored above each box.
        for src in sources:
            for i in range(n_bins):
                total = counts[src][i]
                if total == 0:
                    continue
                cens = censored_counts[src][i]
                if cens == 0:
                    continue
                frac = 100.0 * cens / total
                x = positions_center[i] + offsets[src]
                ax.annotate(
                    f"⌐{frac:.0f}%",
                    xy=(x, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1),
                    ha="center", va="top", fontsize=7,
                    color="#555",
                )
    else:
        data = [g if g else [0.0] for g in groups[None]]
        ax.boxplot(data, positions=positions_center, widths=0.6,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#f5e2df", edgecolor="#653"),
                   medianprops=dict(color="#653"))
        for i, g in enumerate(groups[None]):
            ax.scatter([positions_center[i]] * len(g), g, s=6,
                       color="k", alpha=0.4)

    ax.axhline(5.0, color="tab:blue", lw=0.7, linestyle=":",
               label="current grid spacing (5 nm)")
    ax.axhline(2.5, color="tab:green", lw=0.7, linestyle=":",
               label="candidate 2 nm grid (half-width)")
    ax.set_xticks(positions_center)
    ax.set_xticklabels([_bin_label(edges[i], edges[i + 1])
                        for i in range(n_bins)])
    ax.set_ylabel(f"Δnm needed to reach ΔE = {target_de:g}")
    ax.set_xlabel("base thickness bin")
    ax.set_title(
        f"Perceptibility distance: smallest Δnm on the MOST-SENSITIVE layer "
        f"that changes colour by ΔE={target_de:g}\n"
        f"(⌐N% = fraction of that bin still under ΔE={target_de:g} at "
        f"±{cap_nm:g} nm — right-censored, not plotted in the box)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_comparison(grid_results: List[Dict], out_path: Path,
                         suptitle: str = "Grid granularity vs "
                                          "perceptual snap cost") -> None:
    labels = [g["grid"] for g in grid_results]
    p50 = [g.get("median_snap_dE", 0) for g in grid_results]
    p95 = [g.get("p95_snap_dE", 0) for g in grid_results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, p50, width=0.4, label="median snap ΔE", color="#7ea3d9")
    ax.bar(x + 0.2, p95, width=0.4, label="p95 snap ΔE", color="#c86b6b")
    ax.axhline(2.0, color="k", lw=0.5, linestyle="--",
               label="perceptible-diff threshold (ΔE 2)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("expected ΔE from snapping to grid")
    ax.set_title(suptitle)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_comparison_split(
    grid_results_by_source: Dict[str, List[Dict]], out_path: Path,
) -> None:
    sources = list(grid_results_by_source.keys())
    all_grid_names = [g["grid"] for g in grid_results_by_source[sources[0]]]
    x = np.arange(len(all_grid_names))
    width = 0.8 / max(len(sources), 1)
    colors = {"high_chroma_search": "#c86b6b", "random": "#7ea3d9"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, src in enumerate(sources):
        p95 = [g.get("p95_snap_dE", 0)
               for g in grid_results_by_source[src]]
        ax.bar(x + (i - (len(sources) - 1) / 2) * width, p95,
               width=width * 0.9,
               color=colors.get(src, f"C{i}"),
               label=f"{src} (p95 snap ΔE)")
    ax.axhline(2.0, color="k", lw=0.5, linestyle="--", label="ΔE 2 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_grid_names, rotation=15, ha="right")
    ax.set_ylabel("expected ΔE from snapping (p95)")
    ax.set_title("Grid granularity vs snap cost — high-chroma vs random")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Multiprocessing worker
# ============================================================================
#
# The per-structure sweep is embarrassingly parallel: each structure's
# sweep depends only on its own materials + thicknesses. We spawn N_JOBS
# workers, each with its own RandomLayerSimulation constructed in the
# initializer (so JLL loading + JAX warmup happen once per worker).
#
# The HC directed search and random-structure generation stay serial —
# both consume the main-process RNG state, and parallelising them would
# change which structures come out.

_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(
    jll_dir_str: str, seed: int, p_real: float, lam: float,
    min_layers: int, max_layers: int, incidence_angle: float,
) -> None:
    global _WORKER_STATE
    from create_dataset.src.pool_sampler import split_jll_real
    from create_dataset.src.random_layer import (
        LayerCountConfig, RandomLayerSimulation,
    )
    jll = load_jll_directory(Path(jll_dir_str))
    active, _ = split_jll_real(jll)
    layer_count = LayerCountConfig(
        lam=lam, min_layers=min_layers, max_layers=max_layers,
    )
    _WORKER_STATE["sim"] = RandomLayerSimulation(
        held_in_real=active, layer_count=layer_count,
        incidence_angle=incidence_angle, p_real=p_real, seed=seed,
    )


def _worker_sweep(structure: Dict) -> Optional[StructureSweep]:
    return sweep_structure(_WORKER_STATE["sim"], structure)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-structures", type=int, default=60,
                    help="how many structures PER SOURCE to generate + probe. "
                         "Total sweeps = 2 * n_structures (high_chroma + random).")
    ap.add_argument("--cap-nm", type=float, default=MAX_PROBE_NM,
                    help="largest |Δnm| in the doubling schedule. Crossings "
                         "not found by this reach are marked right-censored.")
    ap.add_argument("--slope-probe-nm", type=float, default=SLOPE_PROBE_NM,
                    help="±probe distance for the per-layer finite-difference "
                         "slope used for layer selection.")
    ap.add_argument("--bisect-tol-nm", type=float, default=BISECT_TOL_NM,
                    help="stop bisecting a threshold when the bracket is "
                         "narrower than this.")
    ap.add_argument("--high-chroma-candidate-count", type=int, default=24)
    ap.add_argument("--high-chroma-refine-iters", type=int, default=12)
    ap.add_argument("--jll-materials-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel workers for the sweep phase. HC search "
                         "and random-gen stay serial. Set to --cpus-per-task "
                         "from your SLURM allocation.")
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

    lam = 4.5
    min_layers = 2
    max_layers = 10
    p_real = 0.15
    incidence_angle = 0

    layer_count = LayerCountConfig(
        lam=lam, min_layers=min_layers, max_layers=max_layers,
    )
    sim = RandomLayerSimulation(
        held_in_real=active, layer_count=layer_count,
        incidence_angle=incidence_angle, p_real=p_real, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    target_cfg = HighChromaTargetConfig()
    search_cfg = HighChromaSearchConfig(
        candidate_count=args.high_chroma_candidate_count,
        refine_iters=args.high_chroma_refine_iters,
    )

    # 1. Generate N high-chroma + N undirected-random structures.
    structures: List[Dict] = []
    print(f"[INFO] Generating {args.n_structures} high-chroma structures…")
    t0 = time.time()
    for i in range(args.n_structures):
        target = sample_high_chroma_target_lab(rng, target_cfg)
        materials, thicks_snapped, achieved_lab, _raw = \
            search_structure_for_target(sim, target, search_cfg, rng)
        structures.append({
            "id": len(structures),
            "structure_source": "high_chroma_search",
            "target_lab": target,
            "materials": materials,
            "thicknesses_nm": thicks_snapped,
            "achieved_lab": tuple(achieved_lab),
        })
    print(f"[INFO] high-chroma done in {time.time() - t0:.1f}s")

    print(f"[INFO] Generating {args.n_structures} random structures…")
    t0 = time.time()
    for i in range(args.n_structures):
        materials, thicks, lab = sim.sample_structure()
        thicks_snapped = [int(round(t / 5) * 5) for t in thicks]
        structures.append({
            "id": len(structures),
            "structure_source": "random",
            "target_lab": None,
            "materials": materials,
            "thicknesses_nm": thicks_snapped,
            "achieved_lab": tuple(lab),
        })
    print(f"[INFO] random done in {time.time() - t0:.1f}s")

    # 2. Adaptive sweep: pick most-sensitive layer per structure, then
    #    doubling+bisection on both sides.
    print(f"[INFO] Adaptive sweep: layer selection by ±{args.slope_probe_nm} nm "
          f"slope; doubling schedule up to ±{args.cap_nm} nm; bisect tol "
          f"{args.bisect_tol_nm} nm; thresholds ΔE ∈ {THRESHOLDS_DE}. "
          f"Workers: {args.n_jobs}.")
    t0 = time.time()

    if args.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=args.n_jobs,
            initializer=_init_worker,
            initargs=(str(jll_dir), args.seed, p_real, lam,
                      min_layers, max_layers, incidence_angle),
        ) as pool:
            raw_results = list(pool.map(_worker_sweep, structures,
                                         chunksize=1))
    else:
        raw_results = [sweep_structure(sim, s) for s in structures]

    all_sweeps: List[StructureSweep] = [
        r for r in raw_results if r is not None
    ]
    n_dropped_empty = len(structures) - len(all_sweeps)
    n_censored_both_sides = sum(
        1 for s in all_sweeps
        if all(s.censored_neg.values()) and all(s.censored_pos.values())
    )
    print(f"[INFO] Sweep done in {time.time() - t0:.1f}s "
          f"({len(all_sweeps)} sweeps; "
          f"{n_dropped_empty} structures had no in-bounds layer; "
          f"{n_censored_both_sides} sweeps are fully censored — "
          f"no threshold reached within ±{args.cap_nm:g} nm on either side)")

    # 3. Flatten to per_layer_stats records.
    per_layer_stats: List[Dict] = []
    for s in all_sweeps:
        row: Dict[str, Any] = {
            "structure_id": s.structure_id,
            "structure_source": s.structure_source,
            "chosen_layer_idx": s.chosen_layer_idx,
            "layer_depth_from_top": s.layer_depth_from_top,
            "material_name": s.material_name,
            "material_category": s.material_category,
            "base_thickness_nm": s.base_thickness_nm,
            "achieved_lab": list(s.achieved_lab),
            "local_slope_dE_per_nm": s.local_slope_dE_per_nm,
            "all_layer_slopes": s.all_layer_slopes,
            "max_de_seen_neg": s.max_de_seen_neg,
            "max_de_seen_pos": s.max_de_seen_pos,
        }
        for thresh in THRESHOLDS_DE:
            key_int = int(thresh)
            row[f"dnm_de{key_int}_neg"] = s.crossings_neg[thresh]
            row[f"dnm_de{key_int}_pos"] = s.crossings_pos[thresh]
            row[f"cens_de{key_int}_neg"] = bool(s.censored_neg[thresh])
            row[f"cens_de{key_int}_pos"] = bool(s.censored_pos[thresh])
        per_layer_stats.append(row)

    # 4. Grid comparison — overall AND per structure_source.
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

    def _print_grid_table(subset, title):
        header = ("grid                       n     median ΔE   p95 ΔE    "
                  "max ΔE   > 2 ΔE   > 3 ΔE")
        print(f"\n[grid comparison — {title}]")
        print(header)
        print("-" * len(header))
        rows: List[str] = []
        results: List[Dict] = []
        for name, fn in grids:
            g = evaluate_grid(subset, fn, name, threshold_de=(2.0, 3.0))
            results.append(g)
            if "median_snap_dE" not in g:
                continue
            line = (f"{g['grid']:26s}  {g['n_layers']:4d}   "
                    f"{g['median_snap_dE']:7.3f}    "
                    f"{g['p95_snap_dE']:7.3f}   "
                    f"{g['max_snap_dE']:7.3f}   "
                    f"{g['n_layers_over_2']:5d}    "
                    f"{g['n_layers_over_3']:5d}")
            print(line)
            rows.append(line)
        block = header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)
        return results, block

    results_all, table_all = _print_grid_table(per_layer_stats, "all sources")
    hc_subset = [r for r in per_layer_stats
                 if r["structure_source"] == "high_chroma_search"]
    rand_subset = [r for r in per_layer_stats
                   if r["structure_source"] == "random"]
    results_hc, table_hc = _print_grid_table(hc_subset, "high_chroma_search")
    results_rand, table_rand = _print_grid_table(rand_subset, "random")

    (args.output_dir / "grid_comparison.txt").write_text(
        "=== ALL SOURCES ===\n" + table_all + "\n\n"
        "=== HIGH-CHROMA SEARCH ===\n" + table_hc + "\n\n"
        "=== RANDOM ===\n" + table_rand + "\n"
    )

    # 5. Persist raw + aggregate data as JSON.
    payload: Dict[str, Any] = {
        "config": {
            "n_structures_per_source": args.n_structures,
            "layer_selection": "most_sensitive_by_slope",
            "slope_probe_nm": args.slope_probe_nm,
            "cap_nm": args.cap_nm,
            "bisect_tol_nm": args.bisect_tol_nm,
            "doubling_schedule_nm": list(DOUBLING_SCHEDULE_NM),
            "de_thresholds_reported": list(THRESHOLDS_DE),
            "high_chroma_candidate_count": args.high_chroma_candidate_count,
            "high_chroma_refine_iters": args.high_chroma_refine_iters,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
        },
        "grids_all_sources": results_all,
        "grids_high_chroma_search": results_hc,
        "grids_random": results_rand,
        "per_layer": per_layer_stats,
        "sweeps": [
            {
                "structure_id": s.structure_id,
                "structure_source": s.structure_source,
                "chosen_layer_idx": s.chosen_layer_idx,
                "material_name": s.material_name,
                "material_category": s.material_category,
                "base_thickness_nm": s.base_thickness_nm,
                "achieved_lab": list(s.achieved_lab),
                "sweep_delta_nm": s.sweep_delta_nm,
                "sweep_delta_e": s.sweep_delta_e,
            } for s in all_sweeps
        ],
    }
    (args.output_dir / "sensitivity.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(f"[json] wrote {args.output_dir / 'sensitivity.json'}")

    # 6. CSV exports. per_layer.csv is the flat summary. sweeps_long.csv
    #    is one row per (structure, probe) point — variable count per
    #    structure since the adaptive probe visits different Δnm's.
    per_layer_csv = args.output_dir / "per_layer.csv"
    fieldnames = [
        "structure_id", "structure_source",
        "chosen_layer_idx", "layer_depth_from_top",
        "material_name", "material_category",
        "base_thickness_nm",
        "L_base", "a_base", "b_base",
        "local_slope_dE_per_nm",
        "max_de_seen_neg", "max_de_seen_pos",
    ]
    for thresh in THRESHOLDS_DE:
        k = int(thresh)
        fieldnames.extend([
            f"dnm_de{k}_neg", f"dnm_de{k}_pos",
            f"cens_de{k}_neg", f"cens_de{k}_pos",
        ])

    with open(per_layer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_layer_stats:
            L, a, b = r["achieved_lab"]
            row = {
                "structure_id":         r["structure_id"],
                "structure_source":     r["structure_source"],
                "chosen_layer_idx":     r["chosen_layer_idx"],
                "layer_depth_from_top": r["layer_depth_from_top"],
                "material_name":        r["material_name"],
                "material_category":    r["material_category"],
                "base_thickness_nm":    r["base_thickness_nm"],
                "L_base": L, "a_base": a, "b_base": b,
                "local_slope_dE_per_nm": r["local_slope_dE_per_nm"],
                "max_de_seen_neg":       r["max_de_seen_neg"],
                "max_de_seen_pos":       r["max_de_seen_pos"],
            }
            for thresh in THRESHOLDS_DE:
                k = int(thresh)
                row[f"dnm_de{k}_neg"]  = r[f"dnm_de{k}_neg"]
                row[f"dnm_de{k}_pos"]  = r[f"dnm_de{k}_pos"]
                row[f"cens_de{k}_neg"] = int(bool(r[f"cens_de{k}_neg"]))
                row[f"cens_de{k}_pos"] = int(bool(r[f"cens_de{k}_pos"]))
            writer.writerow(row)
    print(f"[csv]  wrote {per_layer_csv} ({len(per_layer_stats)} rows)")

    sweeps_long_csv = args.output_dir / "sweeps_long.csv"
    n_rows = 0
    with open(sweeps_long_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "structure_id", "structure_source",
            "chosen_layer_idx", "material_name", "material_category",
            "base_thickness_nm",
            "delta_nm", "delta_e",
        ])
        writer.writeheader()
        for s in all_sweeps:
            for dnm, de in zip(s.sweep_delta_nm, s.sweep_delta_e):
                writer.writerow({
                    "structure_id":     s.structure_id,
                    "structure_source": s.structure_source,
                    "chosen_layer_idx": s.chosen_layer_idx,
                    "material_name":    s.material_name,
                    "material_category": s.material_category,
                    "base_thickness_nm": s.base_thickness_nm,
                    "delta_nm": dnm, "delta_e": de,
                })
                n_rows += 1
    print(f"[csv]  wrote {sweeps_long_csv} ({n_rows} rows)")

    # 7. Plots — deliberately LAST so a timeout during plotting can't
    #    destroy the raw data (JSON + CSVs).
    plot_example_curves(all_sweeps, args.output_dir / "curves_examples.png")
    plot_slope_by_bin(per_layer_stats,
                      args.output_dir / "sensitivity_by_bin.png")
    plot_delta_nm_for_de(per_layer_stats, target_de=2.0,
                         out_path=args.output_dir / "delta_e_2_by_bin.png",
                         cap_nm=args.cap_nm)
    plot_delta_nm_for_de(per_layer_stats, target_de=3.0,
                         out_path=args.output_dir / "delta_e_3_by_bin.png",
                         cap_nm=args.cap_nm)
    plot_grid_comparison(results_all,
                         args.output_dir / "grid_comparison.png",
                         suptitle="Grid snap cost — most-sensitive layer, "
                                   "all sources")
    plot_grid_comparison_split(
        {"high_chroma_search": results_hc, "random": results_rand},
        args.output_dir / "grid_comparison_by_source.png",
    )
    print(f"\n[plots] wrote 6 PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
