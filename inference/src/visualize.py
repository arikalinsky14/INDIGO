"""
Inference-time visualizations.

  Result JSON   ─►   one composite PNG per run with:
                     1. Target / achieved color swatches (sRGB) with ΔE
                     2. Reflectance spectrum of the chosen structure
                     3. Layer diagram (material × thickness bars)

Designed for headless slurm runs — uses matplotlib's Agg backend and writes
PNGs to disk; never opens a window. If matplotlib isn't installed we skip
silently and let the rest of the pipeline carry on (a Result JSON without
images is still useful).

The figure layout
-----------------
  ┌──────────────────────────────────────────────────────────┐
  │  Target [Lab]    |    Achieved [Lab]    |    ΔE = 0.59   │   swatches
  ├──────────────────────────────────────────────────────────┤
  │                                                          │
  │   Reflectance vs wavelength (chosen, optionally + alts)  │
  │                                                          │
  ├──────────────────────────────────────────────────────────┤
  │     │██████   │████████   │████  │██████████             │   layer bars
  │     Ag(30)    SiO2(100)   ...                            │
  └──────────────────────────────────────────────────────────┘

Failure modes
-------------
- matplotlib missing  → emit warning, return None.
- Result.chosen is None (failure-Result) → emit a single "FAILED" swatch
  panel so the GUI / human report still shows that something happened.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.material_features import CANONICAL_LAMBDA_NM
from src.color_utils import lab_to_srgb_int

from inference.src.schema import Candidate, Result


def _lab_to_mpl_rgb(lab) -> tuple:
    """Lab (L*, a*, b*) → matplotlib RGB tuple in [0, 1].

    sRGB clipping is handled inside lab_to_srgb_int (clamped to gamut).
    """
    r, g, b = lab_to_srgb_int(list(lab))
    return (r / 255.0, g / 255.0, b / 255.0)


def _stack_summary(cand: Candidate) -> str:
    """One-line human summary of the layer stack."""
    if cand is None or not cand.material_names:
        return "(no structure)"
    parts = [f"{m}@{t:.0f}nm"
             for m, t in zip(cand.material_names, cand.thicknesses_nm)]
    return "  ─  ".join(parts)


def _draw_swatch_panel(ax, target_lab, achieved_lab, delta_e: float) -> None:
    """Two side-by-side color squares with Lab labels and ΔE in the middle."""
    from matplotlib.patches import Rectangle
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    target_rgb = _lab_to_mpl_rgb(target_lab)
    achieved_rgb = _lab_to_mpl_rgb(achieved_lab)

    ax.add_patch(Rectangle((0.5, 0.5), 3.5, 3.0, facecolor=target_rgb,
                            edgecolor="black", linewidth=1.5))
    ax.add_patch(Rectangle((6.0, 0.5), 3.5, 3.0, facecolor=achieved_rgb,
                            edgecolor="black", linewidth=1.5))

    ax.text(2.25, 3.7, "TARGET", ha="center", fontsize=10, weight="bold")
    ax.text(2.25, 0.2, f"L*={target_lab[0]:.1f}  a*={target_lab[1]:.1f}  "
            f"b*={target_lab[2]:.1f}", ha="center", fontsize=9)

    ax.text(7.75, 3.7, "ACHIEVED", ha="center", fontsize=10, weight="bold")
    ax.text(7.75, 0.2, f"L*={achieved_lab[0]:.1f}  a*={achieved_lab[1]:.1f}  "
            f"b*={achieved_lab[2]:.1f}", ha="center", fontsize=9)

    # ΔE callout in the gap.
    ax.text(5.0, 2.0, f"ΔE_00\n{delta_e:.3f}", ha="center", va="center",
            fontsize=14, weight="bold",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5",
                  "edgecolor": "gray"})


def _draw_reflectance_panel(ax, chosen: Candidate,
                            alternatives: Optional[List[Candidate]] = None,
                            ) -> None:
    """Reflectance vs wavelength for the chosen candidate; alts as faint lines."""
    lam = np.asarray(CANONICAL_LAMBDA_NM, dtype=np.float64)
    R = np.asarray(chosen.reflectance, dtype=np.float64)

    # Faint alternatives behind the chosen line so the chosen stays readable.
    if alternatives:
        for alt in alternatives:
            R_alt = np.asarray(alt.reflectance, dtype=np.float64)
            ax.plot(lam, R_alt, color="gray", alpha=0.25, linewidth=1.0)

    ax.plot(lam, R, color="C0", linewidth=2.0, label="chosen")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(lam.min(), lam.max())
    # Clip y to a sensible window unless the spectrum legitimately exceeds 1.
    y_hi = float(min(1.05, max(0.8, R.max() * 1.1)))
    ax.set_ylim(0, y_hi)
    # Visible-band shading for orientation (380–780 nm).
    ax.axvspan(380, 780, color="C2", alpha=0.05, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.set_title(_stack_summary(chosen), fontsize=10, loc="left")
    if alternatives:
        ax.legend(loc="upper right", fontsize=9)


def _draw_layer_diagram(ax, cand: Candidate) -> None:
    """Stacked horizontal bar — each layer drawn at its physical thickness.

    Renders bottom-up so the substrate end is at the left, the topmost
    decoded layer at the right (matches how the AR model decodes).
    """
    ax.axis("off")
    if not cand.material_names:
        ax.text(0.5, 0.5, "(no layers)", ha="center", va="center",
                fontsize=12, color="gray")
        return

    total = float(sum(cand.thicknesses_nm)) or 1.0
    x = 0.0
    palette = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
               "#06b6d4", "#ec4899", "#84cc16", "#a855f7", "#f97316"]
    name_to_color: dict = {}
    for name in cand.material_names:
        if name not in name_to_color:
            name_to_color[name] = palette[len(name_to_color) % len(palette)]

    from matplotlib.patches import Rectangle
    height = 1.0
    for name, thick in zip(cand.material_names, cand.thicknesses_nm):
        w = thick / total
        ax.add_patch(Rectangle((x, 0), w, height,
                                facecolor=name_to_color[name],
                                edgecolor="white", linewidth=1.0))
        # Label inside the bar when there's space, below when there isn't.
        label = f"{name}\n{thick:.0f}nm"
        if w > 0.07:
            ax.text(x + w / 2, height / 2, label, ha="center", va="center",
                    fontsize=8, color="white", weight="bold")
        x += w
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, height + 0.05)
    ax.set_title(f"Total thickness: {int(total)} nm  "
                 f"({len(cand.material_names)} layers)", fontsize=9, loc="left")


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def render_result(result: Result, out_path: Path,
                  include_alternatives: bool = True) -> Optional[Path]:
    """Write a composite PNG summarising the Result. Returns the path on
    success, None if matplotlib is unavailable.

    Layout: swatches | reflectance | layer diagram.

    Failure-Results (chosen=None) get a single "FAILED" swatch panel so the
    file still exists and a downstream GUI can render the failure case with
    one code path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualize] matplotlib not available — skipping image", flush=True)
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if result.chosen is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.6, "INFERENCE FAILED", ha="center", va="center",
                fontsize=16, weight="bold", color="C3")
        errs = "; ".join(result.errors[:3]) if result.errors else "(no error message)"
        ax.text(0.5, 0.3, errs, ha="center", va="center", fontsize=10,
                wrap=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    chosen = result.chosen
    alternatives = result.alternatives if include_alternatives else []

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 0.8],
                          hspace=0.3, left=0.07, right=0.95, top=0.94, bottom=0.06)
    ax_swatch = fig.add_subplot(gs[0])
    ax_refl   = fig.add_subplot(gs[1])
    ax_stack  = fig.add_subplot(gs[2])

    _draw_swatch_panel(ax_swatch,
                       target_lab=result.spec_echo.target_lab_raw,
                       achieved_lab=chosen.achieved_lab,
                       delta_e=chosen.delta_e)
    _draw_reflectance_panel(ax_refl, chosen, alternatives=alternatives)
    _draw_layer_diagram(ax_stack, chosen)

    # Footer: provenance + ensemble counts.
    s = result.ensemble_stats
    foot = (
        f"seed={result.provenance.seed}   "
        f"N={s.n_sampled}→unique{s.n_unique_after_dedup}→feasible{s.n_feasible}"
        f"→refined{s.n_refined}→returned{s.n_returned}   "
        f"R_l2={chosen.robustness.grad_l2_shift:.3f}   "
        f"J={chosen.objective:.3f}"
    )
    if chosen.robustness.mc_samples > 0:
        foot += (f"   MC (p50/p95/worst, K={chosen.robustness.mc_samples}): "
                 f"{chosen.robustness.mc_p50:.2f}/"
                 f"{chosen.robustness.mc_p95:.2f}/"
                 f"{chosen.robustness.mc_worst:.2f}")
    fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8,
             color="#444444")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
