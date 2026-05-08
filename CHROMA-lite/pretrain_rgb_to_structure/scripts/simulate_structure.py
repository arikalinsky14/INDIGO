#!/usr/bin/env python3
"""
simulate_structure.py

Simulate the reflectance spectrum and predicted sRGB color for a fixed
thin-film structure using the existing OpticalSimulator infrastructure.

Structure (top → bottom):
    TiN    55 nm
    Al2O3  85 nm
    Pd     15 nm
    Al     80 nm
    ─────────────
    Total: 235 nm

Usage:
    python scripts/simulate_structure.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── repo root on sys.path so absolute imports work ──────────────────────────
_repo_root = Path(__file__).resolve().parents[2]   # adjust depth if needed
sys.path.insert(0, str(_repo_root))

from src.optical_sim import OpticalSimulator, is_available

# ── Structure definition ─────────────────────────────────────────────────────
MATERIALS   = ["TiO2", "Ag"]  # top → bottom
THICKNESSES = [62, 50]          # nm, top → bottom

# ── Simulate ─────────────────────────────────────────────────────────────────
def main():
    if not is_available():
        print("[ERROR] jaxlayerlumos not available — install it first.")
        sys.exit(1)

    sim = OpticalSimulator(incidence_angle=0)

    R   = sim.compute_reflectance(MATERIALS, THICKNESSES)
    rgb = sim.compute_color(MATERIALS, THICKNESSES)

    print(f"Predicted sRGB : {rgb}")
    print(f"Hex            : #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: reflectance spectrum
    ax = axes[0]
    ax.plot(sim.wavelength_nm, R, color="steelblue", linewidth=1.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Reflectance spectrum")
    ax.set_xlim(sim.wavelength_nm[0], sim.wavelength_nm[-1])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Annotate structure
    struct_str = "\n".join(
        f"{m}  {t} nm" for m, t in zip(MATERIALS, THICKNESSES)
    )
    ax.text(0.97, 0.97, struct_str, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7))

    # Right: predicted color swatch
    ax2 = axes[1]
    ax2.set_facecolor([c / 255 for c in rgb])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"Predicted color\n#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}  {rgb}")
    for spine in ax2.spines.values():
        spine.set_edgecolor("gray")

    plt.tight_layout()
    out = Path(__file__).parent / "simulate_structure_output.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out}")
    plt.show()


if __name__ == "__main__":
    main()