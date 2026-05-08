#!/usr/bin/env python3
"""
Grant-Proposal Figure: Reflectance Spectra + Color Swatch Grid

Creates a compact, portrait-oriented figure with:
  - Top row: 2 reflectance spectra (GT vs Generated), lines colored by structure color
  - Bottom grid: 4x4 clean color swatch pairs (Target | Prediction) with dE labels

Designed to fit on a standard portrait page for grant proposals.

Usage:
    python pretrain_rgb_to_structure/scripts/plot_reflectance_comparison.py \
        --checkpoint data/checkpoints/mlp_d1024_L8_do0.1_lr0.0001_bs256_ep200/latest
"""

import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import denormalize_rgb
from src.dataset import ThinFilmDataset, find_repo_root
from src.optical_sim import OpticalSimulator, is_available as optical_is_available
from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig, generate_structure

# ============================================================================
# Configuration
# ============================================================================

N_SPECTRA = 2          # Reflectance spectrum plots (top row)
N_SWATCHES = 16        # Color swatch pairs (4x4 grid)
N_EXAMPLES = N_SPECTRA + N_SWATCHES
TEMPERATURE = 1.0
MIN_CHROMA = 15.0
MIN_SPEC_DIFF = 0.05

SWATCH_COLS = 4
SWATCH_ROWS = math.ceil(N_SWATCHES / SWATCH_COLS)

# ============================================================================
# Color Science
# ============================================================================

def sRGB_to_Lab(sRGB: List[int]) -> Tuple[float, float, float]:
    rgb = np.array(sRGB, dtype=np.float64) / 255.0
    linear_rgb = np.where(
        rgb <= 0.04045, rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = M @ linear_rgb
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / white
    def f(t):
        delta = 6 / 29
        return np.where(t > delta**3, t**(1/3), t / (3 * delta**2) + 4 / 29)
    fv = f(xyz_n)
    L = 116 * fv[1] - 16
    a = 500 * (fv[0] - fv[1])
    b = 200 * (fv[1] - fv[2])
    return float(L), float(a), float(b)


def lab_chroma(sRGB: List[int]) -> float:
    _, a, b = sRGB_to_Lab(sRGB)
    return math.sqrt(a**2 + b**2)


def ciede2000(lab1, lab2) -> float:
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
    dTh = 30 * math.exp(-((hbp - 275) / 25)**2)
    RC = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    SL = 1 + 0.015 * (Lbp - 50)**2 / math.sqrt(20 + (Lbp - 50)**2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2 * dTh)) * RC
    return math.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH)
    )


def compute_color_difference(rgb1, rgb2) -> float:
    return ciede2000(sRGB_to_Lab(rgb1), sRGB_to_Lab(rgb2))


# ============================================================================
# Model Utilities
# ============================================================================

def load_model(checkpoint_dir: Path, device: torch.device):
    config_path = checkpoint_dir / 'config.json'
    model_path = checkpoint_dir / 'model.pt'
    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    model = ThinFilmMLP(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, config


def format_structure(materials: List[str], thicknesses: List[int]) -> str:
    if not materials:
        return "(empty)"
    return "\n".join(f"  {m}: {t} nm" for m, t in zip(materials, thicknesses))


def generate_structure_tempered(model, rgb, device, temperature=0.5):
    """
    Generate structure with temperature sampling.
    
    Delegates to generate_structure with sample=True.
    If temperature <= 0, uses greedy decoding.
    """
    if temperature <= 0:
        return generate_structure(model, rgb, device)
    return generate_structure(model, rgb, device, sample=True, temperature=temperature)


def _text_color_for_bg(sRGB):
    """Return black or white text depending on background luminance."""
    r, g, b = sRGB
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000' if lum > 140 else '#fff'


# ============================================================================
# Plotting
# ============================================================================

def plot_spectrum(ax, sim, data, idx_label):
    """Plot a single GT vs Gen reflectance spectrum with inset swatches."""
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    R_gt = sim.compute_reflectance(data['gt_materials'], data['gt_thicknesses'])
    R_pred = sim.compute_reflectance(data['pred_materials'], data['pred_thicknesses'])

    gt_color = tuple(c / 255 for c in data['gt_sRGB'])
    pred_color = tuple(c / 255 for c in data['pred_sRGB'])

    ax.plot(sim.wavelength_nm, R_gt, color=gt_color, lw=2.0, ls='-', label='Target')
    ax.plot(sim.wavelength_nm, R_pred, color=pred_color, lw=2.0, ls='--', label='Design')

    ax.set_xlabel('Wavelength (nm)', fontsize=8, labelpad=3)
    ax.set_ylabel('Reflectance', fontsize=8, labelpad=3)
    ax.set_xlim(400, 750)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=7)
    # FIX: reduced fontsize + pad to avoid collision with suptitle
    ax.set_title(f'Example {idx_label}  \u2014  \u0394E\u2080\u2080 = {data["delta_e"]:.2f}',
                 fontsize=8.5, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.25, lw=0.5)

    gt_c = [c / 255 for c in data['gt_sRGB']]
    pr_c = [c / 255 for c in data['pred_sRGB']]

    # FIX: inset swatches moved down (bbox_to_anchor y=-0.08) so T/P labels
    # don't collide with the plot title above
    ax_t = inset_axes(ax, width="12%", height="16%", loc='upper right',
                      bbox_to_anchor=(-0.15, -0.08, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    ax_t.set_facecolor(gt_c)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for sp in ax_t.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(0.8)
    # FIX: T/P labels INSIDE the swatch (not as titles above) to avoid overlap
    ax_t.text(0.5, 0.5, 'T', transform=ax_t.transAxes, fontsize=6,
              ha='center', va='center', fontweight='bold',
              color=_text_color_for_bg(data['gt_sRGB']), alpha=0.7)

    ax_p = inset_axes(ax, width="12%", height="16%", loc='upper right',
                      bbox_to_anchor=(0.0, -0.08, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    ax_p.set_facecolor(pr_c)
    ax_p.set_xticks([]); ax_p.set_yticks([])
    for sp in ax_p.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(0.8)
    ax_p.text(0.5, 0.5, 'D', transform=ax_p.transAxes, fontsize=6,
              ha='center', va='center', fontweight='bold',
              color=_text_color_for_bg(data['pred_sRGB']), alpha=0.7)

    # Legend
    handles = [
        Line2D([0], [0], color='black', lw=1.5, ls='-', label='Target'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', label='Design'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=6.5, framealpha=0.85)




def plot_swatch_grid(fig, grid_spec, examples):
    """
    Draw a grid of Target|Design color swatch pairs.
    Each cell shows clean swatches with dE label below.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    inner = GridSpecFromSubplotSpec(
        SWATCH_ROWS, SWATCH_COLS, subplot_spec=grid_spec,
        hspace=0.35, wspace=0.08,
    )

    for idx, data in enumerate(examples):
        row = idx // SWATCH_COLS
        col = idx % SWATCH_COLS

        cell = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=inner[row, col],
            wspace=0.1,
        )

        gt_c = [c / 255 for c in data['gt_sRGB']]
        pr_c = [c / 255 for c in data['pred_sRGB']]

        # Target swatch
        ax_t = fig.add_subplot(cell[0, 0])
        ax_t.set_facecolor(gt_c)
        ax_t.set_xticks([]); ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(0.6)

        # Design swatch
        ax_p = fig.add_subplot(cell[0, 1])
        ax_p.set_facecolor(pr_c)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        for sp in ax_p.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(0.6)

        # dE label below the pair (centered between the two swatches)
        de = data['delta_e']
        ax_t.text(1.05, -0.05, f'\u0394E={de:.1f}',
                  transform=ax_t.transAxes, fontsize=5.5,
                  ha='center', va='top', color='#333')

        # Column headers on first row only
        if row == 0:
            ax_t.set_title('Target', fontsize=5.5, pad=3, color='#555')
            ax_p.set_title('Design', fontsize=5.5, pad=3, color='#555')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grant-figure: reflectance spectra + color swatch grid')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    args = parser.parse_args()

    temperature = min(args.temperature, 1.0)
    if temperature != args.temperature:
        print(f"[INFO] Temperature clamped to {temperature}")

    if not optical_is_available():
        print("[ERROR] Optical simulation not available")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    checkpoint_dir = Path(args.checkpoint)
    model, config = load_model(checkpoint_dir, device)
    print(f"[INFO] Model: {config.tag()}")
    print(f"[INFO] Temperature: {temperature}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / 'create_dataset' / 'data_prompts'

    dataset = ThinFilmDataset(data_dir, seed=args.seed, split=args.split, verbose=True)
    all_examples = list(dataset)
    print(f"[INFO] {len(all_examples)} examples from {args.split} split")

    rng = random.Random(args.seed)
    candidates = list(all_examples)
    rng.shuffle(candidates)

    sim = OpticalSimulator(incidence_angle=0)

    # ================================================================
    # Collect examples
    # ================================================================

    # Collect in two passes:
    #   1. Spectra examples: require spec_diff >= MIN_SPEC_DIFF (interesting spectra)
    #   2. Swatch examples: no spec_diff filter (show full range including exact matches)

    used_indices = set()

    def collect_examples(candidate_list, n_needed, require_spec_diff,
                         require_chroma, label):
        collected = []
        n_skip_gray, n_skip_spec = 0, 0
        for ci, ex in enumerate(candidate_list):
            if len(collected) >= n_needed:
                break
            if ci in used_indices:
                continue

            gt_sRGB = denormalize_rgb(ex.rgb)
            if require_chroma and lab_chroma(gt_sRGB) < MIN_CHROMA:
                n_skip_gray += 1
                continue

            pred_mats, pred_thick, stop = generate_structure_tempered(
                model, ex.rgb, device, temperature=temperature)
            if not pred_mats:
                continue

            R_gt = sim.compute_reflectance(ex.target_materials, ex.target_thicknesses)
            R_pred = sim.compute_reflectance(pred_mats, pred_thick)
            spec_diff = float(np.mean(np.abs(R_gt - R_pred)))

            if require_spec_diff and spec_diff < MIN_SPEC_DIFF:
                n_skip_spec += 1
                continue

            pred_sRGB = sim.compute_color(pred_mats, pred_thick)
            de = compute_color_difference(gt_sRGB, pred_sRGB)

            collected.append({
                'gt_materials': ex.target_materials,
                'gt_thicknesses': ex.target_thicknesses,
                'gt_sRGB': gt_sRGB,
                'pred_materials': pred_mats,
                'pred_thicknesses': pred_thick,
                'pred_sRGB': pred_sRGB,
                'delta_e': de,
                'stop_reason': stop,
            })
            used_indices.add(ci)

            i = len(collected)
            print(f"  [{label} {i}/{n_needed}] dE={de:.2f}  spec_diff={spec_diff:.3f}  "
                  f"GT={gt_sRGB}  Pred={pred_sRGB}  "
                  f"layers={len(ex.target_materials)}->{len(pred_mats)}")

        print(f"[INFO] {label}: collected {len(collected)} "
              f"(skipped {n_skip_gray} gray, {n_skip_spec} near-identical)")
        return collected

    # Pass 1: spectra — require distinct spectra + colorful examples
    spectra_examples = collect_examples(
        candidates, N_SPECTRA,
        require_spec_diff=True, require_chroma=True, label="Spectra")

    # Pass 2: swatches — no spec_diff filter, but exclude whites/grays
    swatch_examples = collect_examples(
        candidates, N_SWATCHES,
        require_spec_diff=False, require_chroma=True, label="Swatch")

    examples_data = spectra_examples + swatch_examples

    if len(spectra_examples) < N_SPECTRA:
        print(f"[ERROR] Only collected {len(spectra_examples)} spectra examples")
        sys.exit(1)

    # ================================================================
    # Build figure -- portrait, grant-proposal sized
    # ================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.5, 8.0))

    gs_main = GridSpec(2, 1, figure=fig,
                       height_ratios=[0.33, 0.67],
                       hspace=0.38,
                       top=0.93, bottom=0.03,
                       left=0.08, right=0.97)

    # Top: spectra side by side
    gs_spectra = gs_main[0].subgridspec(1, N_SPECTRA, wspace=0.38)
    for i, data in enumerate(spectra_examples):
        ax = fig.add_subplot(gs_spectra[0, i])
        plot_spectrum(ax, sim, data, idx_label=i + 1)

    # Bottom: swatch grid
    plot_swatch_grid(fig, gs_main[1], swatch_examples)

    # FIX: main title at y=0.97 -- well above subplot titles (which are at ~0.93)
    fig.suptitle('Inverse Optical Design: Target vs Design Colors',
                 fontsize=11, fontweight='bold', y=0.97)

    # FIX: swatch header positioned in the gap between spectra and swatches
    fig.text(0.5, 0.575,
             'Color Swatch Comparison  (Target | Design)',
             ha='center', fontsize=8, color='#444', style='italic')

    # Summary stats line below header
    all_de = [d['delta_e'] for d in examples_data]
    fig.text(0.5, 0.555,
             f'mean \u0394E\u2080\u2080 = {np.mean(all_de):.2f}  |  '
             f'median \u0394E\u2080\u2080 = {np.median(all_de):.2f}',
             ha='center', fontsize=7, color='#777')

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        temp_str = f"{temperature:.1f}".replace('.', 'p')
        output_path = (repo_root / 'pretrain_rgb_to_structure' / 'outputs' /
                       f'reflectance_comparison_{config.tag()}_T{temp_str}.png')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Figure saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()