#!/usr/bin/env python3
"""
Thickness Distribution Plot — winning material per position, faceted
====================================================================

For a batch of val examples, run one forward pass of a checkpoint,
then for every layer position of every example extract the softmax
distribution over the 100 thickness bins CONDITIONAL on the winning
(argmax) material at that position. Plot as a faceted grid — one
subplot per example, one line per layer position, x = thickness (nm),
y = P(thickness | winning slot, position).

Answers: are the model's thickness picks sharply peaked, broadly
supported, or bimodal? Do bimodal thickness distributions appear at
particular layer indices or example types?

GT thickness for each layer is overlaid as a vertical dashed line so
you can see how far the peak is from the correct pick.

Usage
-----
    python analyses/de_finetune/thickness_distribution_plot.py \\
        --checkpoint data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best \\
        --data-dir data/finetune --n-examples 32 \\
        --output-path analyses/de_finetune/results/thickness_dist_best.png

The script writes ONE PNG. No SLURM wrapper — this runs in <1 min on a
laptop-class GPU or CPU (32 examples, one forward pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import FlexThinFilmDataset, find_repo_root
from src.de_finetune import collate_fn
from src.materials_vocab import (
    M_MAX,
    MAX_LAYERS,
    NUM_THICKNESSES,
    THICKNESSES,
    denormalize_lab,
)
from src.model import ModelConfig, build_model


# ============================================================================
# Model loading — supports both meta.json (finetune) and config.json (inference)
# ============================================================================


def _load_config(checkpoint_dir: Path) -> ModelConfig:
    cfg_path = checkpoint_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return ModelConfig(**json.load(f))
    meta_path = checkpoint_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return ModelConfig(**meta["config"])
    raise FileNotFoundError(
        f"No config.json or meta.json at {checkpoint_dir}"
    )


def load_model(checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    config = _load_config(checkpoint_dir)
    model = build_model(config).to(device).eval()
    state = torch.load(
        checkpoint_dir / "model.pt", map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    return model


# ============================================================================
# Extract per-position thickness distributions on the winning slot
# ============================================================================


def compute_thickness_distributions(
    model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
) -> List[Dict]:
    """One forward pass. Returns a list (one per example) of dicts:

        {
          "target_lab_denorm": [L, a, b],
          "gt_slots":          [s0, s1, ...],
          "gt_thicknesses":    [nm0, nm1, ...],
          "positions": [
              {
                "layer_idx":      int,
                "winning_slot":   int,
                "thick_probs":    np.ndarray[NUM_THICKNESSES],
                "winning_thick_bin": int,
                "gt_thickness_nm":   int | None,
              },
              ...
          ],
        }
    """
    from src.materials_vocab import build_structure_matrix

    B = batch["lab"].size(0)
    structure_matrices = []
    for b in range(B):
        gt_slots = list(batch["target_slots"][b])[:MAX_LAYERS]
        gt_thicknesses = list(batch["target_thicknesses"][b])[:MAX_LAYERS]
        structure_matrices.append(
            build_structure_matrix(gt_slots, gt_thicknesses),
        )
    structure_matrix = torch.stack(structure_matrices, dim=0).to(device)

    with torch.no_grad():
        logits = model(
            lab=batch["lab"].to(device),
            pool_features=batch["pool_features"].to(device),
            pool_mask=batch["pool_mask"].to(device),
            pool_size=batch["pool_size"].to(device),
            structure_matrix=structure_matrix,
            apply_output_mask=True,
        )  # [B, MAX_LAYERS+1, VOCAB_SIZE]

    # Clip -inf/+inf to finite so softmax is well defined.
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)

    out: List[Dict] = []
    for b in range(B):
        gt_slots = list(batch["target_slots"][b])[:MAX_LAYERS]
        gt_thicknesses = list(batch["target_thicknesses"][b])[:MAX_LAYERS]
        target_lab = denormalize_lab(batch["lab"][b])

        positions = []
        for k in range(len(gt_slots)):
            # [M_MAX, NUM_THICKNESSES] joint logits at this position
            layer_logits = logits[b, k, : M_MAX * NUM_THICKNESSES].view(
                M_MAX, NUM_THICKNESSES,
            )
            # Per-slot marginal = max over thickness (same pool as ste_pick)
            slot_scores = layer_logits.max(dim=-1).values  # [M_MAX]
            # Winning slot = argmax over active slots
            pool_mask_b = batch["pool_mask"][b].to(device).bool()
            slot_scores_masked = slot_scores.masked_fill(
                ~pool_mask_b, float("-inf"),
            )
            winning_slot = int(slot_scores_masked.argmax().item())

            # Thickness distribution CONDITIONAL on winning slot: softmax
            # over the thickness logits for that slot.
            thick_logits = layer_logits[winning_slot]  # [NUM_THICKNESSES]
            thick_probs = torch.softmax(thick_logits, dim=-1).cpu().numpy()

            positions.append({
                "layer_idx": k,
                "winning_slot": winning_slot,
                "thick_probs": thick_probs,
                "winning_thick_bin": int(thick_probs.argmax()),
                "gt_thickness_nm": int(gt_thicknesses[k]),
                "gt_slot": int(gt_slots[k]),
            })
        out.append({
            "target_lab_denorm": list(target_lab),
            "gt_slots": gt_slots,
            "gt_thicknesses": gt_thicknesses,
            "positions": positions,
        })
    return out


# ============================================================================
# Plotting
# ============================================================================


def make_faceted_plot(
    per_example: List[Dict],
    output_path: Path,
    grid_cols: int = 4,
) -> None:
    n = len(per_example)
    grid_rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * 3.0, grid_rows * 2.4),
        sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    thickness_axis = np.asarray(THICKNESSES)  # 2..200 nm

    layer_colors = plt.cm.viridis(np.linspace(0.15, 0.85, MAX_LAYERS))

    for i, ex in enumerate(per_example):
        row, col = divmod(i, grid_cols)
        ax = axes[row, col]

        for pos in ex["positions"]:
            k = pos["layer_idx"]
            color = layer_colors[k]
            ax.plot(
                thickness_axis, pos["thick_probs"],
                color=color, alpha=0.9, linewidth=1.4,
                label=f"L{k}: slot={pos['winning_slot']}",
            )
            # GT thickness for this layer as vertical dashed line (same
            # colour). Mark whether the slot picked matches GT slot.
            slot_ok = "✓" if pos["winning_slot"] == pos["gt_slot"] else "✗"
            ax.axvline(
                pos["gt_thickness_nm"], color=color, linestyle="--",
                linewidth=1.0, alpha=0.7,
            )
            # Star at GT thickness position on top edge (small).
            top_y = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
            ax.plot(
                pos["gt_thickness_nm"], 0, marker="*",
                color=color, markersize=5,
            )

        lab = ex["target_lab_denorm"]
        ax.set_title(
            f"ex{i}: L={lab[0]:.0f} a={lab[1]:.0f} b={lab[2]:.0f}",
            fontsize=8,
        )
        ax.set_xlim(0, 202)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
        if col == 0:
            ax.set_ylabel("P(thick | slot)", fontsize=8)
        if row == grid_rows - 1:
            ax.set_xlabel("thickness (nm)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc="best")

    # Blank unused axes if n < rows*cols.
    for j in range(n, grid_rows * grid_cols):
        row, col = divmod(j, grid_cols)
        axes[row, col].axis("off")

    fig.suptitle(
        "Thickness distributions P(thick | winning slot) per layer position\n"
        "dashed vertical = GT thickness, star at bottom marks GT position,\n"
        "each color = one decoding layer",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[thickness-plot] wrote {output_path}")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Finetune checkpoint dir (best/ or step_N/)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Finetune data dir (default: <repo>/data/finetune)")
    p.add_argument("--split", type=str, default="validation",
                   help="dataset split (default: validation)")
    p.add_argument("--n-examples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=Path, required=True,
                   help="Where to write the PNG")
    p.add_argument("--grid-cols", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).parent)
    if args.data_dir is None:
        args.data_dir = repo_root / "data" / "finetune"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[thickness-plot] device={device}", flush=True)
    print(f"[thickness-plot] loading model from {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device)

    print(f"[thickness-plot] loading data from {args.data_dir}", flush=True)
    ds = FlexThinFilmDataset(
        data_prompts_dir=args.data_dir,
        seed=args.seed, split=args.split,
        limit_examples=args.n_examples, streaming=True,
    )
    examples = [ex for ex in ds][: args.n_examples]
    batch = collate_fn(examples)
    print(f"[thickness-plot] batch size = {len(examples)}", flush=True)

    per_example = compute_thickness_distributions(model, batch, device)
    make_faceted_plot(per_example, args.output_path, grid_cols=args.grid_cols)


if __name__ == "__main__":
    main()
