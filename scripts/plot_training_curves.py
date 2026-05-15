#!/usr/bin/env python3
"""
Plot Training Loss vs. Teacher Forcing Validation Loss Across Checkpoints.

Loads checkpoints at regular step intervals and computes:
1. Training loss on a small subset of the train split
2. Teacher-forcing validation loss on the validation split

Usage:
    python scripts/plot_training_curves.py \
        --checkpoint-dir data/checkpoints/<config-tag> \
        --start-step 1000 --end-step 100000 --step-interval 5000 \
        --eval-examples 5000 --train-examples 30000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from scripts.training import collate_fn
from src.dataset import FlexThinFilmDataset, find_repo_root
from src.model import FlexMaterialMLP, ModelConfig, compute_loss


def load_model_from_checkpoint(checkpoint_dir: Path, device: torch.device) -> Tuple[FlexMaterialMLP, ModelConfig]:
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    model = FlexMaterialMLP(config)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True), strict=False
    )
    model.to(device)
    model.eval()
    return model, config


def get_training_loss_from_meta(checkpoint_dir: Path) -> Optional[float]:
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("loss")


def evaluate_teacher_forcing(model, dataset, device, batch_size=64, num_workers=4) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True,
    )
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            losses = compute_loss(model, batch_on_device)
            count = batch_on_device["lab"].size(0)
            total_loss += losses["loss"].item() * count
            total_correct += int(losses["accuracy"].item() * count)
            total_samples += count
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "n_samples": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves across checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--end-step", type=int, default=174000)
    parser.add_argument("--step-interval", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=5000)
    parser.add_argument("--train-examples", type=int, default=30000)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    checkpoint_base = Path(args.checkpoint_dir)
    if not checkpoint_base.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_base}")
        sys.exit(1)

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "create_dataset" / "data_prompts"
    print(f"[INFO] Loading validation data from {data_dir}")
    val_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="validation",
        limit_examples=args.eval_examples, verbose=True,
    )
    print(f"[INFO] Validation set: {len(val_dataset)} examples")

    print(f"[INFO] Loading training data from {data_dir}")
    train_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="train",
        limit_examples=args.train_examples, verbose=True,
    )
    print(f"[INFO] Training set: {len(train_dataset)} examples")

    steps_to_eval = list(range(args.start_step, args.end_step + 1, args.step_interval))
    print(f"[INFO] Will evaluate {len(steps_to_eval)} checkpoints from step "
          f"{args.start_step} to {args.end_step}")

    results: Dict[str, List] = {
        "steps": [], "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
    }

    for i, step in enumerate(steps_to_eval):
        checkpoint_path = checkpoint_base / f"step_{step}"
        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        print(f"\n[{i + 1}/{len(steps_to_eval)}] Evaluating step {step}...")
        try:
            model, config = load_model_from_checkpoint(checkpoint_path, device)
            train_results = evaluate_teacher_forcing(
                model, train_dataset, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            val_results = evaluate_teacher_forcing(
                model, val_dataset, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            train_loss = train_results["loss"]
            train_acc = train_results["accuracy"]
            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate: {e}")
            train_loss = float("nan"); train_acc = float("nan")
            val_loss = float("nan"); val_acc = float("nan")
        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.3f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.3f}")
        results["steps"].append(step)
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_acc)

    if args.output:
        output_path = Path(args.output)
    else:
        tag = checkpoint_base.name
        output_path = repo_root / "outputs" / f"training_curves_{tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("TRAINING CURVES SUMMARY")
    print("=" * 80)
    print(f"{'Step':>10} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 60)
    for i in range(len(results["steps"])):
        print(f"{results['steps'][i]:>10} {results['train_loss'][i]:>12.4f} "
              f"{results['train_accuracy'][i]:>10.3f} {results['val_loss'][i]:>12.4f} "
              f"{results['val_accuracy'][i]:>10.3f}")
    print("=" * 80)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot generation")
            return
        steps = np.array(results["steps"]) / 1000  # thousands
        train_loss = np.array(results["train_loss"])
        val_loss = np.array(results["val_loss"])
        train_acc = np.array(results["train_accuracy"])
        val_acc = np.array(results["val_accuracy"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("INDIGO Training Curves", fontsize=14, fontweight="bold")

        ax1.plot(steps, train_loss, "b-", alpha=0.3, linewidth=1, label="Train Loss (raw)")
        ax1.plot(steps, val_loss, "r-", alpha=0.3, linewidth=1, label="Val Loss (raw)")
        window = args.smoothing_window
        steps_smooth = steps
        train_smooth = train_loss
        val_smooth = val_loss
        if len(steps) > window:
            kernel = np.ones(window) / window
            train_smooth = np.convolve(train_loss, kernel, mode="valid")
            val_smooth = np.convolve(val_loss, kernel, mode="valid")
            steps_smooth = steps[window // 2:-(window // 2)] if window % 2 == 0 else steps[window // 2:-(window // 2) or None]
            min_len = min(len(steps_smooth), len(train_smooth), len(val_smooth))
            steps_smooth = steps_smooth[:min_len]
            train_smooth = train_smooth[:min_len]
            val_smooth = val_smooth[:min_len]
            ax1.plot(steps_smooth, train_smooth, "b-", linewidth=2.5, label="Train Loss (smoothed)")
            ax1.plot(steps_smooth, val_smooth, "r-", linewidth=2.5, label="Val Loss (smoothed)")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.3)
        if np.any(~np.isnan(val_loss)):
            best_val_idx = int(np.nanargmin(val_loss))
            ax1.annotate(f"Best Val: {val_loss[best_val_idx]:.4f}",
                         xy=(steps[best_val_idx], val_loss[best_val_idx]),
                         xytext=(steps[-1] * 0.7, np.nanmax(val_loss) * 0.9),
                         fontsize=10,
                         arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

        ax2.plot(steps, train_acc * 100, "b-", alpha=0.3, linewidth=1, label="Train Acc (raw)")
        ax2.plot(steps, val_acc * 100, "g-", alpha=0.3, linewidth=1, label="Val Acc (raw)")
        if len(steps) > window:
            min_len = len(steps_smooth)
            train_acc_smooth = np.convolve(train_acc * 100, kernel, mode="valid")[:min_len]
            val_acc_smooth = np.convolve(val_acc * 100, kernel, mode="valid")[:min_len]
            ax2.plot(steps_smooth, train_acc_smooth, "b-", linewidth=2.5, label="Train Acc (smoothed)")
            ax2.plot(steps_smooth, val_acc_smooth, "g-", linewidth=2.5, label="Val Acc (smoothed)")
        ax2.set_xlabel("Training Steps (thousands)", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.legend(loc="lower right"); ax2.grid(True, alpha=0.3)
        if np.any(~np.isnan(val_acc)):
            best_acc_idx = int(np.nanargmax(val_acc))
            ax2.annotate(f"Best Val: {val_acc[best_acc_idx] * 100:.1f}%",
                         xy=(steps[best_acc_idx], val_acc[best_acc_idx] * 100),
                         xytext=(steps[-1] * 0.7, np.nanmin(val_acc) * 100 + 2),
                         fontsize=10,
                         arrowprops=dict(arrowstyle="->", color="green", alpha=0.7))

        plt.tight_layout()
        plot_path = output_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
