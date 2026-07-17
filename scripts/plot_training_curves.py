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
from src.model import ModelConfig, build_model, compute_loss


def load_model_from_checkpoint(checkpoint_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, ModelConfig]:
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    model = build_model(config)
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


def evaluate_teacher_forcing(model, dataset, device, batch_size=64,
                              num_workers=4, prefetch_factor=1) -> Dict[str, float]:
    model.eval()
    loader_kw = {}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, **loader_kw,
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
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Training data dir. Train + validation slices "
                             "come from here.")
    parser.add_argument("--test-a-dir", type=str, default=None,
                        help="Tier-A test dir (data/test/tier_a). If given, "
                             "adds a 'test_a' curve.")
    parser.add_argument("--test-b-dir", type=str, default=None,
                        help="Tier-B test dir (data/test/tier_b). If given, "
                             "adds a 'test_b' curve.")
    parser.add_argument("--test-examples", type=int, default=5000,
                        help="Max rows per test tier per checkpoint (0 = "
                             "read all shards).")
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--end-step", type=int, default=174000)
    parser.add_argument("--step-interval", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=5000)
    parser.add_argument("--train-examples", type=int, default=30000)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="DataLoader prefetch_factor (default: 1; matches "
                             "training.py — pipeline is producer-bound)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream dataset shard-by-shard (recommended at "
                             "production scale; the legacy mode OOMs).")

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

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "train"
    print(f"[INFO] Loading validation data from {data_dir}")
    val_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="validation",
        limit_examples=args.eval_examples, verbose=True,
        streaming=args.streaming,
    )
    print(f"[INFO] Validation set: {len(val_dataset)} examples")

    print(f"[INFO] Loading training data from {data_dir}")
    train_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="train",
        limit_examples=args.train_examples, verbose=True,
        streaming=args.streaming,
    )
    print(f"[INFO] Training set: {len(train_dataset)} examples")

    # Optional held-out test tiers. Loaded once, re-used across every
    # checkpoint. Tier_a = seen materials + novel structures; tier_b =
    # unseen materials + novel structures (the harder generalisation test).
    test_a_dataset = None
    test_b_dataset = None
    # limit_examples=0 in FlexThinFilmDataset is falsy — read all shards.
    # Use None for "read all", so translate 0 → None here.
    test_limit = args.test_examples or None
    if args.test_a_dir:
        print(f"[INFO] Loading tier_A test data from {args.test_a_dir}")
        test_a_dataset = FlexThinFilmDataset(
            Path(args.test_a_dir), seed=args.seed, split="train",
            limit_examples=test_limit, verbose=True,
            streaming=args.streaming,
        )
        print(f"[INFO] Test-A set: {len(test_a_dataset)} examples")
    if args.test_b_dir:
        print(f"[INFO] Loading tier_B test data from {args.test_b_dir}")
        test_b_dataset = FlexThinFilmDataset(
            Path(args.test_b_dir), seed=args.seed, split="train",
            limit_examples=test_limit, verbose=True,
            streaming=args.streaming,
        )
        print(f"[INFO] Test-B set: {len(test_b_dataset)} examples")

    steps_to_eval = list(range(args.start_step, args.end_step + 1, args.step_interval))
    print(f"[INFO] Will evaluate {len(steps_to_eval)} checkpoints from step "
          f"{args.start_step} to {args.end_step}")

    results: Dict[str, List] = {
        "steps": [], "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "test_a_loss": [], "test_a_accuracy": [],
        "test_b_loss": [], "test_b_accuracy": [],
    }

    for i, step in enumerate(steps_to_eval):
        checkpoint_path = checkpoint_base / f"step_{step}"
        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        print(f"\n[{i + 1}/{len(steps_to_eval)}] Evaluating step {step}...")
        # Default per-tier metrics — NaN so the JSON stays rectangular
        # even when a tier is disabled or crashes.
        train_loss = train_acc = float("nan")
        val_loss = val_acc = float("nan")
        test_a_loss = test_a_acc = float("nan")
        test_b_loss = test_b_acc = float("nan")
        try:
            model, config = load_model_from_checkpoint(checkpoint_path, device)
            train_results = evaluate_teacher_forcing(
                model, train_dataset, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            val_results = evaluate_teacher_forcing(
                model, val_dataset, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            train_loss = train_results["loss"]
            train_acc = train_results["accuracy"]
            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]
            if test_a_dataset is not None:
                r = evaluate_teacher_forcing(
                    model, test_a_dataset, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
                test_a_loss, test_a_acc = r["loss"], r["accuracy"]
            if test_b_dataset is not None:
                r = evaluate_teacher_forcing(
                    model, test_b_dataset, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
                test_b_loss, test_b_acc = r["loss"], r["accuracy"]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate: {e}")
        parts = [
            f"Train loss={train_loss:.4f} acc={train_acc:.3f}",
            f"Val loss={val_loss:.4f} acc={val_acc:.3f}",
        ]
        if test_a_dataset is not None:
            parts.append(f"TestA loss={test_a_loss:.4f} acc={test_a_acc:.3f}")
        if test_b_dataset is not None:
            parts.append(f"TestB loss={test_b_loss:.4f} acc={test_b_acc:.3f}")
        print("  " + "  ".join(parts))
        results["steps"].append(step)
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_acc)
        results["test_a_loss"].append(test_a_loss)
        results["test_a_accuracy"].append(test_a_acc)
        results["test_b_loss"].append(test_b_loss)
        results["test_b_accuracy"].append(test_b_acc)

    if args.output:
        output_path = Path(args.output)
    else:
        tag = checkpoint_base.name
        output_path = repo_root / "outputs" / f"training_curves_{tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    have_test_a = any(not np.isnan(x) for x in results["test_a_loss"])
    have_test_b = any(not np.isnan(x) for x in results["test_b_loss"])
    header_bits = [f"{'Step':>10}", f"{'Train Loss':>11}", f"{'Train Acc':>10}",
                   f"{'Val Loss':>10}", f"{'Val Acc':>8}"]
    if have_test_a:
        header_bits += [f"{'TestA Loss':>11}", f"{'TestA Acc':>10}"]
    if have_test_b:
        header_bits += [f"{'TestB Loss':>11}", f"{'TestB Acc':>10}"]
    header = " ".join(header_bits)
    print("\n" + "=" * len(header))
    print("TRAINING CURVES SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(len(results["steps"])):
        row = [
            f"{results['steps'][i]:>10}",
            f"{results['train_loss'][i]:>11.4f}",
            f"{results['train_accuracy'][i]:>10.3f}",
            f"{results['val_loss'][i]:>10.4f}",
            f"{results['val_accuracy'][i]:>8.3f}",
        ]
        if have_test_a:
            row += [f"{results['test_a_loss'][i]:>11.4f}",
                    f"{results['test_a_accuracy'][i]:>10.3f}"]
        if have_test_b:
            row += [f"{results['test_b_loss'][i]:>11.4f}",
                    f"{results['test_b_accuracy'][i]:>10.3f}"]
        print(" ".join(row))
    print("=" * len(header))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot generation")
            return
        steps = np.array(results["steps"]) / 1000  # thousands
        # Colour + label per series. Test tiers only rendered when the
        # data was actually collected — otherwise the arrays are all-NaN
        # and matplotlib would draw an empty legend entry.
        series = [
            ("train_loss",  "train_accuracy",  "#1f77b4", "train"),
            ("val_loss",    "val_accuracy",    "#d62728", "val (held-out from train)"),
            ("test_a_loss", "test_a_accuracy", "#2ca02c", "test-A (seen materials · novel structures)"),
            ("test_b_loss", "test_b_accuracy", "#9467bd", "test-B (unseen materials · novel structures)"),
        ]
        window = args.smoothing_window
        kernel = np.ones(window) / window if window > 0 else None

        def _smooth(arr: np.ndarray):
            """Convolve, return (steps_smooth, values_smooth) or None if too short."""
            if kernel is None or len(steps) <= window:
                return None
            values_smooth = np.convolve(arr, kernel, mode="valid")
            # matplotlib's default steps_smooth alignment for `mode="valid"`.
            offset = window // 2
            end = -(window // 2) if window % 2 == 0 else -(window // 2) or None
            steps_smooth = steps[offset:end]
            min_len = min(len(steps_smooth), len(values_smooth))
            return steps_smooth[:min_len], values_smooth[:min_len]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("INDIGO Training Curves", fontsize=14, fontweight="bold")

        # ---- Loss panel ------------------------------------------------
        for loss_key, _acc_key, colour, label in series:
            arr = np.array(results[loss_key])
            if not np.any(~np.isnan(arr)):
                continue
            ax1.plot(steps, arr, color=colour, alpha=0.28, linewidth=1,
                     label=f"{label} (raw)")
            sm = _smooth(arr)
            if sm is not None:
                ss, sv = sm
                ax1.plot(ss, sv, color=colour, linewidth=2.4,
                         label=f"{label} (smoothed)")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Annotate the best val loss (traditional stopping-point marker).
        val_loss = np.array(results["val_loss"])
        if np.any(~np.isnan(val_loss)):
            best = int(np.nanargmin(val_loss))
            ax1.annotate(
                f"Best val: {val_loss[best]:.4f}",
                xy=(steps[best], val_loss[best]),
                xytext=(steps[-1] * 0.7, np.nanmax(val_loss) * 0.9),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7),
            )

        # ---- Accuracy panel --------------------------------------------
        for _loss_key, acc_key, colour, label in series:
            arr = np.array(results[acc_key]) * 100
            if not np.any(~np.isnan(arr)):
                continue
            ax2.plot(steps, arr, color=colour, alpha=0.28, linewidth=1,
                     label=f"{label} (raw)")
            sm = _smooth(arr)
            if sm is not None:
                ss, sv = sm
                ax2.plot(ss, sv, color=colour, linewidth=2.4,
                         label=f"{label} (smoothed)")
        ax2.set_xlabel("Training Steps (thousands)", fontsize=12)
        ax2.set_ylabel("Token-level Accuracy (%)", fontsize=12)
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        val_acc = np.array(results["val_accuracy"])
        if np.any(~np.isnan(val_acc)):
            best = int(np.nanargmax(val_acc))
            ax2.annotate(
                f"Best val: {val_acc[best] * 100:.1f}%",
                xy=(steps[best], val_acc[best] * 100),
                xytext=(steps[-1] * 0.7, np.nanmin(val_acc) * 100 + 2),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7),
            )

        plt.tight_layout()
        plot_path = output_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
