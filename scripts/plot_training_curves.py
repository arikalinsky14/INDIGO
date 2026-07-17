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
    """Legacy path — re-reads the dataset every call. Kept for the rare
    case where a caller wants a stream. Every checkpoint-loop caller
    should use `precollate_dataset` + `evaluate_precollated` instead."""
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


def precollate_dataset(dataset, batch_size: int, num_workers: int,
                       prefetch_factor: int, label: str) -> List[Dict[str, torch.Tensor]]:
    """Read the dataset ONCE and collect every collated batch as CPU
    tensors. Massive speed-up over the streaming path when we're going
    to iterate the same dataset dozens of times (35 checkpoints × 4
    datasets = 140 iterations in the default sweep).

    Per-checkpoint cost drops from "re-read shards + respawn workers"
    to "iterate a Python list of tensors". Memory: ~50 KB per row, so
    5 k rows fits in ~250 MB CPU RAM — comfortably under any reasonable
    SLURM allocation.
    """
    import time
    loader_kw = {}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, **loader_kw,
    )
    batches: List[Dict[str, torch.Tensor]] = []
    t0 = time.time()
    for batch in loader:
        batches.append(batch)
    elapsed = time.time() - t0
    n_rows = sum(b["lab"].size(0) for b in batches)
    print(f"[INFO] Pre-collated {label}: {n_rows} rows in {len(batches)} "
          f"batches ({elapsed:.1f}s)")
    return batches


def evaluate_precollated(model, batches: List[Dict[str, torch.Tensor]],
                         device) -> Dict[str, float]:
    """Iterate pre-collated batches. No disk I/O, no worker spawn — the
    per-checkpoint cost is a pure GPU forward-pass loop.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in batches:
            batch_on_device = {k: v.to(device, non_blocking=True)
                               for k, v in batch.items()}
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
    parser.add_argument("--train-loss-source",
                        choices=("meta", "eval"), default="meta",
                        help="'meta' (default, fast): read the running "
                             "training loss from each checkpoint's "
                             "meta.json — no re-eval, saves ~75%% of the "
                             "sweep. 'eval': re-evaluate a train subset "
                             "the same way val is scored — slower but on "
                             "the same scale as val (no batch-noise "
                             "artefacts).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="If the output JSON already contains "
                             "completed steps, skip them and continue. "
                             "Combined with the incremental save (always "
                             "on), a timed-out sweep can be resumed by "
                             "just re-sbatching the same job.")

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

    # Resolve output path first — needed for --resume before we pre-collate.
    if args.output:
        output_path = Path(args.output)
    else:
        tag = checkpoint_base.name
        output_path = repo_root / "outputs" / f"training_curves_{tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only pre-collate a train set if the user actually wants to
    # re-evaluate train loss. Default path reads train loss from each
    # checkpoint's meta.json (running training loss, no re-eval needed).
    train_batches = None
    if args.train_loss_source == "eval":
        print(f"[INFO] Loading training data from {data_dir} (--train-loss-source=eval)")
        train_dataset = FlexThinFilmDataset(
            data_dir, seed=args.seed, split="train",
            limit_examples=args.train_examples, verbose=True,
            streaming=args.streaming,
        )
        print(f"[INFO] Training set: {len(train_dataset)} examples")
        train_batches = precollate_dataset(
            train_dataset, args.batch_size, args.num_workers,
            args.prefetch_factor, "train",
        )
        del train_dataset
    else:
        print(f"[INFO] Train loss will be read from meta.json per checkpoint "
              f"(--train-loss-source=meta; skips train re-eval).")

    # Pre-collate val + optional test tiers ONCE. Every subsequent
    # per-checkpoint eval iterates a list of CPU tensors instead of
    # spawning DataLoader workers + re-reading parquet shards.
    val_batches = precollate_dataset(
        val_dataset, args.batch_size, args.num_workers,
        args.prefetch_factor, "val",
    )
    del val_dataset

    test_a_batches = None
    test_b_batches = None
    # limit_examples=0 → None (read all shards).
    test_limit = args.test_examples or None
    if args.test_a_dir:
        print(f"[INFO] Loading tier_A test data from {args.test_a_dir}")
        test_a_dataset = FlexThinFilmDataset(
            Path(args.test_a_dir), seed=args.seed, split="train",
            limit_examples=test_limit, verbose=True,
            streaming=args.streaming,
        )
        print(f"[INFO] Test-A set: {len(test_a_dataset)} examples")
        test_a_batches = precollate_dataset(
            test_a_dataset, args.batch_size, args.num_workers,
            args.prefetch_factor, "test_a",
        )
        del test_a_dataset
    if args.test_b_dir:
        print(f"[INFO] Loading tier_B test data from {args.test_b_dir}")
        test_b_dataset = FlexThinFilmDataset(
            Path(args.test_b_dir), seed=args.seed, split="train",
            limit_examples=test_limit, verbose=True,
            streaming=args.streaming,
        )
        print(f"[INFO] Test-B set: {len(test_b_dataset)} examples")
        test_b_batches = precollate_dataset(
            test_b_dataset, args.batch_size, args.num_workers,
            args.prefetch_factor, "test_b",
        )
        del test_b_dataset

    steps_to_eval = list(range(args.start_step, args.end_step + 1, args.step_interval))
    print(f"[INFO] Will evaluate {len(steps_to_eval)} checkpoints from step "
          f"{args.start_step} to {args.end_step}")

    # Resume support: if the output JSON already contains completed
    # steps, skip those. Combined with the incremental save below, a
    # timed-out sweep can be resumed by just re-sbatching the same job.
    results: Dict[str, List] = {
        "steps": [], "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "test_a_loss": [], "test_a_accuracy": [],
        "test_b_loss": [], "test_b_accuracy": [],
    }
    if args.resume and output_path.exists():
        try:
            with open(output_path) as f:
                prior = json.load(f)
            for k in results:
                if k in prior and isinstance(prior[k], list):
                    results[k] = list(prior[k])
            done_steps = set(results["steps"])
            skipped = [s for s in steps_to_eval if s in done_steps]
            steps_to_eval = [s for s in steps_to_eval if s not in done_steps]
            print(f"[INFO] Resume: {len(skipped)} steps already in "
                  f"{output_path.name}, {len(steps_to_eval)} to go.")
        except Exception as exc:
            print(f"[WARN] Resume disabled — could not parse existing "
                  f"{output_path}: {exc}")

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
            # Val + optional test tiers via pre-collated batches — no
            # disk I/O or worker spawn per checkpoint.
            val_r = evaluate_precollated(model, val_batches, device)
            val_loss, val_acc = val_r["loss"], val_r["accuracy"]
            if test_a_batches is not None:
                r = evaluate_precollated(model, test_a_batches, device)
                test_a_loss, test_a_acc = r["loss"], r["accuracy"]
            if test_b_batches is not None:
                r = evaluate_precollated(model, test_b_batches, device)
                test_b_loss, test_b_acc = r["loss"], r["accuracy"]
            # Train loss: either the running training loss from meta.json
            # (fast, default) or a fresh re-eval on a train subset (slow).
            if train_batches is not None:
                r = evaluate_precollated(model, train_batches, device)
                train_loss, train_acc = r["loss"], r["accuracy"]
            else:
                meta_loss = get_training_loss_from_meta(checkpoint_path)
                if meta_loss is not None:
                    train_loss = float(meta_loss)
                # train_acc stays NaN — meta.json doesn't store accuracy.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate: {e}")
        parts = [
            f"Train loss={train_loss:.4f}"
            + (f" acc={train_acc:.3f}" if not np.isnan(train_acc) else ""),
            f"Val loss={val_loss:.4f} acc={val_acc:.3f}",
        ]
        if test_a_batches is not None:
            parts.append(f"TestA loss={test_a_loss:.4f} acc={test_a_acc:.3f}")
        if test_b_batches is not None:
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

        # Incremental save — a timeout leaves every completed checkpoint
        # on disk, and --resume picks up where we left off on the next
        # sbatch. Cheap: rewriting a few dozen KB per checkpoint.
        with open(output_path, "w") as _f:
            json.dump(results, _f, indent=2)

    # output_path already resolved (and updated incrementally after each
    # checkpoint). Nothing to re-dump here.
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
