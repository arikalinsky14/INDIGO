#!/usr/bin/env python3
"""
Learning Rate Finder for INDIGO (FlexMaterialMLP / FlexMaterialCrossAttn).

Imports collate / step utilities from `scripts/training.py` so the search
loop is bit-identical to a real training run. Tune LR per head — the
optimum is architecture-dependent, so re-run this sweep whenever you
change `--head-mode`.

For a single-pass production run on N rows, the optimal LR scales as a
power law in N. The recommended workflow is:

1. Run this script multiple times with different `--limit-examples`,
   each at `--epochs 1`. Each run writes a JSON to
   `outputs/lr_search/lr_search_ep1_lim<N>.json`.

      for N in 500000 1000000 2000000; do
          python scripts/lr_tuning.py \
              --data-dir data/train --epochs 1 \
              --limit-examples ${N} --limit-val-examples 10000 \
              --n-lrs 6 --lr-min 1e-5 --lr-max 5e-3
      done

2. Fit log(lr_opt) = a + b * log(N) and extrapolate to your production N:

      python scripts/fit_lr_scaling.py \
          --results-dir outputs/lr_search --target-examples 10000000 --plot

The multi-N approach is more principled than running multiple epochs on a
small subset: AdamW dynamics with repeated examples differ from those
with fresh examples, so the latter contaminates the fit.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from scripts.training import collate_fn, run_one_epoch
from src.dataset import FlexThinFilmDataset, find_repo_root
from src.model import ModelConfig, build_model, compute_loss


@dataclass
class LRSearchResult:
    lr: float
    epochs: int
    final_train_loss: float
    final_val_loss: float
    final_val_acc: float
    best_val_loss: float
    best_val_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    val_accs: List[float]


def evaluate_validation(model, val_loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            losses = compute_loss(model, batch_on_device)
            count = batch_on_device["lab"].size(0)
            total_loss += losses["loss"].item() * count
            total_correct += int(losses["accuracy"].item() * count)
            total_samples += count
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_with_lr(
    lr: float,
    epochs: int,
    train_dataset: FlexThinFilmDataset,
    val_dataset: FlexThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    log_every: int = 100,
    verbose: bool = True,
) -> LRSearchResult:
    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader_kw = {}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, **loader_kw,
    )

    n_examples = len(train_dataset)
    steps_per_epoch = math.ceil(n_examples / batch_size)
    total_steps = steps_per_epoch * epochs

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    global_step = 0

    for epoch in range(epochs):
        epoch_out = run_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            total_steps=total_steps,
            base_lr=lr,
            warmup_fraction=warmup_fraction,
            grad_clip=grad_clip,
            log_every=log_every,
            verbose=verbose,
            global_step_start=global_step,
        )
        global_step = epoch_out["global_step"]
        avg_train_loss = epoch_out["avg_loss"]
        train_losses.append(avg_train_loss)

        val_loss, val_acc = evaluate_validation(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1

        if verbose:
            warmup_steps = int(total_steps * warmup_fraction)
            phase = "warmup" if global_step <= warmup_steps else "decay"
            print(
                f"    Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                f"lr={epoch_out['final_lr']:.2e} [{phase}]",
                flush=True,
            )

    return LRSearchResult(
        lr=lr,
        epochs=epochs,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        final_val_acc=val_accs[-1],
        best_val_loss=best_val_loss,
        best_val_epoch=best_val_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accs=val_accs,
    )


def lr_tuning(
    epochs: int,
    train_dataset: FlexThinFilmDataset,
    val_dataset: FlexThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    n_lrs: int = 8,
    batch_size: int = 64,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    log_every: int = 100,
    verbose: bool = True,
) -> Tuple[float, List[LRSearchResult]]:
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), n_lrs)
    print(f"\n{'=' * 70}")
    print(f"LEARNING RATE SEARCH - {epochs} Epochs")
    print(f"{'=' * 70}")
    print(f"LR range: {lr_min:.0e} to {lr_max:.0e}")
    print(f"Testing {n_lrs} values: {[f'{lr:.2e}' for lr in lrs]}")
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")
    print(f"Warmup fraction: {warmup_fraction:.1%}")
    print(f"{'=' * 70}\n")

    results: List[LRSearchResult] = []
    for i, lr in enumerate(lrs):
        print(f"[{i + 1}/{n_lrs}] Training with LR = {lr:.2e}")
        result = train_with_lr(
            lr=lr, epochs=epochs,
            train_dataset=train_dataset, val_dataset=val_dataset,
            config=config, device=device,
            batch_size=batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            weight_decay=weight_decay, grad_clip=grad_clip,
            warmup_fraction=warmup_fraction,
            log_every=log_every, verbose=verbose,
        )
        results.append(result)
        print(f"    Final: train_loss={result.final_train_loss:.4f}, "
              f"val_loss={result.final_val_loss:.4f}, "
              f"best_val_loss={result.best_val_loss:.4f}\n")

    best_result = min(results, key=lambda r: r.best_val_loss)
    return best_result.lr, results


def plot_lr_search(results: List[LRSearchResult], output_path: Path, epochs: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot")
        return
    lrs = [r.lr for r in results]
    best_val_losses = [r.best_val_loss for r in results]
    final_val_losses = [r.final_val_loss for r in results]
    final_train_losses = [r.final_train_loss for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.semilogx(lrs, best_val_losses, "b-o", label="Best Val Loss", linewidth=2, markersize=8)
    ax1.semilogx(lrs, final_val_losses, "r--s", label="Final Val Loss", linewidth=1.5, markersize=6)
    ax1.semilogx(lrs, final_train_losses, "g:^", label="Final Train Loss", linewidth=1.5, markersize=6)
    best_idx = int(np.argmin(best_val_losses))
    ax1.axvline(x=lrs[best_idx], color="blue", linestyle="--", alpha=0.5)
    ax1.scatter([lrs[best_idx]], [best_val_losses[best_idx]], color="blue", s=150,
                zorder=5, marker="*", label=f"Optimal LR: {lrs[best_idx]:.2e}")
    ax1.set_xlabel("Learning Rate", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"LR Search Results ({epochs} Epochs)", fontsize=14)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    best_result = results[best_idx]
    ax2 = axes[1]
    epoch_range = range(1, epochs + 1)
    ax2.plot(epoch_range, best_result.train_losses, "b-", label="Train Loss", linewidth=2)
    ax2.plot(epoch_range, best_result.val_losses, "r-", label="Val Loss", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title(f"Training Curves (LR={best_result.lr:.2e})", fontsize=14)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal LR for INDIGO")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Subsetting: limit how many rows of the train / validation pool the LR
    # sweep uses. Set --limit-examples for the multi-N scaling-law workflow
    # (see module docstring).
    parser.add_argument("--limit-examples", type=int, default=None,
                        help="Limit training rows used by the LR sweep")
    parser.add_argument("--limit-val-examples", type=int, default=None,
                        help="Limit validation rows used by the LR sweep")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream dataset shard-by-shard (recommended at "
                             "production scale; the legacy mode OOMs).")

    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--n-lrs", type=int, default=8)

    parser.add_argument("--feature-mode", type=str, default="raw_spectrum",
                        choices=["raw_spectrum", "compact"])
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=8)
    # Keep aligned with scripts/training.py: optimal LR depends on regularisation.
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-mode", type=str, default="mlp",
                        choices=["mlp", "cross_attn"],
                        help="Must match the architecture you plan to train; "
                             "optimal LR is head-dependent.")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Attention heads (cross_attn only).")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="DataLoader prefetch_factor (default: 1; matches "
                             "training.py — pipeline is producer-bound)")
    # Keep aligned with scripts/training.py: optimal LR is regulariser-sensitive.
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")

    # Logging
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print per-step loss every N optimizer steps "
                             "within each LR trial (default: 100)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print per-step loss + per-epoch val (default: on). "
                             "Pass --no-verbose to silence.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    data_dir = Path(args.data_dir)
    if args.limit_examples is None:
        print(f"[INFO] Loading FULL training data...")
    else:
        print(f"[INFO] Loading training data (limited to {args.limit_examples} rows)...")
    train_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="train", verbose=True,
        limit_examples=args.limit_examples, streaming=args.streaming,
    )
    print(f"[INFO] Loading validation data...")
    val_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="validation", verbose=True,
        limit_examples=args.limit_val_examples, streaming=args.streaming,
    )

    config = ModelConfig(
        feature_mode=args.feature_mode,
        encoder_hidden=args.encoder_hidden,
        encoder_out=args.encoder_out,
        encoder_dropout=args.encoder_dropout,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        head_mode=args.head_mode,
        n_heads=args.n_heads,
    )
    print(f"[INFO] Model config: head_mode={args.head_mode}, "
          f"d_model={args.d_model}, n_layers={args.n_layers}")

    optimal_lr, results = lr_tuning(
        epochs=args.epochs,
        train_dataset=train_dataset, val_dataset=val_dataset,
        config=config, device=device,
        lr_min=args.lr_min, lr_max=args.lr_max, n_lrs=args.n_lrs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
        log_every=args.log_every, verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print("LR SEARCH RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'LR':>12} | {'Best Val Loss':>14} | {'Final Val Loss':>14} | {'Final Train':>12}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.lr):
        marker = " *" if r.lr == optimal_lr else ""
        print(f"{r.lr:>12.2e} | {r.best_val_loss:>14.4f} | {r.final_val_loss:>14.4f} | {r.final_train_loss:>12.4f}{marker}")
    print("-" * 60)
    print(f"\n* OPTIMAL LR for {args.epochs} epochs: {optimal_lr:.2e}")
    best_result = next(r for r in results if r.lr == optimal_lr)
    print(f"  Best validation loss: {best_result.best_val_loss:.4f} (epoch {best_result.best_val_epoch})")
    print("=" * 70)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        try:
            repo_root = find_repo_root()
            output_dir = repo_root / "outputs" / "lr_search"
        except Exception:
            output_dir = Path("./outputs/lr_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag output filenames with the train-subset size so multi-N runs
    # (the scaling-law workflow) don't clobber each other.
    n_train = len(train_dataset)
    tag = f"ep{args.epochs}_lim{n_train}"
    results_file = output_dir / f"lr_search_{tag}.json"
    with open(results_file, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "optimal_lr": optimal_lr,
            "lr_range": [args.lr_min, args.lr_max],
            "n_lrs": args.n_lrs,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "warmup_fraction": args.warmup_fraction,
            "train_examples": n_train,
            "val_examples": len(val_dataset),
            "limit_examples": args.limit_examples,
            "limit_val_examples": args.limit_val_examples,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\n[INFO] Results saved to {results_file}")

    if args.plot:
        plot_path = output_dir / f"lr_search_{tag}.png"
        plot_lr_search(results, plot_path, args.epochs)

    print(f"\n# Machine-readable output for log-log fitting:")
    print(f"EPOCHS={args.epochs}")
    print(f"OPTIMAL_LR={optimal_lr}")
    print(f"BEST_VAL_LOSS={best_result.best_val_loss}")


if __name__ == "__main__":
    main()
