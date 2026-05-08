#!/usr/bin/env python3
"""
Learning Rate Finder for CHROMA-Lite

Finds the optimal learning rate for a given number of epochs by training
with multiple LR values and selecting the one with lowest validation loss.

This script IMPORTS from training.py to ensure identical training behavior.

Usage:
    # Find optimal LR for 4 epochs
    python lr_tuning.py --epochs 4 --data-dir /path/to/data_prompts
    
    # Custom LR range
    python lr_tuning.py --epochs 8 --lr-min 1e-5 --lr-max 1e-2 --n-lrs 10

After running for multiple epoch values (4, 6, 8, 12), use fit_lr_scaling.py
to fit a line and extrapolate to 200 epochs.
"""

import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

# Setup path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

# Import from existing modules - SINGLE SOURCE OF TRUTH
from pretrain_rgb_to_structure.scripts.training import collate_fn, get_lr_schedule, set_lr, train_step
from src.dataset import ThinFilmDataset, find_repo_root
from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig, compute_loss


@dataclass
class LRSearchResult:
    """Result of a single LR training run."""
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
    """
    Evaluate model on validation set using teacher forcing.
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            structure_matrix = batch['structure_matrix'].to(device)
            target_token = batch['target_token'].to(device)
            
            losses = compute_loss(model, rgb, structure_matrix, target_token)
            
            count = rgb.size(0)
            total_loss += losses['loss'].item() * count
            total_correct += int(losses['accuracy'].item() * count)
            total_samples += count
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    
    return avg_loss, avg_acc


def train_with_lr(
    lr: float,
    epochs: int,
    train_dataset: ThinFilmDataset,
    val_dataset: ThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    verbose: bool = False,
) -> LRSearchResult:
    """
    Train a model with a specific learning rate and return results.
    
    Uses the same training loop structure as training.py.
    """
    # Create fresh model
    model = ThinFilmMLP(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create data loaders using collate_fn from training.py
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )
    
    # Calculate total steps for LR schedule (same as training.py)
    n_examples = len(train_dataset)
    steps_per_epoch = math.ceil(n_examples / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_fraction)
    
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    best_val_epoch = 0
    
    global_step = 0
    
    # Training loop - mirrors training.py exactly
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # Update learning rate based on schedule (from training.py)
            current_lr = get_lr_schedule(global_step, total_steps, lr, warmup_fraction)
            set_lr(optimizer, current_lr)
            
            # Training step (from training.py)
            optimizer.zero_grad()
            losses = train_step(model, batch, device)
            losses['loss'].backward()
            
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            epoch_loss += losses['loss'].item()
            epoch_acc += losses['accuracy'].item()
            n_batches += 1
            global_step += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_validation(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
        
        if verbose:
            phase = "warmup" if global_step <= warmup_steps else "decay"
            print(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, lr={current_lr:.2e} [{phase}]")
    
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
    train_dataset: ThinFilmDataset,
    val_dataset: ThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    n_lrs: int = 8,
    batch_size: int = 256,
    num_workers: int = 4,
    weight_decay: float = 0.00,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    verbose: bool = True,
) -> Tuple[float, List[LRSearchResult]]:
    """
    Find optimal learning rate by grid search.
    
    Returns:
        Tuple of (optimal_lr, list of all results)
    """
    # Generate log-spaced learning rates
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), n_lrs)
    
    print(f"\n{'='*70}")
    print(f"LEARNING RATE SEARCH - {epochs} Epochs")
    print(f"{'='*70}")
    print(f"LR range: {lr_min:.0e} to {lr_max:.0e}")
    print(f"Testing {n_lrs} values: {[f'{lr:.2e}' for lr in lrs]}")
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")
    print(f"Warmup fraction: {warmup_fraction:.1%}")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, lr in enumerate(lrs):
        print(f"[{i+1}/{n_lrs}] Training with LR = {lr:.2e}")
        
        result = train_with_lr(
            lr=lr,
            epochs=epochs,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_fraction=warmup_fraction,
            verbose=verbose,
        )
        results.append(result)
        
        print(f"    Final: train_loss={result.final_train_loss:.4f}, "
              f"val_loss={result.final_val_loss:.4f}, best_val_loss={result.best_val_loss:.4f}\n")
    
    # Find optimal LR (lowest validation loss)
    best_result = min(results, key=lambda r: r.best_val_loss)
    
    return best_result.lr, results


def plot_lr_search(results: List[LRSearchResult], output_path: Path, epochs: int):
    """Plot LR vs validation loss."""
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
    
    # Plot 1: LR vs Loss
    ax1 = axes[0]
    ax1.semilogx(lrs, best_val_losses, 'b-o', label='Best Val Loss', linewidth=2, markersize=8)
    ax1.semilogx(lrs, final_val_losses, 'r--s', label='Final Val Loss', linewidth=1.5, markersize=6)
    ax1.semilogx(lrs, final_train_losses, 'g:^', label='Final Train Loss', linewidth=1.5, markersize=6)
    
    # Mark the optimal LR
    best_idx = np.argmin(best_val_losses)
    ax1.axvline(x=lrs[best_idx], color='blue', linestyle='--', alpha=0.5)
    ax1.scatter([lrs[best_idx]], [best_val_losses[best_idx]], color='blue', s=150, 
                zorder=5, marker='*', label=f'Optimal LR: {lrs[best_idx]:.2e}')
    
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'LR Search Results ({epochs} Epochs)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training curves for best LR
    best_result = results[best_idx]
    ax2 = axes[1]
    epoch_range = range(1, epochs + 1)
    ax2.plot(epoch_range, best_result.train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epoch_range, best_result.val_losses, 'r-', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Training Curves (LR={best_result.lr:.2e})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Find optimal learning rate for CHROMA-Lite')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data_prompts/ directory')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs to train for each LR')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # LR search parameters
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Minimum learning rate to try (default: 1e-5)')
    parser.add_argument('--lr-max', type=float, default=1e-2,
                        help='Maximum learning rate to try (default: 1e-2)')
    parser.add_argument('--n-lrs', type=int, default=8,
                        help='Number of learning rates to try (default: 8)')
    
    # Model architecture (should match main training)
    parser.add_argument('--d-model', type=int, default=1024,
                        help='Hidden layer dimension (default: 1024)')
    parser.add_argument('--n-layers', type=int, default=8,
                        help='Number of hidden layers (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    
    # Training parameters (should match training.py defaults)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Weight decay (default: 0.00)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (default: 1.0)')
    parser.add_argument('--warmup-fraction', type=float, default=0.02,
                        help='Warmup fraction (default: 0.02)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: outputs/lr_search/)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training progress')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load FULL datasets (no limiting)
    print(f"[INFO] Loading FULL training data...")
    train_dataset = ThinFilmDataset(
        data_dir, seed=args.seed, split='train', verbose=True
    )
    
    print(f"[INFO] Loading FULL validation data...")
    val_dataset = ThinFilmDataset(
        data_dir, seed=args.seed, split='validation', verbose=True
    )
    
    # Create model config
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    
    print(f"[INFO] Model config: d_model={args.d_model}, n_layers={args.n_layers}")
    
    # Run LR search
    optimal_lr, results = lr_tuning(
        epochs=args.epochs,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        n_lrs=args.n_lrs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
        verbose=args.verbose,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("LR SEARCH RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'LR':>12} | {'Best Val Loss':>14} | {'Final Val Loss':>14} | {'Final Train':>12}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x.lr):
        marker = " *" if r.lr == optimal_lr else ""
        print(f"{r.lr:>12.2e} | {r.best_val_loss:>14.4f} | {r.final_val_loss:>14.4f} | {r.final_train_loss:>12.4f}{marker}")
    
    print("-" * 60)
    print(f"\n✓ OPTIMAL LR for {args.epochs} epochs: {optimal_lr:.2e}")
    
    best_result = next(r for r in results if r.lr == optimal_lr)
    print(f"  Best validation loss: {best_result.best_val_loss:.4f} (epoch {best_result.best_val_epoch})")
    print("=" * 70)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        try:
            repo_root = find_repo_root()
            output_dir = repo_root / 'pretrain_rgb_to_structure' / 'outputs' / 'lr_search'
        except:
            output_dir = Path('./outputs/lr_search')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'lr_search_ep{args.epochs}.json'
    results_data = {
        'epochs': args.epochs,
        'optimal_lr': optimal_lr,
        'lr_range': [args.lr_min, args.lr_max],
        'n_lrs': args.n_lrs,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'warmup_fraction': args.warmup_fraction,
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset),
        'results': [asdict(r) for r in results],
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[INFO] Results saved to {results_file}")
    
    # Generate plot if requested
    if args.plot:
        plot_path = output_dir / f'lr_search_ep{args.epochs}.png'
        plot_lr_search(results, plot_path, args.epochs)
    
    # Print machine-readable output for scripting
    print(f"\n# Machine-readable output for log-log fitting:")
    print(f"EPOCHS={args.epochs}")
    print(f"OPTIMAL_LR={optimal_lr}")
    print(f"BEST_VAL_LOSS={best_result.best_val_loss}")


if __name__ == "__main__":
    main()