#!/usr/bin/env python3
"""
Learning Rate Finder for Full Text → Structure Model

Finds the optimal learning rate by training with multiple LR values and
selecting the one with lowest validation loss.

Supports --limit-examples to quickly test at different data scales, enabling
log(examples) vs log(optimal_lr) extrapolation plots.

The LLM backbone (TinyLlama) is loaded ONCE and reused across all trials.
Only the trainable components (LoRA adapters + ConstraintMLP + MixingMLP)
are reinitialized for each LR trial.

Usage:
    # Standard LR search:
    python train_full_model/scripts/lr_tuning.py \\
        --cache-dir-train train_full_model/data/cache_train \\
        --cache-dir-val train_full_model/data/cache_test \\
        --epochs 3 --verbose

    # Quick search with limited data:
    python train_full_model/scripts/lr_tuning.py \\
        --cache-dir-train train_full_model/data/cache_train \\
        --cache-dir-val train_full_model/data/cache_test \\
        --epochs 3 --limit-examples 10000

    # Custom LR range:
    python train_full_model/scripts/lr_tuning.py \\
        --cache-dir-train train_full_model/data/cache_train \\
        --cache-dir-val train_full_model/data/cache_test \\
        --epochs 3 --lr-min 1e-5 --lr-max 1e-2 --n-lrs 10

After running for multiple --limit-examples values (e.g. 5000, 20000, 100000),
plot log(examples) vs log(optimal_lr) and fit a line to extrapolate.
"""

import sys
import json
import argparse
import math
import time
import copy
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

# Import from existing full model modules
from train_full_model.scripts.training import (
    CachedFullModelDataset, collate_full_model,
    get_lr, set_lr, log,
)
from train_full_model.src.model import (
    FullModelConfig, FullModel, collate_constraint_inputs,
)


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class LRSearchResult:
    """Result of a single LR training run."""
    lr: float
    epochs: int
    limit_examples: int
    final_train_loss: float
    final_val_loss: float
    final_val_acc: float
    best_val_loss: float
    best_val_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    val_accs: List[float]


# ============================================================================
# Validation
# ============================================================================

def evaluate_validation(
    model: FullModel,
    val_loader: DataLoader,
    device: torch.device,
    config: FullModelConfig,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, float]:
    """
    Evaluate on validation set. Returns (loss, accuracy).
    """
    model.constraint_mlp.eval()
    model.mixing_mlp.eval()
    if model.llm is not None:
        model.llm.eval()

    use_amp = autocast_dtype is not None
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Tokenization done in collate function (DataLoader workers)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            base_logits = batch['base_logits'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            structure_matrices = batch['structure_matrices'].to(device)
            n_steps = batch['n_steps'].to(device)

            with torch.amp.autocast(
                    device_type='cuda', dtype=autocast_dtype,
                    enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cached_base_logits=base_logits,
                    n_steps=n_steps,
                    structure_matrices=structure_matrices,
                    target_tokens=target_tokens,
                )

            total_loss += outputs['loss'].item()
            total_acc += outputs['accuracy'].item()
            n_batches += 1

    return (total_loss / max(n_batches, 1),
            total_acc / max(n_batches, 1))


# ============================================================================
# Single LR Trial
# ============================================================================

def get_initial_trainable_state(model: FullModel) -> dict:
    """Capture initial state of all trainable parameters for resetting between trials."""
    state = {}
    state['constraint_mlp'] = copy.deepcopy(model.constraint_mlp.state_dict())
    state['mixing_mlp'] = copy.deepcopy(model.mixing_mlp.state_dict())
    if model.llm is not None:
        # Save only LoRA adapter weights (the trainable ones)
        state['lora'] = {
            name: param.data.clone()
            for name, param in model.llm.named_parameters()
            if param.requires_grad
        }
    return state


def reset_trainable_params(model: FullModel, initial_state: dict):
    """Reset all trainable parameters to initial state (fresh model for each LR trial)."""
    model.constraint_mlp.load_state_dict(initial_state['constraint_mlp'])
    model.mixing_mlp.load_state_dict(initial_state['mixing_mlp'])
    if 'lora' in initial_state and model.llm is not None:
        for name, param in model.llm.named_parameters():
            if param.requires_grad and name in initial_state['lora']:
                param.data.copy_(initial_state['lora'][name])


def reinitialize_trainable_params(model: FullModel):
    """Reinitialize trainable params from scratch (new random init each trial)."""
    # Reinit ConstraintMLP
    model.constraint_mlp._init_weights()
    # Reinit MixingMLP
    model.mixing_mlp._init_weights()
    # Reinit LoRA adapters (they use Kaiming uniform by default in peft)
    if model.llm is not None:
        for name, param in model.llm.named_parameters():
            if param.requires_grad:
                if 'lora_A' in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif 'lora_B' in name:
                    nn.init.zeros_(param)


def train_with_lr(
    lr: float,
    epochs: int,
    model: FullModel,
    config: FullModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    verbose: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> LRSearchResult:
    """
    Train with a specific LR and return results.

    The model's trainable parameters are reinitialized before training.
    The frozen LLM backbone is shared across trials.
    """
    # Fresh random init for all trainable params
    reinitialize_trainable_params(model)

    use_amp = autocast_dtype is not None
    use_scaler = scaler is not None and scaler.is_enabled()

    # Optimizer
    param_groups = [
        {'params': [p for p in model.llm.parameters() if p.requires_grad],
         'lr': lr, 'weight_decay': config.weight_decay},
        {'params': list(model.constraint_mlp.parameters()),
         'lr': lr, 'weight_decay': config.weight_decay},
        {'params': list(model.mixing_mlp.parameters()),
         'lr': lr, 'weight_decay': config.weight_decay},
    ]
    optimizer = AdamW(param_groups)

    # Schedule
    n_examples = len(train_loader.dataset)
    effective_batch = config.batch_size * config.grad_accum_steps
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * config.warmup_fraction)

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    best_val_epoch = 0
    global_step = 0

    for epoch in range(epochs):
        # --- Train ---
        model.constraint_mlp.train()
        model.mixing_mlp.train()
        if model.llm is not None:
            # Gradient checkpointing requires train() mode
            if getattr(model, '_gradient_checkpointing', False):
                model.llm.train()
            else:
                model.llm.eval()

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_opt_steps = 0
        accum_loss = 0.0
        accum_acc = 0.0

        for batch_idx, batch in enumerate(train_loader):
            current_lr = get_lr(global_step, total_steps, lr, config.warmup_fraction)
            set_lr(optimizer, current_lr)

            # Tokenization done in collate function (DataLoader workers)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            base_logits = batch['base_logits'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            structure_matrices = batch['structure_matrices'].to(device)
            n_steps = batch['n_steps'].to(device)

            with torch.amp.autocast(
                    device_type='cuda', dtype=autocast_dtype,
                    enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cached_base_logits=base_logits,
                    n_steps=n_steps,
                    structure_matrices=structure_matrices,
                    target_tokens=target_tokens,
                )
                loss = outputs['loss'] / config.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += outputs['loss'].item()
            accum_acc += outputs['accuracy'].item()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if config.grad_clip > 0:
                    trainable_params = (
                        list(model.constraint_mlp.parameters()) +
                        list(model.mixing_mlp.parameters()) +
                        [p for p in model.llm.parameters() if p.requires_grad]
                    )
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / config.grad_accum_steps
                avg_acc = accum_acc / config.grad_accum_steps
                epoch_loss += avg_loss
                epoch_acc += avg_acc
                n_opt_steps += 1
                global_step += 1
                accum_loss = 0.0
                accum_acc = 0.0

        avg_train_loss = epoch_loss / max(n_opt_steps, 1)
        avg_train_acc = epoch_acc / max(n_opt_steps, 1)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        val_loss, val_acc = evaluate_validation(
            model, val_loader, device, config,
            autocast_dtype=autocast_dtype)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1

        if verbose:
            phase = "warmup" if global_step <= warmup_steps else "decay"
            log(f"    Ep {epoch+1}/{epochs}: train={avg_train_loss:.4f}, "
                f"val={val_loss:.4f}, acc={val_acc:.3f}, "
                f"lr={current_lr:.2e} [{phase}]")

    return LRSearchResult(
        lr=lr,
        epochs=epochs,
        limit_examples=len(train_loader.dataset),
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        final_val_acc=val_accs[-1],
        best_val_loss=best_val_loss,
        best_val_epoch=best_val_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accs=val_accs,
    )


# ============================================================================
# LR Search
# ============================================================================

def lr_search(
    epochs: int,
    model: FullModel,
    config: FullModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    n_lrs: int = 8,
    verbose: bool = True,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, List[LRSearchResult]]:
    """Run grid search over log-spaced learning rates."""
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), n_lrs)

    log(f"\n{'='*70}")
    log(f"LR SEARCH: {n_lrs} values in [{lr_min:.1e}, {lr_max:.1e}], "
        f"{epochs} epochs, {len(train_loader.dataset):,} train examples")
    log(f"{'='*70}")

    results = []
    for i, lr in enumerate(lrs):
        log(f"\n--- Trial {i+1}/{n_lrs}: LR = {lr:.2e} ---")
        t0 = time.time()

        result = train_with_lr(
            lr=lr,
            epochs=epochs,
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            verbose=verbose,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
        )
        results.append(result)

        elapsed = time.time() - t0
        log(f"    Result: val={result.best_val_loss:.4f} "
            f"(ep {result.best_val_epoch}), "
            f"train={result.final_train_loss:.4f}, "
            f"time={elapsed:.0f}s")

    best_result = min(results, key=lambda r: r.best_val_loss)
    return best_result.lr, results


# ============================================================================
# Plotting
# ============================================================================

def plot_lr_search(results: List[LRSearchResult], output_path: Path, epochs: int):
    """Plot LR vs validation loss."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log("[WARN] matplotlib not available, skipping plot")
        return

    lrs = [r.lr for r in results]
    best_val = [r.best_val_loss for r in results]
    final_val = [r.final_val_loss for r in results]
    final_train = [r.final_train_loss for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: LR vs Loss
    ax1 = axes[0]
    ax1.semilogx(lrs, best_val, 'b-o', label='Best Val Loss', lw=2, ms=8)
    ax1.semilogx(lrs, final_val, 'r--s', label='Final Val Loss', lw=1.5, ms=6)
    ax1.semilogx(lrs, final_train, 'g:^', label='Final Train Loss', lw=1.5, ms=6)

    best_idx = int(np.argmin(best_val))
    ax1.axvline(x=lrs[best_idx], color='blue', ls='--', alpha=0.5)
    ax1.scatter([lrs[best_idx]], [best_val[best_idx]], color='blue', s=150,
                zorder=5, marker='*', label=f'Optimal: {lrs[best_idx]:.2e}')

    n_examples = results[0].limit_examples
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'LR Search ({epochs} ep, {n_examples:,} examples)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training curves for best LR
    best = results[best_idx]
    ax2 = axes[1]
    ep_range = range(1, epochs + 1)
    ax2.plot(ep_range, best.train_losses, 'b-', label='Train', lw=2)
    ax2.plot(ep_range, best.val_losses, 'r-', label='Val', lw=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Best LR={best.lr:.2e}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"[INFO] Plot saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='LR finder for full text → structure model')

    # Data
    parser.add_argument('--cache-dir-train', type=str, required=True,
                        help='Path to cached train data')
    parser.add_argument('--cache-dir-val', type=str, required=True,
                        help='Path to cached validation/test data')
    parser.add_argument('--limit-examples', type=int, default=None,
                        help='Limit training examples (for log-log extrapolation)')
    parser.add_argument('--limit-val', type=int, default=None,
                        help='Limit validation examples (default: use all)')

    # LR search
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--lr-max', type=float, default=1e-2)
    parser.add_argument('--n-lrs', type=int, default=8)
    parser.add_argument('--epochs', type=int, required=True)

    # Architecture
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--constraint-layers', type=int, default=4)
    parser.add_argument('--mixing-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # LLM / LoRA
    parser.add_argument('--encoder', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-text-len', type=int, default=756)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', type=str, default='q_proj,v_proj')

    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-fraction', type=float, default=0.02)
    parser.add_argument('--num-workers', type=int, default=4)

    # Performance
    parser.add_argument('--mixed-precision', type=str, default='bf16',
                        choices=['bf16', 'fp16', 'none'],
                        help='Mixed precision mode (default: bf16)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        default=False,
                        help='Enable gradient checkpointing on LLM (saves memory)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='torch.compile() the MLPs (requires PyTorch 2.0+)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[INFO] Device: {device}")

    try:
        from src.dataset import find_repo_root
        repo_root = find_repo_root()
    except Exception:
        repo_root = _repo_root

    # ================================================================
    # Mixed precision setup
    # ================================================================

    if args.mixed_precision == 'bf16':
        autocast_dtype = torch.bfloat16
        scaler = None
        log("[INFO] Mixed precision: bfloat16")
    elif args.mixed_precision == 'fp16':
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
        log("[INFO] Mixed precision: float16 with GradScaler")
    else:
        autocast_dtype = None
        scaler = None
        log("[INFO] Mixed precision: disabled (float32)")

    # ================================================================
    # Load cached datasets
    # ================================================================

    log(f"\n[INFO] Loading cached train data: {args.cache_dir_train}")
    train_dataset = CachedFullModelDataset(
        args.cache_dir_train, limit_examples=args.limit_examples)

    log(f"[INFO] Loading cached val data: {args.cache_dir_val}")
    val_dataset = CachedFullModelDataset(
        args.cache_dir_val, limit_examples=args.limit_val)

    # ================================================================
    # Build model config (LR placeholder — overridden per trial)
    # ================================================================

    config = FullModelConfig(
        d_model=args.d_model,
        constraint_layers=args.constraint_layers,
        mixing_layers=args.mixing_layers,
        dropout=args.dropout,
        encoder_name=args.encoder,
        max_text_len=args.max_text_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        learning_rate=0.0,  # placeholder
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        limit_examples=args.limit_examples,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
    )

    # ================================================================
    # Load model ONCE (LLM is expensive)
    # ================================================================

    log(f"\n[INFO] Loading model (LLM loaded once, reused across trials)...")
    model = FullModel(config)
    model.load_llm(device, gradient_checkpointing=args.gradient_checkpointing)
    model.constraint_mlp.to(device)
    model.mixing_mlp.to(device)

    # Optional torch.compile for MLPs
    if args.compile:
        log("[INFO] Compiling MLPs with torch.compile()...")
        model.constraint_mlp = torch.compile(model.constraint_mlp)
        model.mixing_mlp = torch.compile(model.mixing_mlp)

    n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    n_constraint = sum(p.numel() for p in model.constraint_mlp.parameters())
    n_mixing = sum(p.numel() for p in model.mixing_mlp.parameters())
    log(f"[INFO] Trainable: LoRA={n_lora:,}, ConstraintMLP={n_constraint:,}, "
        f"MixingMLP={n_mixing:,}, Total={n_lora+n_constraint+n_mixing:,}")

    # ================================================================
    # Data loaders (with tokenization in collate for worker parallelism)
    # ================================================================

    def make_collate_fn(tokenizer, max_text_len):
        """Wrap collate to include tokenization in DataLoader workers."""
        def collate_fn(batch):
            result = collate_full_model(batch)
            tokenized = collate_constraint_inputs(
                result['texts'], tokenizer, max_text_len)
            result['input_ids'] = tokenized['input_ids']
            result['attention_mask'] = tokenized['attention_mask']
            return result
        return collate_fn

    collate_fn = make_collate_fn(model.tokenizer, config.max_text_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None)

    # ================================================================
    # Run LR search
    # ================================================================

    optimal_lr, results = lr_search(
        epochs=args.epochs,
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        n_lrs=args.n_lrs,
        verbose=args.verbose,
        scaler=scaler,
        autocast_dtype=autocast_dtype,
    )

    # ================================================================
    # Summary
    # ================================================================

    n_train = len(train_dataset)
    log(f"\n{'='*70}")
    log(f"LR SEARCH RESULTS — {args.epochs} epochs, {n_train:,} examples")
    log(f"{'='*70}")
    log(f"\n{'LR':>12} | {'Best Val':>10} | {'Final Val':>10} | "
        f"{'Final Train':>12} | {'Best Ep':>7}")
    log(f"{'-'*62}")

    for r in sorted(results, key=lambda x: x.lr):
        marker = " *" if r.lr == optimal_lr else ""
        log(f"{r.lr:>12.2e} | {r.best_val_loss:>10.4f} | "
            f"{r.final_val_loss:>10.4f} | {r.final_train_loss:>12.4f} | "
            f"{r.best_val_epoch:>7}{marker}")

    log(f"{'-'*62}")
    best = next(r for r in results if r.lr == optimal_lr)
    log(f"\n✓ OPTIMAL LR: {optimal_lr:.2e}")
    log(f"  Best val loss: {best.best_val_loss:.4f} (epoch {best.best_val_epoch})")
    log(f"  Train examples: {n_train:,}")
    log(f"{'='*70}")

    # ================================================================
    # Save results
    # ================================================================

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / 'train_full_model' / 'outputs' / 'lr_search'

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_lim{args.limit_examples}" if args.limit_examples else ""
    results_file = output_dir / f'lr_search_ep{args.epochs}{suffix}.json'

    results_data = {
        'epochs': args.epochs,
        'optimal_lr': optimal_lr,
        'lr_range': [args.lr_min, args.lr_max],
        'n_lrs': args.n_lrs,
        'limit_examples': args.limit_examples,
        'train_examples': n_train,
        'val_examples': len(val_dataset),
        'd_model': args.d_model,
        'constraint_layers': args.constraint_layers,
        'mixing_layers': args.mixing_layers,
        'lora_rank': args.lora_rank,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'warmup_fraction': args.warmup_fraction,
        'results': [asdict(r) for r in results],
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    log(f"\n[INFO] Results saved to {results_file}")

    if args.plot:
        plot_path = output_dir / f'lr_search_ep{args.epochs}{suffix}.png'
        plot_lr_search(results, plot_path, args.epochs)

    # Machine-readable output for scripting
    log(f"\n# For log-log extrapolation:")
    log(f"EPOCHS={args.epochs}")
    log(f"LIMIT_EXAMPLES={args.limit_examples or n_train}")
    log(f"OPTIMAL_LR={optimal_lr}")
    log(f"BEST_VAL_LOSS={best.best_val_loss}")


if __name__ == '__main__':
    main()
