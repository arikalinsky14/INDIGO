#!/usr/bin/env python3
"""
Learning Rate Tuner for CHROMA-Lite - Constrained LLM Fine-tuning

Trains the constrained LLM at multiple learning rates for 1 epoch on a fixed
data size, evaluating on a held-out validation set to find the optimal LR.

Strategy: instead of sweeping epoch counts, we sweep data sizes via
--limit-examples. One pass through the data is sufficient for LLM fine-tuning;
more data is better than more epochs over the same data.

Each LR trial:
  1. Loads a fresh base model + performs lm_head surgery
  2. Configures fine-tuning (LoRA / last_n / full)
  3. Trains for 1 epoch on the training set
  4. Evaluates: val CE loss (teacher forcing)

Since each trial requires a full model load, this is slower per-trial than the
MLP version. To keep wall time manageable:
  - Use --limit-examples to control training set size
  - Use fewer LR candidates (--n-lrs 8)

After running at multiple data sizes, use the results to pick the optimal LR
for your target training scale.

Usage:
    # Default: 8 LRs from 5e-5 to 5e-3, 1 epoch
    python pretrain_text_to_rgb/scripts/lr_tuning.py --limit-examples 5000

    # Custom range:
    python pretrain_text_to_rgb/scripts/lr_tuning.py \\
        --limit-examples 5000 --lr-min 1e-5 --lr-max 1e-2 --n-lrs 12

    # Sweep data sizes (submit separately via SLURM):
    for n in 1000 5000 10000 50000; do
      python pretrain_text_to_rgb/scripts/lr_tuning.py --limit-examples $n
    done
"""

import sys
import json
import argparse
import math
import time
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / 'pretrain_text_to_rgb'))

from src.dataset import TextThinFilmDataset, find_repo_root
from pretrain_text_to_rgb.src.model import (
    ConstrainedTextToRGBConfig,
    ConstrainedTextToRGBModel,
    format_training_example,
    format_chat_input,
    collate_fn,
    parse_rgb_string,
    normalized_to_rgb,
)


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# LR Schedule (same as training.py)
# ============================================================================

def get_lr_schedule(step: int, total_steps: int, base_lr: float,
                    warmup_fraction: float = 0.03) -> float:
    warmup_steps = int(total_steps * warmup_fraction)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    else:
        decay_steps = total_steps - warmup_steps
        decay_progress = (step - warmup_steps) / max(decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * decay_progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LRSearchResult:
    """Result of a single LR training run."""
    lr: float
    epochs: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_val_epoch: int
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    total_time_s: float = 0.0


# ============================================================================
# Evaluation (teacher-forcing CE loss on val set)
# ============================================================================

def evaluate_val(
    model: ConstrainedTextToRGBModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average CE loss on validation set (teacher forcing)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Single LR Trial
# ============================================================================

def run_single_lr(
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ConstrainedTextToRGBConfig,
    device: torch.device,
    epochs: int,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.03,
    grad_accum_steps: int = 4,
) -> LRSearchResult:
    """
    Train from scratch at a given LR, return validation metrics.

    Loads a fresh model for each trial to avoid any contamination.
    """
    log(f"\n{'='*50}")
    log(f"  LR = {lr:.2e}")
    log(f"{'='*50}")

    # Fresh model for this trial
    model = ConstrainedTextToRGBModel(config)
    model.load_model(device=device)

    # Optimizer: model backbone + compact lm_head
    trainable_params = [
        {'params': [p for p in model.model.parameters() if p.requires_grad],
         'lr': lr},
        {'params': model.compact_lm_head.parameters(), 'lr': lr},
    ]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=config.weight_decay)

    accum = grad_accum_steps
    micro_batches_per_epoch = len(train_loader)
    optim_steps_per_epoch = math.ceil(micro_batches_per_epoch / accum)
    total_steps = optim_steps_per_epoch * epochs

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_epoch = 0

    t0 = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        micro_step = 0
        optim_step = epoch * optim_steps_per_epoch

        optimizer.zero_grad()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / accum
            loss.backward()

            epoch_loss += outputs['loss'].item()  # unscaled for logging
            micro_step += 1

            if micro_step % accum == 0 or micro_step == micro_batches_per_epoch:
                step = optim_step + (micro_step // accum)
                current_lr = get_lr_schedule(step, total_steps, lr, warmup_fraction)
                set_lr(optimizer, current_lr)

                if grad_clip > 0:
                    all_params = list(model.model.parameters()) + \
                                 list(model.compact_lm_head.parameters())
                    trainable = [p for p in all_params if p.requires_grad]
                    nn.utils.clip_grad_norm_(trainable, grad_clip)

                optimizer.step()
                optimizer.zero_grad()

        avg_train = epoch_loss / max(micro_step, 1)
        train_losses.append(avg_train)

        # --- Validate ---
        val_loss = evaluate_val(model, val_loader, device)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1

        elapsed = time.time() - t0
        log(f"  Epoch {epoch+1}/{epochs}: "
            f"train_ce={avg_train:.4f}, val_ce={val_loss:.4f}, "
            f"best_val={best_val_loss:.4f} (ep{best_val_epoch}), "
            f"time={elapsed:.0f}s")

    total_time = time.time() - t0

    result = LRSearchResult(
        lr=lr,
        epochs=epochs,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        best_val_loss=best_val_loss,
        best_val_epoch=best_val_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        total_time_s=total_time,
    )

    # Cleanup to free GPU memory
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(
    data_dir: Path,
    config: ConstrainedTextToRGBConfig,
    seed: int = 42,
    val_fraction: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and format training data, split into train/val.

    We need the tokenizer to format examples, so we load a temporary model
    just to get the tokenizer + compact vocab, then discard it.
    """
    from transformers import AutoTokenizer
    from pretrain_text_to_rgb.src.model import build_compact_vocab

    log("[INFO] Loading tokenizer for data formatting...")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compact_vocab = build_compact_vocab(tokenizer)

    log(f"[INFO] Loading data from {data_dir}")
    dataset = TextThinFilmDataset(
        data_dir, seed=seed, split='train',
        verbose=True, limit_examples=config.limit_examples,
    )

    input_ids_list = []
    labels_list = []
    skipped_text = 0
    skipped_format = 0

    log("[INFO] Formatting examples...")
    t0 = time.time()

    for i, example in enumerate(dataset):
        if example.text is None or not isinstance(example.text, str):
            skipped_text += 1
            continue
        if not example.text.strip():
            skipped_text += 1
            continue

        rgb_int = normalized_to_rgb(example.rgb)
        formatted = format_training_example(
            user_text=example.text,
            rgb=rgb_int,
            tokenizer=tokenizer,
            compact_vocab=compact_vocab,
            max_text_len=config.max_text_len,
        )
        if formatted is None:
            skipped_format += 1
            continue

        input_ids_list.append(formatted['input_ids'])
        labels_list.append(formatted['labels'])

        n_done = len(input_ids_list) + skipped_text + skipped_format
        if n_done % 5000 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            log(f"  Processed {n_done:,} rows, formatted {len(input_ids_list):,} — "
                f"{rate:.0f} rows/s")

    elapsed = time.time() - t0
    log(f"[INFO] Formatted {len(input_ids_list):,} examples in {elapsed:.1f}s "
        f"({skipped_text} skipped text, {skipped_format} skipped format)")

    if len(input_ids_list) == 0:
        log("[ERROR] No valid examples found!")
        sys.exit(1)

    # Split into train/val
    n_total = len(input_ids_list)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    # Deterministic split
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Simple dataset class
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, ids_list, lab_list, idx):
            self.items = [(ids_list[i], lab_list[i]) for i in idx]
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            return {'input_ids': self.items[i][0], 'labels': self.items[i][1]}

    train_ds = IndexedDataset(input_ids_list, labels_list, train_indices)
    val_ds = IndexedDataset(input_ids_list, labels_list, val_indices)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
    )

    log(f"[INFO] Train: {n_train:,}, Val: {n_val:,}")
    return train_loader, val_loader


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Learning rate search for constrained LLM fine-tuning')

    # Data
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.1)

    # Model
    parser.add_argument('--encoder', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-text-len', type=int, default=756)

    # Fine-tuning mode
    parser.add_argument('--finetune-mode', type=str, default='lora',
                        choices=['lora', 'last_n', 'full'])
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', type=str, default='q_proj,v_proj')
    parser.add_argument('--unfreeze-layers', type=int, default=4)

    # LR search range
    parser.add_argument('--lr-min', type=float, default=5e-5)
    parser.add_argument('--lr-max', type=float, default=5e-3)
    parser.add_argument('--n-lrs', type=int, default=8,
                        help='Number of LR candidates (log-spaced)')

    # Training
    parser.add_argument('--epochs', type=int, default=1,
                        help='Epochs per trial (default: 1, single pass)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Micro batch size per forward pass')
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-fraction', type=float, default=0.03)

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for results JSON')

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[INFO] Device: {device}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    # Generate LR candidates (log-spaced)
    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.n_lrs)
    log(f"[INFO] LR candidates ({args.n_lrs}): "
        f"{', '.join(f'{lr:.2e}' for lr in lrs)}")

    # Build config (LR will be overridden per-trial)
    config = ConstrainedTextToRGBConfig(
        encoder_name=args.encoder,
        max_text_len=args.max_text_len,
        finetune_mode=args.finetune_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        unfreeze_layers=args.unfreeze_layers,
        learning_rate=lrs[0],  # placeholder
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit_examples=args.limit_examples,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
    )

    # Prepare data (done once, reused for all LR trials)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'

    train_loader, val_loader = prepare_data(
        data_dir, config, seed=args.seed, val_fraction=args.val_fraction,
    )

    # ================================================================
    # Run LR sweep
    # ================================================================

    log(f"\n{'='*60}")
    log(f"LEARNING RATE SEARCH")
    log(f"{'='*60}")
    log(f"  Fine-tune mode: {args.finetune_mode}")
    log(f"  Epochs/trial:   {args.epochs}")
    log(f"  Batch size:     {args.batch_size} micro x {args.grad_accum_steps} accum = "
        f"{args.batch_size * args.grad_accum_steps} effective")
    log(f"  N candidates:   {args.n_lrs}")
    log(f"  LR range:       [{args.lr_min:.1e}, {args.lr_max:.1e}]")
    log(f"  Train batches:  {len(train_loader)}")
    log(f"  Val batches:    {len(val_loader)}")
    log(f"{'='*60}")

    results: List[LRSearchResult] = []
    total_t0 = time.time()

    for i, lr in enumerate(lrs):
        log(f"\n[Trial {i+1}/{len(lrs)}] LR = {lr:.2e}")

        result = run_single_lr(
            lr=lr,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
            warmup_fraction=args.warmup_fraction,
            grad_accum_steps=args.grad_accum_steps,
        )
        results.append(result)

        log(f"  => best_val_ce={result.best_val_loss:.4f} (epoch {result.best_val_epoch}), "
            f"time={result.total_time_s:.0f}s")

    total_time = time.time() - total_t0

    # ================================================================
    # Find optimal LR
    # ================================================================

    best_idx = min(range(len(results)), key=lambda i: results[i].best_val_loss)
    best = results[best_idx]

    log(f"\n{'='*60}")
    log(f"LEARNING RATE SEARCH RESULTS")
    log(f"{'='*60}")
    log(f"{'LR':>12} | {'Best Val CE':>12} | {'Final Val CE':>12} | "
        f"{'Final Train CE':>14} | {'Best Epoch':>10}")
    log(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}-+-{'-'*10}")

    for r in results:
        marker = " <-- BEST" if r.lr == best.lr else ""
        log(f"{r.lr:>12.2e} | {r.best_val_loss:>12.4f} | {r.final_val_loss:>12.4f} | "
            f"{r.final_train_loss:>14.4f} | {r.best_val_epoch:>10d}{marker}")

    log(f"\n  Optimal LR:       {best.lr:.6e}")
    log(f"  Best val CE loss: {best.best_val_loss:.4f}")
    log(f"  Total time:       {total_time/60:.1f} min")
    log(f"{'='*60}")

    # ================================================================
    # Save results
    # ================================================================

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / 'pretrain_text_to_rgb' / 'outputs' / 'lr_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = args.finetune_mode
    if args.finetune_mode == 'lora':
        mode_tag += f"_r{args.lora_rank}"
    elif args.finetune_mode == 'last_n':
        mode_tag += f"_un{args.unfreeze_layers}"

    lim_tag = f"_lim{args.limit_examples}" if args.limit_examples else "_full"
    output_file = output_dir / f'lr_search_{mode_tag}_ep{args.epochs}_bs{args.batch_size}{lim_tag}.json'

    save_data = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'finetune_mode': args.finetune_mode,
        'lora_rank': args.lora_rank if args.finetune_mode == 'lora' else None,
        'lora_alpha': args.lora_alpha if args.finetune_mode == 'lora' else None,
        'unfreeze_layers': args.unfreeze_layers if args.finetune_mode == 'last_n' else None,
        'n_train_examples': len(train_loader.dataset),
        'n_val_examples': len(val_loader.dataset),
        'optimal_lr': best.lr,
        'best_val_loss': best.best_val_loss,
        'total_time_s': total_time,
        'results': [asdict(r) for r in results],
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"[INFO] Results saved to {output_file}")


if __name__ == '__main__':
    main()