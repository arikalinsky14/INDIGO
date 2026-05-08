#!/usr/bin/env python3
"""
Training Script for CHROMA-Lite - Constrained LLM Fine-tuning for Text-to-RGB

Fine-tunes TinyLlama with a surgically reduced lm_head so the model can
ONLY output valid [R,G,B] tokens. The full input embedding is preserved
so arbitrary text prompts can be processed.

Training approach:
1. Load dataset (same TextThinFilmDataset as MLP head approach)
2. Format each example: system prompt + user text -> [R,G,B] target
3. Tokenize and create labels (loss only on assistant response tokens)
4. Fine-tune with LoRA / last-N layers / full (configurable)

Usage:
    # LoRA fine-tuning (recommended starting point):
    python pretrain_text_to_rgb/scripts/training.py \\
        --data-dir create_dataset/data_prompts \\
        --finetune-mode lora --lora-rank 16 \\
        --epochs 3 --lr 2e-4

    # Last-4 layers fine-tuning:
    python pretrain_text_to_rgb/scripts/training.py \\
        --finetune-mode last_n --unfreeze-layers 4 \\
        --epochs 5 --lr 5e-5

    # Quick test:
    python pretrain_text_to_rgb/scripts/training.py \\
        --limit-examples 500 --epochs 2 --batch-size 4
"""

import sys
import json
import argparse
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
    collate_fn,
    normalized_to_rgb,
)


def log(msg: str):
    """Print with immediate flush for SLURM log visibility."""
    print(msg)
    sys.stdout.flush()


def get_lr_schedule(step: int, total_steps: int, base_lr: float,
                    warmup_fraction: float = 0.03) -> float:
    """Linear warmup + cosine decay."""
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
# Dataset Wrapper
# ============================================================================

class TextToRGBFineTuneDataset(Dataset):
    """
    Pre-formatted dataset of (input_ids, labels) for constrained LLM training.

    Takes raw text + RGB pairs and converts them into tokenized chat sequences
    with compact vocabulary labels.
    """
    def __init__(
        self,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ):
        assert len(input_ids_list) == len(labels_list)
        self.input_ids_list = input_ids_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.labels_list[idx],
        }


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: ConstrainedTextToRGBModel,
    config: ConstrainedTextToRGBConfig,
    optimizer,
    step: int,
    loss: float,
    save_dir: Path,
    lr: float = None,
):
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save compact lm_head
    torch.save(model.compact_lm_head.state_dict(), save_dir / 'compact_lm_head.pt')

    # Save compact vocab mapping
    torch.save(model.compact_vocab, save_dir / 'compact_vocab.pt')

    # Save model weights (LoRA adapters or full model depending on mode)
    if config.finetune_mode == 'lora':
        # Save only LoRA adapters (small)
        model.model.save_pretrained(save_dir / 'lora_adapters')
    else:
        # Save the full transformer state (large)
        # Only save trainable parameters to save space
        trainable_state = {
            name: param.data
            for name, param in model.model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state, save_dir / 'trainable_params.pt')

    # Save optimizer
    torch.save(optimizer.state_dict(), save_dir / 'optimizer.pt')

    # Save config and metadata
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    meta = {'step': step, 'loss': loss, 'tag': config.tag()}
    if lr is not None:
        meta['lr'] = lr
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    log(f"[Checkpoint] Saved to {save_dir} at step {step}")


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_dataset(
    data_dir: Path,
    tokenizer,
    compact_vocab: Dict[str, torch.Tensor],
    config: ConstrainedTextToRGBConfig,
    split: str = 'train',
    seed: int = 42,
) -> TextToRGBFineTuneDataset:
    """
    Load dataset and format all examples for training.

    Follows the same data loading pattern as the MLP head training script:
    - Uses TextThinFilmDataset
    - Filters out None/non-string text entries
    - Converts normalized RGB to integer (0-255) for formatting
    """
    log(f"[INFO] Loading {split} data from {data_dir}")
    dataset = TextThinFilmDataset(
        data_dir, seed=seed, split=split,
        verbose=True, limit_examples=config.limit_examples,
    )

    input_ids_list = []
    labels_list = []
    skipped_text = 0
    skipped_format = 0

    log("[INFO] Formatting examples...")
    t0 = time.time()

    for i, example in enumerate(dataset):
        # Guard: skip None or non-string text (same as MLP head script)
        if example.text is None or not isinstance(example.text, str):
            skipped_text += 1
            continue

        # Guard: skip empty text
        if not example.text.strip():
            skipped_text += 1
            continue

        # Convert normalized RGB [0,1] tensor to integer (0-255) tuple
        rgb_int = normalized_to_rgb(example.rgb)

        # Format into tokenized chat sequence with compact labels
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
            log(f"  Processed {n_done:,} rows, formatted {len(input_ids_list):,} "
                f"({skipped_text} bad text, {skipped_format} bad format) — "
                f"{rate:.0f} rows/s")

    elapsed = time.time() - t0
    log(f"[INFO] Formatted {len(input_ids_list):,} examples in {elapsed:.1f}s "
        f"({skipped_text} skipped text, {skipped_format} skipped format)")

    if len(input_ids_list) == 0:
        log("[ERROR] No valid examples found! Check that data has 'text' column.")
        sys.exit(1)

    # Report sequence length statistics
    lengths = [ids.size(0) for ids in input_ids_list]
    log(f"[INFO] Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Report label token counts (how many tokens of loss per example)
    n_label_tokens = [(labels != -100).sum().item() for labels in labels_list]
    log(f"[INFO] Label tokens/example: min={min(n_label_tokens)}, "
        f"max={max(n_label_tokens)}, mean={np.mean(n_label_tokens):.1f}")

    return TextToRGBFineTuneDataset(input_ids_list, labels_list)


# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune constrained LLM for text-to-RGB')

    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data_prompts/ directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None)

    # Encoder
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

    # Training
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Micro batch size per forward pass')
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-fraction', type=float, default=0.03)
    parser.add_argument('--num-workers', type=int, default=4)

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')

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

    # Build config
    config = ConstrainedTextToRGBConfig(
        encoder_name=args.encoder,
        max_text_len=args.max_text_len,
        finetune_mode=args.finetune_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        unfreeze_layers=args.unfreeze_layers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit_examples=args.limit_examples,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
    )

    # ================================================================
    # Load model + perform surgery
    # ================================================================

    model = ConstrainedTextToRGBModel(config)
    model.load_model(device=device)

    # ================================================================
    # Prepare dataset
    # ================================================================

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'

    train_dataset = prepare_dataset(
        data_dir=data_dir,
        tokenizer=model.tokenizer,
        compact_vocab=model.compact_vocab,
        config=config,
        split='train',
        seed=args.seed,
    )

    pad_id = model.tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
    )

    # ================================================================
    # Optimizer
    # ================================================================

    # Collect trainable parameters: model backbone + compact lm_head
    trainable_params = [
        {'params': [p for p in model.model.parameters() if p.requires_grad],
         'lr': args.lr},
        {'params': model.compact_lm_head.parameters(),
         'lr': args.lr},
    ]

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # ================================================================
    # Checkpoint directory
    # ================================================================

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = (repo_root / 'pretrain_text_to_rgb' / 'data' /
                    'checkpoints' / config.tag())
    log(f"[INFO] Checkpoints: {save_dir}")

    # ================================================================
    # Training loop
    # ================================================================

    n_examples = len(train_dataset)
    micro_batches_per_epoch = math.ceil(n_examples / args.batch_size)
    accum = args.grad_accum_steps
    effective_batch = args.batch_size * accum
    optim_steps_per_epoch = math.ceil(micro_batches_per_epoch / accum)
    total_steps = optim_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    log(f"\n[INFO] Training configuration:")
    log(f"  Examples:        {n_examples:,}")
    log(f"  Micro batch:     {args.batch_size}")
    log(f"  Grad accum:      {accum} steps")
    log(f"  Effective batch: {effective_batch}")
    log(f"  Optim steps/ep:  {optim_steps_per_epoch}")
    log(f"  Total optim steps: {total_steps:,}")
    log(f"  Warmup steps:    {warmup_steps}")
    log(f"  Fine-tune mode:  {config.finetune_mode}")
    log(f"  Config tag:      {config.tag()}")

    global_step = 0  # counts optimizer steps
    best_loss = float('inf')

    log("\n[INFO] Starting training...")
    train_t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_optim_steps = 0
        micro_step = 0
        epoch_t0 = time.time()

        optimizer.zero_grad()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward — scale loss by accumulation steps
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / accum
            loss.backward()

            micro_step += 1
            epoch_loss += outputs['loss'].item()  # unscaled for logging

            # Optimizer step every accum micro-batches (or at end of epoch)
            if micro_step % accum == 0 or micro_step == micro_batches_per_epoch:
                # LR schedule
                current_lr = get_lr_schedule(global_step, total_steps, args.lr,
                                             args.warmup_fraction)
                set_lr(optimizer, current_lr)

                if args.grad_clip > 0:
                    all_params = list(model.model.parameters()) + \
                                 list(model.compact_lm_head.parameters())
                    trainable = [p for p in all_params if p.requires_grad]
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                n_optim_steps += 1

                # Logging
                if global_step % 100 == 0:
                    phase = "warmup" if global_step <= warmup_steps else "decay"
                    elapsed = time.time() - train_t0
                    rate = global_step / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) / rate if rate > 0 else 0
                    avg_recent = epoch_loss / micro_step
                    log(f"  Step {global_step}/{total_steps}: "
                        f"ce_loss={avg_recent:.4f}, lr={current_lr:.2e} [{phase}] "
                        f"({rate:.1f} steps/s, ETA {eta/60:.1f}min)")

                # Periodic checkpoint
                if global_step % args.save_every == 0:
                    save_checkpoint(model, config, optimizer, global_step,
                                    epoch_loss / micro_step,
                                    save_dir / f'step_{global_step}',
                                    lr=current_lr)

        avg_loss = epoch_loss / max(micro_step, 1)
        epoch_time = time.time() - epoch_t0
        final_lr = get_lr_schedule(global_step - 1, total_steps, args.lr,
                                   args.warmup_fraction)

        log(f"[Epoch {epoch+1}/{args.epochs}] ce_loss={avg_loss:.4f}, "
            f"lr={final_lr:.2e}, time={epoch_time:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, config, optimizer, global_step, avg_loss,
                            save_dir / 'best', lr=final_lr)

        # Save latest
        save_checkpoint(model, config, optimizer, global_step, avg_loss,
                        save_dir / 'latest', lr=final_lr)

    total_time = time.time() - train_t0
    log(f"\n[INFO] Training complete! Total time: {total_time/60:.1f} min")
    log(f"[INFO] Best CE loss: {best_loss:.4f}")
    log(f"[INFO] Final checkpoint: {save_dir / 'latest'}")


if __name__ == "__main__":
    main()