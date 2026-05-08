#!/usr/bin/env python3
"""
Plot Training Curves for Constrained LLM Text-to-RGB

Loads the base model once, then for each checkpoint step:
  1. Loads LoRA adapters + compact_lm_head weights
  2. Evaluates teacher-forcing CE loss on train + val splits
  3. Records metrics

Produces a plot of CE loss vs training step.

Usage:
    python pretrain_text_to_rgb/scripts/plot_training_curves.py \
        --checkpoint-dir pretrain_text_to_rgb/data/checkpoints/<tag> \
        --start-step 50 --end-step 5000 --step-interval 50

    # Custom data/eval sizes:
    python pretrain_text_to_rgb/scripts/plot_training_curves.py \
        --checkpoint-dir pretrain_text_to_rgb/data/checkpoints/<tag> \
        --train-examples 5000 --eval-examples 2000 \
        --start-step 100 --end-step 10000 --step-interval 100
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / 'pretrain_text_to_rgb'))

from src.dataset import TextThinFilmDataset, find_repo_root
from pretrain_text_to_rgb.src.model import (
    ConstrainedTextToRGBConfig,
    ConstrainedTextToRGBModel,
    build_compact_vocab,
    format_training_example,
    collate_fn,
    normalized_to_rgb,
)


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# Data Preparation (done once)
# ============================================================================

def prepare_eval_data(
    data_dir: Path,
    config: ConstrainedTextToRGBConfig,
    train_limit: int,
    eval_limit: int,
    seed: int = 42,
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Format train + val examples into DataLoaders for teacher-forcing eval.

    Loads tokenizer to format examples, then discards it (the model will
    have its own tokenizer).
    """
    from transformers import AutoTokenizer

    log("[INFO] Loading tokenizer for data formatting...")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compact_vocab = build_compact_vocab(tokenizer)
    pad_id = tokenizer.pad_token_id

    def format_split(split: str, limit: int):
        log(f"[INFO] Formatting {split} split (limit={limit})...")
        dataset = TextThinFilmDataset(
            data_dir, seed=seed, split=split,
            verbose=True, limit_examples=limit,
        )
        items = []
        skipped = 0
        for ex in dataset:
            if ex.text is None or not isinstance(ex.text, str) or not ex.text.strip():
                skipped += 1
                continue
            rgb_int = normalized_to_rgb(ex.rgb)
            fmt = format_training_example(
                user_text=ex.text, rgb=rgb_int,
                tokenizer=tokenizer, compact_vocab=compact_vocab,
                max_text_len=config.max_text_len,
            )
            if fmt is None:
                skipped += 1
                continue
            items.append(fmt)

        log(f"  {split}: {len(items)} formatted, {skipped} skipped")
        return items

    train_items = format_split('train', train_limit)
    val_items = format_split('validation', eval_limit)

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    train_loader = DataLoader(
        ListDataset(train_items), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
    )
    val_loader = DataLoader(
        ListDataset(val_items), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
    )

    return train_loader, val_loader


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_ce_loss(
    model: ConstrainedTextToRGBModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average CE loss via teacher forcing."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Checkpoint Loading (reuse base model, swap adapters)
# ============================================================================

def load_checkpoint_onto_base(
    base_model: ConstrainedTextToRGBModel,
    checkpoint_dir: Path,
    device: torch.device,
    original_lm_head_state: dict,
) -> ConstrainedTextToRGBModel:
    """
    Load a checkpoint's LoRA adapters + compact_lm_head onto an existing
    base model. This avoids reloading the 1.1B base weights each time.

    For LoRA mode: wraps model in PeftModel, loads adapter weights.
    For other modes: loads trainable_params.pt into the model state.

    Returns the model (which may be wrapped in PeftModel for LoRA).
    """
    # Reset compact_lm_head to checkpoint's weights
    compact_lm_head_path = checkpoint_dir / 'compact_lm_head.pt'
    if compact_lm_head_path.exists():
        base_model.compact_lm_head.load_state_dict(
            torch.load(compact_lm_head_path, map_location=device, weights_only=True)
        )

    # Load LoRA adapters or trainable params
    lora_path = checkpoint_dir / 'lora_adapters'
    trainable_path = checkpoint_dir / 'trainable_params.pt'

    if lora_path.exists():
        from peft import PeftModel
        # Wrap base model with LoRA adapters from this checkpoint
        base_model.model = PeftModel.from_pretrained(
            base_model.model, str(lora_path)
        )
        base_model.model.eval()
    elif trainable_path.exists():
        trainable_state = torch.load(trainable_path, map_location=device,
                                     weights_only=True)
        model_state = base_model.model.state_dict()
        model_state.update(trainable_state)
        base_model.model.load_state_dict(model_state)

    return base_model


def unload_adapters(
    model: ConstrainedTextToRGBModel,
    original_lm_head_state: dict,
    device: torch.device,
):
    """
    Remove LoRA adapters from model, restoring it to the base state
    so we can load the next checkpoint's adapters.
    """
    if hasattr(model.model, 'peft_config'):
        # Unwrap PeftModel -> get back the base model
        model.model = model.model.base_model.model
        model.model.eval()

    # Reset compact_lm_head to original surgery state
    model.compact_lm_head.load_state_dict(original_lm_head_state)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot constrained LLM training curves across checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Base checkpoint directory containing step_N/ subdirs')

    # Step range
    parser.add_argument('--start-step', type=int, default=50)
    parser.add_argument('--end-step', type=int, default=5000)
    parser.add_argument('--step-interval', type=int, default=50)

    # Data
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--train-examples', type=int, default=5000)
    parser.add_argument('--eval-examples', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Eval batch size (kept small for memory)')
    parser.add_argument('--seed', type=int, default=42)

    # Model (for zero-shot baseline comparison)
    parser.add_argument('--encoder', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-text-len', type=int, default=756)

    # Output
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--smoothing-window', type=int, default=5)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[INFO] Device: {device}")

    checkpoint_base = Path(args.checkpoint_dir)
    if not checkpoint_base.exists():
        log(f"[ERROR] Checkpoint directory not found: {checkpoint_base}")
        sys.exit(1)

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    # Load config from first available checkpoint (or build default)
    config = None
    for subdir in sorted(checkpoint_base.iterdir()):
        cfg_path = subdir / 'config.json'
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = ConstrainedTextToRGBConfig.from_dict(json.load(f))
            log(f"[INFO] Loaded config from {cfg_path}")
            break

    if config is None:
        config = ConstrainedTextToRGBConfig(
            encoder_name=args.encoder,
            max_text_len=args.max_text_len,
        )
        log("[INFO] Using default config (no checkpoint config found)")

    # Override encoder if specified
    config.encoder_name = args.encoder
    config.max_text_len = args.max_text_len

    # Prepare eval data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'

    train_loader, val_loader = prepare_eval_data(
        data_dir, config,
        train_limit=args.train_examples,
        eval_limit=args.eval_examples,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    log(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ================================================================
    # Load base model once (surgery only, no fine-tuning config)
    # ================================================================

    log("[INFO] Loading base model (one-time)...")
    t0 = time.time()
    base_model = ConstrainedTextToRGBModel(config)
    base_model.load_model(device=device, inference_only=True)
    log(f"[INFO] Base model loaded in {time.time() - t0:.1f}s")

    # Save original compact_lm_head state (post-surgery, pre-training)
    original_lm_head_state = {
        k: v.clone() for k, v in base_model.compact_lm_head.state_dict().items()
    }

    # ================================================================
    # Optional: evaluate zero-shot baseline
    # ================================================================

    log("\n[INFO] Evaluating zero-shot baseline...")
    zs_train = evaluate_ce_loss(base_model, train_loader, device)
    zs_val = evaluate_ce_loss(base_model, val_loader, device)
    log(f"  Zero-shot: train_ce={zs_train:.4f}, val_ce={zs_val:.4f}")

    # ================================================================
    # Discover available checkpoints
    # ================================================================

    steps_requested = list(range(args.start_step,
                                 args.end_step + 1,
                                 args.step_interval))

    # Also check what step_N directories actually exist
    available_steps = set()
    for p in checkpoint_base.iterdir():
        if p.is_dir() and p.name.startswith('step_'):
            try:
                available_steps.add(int(p.name.split('_')[1]))
            except (ValueError, IndexError):
                pass

    steps_to_eval = sorted(s for s in steps_requested if s in available_steps)

    # Also include any steps outside the requested range that exist
    extra_steps = sorted(available_steps - set(steps_requested))
    if extra_steps:
        log(f"[INFO] Found {len(extra_steps)} additional checkpoints outside "
            f"requested range: {extra_steps[:5]}{'...' if len(extra_steps) > 5 else ''}")

    if not steps_to_eval:
        log(f"[WARN] No checkpoints found in requested range "
            f"[{args.start_step}, {args.end_step}]")
        log(f"[INFO] Available steps: {sorted(available_steps)[:20]}")
        if available_steps:
            steps_to_eval = sorted(available_steps)
            log(f"[INFO] Evaluating all {len(steps_to_eval)} available checkpoints")
        else:
            log("[ERROR] No checkpoints found at all!")
            sys.exit(1)

    log(f"[INFO] Will evaluate {len(steps_to_eval)} checkpoints: "
        f"{steps_to_eval[0]} to {steps_to_eval[-1]}")

    # ================================================================
    # Evaluate each checkpoint
    # ================================================================

    results = {
        'steps': [],
        'train_ce': [],
        'val_ce': [],
        'zero_shot_train_ce': zs_train,
        'zero_shot_val_ce': zs_val,
    }

    for i, step in enumerate(steps_to_eval):
        checkpoint_path = checkpoint_base / f'step_{step}'
        t0 = time.time()

        try:
            load_checkpoint_onto_base(base_model, checkpoint_path, device,
                                      original_lm_head_state)

            train_ce = evaluate_ce_loss(base_model, train_loader, device)
            val_ce = evaluate_ce_loss(base_model, val_loader, device)

            elapsed = time.time() - t0
            log(f"  [{i+1}/{len(steps_to_eval)}] Step {step}: "
                f"train_ce={train_ce:.4f}, val_ce={val_ce:.4f} "
                f"({elapsed:.1f}s)")

            results['steps'].append(step)
            results['train_ce'].append(train_ce)
            results['val_ce'].append(val_ce)

        except Exception as e:
            log(f"  [{i+1}/{len(steps_to_eval)}] Step {step}: ERROR - {e}")

        finally:
            unload_adapters(base_model, original_lm_head_state, device)
            torch.cuda.empty_cache()
            gc.collect()

    # Also evaluate 'best' and 'latest' if they exist
    for special in ['best', 'latest']:
        sp = checkpoint_base / special
        if sp.exists():
            try:
                load_checkpoint_onto_base(base_model, sp, device,
                                          original_lm_head_state)
                val_ce = evaluate_ce_loss(base_model, val_loader, device)
                log(f"\n  {special}: val_ce={val_ce:.4f}")
                results[f'{special}_val_ce'] = val_ce
            except Exception as e:
                log(f"  {special}: ERROR - {e}")
            finally:
                unload_adapters(base_model, original_lm_head_state, device)
                torch.cuda.empty_cache()

    # ================================================================
    # Save results
    # ================================================================

    if args.output:
        output_path = Path(args.output)
    else:
        tag = checkpoint_base.name
        output_path = (repo_root / 'pretrain_text_to_rgb' / 'outputs' /
                       f'training_curves_{tag}.json')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n[INFO] Results saved to {output_path}")

    # Summary table
    log("\n" + "=" * 65)
    log("TRAINING CURVES SUMMARY")
    log("=" * 65)
    log(f"  Zero-shot baseline: train_ce={zs_train:.4f}, val_ce={zs_val:.4f}")
    log(f"{'Step':>10} {'Train CE':>12} {'Val CE':>12}")
    log("-" * 38)
    for i in range(len(results['steps'])):
        log(f"{results['steps'][i]:>10} {results['train_ce'][i]:>12.4f} "
            f"{results['val_ce'][i]:>12.4f}")
    log("=" * 65)

    # ================================================================
    # Plot
    # ================================================================

    if args.plot and len(results['steps']) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            steps = np.array(results['steps'])
            train_ce = np.array(results['train_ce'])
            val_ce = np.array(results['val_ce'])

            fig, ax = plt.subplots(figsize=(10, 6))

            # Raw curves
            ax.plot(steps, train_ce, 'b-', alpha=0.3, lw=1, label='Train CE (raw)')
            ax.plot(steps, val_ce, 'r-', alpha=0.3, lw=1, label='Val CE (raw)')

            # Smoothed
            w = args.smoothing_window
            if len(steps) > w:
                kernel = np.ones(w) / w
                train_smooth = np.convolve(train_ce, kernel, mode='valid')
                val_smooth = np.convolve(val_ce, kernel, mode='valid')
                min_len = min(len(train_smooth), len(val_smooth))
                steps_smooth = steps[w // 2: w // 2 + min_len]
                ax.plot(steps_smooth, train_smooth[:min_len], 'b-', lw=2.5,
                        label=f'Train CE (smoothed, w={w})')
                ax.plot(steps_smooth, val_smooth[:min_len], 'r-', lw=2.5,
                        label=f'Val CE (smoothed, w={w})')

            # Zero-shot baseline
            ax.axhline(y=zs_val, color='gray', ls='--', lw=1.5, alpha=0.7,
                        label=f'Zero-shot val CE ({zs_val:.3f})')

            # Best checkpoint marker
            best_idx = np.nanargmin(val_ce)
            ax.annotate(
                f'Best: {val_ce[best_idx]:.4f} (step {steps[best_idx]})',
                xy=(steps[best_idx], val_ce[best_idx]),
                xytext=(0.6, 0.85), textcoords='axes fraction',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='red', alpha=0.8),
            )

            ax.set_xlabel('Training Step', fontsize=13)
            ax.set_ylabel('Cross-Entropy Loss', fontsize=13)
            ax.set_title(f'Constrained LLM Training Curves\n{checkpoint_base.name}',
                         fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            log(f"[INFO] Plot saved to {plot_path}")
            plt.close()

        except ImportError:
            log("[WARN] matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()