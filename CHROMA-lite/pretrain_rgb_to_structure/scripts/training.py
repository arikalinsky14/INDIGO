#!/usr/bin/env python3
"""
Training Script for CHROMA-Lite - MLP Version

Simplified feedforward MLP architecture replacing the transformer.

Training approach (multi-pass autoregressive):
- For an N-layer structure, we create N+1 training samples:
  - Step 0: RGB + empty structure -> predict layer 0 token
  - Step 1: RGB + layer 0 -> predict layer 1 token
  - ...
  - Step N: RGB + layers 0..N-1 -> predict EOS token

Features:
- Learning rate schedule: 2% linear warmup + cosine decay
- Checkpoints saved to: chroma-lite/data/checkpoints/<hparams_tag>/

Usage:
    python scripts/training.py --data-dir /path/to/data_prompts --epochs 10 --verbose
    
Key hyperparameters:
    --d-model: Hidden layer dimension (default: 256)
    --n-layers: Number of hidden layers (default: 4)
    --dropout: Dropout rate (default: 0.1)
"""

import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import NUM_MATERIALS, MAX_LAYERS, EOS_TOKEN, build_structure_matrix, encode_layer
from src.dataset import ThinFilmDataset, TrainingExample, find_repo_root
from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig, compute_loss


def collate_fn(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """
    Create training samples for autoregressive prediction.
    
    For each example with N layers (N < 8): creates N+1 samples
        - Steps 0 to N-1: predict layer tokens
        - Step N: predict EOS
    
    For each example with N=8 layers: creates 8 samples
        - Steps 0 to 7: predict layer tokens
        - No EOS step (max length reached, generation stops automatically)
    
    Returns:
        dict with:
        - 'rgb': [total_samples, 3]
        - 'structure_matrix': [total_samples, 25, 8] 
        - 'target_token': [total_samples]
    """
    all_rgb = []
    all_matrices = []
    all_targets = []
    
    for ex in examples:
        n_layers = len(ex.target_materials)
        
        # Determine how many steps to create:
        # - For N < 8 layers: N+1 steps (N layer predictions + 1 EOS prediction)
        # - For N = 8 layers: 8 steps (8 layer predictions, no EOS needed)
        max_step = n_layers if n_layers < MAX_LAYERS else MAX_LAYERS - 1
        
        for step in range(max_step + 1):
            all_rgb.append(ex.rgb)
            
            # Build structure matrix with layers 0..step-1 filled
            if step == 0:
                # Empty structure for predicting first layer
                all_matrices.append(torch.zeros(NUM_MATERIALS, MAX_LAYERS))
            else:
                # Structure with first 'step' layers filled
                all_matrices.append(build_structure_matrix(
                    ex.target_materials[:step], 
                    ex.target_thicknesses[:step]
                ))
            
            # Target: layer token for steps 0..N-1, EOS for step N (if N < 8)
            if step < n_layers:
                all_targets.append(encode_layer(
                    ex.target_materials[step], 
                    ex.target_thicknesses[step]
                ))
            else:
                all_targets.append(EOS_TOKEN)
    
    return {
        'rgb': torch.stack(all_rgb),
        'structure_matrix': torch.stack(all_matrices),
        'target_token': torch.tensor(all_targets, dtype=torch.long),
    }


def get_lr_schedule(step: int, total_steps: int, base_lr: float, warmup_fraction: float = 0.02) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.
    
    Args:
        step: Current training step (0-indexed)
        total_steps: Total number of training steps
        base_lr: Base learning rate (peak LR after warmup)
        warmup_fraction: Fraction of steps for warmup (default: 2%)
    
    Returns:
        Learning rate for current step
    """
    warmup_steps = int(total_steps * warmup_fraction)
    
    if step < warmup_steps:
        # Linear warmup: scale from 0 to base_lr
        return base_lr * (step + 1) / max(warmup_steps, 1)
    else:
        # Cosine decay: from base_lr to 0
        decay_steps = total_steps - warmup_steps
        decay_progress = (step - warmup_steps) / max(decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * decay_progress))


def set_lr(optimizer, lr: float):
    """Update learning rate for all parameter groups."""
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def save_checkpoint(model, config, optimizer, step, loss, save_dir: Path, lr: float = None):
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'model.pt')
    torch.save(optimizer.state_dict(), save_dir / 'optimizer.pt')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    meta = {'step': step, 'loss': loss, 'tag': config.tag()}
    if lr is not None:
        meta['lr'] = lr
    
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[Checkpoint] Saved to {save_dir} at step {step}")


def train_step(model, batch, device) -> Dict[str, torch.Tensor]:
    """
    Perform a single training step.
    
    With the MLP, all samples can be processed together in a single forward pass.
    """
    rgb = batch['rgb'].to(device)
    structure_matrix = batch['structure_matrix'].to(device)
    target_token = batch['target_token'].to(device)
    
    # Simple forward pass - MLP processes all samples at once
    losses = compute_loss(model, rgb, structure_matrix, target_token)
    
    return losses


def parse_args():
    parser = argparse.ArgumentParser(description='Train CHROMA-Lite MLP model')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data_prompts/ directory')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None,
                        help='Limit to first N examples (for testing/debugging)')
    
    # Model hyperparameters (simplified for MLP)
    parser.add_argument('--d-model', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-fraction', type=float, default=0.02,
                        help='Fraction of total steps for LR warmup (default: 0.02 = 2%)')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override checkpoint dir (default: data/checkpoints/<hparams_tag>/)')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # Find repo root for default paths
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = Path(__file__).parent.parent
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'
    print(f"[INFO] Loading data from {data_dir}")
    
    # Load dataset
    dataset = ThinFilmDataset(
        data_dir, 
        seed=args.seed, 
        split=args.split, 
        verbose=args.verbose,
        limit_examples=args.limit_examples
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Create model config (simplified for MLP)
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        limit_examples=args.limit_examples,
        epochs=args.epochs
    )
    
    # Create model
    model = ThinFilmMLP(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: MLP with {args.n_layers} hidden layers of dim {args.d_model}")
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Config tag: {config.tag()}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Checkpoint directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = repo_root / 'pretrain_rgb_to_structure' / 'data' / 'checkpoints' / config.tag()
    print(f"[INFO] Checkpoints will be saved to: {save_dir}")
    
    # Calculate total steps for LR schedule
    n_examples = len(dataset)
    steps_per_epoch = math.ceil(n_examples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)
    
    print(f"[INFO] Dataset size: {n_examples:,} examples")
    print(f"[INFO] Steps per epoch: {steps_per_epoch:,}")
    print(f"[INFO] Total steps: {total_steps:,}")
    print(f"[INFO] Warmup steps: {warmup_steps:,} ({args.warmup_fraction:.1%} of total)")
    print(f"[INFO] LR schedule: {args.warmup_fraction:.1%} warmup + {1-args.warmup_fraction:.1%} cosine decay")
    
    global_step = 0
    
    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        
        for batch in loader:
            # Update learning rate based on schedule
            current_lr = get_lr_schedule(global_step, total_steps, args.lr, args.warmup_fraction)
            set_lr(optimizer, current_lr)
            
            # Training step
            optimizer.zero_grad()
            losses = train_step(model, batch, device)
            losses['loss'].backward()
            
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            epoch_loss += losses['loss'].item()
            epoch_acc += losses['accuracy'].item()
            n_batches += 1
            global_step += 1
            
            # Progress logging
            if global_step % 100 == 0:
                phase = "warmup" if global_step <= warmup_steps else "decay"
                print(f"  Step {global_step}: loss={losses['loss'].item():.4f}, "
                      f"acc={losses['accuracy'].item():.3f}, lr={current_lr:.2e} [{phase}]")
            
            # Periodic checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(
                    model, config, optimizer, global_step,
                    losses['loss'].item(), save_dir / f'step_{global_step}', 
                    lr=current_lr
                )
        
        # Epoch summary
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        final_lr = get_lr_schedule(global_step - 1, total_steps, args.lr, args.warmup_fraction)
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}, acc={avg_acc:.3f}, final_lr={final_lr:.2e}")
        
        # Epoch checkpoint
        save_checkpoint(model, config, optimizer, global_step, avg_loss, save_dir / 'latest', lr=final_lr)
    
    print("[INFO] Training complete!")
    print(f"[INFO] Final checkpoint: {save_dir / 'latest'}")


if __name__ == "__main__":
    main()