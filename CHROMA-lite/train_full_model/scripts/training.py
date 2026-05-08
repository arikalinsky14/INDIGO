#!/usr/bin/env python3
"""
Training Script — Full Text → Structure Model

Trains the constraint-informed structure prediction model:
    Cached base logits + [TinyLlama + LoRA → ConstraintMLP → MixingMLP] → final logits

Trainable components:
    - Constraint LoRA adapters on TinyLlama (extracts structural constraints from text)
    - ConstraintMLP (maps LLM hidden state → constraint embedding)
    - MixingMLP (fuses base logits + constraint embedding → final logits, residual)

Frozen / pre-cached:
    - TinyLlama base weights (frozen, only LoRA adapters train)
    - TextToRGB pipeline (cached in base logits)
    - RGB→Structure MLP (cached in base logits)

Data flow per batch:
    1. Load cached base_logits [total_steps, 1002] + target_tokens [total_steps]
    2. Reconstruct per-step structure matrices from target_tokens (on-the-fly)
    3. Tokenize texts → run TinyLlama + LoRA → last hidden state [B, 2048]
    4. ConstraintMLP → constraint embedding [B, 1024] (once per example)
    5. Expand constraint embedding to match total steps (repeat_interleave)
    6. MixingMLP(base_logits, constraint_expanded, structure_flat) → final logits
    7. Cross-entropy loss vs targets (ERROR token for incorrect examples)

Gradients flow through all steps back to the constraint embedding, through
the ConstraintMLP, and into the LoRA adapters.

Usage:
    # Standard training:
    python train_full_model/scripts/training.py \\
        --cache-dir train_full_model/data/cache_train \\
        --epochs 10 --lr 7e-4

    # Quick test:
    python train_full_model/scripts/training.py \\
        --cache-dir train_full_model/data/cache_train \\
        --limit-examples 1000 --epochs 2
"""

import sys
import json
import argparse
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import (
    VOCAB_SIZE_WITH_ERROR, NUM_MATERIALS, MAX_LAYERS,
    decode_token, MATERIAL_TO_IDX, normalize_thickness,
)
from train_full_model.src.model import (
    FullModelConfig, FullModel,
    collate_constraint_inputs,
)


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# LR Schedule
# ============================================================================

def get_lr(step: int, total_steps: int, base_lr: float,
           warmup_fraction: float = 0.02) -> float:
    """Linear warmup + cosine decay."""
    warmup_steps = int(total_steps * warmup_fraction)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    decay_steps = total_steps - warmup_steps
    decay_progress = (step - warmup_steps) / max(decay_steps, 1)
    decay_progress = min(decay_progress, 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * decay_progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ============================================================================
# Cached Dataset
# ============================================================================

def build_structure_matrices_from_tokens(target_tokens: torch.Tensor,
                                         n_steps: int) -> torch.Tensor:
    """
    Reconstruct per-step structure matrices from cached target tokens.

    At step k, the structure matrix contains ground truth layers 0..k-1.
    Step 0 gets an all-zeros matrix (nothing generated yet).

    Uses vectorized token decomposition to avoid per-step decode_token() and
    normalize_thickness() calls.

    Args:
        target_tokens: [MAX_STEPS] token IDs (padded with -1 beyond n_steps)
        n_steps: number of valid autoregressive steps

    Returns:
        [n_steps, NUM_MATERIALS * MAX_LAYERS] flattened structure at each step
    """
    structure_dim = NUM_MATERIALS * MAX_LAYERS
    structures = torch.zeros(n_steps, structure_dim, dtype=torch.float32)

    tokens = target_tokens[:n_steps]

    # Precompute material/thickness for all valid tokens vectorized
    valid_mask = tokens < 1000
    if not valid_mask.any():
        return structures

    valid_tokens = tokens[valid_mask]
    # Token ID = material_idx * 40 + thickness_idx
    mat_indices = (valid_tokens // 40).long()
    thick_indices = (valid_tokens % 40).long()
    # thickness_nm = thick_idx * 5 + 5; normalized = thickness_nm / 200
    thickness_vals = (thick_indices * 5 + 5).float() / 200.0

    valid_steps = torch.where(valid_mask)[0]

    # Cumulative fill: step k sees layers 0..k-1
    # We still need a loop for the cumulative update, but the per-step
    # work is now a single index assignment (no decode_token/normalize calls)
    matrix = torch.zeros(structure_dim, dtype=torch.float32)
    vi = 0  # pointer into valid_steps
    for step in range(n_steps):
        structures[step] = matrix
        if vi < len(valid_steps) and valid_steps[vi] == step:
            flat_idx = mat_indices[vi] * MAX_LAYERS + step
            matrix[flat_idx] = thickness_vals[vi]
            vi += 1

    return structures


class CachedFullModelDataset(Dataset):
    """
    Random-access dataset over cached pretrained pipeline outputs.

    Loads sharded .pt files produced by cache_pretrained.py.
    Each shard contains:
        texts:         List[str]
        base_logits:   [N, MAX_STEPS, VOCAB_SIZE_WITH_ERROR] float16
        target_tokens: [N, MAX_STEPS] long, padded with -1
        n_steps:       [N] int8
        incorrect:     [N] bool

    Structure matrices are reconstructed on-the-fly from target_tokens
    (no re-caching needed).
    """

    def __init__(self, cache_dir: Path, limit_examples: Optional[int] = None,
                 seed: int = 42):
        cache_dir = Path(cache_dir)

        # Load index
        with open(cache_dir / 'index.json') as f:
            self.index = json.load(f)

        self.n_shards = self.index['n_shards']

        # Discover shard files
        self.shard_paths = sorted(cache_dir.glob('shard_*.pt'))
        assert len(self.shard_paths) == self.n_shards, (
            f"Expected {self.n_shards} shards, found {len(self.shard_paths)}")

        # Compute cumulative example counts for index → shard mapping
        self._shard_sizes = []
        self._cumulative = [0]
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location='cpu', weights_only=False)
            n = len(shard['texts'])
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)

        self._full_total = self._cumulative[-1]

        # Deterministic permutation for representative limit_examples subsets.
        # The cache stores correct examples first, then incorrect (following
        # FullModelDataset's stratified ordering). Without this permutation,
        # limit_examples=20k would grab only correct examples since incorrect
        # don't start until position ~2.98M. The permutation ensures any
        # subset has the correct ~5.7% incorrect ratio.
        # When limit_examples is None, this is an identity-like shuffle that
        # the DataLoader's own shuffle=True will further randomize.
        self._index_map = None
        if limit_examples is not None and limit_examples < self._full_total:
            g = torch.Generator()
            g.manual_seed(seed)
            self._index_map = torch.randperm(self._full_total, generator=g)

        self.total = self._full_total
        if limit_examples is not None and limit_examples < self.total:
            self.total = limit_examples

        log(f"[CachedDataset] {self.total:,} examples across {self.n_shards} shards"
            f" (of {self._full_total:,} total)")

    @lru_cache(maxsize=4)
    def _load_shard(self, shard_idx: int):
        return torch.load(self.shard_paths[shard_idx],
                          map_location='cpu', weights_only=False)

    def _global_to_shard(self, global_idx: int) -> Tuple[int, int]:
        """Map global example index to (shard_idx, local_idx)."""
        for s in range(self.n_shards):
            if global_idx < self._cumulative[s + 1]:
                return s, global_idx - self._cumulative[s]
        raise IndexError(f"Index {global_idx} out of range")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict:
        # When limit_examples is active, map through permutation so the
        # subset has a representative mix of correct + incorrect examples
        real_idx = self._index_map[idx].item() if self._index_map is not None else idx
        shard_idx, local_idx = self._global_to_shard(real_idx)
        shard = self._load_shard(shard_idx)

        n_steps = shard['n_steps'][local_idx].item()
        target_tokens = shard['target_tokens'][local_idx]  # [MAX_STEPS]

        # Reconstruct per-step structure matrices from target tokens
        structure_matrices = build_structure_matrices_from_tokens(
            target_tokens, n_steps)  # [n_steps, NUM_MATERIALS * MAX_LAYERS]

        return {
            'text': shard['texts'][local_idx],
            'base_logits': shard['base_logits'][local_idx, :n_steps],    # [n_steps, V]
            'target_tokens': target_tokens[:n_steps],                     # [n_steps]
            'structure_matrices': structure_matrices,                      # [n_steps, 200]
            'n_steps': n_steps,
            'incorrect': shard['incorrect'][local_idx].item(),
        }


# ============================================================================
# Collation
# ============================================================================

def collate_full_model(batch: List[Dict]) -> Dict:
    """
    Collate variable-length cached examples into a flat training batch.

    Flattens all autoregressive steps across the batch so MixingMLP processes
    them in one pass. The ConstraintMLP output is expanded per-example
    using n_steps + repeat_interleave in the model's forward pass.

    Returns:
        texts:              List[str], length B
        base_logits:        [total_steps, VOCAB_SIZE_WITH_ERROR] float
        target_tokens:      [total_steps] long
        structure_matrices: [total_steps, NUM_MATERIALS * MAX_LAYERS] float
        n_steps:            [B] long — steps per example (for repeat_interleave)
    """
    texts = [item['text'] for item in batch]
    n_steps = torch.tensor([item['n_steps'] for item in batch], dtype=torch.long)

    # Flatten all steps across examples
    all_logits = [item['base_logits'] for item in batch]
    all_targets = [item['target_tokens'] for item in batch]
    all_structures = [item['structure_matrices'] for item in batch]

    base_logits = torch.cat(all_logits, dim=0)           # [total_steps, V]
    target_tokens = torch.cat(all_targets, dim=0)         # [total_steps]
    structure_matrices = torch.cat(all_structures, dim=0)  # [total_steps, 200]

    return {
        'texts': texts,
        'base_logits': base_logits,
        'target_tokens': target_tokens,
        'structure_matrices': structure_matrices,
        'n_steps': n_steps,
    }


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: FullModel,
    config: FullModelConfig,
    optimizer,
    step: int,
    loss: float,
    save_dir: Path,
    lr: float = None,
    total_steps: int = None,
    steps_per_epoch: int = None,
    epoch: int = None,
    scaler: Optional[torch.amp.GradScaler] = None,
):
    """Save full model checkpoint (LoRA adapters + ConstraintMLP + MixingMLP)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save ConstraintMLP
    torch.save(model.constraint_mlp.state_dict(),
               save_dir / 'constraint_mlp.pt')

    # Save MixingMLP
    torch.save(model.mixing_mlp.state_dict(),
               save_dir / 'mixing_mlp.pt')

    # Save LoRA adapters
    if model.llm is not None and hasattr(model.llm, 'save_pretrained'):
        model.llm.save_pretrained(save_dir / 'constraint_lora')

    # Save optimizer
    torch.save(optimizer.state_dict(), save_dir / 'optimizer.pt')

    # Save GradScaler state (fp16 only)
    if scaler is not None and scaler.is_enabled():
        torch.save(scaler.state_dict(), save_dir / 'scaler.pt')

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save metadata (backwards-compatible: new fields are additive)
    meta = {'step': step, 'loss': loss, 'tag': config.tag()}
    if lr is not None:
        meta['lr'] = lr
    if total_steps is not None:
        meta['total_steps'] = total_steps
    if steps_per_epoch is not None:
        meta['steps_per_epoch'] = steps_per_epoch
    if epoch is not None:
        meta['epoch'] = epoch
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    log(f"[Checkpoint] Saved to {save_dir} at step {step}, loss={loss:.4f}")


def find_latest_checkpoint(save_dir: Path) -> Optional[Tuple[Path, dict]]:
    """
    Scan save_dir for the checkpoint with the highest step count.

    Checks both step_*/meta.json (periodic) and latest/meta.json (epoch).
    Backwards compatible with old meta.json formats.

    Returns:
        (checkpoint_path, meta_dict) for the highest-step checkpoint,
        or None if no valid checkpoint exists.
    """
    best_step = -1
    best_path = None
    best_meta = None

    candidates = list(save_dir.glob('step_*/meta.json'))
    latest_meta = save_dir / 'latest' / 'meta.json'
    if latest_meta.exists():
        candidates.append(latest_meta)

    for meta_path in candidates:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get('step', -1)
            if step > best_step:
                best_step = step
                best_path = meta_path.parent
                best_meta = meta
        except (json.JSONDecodeError, OSError):
            continue

    if best_path is None:
        return None
    return best_path, best_meta


def load_checkpoint_for_resume(
    checkpoint_dir: Path,
    model: FullModel,
    optimizer: AdamW,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    """
    Load model weights and optimizer state from a checkpoint for training resume.

    The model must already have load_llm() called (LLM loaded with fresh LoRA).
    This function overwrites the fresh LoRA weights with the checkpoint's
    trained ones via peft.set_peft_model_state_dict().

    Returns:
        global_step from the checkpoint's meta.json
    """
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / 'meta.json') as f:
        meta = json.load(f)
    global_step = meta['step']

    # --- ConstraintMLP ---
    cmlp_path = checkpoint_dir / 'constraint_mlp.pt'
    model.constraint_mlp.load_state_dict(
        torch.load(cmlp_path, map_location=device, weights_only=True))
    log(f"[Resume] Loaded ConstraintMLP from {cmlp_path}")

    # --- MixingMLP ---
    mmlp_path = checkpoint_dir / 'mixing_mlp.pt'
    model.mixing_mlp.load_state_dict(
        torch.load(mmlp_path, map_location=device, weights_only=True))
    log(f"[Resume] Loaded MixingMLP from {mmlp_path}")

    # --- LoRA adapters ---
    lora_dir = checkpoint_dir / 'constraint_lora'
    if lora_dir.exists() and model.llm is not None:
        from peft import set_peft_model_state_dict
        # Handle both safetensors and bin formats
        adapter_safetensors = lora_dir / 'adapter_model.safetensors'
        adapter_bin = lora_dir / 'adapter_model.bin'
        if adapter_safetensors.exists():
            from safetensors.torch import load_file
            adapter_state = load_file(str(adapter_safetensors))
        elif adapter_bin.exists():
            adapter_state = torch.load(
                str(adapter_bin), map_location=device, weights_only=True)
        else:
            log(f"[WARN] No adapter weights found in {lora_dir}")
            adapter_state = None

        if adapter_state is not None:
            set_peft_model_state_dict(model.llm, adapter_state)
            log(f"[Resume] Loaded LoRA adapters from {lora_dir}")

    # --- Optimizer ---
    opt_path = checkpoint_dir / 'optimizer.pt'
    if opt_path.exists():
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device, weights_only=False))
        log(f"[Resume] Loaded optimizer state (AdamW momentum/variance preserved)")

    # --- GradScaler (fp16 only) ---
    scaler_path = checkpoint_dir / 'scaler.pt'
    if scaler is not None and scaler.is_enabled() and scaler_path.exists():
        scaler.load_state_dict(
            torch.load(scaler_path, map_location=device, weights_only=True))
        log(f"[Resume] Loaded GradScaler state")

    log(f"[Resume] Restored to global_step={global_step}, "
        f"loss={meta.get('loss', '?')}, lr={meta.get('lr', '?')}")

    return global_step


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: FullModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    config: FullModelConfig,
    global_step: int,
    total_steps: int,
    save_dir: Path,
    save_every: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    skip_batches: int = 0,
    steps_per_epoch: int = None,
    epoch: int = None,
) -> Tuple[float, float, int]:
    """
    Train for one epoch.

    Args:
        scaler: GradScaler for fp16 mixed precision (None for bf16 or no AMP).
        autocast_dtype: torch.bfloat16, torch.float16, or None to disable AMP.
        skip_batches: Number of leading batches to skip (for mid-epoch resume).
        steps_per_epoch: Optimizer steps per epoch (forwarded to save_checkpoint).
        epoch: Current epoch index (forwarded to save_checkpoint).

    Returns:
        (avg_loss, avg_accuracy, updated_global_step)
    """
    model.constraint_mlp.train()
    model.mixing_mlp.train()
    if model.llm is not None:
        # Gradient checkpointing requires train() mode
        if getattr(model, '_gradient_checkpointing', False):
            model.llm.train()
        else:
            # eval mode for stable layernorm/dropout in base model
            model.llm.eval()

    use_amp = autocast_dtype is not None
    use_scaler = scaler is not None and scaler.is_enabled()

    epoch_loss = 0.0
    epoch_acc = 0.0
    n_opt_steps = 0
    accum_loss = 0.0
    accum_acc = 0.0

    warmup_steps = int(total_steps * config.warmup_fraction)

    for batch_idx, batch in enumerate(loader):
        # Skip already-processed batches when resuming mid-epoch
        if batch_idx < skip_batches:
            if batch_idx == 0:
                log(f"  [Resume] Skipping {skip_batches} batches "
                    f"to resume mid-epoch...")
            continue
        # Update learning rate
        current_lr = get_lr(
            global_step, total_steps, config.learning_rate, config.warmup_fraction)
        set_lr(optimizer, current_lr)

        # Tokenization is now done in the collate function (DataLoader workers)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        base_logits = batch['base_logits'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        structure_matrices = batch['structure_matrices'].to(device)
        n_steps = batch['n_steps'].to(device)

        # Forward pass with optional mixed precision
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

        # Backward pass (scaler handles fp16 scaling if active)
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += outputs['loss'].item()
        accum_acc += outputs['accuracy'].item()

        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            if config.grad_clip > 0:
                trainable_params = list(model.constraint_mlp.parameters()) + \
                                   list(model.mixing_mlp.parameters())
                if model.llm is not None:
                    trainable_params += [p for p in model.llm.parameters()
                                         if p.requires_grad]
                if use_scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            avg_accum_loss = accum_loss / config.grad_accum_steps
            avg_accum_acc = accum_acc / config.grad_accum_steps
            epoch_loss += avg_accum_loss
            epoch_acc += avg_accum_acc
            n_opt_steps += 1
            global_step += 1

            accum_loss = 0.0
            accum_acc = 0.0

            # Logging
            if global_step % 50 == 0:
                phase = "warmup" if global_step <= warmup_steps else "decay"
                log(f"  Step {global_step}: loss={avg_accum_loss:.4f}, "
                    f"acc={avg_accum_acc:.3f}, lr={current_lr:.2e} [{phase}]")

            # Periodic checkpoint
            if save_every > 0 and global_step % save_every == 0:
                save_checkpoint(
                    model, config, optimizer, global_step,
                    avg_accum_loss, save_dir / f'step_{global_step}',
                    lr=current_lr, total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch, epoch=epoch,
                    scaler=scaler)

    avg_loss = epoch_loss / max(n_opt_steps, 1)
    avg_acc = epoch_acc / max(n_opt_steps, 1)
    return avg_loss, avg_acc, global_step


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train full text → structure model')

    # Data
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to cached pretrained outputs')
    parser.add_argument('--limit-examples', type=int, default=None,
                        help='Limit number of training examples')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=1024,
                        help='Hidden dimension for ConstraintMLP and MixingMLP')
    parser.add_argument('--constraint-layers', type=int, default=4,
                        help='Depth of ConstraintMLP')
    parser.add_argument('--mixing-layers', type=int, default=4,
                        help='Depth of MixingMLP')
    parser.add_argument('--dropout', type=float, default=0.1)

    # LLM / LoRA
    parser.add_argument('--encoder', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-text-len', type=int, default=756)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', type=str, default='q_proj,v_proj')

    # Training
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * this)')
    parser.add_argument('--epochs', type=int, default=10)
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

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override checkpoint dir (default: auto from config tag)')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N optimizer steps (0 to disable)')
    parser.add_argument('--verbose', action='store_true')

    # Resume
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint directory to resume from '
                             '(default: auto-detect from save-dir)')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='Force fresh training, ignore existing checkpoints')

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
        scaler = None  # bf16 doesn't need loss scaling
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
    # Load cached dataset
    # ================================================================

    cache_dir = Path(args.cache_dir)
    dataset = CachedFullModelDataset(cache_dir, limit_examples=args.limit_examples)

    # ================================================================
    # Build model + config
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
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        limit_examples=args.limit_examples,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
    )

    model = FullModel(config)

    # Load LLM + apply LoRA (lazy — this is the expensive step)
    model.load_llm(device, gradient_checkpointing=args.gradient_checkpointing)

    # Move trainable MLP components to device
    model.constraint_mlp.to(device)
    model.mixing_mlp.to(device)

    # Optional torch.compile for MLPs (PyTorch 2.0+)
    if args.compile:
        log("[INFO] Compiling MLPs with torch.compile()...")
        model.constraint_mlp = torch.compile(model.constraint_mlp)
        model.mixing_mlp = torch.compile(model.mixing_mlp)

    # ================================================================
    # DataLoader (with tokenization in collate for worker parallelism)
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

    # Deterministic per-epoch shuffling: seed = 42 + epoch before each epoch.
    # This ensures the same epoch always produces the same batch order,
    # enabling exact mid-epoch resume with consistent data ordering.
    shuffle_generator = torch.Generator()
    sampler = RandomSampler(dataset, generator=shuffle_generator)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ================================================================
    # Parameter counts
    # ================================================================

    n_constraint = sum(p.numel() for p in model.constraint_mlp.parameters())
    n_mixing = sum(p.numel() for p in model.mixing_mlp.parameters())
    n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    n_total = n_constraint + n_mixing + n_lora
    n_frozen = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)

    log(f"\n[INFO] Config tag: {config.tag()}")
    log(f"[INFO] Trainable parameters: {n_total:,}")
    log(f"       LoRA adapters:    {n_lora:,}")
    log(f"       ConstraintMLP:    {n_constraint:,}")
    log(f"       MixingMLP:        {n_mixing:,}")
    log(f"       Frozen LLM base:  {n_frozen:,}")

    # ================================================================
    # Optimizer — separate param groups for potential per-group LR later
    # ================================================================

    param_groups = [
        {
            'params': [p for p in model.llm.parameters() if p.requires_grad],
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'name': 'lora',
        },
        {
            'params': list(model.constraint_mlp.parameters()),
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'name': 'constraint_mlp',
        },
        {
            'params': list(model.mixing_mlp.parameters()),
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'name': 'mixing_mlp',
        },
    ]

    optimizer = AdamW(param_groups)

    # ================================================================
    # Checkpoint directory
    # ================================================================

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = (repo_root / 'train_full_model' / 'data' / 'checkpoints'
                    / config.tag())
    log(f"[INFO] Checkpoints: {save_dir}")

    # ================================================================
    # Training schedule
    # ================================================================

    n_examples = len(dataset)
    effective_batch = args.batch_size * config.grad_accum_steps
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)

    log(f"\n[INFO] Dataset:             {n_examples:,} examples")
    log(f"[INFO] Batch size:          {args.batch_size}")
    log(f"[INFO] Grad accum steps:    {config.grad_accum_steps}")
    log(f"[INFO] Effective batch:     {effective_batch}")
    log(f"[INFO] Steps per epoch:     {steps_per_epoch:,}")
    log(f"[INFO] Total steps:         {total_steps:,}")
    log(f"[INFO] Warmup steps:        {warmup_steps:,}")
    log(f"[INFO] LR:                  {config.learning_rate}")
    log(f"[INFO] Save every:          {args.save_every} steps")

    # ================================================================
    # Resume detection
    # ================================================================

    global_step = 0
    start_epoch = 0
    skip_batches_first_epoch = 0

    if args.no_resume:
        checkpoint_result = None
        log("\n[INFO] --no-resume: ignoring existing checkpoints")
    elif args.resume_from:
        resume_path = Path(args.resume_from)
        if (resume_path / 'meta.json').exists():
            with open(resume_path / 'meta.json') as f:
                checkpoint_result = (resume_path, json.load(f))
        else:
            log(f"[WARN] --resume-from {resume_path} has no meta.json, "
                f"starting fresh")
            checkpoint_result = None
    else:
        checkpoint_result = find_latest_checkpoint(save_dir)

    if checkpoint_result is not None:
        ckpt_path, ckpt_meta = checkpoint_result
        ckpt_step = ckpt_meta.get('step', 0)

        if ckpt_step >= total_steps:
            log(f"\n[INFO] Training already complete! "
                f"Checkpoint at step {ckpt_step} >= total_steps {total_steps}.")
            log(f"       Checkpoint: {ckpt_path}")
            return

        # Use total_steps from checkpoint if available (preserves original
        # LR schedule). Fall back to current args for old checkpoints.
        saved_total = ckpt_meta.get('total_steps')
        if saved_total is not None and saved_total != total_steps:
            log(f"[WARN] total_steps changed: checkpoint has {saved_total}, "
                f"current run expects {total_steps}. "
                f"Using checkpoint's total_steps={saved_total} for LR schedule "
                f"consistency.")
            total_steps = saved_total
            warmup_steps = int(total_steps * config.warmup_fraction)

        log(f"\n[Resume] Found checkpoint at step {ckpt_step}/{total_steps}")
        global_step = load_checkpoint_for_resume(
            ckpt_path, model, optimizer, device, scaler)

        # Calculate epoch and batch position
        start_epoch = global_step // steps_per_epoch
        steps_into_epoch = global_step % steps_per_epoch
        skip_batches_first_epoch = steps_into_epoch * config.grad_accum_steps

        log(f"[Resume] Resuming from epoch {start_epoch + 1}, "
            f"step {global_step}/{total_steps}")
        if skip_batches_first_epoch > 0:
            log(f"[Resume] Will skip {skip_batches_first_epoch} batches "
                f"in first resumed epoch ({steps_into_epoch} optimizer steps)")
    else:
        if not args.no_resume:
            log(f"\n[INFO] No existing checkpoint found. Starting fresh training.")

    # ================================================================
    # Train
    # ================================================================

    log(f"\n{'='*60}")
    if global_step > 0:
        log(f"[INFO] Resuming training from step {global_step}...")
    else:
        log("[INFO] Starting training...")
    log(f"{'='*60}\n")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        log(f"--- Epoch {epoch + 1}/{args.epochs} ---")

        # Deterministic per-epoch shuffle: same epoch always produces
        # the same batch order, enabling exact mid-epoch resume.
        shuffle_generator.manual_seed(42 + epoch)

        # Skip batches only in the first resumed epoch
        skip = skip_batches_first_epoch if epoch == start_epoch else 0

        avg_loss, avg_acc, global_step = train_epoch(
            model, loader, optimizer, device, config,
            global_step, total_steps, save_dir, args.save_every,
            scaler=scaler, autocast_dtype=autocast_dtype,
            skip_batches=skip, steps_per_epoch=steps_per_epoch,
            epoch=epoch)

        epoch_time = time.time() - epoch_t0
        log(f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f}, acc={avg_acc:.3f}, "
            f"time={epoch_time:.0f}s")

        # Epoch checkpoint
        final_lr = get_lr(
            global_step - 1, total_steps, config.learning_rate,
            config.warmup_fraction)
        save_checkpoint(
            model, config, optimizer, global_step,
            avg_loss, save_dir / 'latest', lr=final_lr,
            total_steps=total_steps, steps_per_epoch=steps_per_epoch,
            epoch=epoch, scaler=scaler)

    total_time = time.time() - t0
    log(f"\n{'='*60}")
    log(f"[INFO] Training complete!")
    log(f"       Total time:      {total_time:.0f}s ({total_time/60:.1f}min)")
    log(f"       Final loss:      {avg_loss:.4f}")
    log(f"       Final accuracy:  {avg_acc:.3f}")
    log(f"       Checkpoint:      {save_dir / 'latest'}")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()