#!/usr/bin/env python3
"""
Training Script for INDIGO (FlexMaterialMLP / FlexMaterialCrossAttn).

Single-stage training: Lab + per-example material pool → next-layer token.

Training approach (autoregressive expansion):
- For an N-layer structure with N < MAX_LAYERS we generate N+1 samples per
  example (N layer-token predictions + 1 EOS prediction).
- For an N=MAX_LAYERS structure we generate N samples (no EOS step;
  generation stops at the length cap).

Each expanded sample carries the same material pool (featurized once
per example). The model receives:
    lab              : [B, 3]
    pool_features    : [B, M_MAX, 2, NUM_LAMBDA]
    pool_mask        : [B, M_MAX]  bool
    pool_size        : [B]         int
    structure_matrix : [B, M_MAX, MAX_LAYERS]
and predicts:
    target_token     : [B]         int in [0, VOCAB_SIZE)

LR schedule: 2% linear warmup + cosine decay.
Checkpoints: data/checkpoints/<config.tag()>/step_<N>/  and  .../latest/
"""

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch


@contextlib.contextmanager
def _nullcontext():
    yield
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import FlexThinFilmDataset, TrainingExample, find_repo_root
from src.material_features import featurize_pool, pad_pool_features
from src.materials_vocab import (
    EOS_TOKEN,
    M_MAX,
    MAX_LAYERS,
    build_structure_matrix,
    encode_layer,
)
from src.model import ModelConfig, build_model, compute_loss, compute_loss_packed


def collate_fn(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """Expand each example into autoregressive sub-samples and batch them.

    The pool is featurized once per example and broadcast across all
    expanded steps (same pool throughout the autoregressive trajectory).
    """
    all_lab: List[torch.Tensor] = []
    all_pool_feats: List[torch.Tensor] = []
    all_pool_masks: List[torch.Tensor] = []
    all_pool_sizes: List[int] = []
    all_structures: List[torch.Tensor] = []
    all_targets: List[int] = []

    for ex in examples:
        pool_feats_unpadded = featurize_pool(ex.pool, mode="raw_spectrum")
        pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
        pool_size = len(ex.pool)
        n_layers = len(ex.target_slots)
        max_step = n_layers if n_layers < MAX_LAYERS else MAX_LAYERS - 1

        for step in range(max_step + 1):
            all_lab.append(ex.lab)
            all_pool_feats.append(pool_feats)
            all_pool_masks.append(pool_mask)
            all_pool_sizes.append(pool_size)

            if step == 0:
                all_structures.append(torch.zeros(M_MAX, MAX_LAYERS))
            else:
                all_structures.append(build_structure_matrix(
                    ex.target_slots[:step],
                    ex.target_thicknesses[:step],
                ))

            if step < n_layers:
                all_targets.append(encode_layer(
                    ex.target_slots[step],
                    ex.target_thicknesses[step],
                ))
            else:
                all_targets.append(EOS_TOKEN)

    return {
        "lab": torch.stack(all_lab),
        "pool_features": torch.stack(all_pool_feats),
        "pool_mask": torch.stack(all_pool_masks),
        "pool_size": torch.tensor(all_pool_sizes, dtype=torch.long),
        "structure_matrix": torch.stack(all_structures),
        "target_token": torch.tensor(all_targets, dtype=torch.long),
    }


# Sequence length for the packed cross_attn forward: start token + MAX_LAYERS
# past tokens / prediction positions.
_PACKED_SEQ_LEN = MAX_LAYERS + 1


def collate_fn_packed(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """Packed teacher-forcing collate for the cross_attn head.

    Each example becomes ONE batch row carrying:
      - lab, pool_features, pool_mask, pool_size : as before, 1 copy per example
      - structure_matrix : the FULL deposited structure (all layers laid down)
      - target_tokens [SEQ_LEN] : encoded layer tokens at positions
        0..n_layers-1, EOS_TOKEN at position n_layers, and -100 at positions
        n_layers+1..MAX_LAYERS so they are excluded from loss.

    The packed forward predicts at every sequence position in a single pass —
    one slot-encoder run amortised across all (n_layers + 1) token decisions
    per example.
    """
    all_lab: List[torch.Tensor] = []
    all_pool_feats: List[torch.Tensor] = []
    all_pool_masks: List[torch.Tensor] = []
    all_pool_sizes: List[int] = []
    all_structures: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for ex in examples:
        pool_feats_unpadded = featurize_pool(ex.pool, mode="raw_spectrum")
        pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
        pool_size = len(ex.pool)

        n_layers = len(ex.target_slots)
        # Whether EOS gets a prediction position: yes if structure ended before
        # MAX_LAYERS (so n_layers < MAX_LAYERS); no if it filled to the cap.
        emits_eos = n_layers < MAX_LAYERS

        # Full deposited structure (every layer the model is supposed to
        # produce). The causal mask in the model ensures position p only
        # sees layers 0..p-1, so feeding the full structure here does not
        # leak information.
        full_structure = build_structure_matrix(
            ex.target_slots, ex.target_thicknesses
        )

        targets = torch.full((_PACKED_SEQ_LEN,), -100, dtype=torch.long)
        for k in range(n_layers):
            targets[k] = encode_layer(ex.target_slots[k], ex.target_thicknesses[k])
        if emits_eos:
            targets[n_layers] = EOS_TOKEN

        all_lab.append(ex.lab)
        all_pool_feats.append(pool_feats)
        all_pool_masks.append(pool_mask)
        all_pool_sizes.append(pool_size)
        all_structures.append(full_structure)
        all_targets.append(targets)

    return {
        "lab": torch.stack(all_lab),
        "pool_features": torch.stack(all_pool_feats),
        "pool_mask": torch.stack(all_pool_masks),
        "pool_size": torch.tensor(all_pool_sizes, dtype=torch.long),
        "structure_matrix": torch.stack(all_structures),
        "target_tokens": torch.stack(all_targets),
    }


def get_lr_schedule(
    step: int, total_steps: int, base_lr: float, warmup_fraction: float = 0.02
) -> float:
    """Linear warmup followed by cosine decay to zero."""
    warmup_steps = int(total_steps * warmup_fraction)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    decay_steps = total_steps - warmup_steps
    decay_progress = (step - warmup_steps) / max(decay_steps, 1)
    decay_progress = min(decay_progress, 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * decay_progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_checkpoint(model, config, optimizer, step, loss, save_dir: Path, lr=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    with open(save_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    meta = {"step": step, "loss": loss, "tag": config.tag()}
    if lr is not None:
        meta["lr"] = lr
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Checkpoint] Saved to {save_dir} at step {step}")


def train_step(model, batch, device, loss_fn=compute_loss) -> Dict[str, torch.Tensor]:
    """Move batch to device and run one forward + loss pass.

    `loss_fn` decides which target field is consumed:
      - `compute_loss`        -> `target_token` (fanned-out collate)
      - `compute_loss_packed` -> `target_tokens` (packed collate)
    """
    batch_on_device = {
        "lab": batch["lab"].to(device),
        "pool_features": batch["pool_features"].to(device),
        "pool_mask": batch["pool_mask"].to(device),
        "pool_size": batch["pool_size"].to(device),
        "structure_matrix": batch["structure_matrix"].to(device),
    }
    if "target_token" in batch:
        batch_on_device["target_token"] = batch["target_token"].to(device)
    if "target_tokens" in batch:
        batch_on_device["target_tokens"] = batch["target_tokens"].to(device)
    return loss_fn(model, batch_on_device)


def run_one_epoch(
    model,
    optimizer,
    loader,
    device,
    *,
    total_steps: int,
    base_lr: float,
    warmup_fraction: float,
    grad_clip: float,
    log_every: int,
    verbose: bool,
    global_step_start: int = 0,
    save_dir: "Path | None" = None,
    save_every: "int | None" = None,
    config: "ModelConfig | None" = None,
    loss_fn=compute_loss,
    bf16: bool = False,
) -> Dict[str, float]:
    """Run one epoch of training. Shared by `scripts/training.py` (single
    full run) and `scripts/lr_tuning.py` (one trial per candidate LR).

    Per-step loss logging fires every `log_every` steps when `verbose=True`.
    Pass `save_dir`/`save_every`/`config` to enable mid-epoch checkpointing
    (the lr_tuning case leaves these `None`).
    """
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    n_batches = 0
    global_step = global_step_start
    warmup_steps = int(total_steps * warmup_fraction)

    current_lr = base_lr
    last_loss = float("nan")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device.type == "cuda"
        else _nullcontext()
    )
    for batch in loader:
        current_lr = get_lr_schedule(global_step, total_steps, base_lr, warmup_fraction)
        set_lr(optimizer, current_lr)

        optimizer.zero_grad()
        with amp_ctx:
            losses = train_step(model, batch, device, loss_fn=loss_fn)
        # bf16 has the same dynamic range as fp32, so no GradScaler is needed.
        losses["loss"].backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        last_loss = losses["loss"].item()
        epoch_loss += last_loss
        epoch_acc += losses["accuracy"].item()
        n_batches += 1
        global_step += 1

        if verbose and global_step % log_every == 0:
            phase = "warmup" if global_step <= warmup_steps else "decay"
            print(
                f"  Step {global_step}/{total_steps}: "
                f"loss={last_loss:.4f}, "
                f"acc={losses['accuracy'].item():.3f}, "
                f"lr={current_lr:.2e} [{phase}]",
                flush=True,
            )

        if save_dir is not None and save_every and config is not None \
                and global_step % save_every == 0:
            save_checkpoint(
                model, config, optimizer, global_step,
                last_loss, save_dir / f"step_{global_step}",
                lr=current_lr,
            )

    return {
        "global_step": global_step,
        "avg_loss": epoch_loss / max(n_batches, 1),
        "avg_acc": epoch_acc / max(n_batches, 1),
        "last_loss": last_loss,
        "final_lr": current_lr,
        "n_batches": n_batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train INDIGO flex-material model")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to a directory of INDIGO parquet shards "
                             "(default: <repo>/data/train, matching the "
                             "OUTPUT_DIR default of slurms/generate_data.sh)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-examples", type=int, default=None,
                        help="Limit to first N examples (for testing/debugging)")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream the dataset shard-by-shard (one parquet "
                             "table in scope at a time, ~140 MB/worker). The "
                             "legacy in-memory order-preserving path OOMs at "
                             "production scale (~40 KB/example × millions); "
                             "set --streaming for any full-dataset run. "
                             "Trade-off: rows come out shard-by-shard rather "
                             "than in the global `self.order` permutation.")

    # Model hyperparameters
    parser.add_argument("--feature-mode", type=str, default="raw_spectrum",
                        choices=["raw_spectrum", "compact"])
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-mode", type=str, default="mlp",
                        choices=["mlp", "cross_attn"],
                        help="Backbone architecture: 'mlp' (flatten-then-MLP, "
                             "original) or 'cross_attn' (packed transformer "
                             "decoder: causal self-attn over the partial "
                             "structure + cross-attn onto the pool).")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Attention heads (cross_attn only).")
    parser.add_argument("--slot-encoder-layers", type=int, default=0,
                        help="Depth of the slot self-attention encoder "
                             "(cross_attn only). 0 = use --n-layers. "
                             "4 is recommended — 8-layer self-attn over "
                             "≤32 set elements is overkill.")
    parser.add_argument("--decoder-layers", type=int, default=1,
                        help="Decoder depth (cross_attn only). Each layer "
                             "does causal self-attn + cross-attn + FFN.")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4.42e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="DataLoader prefetch_factor (default: 1). Our "
                             "pipeline is producer-bound — workers are ~10x "
                             "slower than the GPU consumer — so a deeper queue "
                             "buys no throughput, only memory. Bump only if "
                             "the model + batch grow enough to make the "
                             "consumer the bottleneck.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)

    # Performance knobs
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Wrap forward+loss in torch.amp.autocast bfloat16. "
                             "~2x speedup on L40s/H100, no GradScaler needed. "
                             "Default off so CI/CPU-only runs stay fp32.")
    parser.add_argument("--packed-tf", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Use packed teacher-forcing collate (1 row per "
                             "example, all sequence positions scored in one "
                             "pass). cross_attn only. Default: on for "
                             "cross_attn, off for mlp (mlp can't benefit).")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Wrap the model in torch.compile(). Extra "
                             "1.2-1.5x speedup once the trace stabilises.")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1000)

    # Logging
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print per-step loss every N optimizer steps "
                             "when --verbose is set (default: 100)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print per-step loss + LR (default: on). "
                             "Pass --no-verbose to silence.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "train"
    print(f"[INFO] Loading data from {data_dir}")

    dataset = FlexThinFilmDataset(
        data_dir,
        seed=args.seed,
        split=args.split,
        verbose=args.verbose,
        limit_examples=args.limit_examples,
        streaming=args.streaming,
    )

    # Packed teacher-forcing defaults to on for cross_attn, off for mlp.
    packed_tf = args.packed_tf if args.packed_tf is not None else (args.head_mode == "cross_attn")
    if packed_tf and args.head_mode == "mlp":
        raise ValueError("--packed-tf is incompatible with --head-mode mlp")
    active_collate = collate_fn_packed if packed_tf else collate_fn
    loss_fn = compute_loss_packed if packed_tf else compute_loss
    print(f"[INFO] Collate: {active_collate.__name__} "
          f"(packed_tf={packed_tf})")

    loader_kw = {}
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=active_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        **loader_kw,
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
        slot_encoder_layers=args.slot_encoder_layers,
        decoder_layers=args.decoder_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit_examples=args.limit_examples,
    )

    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {type(model).__name__} (head_mode={args.head_mode}, "
          f"d_model={args.d_model}, n_layers={args.n_layers})")
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Config tag: {config.tag()}")

    if args.compile:
        print(f"[INFO] torch.compile(model) — first batch will be slow to trace")
        model = torch.compile(model)
    if args.bf16:
        print(f"[INFO] bf16 autocast: on")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = repo_root / "data" / "checkpoints" / config.tag()
    print(f"[INFO] Checkpoints will be saved to: {save_dir}")

    n_examples = len(dataset)
    steps_per_epoch = math.ceil(n_examples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)
    print(f"[INFO] Dataset size: {n_examples:,} examples")
    print(f"[INFO] Steps per epoch: {steps_per_epoch:,}")
    print(f"[INFO] Total steps: {total_steps:,}")
    print(f"[INFO] Warmup steps: {warmup_steps:,} ({args.warmup_fraction:.1%} of total)")

    global_step = 0
    print("[INFO] Starting training...")
    print(f"[INFO] Logging every {args.log_every} step(s) "
          f"(verbose={args.verbose})")

    for epoch in range(args.epochs):
        epoch_out = run_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=loader,
            device=device,
            total_steps=total_steps,
            base_lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            verbose=args.verbose,
            global_step_start=global_step,
            save_dir=save_dir,
            save_every=args.save_every,
            config=config,
            loss_fn=loss_fn,
            bf16=args.bf16,
        )
        global_step = epoch_out["global_step"]
        avg_loss = epoch_out["avg_loss"]
        avg_acc = epoch_out["avg_acc"]
        final_lr = epoch_out["final_lr"]
        print(f"[Epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.4f}, "
              f"acc={avg_acc:.3f}, final_lr={final_lr:.2e}")

        save_checkpoint(model, config, optimizer, global_step, avg_loss,
                        save_dir / "latest", lr=final_lr)

    print("[INFO] Training complete!")
    print(f"[INFO] Final checkpoint: {save_dir / 'latest'}")


if __name__ == "__main__":
    main()
