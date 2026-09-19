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
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# NOTE: must be contextlib.nullcontext (a reusable class), NOT a
# @contextlib.contextmanager generator. The amp context below is built once
# and re-entered every step, and a generator-based context manager is
# single-use -- it raises "'_GeneratorContextManager' object has no attribute
# 'args'" on the second step. That only bites when the autocast branch is not
# taken (any CPU run, or a GPU run without --bf16), which is why it went
# unnoticed: the production runs all used --bf16 on CUDA, where
# torch.amp.autocast is itself reusable.
_nullcontext = contextlib.nullcontext
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
from src.delta_e_eval import (
    evaluate_delta_e,
    primary_metric as delta_e_primary_metric,
    OPTICAL_SIM_AVAILABLE,
)


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


def evaluate_validation(
    model,
    val_loader,
    device,
    loss_fn,
    bf16: bool,
) -> Tuple[float, float]:
    """Compute val loss + accuracy. Restores the model's train() state on exit."""
    was_training = model.training
    model.eval()
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device.type == "cuda"
        else _nullcontext()
    )
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                with amp_ctx:
                    losses = loss_fn(model, batch_on_device)
                # Weight by scored TOKENS, not by batch size. `loss` and
                # `accuracy` are per-token means, so example-weighting them
                # gives a biased estimator whenever tokens-per-example varies
                # (structures are 2-10 layers). `n_correct` is an exact
                # integer count -- deriving it as int(accuracy * count) used
                # to truncate, which reads as exactly 0.0 whenever accuracy
                # is below 1/batch_size (the regime small models start in).
                n_tok = int(losses["n_tokens"].item())
                total_loss += losses["loss"].item() * n_tok
                total_correct += int(losses["n_correct"].item())
                total_tokens += n_tok
    finally:
        if was_training:
            model.train()
    denom = max(total_tokens, 1)
    return total_loss / denom, total_correct / denom


def append_history(history_path: Path, entry: Dict[str, Any]) -> None:
    """Append one JSON line to <save_dir>/history.jsonl. Never raises — a
    history-write failure must not kill training."""
    try:
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        print(f"[History] WARN: failed to append to {history_path}: {exc}",
              flush=True)


def save_checkpoint(model, config, optimizer, step, loss, save_dir: Path, lr=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        # If the model was wrapped by torch.compile, save the *original* module's
        # state_dict so it loads back into a non-compiled model without the
        # _orig_mod. prefix.
        sd_model = getattr(model, "_orig_mod", model)
        torch.save(sd_model.state_dict(), save_dir / "model.pt")
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        meta = {"step": step, "loss": loss, "tag": config.tag()}
        if lr is not None:
            meta["lr"] = lr
        with open(save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        # Saves are critical — don't let them fail silently. Log the
        # traceback so slurm err files surface the cause.
        import traceback
        print(f"[Checkpoint] FAILED to save to {save_dir} at step {step}: {exc}",
              flush=True)
        traceback.print_exc()
        raise
    print(f"[Checkpoint] Saved to {save_dir} at step {step}", flush=True)


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
    checkpoint_hook: "Optional[Callable[[int, float, float], None]]" = None,
    epoch: "int | None" = None,
) -> Dict[str, float]:
    """Run one epoch of training. Shared by `scripts/training.py` (single
    full run) and `scripts/lr_tuning.py` (one trial per candidate LR).

    Per-step loss logging fires every `log_every` steps when `verbose=True`,
    including EMA'd step-time / samples-per-sec measured between log ticks.

    Checkpointing: if `checkpoint_hook` is supplied it's called on every
    `save_every`-th step as `checkpoint_hook(step, last_loss, current_lr)` —
    the hook owns everything (val eval, history write, actual save call).
    If `checkpoint_hook` is None but `save_dir`/`save_every`/`config` are
    set, we fall back to a plain `save_checkpoint(...)` (the lr_tuning path
    leaves all four None, so nothing writes).
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

    last_log_step = global_step
    last_log_time = time.perf_counter()
    samples_since_log = 0

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
        samples_since_log += int(batch["lab"].size(0))

        if verbose and global_step % log_every == 0:
            phase = "warmup" if global_step <= warmup_steps else "decay"
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            steps_delta = max(global_step - last_log_step, 1)
            ms_per_step = 1000.0 * elapsed / steps_delta
            ex_per_s = samples_since_log / elapsed
            print(
                f"  Step {global_step}/{total_steps}: "
                f"loss={last_loss:.4f}, "
                f"acc={losses['accuracy'].item():.3f}, "
                f"lr={current_lr:.2e} [{phase}] "
                f"dt={ms_per_step:.1f}ms/step ({ex_per_s:.0f} ex/s)",
                flush=True,
            )
            last_log_time = now
            last_log_step = global_step
            samples_since_log = 0

        # `save_every` is truthy (int > 0) only when the caller enabled saves.
        # Short-circuit BEFORE the modulo so lr_tuning.py (save_every=None)
        # doesn't crash with ZeroDivisionError.
        if save_every and global_step % save_every == 0:
            if checkpoint_hook is not None:
                checkpoint_hook(global_step, last_loss, current_lr)
            elif save_dir is not None and config is not None:
                save_checkpoint(
                    model, config, optimizer, global_step,
                    last_loss, save_dir / f"step_{global_step}",
                    lr=current_lr,
                )

        # Stop at the planned budget. This is a no-op for a fresh run (the
        # loader yields exactly steps_per_epoch batches), but it is essential
        # after a MID-EPOCH --resume: start_epoch is computed as
        # global_step // steps_per_epoch, so the resumed epoch would
        # otherwise run a full loader pass on top of the steps already done
        # and overshoot total_steps. A run resumed at step 1500 of a
        # 2400-step plan would end at 3900 -- 62% more compute than planned,
        # with the cosine schedule running off its own end. Resuming exactly
        # at an epoch boundary (what save_dir/"latest" holds) was always
        # safe; this makes every other resume point safe too.
        if global_step >= total_steps:
            break

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
    parser.add_argument("--limit-val-examples", type=int, default=5000,
                        help="Cap on val examples per checkpoint eval "
                             "(default: 5000, the full 0.05%% val split). "
                             "Set to 0 to skip val eval entirely.")
    # -- DeltaE validation. This is the metric that matters: the CE/DeltaE
    # decoupling is verified on INDIGO, so val_loss is NOT a usable proxy
    # for deployed quality, and the compute-optimal scaling study fits its
    # IsoFLOP parabolas on DeltaE. Kept separate from --limit-val-examples
    # because DeltaE costs ~100x more per example than CE (autoregressive
    # decode + optical sim), so it runs on a much smaller slice.
    parser.add_argument("--limit-de-examples", type=int, default=256,
                        help="Examples per DeltaE_00 val eval (default: 256). "
                             "Costs roughly 50ms/example of optical sim plus "
                             "the autoregressive decode, i.e. a few percent "
                             "overhead at the default save cadence. Set to 0 "
                             "to skip DeltaE eval entirely.")
    parser.add_argument("--de-every", type=int, default=0,
                        help="Run the DeltaE eval every N steps. 0 (default) "
                             "means every save tick, matching --save-every. "
                             "Use a multiple of --save-every to sample DeltaE "
                             "more coarsely than val_loss on short runs.")
    parser.add_argument("--de-sample", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Temperature-sample instead of greedy-decoding "
                             "the DeltaE eval. Default greedy, so the metric "
                             "is deterministic and comparable across "
                             "checkpoints. Note slurms/de_curve.sh defaults "
                             "the other way (SAMPLE_PREDICTIONS=1), so its "
                             "curves are not comparable to this one.")
    parser.add_argument("--de-temperature", type=float, default=1.0,
                        help="Temperature for --de-sample.")
    parser.add_argument("--de-on-epoch-end", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Force a DeltaE eval at every epoch boundary "
                             "(default on). Turn OFF for multi-epoch runs "
                             "where only the end-of-run value is wanted: "
                             "otherwise an 8-epoch run pays 8 DeltaE evals "
                             "while a 1-epoch run pays one, which both wastes "
                             "time and makes the two cost different amounts. "
                             "The FINAL save always evaluates regardless.")
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved checkpoint dir (e.g. step_91000/ "
                             "or latest/) to resume from. Loads model.pt + "
                             "optimizer.pt + step from meta.json and continues. "
                             "All other CLI args must match the original run so "
                             "save_dir, total_steps, and the LR schedule align.")

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

    # ---- Validation dataset (small held-out slice from the same DATA_DIR).
    # 5k examples at the default 99.95/0.05 split (see src/dataset.py). Used
    # for the per-save-tick val-loss line in history.jsonl and for early
    # divergence detection during long runs. Set --limit-val-examples 0 to
    # skip. Val workers are capped at 2: this loader is iterated only every
    # `save-every` steps, so a large worker pool sits idle 99% of the time.
    val_loader = None
    if args.limit_val_examples and args.limit_val_examples > 0:
        val_dataset = FlexThinFilmDataset(
            data_dir,
            seed=args.seed,
            split="validation",
            verbose=False,
            limit_examples=args.limit_val_examples,
            streaming=args.streaming,
        )
        val_workers = min(2, args.num_workers)
        val_loader_kw = {}
        if val_workers > 0:
            val_loader_kw["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=active_collate,
            num_workers=val_workers,
            pin_memory=True,
            **val_loader_kw,
        )
        print(f"[INFO] Val split: {len(val_dataset):,} examples "
              f"(evaluated on every checkpoint save)")
    else:
        print("[INFO] Val eval disabled (--limit-val-examples 0)")

    # ---- DeltaE validation slice. Held as a materialised list of
    # TrainingExamples rather than a DataLoader: the DeltaE path decodes
    # autoregressively one example at a time and needs each example's raw
    # material pool, not a collated batch. Reading it once up front keeps
    # every later eval off the parquet shards.
    de_examples = None
    de_simulator = None
    if args.limit_de_examples and args.limit_de_examples > 0:
        if not OPTICAL_SIM_AVAILABLE:
            print("[INFO] DeltaE eval requested but the optical simulator is "
                  "unavailable (jaxlayerlumos missing); skipping. val_de will "
                  "be absent from history.jsonl.", flush=True)
        else:
            de_dataset = FlexThinFilmDataset(
                data_dir,
                seed=args.seed,
                split="validation",
                verbose=False,
                limit_examples=args.limit_de_examples,
                streaming=args.streaming,
            )
            de_examples = list(de_dataset)
            # Reused across evals so jaxlayerlumos' trace cache stays warm
            # (it re-traces per stack depth; see src/delta_e_eval.py).
            from src.optical_sim import OpticalSimulator
            de_simulator = OpticalSimulator(incidence_angle=0)
            print(f"[INFO] DeltaE val slice: {len(de_examples):,} examples, "
                  f"{'sampled T=' + str(args.de_temperature) if args.de_sample else 'greedy'}, "
                  f"every {args.de_every or args.save_every} steps", flush=True)

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
    # Create the directory NOW (parents=True) so any permission/disk error
    # surfaces before training starts rather than at the first save attempt.
    save_dir.mkdir(parents=True, exist_ok=True)
    # And immediately write a sentinel so we can confirm writes work on this
    # filesystem before sinking hours into training.
    try:
        (save_dir / ".write_test").write_text("ok")
        (save_dir / ".write_test").unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write to checkpoint directory {save_dir}: {exc}. "
            "Check permissions and free disk before re-running."
        ) from exc
    print(f"[INFO] Checkpoints will be saved to: {save_dir.resolve()}", flush=True)

    n_examples = len(dataset)
    steps_per_epoch = math.ceil(n_examples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)
    print(f"[INFO] Dataset size: {n_examples:,} examples")
    print(f"[INFO] Steps per epoch: {steps_per_epoch:,}")
    print(f"[INFO] Total steps: {total_steps:,}")
    print(f"[INFO] Warmup steps: {warmup_steps:,} ({args.warmup_fraction:.1%} of total)")

    global_step = 0
    start_epoch = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if not (resume_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"--resume {resume_dir}: no model.pt found. Expected a "
                "checkpoint dir like .../step_91000/ or .../latest/."
            )
        # Strip torch.compile wrapper for loading too.
        sd_model = getattr(model, "_orig_mod", model)
        sd_model.load_state_dict(torch.load(
            resume_dir / "model.pt", map_location=device, weights_only=True
        ))
        optimizer.load_state_dict(torch.load(
            resume_dir / "optimizer.pt", map_location=device, weights_only=True
        ))
        with open(resume_dir / "meta.json") as f:
            meta = json.load(f)
        global_step = int(meta["step"])
        start_epoch = global_step // max(steps_per_epoch, 1)
        print(f"[Resume] Loaded {resume_dir}", flush=True)
        print(f"[Resume] global_step={global_step:,} (of {total_steps:,}), "
              f"resuming at epoch {start_epoch + 1}/{args.epochs}", flush=True)
        if global_step >= total_steps:
            print(f"[Resume] Already at or past total_steps "
                  f"({total_steps:,}); writing final save and exiting.",
                  flush=True)
            save_checkpoint(model, config, optimizer, global_step,
                            meta.get("loss", float("nan")),
                            save_dir / "final",
                            lr=meta.get("lr"))
            save_checkpoint(model, config, optimizer, global_step,
                            meta.get("loss", float("nan")),
                            save_dir / "latest",
                            lr=meta.get("lr"))
            return

    print("[INFO] Starting training...")
    print(f"[INFO] Logging every {args.log_every} step(s) "
          f"(verbose={args.verbose})")

    # -- Checkpoint hook: eval val, append history.jsonl, save. Called from
    # inside run_one_epoch at every save_every step and from main() at the
    # per-epoch and final saves. Everything the hook needs (loaders, device,
    # loss_fn, bf16 flag) is captured in this closure, so run_one_epoch only
    # needs to know (step, train_loss, lr).
    history_path = save_dir / "history.jsonl"
    print(f"[INFO] Training history: {history_path.resolve()}", flush=True)

    de_every = args.de_every or args.save_every

    def _save_and_log(step: int, train_loss: float, lr: float,
                      subdir: Path, *, epoch: "int | None" = None,
                      force_de: bool = False) -> None:
        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            val_loss, val_acc = evaluate_validation(
                model, val_loader, device, loss_fn, bf16=args.bf16,
            )

        # DeltaE_00 on the held-out slice. This is the metric checkpoint
        # selection and the scaling study's IsoFLOP fits key on -- val_loss
        # is recorded alongside it as a diagnostic only. Can run at a
        # coarser cadence than val_loss since it is far more expensive.
        de_result = None
        if de_examples and (force_de or step % de_every == 0):
            de_t0 = time.perf_counter()
            de_result = evaluate_delta_e(
                model, de_examples, device,
                limit=args.limit_de_examples,
                sample=args.de_sample,
                temperature=args.de_temperature,
                seed=args.seed if args.de_sample else None,
                simulator=de_simulator,
            )
            de_result["wall_seconds"] = time.perf_counter() - de_t0

        save_checkpoint(model, config, optimizer, step, train_loss,
                        subdir, lr=lr)

        entry = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if de_result is not None and de_result.get("available"):
            # Contract: a val_de_* DISTRIBUTION key is present only when it
            # holds a real number. When the eval ran but scored nothing (every
            # generation invalid, or every sim failed) we still record that it
            # ran, via the count/rate keys, but omit the stats rather than
            # writing nulls -- so downstream plotting and the scaling fits can
            # treat "val_de_median present" as "usable".
            entry["val_de_n"] = de_result.get("n_scored", 0)
            entry["val_de_valid_rate"] = de_result.get("valid_rate")
            entry["val_de_greedy"] = de_result.get("greedy")
            entry["val_de_seconds"] = de_result.get("wall_seconds")
            if de_result.get("n_scored"):
                # Flat val_de_* keys keep history.jsonl one level deep and
                # directly plottable; by_chroma is the nested exception, and
                # is what the chroma-conditioned scaling fits read.
                entry["val_de_median"] = de_result["delta_e_median"]
                entry["val_de_mean"] = de_result["delta_e_mean"]
                entry["val_de_p75"] = de_result["delta_e_p75"]
                entry["val_de_p95"] = de_result["delta_e_p95"]
                entry["val_de_by_chroma"] = {
                    bucket: {k: stats[k] for k in
                             ("n", "n_examples", "median", "mean", "p75", "p95")
                             if k in stats}
                    for bucket, stats in de_result.get("by_chroma", {}).items()
                }
        append_history(history_path, entry)

        if val_loader is not None:
            line = f"[Val] step={step} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            if de_result is not None and de_result.get("n_scored"):
                by_c = de_result.get("by_chroma", {})
                buckets = " ".join(
                    f"{b}={by_c[b]['median']:.2f}"
                    for b in ("low", "mid", "high")
                    if by_c.get(b, {}).get("n")
                )
                line += (f" val_de_median={de_result['delta_e_median']:.3f}"
                         f" p95={de_result['delta_e_p95']:.3f}"
                         f" [{buckets}]"
                         f" ({de_result['wall_seconds']:.0f}s)")
            print(line, flush=True)

    def mid_epoch_hook(step: int, train_loss: float, lr: float) -> None:
        _save_and_log(step, train_loss, lr, save_dir / f"step_{step}")

    avg_loss = float("nan")
    final_lr = args.lr
    for epoch in range(start_epoch, args.epochs):
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
            checkpoint_hook=mid_epoch_hook,
            epoch=epoch,
        )
        global_step = epoch_out["global_step"]
        avg_loss = epoch_out["avg_loss"]
        avg_acc = epoch_out["avg_acc"]
        final_lr = epoch_out["final_lr"]
        print(f"[Epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.4f}, "
              f"acc={avg_acc:.3f}, final_lr={final_lr:.2e}")

        _save_and_log(global_step, avg_loss, final_lr,
                      save_dir / "latest", epoch=epoch,
                      force_de=args.de_on_epoch_end)

    # Belt-and-suspenders: explicit save after the epoch loop exits, even if
    # args.epochs is somehow 0 or run_one_epoch returned early. Overwrites
    # the per-epoch "latest" with identical content if everything ran.
    _save_and_log(global_step, avg_loss, final_lr,
                  save_dir / "final", epoch=args.epochs - 1, force_de=True)

    print("[INFO] Training complete!")
    print(f"[INFO] Final checkpoint:  {save_dir / 'final'}", flush=True)
    print(f"[INFO] Latest checkpoint: {save_dir / 'latest'}", flush=True)


if __name__ == "__main__":
    main()
