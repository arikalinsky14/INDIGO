#!/usr/bin/env python3
"""
INDIGO Post-Training ΔE Finetune
================================

Loads a pretrained cross-attn checkpoint and finetunes it against ΔE₀₀
directly, using the differentiable full-stack rollout defined in
`src/de_finetune.py`. See `analyses/de_finetune/README.md` for the full
design (motivation, credit-assignment, STE, PyTorch↔JAX bridge).

Two supported experiments, selected by --freeze-encoder:
    --freeze-encoder        (Experiment A) freeze MaterialEncoder +
                            slot_encoder; train decoder + heads. LR ~5e-6.
    (default: no freezing)  (Experiment B) full-model finetune. LR ~1e-6.

Model hyperparameters (d_model, n_layers, head_mode, ...) MUST match the
pretrained checkpoint being loaded — they're used both to construct the
model and to look up the pretrained tag if --checkpoint isn't given.

Outputs
-------
    <save_dir>/step_<N>/    per-checkpoint model + optimizer + meta
    <save_dir>/latest/      most recent (updated at every save)
    <save_dir>/history.jsonl one row per checkpoint (train + val ΔE + metrics)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import FlexThinFilmDataset, find_repo_root
from src.de_finetune import (
    collate_fn,
    finetune_de_loss,
    freeze_encoder_for_decoder_only,
    unfreeze_all,
)
from src.model import ModelConfig, build_model


# ============================================================================
# LR schedule (cosine with warmup, matches pretrain conventions)
# ============================================================================


def _lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    schedule: str = "cosine",
) -> float:
    """LR at `step`. Linear warmup then either cosine decay to 0
    (default, matches pretrain) or a flat plateau at base_lr
    (long-training-friendly — the cosine decay to zero appears to hurt
    once past a mid-training minimum; see Sept 10 note in
    analyses/de_finetune/)."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if schedule == "constant":
        return base_lr
    # cosine to zero
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# Back-compat alias — callers that imported by the old name still work.
_cosine_warmup_lr = _lr_at_step


def _epsilon_at_step(
    step: int,
    total_steps: int,
    eps_start: float,
    eps_end: float,
    decay_fraction: float,
) -> float:
    """ε-exploration fraction at `step`. Linear anneal from `eps_start`
    (at step 0) to `eps_end` (at step decay_fraction * total_steps),
    then held at `eps_end`. Set both to 0 to disable exploration
    (default). Set both > 0 for constant ε.
    """
    if eps_start == 0.0 and eps_end == 0.0:
        return 0.0
    decay_steps = max(1, int(total_steps * max(0.0, min(1.0, decay_fraction))))
    frac = min(1.0, step / decay_steps)
    return eps_start + (eps_end - eps_start) * frac


# ============================================================================
# Checkpoint IO (mirrors scripts/training.py conventions)
# ============================================================================


def save_checkpoint(model, config: ModelConfig, optimizer, step: int,
                    loss_de: float, subdir: Path, *,
                    lr: Optional[float] = None) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    sd_model = getattr(model, "_orig_mod", model)
    torch.save(sd_model.state_dict(), subdir / "model.pt")
    torch.save(optimizer.state_dict(), subdir / "optimizer.pt")
    meta = {
        "step": step,
        "loss_de": loss_de,
        "lr": lr,
        "config": config.__dict__,
        "phase": "finetune_de",
        "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    # Also write config.json alongside — the inference pipeline
    # (inference.src.generate.load_inference_model) expects it as a
    # top-level file. This makes finetune checkpoints drop-in usable
    # with inference/scripts/test_eval.py, gamut_eval.py, etc.
    with open(subdir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)


def append_history(history_path: Path, row: Dict) -> None:
    with open(history_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ============================================================================
# Validation eval (mean ΔE₀₀ on a held-out slice)
# ============================================================================


def evaluate_val(model, val_loader, device, ce_loss_weight: float = 0.0,
                 sim_feedback: bool = False) -> Dict[str, float]:
    """Val eval. Threads ce_loss_weight through so val loss_ce is also
    reported when the anchor is on; val_loss_de is always the primary
    metric regardless of weight (that's the target objective).

    NOTE: val ALWAYS runs the STE greedy path (real_sim_topk=0) so that
    val_loss_de measures the ΔE at the model's own greedy pick — the
    same quantity across all training modes (STE, top-K K=5/10/15,
    with/without CE anchor). Otherwise different K would spend
    different sim budgets on val and val_loss_de wouldn't be
    apples-to-apples across runs.

    sim_feedback MUST match the training-time setting: if training used
    residual conditioning, val must also compute + pass residuals or the
    model's forward is under-conditioned and val_de is pessimistic."""
    model.eval()
    metrics_accum: Dict[str, float] = {}
    n_total_positions = 0
    with torch.no_grad():
        for batch in val_loader:
            _, m = finetune_de_loss(
                model, batch, device=device,
                ce_loss_weight=ce_loss_weight,
                real_sim_topk=0,
                sim_feedback=sim_feedback,
            )
            n_pos = int(m.get("n_positions", 0))
            if n_pos == 0:
                continue
            n_total_positions += n_pos
            for k, v in m.items():
                if k in ("batch_size", "n_positions"):
                    continue
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v * n_pos
    model.train()
    if n_total_positions == 0:
        return {"val_loss_de": float("nan")}
    out: Dict[str, float] = {"val_n_positions": n_total_positions}
    for k, v in metrics_accum.items():
        out[f"val_{k}"] = v / n_total_positions
    # Backfill val_loss_de from the metrics dict so callers don't need
    # to guess where the ΔE component came from.
    if "val_loss_de" not in out and "loss_de" in metrics_accum:
        out["val_loss_de"] = metrics_accum["loss_de"] / n_total_positions
    return out


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Finetune parquet dir (default: <repo>/data/finetune)")
    p.add_argument("--val-data-dir", type=Path, default=None,
                   help="Optional held-out val dir (default: uses --data-dir "
                        "validation split)")
    p.add_argument("--limit-examples", type=int, default=None)
    p.add_argument("--limit-val-examples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Model architecture (must match pretrained checkpoint)
    p.add_argument("--feature-mode", type=str, default="raw_spectrum")
    p.add_argument("--encoder-hidden", type=int, default=128)
    p.add_argument("--encoder-out", type=int, default=64)
    p.add_argument("--encoder-dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-mode", type=str, default="cross_attn",
                   choices=["cross_attn"],
                   help="finetune requires cross_attn (3-D logits output)")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--slot-encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=1)

    # Optimisation
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Small: 5e-6 for decoder-only (A), 1e-6 for full-model (B)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-fraction", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=1)

    # Freezing
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Experiment A: freeze MaterialEncoder + slot_encoder")

    # Loading + saving
    p.add_argument("--pretrained-checkpoint", type=Path, required=True,
                   help="Path to the pretrained cross_attn checkpoint dir "
                        "(e.g. .../step_13000/ — val-optimal is recommended)")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume a finetune from a previous checkpoint dir")
    p.add_argument("--save-every", type=int, default=500)

    # Optical sim
    p.add_argument("--incidence-angle", type=float, default=0.0)

    # CE anchor loss (see src/de_finetune.py:ce_anchor_loss and
    # analyses/de_finetune/GRADIENT_FLOW_EXPLAINER.md). Default 0.0 =
    # pure ΔE. Values in [0.1, 10] pin the model to the pretrain
    # manifold with progressively stronger weight; needed because the
    # STE gradient's linearization has only ~41% top-1 accuracy at the
    # model's argmax anchor points and drifts the finetune away from
    # the pretrained CE optimum without an anchor.
    p.add_argument("--ce-loss-weight", type=float, default=0.0,
                   help="Weight for the CE anchor in the combined loss "
                        "L = ΔE + λ*CE. 0 disables the anchor (pure ΔE).")

    # Top-K real-sim loss (C1). See src/de_finetune.py:_topK_sim_loss_for_example.
    # 0 = STE mode (original). >0 replaces the STE ΔE primary loss with a
    # listwise CE loss whose target is softmax(-β·ΔE_real) over the top-K
    # candidates per position. Sim cost scales linearly with K.
    p.add_argument("--real-sim-topk", type=int, default=0,
                   help="0=STE mode. K>0 uses top-K real-sim listwise CE loss.")
    p.add_argument("--sim-target-beta", type=float, default=1.0,
                   help="Target-dist sharpness for real-sim CE: "
                        "softmax(-β·ΔE). Higher β = sharper on argmin.")
    p.add_argument("--topk-mode", type=str, default="slot",
                   choices=["slot", "joint", "hierarchical"],
                   help="'slot': top-K over per-slot scores (max-over-thickness), "
                        "each candidate uses its argmax thickness bin. "
                        "'joint': top-K over the flat (slot × thickness) "
                        "grid — Sept 10 finding: concentrates on 1-2 slots' "
                        "neighbor-thickness bins, learning stalls. "
                        "'hierarchical': top-K slots AND top-N thicknesses "
                        "per slot (N via --thickness-topn). Total K·N sims "
                        "per position, distinct (slot, thick) pairs. Only "
                        "used when --real-sim-topk > 0.")
    p.add_argument("--thickness-topn", type=int, default=1,
                   help="Only used with --topk-mode hierarchical. N "
                        "thickness bins per slot in the K·N candidate grid. "
                        "N=1 = equivalent to 'slot' mode.")

    # Sim-feedback residual (Sept 13). At each decoding position, feed
    # the model residual = target - sim(GT prefix) so it knows how far
    # off the prefix already is. Zero-init projection in the model
    # means enabling this on top of a pretrained checkpoint is safe
    # (behaves identically at init, learns to use residual during
    # finetune). Adds ~N-1 partial sims per example (~25% overhead).
    p.add_argument("--sim-feedback", action="store_true",
                   help="Enable sim-feedback residual conditioning "
                        "(finetune-only architectural addition).")

    # LR schedule.
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["cosine", "constant"],
                   help="'cosine' (default) decays LR to 0 by end of training. "
                        "'constant' holds base_lr flat after warmup — better "
                        "for long runs where cosine decay to 0 hurts.")

    # ε-exploration for the top-K real-sim loss. Some fraction of the K
    # candidates per position are drawn uniformly at random from the
    # active grid instead of top-K by logit. Anneals linearly from
    # `--epsilon-start` (early training) to `--epsilon-end` over
    # `--epsilon-decay-fraction · total_steps`, then holds at end. Both
    # 0 = disabled (only top-K by logit). Only used when
    # --real-sim-topk > 0.
    p.add_argument("--epsilon-start", type=float, default=0.0,
                   help="Initial ε-exploration fraction (0 disables).")
    p.add_argument("--epsilon-end", type=float, default=0.0,
                   help="Final ε-exploration fraction after decay.")
    p.add_argument("--epsilon-decay-fraction", type=float, default=1.0,
                   help="Fraction of training over which ε anneals from "
                        "start to end (rest holds at end).")

    # Logging
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--verbose", dest="verbose", action="store_true", default=True)
    p.add_argument("--no-verbose", dest="verbose", action="store_false")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = find_repo_root(Path(__file__).parent)
    if args.data_dir is None:
        args.data_dir = repo_root / "data" / "finetune"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Data dir: {args.data_dir}", flush=True)
    print(f"[INFO] Save dir: {args.save_dir}", flush=True)
    print(f"[INFO] Pretrained ckpt: {args.pretrained_checkpoint}", flush=True)

    # --- Model config + build ---
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
    )
    model = build_model(config).to(device)

    # --- Load pretrained weights ---
    pretrained_state = torch.load(
        args.pretrained_checkpoint / "model.pt",
        map_location=device, weights_only=True,
    )
    # strict=False so the residual_proj module (added in Sept 13 for
    # sim-feedback finetune, zero-init) can be absent from older
    # pretrained checkpoints. Any keys present in state but not in model
    # (unexpected_keys) still surface here for review.
    load_result = model.load_state_dict(pretrained_state, strict=False)
    if load_result.missing_keys:
        print(
            f"[INFO] Loaded pretrained state; missing_keys (using init): "
            f"{load_result.missing_keys}", flush=True,
        )
    if load_result.unexpected_keys:
        print(
            f"[WARN] Loaded pretrained state; unexpected_keys (ignored): "
            f"{load_result.unexpected_keys}", flush=True,
        )
    if not load_result.missing_keys and not load_result.unexpected_keys:
        print(f"[INFO] Loaded pretrained state from {args.pretrained_checkpoint}",
              flush=True)

    # --- Freezing (Experiment A vs B) ---
    if args.freeze_encoder:
        stats = freeze_encoder_for_decoder_only(model)
        print(f"[INFO] Experiment A (decoder-only): "
              f"frozen={stats['frozen']:,} params, "
              f"trainable={stats['trainable']:,} params", flush=True)
    else:
        stats = unfreeze_all(model)
        print(f"[INFO] Experiment B (full-model): "
              f"trainable={stats['trainable']:,} params", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr,
                      weight_decay=args.weight_decay)

    # --- Datasets ---
    train_ds = FlexThinFilmDataset(
        data_prompts_dir=args.data_dir,
        seed=args.seed,
        split="train",
        limit_examples=args.limit_examples,
        streaming=True,
    )
    val_data_dir = args.val_data_dir or args.data_dir
    val_ds = FlexThinFilmDataset(
        data_prompts_dir=val_data_dir,
        seed=args.seed,
        split="validation" if args.val_data_dir is None else "all",
        limit_examples=args.limit_val_examples,
        streaming=True,
    )
    print(f"[INFO] Train examples: {len(train_ds):,}", flush=True)
    print(f"[INFO] Val examples:   {len(val_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=min(2, args.num_workers), pin_memory=True,
    )

    # --- Save-dir bootstrap ---
    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        (args.save_dir / ".write_test").write_text("ok")
        (args.save_dir / ".write_test").unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write to {args.save_dir}: {exc}"
        ) from exc
    history_path = args.save_dir / "history.jsonl"
    print(f"[INFO] History: {history_path.resolve()}", flush=True)

    # --- LR schedule setup ---
    n_examples = len(train_ds)
    steps_per_epoch = math.ceil(n_examples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)
    print(f"[INFO] Steps per epoch: {steps_per_epoch:,}", flush=True)
    print(f"[INFO] Total steps: {total_steps:,}  (warmup={warmup_steps:,})",
          flush=True)

    # --- Resume ---
    global_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if not (resume_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"--resume {resume_dir}: no model.pt found."
            )
        sd_model = getattr(model, "_orig_mod", model)
        sd_model.load_state_dict(torch.load(
            resume_dir / "model.pt", map_location=device, weights_only=True,
        ))
        optimizer.load_state_dict(torch.load(
            resume_dir / "optimizer.pt", map_location=device, weights_only=True,
        ))
        with open(resume_dir / "meta.json") as f:
            meta = json.load(f)
        global_step = int(meta["step"])
        print(f"[Resume] step={global_step:,}", flush=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    ema_step_time = None
    running_loss = 0.0
    running_n = 0
    t_last_log = time.time()

    # Best-val tracking. Every val eval that improves on the running
    # best gets an extra copy saved as `<save_dir>/best/`, so long runs
    # that peak mid-training (Sept 10 K=3 run: best at step 600 of 1665)
    # don't lose the best model. `best_val_loss_de = +inf` at start so
    # the first val always wins.
    best_val_loss_de = float("inf")
    best_step = -1

    for epoch in range(args.epochs):
        for batch in train_loader:
            if global_step >= total_steps:
                break
            step_t0 = time.time()
            lr = _lr_at_step(
                global_step, total_steps, warmup_steps, args.lr,
                schedule=args.lr_schedule,
            )
            epsilon = _epsilon_at_step(
                global_step, total_steps,
                args.epsilon_start, args.epsilon_end,
                args.epsilon_decay_fraction,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = finetune_de_loss(
                model, batch, incidence_angle=args.incidence_angle,
                device=device, ce_loss_weight=args.ce_loss_weight,
                real_sim_topk=args.real_sim_topk,
                sim_target_beta=args.sim_target_beta,
                topk_mode=args.topk_mode,
                epsilon=epsilon,
                thickness_topn=args.thickness_topn,
                sim_feedback=args.sim_feedback,
            )
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {global_step}, skipping",
                      flush=True)
                global_step += 1
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            step_time = time.time() - step_t0
            ema_step_time = (
                step_time if ema_step_time is None
                else 0.9 * ema_step_time + 0.1 * step_time
            )
            # Track loss_de (the target metric) separately from the combined
            # loss the optimizer minimizes, so log lines and running means
            # stay comparable across ce_loss_weight settings.
            running_loss += metrics.get("loss_de", loss.item()) * metrics.get("n_positions", 1)
            running_n += metrics.get("n_positions", 1)

            if args.verbose and (global_step + 1) % args.log_every == 0:
                mean_loss_de = running_loss / max(running_n, 1)
                ex_per_s = metrics.get("batch_size", 0) / max(step_time, 1e-6)
                ce_str = (
                    f"  loss_ce={metrics.get('loss_ce', float('nan')):.3f}"
                    if args.ce_loss_weight > 0 else ""
                )
                eps_str = (
                    f"  eps={epsilon:.2f}"
                    if args.real_sim_topk > 0 and (args.epsilon_start > 0 or args.epsilon_end > 0)
                    else ""
                )
                topk_str = (
                    f"  loss_topk={metrics.get('loss_topk', float('nan')):.3f}"
                    f"  argmin_hit={metrics.get('topk_argmin_matches_model', 0):.2f}"
                    f"{eps_str}"
                    if args.real_sim_topk > 0 else ""
                )
                print(
                    f"[step {global_step + 1:>6}/{total_steps:>6}]  "
                    f"loss_de={mean_loss_de:.3f}{topk_str}{ce_str}  lr={lr:.2e}  "
                    f"slot_ent={metrics.get('slot_entropy', 0):.2f}  "
                    f"thick_ent={metrics.get('thickness_entropy', 0):.2f}  "
                    f"slot_match_gt={metrics.get('slot_match_gt', 0):.2f}  "
                    f"thick_match_gt={metrics.get('thickness_match_gt', 0):.2f}  "
                    f"step={ema_step_time:.2f}s  ex/s={ex_per_s:.1f}",
                    flush=True,
                )
                running_loss = 0.0
                running_n = 0

            # Periodic checkpoint + val
            if (global_step + 1) % args.save_every == 0:
                step_id = global_step + 1
                val_out = evaluate_val(
                    model, val_loader, device,
                    ce_loss_weight=args.ce_loss_weight,
                    sim_feedback=args.sim_feedback,
                )
                # Save the ΔE-only component in checkpoint meta so we can
                # compare across ce_loss_weight settings later.
                loss_de_scalar = float(metrics.get("loss_de", loss.item()))
                save_checkpoint(
                    model, config, optimizer, step_id,
                    loss_de_scalar, args.save_dir / f"step_{step_id}", lr=lr,
                )
                save_checkpoint(
                    model, config, optimizer, step_id,
                    loss_de_scalar, args.save_dir / "latest", lr=lr,
                )

                # Best-val checkpoint: save `best/` whenever val_loss_de
                # improves. Poor-man's early stopping — the final model
                # can degrade past mid-training peak; this preserves it.
                val_de = val_out.get("val_loss_de", float("inf"))
                new_best = False
                if val_de is not None and val_de < best_val_loss_de:
                    best_val_loss_de = float(val_de)
                    best_step = step_id
                    save_checkpoint(
                        model, config, optimizer, step_id,
                        loss_de_scalar, args.save_dir / "best", lr=lr,
                    )
                    new_best = True

                row = {
                    "step": step_id,
                    "epoch": epoch,
                    "lr": lr,
                    "epsilon": epsilon,
                    "ce_loss_weight": args.ce_loss_weight,
                    "real_sim_topk": args.real_sim_topk,
                    "sim_target_beta": args.sim_target_beta,
                    "topk_mode": args.topk_mode,
                    "lr_schedule": args.lr_schedule,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_end": args.epsilon_end,
                    "epsilon_decay_fraction": args.epsilon_decay_fraction,
                    "sim_feedback": args.sim_feedback,
                    "best_val_loss_de": best_val_loss_de,
                    "best_step": best_step,
                    "is_new_best": new_best,
                    "wall_time_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
                    ),
                }
                # metrics carries loss_de / loss_ce / loss_total etc.;
                # prefix with `train_` and let the update populate them.
                row.update({
                    f"train_{k}": v for k, v in metrics.items()
                    if k not in ("batch_size", "n_positions")
                })
                row.update(val_out)
                append_history(history_path, row)
                ce_str = (
                    f"  val_loss_ce={val_out.get('val_loss_ce', float('nan')):.3f}"
                    if args.ce_loss_weight > 0 else ""
                )
                best_str = (
                    f"  best={best_val_loss_de:.3f}@{best_step}"
                    + ("  ★NEW" if new_best else "")
                )
                print(
                    f"[Val]  step={step_id}  "
                    f"val_loss_de={val_out.get('val_loss_de', float('nan')):.3f}"
                    f"{ce_str}  "
                    f"val_slot_match_gt="
                    f"{val_out.get('val_slot_match_gt', float('nan')):.2f}"
                    f"{best_str}",
                    flush=True,
                )

            global_step += 1

        if global_step >= total_steps:
            break

    # Final save. Use loss_de from the last valid metrics if we have it;
    # fall back to the combined loss otherwise.
    final_loss_de = float(
        metrics.get("loss_de", loss.item()) if metrics else loss.item()
    ) if torch.isfinite(loss) else float("nan")
    save_checkpoint(
        model, config, optimizer, global_step,
        final_loss_de,
        args.save_dir / "final", lr=lr,
    )
    save_checkpoint(
        model, config, optimizer, global_step,
        final_loss_de,
        args.save_dir / "latest", lr=lr,
    )
    print(f"[INFO] Finetune complete. Final step: {global_step}", flush=True)
    print(f"[INFO] Latest checkpoint: {args.save_dir / 'latest'}", flush=True)
    if best_step > 0:
        print(
            f"[INFO] Best val_loss_de: {best_val_loss_de:.4f} at step {best_step}  "
            f"(checkpoint: {args.save_dir / 'best'})",
            flush=True,
        )


if __name__ == "__main__":
    main()
