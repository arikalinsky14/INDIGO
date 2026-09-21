#!/usr/bin/env python3
"""
Learning Rate Finder for INDIGO (FlexMaterialMLP / FlexMaterialCrossAttn).

Imports collate / step utilities from `scripts/training.py` so the search
loop is bit-identical to a real training run. Tune LR per head — the
optimum is architecture-dependent, so re-run this sweep whenever you
change `--head-mode`.

For a single-pass production run on N rows, the optimal LR scales as a
power law in N. The recommended workflow is:

1. Run this script multiple times with different `--limit-examples`,
   each at `--epochs 1`. Each run writes a JSON to
   `outputs/lr_search/lr_search_ep1_lim<N>.json`.

      for N in 500000 1000000 2000000; do
          python scripts/lr_tuning.py \
              --data-dir data/train --epochs 1 \
              --limit-examples ${N} --limit-val-examples 10000 \
              --n-lrs 6 --lr-min 1e-5 --lr-max 5e-3
      done

2. Fit log(lr_opt) = a + b * log(N) and extrapolate to your production N:

      python scripts/fit_lr_scaling.py \
          --results-dir outputs/lr_search --target-examples 10000000 --plot

The multi-N approach is more principled than running multiple epochs on a
small subset: AdamW dynamics with repeated examples differ from those
with fresh examples, so the latter contaminates the fit.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from scripts.training import (
    collate_fn,
    collate_fn_packed,
    run_one_epoch,
)
from src.dataset import FlexThinFilmDataset, find_repo_root
from src.model import ModelConfig, build_model, compute_loss, compute_loss_packed
from src.materials_vocab import VOCAB_SIZE
from src.delta_e_eval import (
    evaluate_delta_e,
    primary_metric as delta_e_primary_metric,
    OPTICAL_SIM_AVAILABLE,
)


# A run whose cross-entropy ends above this has diverged, not learned: it is
# worse than predicting uniformly at random over the vocabulary. Derived from
# the vocab rather than hand-tuned, so it travels with the task.
#
# This guard exists because DeltaE CANNOT detect divergence on its own. A
# diverged model still emits some structure, the simulator still colours it,
# and the resulting DeltaE lands in the same range an untrained model
# produces. Observed directly in the d_model=128 and d_model=512 LR sweeps:
# at lr=3e-3 both diverged (val_loss 1426 and inf) yet BOTH reported
# val_de=28.6454 and p95=56.850 -- identical to four decimals from models 23x
# apart in size. Selecting on DeltaE alone picked a diverged model for
# d_model=512.
DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)


@dataclass
class LRSearchResult:
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
    # DeltaE_00 on the held-out slice after the final epoch. This, not
    # val_loss, is what LR selection should key on: the CE/DeltaE decoupling
    # is verified on INDIGO, so the lowest-CE LR is not necessarily the
    # lowest-DeltaE LR. None when the optical simulator is unavailable or
    # DeltaE was disabled.
    final_val_de: Optional[float] = None
    final_val_de_p95: Optional[float] = None
    final_val_de_by_chroma: Optional[dict] = None
    de_result: Optional[dict] = None

    @property
    def diverged(self) -> bool:
        """True when training blew up, whatever DeltaE happens to say."""
        vl = self.best_val_loss
        return (vl is None or not math.isfinite(vl)
                or vl > DIVERGENCE_VAL_LOSS)

    def selection_metric(self, metric: str) -> float:
        """Scalar to minimise. Diverged runs are never selectable.

        The divergence check comes FIRST and applies to both metrics: a
        blown-up run can post a perfectly ordinary-looking DeltaE (see
        DIVERGENCE_VAL_LOSS), so screening on the objective alone would let
        it win.
        """
        if self.diverged:
            return float("inf")
        if metric == "delta_e":
            if self.final_val_de is None:
                return float("inf")
            return self.final_val_de
        if metric == "val_loss":
            return self.best_val_loss
        raise ValueError(f"unknown selection metric {metric!r}")


def evaluate_validation(model, val_loader, device, loss_fn=compute_loss) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            losses = loss_fn(model, batch_on_device)
            # Token-weighted; `n_correct` is exact. See the note in
            # src/model.py:compute_loss_packed. This matters more here than
            # elsewhere: the LR sweep's short runs sit at low accuracy, where
            # the old int(accuracy * batch_size) truncation collapsed to 0.0.
            n_tok = int(losses["n_tokens"].item())
            total_loss += losses["loss"].item() * n_tok
            total_correct += int(losses["n_correct"].item())
            total_tokens += n_tok
    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


def train_with_lr(
    lr: float,
    epochs: int,
    train_dataset: FlexThinFilmDataset,
    val_dataset: FlexThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    log_every: int = 100,
    verbose: bool = True,
    packed_tf: bool = False,
    bf16: bool = False,
    de_examples: Optional[List] = None,
    de_limit: int = 0,
    de_simulator=None,
) -> LRSearchResult:
    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    active_collate = collate_fn_packed if packed_tf else collate_fn
    loss_fn = compute_loss_packed if packed_tf else compute_loss
    loader_kw = {}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=active_collate,
        num_workers=num_workers, pin_memory=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=active_collate,
        num_workers=num_workers, pin_memory=True, **loader_kw,
    )

    n_examples = len(train_dataset)
    steps_per_epoch = math.ceil(n_examples / batch_size)
    total_steps = steps_per_epoch * epochs

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    global_step = 0

    for epoch in range(epochs):
        epoch_out = run_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            total_steps=total_steps,
            base_lr=lr,
            warmup_fraction=warmup_fraction,
            grad_clip=grad_clip,
            log_every=log_every,
            verbose=verbose,
            global_step_start=global_step,
            loss_fn=loss_fn,
            bf16=bf16,
        )
        global_step = epoch_out["global_step"]
        avg_train_loss = epoch_out["avg_loss"]
        train_losses.append(avg_train_loss)

        val_loss, val_acc = evaluate_validation(model, val_loader, device, loss_fn=loss_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1

        if verbose:
            warmup_steps = int(total_steps * warmup_fraction)
            phase = "warmup" if global_step <= warmup_steps else "decay"
            print(
                f"    Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                f"lr={epoch_out['final_lr']:.2e} [{phase}]",
                flush=True,
            )

    # DeltaE on the final weights. Run once per LR rather than per epoch:
    # it is ~100x more expensive per example than CE, and what we need is a
    # single comparable number per LR.
    de_result = None
    if de_examples and de_limit > 0:
        de_result = evaluate_delta_e(
            model, de_examples, device, limit=de_limit,
            simulator=de_simulator,
        )
        if verbose and de_result.get("n_scored"):
            print(f"    dE: median={de_result['delta_e_median']:.3f} "
                  f"p95={de_result['delta_e_p95']:.3f} "
                  f"(n={de_result['n_scored']}, "
                  f"valid={de_result['valid_rate']:.2f})", flush=True)

    scored = bool(de_result and de_result.get("n_scored"))
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
        final_val_de=de_result["delta_e_median"] if scored else None,
        final_val_de_p95=de_result["delta_e_p95"] if scored else None,
        final_val_de_by_chroma=de_result.get("by_chroma") if scored else None,
        de_result=de_result,
    )


def lr_tuning(
    epochs: int,
    train_dataset: FlexThinFilmDataset,
    val_dataset: FlexThinFilmDataset,
    config: ModelConfig,
    device: torch.device,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    n_lrs: int = 8,
    batch_size: int = 64,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_fraction: float = 0.02,
    log_every: int = 100,
    verbose: bool = True,
    packed_tf: bool = False,
    bf16: bool = False,
    de_examples: Optional[List] = None,
    de_limit: int = 0,
    de_simulator=None,
    selection_metric: str = "delta_e",
) -> Tuple[float, List[LRSearchResult]]:
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), n_lrs)
    print(f"\n{'=' * 70}")
    print(f"LEARNING RATE SEARCH - {epochs} Epochs")
    print(f"{'=' * 70}")
    print(f"LR range: {lr_min:.0e} to {lr_max:.0e}")
    print(f"Testing {n_lrs} values: {[f'{lr:.2e}' for lr in lrs]}")
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")
    print(f"Warmup fraction: {warmup_fraction:.1%}")
    print(f"Selection metric: {selection_metric}"
          + (f" (dE on {de_limit} examples)" if selection_metric == "delta_e"
             and de_limit else ""))
    print(f"{'=' * 70}\n")

    results: List[LRSearchResult] = []
    for i, lr in enumerate(lrs):
        print(f"[{i + 1}/{n_lrs}] Training with LR = {lr:.2e}")
        result = train_with_lr(
            lr=lr, epochs=epochs,
            train_dataset=train_dataset, val_dataset=val_dataset,
            config=config, device=device,
            batch_size=batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            weight_decay=weight_decay, grad_clip=grad_clip,
            warmup_fraction=warmup_fraction,
            log_every=log_every, verbose=verbose,
            packed_tf=packed_tf, bf16=bf16,
            de_examples=de_examples, de_limit=de_limit,
            de_simulator=de_simulator,
        )
        results.append(result)
        de_str = ("" if result.final_val_de is None
                  else f", val_de={result.final_val_de:.3f}")
        print(f"    Final: train_loss={result.final_train_loss:.4f}, "
              f"val_loss={result.final_val_loss:.4f}, "
              f"best_val_loss={result.best_val_loss:.4f}{de_str}\n")

    # Select on DeltaE when we have it. If DeltaE was requested but no LR
    # produced a scorable result (simulator missing, or every generation
    # invalid at every LR), fall back to val_loss rather than returning an
    # arbitrary LR -- and say so, loudly, because a silent fallback to the
    # wrong metric is exactly the failure this plumbing exists to prevent.
    n_div = sum(1 for r in results if r.diverged)
    if n_div:
        print(f"[WARN] {n_div} of {len(results)} LR(s) DIVERGED (val_loss above "
              f"{DIVERGENCE_VAL_LOSS:.1f}, i.e. worse than uniform-random over "
              f"the vocabulary) and are excluded from selection. A diverged run "
              f"can still post an ordinary-looking DeltaE, so this screen is on "
              f"cross-entropy, not on the objective.", flush=True)
    if all(r.diverged for r in results):
        print("[ERROR] EVERY LR diverged. The grid is entirely too high -- "
              "lower --lr-max and re-run. Returning the smallest LR so the "
              "caller has something, but it is NOT a tuned value.", flush=True)
        fallback = min(results, key=lambda r: r.lr)
        return fallback.lr, results

    effective_metric = selection_metric
    if selection_metric == "delta_e" and all(
            r.final_val_de is None for r in results if not r.diverged):
        print("[WARN] DeltaE selection requested but no LR produced a scorable "
              "DeltaE (optical sim unavailable, or every generation invalid). "
              "FALLING BACK to val_loss selection. The chosen LR optimises "
              "cross-entropy, which is NOT a reliable proxy for DeltaE on "
              "INDIGO -- treat the result with suspicion.", flush=True)
        effective_metric = "val_loss"

    best_result = min(results, key=lambda r: r.selection_metric(effective_metric))
    print(f"[INFO] Selected LR={best_result.lr:.3e} by {effective_metric}")
    return best_result.lr, results


def plot_lr_search(results: List[LRSearchResult], output_path: Path, epochs: int) -> None:
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
    ax1 = axes[0]
    ax1.semilogx(lrs, best_val_losses, "b-o", label="Best Val Loss", linewidth=2, markersize=8)
    ax1.semilogx(lrs, final_val_losses, "r--s", label="Final Val Loss", linewidth=1.5, markersize=6)
    ax1.semilogx(lrs, final_train_losses, "g:^", label="Final Train Loss", linewidth=1.5, markersize=6)
    best_idx = int(np.argmin(best_val_losses))
    ax1.axvline(x=lrs[best_idx], color="blue", linestyle="--", alpha=0.5)
    ax1.scatter([lrs[best_idx]], [best_val_losses[best_idx]], color="blue", s=150,
                zorder=5, marker="*", label=f"Optimal LR: {lrs[best_idx]:.2e}")
    ax1.set_xlabel("Learning Rate", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"LR Search Results ({epochs} Epochs)", fontsize=14)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    best_result = results[best_idx]
    ax2 = axes[1]
    epoch_range = range(1, epochs + 1)
    ax2.plot(epoch_range, best_result.train_losses, "b-", label="Train Loss", linewidth=2)
    ax2.plot(epoch_range, best_result.val_losses, "r-", label="Val Loss", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title(f"Training Curves (LR={best_result.lr:.2e})", fontsize=14)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal LR for INDIGO")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Subsetting: limit how many rows of the train / validation pool the LR
    # sweep uses. Set --limit-examples for the multi-N scaling-law workflow
    # (see module docstring).
    parser.add_argument("--limit-examples", type=int, default=None,
                        help="Limit training rows used by the LR sweep")
    parser.add_argument("--limit-val-examples", type=int, default=None,
                        help="Limit validation rows used by the LR sweep")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream dataset shard-by-shard (recommended at "
                             "production scale; the legacy mode OOMs).")

    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--n-lrs", type=int, default=8)

    parser.add_argument("--feature-mode", type=str, default="raw_spectrum",
                        choices=["raw_spectrum", "compact"])
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=8)
    # Keep aligned with scripts/training.py: optimal LR depends on regularisation.
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-mode", type=str, default="mlp",
                        choices=["mlp", "cross_attn"],
                        help="Must match the architecture you plan to train; "
                             "optimal LR is head-dependent.")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Attention heads (cross_attn only).")
    parser.add_argument("--slot-encoder-layers", type=int, default=0,
                        help="Slot encoder depth (cross_attn only; 0=use n-layers).")
    parser.add_argument("--decoder-layers", type=int, default=1,
                        help="Decoder depth (cross_attn only).")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False,
                        help="bf16 autocast (~2x speedup on L40s/H100).")
    parser.add_argument("--packed-tf", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Packed teacher-forcing (cross_attn only). Default: "
                             "on for cross_attn, off for mlp.")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="DataLoader prefetch_factor (default: 1; matches "
                             "training.py — pipeline is producer-bound)")
    # Keep aligned with scripts/training.py: optimal LR is regulariser-sensitive.
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    # -- DeltaE-based LR selection. Porian et al. correction #3 says re-tune
    # LR per scale; on INDIGO that has to be done against DeltaE, because CE
    # and DeltaE are decoupled -- the lowest-CE LR is not necessarily the
    # lowest-DeltaE one, and DeltaE is what the scaling study fits.
    parser.add_argument("--limit-de-examples", type=int, default=256,
                        help="Examples for the per-LR DeltaE_00 eval "
                             "(default: 256). Run once per LR on the final "
                             "weights. 0 disables, which forces val_loss "
                             "selection.")
    parser.add_argument("--selection-metric", type=str, default="delta_e",
                        choices=["delta_e", "val_loss"],
                        help="Metric the optimal LR is chosen by. Default "
                             "delta_e. 'val_loss' is offered for "
                             "reproducing pre-DeltaE results only -- CE is "
                             "not a reliable proxy for DeltaE on INDIGO.")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")

    # Logging
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print per-step loss every N optimizer steps "
                             "within each LR trial (default: 100)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print per-step loss + per-epoch val (default: on). "
                             "Pass --no-verbose to silence.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    data_dir = Path(args.data_dir)
    if args.limit_examples is None:
        print(f"[INFO] Loading FULL training data...")
    else:
        print(f"[INFO] Loading training data (limited to {args.limit_examples} rows)...")
    train_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="train", verbose=True,
        limit_examples=args.limit_examples, streaming=args.streaming,
    )
    print(f"[INFO] Loading validation data...")
    val_dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split="validation", verbose=True,
        limit_examples=args.limit_val_examples, streaming=args.streaming,
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
    )

    # Default packed_tf to head_mode == 'cross_attn' if not set.
    packed_tf = args.packed_tf if args.packed_tf is not None else (args.head_mode == "cross_attn")
    if packed_tf and args.head_mode == "mlp":
        raise ValueError("--packed-tf is incompatible with --head-mode mlp")
    print(f"[INFO] Model config: head_mode={args.head_mode}, "
          f"d_model={args.d_model}, n_layers={args.n_layers}, "
          f"packed_tf={packed_tf}, bf16={args.bf16}")

    # DeltaE slice, read once and shared across every LR so all LRs are
    # scored on identical examples. One simulator instance keeps
    # jaxlayerlumos' per-stack-depth trace cache warm across the sweep.
    de_examples = None
    de_simulator = None
    de_limit = args.limit_de_examples if args.selection_metric == "delta_e" else 0
    if de_limit > 0:
        if not OPTICAL_SIM_AVAILABLE:
            print("[WARN] --selection-metric delta_e requested but the optical "
                  "simulator is unavailable (jaxlayerlumos missing). LR "
                  "selection will fall back to val_loss.", flush=True)
            de_limit = 0
        else:
            de_examples = list(val_dataset)[:de_limit]
            from src.optical_sim import OpticalSimulator
            de_simulator = OpticalSimulator(incidence_angle=0)
            print(f"[INFO] DeltaE selection slice: {len(de_examples):,} examples "
                  f"(greedy), scored once per LR", flush=True)

    optimal_lr, results = lr_tuning(
        epochs=args.epochs,
        train_dataset=train_dataset, val_dataset=val_dataset,
        config=config, device=device,
        lr_min=args.lr_min, lr_max=args.lr_max, n_lrs=args.n_lrs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
        log_every=args.log_every, verbose=args.verbose,
        packed_tf=packed_tf, bf16=args.bf16,
        de_examples=de_examples, de_limit=de_limit,
        de_simulator=de_simulator,
        selection_metric=args.selection_metric,
    )

    print("\n" + "=" * 70)
    print("LR SEARCH RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'LR':>12} | {'val_de (sel)':>13} | {'dE p95':>8} | "
          f"{'Best Val Loss':>14} | {'Final Train':>12}")
    print("-" * 74)
    for r in sorted(results, key=lambda x: x.lr):
        marker = " *" if r.lr == optimal_lr else ""
        de_s = "n/a" if r.final_val_de is None else f"{r.final_val_de:.4f}"
        p95_s = "n/a" if r.final_val_de_p95 is None else f"{r.final_val_de_p95:.3f}"
        flag = "  DIVERGED (excluded)" if r.diverged else ""
        print(f"{r.lr:>12.2e} | {de_s:>13} | {p95_s:>8} | "
              f"{r.best_val_loss:>14.4f} | {r.final_train_loss:>12.4f}{marker}{flag}")
    print("-" * 74)
    print(f"\n* OPTIMAL LR for {args.epochs} epochs: {optimal_lr:.2e}")
    best_result = next(r for r in results if r.lr == optimal_lr)
    if best_result.final_val_de is not None:
        print(f"  Selected on val_de (median): {best_result.final_val_de:.4f}")
        by_c = best_result.final_val_de_by_chroma or {}
        parts = [f"{b}={by_c[b]['median']:.3f}" for b in ("low", "mid", "high")
                 if by_c.get(b, {}).get("n")]
        if parts:
            print(f"  Per-chroma median: {'  '.join(parts)}")
    print(f"  Best validation loss (diagnostic): {best_result.best_val_loss:.4f} "
          f"(epoch {best_result.best_val_epoch})")
    # An explicit disagreement check: if CE would have picked a different LR,
    # that is the decoupling showing up in this very sweep, and worth
    # recording in the log rather than leaving implicit.
    ce_pick = min(results, key=lambda r: r.best_val_loss).lr
    if best_result.final_val_de is not None and ce_pick != optimal_lr:
        print(f"  NOTE: val_loss would have selected LR={ce_pick:.3e} instead "
              f"-- CE/DeltaE disagree on this sweep.")
    print("=" * 70)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Partition by head_mode so MLP and cross_attn runs don't share files —
        # fit_lr_scaling.py reads one head's results at a time.
        try:
            repo_root = find_repo_root()
            output_dir = repo_root / "outputs" / "lr_search" / args.head_mode
        except Exception:
            output_dir = Path("./outputs/lr_search") / args.head_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag output filenames with the train-subset size so multi-N runs
    # (the scaling-law workflow) don't clobber each other.
    n_train = len(train_dataset)
    tag = f"ep{args.epochs}_lim{n_train}"
    results_file = output_dir / f"lr_search_{tag}.json"
    with open(results_file, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "optimal_lr": optimal_lr,
            "selection_metric": args.selection_metric,
            "optimal_val_de": best_result.final_val_de,
            "optimal_val_de_p95": best_result.final_val_de_p95,
            "optimal_val_de_by_chroma": best_result.final_val_de_by_chroma,
            "limit_de_examples": de_limit,
            "val_loss_would_pick_lr": min(
                results, key=lambda r: r.best_val_loss).lr,
            "lr_range": [args.lr_min, args.lr_max],
            "n_lrs": args.n_lrs,
            "head_mode": args.head_mode,
            "n_heads": args.n_heads,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "warmup_fraction": args.warmup_fraction,
            "train_examples": n_train,
            "val_examples": len(val_dataset),
            "limit_examples": args.limit_examples,
            "limit_val_examples": args.limit_val_examples,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\n[INFO] Results saved to {results_file}")

    if args.plot:
        plot_path = output_dir / f"lr_search_{tag}.png"
        plot_lr_search(results, plot_path, args.epochs)

    print(f"\n# Machine-readable output for log-log fitting:")
    print(f"EPOCHS={args.epochs}")
    print(f"OPTIMAL_LR={optimal_lr}")
    print(f"BEST_VAL_LOSS={best_result.best_val_loss}")
    print(f"SELECTION_METRIC={args.selection_metric}")
    if best_result.final_val_de is not None:
        print(f"OPTIMAL_VAL_DE={best_result.final_val_de}")


if __name__ == "__main__":
    main()
