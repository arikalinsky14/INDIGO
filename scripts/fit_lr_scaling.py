#!/usr/bin/env python3
"""
Fit a power-law scaling of optimal LR vs training-set size.

Reads the JSON outputs from `scripts/lr_tuning.py` (one per training-set
size N), fits

    log(lr_opt) = a + b * log(N)        ->     lr_opt(N) = exp(a) * N^b

via least squares, and extrapolates to a target N (usually your
production-run size).

Why this is the right thing to fit
----------------------------------
For a single-pass run with AdamW + linear warmup + cosine decay, optimal
LR depends primarily on the total number of optimization steps. Holding
batch size constant and varying the dataset size at epochs=1 gives a
clean (lr_opt, steps) curve. Using multi-epoch runs to vary step count
instead would contaminate the fit, because AdamW dynamics with repeated
data differ from those with fresh data.

Inputs
------
A directory of `lr_search_ep1_lim<N>.json` files produced by
`scripts/lr_tuning.py --limit-examples N --epochs 1`. The script reads
`train_examples` and `optimal_lr` from each.

Outputs are partitioned by head_mode: lr_tuning.py writes into
`outputs/lr_search/<head_mode>/`. Use `--head-mode mlp` or
`--head-mode cross_attn` (or pass `--results-dir` explicitly) so the fit
stays within one architecture.

Examples
--------
    # Generate the inputs (MLP head — default)
    for N in 500000 1000000 2000000; do
        EPOCHS=1 LIMIT_EXAMPLES=${N} sbatch slurms/lr_tuning.sh
    done

    # Fit and extrapolate
    python scripts/fit_lr_scaling.py \
        --head-mode mlp --target-examples 10000000 --plot

    # Same workflow for the cross-attention head (its outputs land in a
    # separate subdir, so fits never mix architectures)
    for N in 500000 1000000 2000000; do
        HEAD_MODE=cross_attn EPOCHS=1 LIMIT_EXAMPLES=${N} sbatch slurms/lr_tuning.sh
    done
    python scripts/fit_lr_scaling.py \
        --head-mode cross_attn --target-examples 10000000 --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_runs(results_dir: Path) -> List[Tuple[int, float, float, dict]]:
    """Return a list of (n_train, lr_opt, best_val_loss, raw_json) tuples."""
    runs = []
    for json_path in sorted(results_dir.glob("lr_search_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if "train_examples" not in data or "optimal_lr" not in data:
            print(f"[WARN] {json_path.name}: missing required fields, skipping")
            continue
        n_train = int(data["train_examples"])
        lr_opt = float(data["optimal_lr"])
        best_val = min(
            float(r["best_val_loss"]) for r in data.get("results", [])
            if float(r["lr"]) == lr_opt
        ) if data.get("results") else float("nan")
        runs.append((n_train, lr_opt, best_val, data))
    return runs


def fit_power_law(n_values: np.ndarray, lr_values: np.ndarray) -> Tuple[float, float]:
    """Fit log(lr) = a + b * log(N). Returns (a, b)."""
    if len(n_values) < 2:
        raise ValueError("Need at least 2 (N, lr_opt) pairs to fit a line")
    log_n = np.log(n_values)
    log_lr = np.log(lr_values)
    b, a = np.polyfit(log_n, log_lr, 1)
    return float(a), float(b)


def predict(a: float, b: float, n: float) -> float:
    return float(np.exp(a) * n ** b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit LR scaling law from multi-N sweeps")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing lr_search_*.json files. If "
                             "omitted, defaults to outputs/lr_search/<head_mode>/.")
    parser.add_argument("--head-mode", type=str, default="mlp",
                        choices=["mlp", "cross_attn"],
                        help="Architecture whose sweeps to fit. Only consulted "
                             "when --results-dir is not given.")
    parser.add_argument("--target-examples", type=int, required=True,
                        help="N for which to predict optimal LR (e.g. 10000000 for "
                             "your 10M-row production run)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a log-log diagnostic plot next to the JSON output")
    parser.add_argument("--output", type=str, default=None,
                        help="Override path for the result JSON")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path("outputs/lr_search") / args.head_mode
    if not results_dir.exists():
        print(f"[ERROR] results-dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(results_dir)
    if len(runs) < 2:
        print(f"[ERROR] Found {len(runs)} valid lr_search_*.json files in "
              f"{results_dir}; need at least 2 to fit.", file=sys.stderr)
        sys.exit(1)
    runs.sort()

    n_values = np.array([r[0] for r in runs], dtype=np.float64)
    lr_values = np.array([r[1] for r in runs], dtype=np.float64)
    val_losses = np.array([r[2] for r in runs], dtype=np.float64)

    print(f"[INFO] {len(runs)} runs loaded from {results_dir}")
    print(f"{'N':>12} {'lr_opt':>10} {'best_val_loss':>14}")
    print("-" * 40)
    for n, lr, vl in zip(n_values, lr_values, val_losses):
        print(f"{int(n):>12,d} {lr:>10.2e} {vl:>14.4f}")

    a, b = fit_power_law(n_values, lr_values)
    target = float(args.target_examples)
    lr_predicted = predict(a, b, target)

    # Residuals on the fit points (in log space).
    log_predicted = np.log(np.array([predict(a, b, n) for n in n_values]))
    log_actual = np.log(lr_values)
    rms_log_residual = float(np.sqrt(np.mean((log_predicted - log_actual) ** 2)))

    print()
    print("=" * 60)
    print("LR SCALING FIT")
    print("=" * 60)
    print(f"  lr_opt(N) = {np.exp(a):.4e} * N^{b:.4f}")
    print(f"  log-space RMS residual on fit points: {rms_log_residual:.4f}")
    print(f"  Predicted lr_opt({int(target):,d}): {lr_predicted:.4e}")
    print("=" * 60)

    output_path = (
        Path(args.output)
        if args.output
        else results_dir / f"lr_scaling_fit_target{int(target)}.json"
    )
    with open(output_path, "w") as f:
        json.dump({
            "fit_form": "lr_opt(N) = exp(a) * N**b",
            "a": a,
            "b": b,
            "log_rms_residual": rms_log_residual,
            "target_examples": int(target),
            "predicted_optimal_lr": lr_predicted,
            "inputs": [
                {"n_train": int(n), "lr_opt": lr, "best_val_loss": vl}
                for n, lr, vl in zip(n_values, lr_values, val_losses)
            ],
        }, f, indent=2)
    print(f"\n[INFO] Fit written to {output_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot")
            return

        n_grid = np.geomspace(n_values.min() / 2, max(n_values.max(), target) * 1.5, 100)
        lr_grid = np.array([predict(a, b, n) for n in n_grid])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.loglog(n_values, lr_values, "bo", markersize=12, label="LR-search results")
        ax.loglog(n_grid, lr_grid, "b-", linewidth=2, alpha=0.7, label="Power-law fit")
        ax.loglog([target], [lr_predicted], "r*", markersize=20,
                  label=f"Predicted at N={int(target):,d}: {lr_predicted:.2e}")
        ax.axvline(target, color="r", linestyle="--", alpha=0.3)
        ax.set_xlabel("Training-set size N (rows)", fontsize=12)
        ax.set_ylabel("Optimal learning rate", fontsize=12)
        ax.set_title(f"LR Scaling: lr_opt(N) = {np.exp(a):.2e} * N^{b:.3f}", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11)
        plot_path = output_path.with_suffix(".png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
