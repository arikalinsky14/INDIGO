#!/usr/bin/env python3
"""Fit the learning-rate law the way Porian et al. 2024 do.

Their protocol (`get_interpolated_hparams_dfs` in the authors' released code):

  1. Sweep the hyperparameter grid exhaustively at every configuration up to
     some scale, rather than at a couple of sizes.
  2. Per configuration, take the optimum as the argmin of an Akima
     interpolation of loss vs the hyperparameter in log-log space, NOT the best
     grid point. A grid of three cannot resolve an optimum; an interpolated one
     can, and it also makes the next check meaningful.
  3. Flag `on_edge` when that argmin falls outside the second and second-to-last
     grid points. Such a sweep did not bracket its own optimum and cannot be
     used to anchor a law.
  4. Fit the power law only over a WINDOW of configurations, then extrapolate
     above it. Fitting across the whole range lets one unbracketed endpoint set
     the slope, which is what happened to INDIGO's existing law.

This script implements 2 to 4 over the JSONs `scripts/lr_tuning.py` writes.
Step 1 is a compute decision: see `slurms/lr_grid.sh`.

Where the grid also varies the dataset size at fixed model size, a two-
dimensional law lr(N, D) = A * N^b * D^c is fitted as well. INDIGO's deployed
law has no D term at all, while the sweep's runs span 9x in D at fixed N, so
that exponent is the one this is really here to measure.

    python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \\
        --min-params 2e5 --max-params 3e6 --target-params 17.8e6
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling.porian import tuned_optimum, power_law_fit    # noqa: E402

METRIC_FIELD = {"delta_e": "final_val_de", "val_loss": "best_val_loss"}


def load_sweeps(results_dir: Path, metric: str) -> List[dict]:
    """One record per lr_tuning.py run: its grid, its interpolated optimum."""
    field = METRIC_FIELD[metric]
    out = []
    for path in sorted(glob.glob(str(results_dir / "lr_search_*.json"))):
        d = json.load(open(path))
        trials = [(t["lr"], t.get(field)) for t in d.get("results", [])]
        trials = [(lr, v) for lr, v in trials if v is not None and np.isfinite(v)]
        if len(trials) < 2:
            print(f"[WARN] {Path(path).name}: {len(trials)} usable trials, skipping")
            continue
        n_params = d.get("n_params")
        if n_params is None:
            print(f"[WARN] {Path(path).name}: no n_params recorded, skipping. "
                  f"Re-run with the current lr_tuning.py, which saves it.")
            continue
        lrs = [t[0] for t in trials]
        best, on_edge = tuned_optimum(lrs, [t[1] for t in trials])
        out.append({
            "file": Path(path).name, "n_params": int(n_params),
            "d_model": d.get("d_model"),
            "slot_encoder_layers": d.get("slot_encoder_layers"),
            "batch_size": d.get("batch_size"),
            "train_examples": d.get("train_examples"),
            "epochs": d.get("epochs"),
            "n_grid": len(trials), "grid_min": min(lrs), "grid_max": max(lrs),
            "lr_grid_argmin": min(trials, key=lambda t: t[1])[0],
            "lr_star": best, "on_edge": bool(on_edge),
        })
    return out


def fit_2d(rows: List[dict]) -> Optional[dict]:
    """lr(N, D) = A * N^b * D^c by least squares in log space."""
    usable = [r for r in rows if not r["on_edge"] and r["train_examples"]]
    if len({r["n_params"] for r in usable}) < 2 or len({r["train_examples"] for r in usable}) < 2:
        return None
    A = np.array([[np.log(r["n_params"]), np.log(r["train_examples"]), 1.0]
                  for r in usable])
    y = np.log([r["lr_star"] for r in usable])
    if A.shape[0] < 4:
        return {"status": f"only {A.shape[0]} bracketed runs, need 4 for a 2-D fit"}
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"status": "ok", "n_exponent": float(coef[0]),
            "d_exponent": float(coef[1]), "coef": float(np.exp(coef[2])),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_points": int(A.shape[0])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--metric", choices=sorted(METRIC_FIELD), default="delta_e")
    p.add_argument("--min-params", type=float, default=None,
                   help="lower edge of the fit window (their min_params_for_fit)")
    p.add_argument("--max-params", type=float, default=None,
                   help="upper edge of the fit window; configs above it are "
                        "held out so the extrapolation can be checked")
    p.add_argument("--target-params", type=float, default=None,
                   help="extrapolate the fitted law to this model size")
    p.add_argument("--output", default="analyses/scaling/results/lr_law_fit.json")
    a = p.parse_args()

    rows = load_sweeps(Path(a.results_dir), a.metric)
    if not rows:
        sys.exit(f"no usable lr_search JSONs under {a.results_dir}")

    print(f"\n{'config':<28} {'N':>10} {'D':>10} {'grid':>5} "
          f"{'grid argmin':>12} {'lr* interp':>11} {'bracketed?':>11}")
    for r in sorted(rows, key=lambda r: r["n_params"]):
        tag = f"d{r['d_model']}/se{r['slot_encoder_layers']}/bs{r['batch_size']}"
        d = f"{r['train_examples']:,}" if r["train_examples"] else "?"
        print(f"{tag:<28} {r['n_params']:>10,} {d:>10} {r['n_grid']:>5} "
              f"{r['lr_grid_argmin']:>12.3e} {r['lr_star']:>11.3e} "
              f"{'no' if r['on_edge'] else 'yes':>11}")

    bracketed = [r for r in rows if not r["on_edge"]]
    print(f"\n{len(bracketed)} of {len(rows)} sweeps bracketed their own optimum.")
    if len(rows) - len(bracketed):
        print("   Unbracketed sweeps are excluded from the fit: their optimum "
              "sits at a grid endpoint, so its true value is unknown. Widen "
              "--lr-min / --lr-max and re-run those configs.")

    window = bracketed
    if a.min_params:
        window = [r for r in window if r["n_params"] >= a.min_params]
    if a.max_params:
        window = [r for r in window if r["n_params"] <= a.max_params]

    result = {"results_dir": a.results_dir, "metric": a.metric, "runs": rows,
              "n_bracketed": len(bracketed),
              "window": {"min_params": a.min_params, "max_params": a.max_params,
                         "n_in_window": len(window)}}

    if len(window) >= 2:
        law = power_law_fit([r["n_params"] for r in window],
                            [r["lr_star"] for r in window])
        print(f"\nlr(N) = {law.coef:.4g} * N^{law.exponent:+.4f}   "
              f"(r2 = {law.r2:.3f}, {law.n_points} configs in window)")
        result["lr_vs_n"] = {"exponent": law.exponent, "coef": law.coef,
                             "r2": law.r2, "n_points": law.n_points}
        held = [r for r in bracketed if r not in window]
        if held:
            print("\nHeld-out configs above the window (the extrapolation test):")
            for r in sorted(held, key=lambda r: r["n_params"]):
                pred = law(r["n_params"])
                print(f"   N={r['n_params']:>10,}  measured {r['lr_star']:.3e}  "
                      f"predicted {pred:.3e}  ratio {r['lr_star']/pred:.2f}x")
            result["held_out"] = [
                {"n_params": r["n_params"], "measured": r["lr_star"],
                 "predicted": float(law(r["n_params"])),
                 "ratio": float(r["lr_star"] / law(r["n_params"]))}
                for r in held]
        if a.target_params:
            print(f"\nExtrapolated to N = {a.target_params:,.0f}: "
                  f"lr = {law(a.target_params):.3e}")
            result["extrapolated"] = {"n_params": a.target_params,
                                      "lr": float(law(a.target_params))}
    else:
        print(f"\n[WARN] only {len(window)} bracketed configs in the window; "
              f"no law fitted")
        result["lr_vs_n"] = {"status": "too few bracketed configs in window"}

    two_d = fit_2d(rows)
    if two_d is None:
        print("\nNo D term fitted: the sweep varies only one of N and D. To "
              "measure it, run lr_tuning.py at one model size over several "
              "--limit-examples (see slurms/lr_grid.sh).")
        result["lr_vs_n_and_d"] = {"status": "D held fixed across the sweep"}
    elif two_d["status"] != "ok":
        print(f"\n2-D fit: {two_d['status']}")
        result["lr_vs_n_and_d"] = two_d
    else:
        print(f"\nlr(N, D) = {two_d['coef']:.4g} * N^{two_d['n_exponent']:+.4f} "
              f"* D^{two_d['d_exponent']:+.4f}   (r2 = {two_d['r2']:.3f}, "
              f"{two_d['n_points']} configs)")
        print(f"   The D exponent is the term the deployed law omits. "
              f"|c| = {abs(two_d['d_exponent']):.3f}")
        result["lr_vs_n_and_d"] = two_d

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.output, "w"), indent=1)
    print(f"\n[INFO] written to {a.output}")


if __name__ == "__main__":
    main()
