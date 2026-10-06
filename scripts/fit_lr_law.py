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
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling.porian import (nested_hparam_optimum,          # noqa: E402
                                power_law_fit, tuned_optimum)

METRIC_FIELD = {"delta_e": "final_val_de", "val_loss": "best_val_loss"}


def load_cells(results_dir: Path, metric: str,
               beta2: Optional[float] = None) -> List[dict]:
    """Every (N, D, bs, beta2, lr) cell the sweep ran, one row per trial.

    Only shard-aligned results, and with `beta2` only that beta2's: the
    directory also holds leftovers from before the stages had their own
    directories (the first stage-1 submission ran unaligned, at three beta2
    values and three rates), and those are not evidence for this law.
    """
    field = METRIC_FIELD[metric]
    out = []
    for path in sorted(glob.glob(str(results_dir / "lr_search_*.json"))):
        d = json.load(open(path))
        if not d.get("limit_shard_aligned"):
            print(f"[WARN] {Path(path).name}: not shard-aligned, skipped")
            continue
        if beta2 is not None and abs(float(d.get("beta2", 0.999)) - beta2) > 1e-12:
            print(f"[WARN] {Path(path).name}: beta2 {d.get('beta2')}, not "
                  f"{beta2:g}; skipped")
            continue
        if d.get("n_params") is None:
            print(f"[WARN] {Path(path).name}: no n_params recorded, skipping. "
                  f"Re-run with the current lr_tuning.py, which saves it.")
            continue
        for t in d.get("results", []):
            v = t.get(field)
            if v is None or not np.isfinite(v):
                continue
            out.append({
                "n_params": int(d["n_params"]), "lr": float(t["lr"]),
                "batch_size": float(t.get("batch_size", d.get("batch_size"))),
                "beta2": float(t.get("beta2", d.get("beta2", 0.999))),
                "train_examples": d.get("train_examples"),
                "d_model": d.get("d_model"),
                "slot_encoder_layers": d.get("slot_encoder_layers"),
                "value": float(v), "file": Path(path).name,
            })
    return out


def tune_per_config(cells: List[dict],
                    batch_size_swept: bool = False) -> List[dict]:
    """Nested minimisation per (model size, dataset size), as they do."""
    groups = {}
    for c in cells:
        groups.setdefault((c["n_params"], c["train_examples"]), []).append(c)
    rows = []
    for (n, D), group in sorted(groups.items()):
        try:
            o = nested_hparam_optimum(group, n,
                                     batch_size_swept=batch_size_swept)
        except ValueError as exc:
            print(f"[WARN] N={n:,} D={D}: {exc}")
            continue
        rows.append({
            "n_params": n, "train_examples": D,
            "d_model": group[0]["d_model"],
            "slot_encoder_layers": group[0]["slot_encoder_layers"],
            "lr_star": o.lr, "bs_star": o.batch_size, "beta2_star": o.beta2,
            "value": o.value, "lr_on_edge": o.lr_on_edge,
            "bs_on_edge": o.bs_on_edge, "usable": o.usable,
            "n_cells": o.n_cells,
            "n_batch_sizes": len({c["batch_size"] for c in group}),
            "n_lrs": len({c["lr"] for c in group}),
            "n_beta2": len({c["beta2"] for c in group}),
        })
    return rows


def fit_2d(rows: List[dict]) -> dict:
    """lr(N, D) = A * N^b * D^c by least squares in log space.

    Only configurations that bracketed on both axes are used, since an
    endpoint optimum carries no information about where the true one is.
    """
    usable = [r for r in rows if r["usable"] and r["train_examples"]]
    n_axis = len({r["n_params"] for r in usable})
    d_axis = len({r["train_examples"] for r in usable})
    if n_axis < 2 or d_axis < 2:
        varies = len({r["n_params"] for r in rows}) > 1, len(
            {r["train_examples"] for r in rows}) > 1
        if all(varies):
            return {"status": (f"the grid varies both axes, but only {n_axis} "
                               f"model size(s) and {d_axis} dataset size(s) "
                               f"bracketed; widen the unbracketed cells")}
        return {"status": "the grid varies only one of N and D"}
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


def rung_of(rows: List[dict], sweep_fit_path: str) -> List[Optional[float]]:
    """The IsoFLOP budget each tuned configuration belongs to, matched on
    (N, D) against the sweep it re-runs; None when it matches no sweep point."""
    runs = json.load(open(sweep_fit_path))["runs"]
    budgets = sorted(json.load(open(sweep_fit_path))["budgets"])
    out = []
    for r in rows:
        hit = [x for x in runs if x["n_params"] == r["n_params"]
               and r["train_examples"]
               and abs(x["passes"] - r["train_examples"]) <= 0.01 * x["passes"]]
        out.append(min(budgets, key=lambda b: abs(b - hit[0]["flops"]))
                   if hit else None)
    return out


def leave_one_rung_out(rows: List[dict], sweep_fit_path: str) -> dict:
    """Fit lr(N, D) without one rung, predict that rung's tuned optima.

    The held-out TOP rung is the extrapolation test that matters: it is what
    stage 3 does to the rungs above, done here on a rung whose answer was
    measured. Lower rungs held out are interpolation checks. A law that
    misses its own top rung will miss the rungs above it by at least as much.
    """
    usable = [r for r in rows if r["usable"] and r["train_examples"]]
    rung = dict(zip(map(id, usable), rung_of(usable, sweep_fit_path)))
    budgets = sorted({b for b in rung.values() if b is not None})
    out = {"rungs": []}
    if len(budgets) < 3:
        out["status"] = (f"{len(budgets)} tuned rung(s); holding one out needs "
                         f"at least 3 (two to fit N and M apart)")
        return out
    for b in budgets:
        train = [r for r in usable if rung[id(r)] not in (None, b)]
        test = [r for r in usable if rung[id(r)] == b]
        law = fit_2d(train)
        if law["status"] != "ok" or not test:
            out["rungs"].append({"budget": b, "status": law["status"]})
            continue
        ratios = [r["lr_star"] / (law["coef"] * r["n_params"] ** law["n_exponent"]
                                  * r["train_examples"] ** law["d_exponent"])
                  for r in test]
        out["rungs"].append({
            "budget": b, "held_out": "top (extrapolation)" if b == budgets[-1]
            else "interior", "n_points": len(test),
            "measured_over_predicted": ratios,
            "median_ratio": float(np.median(ratios)),
            "worst_ratio": float(max(ratios, key=lambda x: abs(np.log(x)))),
            "law_without_it": {k: law[k] for k in ("coef", "n_exponent",
                                                   "d_exponent")}})
    out["status"] = "ok"
    return out


def report_coverage(law_coef: float, law_exp: float, window: List[dict],
                    fit_json: str) -> dict:
    """Which of the sweep's rungs does the fitted law actually cover?

    A power law fitted over one range and applied outside it is an
    extrapolation, and the sweep's SMALLEST rungs are both the most sensitive
    to learning rate and the furthest from where anyone usually tunes. This
    lists every distinct model size the sweep trains and says whether the law
    interpolates for it or reaches past the tuned range, and by how far.
    """
    try:
        with open(fit_json) as fh:
            sweep = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"\n[WARN] could not read {fit_json} for the coverage check: {exc}")
        return {"status": f"unreadable: {exc}"}

    sizes = sorted({int(r["n_params"]) for r in sweep.get("runs", [])})
    if not sizes:
        return {"status": "no runs in the fit artifact"}
    lo = min(r["n_params"] for r in window)
    hi = max(r["n_params"] for r in window)

    print(f"\nCoverage against the sweep's ladder "
          f"({len(sizes)} distinct sizes, tuned range {lo:,} to {hi:,}):")
    rows, n_extrap = [], 0
    for n in sizes:
        if n < lo:
            where, factor = "below", lo / n
        elif n > hi:
            where, factor = "above", n / hi
        else:
            where, factor = "covered", 1.0
        if where != "covered":
            n_extrap += 1
        rows.append({"n_params": n, "coverage": where, "factor": factor,
                     "lr": float(law_coef * n ** law_exp)})
    for r in rows:
        mark = ("covered" if r["coverage"] == "covered"
                else f"{r['coverage']} by {r['factor']:.1f}x")
        print(f"   N={r['n_params']:>10,}   lr = {r['lr']:.3e}   {mark}")
    if n_extrap:
        print(f"\n   {n_extrap} of {len(sizes)} sizes fall outside the tuned "
              f"range. Add those sizes to STAGE=2's N_LADDER and re-run rather "
              f"than trusting the extrapolation; the small rungs are where the "
              f"learning rate matters most.")
    else:
        print(f"\n   Every size the sweep trains is inside the tuned range. "
              f"The law interpolates everywhere and extrapolates nowhere.")
    return {"status": "ok", "n_sizes": len(sizes), "n_extrapolated": n_extrap,
            "tuned_min": lo, "tuned_max": hi, "sizes": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--metric", choices=sorted(METRIC_FIELD), default="delta_e")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="fit only results at this beta2 (stage 2 ran 0.999); "
                        "pass a negative value to keep every beta2")
    p.add_argument("--min-params", type=float, default=None,
                   help="lower edge of the fit window (their min_params_for_fit)")
    p.add_argument("--max-params", type=float, default=None,
                   help="upper edge of the fit window; configs above it are "
                        "held out so the extrapolation can be checked")
    p.add_argument("--target-params", type=float, default=None,
                   help="extrapolate the fitted law to this model size")
    p.add_argument("--coverage-from", default=None,
                   help="a fit_scaling.py artifact. Reports, for every model "
                        "size the sweep trains, whether the fitted law "
                        "interpolates or extrapolates to it.")
    p.add_argument("--batch-size-swept", action="store_true",
                   help="treat batch size as a tuned axis. Off by default: on "
                        "INDIGO it is a VRAM decision held fixed at "
                        "configs.DEFAULT_BATCH_SIZE, so a single value is "
                        "deliberate rather than an axis that failed to "
                        "bracket.")
    p.add_argument("--output", default="analyses/scaling/results/lr_law_fit.json")
    a = p.parse_args()

    cells = load_cells(Path(a.results_dir), a.metric,
                       a.beta2 if a.beta2 >= 0 else None)
    if not cells:
        sys.exit(f"no usable lr_search JSONs under {a.results_dir}")
    rows = tune_per_config(cells, batch_size_swept=a.batch_size_swept)
    if not rows:
        sys.exit("no configuration produced a tuned optimum")

    print(f"\n{len(cells)} cells across {len(rows)} configurations "
          f"(model size x dataset size).\n")
    print(f"{'config':<22} {'N':>10} {'D':>11} {'cells':>6} "
          f"{'lr*':>10} {'bs*':>7} {'beta2':>6} {'bracketed':>10}")
    for r in sorted(rows, key=lambda r: (r["n_params"], r["train_examples"] or 0)):
        tag = f"d{r['d_model']}/se{r['slot_encoder_layers']}"
        D = f"{r['train_examples']:,}" if r["train_examples"] else "?"
        flag = ("yes" if r["usable"] else
                ",".join(x for x, on in (("lr", r["lr_on_edge"]),
                                         ("bs", r["bs_on_edge"])) if on) + " edge")
        b2 = f"{r['beta2_star']:.3g}" if r["beta2_star"] is not None else "-"
        print(f"{tag:<22} {r['n_params']:>10,} {D:>11} {r['n_cells']:>6} "
              f"{r['lr_star']:>10.3e} {r['bs_star']:>7.1f} {b2:>6} {flag:>10}")

    usable = [r for r in rows if r["usable"]]
    print(f"\n{len(usable)} of {len(rows)} configurations bracketed on both axes.")
    if len(rows) - len(usable):
        print("   The rest are excluded: an optimum at a grid endpoint is a "
              "wall the sweep hit, not a minimum it found. Widen that axis "
              "for those cells and re-run.")
    if usable:
        b2s = collections.Counter(r["beta2_star"] for r in usable
                                  if r["beta2_star"] is not None)
        if b2s:
            picked = ", ".join(f"{v:g} ({n})" for v, n in b2s.most_common())
            print(f"\n   beta2 chosen per configuration: {picked}")
            if len(b2s) > 1:
                print("   It moves across configurations, so a single global "
                      "value is a scale-dependent handicap.")

    window = usable
    if a.min_params:
        window = [r for r in window if r["n_params"] >= a.min_params]
    if a.max_params:
        window = [r for r in window if r["n_params"] <= a.max_params]

    result = {"results_dir": a.results_dir, "metric": a.metric,
              "n_cells": len(cells), "configs": rows,
              "n_usable": len(usable),
              "window": {"min_params": a.min_params, "max_params": a.max_params,
                         "n_in_window": len(window)}}

    if len(window) >= 2:
        keys = [("lr_star", "lr")] + ([("bs_star", "bs")] if a.batch_size_swept
                                      else [])
        if not a.batch_size_swept:
            held = sorted({r["bs_star"] for r in window})
            print(f"\nBatch size held fixed at {', '.join(f'{b:g}' for b in held)}"
                  f" (VRAM decision, not tuned). No bs(N) law is fitted.")
            result["bs_vs_n"] = {"status": "held fixed by design",
                                 "values": [float(b) for b in held]}
        for key, name in keys:
            if len({r[key] for r in window}) < 2:
                print(f"\n[WARN] {name} is constant across the window; no law fitted")
                result[f"{name}_vs_n"] = {"status": "constant across the window"}
                continue
            law = power_law_fit([r["n_params"] for r in window],
                                [r[key] for r in window])
            print(f"\n{name}(N) = {law.coef:.4g} * N^{law.exponent:+.4f}   "
                  f"(r2 = {law.r2:.3f}, {law.n_points} configs in window)")
            result[f"{name}_vs_n"] = {"exponent": law.exponent, "coef": law.coef,
                                      "r2": law.r2, "n_points": law.n_points}
            held = [r for r in usable if r not in window]
            if held:
                print(f"   held out above the window:")
                for r in sorted(held, key=lambda r: r["n_params"]):
                    pred = law(r["n_params"])
                    print(f"      N={r['n_params']:>10,}  measured {r[key]:.3e}  "
                          f"predicted {pred:.3e}  ratio {r[key]/pred:.2f}x")
            if a.target_params:
                print(f"   extrapolated to N = {a.target_params:,.0f}: "
                      f"{law(a.target_params):.4g}")
                result.setdefault("extrapolated", {})[name] = float(law(a.target_params))
            if name == "lr" and a.coverage_from:
                result["coverage"] = report_coverage(
                    law.coef, law.exponent, window, a.coverage_from)
    else:
        print(f"\n[WARN] only {len(window)} bracketed configs in the window; "
              f"no laws fitted")
        result["lr_vs_n"] = {"status": "too few bracketed configs in window"}

    two_d = fit_2d(rows)
    if two_d["status"] != "ok":
        print(f"\nNo D term fitted: {two_d['status']}. To measure it, run "
              f"lr_tuning.py at one model size over several --limit-examples "
              f"(see slurms/lr_grid.sh).")
        result["lr_vs_n_and_d"] = two_d
    else:
        print(f"\nlr(N, D) = {two_d['coef']:.4g} * N^{two_d['n_exponent']:+.4f} "
              f"* D^{two_d['d_exponent']:+.4f}   (r2 = {two_d['r2']:.3f}, "
              f"{two_d['n_points']} configs)")
        print(f"   The D exponent is the term the deployed law omits. "
              f"|c| = {abs(two_d['d_exponent']):.3f}")
        result["lr_vs_n_and_d"] = two_d

    if a.coverage_from:
        loro = leave_one_rung_out(rows, a.coverage_from)
        result["leave_one_rung_out"] = loro
        print("\nLeave one rung out: fit lr(N, D) without a rung, predict its "
              "tuned optima (measured / predicted).")
        if loro["status"] != "ok":
            print(f"   not run: {loro['status']}")
        for r in loro["rungs"]:
            if "median_ratio" not in r:
                print(f"   C={r['budget']:.2e}: {r['status']}")
                continue
            print(f"   C={r['budget']:.2e} ({r['held_out']}, {r['n_points']} "
                  f"points): median {r['median_ratio']:.2f}x, worst "
                  f"{r['worst_ratio']:.2f}x")
        top = [r for r in loro["rungs"] if r.get("held_out", "").startswith("top")]
        if top and "worst_ratio" in top[0]:
            print("   The top rung is the extrapolation test: if its worst "
                  "point is off by more than ~1.4x (one grid step is 1.85x), "
                  "the law does not reach the rungs above.")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.output, "w"), indent=1)
    print(f"\n[INFO] written to {a.output}")


if __name__ == "__main__":
    main()
