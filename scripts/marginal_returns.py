#!/usr/bin/env python3
"""Marginal returns: pretraining compute against inference-time search.

Greedy DeltaE00 at the lowest and highest usable compute budget of the tuned
scaling study, beside the production model's greedy DeltaE00 and its
search-based medians.

Comparability (checked in code, written into the output):
  scaling study   analyses/scaling/results/tuned/porian_fit_tuned.json,
                  by_metric.pooled.best_per_budget: per budget, the best model
                  size's seed-averaged lr_tuning.py final_val_de, which is
                  src.delta_e_eval.evaluate_delta_e(sample=False), the MEDIAN
                  greedy DeltaE00 over the first 2,048 rows of the
                  shard-aligned validation split. On tuned rungs each point
                  is the best of 7 learning rates on that same number.
  production      the production run's own lr_tuning.py final_val_de
                  (analyses/scaling/results/tuned/lr_curves_prod.csv): same
                  function, decoding, statistic and kind of slice. Given at
                  the shipped rate (3.1e-4, chosen on the search pipeline) and
                  at its best greedy rate (selected the way the study's points
                  were).
  bridge          results/prod_eval/eval_tier_a_greedy.json (evaluate.py,
                  greedy, median, 2,000 tier A TEST rows) and
                  eval_val_greedy.json: same decoding and statistic, other
                  rows; shows the validation-to-test shift.
  search          test_eval.py rows 0-499 of tiers A and B (test set), median
                  with failed rows as +inf. Not the validation slice: the
                  bridge row says how far greedy moves between the two.

    python scripts/marginal_returns.py   # -> results/paper/marginal_returns.{json,md}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SEARCH_RUNS = (("light: 16 candidates, top-1, 5 steps", "results/prod_eval/test_eval_light"),
               ("mid: 50 candidates, top-1, 20 steps", "results/prod_eval/test_eval_mid"),
               ("high: 250 candidates, top-1, 40 steps", "results/prod_eval/test_eval_high_model"),
               ("strong: 250 candidates, top-10, 100 steps", "results/paper_runs/strong_model"))


def search_medians(folder: Path) -> dict:
    out = {}
    for tier in ("a", "b"):
        by_row = {}
        for rp in sorted(glob.glob(str(folder / "**" / f"tier_{tier}" / "rows.json"),
                                   recursive=True)):
            for r in json.load(open(rp)):
                # A rerun after the sampler crash (fixed Nov 18) replaces the crash.
                old = by_row.get(r["row_idx"])
                if old is None or ("out of range [0, 3201)" in str(old.get("error") or "")
                                   and not r.get("error")):
                    by_row[r["row_idx"]] = r
        des = []
        for r in by_row.values():
            d = r.get("delta_e")
            des.append(float(d) if d is not None and math.isfinite(float(d)) else math.inf)
        if des:
            out[tier] = {"median": float(np.median(des)), "n": len(des),
                         "n_failed": int(np.sum(~np.isfinite(des)))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="analyses/scaling/results/tuned/porian_fit_tuned.json")
    p.add_argument("--prod-curve", default="analyses/scaling/results/tuned/lr_curves_prod.csv")
    p.add_argument("--shipped-lr", type=float, default=3.104e-4)
    p.add_argument("--out", default="results/paper/marginal_returns.json")
    a = p.parse_args()

    fit = json.load(open(REPO / a.fit))
    pooled = fit["by_metric"]["pooled"]
    usable = [r for r in pooled["rungs"] if r["usable"]]
    best = pooled["best_per_budget"]

    def best_at(b):
        return next(v for k, v in best.items() if abs(float(k) / b - 1) < 1e-3)

    lo, hi = usable[0], usable[-1]
    study = [{"budget_flops": r["budget"], "greedy_dE_best": best_at(r["budget"]),
              "usable": r["usable"]} for r in pooled["rungs"]]

    rows = list(csv.DictReader(open(REPO / a.prod_curve)))
    stable = [r for r in rows if r["diverged"] == "0" and r["greedy_de"]]
    shipped = next(r for r in stable if abs(float(r["lr"]) / a.shipped_lr - 1) < 1e-3)
    best_g = min(stable, key=lambda r: float(r["greedy_de"]))
    from src.scaling.flops import train_flops_per_example
    from lr_edges import arch_for
    d, se, D = int(shipped["d_model"]), int(shipped["se"]), int(shipped["examples"])
    C_prod = float(train_flops_per_example(arch_for(d, se)) * D)

    bridge = {}
    for name in ("eval_tier_a_greedy", "eval_tier_b_greedy", "eval_val_greedy"):
        f = REPO / "results" / "prod_eval" / f"{name}.json"
        if f.is_file():
            m = json.load(open(f))["metrics"]
            bridge[name] = {"median": m["ciede2000_median"], "n": m["ciede2000_n_computed"],
                            "greedy": not m.get("sample_predictions", False)}

    search = {}
    for label, folder in SEARCH_RUNS:
        if (REPO / folder).is_dir():
            s = search_medians(REPO / folder)
            if s:
                search[label] = {"folder": folder, **s}

    out = {
        "comparability": {
            "decoding": "greedy everywhere (evaluate_delta_e sample=False; evaluate.py "
                        "sample_predictions=False)",
            "statistic": "median DeltaE00 everywhere",
            "slice": "study and production greedy: the 2,048-row validation slice of their "
                     "own run (same construction, rows depend on the run's D); bridge and "
                     "search: test rows",
            "selection": "study points: best over model size and (tuned rungs) 7 learning "
                         "rates on this metric; production shown at the shipped rate and at "
                         "its best greedy rate",
            "recomputed": "nothing: the production run's lr_tuning.py final_val_de is already "
                          "on the study's footing, so eval_tier_a_greedy.json is a bridge, "
                          "not the comparison"},
        "study": study,
        "lowest_usable": {"budget_flops": lo["budget"], "greedy_dE": best_at(lo["budget"])},
        "highest_usable": {"budget_flops": hi["budget"], "greedy_dE": best_at(hi["budget"])},
        "production": {"budget_flops": C_prod, "n_params": int(shipped["n_params"]),
                       "examples": D,
                       "greedy_dE_shipped_lr": {"lr": float(shipped["lr"]),
                                                "value": float(shipped["greedy_de"])},
                       "greedy_dE_best_lr": {"lr": float(best_g["lr"]),
                                             "value": float(best_g["greedy_de"])}},
        "bridge": bridge, "search": search,
        "sources": {"fit": a.fit, "production_curve": a.prod_curve}}
    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)

    f = lambda x: f"{x:.2f}"
    L = ["# Marginal returns: pretraining compute vs search", "",
         "Median DeltaE00. Greedy rows are on a validation slice, built the same way for "
         "the study and the production run; search rows are on test rows 0-499 (failed rows "
         "count as +inf).", "",
         "| system | compute (FLOPs) | decoding | median ΔE₀₀ |", "|---|---|---|---|",
         f"| study, lowest usable budget | {lo['budget']:.2e} | greedy (best size, best LR) | "
         f"{f(best_at(lo['budget']))} |",
         f"| study, highest usable budget | {hi['budget']:.2e} | greedy (best size, best LR) | "
         f"{f(best_at(hi['budget']))} |",
         f"| production, best greedy LR ({float(best_g['lr']):.2g}) | {C_prod:.2e} | greedy | "
         f"{f(float(best_g['greedy_de']))} |",
         f"| production, shipped LR ({float(shipped['lr']):.2g}) | {C_prod:.2e} | greedy | "
         f"{f(float(shipped['greedy_de']))} |"]
    for label, s in search.items():
        L.append(f"| production, shipped LR | {C_prod:.2e} | search, {label} | "
                 + " / ".join(f"{t.upper()} {f(v['median'])}" for t, v in s.items()
                              if t in ("a", "b")) + " |")
    L += ["", "Bridge, same decoding and statistic on other rows: "
          + "; ".join(f"{k} {f(v['median'])} (n={v['n']})" for k, v in bridge.items()) + ".", ""]
    (outp.with_suffix(".md")).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
