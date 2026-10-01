#!/usr/bin/env python3
"""IsoFLOP fit following Porian et al. 2024, for comparison with our own.

`scripts/fit_scaling.py` fits a parabola in log N per rung and an unweighted
power law across rungs. This script runs the estimator from the authors'
released code instead: Akima interpolation, boundary rejection, a seed-noise
bootstrap whose median is the observation, and a 1/sigma^2-weighted power law.
See `src/scaling/porian.py` for what each of those changes and why.

Input is the JSON `fit_scaling.py --output` already writes, so no runs need
re-reading. Output is a JSON with the same shape plus the extra laws Porian et
al. report (the D*/N* multiplier, and a saturating fit with an irreducible
floor).

    python scripts/fit_scaling_porian.py \
        --fit analyses/scaling/results/isoflop_fit.json \
        --output analyses/scaling/results/porian_fit.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling import porian as P                       # noqa: E402
from src.scaling.credits import CreditModel               # noqa: E402

BUCKETS = ("pooled", "low", "mid", "high")


def value_of(run: dict, metric: str) -> float:
    return run["val_loss"] if metric == "ce" else run["val_de"][metric]


def group_by_budget(runs: List[dict], budgets: List[float]) -> Dict[float, List[dict]]:
    out = collections.defaultdict(list)
    for r in runs:
        out[min(budgets, key=lambda b: abs(b - r["flops"]))].append(r)
    return dict(out)


def noise_for(runs: List[dict], metric: str, constant: bool) -> P.NoiseModel:
    """Calibrate the bootstrap noise from this metric's repeat-seed clusters."""
    clusters = collections.defaultdict(list)
    for r in runs:
        clusters[(round(r["flops"], -11), r["n_params"])].append(value_of(r, metric))
    return P.NoiseModel.from_clusters(list(clusters.values()), constant=constant)


def law_json(law: P.PowerLaw) -> dict:
    return {"exponent": law.exponent, "coef": law.coef, "r2": law.r2,
            "n_points": law.n_points, "weighted": law.weighted,
            "ci_low": law.ci_low, "ci_high": law.ci_high}


def analyse(runs: List[dict], budgets: List[float], metric: str,
            n_boot: int, n_draws: int, constant_noise: bool,
            credits: CreditModel, seed: int) -> dict:
    by_b = group_by_budget(runs, budgets)
    noise = noise_for(runs, metric, constant_noise)
    rungs, rung_json = [], []
    for b in sorted(by_b):
        rs = by_b[b]
        f = P.fit_rung(b, [r["n_params"] for r in rs],
                       [value_of(r, metric) for r in rs],
                       d_vals=[r["passes"] for r in rs], noise=noise,
                       n_boot=n_boot, rng=np.random.default_rng(seed))
        rungs.append(f)
        rung_json.append({
            "budget": b, "n_points": len(rs), "on_edge": f.on_edge,
            "usable": f.usable,
            "n_star_akima": f.n_star, "value_at_star": f.y_star,
            "n_star_median": f.n_star_median, "log_sigma": f.log_sigma,
            "valid_fraction": f.valid_fraction,
            "d_star": f.d_star_median if f.usable else None,
            "credits": credits.from_passes(f.d_star_median) if f.usable else None,
        })

    out = {"metric": metric, "noise_sigma_lo": noise.sigma_lo,
           "noise_sigma_hi": noise.sigma_hi, "rungs": rung_json,
           "n_usable": sum(r.usable for r in rungs)}
    if out["n_usable"] < 2:
        out["laws"] = {"status": f"only {out['n_usable']} usable rungs"}
        return out

    good = [r for r in rungs if r.usable]
    B = [r.budget for r in good]
    S = [r.log_sigma for r in good]
    n_law, n_exps = P.bootstrap_power_law(rungs, "n", n_draws=n_draws,
                                          weighted=True,
                                          rng=np.random.default_rng(seed + 1))
    n_law_unw, _ = P.bootstrap_power_law(rungs, "n", n_draws=n_draws,
                                         weighted=False,
                                         rng=np.random.default_rng(seed + 1))
    d_law = P.power_law_fit(B, [r.d_star_median for r in good], S)
    m_law, _ = P.bootstrap_power_law(rungs, "multiplier", n_draws=n_draws,
                                     weighted=True,
                                     rng=np.random.default_rng(seed + 2))
    # Also report the exponent in effective parameters, the N that makes
    # C = 6 N D exact (src/scaling/flops.py:effective_params). It is the unit
    # Porian et al. actually plot, and in it alpha + beta = 1 by construction,
    # so the gap from 1 in the parameter-count version measures how far the
    # 6ND shorthand is from holding rather than anything physical.
    N_EFF_EXPONENT = 0.9398        # N_eff ~ 98.9 * N^0.940, r2 = 0.9999
    out["laws"] = {"N_star": law_json(n_law), "N_star_unweighted": law_json(n_law_unw),
                   "D_star": law_json(d_law), "multiplier": law_json(m_law),
                   "alpha_plus_beta": n_law.exponent + d_law.exponent,
                   "N_star_effective_params": {
                       "exponent": n_law.exponent / N_EFF_EXPONENT,
                       "ci_low": (n_law.ci_low / N_EFF_EXPONENT
                                  if n_law.ci_low is not None else None),
                       "ci_high": (n_law.ci_high / N_EFF_EXPONENT
                                   if n_law.ci_high is not None else None),
                       "note": "alpha in units where C = 6 N D is exact"},
                   "exponent_draws": n_exps.tolist()}

    # Best-per-budget, seed-averaged, and the saturating fit on it.
    best = []
    for b in sorted(by_b):
        g = collections.defaultdict(list)
        for r in by_b[b]:
            g[r["n_params"]].append(value_of(r, metric))
        best.append(min(sum(v) / len(v) for v in g.values()))
    sat = P.saturating_fit(sorted(by_b), best)
    out["best_per_budget"] = dict(zip([f"{b:.4g}" for b in sorted(by_b)], best))
    out["saturating"] = {"floor": sat.floor, "alpha": sat.alpha, "rmse": sat.rmse,
                         "n_points": sat.n_points, "identified": sat.identified,
                         "note": sat.note}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="analyses/scaling/results/isoflop_fit.json",
                   help="JSON written by scripts/fit_scaling.py --output")
    p.add_argument("--output", default="analyses/scaling/results/porian_fit.json")
    p.add_argument("--metrics", nargs="+", default=[*BUCKETS, "ce"])
    p.add_argument("--bootstrap-iters", type=int, default=P.BOOTSTRAP_ITERS)
    p.add_argument("--law-draws", type=int, default=200,
                   help="bootstrap refits of the power law (their bootstrap_num)")
    p.add_argument("--noise-varies-with-metric", action="store_true",
                   help="interpolate sigma between low and high metric values, "
                        "as they do for CE. Off by default: INDIGO's six "
                        "repeat-seed clusters show no clean trend, so a "
                        "pooled constant is what the data supports")
    p.add_argument("--su-per-gpu-hour", type=float, default=None,
                   help="service units per GPU-hour for the credits column")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    src = json.load(open(a.fit))
    credits = (CreditModel(su_per_gpu_hour=a.su_per_gpu_hour)
               if a.su_per_gpu_hour else CreditModel())
    if credits.caveat:
        print(f"[WARN] {credits.caveat}")

    result = {"source": a.fit, "budgets": src["budgets"],
              "n_runs": len(src["runs"]), "estimator": "porian2024",
              "su_per_gpu_hour": credits.su_per_gpu_hour,
              "su_rates_confirmed": credits.confirmed, "by_metric": {}}
    for m in a.metrics:
        result["by_metric"][m] = analyse(
            src["runs"], src["budgets"], m, a.bootstrap_iters, a.law_draws,
            not a.noise_varies_with_metric, credits, a.seed)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.output, "w"), indent=1)

    print(f"\n{'metric':>8} {'sigma':>7} {'rungs':>6} {'alpha (N*)':>24} "
          f"{'r2':>6} {'beta (D*)':>10} {'a+b':>6} {'multiplier':>11} "
          f"{'alpha (N_eff)':>14}")
    for m in a.metrics:
        r = result["by_metric"][m]
        laws = r["laws"]
        if "N_star" not in laws:
            print(f"{m:>8} {r['noise_sigma_lo']:7.3f} {r['n_usable']:6d}   {laws['status']}")
            continue
        n, d, mu = laws["N_star"], laws["D_star"], laws["multiplier"]
        print(f"{m:>8} {r['noise_sigma_lo']:7.3f} {r['n_usable']:6d}  "
              f"{n['exponent']:+.3f} [{n['ci_low']:+.3f},{n['ci_high']:+.3f}] "
              f"{n['r2']:6.3f} {d['exponent']:+10.3f} "
              f"{laws['alpha_plus_beta']:6.3f} {mu['exponent']:+11.3f} "
              f"{laws['N_star_effective_params']['exponent']:+14.3f}")
    print(f"\n[INFO] written to {a.output}")


if __name__ == "__main__":
    main()
