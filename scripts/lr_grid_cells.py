#!/usr/bin/env python3
"""Enumerate the (model, D, beta2) cells for a learning-rate tuning stage.

Lives outside the SLURM script because stage 2's cells are derived from the
sweep's own compute-optimal points rather than written down by hand.

Why that matters, and where it differs from Porian et al.
---------------------------------------------------------
They tune at a CONSTANT token multiplier: across a 42x range in parameters,
their sweep's M = tokens/params stays between 20.0 and 21.1. So in their grid N
and M are decoupled by construction, M is pinned, and a law in N alone is the
right object.

INDIGO cannot borrow that, because our compute-optimal multiplier is not
constant. D*/N* runs from 29.3 at C = 1e14 down to 0.86 at C = 8.3e15, a 34x
range, and the 48 sweep runs occupy M from 0.28 to 52. Tuning at a fixed D
instead would be worse still: M = D/N would then vary as 1/N across the ladder,
220x, and the fitted "lr(N)" would really be lr along a trajectory in M. That is
the aspect-ratio confound again, in a different variable.

So stage 2 tunes at the (N*, D*) pairs the sweep actually found: the points that
determine the IsoFLOP minima. D* barely moves, so this is affordable, and it
spans N by 61x. It is a one-dimensional path through (N, M), so it cannot
separate the two exponents on its own; stage 3 varies M at fixed N for that,
and the two together support the 2-D fit.

    python scripts/lr_grid_cells.py --stage 2
    python scripts/lr_grid_cells.py --stage 2 --format table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling.configs import achievable_sizes          # noqa: E402

#: beta2 at the smallest and largest rung (stage 1), then the winner.
STAGE1_BETA2 = (0.95, 0.99, 0.999)


def nearest_config(n_target: float):
    """The achievable (n_params, d_model, slot_encoder_layers) nearest in log N."""
    import math
    return min(achievable_sizes(),
               key=lambda s: abs(math.log(s[0] / max(n_target, 1.0))))


def sweep_n_range(fit: dict) -> tuple[int, int]:
    """Smallest and largest model the sweep actually trains."""
    ns = [r["n_params"] for r in fit.get("runs", [])]
    if not ns:                       # porian_fit.json carries rungs, not runs
        ns = [r["n_star_median"] for r in fit["by_metric"]["pooled"]["rungs"]
              if r.get("usable")]
    return int(min(ns)), int(max(ns))


def cells_for(stage, fit_path: str, beta2: float, rungs: int = 3) -> list[dict]:
    fit = json.load(open(fit_path))

    rungs_ = [r for r in fit["by_metric"]["pooled"]["rungs"] if r.get("usable")]
    if not rungs_:
        raise SystemExit(f"no usable rungs in {fit_path}")

    # The sweep's own grid, budget -> {N: D}. Stage 2 tunes every point of it
    # on the lowest curves, so it needs the whole grid rather than each rung's
    # optimum. porian_fit.json carries the rungs; the runs live in the artifact
    # it was built from.
    src = fit.get("runs")
    if src is None:
        src = json.load(open(fit["source"]))["runs"]
    budgets = sorted(r["budget"] for r in rungs_)
    points_at = {b: {} for b in budgets}
    for run in src:
        b = min(budgets, key=lambda x: abs(x - run["flops"]))
        if abs(run["flops"] - b) / b < 0.25:
            points_at[b][run["n_params"]] = run["passes"]

    if stage == "probe":
        # One mid-ladder size, one learning rate, a short budget. The only
        # question is examples per second.
        #
        # It exists because stage 1 was sized at the planner's 1301 ex/s and
        # ran at a median of 204, so every cell hit its wall. The sweep itself
        # reaches 2171 ex/s on the same shards, so the gap is contention or
        # configuration rather than a floor, and the difference decides whether
        # the rest of the tuning costs 20 GPU-hours or 200. Run it alone:
        # --array=0-0, nothing else of yours queued.
        n, d, se = nearest_config(rungs_[len(rungs_) // 2]["n_star_median"])
        return [{"n_params": n, "d_model": d, "se": se, "D": 614_400,
                 "beta2": beta2}]

    if stage == 1:
        # beta2, at both ends of the ladder the SWEEP uses. (Not of every
        # achievable shape: achievable_sizes runs to 121M parameters, seven
        # times anything this study trains.)
        #
        # Each beta2 gets its own small spread of learning rates rather than
        # one shared value, because the two interact: comparing beta2 at a
        # single fixed LR can pick whichever beta2 happens to suit that LR.
        # Porian et al. collapse beta2 by taking the best cell over LR, which
        # needs the grid. Three rates is enough to see whether the beta2
        # ranking is STABLE across LR, which is the question here; finding the
        # LR optimum itself is stage 2's job, with seven.
        #
        # And they run at the rung's own D*, not at the historical 614,400.
        # The first attempt used that smaller budget and nothing learned:
        # accuracy sat at the EOS base rate for every trial and DeltaE came
        # back as scatter between 25 and 37, so no beta2 could be ranked
        # against another.
        ends = [rungs_[0], rungs_[-1]]
        out = []
        for r in ends:
            n, d, se = nearest_config(r["n_star_median"])
            D = int(round(r["d_star"] * 1.1 / 1000) * 1000)
            out.extend({"n_params": n, "d_model": d, "se": se, "D": D,
                        "beta2": b} for b in STAGE1_BETA2)
        return out

    if stage == 2:
        # EVERY model on the lowest `rungs` IsoFLOP curves, tuned directly.
        #
        # Not one representative point per rung. The whole curve, because the
        # curve is what the parabola is fitted through: a point whose learning
        # rate was extrapolated rather than measured moves the minimum as
        # surely as one that was trained wrong.
        #
        # Each point is its own (N, M) pair, and that is what makes the law
        # identifiable without a separate experiment. Within ONE rung C is
        # fixed, so M = C/(k N^2) and log M = const - 2 log N: the two columns
        # are collinear (corr -0.9998) and only the combination b - 2c can be
        # recovered. A second rung shifts the intercept and separates them. At
        # three rungs the design matrix has condition number 401, and on
        # synthetic data with 10% noise on log lr* it recovers both exponents
        # to +/- 0.05 and +/- 0.03.
        #
        # So the M axis comes free from the IsoFLOP geometry. It does not need
        # the fractional-scoring trick, and it does not need a constant
        # learning rate to get it, which means this measures the law under the
        # schedule the sweep actually trains with.
        out = []
        for b in budgets[:rungs]:
            for n, D in sorted(points_at[b].items()):
                _, d_model, se = nearest_config(n)
                out.append({"n_params": n, "d_model": d_model, "se": se,
                            "D": int(D), "beta2": beta2, "budget": b,
                            "M": D / n})
        return out

    if stage == 3:
        # The extrapolation check. Stage 2 measures the low rungs; the upper
        # ones get the fitted law rather than a measurement, so tune ONE point
        # on a high rung and compare what the law predicted against what the
        # sweep actually wanted there.
        #
        # The compute-optimal point of the highest rung that the IsoFLOP fit
        # could use: it is where an error in the law does the most damage.
        r = rungs_[-1]
        n, d_model, se = nearest_config(r["n_star_median"])
        return [{"n_params": n, "d_model": d_model, "se": se,
                 "D": int(r["d_star"]), "beta2": beta2,
                 "M": r["d_star"] / r["n_star_median"]}]

    raise SystemExit(f"unknown stage {stage}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("probe", "1", "2", "3"),
                   help="'probe' is one cell at one learning rate: it measures "
                        "throughput and nothing else, and decides whether the "
                        "rest is affordable.")
    p.add_argument("--fit", default="analyses/scaling/results/porian_fit.json")
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--rungs", type=int, default=3,
                   help="stage 2: how many of the LOWEST IsoFLOP curves to "
                        "tune in full. Two is the minimum that separates the "
                        "N and M exponents; three gives a condition number of "
                        "401 against 630 at two. Above three the cost roughly "
                        "doubles per rung for little extra conditioning.")
    p.add_argument("--format", choices=("lines", "table"), default="lines")
    p.add_argument("--rate", type=float, default=204.0,
                   help="examples per second. The default is the MEASURED "
                        "median from the first stage-1 submission (227 step "
                        "samples, min 13, median 204, max 395), not the 1301 "
                        "the sweep planner assumes. Tuning cells run far "
                        "slower than sweep runs: six array tasks stream the "
                        "same shards at once and these models are input-bound, "
                        "so concurrency costs more than model size does. "
                        "Sizing stage 1 at 1301 under-estimated it by 6x and "
                        "every cell hit the 6-hour wall.")
    p.add_argument("--wall-hours", type=float, default=6.0,
                   help="the --time the array will be submitted with, so the "
                        "table can say which cells do not fit")
    p.add_argument("--n-lrs", type=int, default=7)
    a = p.parse_args()

    stage = a.stage if a.stage == "probe" else int(a.stage)
    cells = cells_for(stage, a.fit, a.beta2, a.rungs)
    # Stage 1 ranks beta2; stage 2 locates the LR optimum. Different jobs,
    # different grid widths.
    n_lrs = {"probe": 1, 1: 3}.get(stage, a.n_lrs)
    if a.format == "lines":
        for c in cells:
            print(f"{c['d_model']} {c['se']} {c['D']} {c['beta2']:g}")
        return

    print(f"STAGE {a.stage}: {len(cells)} cell(s), "
          f"{n_lrs} learning rate(s) each")
    print(f"{'idx':>4} {'model':<12} {'N':>11} {'D':>12} {'M = D/N':>9} "
          f"{'beta2':>6} {'GPU-h':>7}")
    total = 0.0
    for i, c in enumerate(cells):
        h = n_lrs * c["D"] / a.rate / 3600
        total += h
        print(f"{i:>4} d{c['d_model']}/se{c['se']:<8} {c['n_params']:>11,} "
              f"{c['D']:>12,} {c['D'] / c['n_params']:>9.2f} "
              f"{c['beta2']:>6g} {h:>7.1f}")
    print(f"\n{'':>4} {'total':<12} {'':>11} {'':>12} {'':>9} {'':>6} {total:>7.1f}")
    # A cell that needs most of its wall will die on the tail, because the
    # rate is a median and the slow end of the distribution is 15x below it.
    # The first stage-1 submission was estimated at 0.9h per cell against a
    # 6h wall and still hit the limit.
    MARGIN = 0.6
    hours = [n_lrs * c["D"] / a.rate / 3600 for c in cells]
    at_risk = [h for h in hours if h > a.wall_hours * MARGIN]
    if at_risk:
        worst = max(hours)
        need = int(worst / MARGIN) + 1
        print(f"\n[WARN] {len(at_risk)} of {len(cells)} cells need more than "
              f"{MARGIN:.0%} of the {a.wall_hours:g}h wall "
              f"({worst:.1f}h for the largest at {a.rate:g} ex/s).")
        print(f"       The rate is a MEDIAN and the slow tail runs 15x under "
              f"it, so a cell sized near the wall will not finish.")
        print(f"       Submit with --time={need:02d}:00:00 --qos=long, and "
              f"throttle the array (--array=0-{len(cells) - 1}%2): these runs "
              f"are input-bound, so concurrent cells slow each other down.")


if __name__ == "__main__":
    main()
