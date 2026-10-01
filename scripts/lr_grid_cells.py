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

So the learning-rate search is embedded in the IsoFLOP test itself: stage 2
tunes EVERY model on the lowest few curves, which is where the parabolas are
fitted and where a borrowed learning rate would move a minimum. Each of those
points carries its own (N, M), and because a second rung shifts the intercept,
the geometry separates the two exponents without a separate experiment.

Stage "check" then tests the fitted law at one upper-rung point, and stage 3
finishes the IsoFLOP: every (N, D) point of the remaining curves trained once
at the law's learning rate, plus every rung's repeat seeds. The learning rate
is the only extrapolated quantity. Stage 2's winning trials and stage 3's runs
together are the final figure; scripts/collect_isoflop.py assembles them.

How many curves to tune in full, against how many to project, is the second
objective of the throughput probe: see the stage-"probe" block below for the
cost and conditioning at two, three and four.

    python scripts/lr_grid_cells.py --stage 2
    python scripts/lr_grid_cells.py --stage 2 --format table
    python scripts/lr_grid_cells.py --stage 3 --format table   # before stage 2 too
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling.configs import (CORPUS_EXAMPLES,                  # noqa: E402
                                 DEFAULT_BATCH_SIZE, SEC_PER_DE_EXAMPLE,
                                 STARTUP_SEC, achievable_sizes)

#: Longest --time CRC's --qos=short accepts on the l40s partition.
QOS_SHORT_MAX_HOURS = 24

#: beta2 at the smallest and largest rung (stage 1), then the winner.
STAGE1_BETA2 = (0.95, 0.99, 0.999)


def nearest_config(n_target: float):
    """The achievable (n_params, d_model, slot_encoder_layers) nearest in log N."""
    return min(achievable_sizes(),
               key=lambda s: abs(math.log(s[0] / max(n_target, 1.0))))


def sweep_n_range(fit: dict) -> tuple[int, int]:
    """Smallest and largest model the sweep actually trains."""
    ns = [r["n_params"] for r in fit.get("runs", [])]
    if not ns:                       # porian_fit.json carries rungs, not runs
        ns = [r["n_star_median"] for r in fit["by_metric"]["pooled"]["rungs"]
              if r.get("usable")]
    return int(min(ns)), int(max(ns))


def split_passes(passes: int, batch_size: int = DEFAULT_BATCH_SIZE,
                 corpus: int = CORPUS_EXAMPLES) -> tuple[int, int]:
    """(epochs, limit_examples) for `passes` example-passes, as the sweep does.

    Same rule as src/scaling/configs.py: as many epochs as it takes to fit the
    corpus, each a whole number of batches. Only the 2.5e16 rung's smallest
    model needs more than one.
    """
    epochs = max(1, math.ceil(passes / corpus))
    return epochs, (passes // batch_size // epochs) * batch_size


def _cell(n: int, D: int, beta2: float, *, seed: int = 42,
          lr: Optional[float] = None, budget: Optional[float] = None) -> dict:
    """One cell. `D` is total example-passes; `lr` set means ONE fixed rate."""
    n_params, d_model, se = nearest_config(n)
    epochs, limit = split_passes(int(D))
    return {"n_params": n_params, "d_model": d_model, "se": se, "D": int(D),
            "epochs": epochs, "limit": limit, "beta2": beta2, "seed": seed,
            "lr": lr, "budget": budget, "M": D / n_params}


def sweep_points(fit: dict) -> tuple[list[float], list[dict]]:
    """The finished sweep's grid: every (budget, N, D, seed) it trained.

    Repeat seeds are kept, because the final IsoFLOP needs them for the same
    reason the first sweep did: they calibrate the seed noise the bootstrap
    draws from.
    """
    src = fit.get("runs")
    if src is None:                       # porian_fit.json points at its source
        src = json.load(open(fit["source"]))["runs"]
    budgets = sorted(fit["budgets"])
    points = []
    for run in src:
        b = min(budgets, key=lambda x: abs(x - run["flops"]))
        if abs(run["flops"] - b) / b >= 0.25:
            continue
        m = re.search(r"_s(\d+)$", run.get("name", ""))
        points.append({"budget": b, "n": int(run["n_params"]),
                       "D": int(run["passes"]),
                       "seed": int(m.group(1)) if m else 42})
    points.sort(key=lambda p: (p["budget"], p["n"], p["seed"]))
    return budgets, points


def stage2_optimum(stage2_dir: Path, cell: dict) -> Optional[float]:
    """The learning rate stage 2 selected for this cell's (N, D, beta2), or None.

    Read from lr_tuning.py's own "optimal_lr", the trial whose run becomes the
    IsoFLOP point, so a repeat seed runs at exactly the rate its seed-42 twin
    was scored at and the pair is a clean noise measurement.
    """
    path = (stage2_dir / f"lr_search_ep{cell['epochs']}_lim{cell['limit']}"
            f"_d{cell['d_model']}_se{cell['se']}_bs{DEFAULT_BATCH_SIZE}"
            f"_b2{cell['beta2']:g}.json")
    if not path.is_file():
        return None
    return float(json.load(open(path))["optimal_lr"])


def cells_for(stage, fit_path: str, beta2: float, rungs: int = 3,
              stage2_dir: Optional[str] = None) -> list[dict]:
    fit = json.load(open(fit_path))

    rungs_ = [r for r in fit["by_metric"]["pooled"]["rungs"] if r.get("usable")]
    if not rungs_:
        raise SystemExit(f"no usable rungs in {fit_path}")

    # The sweep's own grid. Stages 2 and 3 between them re-run ALL of it, at
    # tuned hyperparameters, and those runs are the final IsoFLOP figure:
    # stage 2 the lowest `rungs` curves with the learning rate measured,
    # stage 3 the rest with it extrapolated. Same (N, D) points the first
    # sweep trained, so the two figures compare point for point.
    budgets, points = sweep_points(fit)
    lower = set(budgets[:rungs])

    if stage == "probe":
        # One mid-ladder size, one learning rate, a short budget. It measures
        # examples per second, and that one number settles TWO things.
        #
        # 1. WHETHER THE REST IS AFFORDABLE. Stage 1 was sized at the planner's
        #    1301 ex/s and ran at a median of 204, so every cell hit its wall.
        #    The sweep itself reaches 2171 ex/s on the same shards, so the gap
        #    is contention or configuration rather than a floor.
        #
        # 2. HOW MANY IsoFLOP CURVES STAGE 2 TUNES IN FULL, the rest being
        #    projected from the fitted law. Each added rung improves the
        #    conditioning of the (N, M) fit and roughly doubles the cost:
        #
        #      rungs  cells   GPU-h @2171   GPU-h @204   cond   sd(b)  sd(c)  extrap
        #          2     12          31.3        245.3    630   0.100  0.054   13/24
        #          3     18          56.7        471.3    401   0.050  0.031    9/24
        #          4     24          103.7        927.6    303   0.032  0.021    5/24
        #
        #    sd(b), sd(c) are the spreads of the recovered N and M exponents
        #    over 2000 synthetic draws with 0.10 of noise in log-lr units, and
        #    "extrap" counts the sweep's 24 distinct model sizes that fall
        #    ABOVE the largest size tuned. So the fourth rung buys coverage as
        #    well as conditioning, and coverage is the stronger argument: an
        #    extrapolated size is a size whose learning rate is a guess.
        #    At 2171 it tightens b 1.6x and halves the extrapolated sizes for
        #    2x the compute, a real choice; at 204 even three rungs is 471
        #    GPU-hours and the choice is three or nothing. Carry the answer
        #    into stage 2 as --rungs (RUNGS in the SLURM wrapper).
        #
        # Run it ALONE: --array=0-0, nothing else of yours queued, or it
        # measures contention rather than throughput.
        n, _, _ = nearest_config(rungs_[len(rungs_) // 2]["n_star_median"])
        return [_cell(n, 614_400, beta2)]

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
            n, _, _ = nearest_config(r["n_star_median"])
            D = int(round(r["d_star"] * 1.1 / 1000) * 1000)
            out.extend(_cell(n, D, b) for b in STAGE1_BETA2)
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
        return [_cell(p["n"], p["D"], beta2, budget=p["budget"])
                for p in points if p["budget"] in lower and p["seed"] == 42]

    if stage == "check":
        # The extrapolation check, run between stages 2 and 3 and BEFORE the
        # spend. Stage 3 trusts the fitted law at every upper-rung point, so
        # tune ONE point there in full and compare what the law predicted
        # against what that point actually wanted. The compute-optimal point
        # of the highest usable rung: where an error in the law does the most
        # damage. If the ratio is far from 1, tune another rung directly
        # (RUNGS + 1) rather than spend stage 3 on rates that are guesses.
        #
        # Its results go to their own directory, so the law is not fitted on
        # the point that is supposed to test it.
        r = rungs_[-1]
        return [_cell(r["n_star_median"], int(r["d_star"]), beta2)]

    if stage == 3:
        # Finish the IsoFLOP. Every (N, D) point the sweep trained on the
        # upper rungs, at ONE learning rate each, from the law stage 2 fitted:
        # lr(N, D) = a N^b D^c, applied to that point's own N and D. Nothing
        # else is extrapolated; beta2 is stage 1's winner and the (N, D) grid
        # is the sweep's.
        #
        # Plus the lower rungs' repeat seeds. Stage 2 tunes seed 42 only, and
        # its winning trial IS that point's IsoFLOP run, so the repeats run
        # at the rate stage 2 picked. That keeps every seed cluster at one
        # learning rate, which is what makes it a noise measurement.
        from src.scaling.configs import tuned_lr_for
        out = []
        for p in points:
            if p["budget"] in lower:
                if p["seed"] == 42:
                    continue              # stage 2's winning trial is this run
                c = _cell(p["n"], p["D"], beta2, seed=p["seed"],
                          budget=p["budget"])
                c["lr"] = stage2_optimum(Path(stage2_dir), c) if stage2_dir else None
                c["lr_source"] = "stage 2"
            else:
                c = _cell(p["n"], p["D"], beta2, seed=p["seed"],
                          budget=p["budget"])
                c["lr"] = tuned_lr_for(c["n_params"], c["D"])
                c["lr_source"] = "law"
            out.append(c)
        return out

    raise SystemExit(f"unknown stage {stage}")


def cell_hours(D: int, n_lrs: int, rate: float, val_examples: int,
               de_examples: int) -> float:
    """GPU-hours for one cell: every cost the job pays, not just training.

    lr_tuning.py runs its learning rates one after another in ONE process, so
    the cell pays the fixed startup once (module load, torch and JAX imports,
    shard scan, validation read, simulator preflight: STARTUP_SEC, measured on
    the sweep) and then, per learning rate, a training pass, one CE validation
    pass and one DeltaE eval. Pricing only the training pass under-stated
    stage 2 by about a third. The validation pass is priced at the training
    rate, which over-states a forward-only pass slightly, in the safe
    direction.
    """
    per_lr = (D + val_examples) / rate + de_examples * SEC_PER_DE_EXAMPLE
    return (STARTUP_SEC + n_lrs * per_lr) / 3600


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=("probe", "1", "2", "check", "3"),
                   help="probe: throughput. 1: beta2. 2: learning rate, tuned "
                        "on every point of the lowest RUNGS IsoFLOP curves. "
                        "check: the fitted law tested at one upper-rung point. "
                        "3: every point of the remaining curves at the law's "
                        "learning rate, plus the lower rungs' repeat seeds. "
                        "Stages 2 and 3 together are the final IsoFLOP.")
    p.add_argument("--stage2-dir", default="outputs/lr_search/cross_attn",
                   help="stage 3: where stage 2's results are, for the "
                        "learning rate each lower-rung repeat seed reuses")
    p.add_argument("--fit", default="analyses/scaling/results/porian_fit.json")
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--rungs", type=int, default=3,
                   help="stage 2: how many of the LOWEST IsoFLOP curves to "
                        "tune in full, the rest being projected from the "
                        "fitted law. Two is the minimum that separates the N "
                        "and M exponents (cond 630); three gives cond 401 at "
                        "18 cells and 56.7 GPU-h at 2171 ex/s; four gives 303 "
                        "at 24 cells and 103.7 GPU-h. THE PROBE DECIDES "
                        "BETWEEN THREE AND FOUR: at 2171 ex/s the fourth rung "
                        "tightens the fitted N exponent 1.6x and cuts the "
                        "sweep sizes left to extrapolation from 9 of 24 to 5, "
                        "for twice the compute; at 204 ex/s three rungs "
                        "already costs 471 GPU-h and four is out of reach.")
    p.add_argument("--format", choices=("lines", "table", "count"),
                   default="lines",
                   help="lines: one runnable cell per line, for the SLURM "
                        "array. table: cells with cost. count: how many "
                        "cells, which stage 3 can answer before its learning "
                        "rates exist.")
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
    p.add_argument("--val-examples", type=int, default=10_000,
                   help="--limit-val-examples the cells run with; one CE "
                        "validation pass per learning rate")
    p.add_argument("--de-examples", type=int, default=2048,
                   help="--limit-de-examples the cells run with; one DeltaE "
                        "eval per learning rate")
    a = p.parse_args()

    stage = a.stage if a.stage in ("probe", "check") else int(a.stage)
    cells = cells_for(stage, a.fit, a.beta2, a.rungs, a.stage2_dir)
    # Stage 1 ranks beta2; stage 2 and the check locate the LR optimum; stage
    # 3 trains each point once at a rate it is given. Different jobs,
    # different grid widths.
    n_lrs = {"probe": 1, 1: 3, 3: 1}.get(stage, a.n_lrs)
    if a.format == "count":
        print(len(cells))
        return
    if a.format == "lines":
        missing = [c for c in cells if stage == 3 and c["lr"] is None]
        if missing:
            src = {c["lr_source"] for c in missing}
            raise SystemExit(
                f"{len(missing)} stage-3 cell(s) have no learning rate yet "
                f"({', '.join(sorted(src))}). Stage 3 needs stage 2 finished "
                f"and fitted: run scripts/fit_lr_law.py so "
                f"analyses/scaling/results/lr_law_fit.json carries "
                f"lr_vs_n_and_d, and check --stage2-dir={a.stage2_dir}.")
        # epochs and limit are what lr_tuning.py is given; lr "-" means the
        # SLURM wrapper builds its usual grid around the prior.
        for c in cells:
            lr = "-" if c["lr"] is None else f"{c['lr']:.6e}"
            print(f"{c['d_model']} {c['se']} {c['limit']} {c['beta2']:g} "
                  f"{c['epochs']} {c['seed']} {lr}")
        return

    print(f"STAGE {a.stage}: {len(cells)} cell(s), "
          f"{n_lrs} learning rate(s) each")
    print(f"{'idx':>4} {'model':<12} {'N':>11} {'D':>12} {'M = D/N':>9} "
          f"{'beta2':>6} {'seed':>5} {'lr':>10} {'GPU-h':>7}")
    hours = [cell_hours(c["D"], n_lrs, a.rate, a.val_examples, a.de_examples)
             for c in cells]
    total = 0.0
    for i, (c, h) in enumerate(zip(cells, hours)):
        total += h
        if stage == 3:
            lr = (f"{c['lr']:.2e}" if c["lr"] is not None
                  else f"({c['lr_source']})")
        else:
            lr = "grid"
        print(f"{i:>4} d{c['d_model']}/se{c['se']:<8} {c['n_params']:>11,} "
              f"{c['D']:>12,} {c['M']:>9.2f} "
              f"{c['beta2']:>6g} {c['seed']:>5} {lr:>10} {h:>7.1f}")
    print(f"\n{'':>4} {'total':<12} {'':>11} {'':>12} {'':>9} {'':>6} "
          f"{'':>5} {'':>10} {total:>7.1f}")
    train_only = sum(n_lrs * c["D"] / a.rate / 3600 for c in cells)
    print(f"{'':>4} (of which training {train_only:.1f}; startup "
          f"{len(cells) * STARTUP_SEC / 3600:.1f}; evals "
          f"{total - train_only - len(cells) * STARTUP_SEC / 3600:.1f})")
    if stage == 3 and any(c["lr"] is None for c in cells):
        print(f"{'':>4} learning rates in parentheses are filled in once "
              f"stage 2 is finished and fitted; the cells and costs are final.")
    # A cell that needs most of its wall will die on the tail, because the
    # rate is a median and the slow end of the distribution is 15x below it.
    # The first stage-1 submission was estimated at 0.9h per cell against a
    # 6h wall and still hit the limit.
    MARGIN = 0.6
    at_risk = [h for h in hours if h > a.wall_hours * MARGIN]
    if at_risk:
        worst = max(hours)
        need = int(worst / MARGIN) + 1
        print(f"\n[WARN] {len(at_risk)} of {len(cells)} cells need more than "
              f"{MARGIN:.0%} of the {a.wall_hours:g}h wall "
              f"({worst:.1f}h for the largest at {a.rate:g} ex/s).")
        print(f"       The rate is a MEDIAN and the slow tail runs 15x under "
              f"it, so a cell sized near the wall will not finish.")
        # CRC's short QoS runs to 24 hours on l40s; only past that is long
        # needed, and long caps how many cards a group holds at once.
        qos = "" if need <= QOS_SHORT_MAX_HOURS else " --qos=long"
        print(f"       Submit with --time={need:02d}:00:00{qos}, and "
              f"throttle the array (--array=0-{len(cells) - 1}%2): these runs "
              f"are input-bound, so concurrent cells slow each other down.")


if __name__ == "__main__":
    main()
