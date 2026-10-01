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


def cells_for(stage: int, fit_path: str, beta2: float) -> list[dict]:
    fit = json.load(open(fit_path))

    if stage == 1:
        # Both ends of the ladder the SWEEP uses, not of every achievable
        # shape: achievable_sizes runs out to 121M parameters, seven times
        # anything this study trains, and tuning beta2 there would answer a
        # question nobody asked.
        lo, hi = sweep_n_range(fit)
        picks = [nearest_config(lo), nearest_config(hi)]
        return [{"n_params": n, "d_model": d, "se": se,
                 "D": 614_400, "beta2": b}
                for (n, d, se) in picks for b in STAGE1_BETA2]

    rungs = [r for r in fit["by_metric"]["pooled"]["rungs"] if r.get("usable")]
    if not rungs:
        raise SystemExit(f"no usable rungs in {fit_path}")

    if stage == 2:
        out = []
        for r in rungs:
            n, d, se = nearest_config(r["n_star_median"])
            # Tune at the D the sweep spends at that rung's optimum, rounded
            # up a little so the cell brackets rather than sits on it.
            out.append({"n_params": n, "d_model": d, "se": se,
                        "D": int(round(r["d_star"] * 1.1 / 1000) * 1000),
                        "beta2": beta2, "M": r["d_star"] / r["n_star_median"]})
        return out

    if stage == 3:
        # Varying M at fixed N is what separates the M exponent from the N one.
        # Cost is 7 learning rates times D, and D = M_max * N, so the cell goes
        # on a SMALL model: the M range is what matters here, not the size.
        # The second rung spans the sweep's full M range at a third of the
        # price of the middle one.
        n, d, se = nearest_config(rungs[1]["n_star_median"])
        m_max = max(r["d_star"] / r["n_star_median"] for r in rungs)
        return [{"n_params": n, "d_model": d, "se": se,
                 "D": int(round(m_max * n / 1000) * 1000), "beta2": beta2,
                 "fractions": [1 / 32, 1 / 10, 1 / 3]}]

    raise SystemExit(f"unknown stage {stage}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    p.add_argument("--fit", default="analyses/scaling/results/porian_fit.json")
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--format", choices=("lines", "table"), default="lines")
    p.add_argument("--rate", type=float, default=1301.0)
    p.add_argument("--n-lrs", type=int, default=7)
    a = p.parse_args()

    cells = cells_for(a.stage, a.fit, a.beta2)
    if a.format == "lines":
        for c in cells:
            print(f"{c['d_model']} {c['se']} {c['D']} {c['beta2']:g}")
        return

    print(f"STAGE {a.stage}: {len(cells)} cells")
    print(f"{'idx':>4} {'model':<12} {'N':>11} {'D':>12} {'M = D/N':>9} "
          f"{'beta2':>6} {'GPU-h':>7}")
    total = 0.0
    for i, c in enumerate(cells):
        h = a.n_lrs * c["D"] / a.rate / 3600
        total += h
        print(f"{i:>4} d{c['d_model']}/se{c['se']:<8} {c['n_params']:>11,} "
              f"{c['D']:>12,} {c['D'] / c['n_params']:>9.2f} "
              f"{c['beta2']:>6g} {h:>7.1f}")
    print(f"\n{'':>4} {'total':<12} {'':>11} {'':>12} {'':>9} {'':>6} {total:>7.1f}")


if __name__ == "__main__":
    main()
