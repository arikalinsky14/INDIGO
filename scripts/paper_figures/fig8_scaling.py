#!/usr/bin/env python3
"""Fig. 8: compute-optimal scaling, LR-tuned IsoFLOP study (re-render).

Publication re-render of analyses/scaling/results/tuned/isoflop_tuned_pooled.png
(panels A and B of analyses/scaling/plot_porian.py, same estimator, same
bootstrap seed). Shows ONLY the tuned study: no untuned first sweep, no 0.92
exponent, no turn at 2.75e15.

Inputs
  analyses/scaling/results/tuned/isoflop_tuned.json      runs[].{n_params, flops,
                                                         val_de.pooled, passes}
  analyses/scaling/results/tuned/porian_fit_tuned.json   by_metric.pooled.laws.N_star
                                                         {exponent, ci_low, ci_high}

The exponent and interval printed on the figure are read from
porian_fit_tuned.json. The curves, stars, error bars and band are recomputed
with src.scaling.porian exactly as plot_porian.py does, and the script refuses
to write if the recomputed exponent disagrees with the JSON.

    python scripts/paper_figures/fig8_scaling.py
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import _style as S
from src.scaling import porian as P

FIT = S.REPO / "analyses/scaling/results/tuned/isoflop_tuned.json"
LAW = S.REPO / "analyses/scaling/results/tuned/porian_fit_tuned.json"
METRIC = "pooled"

LABELS = {
    "a_x": "Parameters, N",
    "a_y": r"Validation $\Delta E_{00}$",
    "b_x": "Training compute, C (FLOP)",
    "b_y": r"Compute-optimal size, $N^*$",
    "nstar": r"$N^*$",
    "edge": "minimum on edge (excluded)",
    "obs": r"$N^*$ per budget",
    "band": "95% CI",
    "law": r"$N^* \propto C^{{{e:.2f}}}$  [{lo:.2f}, {hi:.2f}]",
    "chin": r"Chinchilla, $C^{0.50}$",
    "budget": "C = {b}",
}
# Sequential single-hue ramp (light -> dark = small -> large budget); ordinal,
# readable under all common color-vision deficiencies.
RAMP = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
LAW_C = S.OI["vermilion"]


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return rf"${m}\times10^{{{int(e)}}}$" if m != "1" else rf"$10^{{{int(e)}}}$"


def main() -> None:
    S.apply()
    src = json.load(open(FIT))
    law = json.load(open(LAW))["by_metric"][METRIC]["laws"]["N_star"]

    by_b = collections.defaultdict(list)
    for r in src["runs"]:
        by_b[min(src["budgets"], key=lambda b: abs(b - r["flops"]))].append(r)
    budgets = sorted(by_b)
    clusters = collections.defaultdict(list)
    for r in src["runs"]:
        clusters[(round(r["flops"], -11), r["n_params"])].append(r["val_de"][METRIC])
    noise = P.NoiseModel.from_clusters(list(clusters.values()))
    rungs = [P.fit_rung(b, [r["n_params"] for r in by_b[b]],
                        [r["val_de"][METRIC] for r in by_b[b]],
                        d_vals=[r["passes"] for r in by_b[b]], noise=noise,
                        n_boot=P.BOOTSTRAP_ITERS, rng=np.random.default_rng(0))
             for b in budgets]
    good = [r for r in rungs if r.usable]
    xs = np.array([r.budget for r in good])
    ns = np.array([r.n_star_median for r in good])
    sig = np.array([r.log_sigma for r in good])
    n_law = P.power_law_fit(xs, ns, sig)
    if abs(n_law.exponent - law["exponent"]) > 5e-3:
        sys.exit(f"recomputed exponent {n_law.exponent:.4f} != JSON {law['exponent']:.4f}")
    n_draws = min(200, min(r.samples.size for r in good))
    grid = np.geomspace(xs.min() / 1.6, xs.max() * 1.6, 80)
    band = [P.power_law_fit(xs, np.array([r.samples[i % r.samples.size] for r in good]),
                            sig)(grid) for i in range(n_draws)]
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.DOUBLE, 6.4 * S.CM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.97, wspace=0.28)

    # (a) IsoFLOP curves
    for i, r in enumerate(rungs):
        c = RAMP[i]
        ax.plot(r.n_grid, r.y_grid, "--", color=c, lw=0.9, zorder=2)
        ax.scatter(r.n_vals, r.y_vals, s=9, color=c, edgecolor="white", lw=0.4,
                   zorder=3, label=LABELS["budget"].format(b=sci(r.budget)))
        if r.usable:
            ax.plot(r.n_star_median, r.y_star, "*", ms=8, color=c, mec=S.INK,
                    mew=0.5, zorder=4)
        else:
            ax.plot(r.n_star, r.y_star, "x", ms=5, color=c, mew=1.2, zorder=4)
    ax.plot([], [], "*", ms=7, color="white", mec=S.INK, mew=0.5, ls="none",
            label=LABELS["nstar"])
    ax.plot([], [], "x", ms=4.5, color=S.INK, mew=1.1, ls="none", label=LABELS["edge"])
    ax.set_xscale("log")
    n_lo = min(r.n_vals.min() for r in rungs); n_hi = max(r.n_vals.max() for r in rungs)
    ax.set_xlim(n_lo / 1.5, n_hi * 1.3)
    ax.set_xlabel(LABELS["a_x"]); ax.set_ylabel(LABELS["a_y"])
    ax.legend(loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.2, labelspacing=0.25)
    y0 = min(r.y_vals.min() for r in rungs); y1 = max(r.y_vals.max() for r in rungs)
    ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.42 * (y1 - y0))

    # (b) N*(C)
    bx.fill_between(grid, lo, hi, color=S.LIGHT, lw=0, zorder=1, label=LABELS["band"])
    bx.plot(grid, ns[0] * (grid / xs[0]) ** 0.5, ls=(0, (1.5, 1.5)), color=S.GREY,
            lw=0.9, zorder=2, label=LABELS["chin"])
    bx.plot(grid, n_law(grid), "-", color=LAW_C, lw=1.2, zorder=3,
            label=LABELS["law"].format(e=law["exponent"], lo=law["ci_low"], hi=law["ci_high"]))
    bx.errorbar(xs, ns, yerr=[ns - ns * np.exp(-sig), ns * np.exp(sig) - ns], fmt="o",
                ms=3.5, color=LAW_C, mec="white", mew=0.5, capsize=1.8, elinewidth=0.7,
                capthick=0.7, zorder=4, label=LABELS["obs"])
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(LABELS["b_x"]); bx.set_ylabel(LABELS["b_y"])
    h, l = bx.get_legend_handles_labels()
    order = [l.index(LABELS["obs"]), 2, 0, 1]
    bx.legend([h[i] for i in order], [l[i] for i in order], loc="upper left",
              borderaxespad=0.2, labelspacing=0.3)

    for a_, t in ((ax, "a"), (bx, "b")):
        a_.text(-0.13, 1.0, t, transform=a_.transAxes, fontweight="bold",
                fontsize=S.FONT_PT + 1, va="top")

    S.save(fig, "fig8_scaling")
    S.write_provenance(
        "Fig 8", "`scripts/paper_figures/fig8_scaling.py` -> `fig8_scaling.{pdf,png}`. "
        "Points/curves: `analyses/scaling/results/tuned/isoflop_tuned.json` runs[]."
        "{n_params, flops, passes, val_de.pooled} (55 runs, 6 budgets; top budget "
        "minimum on edge, excluded). Exponent label: `porian_fit_tuned.json` "
        "by_metric.pooled.laws.N_star.{exponent, ci_low, ci_high} = "
        f"{law['exponent']:.2f} [{law['ci_low']:.2f}, {law['ci_high']:.2f}], 5 usable budgets; "
        "curves/band recomputed with src.scaling.porian (bootstrap seed 0), "
        "checked against the JSON exponent.")
    print(f"alpha {n_law.exponent:.4f} (json {law['exponent']:.4f}); "
          f"CI [{law['ci_low']:.2f}, {law['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
