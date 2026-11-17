#!/usr/bin/env python3
"""Fig. 4: INDIGO vs random proposals, the headline comparison.

(a) Cumulative distribution of the final DeltaE00 per test target, tiers A
    and B pooled (n = 1000 per run). Failed rows count as never reaching
    any threshold, so a curve tops out below 100%.
(b) Share of targets with DeltaE00 < THRESHOLD by target chroma bucket.
(c) The same share by tier (A: materials seen in training; B: pools with
    held-out real materials).
Point estimates and 95% bootstrap intervals in (b) and (c) are read from
results/paper/headline.json, never recomputed. The curves in (a) come from
the per-row files that headline.json was built from, and the script stops
if those rows disagree with headline.json.

    python scripts/paper_figures/fig4_headline.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import _style as S
import _runs as R

THRESHOLD = 1          # one of headline.json's thresholds: 1, 2, 3, 5
FLOOR = 1e-3           # DeltaE below this is drawn at the floor (log axis)

LABELS = {
    "a_x": r"Color error of the returned stack, $\Delta E_{00}$",
    "a_y": "Targets reached (%)",
    "b_y": rf"Targets with $\Delta E_{{00}}$ < {THRESHOLD} (%)",
    "b_x": "Target chroma",
    "c_x": "Test tier",
    "tierA": "A\nseen",
    "tierB": "B\nheld out",
    "floor": "≤0.001",
}


def main():
    S.apply()
    h = R.headline()
    data = {k: R.load(k, h) for k in R.RUNS}
    for k, d in data.items():
        R.check_against_headline(k, d, h)

    fig = plt.figure(figsize=(S.DOUBLE, 6.0 * S.CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.55], left=0.075, right=0.99,
                          bottom=0.245, top=0.97, wspace=0.34)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) ECDF
    for k, (name, col) in R.RUNS.items():
        de = data[k]["de"]
        x = np.sort(np.clip(de[np.isfinite(de)], FLOOR, None))
        y = np.arange(1, x.size + 1) / de.size * 100
        ax.step(np.r_[FLOOR, x], np.r_[0, y], where="post", color=col, lw=1.1, label=name)
    for t in (1, 2):
        ax.axvline(t, color=S.LIGHT, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR, 60)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: LABELS["floor"] if v <= FLOOR * 1.0001 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(LABELS["a_x"]); ax.set_ylabel(LABELS["a_y"])
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.2)

    # (b) by chroma, (c) by tier: estimates and intervals from headline.json
    key = f"pass_lt_{THRESHOLD:g}"
    order = list(R.RUNS)
    off = np.linspace(-0.24, 0.24, len(order))

    def dots(axis, groups, getter):
        for j, k in enumerate(order):
            est, lo, hi = zip(*[getter(k, g) for g in groups])
            x = np.arange(len(groups)) + off[j]
            est, lo, hi = (np.array(v) * 100 for v in (est, lo, hi))
            axis.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=3.4,
                          color=R.RUNS[k][1], mec="white", mew=0.4, elinewidth=0.8,
                          capsize=0, zorder=3)

    def by_chroma(k, b):
        c = h["runs"][k]["tiers"]["A+B"]["by_chroma"][b][key]
        return c["est"], c["lo"], c["hi"]

    def by_tier(k, t):
        c = h["runs"][k]["tiers"][t]["all"][key]
        return c["est"], c["lo"], c["hi"]

    dots(bx, R.BUCKETS, by_chroma)
    ns = [h["runs"]["INDIGO"]["tiers"]["A+B"]["by_chroma"][b]["n"] for b in R.BUCKETS]
    bx.set_xticks(range(3))
    bx.set_xticklabels([f"{R.BUCKET_LABEL[b]}\n(n = {n})" for b, n in zip(R.BUCKETS, ns)])
    bx.set_ylabel(LABELS["b_y"]); bx.set_xlabel(LABELS["b_x"])
    dots(cx, ["A", "B"], by_tier)
    nt = [h["runs"]["INDIGO"]["tiers"][t]["all"]["n"] for t in "AB"]
    cx.set_xticks(range(2))
    cx.set_xticklabels([f"{LABELS['tier' + t]}\n(n = {n})" for t, n in zip("AB", nt)])
    cx.set_xlabel(LABELS["c_x"])
    for a_ in (bx, cx):
        a_.set_ylim(0, 102)
        a_.yaxis.grid(True, color=S.LIGHT, lw=0.5); a_.set_axisbelow(True)
        a_.set_xlim(-0.6, len(a_.get_xticks()) - 0.4)
    cx.set_yticklabels([])

    for a_, t, x in ((ax, "a", -0.17), (bx, "b", -0.24), (cx, "c", -0.16)):
        a_.text(x, 1.04, t, transform=a_.transAxes, fontweight="bold",
                fontsize=S.FONT_PT + 1, va="top")

    S.save(fig, "fig4_headline")
    st = h["runs"]["INDIGO"]["settings"]
    S.write_provenance(
        "Fig 4", "`scripts/paper_figures/fig4_headline.py` -> `fig4_headline.{pdf,png}`. "
        "(a) per-row `delta_e` from `results/paper_runs/{strong_random, budget50_model, "
        "strong_model}/**/tier_{a,b}/rows.json`, rows 0-499 per tier (n = 1000 per run), loaded "
        "with scripts/paper_numbers.py load_run/de_of; failed rows = inf; checked equal to "
        "headline.json pass rates. (b, c) `results/paper/headline.json` runs.<run>.tiers."
        f"{{A+B.by_chroma.<bucket>, A.all, B.all}}.pass_lt_{THRESHOLD:g}.{{est, lo, hi}} "
        "(95% bootstrap, 2000 draws). Runs: random N=250, INDIGO N=50, INDIGO N=250; top-k "
        f"{st['top_k']}, {st['refine_iters']} refinement steps, seed {st['seed']}, checkpoint "
        f"{st['model_sha256']}.")


if __name__ == "__main__":
    main()
