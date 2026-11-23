#!/usr/bin/env python3
"""Fig. 5: the same target, two proposers. One point per test target.

x: final DeltaE00 with random proposals (N = 250); y: with INDIGO (N = 250).
Same rows, same pools, same simulator, same refinement budget; only the
proposer differs. Points below the diagonal are targets INDIGO solved
better. Colored by target chroma bucket. Tiers A and B pooled (n = 1000).
Failed rows sit on the "failed" band. Dashed lines mark DeltaE00 = 1; the
corner counts are rows in each quadrant.

The share of targets INDIGO solves better belongs in the caption, not on
the figure: headline.json comparisons["INDIGO - random"]["A+B"].rows_better
(strict, per row). The script recomputes it from the rows, stops if they
disagree, and records it in the provenance line.

    python scripts/paper_figures/fig5_paired.py
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import _style as S
import _runs as R

FLOOR = 1e-3
FAIL = 250.0            # where failed rows are drawn
TOP = 400.0
X_RUN, Y_RUN = "random", "INDIGO"
BUCKET_COL = {"C<8": S.OI["green"], "8<=C<30": S.OI["orange"], "C>=30": S.OI["purple"]}

LABELS = {
    "x": r"$\Delta E_{00}$, baseline",
    "y": r"$\Delta E_{00}$, INDIGO",
    "fail": "failed",
    "floor": "≤0.001",
    "quad": "{n}",
}


def place(de):
    out = np.clip(de, FLOOR, None).astype(float)
    out[~np.isfinite(de)] = FAIL
    return out


def main():
    S.apply()
    h = R.headline()
    dx, dy = R.load(X_RUN, h), R.load(Y_RUN, h)
    for k, d in ((X_RUN, dx), (Y_RUN, dy)):
        R.check_against_headline(k, d, h)
    if not (np.array_equal(dx["idx"], dy["idx"]) and np.array_equal(dx["tier"], dy["tier"])):
        sys.exit("rows are not paired")
    better = h["comparisons"][f"{Y_RUN} - {X_RUN}"]["A+B"]["rows_better"]
    if abs(np.mean(dy["de"] < dx["de"]) - better) > 1e-9:
        sys.exit("rows_better disagrees with headline.json")

    x, y = place(dx["de"]), place(dy["de"])
    any_fail = bool(np.any(~np.isfinite(dx["de"])) or np.any(~np.isfinite(dy["de"])))
    top = TOP if any_fail else 80.0          # no failed rows: no "failed" band
    bk = dy["bucket"]

    fig, ax = plt.subplots(figsize=(S.SINGLE, S.SINGLE * 0.98))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.97)
    for b in R.BUCKETS:
        m = bk == b
        ax.scatter(x[m], y[m], s=4, lw=0, alpha=0.6, color=BUCKET_COL[b], rasterized=True,
                   label=f"{R.BUCKET_LABEL[b]} (n = {m.sum()})", zorder=3)
    diag_hi = FAIL / 1.5 if any_fail else top
    ax.plot([FLOOR / 1.4, diag_hi], [FLOOR / 1.4, diag_hi], color=S.INK, lw=0.6, zorder=2)
    for v in (1,):
        ax.axvline(v, color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axhline(v, color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    if any_fail:
        ax.axvspan(FAIL / 1.5, TOP, color="#F2F2F2", lw=0, zorder=0)
        ax.axhspan(FAIL / 1.5, TOP, color="#F2F2F2", lw=0, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(FLOOR / 1.4, top); ax.set_ylim(FLOOR / 1.4, top)
    ticks = [1e-3, 1e-2, 1e-1, 1, 10] + ([FAIL] if any_fail else [])
    fmt = FuncFormatter(lambda v, _: LABELS["floor"] if v <= FLOOR * 1.0001
                        else LABELS["fail"] if v >= FAIL * 0.999 else f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())
    ax.set_aspect("equal")
    ax.set_xlabel(LABELS["x"]); ax.set_ylabel(LABELS["y"])

    # quadrant counts at DeltaE = 1
    xd, yd = dx["de"], dy["de"]
    q = {(0, 0): ((xd < 1) & (yd < 1)).sum(), (1, 0): ((xd >= 1) & (yd < 1)).sum(),
         (0, 1): ((xd < 1) & (yd >= 1)).sum(), (1, 1): ((xd >= 1) & (yd >= 1)).sum()}
    xr = 120 if any_fail else top / 1.3
    pos = {(0, 0): (1.6e-3, 0.62, "left", "top"), (1, 0): (xr, 0.62, "right", "top"),
           (0, 1): (1.6e-3, 1.6, "left", "bottom"), (1, 1): (xr, 1.6, "right", "bottom")}
    for kq, n in q.items():
        px, py, ha, va = pos[kq]
        ax.text(px, py, LABELS["quad"].format(n=n), ha=ha, va=va, fontsize=S.SMALL_PT,
                fontweight="bold", color=S.INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    lg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), markerscale=2.2,
                   handletextpad=0.1, borderaxespad=0.3,
                   labelspacing=0.2, fontsize=S.SMALL_PT - 0.5)
    for hd in lg.legend_handles:
        hd.set_alpha(1)

    S.save(fig, "fig5_paired")
    S.write_provenance(
        "Fig 5", "`scripts/paper_figures/fig5_paired.py` -> `fig5_paired.{pdf,png}` (8.3 cm). "
        "Per-row `delta_e` and `target_lab` (chroma bucket) from `results/paper_runs/strong_random` "
        "(x) and `results/paper_runs/strong_model` (y), tiers A+B rows 0-499 each, paired by "
        "(tier, row_idx), loaded with paper_numbers.load_run; failed rows on the 'failed' band "
        f"({int(np.sum(~np.isfinite(xd)))} random, {int(np.sum(~np.isfinite(yd)))} INDIGO). "
        "Caption number, not drawn: rows_better = headline.json comparisons['INDIGO - random']['A+B'].rows_better "
        f"= {better:.3f} (recomputed and checked). Quadrant counts at DeltaE00 = 1 from the rows: "
        f"both < 1 {q[(0, 0)]}, INDIGO only {q[(1, 0)]}, random only {q[(0, 1)]}, neither {q[(1, 1)]}.")


if __name__ == "__main__":
    main()
