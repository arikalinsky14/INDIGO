"""The LR law the sweep ran on, and the two things it does not know.

Panel A: every point in MEASURED_LR came off a coarse grid. The first sweep
tried three LRs and all three model sizes returned the SAME middle one, so it
measured no size dependence at all. The slope is set entirely by a later,
separate d512 re-measurement whose optimum landed on its grid's lower edge and
was therefore never bracketed. Leave-one-out slopes differ by 1.7x, which is
8.4x in the LR assigned to the smallest model in the sweep.

Panel B: all of that tuning happened at ONE dataset size, D = 614,400 examples.
lr_for() takes only N, so every run at a given N gets the same LR whether it
trains on 2.5M or 22.4M examples.
"""
import json, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.scaling.configs import MEASURED_LR, fit_lr_law, lr_for

FIT = sys.argv[1] if len(sys.argv) > 1 else "analyses/scaling/results/isoflop_fit.json"
OUT = "analyses/scaling/results/lr_law.png"
runs = json.load(open(FIT))["runs"]

BLUE, ORANGE = "#2a78d6", "#eb6834"        # categorical slots 1 and 2, validated
SURF, INK, INK2, REF, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#d8d7d2"

D_TUNED = 614_400
GRID_V1 = np.geomspace(1e-4, 3e-3, 3)      # first sweep: --lr-min 1e-4 --lr-max 3e-3 --n-lrs 3
GRID_V2 = np.geomspace(1e-4, 5e-4, 4)      # d512 re-run: --lr-min 1e-4 --lr-max 5e-4 --n-lrs 4

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.8); ax.set_axisbelow(True)

# ---- A: the law and how little holds it up ---------------------------------
ax = axes[0]
for n, _ in MEASURED_LR[:2]:
    ax.plot([n] * len(GRID_V1), GRID_V1, "o", mfc="none", mec=REF, mew=1.3,
            markersize=8, zorder=2)
n512 = MEASURED_LR[2][0]
ax.plot([n512] * len(GRID_V2), GRID_V2, "o", mfc="none", mec=REF, mew=1.3,
        markersize=8, zorder=2)

ns = np.array([p[0] for p in MEASURED_LR], float)
lrs = np.array([p[1] for p in MEASURED_LR], float)
grid = np.geomspace(6e4, 2.2e7, 50)
a, b = fit_lr_law()
ax.plot(grid, np.exp(a) * grid ** b, "-", color=BLUE, linewidth=2.2, zorder=3,
        label=f"law the sweep used   b = {b:+.2f}")
# Leave-one-out: drop each of the two duplicated points in turn.
for drop, style, who in ((1, (0, (5, 3)), "d256"), (0, (0, (1.5, 2.5)), "d128")):
    pts = [p for i, p in enumerate(MEASURED_LR) if i != drop]
    a2, b2 = fit_lr_law(pts)
    ax.plot(grid, np.exp(a2) * grid ** b2, ls=style, color=REF, linewidth=1.8,
            zorder=2, label=f"drop the {who} point   b = {b2:+.2f}")
ax.plot(ns, lrs, "o", color=BLUE, markersize=11, markeredgecolor=SURF,
        markeredgewidth=1.6, zorder=5)

ax.annotate("both sizes returned\nthe SAME grid point", xy=(7.67e5, 5.477e-4),
            xytext=(1.05e6, 1.25e-3), fontsize=8.5, color=INK, ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.9,
                            connectionstyle="arc3,rad=0.22"))
ax.annotate("lower edge of its own grid:\nnever bracketed, yet it sets\nthe entire slope",
            xy=(1.62e7, 1.06e-4), xytext=(2.0e6, 4.2e-3), fontsize=8.5, color=INK,
            ha="left", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.9,
                                       connectionstyle="arc3,rad=0.3"))
ax.set_ylim(6.0e-5, 6.0e-2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N (parameters)", color=INK2, fontsize=9.5)
ax.set_ylabel("learning rate", color=INK2, fontsize=9.5)
ax.set_title("A.  Three measurements, two distinct values", color=INK,
             fontsize=11.5, pad=10, loc="left")
ax.plot([], [], "o", mfc="none", mec=REF, mew=1.3, markersize=8,
        label="an LR the grid tried and rejected")
leg = ax.legend(fontsize=8.5, frameon=False, loc="lower left")
for t in leg.get_texts(): t.set_color(INK2)

# ---- B: one D tuned, two decades run ---------------------------------------
ax = axes[1]
by_n = {}
for r in runs: by_n.setdefault(r["n_params"], []).append(r["passes"])
for n, ds in by_n.items():
    ds = sorted(set(ds))
    if len(ds) > 1:
        ax.plot([min(ds), max(ds)], [n, n], "-", color=BLUE, linewidth=1.6,
                alpha=0.55, zorder=2)
ax.plot([r["passes"] for r in runs], [r["n_params"] for r in runs], "o",
        color=BLUE, markersize=7, markeredgecolor=SURF, markeredgewidth=1.2,
        zorder=3, label="the 48 sweep runs")
ax.axvline(D_TUNED, ls=(0, (5, 3)), color=ORANGE, linewidth=2, zorder=2)
ax.plot([D_TUNED] * 3, ns, "s", color=ORANGE, markersize=9,
        markeredgecolor=SURF, markeredgewidth=1.4, zorder=4,
        label="where LR was actually tuned")
ax.annotate("D = 614k, the only dataset size\nany LR was ever measured at",
            xy=(D_TUNED, 1.05e5), xytext=(8.0e5, 9.2e4), fontsize=8.5,
            color=ORANGE, ha="left")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("D (example-passes)", color=INK2, fontsize=9.5)
ax.set_ylabel("N (parameters)", color=INK2, fontsize=9.5)
spread = max(max(v) / min(v) for v in by_n.values())
ax.set_title(f"B.  The axis the law cannot see: one LR per N, up to {spread:.0f}x in D",
             color=INK, fontsize=11.5, pad=10, loc="left")
leg = ax.legend(fontsize=8.5, frameon=False, loc="lower right")
for t in leg.get_texts(): t.set_color(INK2)

fig.suptitle("The learning-rate law: what it rests on, and what it leaves out",
             fontsize=13.5, color=INK, x=0.008, ha="left", y=1.035)
fig.text(0.008, 0.975, "Porian correction #3 is applied on N only. Runs sharing an N share an LR, "
         "however long they train.", fontsize=9.5, color=INK2, ha="left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote", OUT)
