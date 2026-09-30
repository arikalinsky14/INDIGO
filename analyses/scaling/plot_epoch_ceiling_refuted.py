"""Figure: the epoch-ceiling theory is refuted.

Left  -- epochs every one of the 48 sweep runs actually saw. 47 of 48 are
         under one epoch, so they never revisited a single example, and the
         deepest run reached 1.45 epochs.
Right -- three model sizes held EXACTLY fixed while D grows. Each improves,
         bottoms out, then gets worse, entirely within the no-repeat region.

Repetition cannot explain degradation that happens before anything repeats.
Colours are the dataviz skill's ordinal blue ramp (steps 250/450/650),
validated with scripts/validate_palette.js --ordinal: monotone lightness,
all adjacent gaps >= 0.06, light end 2.06:1 on the light surface.
"""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import sys
FIT = sys.argv[1] if len(sys.argv) > 1 else "analyses/scaling/results/isoflop_fit.json"
d = json.load(open(FIT))
CORPUS = 39_980_000
RAMP = ["#86b6ef", "#2a78d6", "#104281"]     # 3 validated ordinal steps
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ACCENT = "#e34948"

runs = d["runs"]
by_n = collections.defaultdict(lambda: collections.defaultdict(list))
for r in runs:
    by_n[r["n_params"]][r["passes"]].append(r)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.8); ax.set_axisbelow(True)

# ---- A: epochs actually seen ------------------------------------------------
ax = axes[0]
eps = sorted(r["passes"] / CORPUS for r in runs)
ax.scatter(eps, range(len(eps)), s=30, color=RAMP[1], edgecolor=SURF,
           linewidth=1.1, zorder=3)
ax.axvline(1.0, color=ACCENT, linewidth=2, zorder=2)
ax.text(0.97, 24, "1 epoch — first repeat", color=ACCENT, fontsize=9.5,
        rotation=90, va="center", ha="right")
ax.axvline(1.5, color=INK2, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
ax.text(1.47, 24, "old EPOCH_CEILING", color=INK2, fontsize=9.5,
        rotation=90, va="center", ha="right")
ax.set_xlabel("epochs over the 40M corpus", color=INK, fontsize=10.5)
ax.set_ylabel("the 48 sweep runs, sorted by epochs", color=INK, fontsize=10.5)
ax.set_title("Every model saw its data at most once", color=INK,
             fontsize=12.5, loc="left", pad=8)
ax.set_xlim(-0.03, 1.72); ax.set_ylim(-2, 50)
ax.text(0.97, 0.06, f"47 of 48 runs under 1 epoch\ndeepest run {max(eps):.2f} epochs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.55", fc="#f4f3ef", ec="#e2e1db"))

# ---- B: iso-N, only the three series with 3 D-points ------------------------
ax = axes[1]
for i, n in enumerate([456549, 767013, 1472773]):
    ds = sorted(by_n[n])
    xs = [p / 1e6 for p in ds]
    ys = [sum(x["val_de"]["pooled"] for x in by_n[n][p]) / len(by_n[n][p]) for p in ds]
    ax.plot(xs, ys, "-o", color=RAMP[i], linewidth=2.2, markersize=9,
            markeredgecolor=SURF, markeredgewidth=1.6, zorder=3)
    # label at the FIRST point, where the three series are well separated;
    # at the right-hand end they converge and the labels collide
    ax.annotate(f"N = {n/1e6:.2f}M", (xs[0], ys[0]), textcoords="offset points",
                xytext=(9, 5), color=RAMP[i], fontsize=10.5, fontweight="bold")
    # mark each curve's own best, and shade what happens after it
    jbest = min(range(len(ys)), key=lambda j: ys[j])
    ax.plot([xs[jbest]], [ys[jbest]], "*", color=RAMP[i], markersize=17,
            markeredgecolor=SURF, markeredgewidth=1.2, zorder=4)
ax.set_xscale("log")
ax.set_xlim(0.32, 34)
ax.set_ylim(9.2, 23.5)
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("D — unique examples seen (millions)", color=INK, fontsize=10.5)
ax.set_ylabel(r"val $\Delta E_{00}$ (median, lower is better)", color=INK, fontsize=10.5)
ax.set_title("Same model, more UNIQUE data, worse result   "
             r"($\star$ = each curve's best)", color=INK,
             fontsize=12.5, loc="left", pad=22)
sec = ax.secondary_xaxis("top", functions=(lambda v: v * 1e6 / CORPUS,
                                           lambda v: v * CORPUS / 1e6))
sec.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
sec.xaxis.set_major_formatter(FixedFormatter(["0.02", "0.05", "0.10", "0.20", "0.50"]))
sec.set_xlabel("the same axis in epochs — nothing here repeats data",
               color=INK2, fontsize=9.5, labelpad=4)
sec.tick_params(colors=INK2, labelsize=8.5)
sec.spines["top"].set_color("#d8d7d2")

fig.suptitle("Degradation is NOT caused by repeating data — the epoch-ceiling theory is refuted",
             color=INK, fontsize=14, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "analyses/scaling/results/epoch_ceiling_refuted.png"
fig.savefig(out, dpi=170, facecolor=SURF)
print("wrote", out)
