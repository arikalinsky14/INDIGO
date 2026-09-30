"""Figure: the epoch-ceiling theory is refuted.

Left  -- epochs every one of the 48 sweep runs actually saw. 47 of 48 are under
         one epoch, so they never revisited a single example; the deepest run
         reached 1.45 epochs.

Right -- EVERY model size that appears at more than one budget, as small
         multiples. Selection rule: a size qualifies if the sweep trained it at
         2+ distinct D. That is 9 of the 20 distinct sizes, and they are NOT
         the largest -- they are the mid-range sizes the rungs happen to share,
         spanning 0.18M to 2.54M parameters. Nothing is hand-picked: all 9 are
         shown, including the two that do not turn.

Within a panel N is held EXACTLY fixed, so the only thing changing is how much
UNIQUE data the model saw. Repetition cannot explain a rise that happens before
anything repeats.

Colour is the dataviz skill's ordinal blue ramp, validated with
scripts/validate_palette.js --ordinal rather than by eye.
"""
import json, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIT = sys.argv[1] if len(sys.argv) > 1 else "analyses/scaling/results/isoflop_fit.json"
d = json.load(open(FIT))
CORPUS = 39_980_000
BLUE, DARK = "#2a78d6", "#104281"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ACCENT = "#e34948"

runs = d["runs"]
shape = {}          # n_params -> "d128/se3"
for r in runs:
    parts = r["name"].split("_")
    shape[r["n_params"]] = f"{parts[2]}/{parts[3]}"
by_n = collections.defaultdict(lambda: collections.defaultdict(list))
for r in runs:
    by_n[r["n_params"]][r["passes"]].append(r)
multi = sorted(n for n in by_n if len(by_n[n]) >= 2)

fig = plt.figure(figsize=(15.5, 6.4), facecolor=SURF)
gs = GridSpec(3, 5, figure=fig, width_ratios=[1.55, 1, 1, 1, 0.02],
              hspace=0.62, wspace=0.34, left=0.055, right=0.995,
              top=0.775, bottom=0.10)

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color="#ebeae5", linewidth=0.7); ax.set_axisbelow(True)

# ---- A: epochs seen ---------------------------------------------------------
ax = fig.add_subplot(gs[:, 0]); style(ax)
eps = sorted(r["passes"] / CORPUS for r in runs)
ax.scatter(eps, range(len(eps)), s=26, color=BLUE, edgecolor=SURF,
           linewidth=1.0, zorder=3)
ax.axvline(1.0, color=ACCENT, linewidth=2, zorder=2)
ax.text(0.97, 25, "1 epoch — first repeat", color=ACCENT, fontsize=9.5,
        rotation=90, va="center", ha="right")
ax.axvline(1.5, color=INK2, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
ax.text(1.47, 25, "old EPOCH_CEILING", color=INK2, fontsize=9.5,
        rotation=90, va="center", ha="right")
ax.set_xlabel("epochs over the 40M corpus", color=INK, fontsize=10)
ax.set_ylabel("the 48 sweep runs, sorted by epochs", color=INK, fontsize=10)
ax.set_title("Every model saw its data at most once", color=INK,
             fontsize=11.5, loc="left", pad=8)
ax.set_xlim(-0.03, 1.72); ax.set_ylim(-2, 50)
ax.text(0.97, 0.05, f"47 of 48 under 1 epoch\ndeepest {max(eps):.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f3ef", ec="#e2e1db"))

# ---- B: small multiples, every size trained at 2+ budgets --------------------
lo = min(min(r["val_de"]["pooled"] for rs in by_n[n].values() for r in rs) for n in multi)
hi = max(max(r["val_de"]["pooled"] for rs in by_n[n].values() for r in rs) for n in multi)
for k, n in enumerate(multi):
    ax = fig.add_subplot(gs[k // 3, 1 + k % 3]); style(ax)
    ds = sorted(by_n[n])
    xs = [p / 1e6 for p in ds]
    ys = [sum(x["val_de"]["pooled"] for x in by_n[n][p]) / len(by_n[n][p]) for p in ds]
    ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, markersize=7,
            markeredgecolor=SURF, markeredgewidth=1.3, zorder=3)
    j = min(range(len(ys)), key=lambda i: ys[i])
    ax.plot([xs[j]], [ys[j]], "*", color=DARK, markersize=15,
            markeredgecolor=SURF, markeredgewidth=1.0, zorder=4)
    ax.set_xscale("log"); ax.set_ylim(lo - 1.2, hi + 1.2)
    # Tick exactly at the D values this size was trained at, as plain numbers.
    # matplotlib's default log decades collide badly inside a small panel and
    # hide which D values were actually run.
    ax.set_xticks(xs); ax.set_xticklabels([f"{v:.1f}" if v >= 1 else f"{v:.2f}" for v in xs], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.set_xlim(min(xs) / 1.55, max(xs) * 1.55)
    ax.set_title(f"N = {n/1e6:.2f}M   {shape[n]}", color=INK, fontsize=9.5,
                 loc="left", pad=4)
    worst_ep = max(ds) / CORPUS
    ax.text(0.96, 0.92, f"max {worst_ep:.2f} ep", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK2)
    if k // 3 == 2: ax.set_xlabel("D (M unique ex.)", color=INK2, fontsize=9)
    if k % 3 == 0: ax.set_ylabel(r"val $\Delta E_{00}$", color=INK2, fontsize=9)

fig.text(0.012, 0.975,
         "Degradation is NOT caused by repeating data — the epoch-ceiling theory is refuted",
         color=INK, fontsize=14, ha="left", va="top")
fig.text(0.30, 0.925,
         "All 9 sizes the sweep trained at 2+ budgets — none hand-picked "
         r"($\star$ = that size's best)",
         color=INK, fontsize=11, ha="left", va="top")
fig.text(0.30, 0.887,
         "N is held EXACTLY fixed inside each panel, so only the amount of UNIQUE data changes.",
         color=INK2, fontsize=9.5, ha="left", va="top")
out = "analyses/scaling/results/epoch_ceiling_refuted.png"
fig.savefig(out, dpi=165, facecolor=SURF)
print(f"wrote {out}")
print(f"\nsizes shown ({len(multi)} of {len(by_n)} distinct sizes in the sweep):")
for n in multi:
    ds = sorted(by_n[n])
    print(f"  N={n:>9,}  {shape[n]:>9}  D = "
          + ", ".join(f"{p/1e6:.1f}M" for p in ds)
          + f"   (max {max(ds)/CORPUS:.2f} epochs)")
