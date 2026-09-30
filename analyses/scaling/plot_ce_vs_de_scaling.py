"""CE alongside DeltaE: the control the study never ran.

The sweep only ever fitted DeltaE, so "CE behaves normally and DeltaE does not"
had never been tested. Running the IDENTICAL IsoFLOP procedure on val_loss is
that test, and it comes back negative: both metrics give alpha near 0.9 rather
than Chinchilla's 0.5, and both turn at C = 2.75e15.

Panel A  N*(C) for each metric, with a Chinchilla alpha=0.5 slope for reference.
Panel B  best CE at each budget, seed-averaged.
Panel C  best DeltaE at each budget, seed-averaged.

B and C are separate panels, not one dual-axis plot: CE and DeltaE have
different units and a shared y-scale would be meaningless.

Seed-averaged, not min-over-runs: the minimum of 8 runs is biased downward and
the bias grows with how many seeds a config had. The 2.75e15 rung's raw
minimum of 9.17 is one seed of a three-seed cluster averaging 10.04.
"""
import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIT = sys.argv[1] if len(sys.argv) > 1 else "analyses/scaling/results/isoflop_fit.json"
d = json.load(open(FIT))
CE_C, DE_C = "#2a78d6", "#eb6834"          # categorical slots 1 & 2, validated
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
REF = "#8a8880"

by_b = collections.defaultdict(list)
for r in d["runs"]:
    by_b[min(d["budgets"], key=lambda x: abs(x - r["flops"]))].append(r)

def isoflop(pts, val):
    x = np.log([p["n_params"] for p in pts]); y = np.array([val(p) for p in pts])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0: return None
    ns = math.exp(-b / (2*a))
    lo, hi = min(p["n_params"] for p in pts), max(p["n_params"] for p in pts)
    return ns if lo <= ns <= hi else None

def seed_avg_best(pts, val):
    g = collections.defaultdict(list)
    for p in pts: g[p["n_params"]].append(val(p))
    return min(sum(v)/len(v) for v in g.values())

buds = sorted(by_b)
ce_ns = [(b, isoflop(by_b[b], lambda p: p["val_loss"])) for b in buds]
de_ns = [(b, isoflop(by_b[b], lambda p: p["val_de"]["pooled"])) for b in buds]
ce_best = [seed_avg_best(by_b[b], lambda p: p["val_loss"]) for b in buds]
de_best = [seed_avg_best(by_b[b], lambda p: p["val_de"]["pooled"]) for b in buds]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.8); ax.set_axisbelow(True)

# ---- A: N*(C), both metrics -------------------------------------------------
ax = axes[0]
alphas = {}
for pts, col, lab in ((ce_ns, CE_C, "cross-entropy"), (de_ns, DE_C, r"$\Delta E_{00}$")):
    ok = [(b, n) for b, n in pts if n]
    xs = [b for b, _ in ok]; ys = [n for _, n in ok]
    al = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    alphas[lab] = al
    ax.plot(xs, ys, "-o", color=col, linewidth=2.2, markersize=9,
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=3,
            label=f"{lab}   " + r"$\alpha$ = " + f"{al:+.2f}")
anchor_x = np.array([3.0e14, 8.3e15])
anchor_y = 2.7e5 * (anchor_x / 3.0e14) ** 0.5
ax.plot(anchor_x, anchor_y, ls=(0, (5, 3)), color=REF, linewidth=2, zorder=2,
        label=r"Chinchilla  $\alpha$ = 0.50")
ax.set_xscale("log"); ax.set_yscale("log")
# Explicit decade ticks: matplotlib's log minor labels collide at this width.
ax.set_xticks([1e14, 1e15, 1e16])
ax.set_xticklabels([r"$10^{14}$", r"$10^{15}$", r"$10^{16}$"])
ax.set_xticks([], minor=True)
ax.set_yticks([3e5, 1e6, 3e6])
ax.set_yticklabels(["0.3M", "1M", "3M"])
ax.set_yticks([], minor=True)
ax.set_xlabel("C (FLOPs)", color=INK, fontsize=10.5)
ax.set_ylabel(r"$N^*$ (parameters)", color=INK, fontsize=10.5)
ax.set_title("A.  Both metrics scale like C, not " + r"$\sqrt{C}$",
             color=INK, fontsize=11.5, loc="left", pad=8)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

# ---- B, C: best achievable per budget --------------------------------------
for ax, vals, col, name, unit in (
        (axes[1], ce_best, CE_C, "B.  Best cross-entropy", "val_loss"),
        (axes[2], de_best, DE_C, "C.  Best " + r"$\Delta E_{00}$", r"val $\Delta E_{00}$ (median)")):
    ax.plot(buds, vals, "-o", color=col, linewidth=2.2, markersize=9,
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
    j = int(np.argmin(vals))
    ax.plot([buds[j]], [vals[j]], "*", color=col, markersize=19,
            markeredgecolor=SURF, markeredgewidth=1.2, zorder=4)
    ax.annotate(f"best at C = {buds[j]:.2e}", (buds[j], vals[j]),
                textcoords="offset points", xytext=(-8, 26), color=INK,
                ha="center",
                fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
    ax.set_xscale("log")
    ax.set_xlabel("C (FLOPs)", color=INK, fontsize=10.5)
    ax.set_ylabel(unit + "  (lower is better)", color=INK, fontsize=10.5)
    ax.set_title(name + " — turns at the same budget", color=INK,
                 fontsize=11.5, loc="left", pad=8)

fig.text(0.006, 0.975,
         "Cross-entropy is not the well-behaved control: it breaks where "
         r"$\Delta E_{00}$ breaks",
         color=INK, fontsize=13.5, ha="left", va="top")
fig.text(0.006, 0.925,
         "Same IsoFLOP procedure, same 48 runs, final checkpoint. "
         "Best-per-budget is seed-averaged, not min-over-runs.",
         color=INK2, fontsize=9.5, ha="left", va="top")
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = "analyses/scaling/results/ce_vs_de_scaling.png"
fig.savefig(out, dpi=170, facecolor=SURF)
print("wrote", out)
for k, v in alphas.items(): print(f"  alpha {k}: {v:+.3f}")
print(f"  CE  best-per-budget: " + ", ".join(f"{v:.4f}" for v in ce_best))
print(f"  dE  best-per-budget: " + ", ".join(f"{v:.3f}" for v in de_best))
