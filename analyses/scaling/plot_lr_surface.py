#!/usr/bin/env python3
"""The tuned learning rate as a surface over model size N and examples D.

Every tuned IsoFLOP point at its (N, D), coloured by its optimal learning
rate: filled where the optimum is bracketed (the points the law is fitted
on), hollow where it sits at the edge of its grid (a bound, not an optimum).
Contour lines are the fitted joint law lr = a N^b D^c, labelled with their
rate; thin grey lines join each IsoFLOP curve's points. D is the examples a
run saw (passes), so every point lands on its own curve.

    python analyses/scaling/plot_lr_surface.py \\
        --law analyses/scaling/results/lr_law_fit.json \\
        --tuned analyses/scaling/results/tuned/isoflop_tuned.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0a2a52"]
SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--law", default="analyses/scaling/results/lr_law_fit.json")
    p.add_argument("--tuned", default="analyses/scaling/results/tuned/isoflop_tuned.json",
                   help="for each point's passes and its IsoFLOP curve")
    p.add_argument("--output", default="analyses/scaling/results/tuned/lr_surface.png")
    a = p.parse_args()

    fit = json.load(open(a.law))
    law = fit["lr_vs_n_and_d"]
    tuned = json.load(open(a.tuned))
    budgets = sorted(tuned["budgets"])
    seed42 = [r for r in tuned["runs"] if r["seed"] == 42]

    pts = []
    for c in fit["configs"]:
        n, d = c["n_params"], c["train_examples"]
        # passes = D x epochs; find the run of this config to get it and its curve.
        match = [r for r in seed42 if r["n_params"] == n
                 and any(abs(r["passes"] - k * d) <= 0.01 * r["passes"] for k in (1, 2, 3))]
        if not match:
            continue
        r = match[0]
        b = min(budgets, key=lambda x: abs(np.log(x / r["flops"])))
        pts.append({"N": n, "D": r["passes"], "lr": c["lr_star"],
                    "ok": bool(c["usable"]), "rung": budgets.index(b)})
    print(f"{len(pts)} tuned points, {sum(q['ok'] for q in pts)} bracketed")

    cmap = LinearSegmentedColormap.from_list("lr", RAMP)
    lrs = np.array([q["lr"] for q in pts])
    norm = LogNorm(vmin=lrs.min() / 1.3, vmax=lrs.max() * 1.3)

    fig, ax = plt.subplots(figsize=(9.6, 6.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.7, which="major")
    ax.set_axisbelow(True)

    Ns = np.array([q["N"] for q in pts])
    Ds = np.array([q["D"] for q in pts])
    gn = np.geomspace(Ns.min() / 1.6, Ns.max() * 3.2, 200)
    gd = np.geomspace(Ds.min() / 1.6, Ds.max() * 1.6, 200)
    GN, GD = np.meshgrid(gn, gd)
    Z = law["coef"] * GN ** law["n_exponent"] * GD ** law["d_exponent"]
    levels = [v for v in (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
              if Z.min() < v < Z.max()]
    cs = ax.contour(GN, GD, Z, levels=levels, colors=REF, linewidths=1.0,
                    linestyles="-", zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.0e}".replace("e-0", "e-"), fontsize=8,
              colors=INK2, inline_spacing=4)

    by_rung = collections.defaultdict(list)
    for q in pts:
        by_rung[q["rung"]].append(q)
    for k, qs in sorted(by_rung.items()):
        qs = sorted(qs, key=lambda q: q["N"])
        ax.plot([q["N"] for q in qs], [q["D"] for q in qs], "-", color=FAINT,
                linewidth=2.2, zorder=2)
        far = max(qs, key=lambda q: q["N"])
        ax.annotate(f"C = {budgets[k]:.2g}", (far["N"], far["D"]), xytext=(7, -2),
                    textcoords="offset points", fontsize=8, color=INK2)

    for q in pts:
        col = cmap(norm(q["lr"]))
        if q["ok"]:
            ax.scatter(q["N"], q["D"], s=90, color=col, edgecolor=INK,
                       linewidth=0.8, zorder=4)
        else:
            ax.scatter(q["N"], q["D"], s=70, facecolor=SURF, edgecolor=col,
                       linewidth=2.2, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(gn.min(), gn.max())
    ax.set_ylim(gd.min(), gd.max())
    ax.set_xlabel("N, model parameters", color=INK2, fontsize=10)
    ax.set_ylabel("D, training examples seen", color=INK2, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("tuned learning rate", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    ax.scatter([], [], s=80, color=RAMP[3], edgecolor=INK, label="optimum bracketed (fitted)")
    ax.scatter([], [], s=60, facecolor=SURF, edgecolor=RAMP[3], linewidth=2,
               label="best rate at the grid edge (a bound)")
    ax.plot([], [], "-", color=REF, label="fitted law, rate labelled")
    ax.plot([], [], "-", color=FAINT, linewidth=2.2, label="IsoFLOP curve")
    leg = ax.legend(fontsize=8.5, frameon=True, facecolor=SURF, edgecolor="none", framealpha=1, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("Optimal learning rate falls with both model size and data",
                 fontsize=13.5, color=INK, x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.945, f"lr* = {law['coef']:.0f} · N^{law['n_exponent']:.2f} · "
             f"D^{law['d_exponent']:.2f}, fitted on {law['n_points']} bracketed points, "
             f"r² = {law['r2']:.2f}", fontsize=9.5, color=INK2)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
