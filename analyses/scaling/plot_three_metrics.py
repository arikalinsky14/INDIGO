#!/usr/bin/env python3
"""IsoFLOP curves and N*(C) side by side, one column per metric.

Top row: each metric's IsoFLOP curves (Akima interpolated, N* starred).
Bottom row: N*(C) with the fitted power law, its 95% band and Chinchilla's
0.5. Same estimator as plot_porian.py. Each column is LABEL=FILE:METRIC, with
an optional @NOISE_FILE whose repeat seeds set that column's noise:

    python analyses/scaling/plot_three_metrics.py \\
        --col "cross-entropy=tuned/isoflop_tuned.json:ce" \\
        --col "ΔE, greedy=tuned/isoflop_tuned.json:pooled" \\
        --col "ΔE, light pipeline=confirm/isoflop_pipeline.json:pooled@pipe_noise.json" \\
        --output three_metrics.png
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scaling import porian as P  # noqa: E402

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#124782", "#0d366b"]
FIT_C = "#eb6834"
SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"


def value_of(run, metric):
    return run["val_loss"] if metric == "ce" else run["val_de"][metric]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color="#ebeae5", linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def fit(path, metric, noise_path, n_boot):
    src = json.load(open(path))
    by_b = collections.defaultdict(list)
    for r in src["runs"]:
        by_b[min(src["budgets"], key=lambda b: abs(b - r["flops"]))].append(r)
    clusters = collections.defaultdict(list)
    for r in json.load(open(noise_path))["runs"] if noise_path else src["runs"]:
        clusters[(round(r["flops"], -11), r["n_params"])].append(value_of(r, metric))
    noise = P.NoiseModel.from_clusters(list(clusters.values()))
    rungs = [P.fit_rung(b, [r["n_params"] for r in by_b[b]],
                        [value_of(r, metric) for r in by_b[b]],
                        d_vals=[r["passes"] for r in by_b[b]], noise=noise,
                        n_boot=n_boot, rng=np.random.default_rng(0))
             for b in sorted(by_b)]
    good = [r for r in rungs if r.usable]
    xs = np.array([r.budget for r in good])
    ns = np.array([r.n_star_median for r in good])
    sig = np.array([r.log_sigma for r in good])
    law = P.power_law_fit(xs, ns, sig)
    k = min(200, min(r.samples.size for r in good))
    grid = np.geomspace(xs.min() / 1.6, xs.max() * 1.6, 60)
    draws = [np.array([r.samples[i % r.samples.size] for r in good]) for i in range(k)]
    band = np.percentile([P.power_law_fit(xs, d, sig)(grid) for d in draws],
                         [2.5, 97.5], axis=0)
    exps = [P.power_law_fit(xs, d, sig).exponent for d in draws]
    return {"rungs": rungs, "xs": xs, "ns": ns, "sig": sig, "law": law,
            "grid": grid, "band": band, "sigma": noise.sigma_lo,
            "ci": np.percentile(exps, [2.5, 97.5])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--col", action="append", required=True,
                   help="LABEL=FILE:METRIC[@NOISE_FILE]")
    p.add_argument("--bootstrap-iters", type=int, default=P.BOOTSTRAP_ITERS)
    p.add_argument("--title", default="IsoFLOP curves and compute-optimal size, by loss")
    p.add_argument("--output", default="analyses/scaling/results/tuned/three_metrics.png")
    a = p.parse_args()

    cols = []
    for spec in a.col:
        label, rest = spec.split("=", 1)
        rest, _, noise = rest.partition("@")
        path, metric = rest.rsplit(":", 1)
        cols.append((label, fit(path, metric, noise or None, a.bootstrap_iters)))

    fig, axes = plt.subplots(2, len(cols), figsize=(5.0 * len(cols), 8.6),
                             facecolor=SURF, squeeze=False)
    n_lo = min(r.n_vals.min() for _, f in cols for r in f["rungs"])
    n_hi = max(r.n_vals.max() for _, f in cols for r in f["rungs"])
    for j, (label, f) in enumerate(cols):
        ax = axes[0][j]
        style(ax)
        for i, r in enumerate(f["rungs"]):
            col = RAMP[i % len(RAMP)]
            ax.scatter(r.n_vals, r.y_vals, s=20, color=col, zorder=3,
                       edgecolor=SURF, linewidth=0.7, label=f"C = {r.budget:.2g}")
            ax.plot(r.n_grid, r.y_grid, "-", color=col, linewidth=1.4, alpha=0.8, zorder=2)
            if r.usable:
                ax.plot([r.n_star_median], [r.y_star], "*", color=col, markersize=14,
                        markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
        ys = np.concatenate([r.y_vals for r in f["rungs"]])
        best = np.array([r.y_star for r in f["rungs"]])
        # Show the bottoms: clip the steep right ends rather than squash the minima.
        top = np.percentile(ys, 90)
        ax.set_ylim(best.min() - 0.12 * (top - best.min()), top)
        ax.set_xscale("log")
        ax.set_xlim(n_lo / 1.6, n_hi * 1.3)
        ax.set_xlabel("N (parameters)", color=INK2, fontsize=9)
        ax.set_title(label, color=INK, fontsize=12, loc="left", pad=8)
        if j == 0:
            ax.set_ylabel("validation value (lower is better)", color=INK2, fontsize=9)
            handles, names = ax.get_legend_handles_labels()

        ax = axes[1][j]
        style(ax)
        ax.fill_between(f["grid"], f["band"][0], f["band"][1], color="#d8d7d2",
                        alpha=0.55, linewidth=0, label="95% band")
        ax.plot(f["grid"], f["law"](f["grid"]), "--", color=FIT_C, linewidth=2,
                label=rf"$N^* \propto C^{{{f['law'].exponent:.2f}}}$")
        anchor = f["law"](np.sqrt(f["xs"].min() * f["xs"].max()))
        mid = np.sqrt(f["xs"].min() * f["xs"].max())
        ax.plot(f["grid"], anchor * (f["grid"] / mid) ** 0.5, "-.", color=REF,
                linewidth=1.4, label="Chinchilla 0.50")
        ax.errorbar(f["xs"], f["ns"], yerr=[f["ns"] - f["ns"] * np.exp(-f["sig"]),
                                            f["ns"] * np.exp(f["sig"]) - f["ns"]],
                    fmt="o", color=FIT_C, markersize=6, capsize=3, linewidth=1.2,
                    markeredgecolor=SURF, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("C (training FLOPs)", color=INK2, fontsize=9)
        lo, hi = f["ci"]
        ax.set_title(f"α = {f['law'].exponent:.2f}  [{lo:.2f}, {hi:.2f}]   "
                     f"{len(f['xs'])}/{len(f['rungs'])} curves",
                     color=INK, fontsize=11, loc="left", pad=6)
        if j == 0:
            ax.set_ylabel("N* (parameters)", color=INK2, fontsize=9)
        leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK2)
        print(f"{label:<28} alpha {f['law'].exponent:+.3f} [{lo:+.2f}, {hi:+.2f}] "
              f"{len(f['xs'])}/{len(f['rungs'])}  sigma {f['sigma']:.3f}")

    fig.suptitle(a.title, fontsize=14, color=INK, x=0.01, ha="left", y=1.03)
    leg = fig.legend(handles, names, fontsize=8.5, frameon=False, ncol=len(names),
                     loc="upper left", bbox_to_anchor=(0.01, 1.005))
    for t in leg.get_texts():
        t.set_color(INK2)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
