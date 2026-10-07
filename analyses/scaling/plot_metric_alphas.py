#!/usr/bin/env python3
"""One IsoFLOP collection, every metric: does N*(C) agree across them?

Panel A: N*(C) per metric (CE, DeltaE in the low, mid and high chroma buckets,
and DeltaE pooled), each with its fitted power law. Panel B: the exponent
alpha with its 95% bootstrap interval per metric, against Chinchilla's 0.5.
Same estimator as plot_porian.py (Akima, boundary rejection, seed-noise
bootstrap, 1/sigma^2 fit), each metric with its own seed noise.

    python analyses/scaling/plot_metric_alphas.py \\
        --fit analyses/scaling/results/tuned/isoflop_tuned.json \\
        --output analyses/scaling/results/tuned/alpha_by_metric.png
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

METRICS = [("ce", "cross-entropy", "#0b0b0b"),
           ("low", "ΔE, low chroma", "#2a78d6"),
           ("mid", "ΔE, mid chroma", "#1c9e77"),
           ("high", "ΔE, high chroma", "#7b4fc9"),
           ("pooled", "ΔE, all colours pooled", "#eb6834")]
SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"


def value_of(run, metric):
    return run["val_loss"] if metric == "ce" else run["val_de"][metric]


def fit_metric(src, metric, n_boot):
    by_b = collections.defaultdict(list)
    for r in src["runs"]:
        by_b[min(src["budgets"], key=lambda b: abs(b - r["flops"]))].append(r)
    clusters = collections.defaultdict(list)
    for r in src["runs"]:
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
    exps = [P.power_law_fit(xs, np.array([r.samples[i % r.samples.size]
                                          for r in good]), sig).exponent
            for i in range(k)]
    return {"xs": xs, "ns": ns, "sig": sig, "law": law,
            "lo": float(np.percentile(exps, 2.5)),
            "hi": float(np.percentile(exps, 97.5)),
            "usable": len(good), "rungs": len(rungs),
            "sigma": float(noise.sigma_lo)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="analyses/scaling/results/tuned/isoflop_tuned.json")
    p.add_argument("--bootstrap-iters", type=int, default=P.BOOTSTRAP_ITERS)
    p.add_argument("--title", default="Compute-optimal model size by metric")
    p.add_argument("--output", default="analyses/scaling/results/tuned/alpha_by_metric.png")
    a = p.parse_args()
    src = json.load(open(a.fit))
    fits = {m: fit_metric(src, m, a.bootstrap_iters) for m, _, _ in METRICS}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in axes:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(FAINT)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color="#ebeae5", linewidth=0.8, which="both")
        ax.set_axisbelow(True)

    ax = axes[0]
    allx = np.concatenate([f["xs"] for f in fits.values()])
    g = np.geomspace(allx.min() / 1.4, allx.max() * 1.4, 40)
    for (m, lab, col), off in zip(METRICS, np.linspace(-0.06, 0.06, len(METRICS))):
        f = fits[m]
        x = f["xs"] * np.exp(off)          # small offset so markers do not overlap
        ax.errorbar(x, f["ns"], yerr=[f["ns"] - f["ns"] * np.exp(-f["sig"]),
                                      f["ns"] * np.exp(f["sig"]) - f["ns"]],
                    fmt="o", color=col, markersize=5, capsize=2.5, linewidth=1,
                    markeredgecolor=SURF, zorder=3)
        ax.plot(g, f["law"](g), "-", color=col, linewidth=1.6, alpha=0.85, zorder=2,
                label=f"{lab}  α = {f['law'].exponent:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("C (training FLOPs)", color=INK2, fontsize=9.5)
    ax.set_ylabel("N* (parameters)", color=INK2, fontsize=9.5)
    ax.set_title("A.  N*(C) on every metric", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[1]
    ys = np.arange(len(METRICS))[::-1]
    ax.axvline(0.5, color=REF, linestyle=(0, (5, 3)), linewidth=1.5,
               label="Chinchilla 0.50")
    for y, (m, lab, col) in zip(ys, METRICS):
        f = fits[m]
        e = f["law"].exponent
        ax.plot([f["lo"], f["hi"]], [y, y], "-", color=col, linewidth=3,
                alpha=0.55, solid_capstyle="round")
        ax.plot(e, y, "o", color=col, markersize=9, markeredgecolor=SURF, zorder=3)
        ax.text(max(f["hi"], e) + 0.03, y,
                f"{e:.2f} [{f['lo']:.2f}, {f['hi']:.2f}]   {f['usable']}/{f['rungs']} curves",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in METRICS], fontsize=9)
    hi = max(f["hi"] for f in fits.values())
    ax.set_xlim(0.3, hi + 0.75)
    ax.set_xlabel("α  (N* ∝ C^α), 95% bootstrap interval", color=INK2, fontsize=9.5)
    ax.set_title("B.  The exponent, per metric", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(a.title, fontsize=13.5, color=INK, x=0.008, ha="left", y=1.02)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")
    for m, lab, _ in METRICS:
        f = fits[m]
        print(f"  {lab:<26} alpha {f['law'].exponent:+.3f} [{f['lo']:+.2f}, "
              f"{f['hi']:+.2f}]  {f['usable']}/{f['rungs']}  sigma {f['sigma']:.3f}")


if __name__ == "__main__":
    main()
