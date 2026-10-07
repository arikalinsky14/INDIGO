#!/usr/bin/env python3
"""The tuned learning-rate law, lr(N, D) = a N^b D^c, and the data under it.

  A  Each tuned configuration's optimum (Akima argmin over its rates) against
     N, coloured by IsoFLOP rung. Filled: bracketed, so the law is fitted on
     it. Hollow: optimum on a grid edge, excluded. Dashed lines: the law along
     each rung, where D falls as N rises (D = C / FLOPs-per-example). Diamonds:
     the rates stage 3 ran on the rung the law extrapolates to. Grey: the old
     three-point law in N alone that the first sweep ran on.
  B  Measured over predicted for every configuration. The band is one stage-2
     grid step (1.85x); inside it the law is as good as the grid can tell.
  C  The landscape: every cell's DeltaE against its rate in units of the law's
     prediction, minus the cell's best. If the law is right the minima line up
     at 1; the rise to the right is the divergence cliff.

    python analyses/scaling/plot_lr_law_tuned.py \\
        --results-dir outputs/lr_search/cross_attn \\
        --law analyses/scaling/results/lr_law_fit.json \\
        --tuned analyses/scaling/results/tuned/isoflop_tuned.json \\
        --output analyses/scaling/results/tuned/lr_law_tuned.png
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from fit_lr_law import load_cells, tune_per_config     # noqa: E402
from lr_edges import arch_for                          # noqa: E402
from src.scaling.flops import train_flops_per_example  # noqa: E402

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#124782", "#0d366b"]
FIT_C, REF = "#eb6834", "#8a8880"
SURF, INK, INK2, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
GRID_STEP = 1.85


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.8, which="both")
    ax.set_axisbelow(True)


def legend(ax, **kw):
    leg = ax.legend(fontsize=7.5, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--law", default="analyses/scaling/results/lr_law_fit.json")
    p.add_argument("--tuned", default="analyses/scaling/results/tuned/isoflop_tuned.json",
                   help="for the rung budgets and the rates stage 3 ran on")
    p.add_argument("--metric", default="delta_e", choices=["delta_e", "val_loss"])
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--output", default="analyses/scaling/results/tuned/lr_law_tuned.png")
    a = p.parse_args()

    law = json.load(open(a.law)).get("lr_vs_n_and_d", {})
    if law.get("status") != "ok":
        sys.exit(f"no lr(N, D) law in {a.law}: {law}")
    A, b, c = law["coef"], law["n_exponent"], law["d_exponent"]

    def pred(n, d):
        return A * np.asarray(n, float) ** b * np.asarray(d, float) ** c

    tuned = json.load(open(a.tuned))
    budgets = sorted(tuned["budgets"])

    def rung(d_model, se, D):
        f = train_flops_per_example(arch_for(int(d_model), int(se))) * D
        return int(np.argmin([abs(np.log(f / x)) for x in budgets]))

    # Without the stage-2 results on disk (a results zip, say), fall back to
    # the per-configuration optima the law file records, and drop panel C,
    # which needs every trial.
    cells = (load_cells(Path(a.results_dir), a.metric, a.beta2)
             if Path(a.results_dir).is_dir() else [])
    if cells:
        rows = tune_per_config(cells)
    else:
        print(f"[WARN] no trials under {a.results_dir}; optima from {a.law}, "
              f"no landscape panel")
        rows = [dict(r) for r in json.load(open(a.law))["configs"]]
    for r in rows:
        r["rung"] = rung(r["d_model"], r["slot_encoder_layers"], r["train_examples"])
        r["pred"] = float(pred(r["n_params"], r["train_examples"]))
    stage3 = [r for r in tuned["runs"] if r.get("stage") == "3" and r["seed"] == 42]
    tuned_rungs = {r["rung"] for r in rows}

    n_pan = 3 if cells else 2
    fig, axes = plt.subplots(1, n_pan, figsize=(5.4 * n_pan, 4.9), facecolor=SURF)
    for ax in axes:
        style(ax)

    # ---- A: optima and the law along each rung ----------------------------
    ax = axes[0]
    for k in sorted(tuned_rungs):
        col = RAMP[k % len(RAMP)]
        rr = sorted((r for r in rows if r["rung"] == k), key=lambda r: r["n_params"])
        for r in rr:
            ax.plot(r["n_params"], r["lr_star"], "o", markersize=6.5, zorder=3,
                    color=col if r["usable"] else SURF, markeredgecolor=col,
                    markeredgewidth=1.3)
        ax.plot([r["n_params"] for r in rr], [r["pred"] for r in rr], "--",
                color=col, linewidth=1.4, zorder=2, label=f"C = {budgets[k]:.2g}")
    if stage3:
        s3 = sorted(stage3, key=lambda r: r["n_params"])
        k = rung_of_run = int(np.argmin([abs(np.log(s3[0]["flops"] / x)) for x in budgets]))
        col = RAMP[k % len(RAMP)]
        ax.plot([r["n_params"] for r in s3], [r["lr"] for r in s3], "D--",
                color=col, markersize=5.5, linewidth=1.2, markeredgecolor=INK,
                markeredgewidth=0.5, zorder=3,
                label=f"C = {budgets[rung_of_run]:.2g}, law-assigned (stage 3)")
    try:
        from src.scaling.configs import MEASURED_LR, fit_lr_law
        a0, b0 = fit_lr_law(MEASURED_LR)
        g = np.geomspace(min(r["n_params"] for r in rows) / 1.3,
                         max([r["n_params"] for r in rows] +
                             [r["n_params"] for r in stage3]) * 1.3, 40)
        ax.plot(g, np.exp(a0) * g ** b0, ls=(0, (1.5, 2.5)), color=REF,
                linewidth=1.8, zorder=1, label=f"old law, N only (b = {b0:+.2f})")
    except Exception as exc:  # the comparison line is optional
        print(f"[WARN] old law not drawn: {exc}")
    ax.plot([], [], "o", color=INK2, markersize=6, label="bracketed (fitted)")
    ax.plot([], [], "o", color=SURF, markeredgecolor=INK2, markersize=6,
            label="optimum on grid edge (excluded)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (parameters)", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"tuned learning rate (Akima argmin of {a.metric})",
                  color=INK2, fontsize=9.5)
    ax.set_title(rf"A.  lr = {A:.3g} N$^{{{b:+.2f}}}$ D$^{{{c:+.2f}}}$",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    # ---- B: measured / predicted ------------------------------------------
    ax = axes[1]
    ax.axhspan(1 / GRID_STEP, GRID_STEP, color=REF, alpha=0.13, zorder=0,
               label=f"one grid step ({GRID_STEP}x)")
    ax.axhline(1, color=REF, linewidth=1.2, zorder=1)
    for k in sorted(tuned_rungs):
        col = RAMP[k % len(RAMP)]
        rr = [r for r in rows if r["rung"] == k]
        ax.plot([r["n_params"] for r in rr], [r["lr_star"] / r["pred"] for r in rr],
                "o", color=col, markersize=6.5, zorder=3, linestyle="none",
                markerfacecolor=None)
        for r in rr:
            if not r["usable"]:
                ax.plot(r["n_params"], r["lr_star"] / r["pred"], "o", color=SURF,
                        markeredgecolor=col, markeredgewidth=1.3, markersize=6.5,
                        zorder=4)
    ok = [r for r in rows if r["usable"]]
    ratios = np.array([r["lr_star"] / r["pred"] for r in ok])
    inside = int(np.sum(np.abs(np.log(ratios)) <= np.log(GRID_STEP)))
    ax.set_xscale("log"); ax.set_yscale("log")
    lim = max(3.0, float(np.exp(np.abs(np.log([r["lr_star"] / r["pred"]
                                                for r in rows])).max())) * 1.2)
    ax.set_ylim(1 / lim, lim)
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([t for t in (0.25, 0.5, 1, 2, 4) if 1 / lim <= t <= lim])
    ax.set_xlabel("N (parameters)", color=INK2, fontsize=9.5)
    ax.set_ylabel("tuned / law-predicted learning rate", color=INK2, fontsize=9.5)
    note = (f"{inside} of {len(ok)} bracketed configs within one grid step; "
            f"r$^2$ = {law.get('r2', float('nan')):.2f}")
    loro = json.load(open(a.law)).get("leave_one_rung_out", {}).get("rungs", [])
    top = [r for r in loro if str(r.get("held_out", "")).startswith("top")]
    if top and "worst_ratio" in top[0]:
        note += (f"\ntop tuned rung held out of the fit: median "
                 f"{top[0]['median_ratio']:.2f}x, worst {top[0]['worst_ratio']:.2f}x")
    ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=8.5, color=INK2,
            va="top")
    ax.set_title("B.  How far each optimum sits from the law", color=INK,
                 fontsize=11.5, loc="left", pad=10)
    legend(ax, loc="upper right")

    # ---- C: the landscape in units of the law ------------------------------
    if cells:
        ax = axes[2]
        by = collections.defaultdict(list)
        for t in cells:
            by[(t["n_params"], t["train_examples"])].append(t)
        for (n, D), ts in by.items():
            ts = sorted(ts, key=lambda t: t["lr"])
            if len(ts) < 3:
                continue
            r0 = next((r for r in rows if r["n_params"] == n and r["train_examples"] == D), None)
            if r0 is None:
                continue
            col = RAMP[r0["rung"] % len(RAMP)]
            v = np.array([t["value"] for t in ts])
            x = np.array([t["lr"] for t in ts]) / float(pred(n, D))
            ax.plot(x, v - v.min(), "-", color=col, alpha=0.75, linewidth=1.2,
                    marker="o", markersize=2.5, zorder=2)
        ax.axvline(1, color=FIT_C, linewidth=1.6, linestyle="--", zorder=3,
                   label="law's prediction")
        ax.axvspan(1 / GRID_STEP, GRID_STEP, color=REF, alpha=0.13, zorder=0)
        ax.set_xscale("log")
        ymax = 6.0 if a.metric == "delta_e" else 0.15
        ax.set_ylim(-0.05 * ymax, ymax)
        ax.set_xlabel("learning rate / law-predicted rate", color=INK2, fontsize=9.5)
        ax.set_ylabel(("DeltaE" if a.metric == "delta_e" else "val CE")
                      + " above the cell's best (diverged rates dropped)",
                      color=INK2, fontsize=9.5)
        ax.set_title("C.  Every cell's LR curve, centred on the law", color=INK,
                     fontsize=11.5, loc="left", pad=10)
        legend(ax, loc="upper left")

    fig.suptitle("Tuned learning-rate law from the stage-2 IsoFLOP cells",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=1.03)
    fig.text(0.008, 0.965, f"{len(rows)} configurations, {len(cells) or 'no'} trials on disk, "
             f"beta2 = {a.beta2:g}; {len(ok)} bracketed and fitted. Stage 3 "
             "trains the rungs above at the law's rate, one rate per point.",
             fontsize=9, color=INK2, ha="left")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")
    print(f"  lr = {A:.4g} N^{b:+.3f} D^{c:+.3f};  {inside}/{len(ok)} within "
          f"{GRID_STEP}x")


if __name__ == "__main__":
    main()
