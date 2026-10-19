#!/usr/bin/env python3
"""Loss against learning rate for selected IsoFLOP points: CE and DeltaE.

One column per selected point, two rows: validation cross-entropy (top) and
greedy DeltaE00 (bottom), every probed rate of the cell, extension rates
(STAGE=2x) merged in. Rates that diverged are drawn as crosses along the top
of the panel rather than at their value. The star is the rate the point was
selected at (by DeltaE, as every stage-2 cell was); the dotted line, when a
law file is given, is what the joint lr(N, D) law predicts for that point.

By default the selection is each IsoFLOP curve's lowest-CE point, i.e. the
point nearest that curve's compute-optimal model. --select picks by hand:

    python analyses/scaling/plot_lr_curves.py \\
        --results-dir outputs/lr_search/cross_attn \\
        --tuned analyses/scaling/results/tuned/isoflop_tuned.json \\
        --law analyses/scaling/results/lr_law_fit.json
    python analyses/scaling/plot_lr_curves.py ... \\
        --select "d64/se1@1e14; d192/se4@2.75e15; d352/se6@2.5e16"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lr_edges import arch_for, cell_trials, diverged            # noqa: E402
from merge_lr_parts import select                               # noqa: E402
from src.scaling.flops import n_params, train_flops_per_example  # noqa: E402
from src.scaling.porian import akima_argmin                     # noqa: E402

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#124782", "#0d366b"]
FIT_C, BAD = "#eb6834", "#d03b3b"
SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"
ROWS = [("final_val_loss", "val cross-entropy"), ("final_val_de", r"val $\Delta E_{00}$ (greedy)")]
PIPE_ROW = ("pipeline", r"$\Delta E_{00}$ after light pipeline")
RERUN = "#eb6834"


def find_cell(rdir: Path, d: int, se: int, budget: float, beta2: float):
    """The seed-42 results file of d<d>/se<se> whose FLOPs land on `budget`."""
    arch = arch_for(d, se)
    for path in sorted(rdir.glob(f"lr_search_ep*_lim*_d{d}_se{se}_bs256_b2{beta2:g}.json")):
        m = re.match(r"lr_search_ep(\d+)_lim(\d+)_", path.name)
        flops = train_flops_per_example(arch) * int(m.group(1)) * int(m.group(2))
        if abs(math.log(flops / budget)) < math.log(1.05):
            return path, int(m.group(1)) * int(m.group(2))
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--tuned", default="analyses/scaling/results/tuned/isoflop_tuned.json",
                   help="for the budgets and the default selection")
    p.add_argument("--law", default=None, help="lr_law_fit.json, to mark the law's rate")
    p.add_argument("--select", default=None,
                   help='"d64/se1@1e14; d192/se4@2.75e15" (default: each curve\'s lowest-CE point)')
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--confirm-dir", default=None,
                   help="collect_confirm.py's output dir: adds each point's "
                        "unselected retrain (seed 45) at its chosen rate, and "
                        "a third row with its light-pipeline DeltaE")
    p.add_argument("--output", default="analyses/scaling/results/tuned/lr_curves.png")
    a = p.parse_args()

    tuned = json.load(open(a.tuned))
    budgets = sorted(tuned["budgets"])
    picks = []
    if a.select:
        for item in filter(None, (x.strip() for x in a.select.split(";"))):
            m = re.fullmatch(r"d(\d+)/se(\d+)@([0-9.eE+-]+)", item)
            if not m:
                sys.exit(f"cannot read {item!r}; expected like d64/se1@1e14")
            b = min(budgets, key=lambda x: abs(math.log(x / float(m.group(3)))))
            picks.append((int(m.group(1)), int(m.group(2)), b))
    else:
        for b in budgets:
            pts = [r for r in tuned["runs"] if r["seed"] == 42 and r.get("val_loss")
                   and abs(math.log(r["flops"] / b)) < math.log(1.05)]
            if pts:
                best = min(pts, key=lambda r: r["val_loss"])
                m = re.search(r"_d(\d+)_se(\d+)_", best["name"])
                picks.append((int(m.group(1)), int(m.group(2)), b))
    law = (json.load(open(a.law)).get("lr_vs_n_and_d") if a.law else None)
    if law and law.get("status") != "ok":
        law = None

    conf, pipe = {}, {}
    if a.confirm_dir:
        cdir = Path(a.confirm_dir)
        for fn, dst in (("isoflop_confirm.json", conf), ("isoflop_pipeline.json", pipe)):
            if (cdir / fn).is_file():
                for r in json.load(open(cdir / fn))["runs"]:
                    m = re.search(r"_d(\d+)_se(\d+)_", r["name"])
                    dst[(int(m.group(1)), int(m.group(2)), r["passes"])] = r
    rows = ROWS + ([PIPE_ROW] if pipe else [])

    cols = []
    for d, se, b in picks:
        path, D = find_cell(Path(a.results_dir), d, se, b, a.beta2)
        if path is None:
            print(f"[WARN] d{d}/se{se} at C={b:.3g}: no results file, skipped")
            continue
        res, trials = cell_trials(path)
        if not trials:
            continue
        ok = [t for t in trials if not diverged(t)]
        chosen = select(trials, res.get("selection_metric", "delta_e"))["lr"]
        cols.append({"d": d, "se": se, "budget": b, "D": D,
                     "N": int(n_params(arch_for(d, se))), "ok": ok,
                     "div": [t["lr"] for t in trials if diverged(t)],
                     "chosen": chosen, "rung": budgets.index(b),
                     "conf": conf.get((d, se, D)), "pipe": pipe.get((d, se, D))})
    if not cols:
        sys.exit("nothing to plot")

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.3 * len(cols), 3.2 * len(rows)),
                             facecolor=SURF, squeeze=False, sharex="col")
    for j, c in enumerate(cols):
        col = RAMP[c["rung"] % len(RAMP)]
        for i, (field, label) in enumerate(rows):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(FAINT)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.grid(True, color="#ebeae5", linewidth=0.8)
            ax.set_axisbelow(True)
            rerun = None
            if c["conf"]:
                rerun = {"final_val_loss": c["conf"]["val_loss"],
                         "final_val_de": c["conf"]["val_de"]["pooled"]}.get(field)
            if field == "pipeline":
                ax.set_xscale("log")
                if c["pipe"]:
                    v = c["pipe"]["val_de"]["pooled"]
                    ax.plot([c["conf"]["lr"]], [v], "D", color=RERUN, markersize=8,
                            markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
                    ax.annotate(f"{v:.2f}", (c["conf"]["lr"], v), xytext=(8, 0),
                                textcoords="offset points", va="center",
                                fontsize=8.5, color=INK2)
                    ax.text(0.03, 0.06, "scored at the chosen rate only",
                            transform=ax.transAxes, fontsize=7.5, color=INK2)
                if j == 0:
                    ax.set_ylabel(label, color=INK2, fontsize=9)
                if i == len(rows) - 1:
                    ax.set_xlabel("learning rate", color=INK2, fontsize=9)
                continue
            pts = sorted((t["lr"], t[field]) for t in c["ok"]
                         if t.get(field) is not None and np.isfinite(t[field]))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o", color=col, markersize=6, markeredgecolor=SURF,
                    markeredgewidth=0.9, zorder=3)
            if len(xs) >= 3:
                grid, yg, _ = akima_argmin(list(xs), list(ys))
                ax.plot(grid, yg, "-", color=col, linewidth=1.5, alpha=0.75, zorder=2)
            yc = dict(pts).get(c["chosen"])
            if yc is not None:
                ax.plot([c["chosen"]], [yc], "*", color=col, markersize=15,
                        markeredgecolor=INK, markeredgewidth=0.7, zorder=4)
            if rerun is not None and np.isfinite(rerun):
                ax.plot([c["conf"]["lr"]], [rerun], "D", color=RERUN, markersize=7,
                        markeredgecolor=INK, markeredgewidth=0.6, zorder=5)
            lo, hi = min(ys + ((rerun,) if rerun is not None else ())), \
                max(ys + ((rerun,) if rerun is not None else ()))
            pad = 0.08 * (hi - lo or abs(hi) * 0.01 or 1)
            ax.set_ylim(lo - pad, hi + 2.2 * pad)
            for x in c["div"]:
                ax.plot([x], [hi + 1.6 * pad], "x", color=BAD, markersize=7,
                        markeredgewidth=1.8, zorder=4, clip_on=False)
            if law:
                ax.axvline(law["coef"] * c["N"] ** law["n_exponent"]
                           * c["D"] ** law["d_exponent"], ls=":", color=FIT_C,
                           linewidth=1.5, zorder=1)
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(label, color=INK2, fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel("learning rate", color=INK2, fontsize=9)
        axes[0][j].set_title(f"C = {c['budget']:.2g}\nd{c['d']}/se{c['se']}, "
                             f"N = {c['N'] / 1e6:.2g}M, D = {c['D'] / 1e6:.3g}M",
                             color=INK, fontsize=9.5, loc="left", pad=6)

    fig.suptitle("Loss against learning rate at selected IsoFLOP points",
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.04)
    fig.text(0.008, 0.995,
             "Every probed rate (extensions merged). Star: the rate the point was "
             "selected at, by DeltaE. Red crosses along the top: rates that "
             "diverged." + (" Dotted: the joint law's rate." if law else "")
             + (" Orange diamond: the seed-45 retrain at the chosen rate." if conf else ""),
             fontsize=8.5, color=INK2, ha="left")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}: " + ", ".join(f"d{c['d']}/se{c['se']}@{c['budget']:.2g}"
                                       for c in cols))


if __name__ == "__main__":
    main()
