#!/usr/bin/env python3
"""Clean loss-against-learning-rate curves from STAGE=lrcurve.

One column per point, three rows: validation cross-entropy, greedy DeltaE00,
and DeltaE00 after the light inference pipeline (slurms/eval_pipeline.sh on
each rate's saved weights). Every rate is a fresh run on the same seed, data
order and eval slice, so each column is one experiment. Rates at or above the
point's first divergence were not rerun; they are drawn as crosses along the
top, from the stage-2 trials. The star on each row is that row's own best
rate.

Stage-2 trials of the same point (same configuration, seed, data and eval
slice) are kept at rates the new grid did not repeat, drawn hollow; one
within --same-rate (default 1.2x) of a new rate is replaced by it. The
pipeline row is new runs only: the earlier pipeline scores are a different
seed's retrain. --no-stage2 draws the new runs alone.

    python analyses/scaling/plot_lr_curves_clean.py \\
        --curves-dir outputs/lr_curves/cross_attn \\
        --stage2-dir outputs/lr_search/cross_attn
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
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
from lr_edges import cell_trials, diverged                     # noqa: E402
from collect_confirm import pipeline_scores                    # noqa: E402
from src.scaling.flops import n_params                         # noqa: E402
from lr_edges import arch_for                                   # noqa: E402
from src.scaling.porian import akima_argmin                    # noqa: E402

COL = "#2a78d6"
BAD = "#d03b3b"
SURF, INK, INK2, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
ROWS = [("ce", "val cross-entropy"), ("de", r"val $\Delta E_{00}$ (greedy)"),
        ("pipe", r"$\Delta E_{00}$ after light pipeline")]
NAME = re.compile(r"lr_search_ep(\d+)_lim(\d+)_d(\d+)_se(\d+)_bs\d+_b2([\d.]+)_r(\d+)\.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curves-dir", default="outputs/lr_curves/cross_attn")
    p.add_argument("--stage2-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--pipeline-dir", default=None, help="default: <curves-dir>/pipeline")
    p.add_argument("--match", default=None,
                   help="only results files whose name contains this, e.g. "
                        "lim39959808 for the production run")
    p.add_argument("--pipeline-label", default=r"$\Delta E_{00}$ after light pipeline",
                   help="y label of the pipeline row, e.g. to say it is scored "
                        "on dev rows")
    p.add_argument("--title", default="Loss against learning rate at the largest compute budget")
    p.add_argument("--same-rate", type=float, default=1.2,
                   help="a stage-2 rate within this factor of a new rate counts "
                        "as repeated and is replaced")
    p.add_argument("--no-stage2", action="store_true",
                   help="draw the new runs only")
    p.add_argument("--output", default="analyses/scaling/results/tuned/lr_curves_clean.png")
    a = p.parse_args()
    cdir = Path(a.curves_dir)
    pdir = Path(a.pipeline_dir) if a.pipeline_dir else cdir / "pipeline"

    cells = collections.defaultdict(list)
    for path in sorted(glob.glob(str(cdir / "lr_search_*_r*.json"))):
        if a.match and a.match not in Path(path).name:
            continue
        m = NAME.match(Path(path).name)
        if not m:
            continue
        ep, lim, d, se, b2, _ = m.groups()
        res = json.load(open(path))
        for t in res.get("results", []):
            if diverged(t):
                continue
            tag = Path(path).name[len("lr_search_"):-len(".json")]
            pipe = pipeline_scores(pdir / tag / f"lr{t['lr']:.6e}").get("pooled")
            cells[(int(d), int(se), int(ep), int(lim), b2)].append(
                {"lr": t["lr"], "ce": t.get("final_val_loss"),
                 "de": t.get("final_val_de"), "pipe": pipe})
    if not cells:
        sys.exit(f"no lr_search_*_r*.json under {cdir}")

    cols = []
    for (d, se, ep, lim, b2), pts in sorted(cells.items(), key=lambda kv: kv[0][0] * 100 + kv[0][1]):
        main_file = Path(a.stage2_dir) / f"lr_search_ep{ep}_lim{lim}_d{d}_se{se}_bs256_b2{b2}.json"
        _, s2 = cell_trials(main_file) if main_file.is_file() else ({}, [])
        div = sorted(t["lr"] for t in s2 if diverged(t))
        new = [q["lr"] for q in pts]
        old = [] if a.no_stage2 else [
            {"lr": t["lr"], "ce": t.get("final_val_loss"), "de": t.get("final_val_de"),
             "pipe": None, "old": True}
            for t in s2 if not diverged(t)
            and all(max(t["lr"] / x, x / t["lr"]) > a.same_rate for x in new)]
        cols.append({"d": d, "se": se, "D": ep * lim, "N": int(n_params(arch_for(d, se))),
                     "pts": sorted(pts + old, key=lambda q: q["lr"]), "div": div})
    cols.sort(key=lambda c: c["N"])

    rows_def = ROWS[:2] + [("pipe", a.pipeline_label)]
    fig, axes = plt.subplots(3, len(cols), figsize=(3.4 * len(cols), 9.4),
                             facecolor=SURF, squeeze=False, sharex="col")
    for j, c in enumerate(cols):
        for i, (field, label) in enumerate(rows_def):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(FAINT)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.grid(True, color="#ebeae5", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xscale("log")
            rows = [q for q in c["pts"]
                    if q[field] is not None and np.isfinite(q[field])]
            pts = [(q["lr"], q[field]) for q in rows]
            if pts:
                xs, ys = zip(*pts)
                if len(xs) >= 3:
                    g, yg, _ = akima_argmin(list(xs), list(ys))
                    ax.plot(g, yg, "-", color=COL, linewidth=1.6, alpha=0.75, zorder=2)
                for q in rows:
                    if q.get("old"):
                        ax.plot([q["lr"]], [q[field]], "o", markerfacecolor=SURF,
                                markeredgecolor=COL, markersize=6,
                                markeredgewidth=1.6, zorder=3)
                    else:
                        ax.plot([q["lr"]], [q[field]], "o", color=COL, markersize=6,
                                markeredgecolor=SURF, markeredgewidth=0.9, zorder=3)
                k = int(np.argmin(ys))
                ax.plot([xs[k]], [ys[k]], "*", color=COL, markersize=16,
                        markeredgecolor=INK, markeredgewidth=0.7, zorder=4)
                lo, hi = min(ys), max(ys)
                pad = 0.08 * (hi - lo or abs(hi) * 0.01 or 1)
                ax.set_ylim(lo - pad, hi + 2.4 * pad)
                for x in c["div"]:
                    ax.plot([x], [hi + 1.7 * pad], "x", color=BAD, markersize=7,
                            markeredgewidth=1.8, zorder=4, clip_on=False)
            else:
                ax.text(0.5, 0.5, "not scored yet", transform=ax.transAxes,
                        ha="center", color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel(label, color=INK2, fontsize=9)
            if i == len(ROWS) - 1:
                ax.set_xlabel("learning rate", color=INK2, fontsize=9)
        axes[0][j].set_title(f"N = {c['N'] / 1e6:.2g}M, D = {c['D'] / 1e6:.3g}M\n"
                             f"d{c['d']}/se{c['se']}", color=INK, fontsize=9.5,
                             loc="left", pad=6)
    fig.suptitle(a.title,
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.035)
    n_old = sum(bool(q.get("old")) for c in cols for q in c["pts"])
    fig.text(0.008, 0.997, "Filled: each rate a fresh run, same seed, data and eval slice."
             + (" Hollow: the earlier run of the same configuration at a rate not "
                "repeated." if n_old else "")
             + " Star: the best rate on that row. Red crosses: rates that diverged.",
             fontsize=8.5, color=INK2, ha="left")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The plotted values, one row per (point, rate), beside the figure.
    with open(out.with_suffix(".csv"), "w") as f:
        f.write("d_model,se,n_params,examples,lr,source,val_ce,greedy_de,pipeline_de,diverged\n")
        for c in cols:
            for q in c["pts"]:
                v = lambda x: "" if x is None else f"{x:.6g}"
                f.write(f"{c['d']},{c['se']},{c['N']},{c['D']},{q['lr']:.6e},"
                        f"{'earlier' if q.get('old') else 'new'},{v(q['ce'])},"
                        f"{v(q['de'])},{v(q['pipe'])},0\n")
            for x in c["div"]:
                f.write(f"{c['d']},{c['se']},{c['N']},{c['D']},{x:.6e},earlier,,,,1\n")
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}: " + ", ".join(
        f"d{c['d']}/se{c['se']} ({sum(not q.get('old') for q in c['pts'])} new, "
        f"{sum(bool(q.get('old')) for q in c['pts'])} earlier, "
        f"{sum(q['pipe'] is not None for q in c['pts'])} scored)" for c in cols))


if __name__ == "__main__":
    main()
