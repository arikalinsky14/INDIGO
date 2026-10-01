"""The learning-rate sweep itself: every probed LR, per configuration.

`scripts/lr_tuning.py` stores one record per probed learning rate, not just the
winner, so the curve that produced each optimum can be inspected after the
fact. This draws it: one small multiple per configuration, the probed points,
the Akima interpolation the optimum is read off, and a marker showing whether
that optimum bracketed.

A cell that did not bracket is the thing to look for. Its optimum sits at an
endpoint, which means the grid was positioned wrong rather than that the
optimum is there, and `scripts/fit_lr_law.py` discards it.

    python analyses/scaling/plot_lr_sweep.py \\
        --results-dir outputs/lr_search/cross_attn
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scaling.porian import akima_argmin, tuned_optimum   # noqa: E402

# Validated ordinal ramp: one hue, monotone lightness, gaps >= 0.06.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#124782", "#0d366b"]
GOOD, BAD = "#0ca30c", "#d03b3b"      # status palette, never reused as series
SURF, INK, INK2, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
FIELD = {"delta_e": ("final_val_de", r"val $\Delta E_{00}$"),
         "val_loss": ("best_val_loss", "val loss")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--metric", choices=sorted(FIELD), default="delta_e")
    p.add_argument("--output", default="analyses/scaling/results/lr_sweep_curves.png")
    a = p.parse_args()

    field, ylabel = FIELD[a.metric]
    cells = []
    for path in sorted(glob.glob(str(Path(a.results_dir) / "lr_search_*.json"))):
        d = json.load(open(path))
        pts = [(t["lr"], t.get(field)) for t in d.get("results", [])]
        pts = [(x, y) for x, y in pts if y is not None and np.isfinite(y)]
        if len(pts) < 2:
            continue
        cells.append({
            "n_params": d.get("n_params"), "d_model": d.get("d_model"),
            "se": d.get("slot_encoder_layers"), "bs": d.get("batch_size"),
            "beta2": d.get("beta2"), "D": d.get("train_examples"),
            "lrs": [x for x, _ in pts], "vals": [y for _, y in pts],
        })
    if not cells:
        sys.exit(f"no lr_search JSONs with usable trials under {a.results_dir}")

    # One panel per (model size, dataset size); beta2 overlays within a panel.
    panels = defaultdict(list)
    for c in cells:
        panels[(c["n_params"] or 0, c["D"] or 0)].append(c)
    keys = sorted(panels)
    ncol = min(3, len(keys))
    nrow = math.ceil(len(keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow),
                             facecolor=SURF, squeeze=False)

    n_edge = 0
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(FAINT)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(True, color="#ebeae5", linewidth=0.8)
        ax.set_axisbelow(True)

        group = sorted(panels[key], key=lambda c: c["beta2"] or 0)
        for j, c in enumerate(group):
            col = RAMP[(j * 2) % len(RAMP)]
            lrs, vals = c["lrs"], c["vals"]
            ax.plot(lrs, vals, "o", color=col, markersize=6,
                    markeredgecolor=SURF, markeredgewidth=0.9, zorder=3,
                    label=rf"$\beta_2$ = {c['beta2']:g}" if c["beta2"] else None)
            if len(set(lrs)) >= 3:
                grid, y_grid, idx = akima_argmin(lrs, vals)
                ax.plot(grid, y_grid, "-", color=col, linewidth=1.5,
                        alpha=0.75, zorder=2)
                best, on_edge = tuned_optimum(lrs, vals)
                n_edge += int(on_edge)
                ax.axvline(best, ls=":", color=col, alpha=0.6, zorder=1)
                ax.plot([best], [y_grid[idx]], marker="*", markersize=14,
                        color=col, markeredgecolor=BAD if on_edge else GOOD,
                        markeredgewidth=1.6, zorder=4)
        c0 = group[0]
        ax.set_xscale("log")
        ax.set_xlabel("learning rate", color=INK2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
        d_str = f"{c0['D']:,}" if c0["D"] else "?"
        ax.set_title(f"d{c0['d_model']}/se{c0['se']}   "
                     f"N = {c0['n_params']:,}\nD = {d_str}, bs = {c0['bs']:g}",
                     color=INK, fontsize=10, pad=8, loc="left")
        if len(group) > 1:
            leg = ax.legend(fontsize=8, frameon=False, loc="upper center")
            for t in leg.get_texts():
                t.set_color(INK2)

    for k in range(len(keys), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # Header sits above the axes. One line height in figure fractions, so the
    # two lines stay apart whatever the panel count.
    line = 0.30 / (3.5 * nrow)
    fig.text(0.006, 1.0 + 2.0 * line,
             "Learning-rate sweep: every probed value, per configuration",
             fontsize=13.5, color=INK, ha="left", va="bottom")
    fig.text(0.006, 1.0 + 0.6 * line,
             f"Points are probed learning rates; the line is the Akima "
             f"interpolation the optimum is read off. A star ringed in green "
             f"bracketed; red means the optimum sat at a grid endpoint and the "
             f"fit discards that cell.  {n_edge} of {len(cells)} did not bracket.",
             fontsize=9, color=INK2, ha="left", va="bottom")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(a.output, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {a.output}")
    print(f"  {len(cells)} cells, {n_edge} did not bracket")


if __name__ == "__main__":
    main()
