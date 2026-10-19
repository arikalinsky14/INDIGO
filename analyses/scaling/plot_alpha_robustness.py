#!/usr/bin/env python3
"""alpha (N* proportional to C^alpha) per metric, across several collections.

One row per metric, one marker per collection that carries it, each with its
95% bootstrap interval and its usable-curve count. For the question a single
fit cannot answer: does the exponent survive a change of seed, of how each
point's learning rate was picked, and of how DeltaE is measured?

Each --fit is LABEL=porian_fit.json (written by fit_scaling_porian.py).
Metrics a file does not carry are skipped. --rename maps a file's metric key
to a row, e.g. pipeline:pooled=pipe for the light-pipeline collection.

    python analyses/scaling/plot_alpha_robustness.py \\
        --fit "stage 2, rate by ΔE=a.json" "stage 2, rate by CE=b.json" \\
        --rename "light pipeline:pooled=pipe" --output alpha_robustness.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [("ce", "cross-entropy"), ("low", "ΔE, low chroma"),
        ("pooled", "ΔE, all colours"), ("pipe", "ΔE after light pipeline")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
MARKERS = ["o", "s", "D", "^"]
SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", nargs="+", required=True, help="LABEL=path ...")
    p.add_argument("--rename", nargs="*", default=[],
                   help="LABEL:metric=row ... (a file's metric shown on another row)")
    p.add_argument("--title", default=None)
    p.add_argument("--output", default="analyses/scaling/results/tuned/alpha_robustness.png")
    a = p.parse_args()

    ren = {}
    for item in a.rename:
        lab, rest = item.rsplit(":", 1)
        src, dst = rest.split("=")
        ren.setdefault(lab, {})[src] = dst
    cols = []
    for item in a.fit:
        lab, path = item.rsplit("=", 1)
        bm = json.load(open(path))["by_metric"]
        pts = {}
        for m, r in bm.items():
            row = ren.get(lab, {}).get(m, m if lab not in ren else None)
            n = r.get("laws", {}).get("N_star")
            if row and n:
                pts[row] = (n["exponent"], n["ci_low"], n["ci_high"],
                            r["n_usable"], len(r["rungs"]))
        cols.append((lab, pts))

    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=9, left=False)
    ax.grid(True, axis="x", color="#ebeae5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0.5, color=REF, linestyle=(0, (5, 3)), linewidth=1.4)
    ax.text(0.5, len(ROWS) - 0.35, "Chinchilla 0.50", color=INK2, fontsize=8.5,
            ha="center", va="bottom")
    k = len(cols)
    for i, (row, _) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        present = [(j, c) for j, c in enumerate(cols) if row in c[1]]
        offs = [0.0] if len(present) == 1 else \
            [0.28 - 0.56 * q / (len(present) - 1) for q in range(len(present))]
        for (j, (lab, pts)), off in zip(present, offs):
            e, lo, hi, u, n = pts[row]
            y = y0 + off
            ax.plot([lo, hi], [y, y], "-", color=COLORS[j % 4], linewidth=2,
                    solid_capstyle="round", alpha=0.7)
            ax.plot(e, y, MARKERS[j % 4], color=COLORS[j % 4], markersize=8,
                    markeredgecolor=SURF, markeredgewidth=1.6, zorder=3)
            ax.text(hi + 0.012, y, f"{e:.2f}  ({u}/{n} curves)", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, lab in ROWS][::-1], fontsize=9.5, color=INK)
    ax.set_ylim(-0.6, len(ROWS) - 0.2)
    ax.set_xlabel("α in N* ∝ C^α, with its 95% bootstrap interval", color=INK2, fontsize=9.5)
    handles = [plt.Line2D([], [], marker=MARKERS[j % 4], color=COLORS[j % 4],
                          linewidth=2, markersize=7, label=lab)
               for j, (lab, _) in enumerate(cols)]
    leg = ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=2,
                    loc="upper left", bbox_to_anchor=(0.0, -0.16))
    keep = {r for r, _ in ROWS}
    allv = [v for _, pts in cols for r, v in pts.items() if r in keep]
    ax.set_xlim(min(v[1] for v in allv) - 0.03, max(v[2] for v in allv) + 0.17)
    for t in leg.get_texts():
        t.set_color(INK2)
    es = [v[0] for v in allv]
    title = a.title or (f"All {len(es)} estimates of α fall between "
                        f"{min(es):.2f} and {max(es):.2f}")
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left", y=1.0)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
