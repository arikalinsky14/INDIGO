#!/usr/bin/env python3
"""Two inference searches on the same model, side by side.

1. Gamut: for every target of the five batteries, the target colour and the
   colour each search achieved, with its DeltaE00 underneath. Two or more
   gamut_eval.py JSONs, same targets and seed, smallest search first.
2. Test rows: per tier, the share of rows under each DeltaE00 (a cumulative
   curve per search), and each row's DeltaE00 under one search against the
   other. Two test_eval.py output folders (each holding tier_a/ and tier_b/
   rows.json), same rows and seed. Uses the first two --labels.

    python analyses/scaling/plot_search_compare.py \\
        --gamut outputs/prod_eval/gamut/gamut_light_dog.json \\
                outputs/prod_eval/gamut/gamut_mid_dog.json \\
                outputs/prod_eval/gamut/gamut_high_dog.json \\
        --test outputs/prod_eval/test_eval_light outputs/prod_eval/test_eval_mid \\
        --labels "light (16, 5 steps)" "mid (50, 20 steps)" "high (250, 40 steps)" \\
        --out-dir outputs/prod_eval/compare
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference" / "scripts"))
from gamut_swatch import _lab_to_srgb01, _is_in_srgb_gamut  # noqa: E402

SURF, INK, INK2, FAINT, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"
GREY, ACCENT = "#9a9890", "#2a78d6"     # first search quiet, last the accent


def gamut_compare(paths, labels, out: Path) -> None:
    runs = [json.load(open(p)) for p in paths]
    names = list(runs[0]["batteries"])
    SW, GAP, PAIR, ROW, LEFT, M = 0.62, 0.05, 0.30, 1.55, 2.3, 0.4
    n_max = max(len(runs[0]["batteries"][b]["per_target"]) for b in names)
    K = len(runs)
    group = (K + 1) * SW + K * GAP + PAIR
    W = 2 * M + LEFT + n_max * group
    H = 2 * M + 1.1 + len(names) * ROW
    fig = plt.figure(figsize=(W, H), facecolor=SURF)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    meds = [r["overall"]["stats"]["median_de"] for r in runs]
    cut = (1 - meds[-1] / meds[0]) * 100
    chain = " → ".join(f"{m:.2f}" for m in meds)
    ax.text(M, H - M, f"Gamut targets: the {labels[K - 1].split(' ')[0]} search cuts the "
            f"median ΔE₀₀ by {cut:.0f}% ({chain})" if cut > 0
            else f"Gamut targets by search: median ΔE₀₀ {chain}",
            fontsize=15, color=INK, va="top")
    ax.text(M, H - M - 0.42, "Each group: target, then " + ", then ".join(labels[:K]) + ". "
            "ΔE₀₀ under each achieved colour. * marks a target outside sRGB "
            "(its swatch is clipped).", fontsize=9.5, color=INK2, va="top")
    for bi, b in enumerate(names):
        y = H - M - 1.1 - (bi + 1) * ROW + 0.45
        ts = [r["batteries"][b]["per_target"] for r in runs]
        st = [r["batteries"][b]["stats"] for r in runs]
        ax.text(M, y + SW + 0.25, b.replace("_", " "), fontsize=11, color=INK, va="bottom")
        ax.text(M, y + SW + 0.02, "median " + "  →  ".join(f"{s['median_de']:.2f}" for s in st),
                fontsize=8.5, color=INK2, va="bottom")
        for i, tgt in enumerate(ts[0]):
            x = M + LEFT + i * group
            lab = tuple(tgt["target_lab"])
            ax.add_patch(Rectangle((x, y), SW, SW, facecolor=_lab_to_srgb01(lab),
                                   edgecolor=EDGE, linewidth=0.7))
            for k in range(K):
                t = ts[k][i]
                xx = x + (k + 1) * (SW + GAP)
                col = (_lab_to_srgb01(tuple(t["achieved_lab"]))
                       if t.get("achieved_lab") else (0.9, 0.9, 0.92))
                ax.add_patch(Rectangle((xx, y), SW, SW, facecolor=col,
                                       edgecolor=EDGE, linewidth=0.7))
                de = t.get("delta_e")
                ax.text(xx + SW / 2, y - 0.08, "fail" if de is None else f"{de:.1f}",
                        fontsize=8, ha="center", va="top",
                        color=ACCENT if k == K - 1 else INK2)
            L, a, bb = lab
            ax.text(x + SW / 2, y - 0.08, f"{L:.0f},{a:+.0f},{bb:+.0f}"
                    + ("*" if not _is_in_srgb_gamut(lab) else ""),
                    fontsize=6.3, ha="center", va="top", color=INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}")


def _rows(folder: Path, tier: str) -> dict:
    hits = glob.glob(str(folder / "**" / f"tier_{tier}" / "rows.json"), recursive=True)
    if not hits:
        sys.exit(f"no tier_{tier}/rows.json under {folder}")
    return {r["row_idx"]: float(r["delta_e"]) for r in json.load(open(hits[0]))
            if r.get("delta_e") is not None}


def test_compare(folders, labels, out: Path) -> None:
    tiers = [("a", "tier A: training materials, new structures"),
             ("b", "tier B: held-out real materials")]
    data = {t: [_rows(Path(f), t) for f in folders] for t, _ in tiers}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), facecolor=SURF,
                             gridspec_kw={"height_ratios": [1, 1.05]})
    for j, (t, title) in enumerate(tiers):
        a, b = data[t]
        common = sorted(set(a) & set(b))
        xa = np.array([a[k] for k in common]); xb = np.array([b[k] for k in common])
        for ax in axes[:, j]:
            ax.set_facecolor(SURF)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(FAINT)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(True, color="#ebeae5", linewidth=0.7)
            ax.set_axisbelow(True)
        ax = axes[0][j]
        grid = np.linspace(0, 10, 401)
        for vals, lab, col, lw in ((xa, labels[0], GREY, 1.8), (xb, labels[1], ACCENT, 2.4)):
            ax.plot(grid, [(vals <= g).mean() * 100 for g in grid], color=col,
                    linewidth=lw, label=lab)
        for thr in (1, 2, 5):
            ax.axvline(thr, color=FAINT, linewidth=1, zorder=1)
            pa, pb = (xa < thr).mean() * 100, (xb < thr).mean() * 100
            ax.annotate(f"{pb:.0f}%", (thr, pb), xytext=(4, -2), textcoords="offset points",
                        fontsize=8.5, color=ACCENT, va="top")
            ax.annotate(f"{pa:.0f}%", (thr, pa), xytext=(4, -2), textcoords="offset points",
                        fontsize=8.5, color=INK2, va="top")
        ax.set_xlim(0, 10); ax.set_ylim(0, 101)
        ax.set_xlabel("ΔE₀₀", color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel("rows at or under this ΔE₀₀ (%)", color=INK2, fontsize=9)
        ax.set_title(f"{title}\nmedian {np.median(xa):.2f} → {np.median(xb):.2f}  "
                     f"({len(common)} rows)", color=INK, fontsize=10.5, loc="left")
        if j == 1:
            leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
            for tx in leg.get_texts():
                tx.set_color(INK2)
        ax = axes[1][j]
        lo = max(min(xa.min(), xb.min()), 0.01)
        ax.scatter(np.clip(xa, lo, None), np.clip(xb, lo, None), s=10, color=ACCENT,
                   alpha=0.45, linewidths=0)
        ax.plot([lo, 100], [lo, 100], color=EDGE, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, 100); ax.set_ylim(lo, 100)
        better = (xb < xa).mean() * 100
        ax.text(0.03, 0.97, f"{better:.0f}% of rows better with {labels[1].split(' ')[0]}",
                transform=ax.transAxes, fontsize=9, color=INK2, va="top")
        ax.set_xlabel(f"ΔE₀₀, {labels[0]}", color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"ΔE₀₀, {labels[1]}", color=INK2, fontsize=9)
    med = [np.median(np.concatenate([list(data[t][k].values()) for t, _ in tiers]))
           for k in range(2)]
    fig.suptitle(f"Test rows: the {labels[1].split(' ')[0]} search cuts the median "
                 f"ΔE₀₀ from {med[0]:.2f} to {med[1]:.2f}",
                 fontsize=14, color=INK, x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.962, "Same model, same 500 rows per tier, same seed. Top: share "
             "of rows at or under each ΔE₀₀. Bottom: each row under both searches; "
             "below the dashed line is better with the larger search.",
             fontsize=9, color=INK2, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamut", nargs="+", default=None,
                   help="two or more gamut_eval JSONs, smallest search first")
    p.add_argument("--test", nargs=2, default=None, help="two test_eval output folders")
    p.add_argument("--labels", nargs="+", default=["light (16, 5 steps)", "mid (50, 20 steps)",
                                                   "high (250, 40 steps)"])
    p.add_argument("--out-dir", default="outputs/prod_eval/compare")
    a = p.parse_args()
    out = Path(a.out_dir)
    if a.gamut:
        names = "_vs_".join(l.split(" ")[0] for l in a.labels[:len(a.gamut)])
        gamut_compare(a.gamut, a.labels, out / f"gamut_{names}.png")
    if a.test:
        test_compare(a.test, a.labels, out / "test_light_vs_mid.png")


if __name__ == "__main__":
    main()
