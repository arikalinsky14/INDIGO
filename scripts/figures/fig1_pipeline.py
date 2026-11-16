#!/usr/bin/env python3
"""Fig. 1: INDIGO inference pipeline (schematic, no data) + TOC graphic.

Stages follow inference/src/solve.py:
  inputs (material pool n,k + target L*a*b*)
  -> generate.py  model samples N candidate stacks; constraint decode-masks
                  are ANDed into the next-layer distribution (the gate)
  -> simulate.py  transfer-matrix reflectance -> L*a*b*
  -> select.py    rank candidates against the target, keep top k
  -> refine.py    projected-gradient (DoG) descent on continuous thicknesses,
                  clamped to per-layer bounds
  -> output stack + color swatch

All curves, stacks and colors are illustrative glyphs. Nothing is read from
results. Edit wording in LABELS.

    python scripts/figures/fig1_pipeline.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

import _style as S

LABELS = {
    "h_in": "Inputs",
    "s_in": "",
    "pool": "Material pool (n, k)",
    "target": "Target color",
    "target_sub": "L*a*b*",
    "h_gen": "Sample",
    "s_gen": "N candidate stacks",
    "model": "INDIGO model",
    "next": "next-layer options",
    "gate": "constraint\nmask",
    "loop": "each layer",
    "h_sim": "Simulate",
    "s_sim": "transfer-matrix optics",
    "refl": "R(λ)",
    "light": "light",
    "h_rank": "Rank",
    "s_rank": r"by $\Delta E_{00}$ to target",
    "de": r"$\Delta E_{00}$",
    "topk": "top k",
    "h_ref": "Refine",
    "s_ref": "projected gradient (DoG)",
    "thick": "thicknesses\nwithin bounds",
    "iter": "step",
    "h_out": "Output",
    "s_out": "",
    "stack": "stack",
    "swatch_t": "target",
    "swatch_r": "result",
    # TOC
    "toc_pool": "your materials",
    "toc_target": "target color",
    "toc_model": "INDIGO",
    "toc_out": "thin-film stack",
}

# Materials: color per pool entry (Okabe-Ito). Ag is the masked one.
MATS = [("SiO$_2$", S.OI["sky"]), ("TiO$_2$", S.OI["green"]),
        ("aSi", S.OI["orange"]), ("Ag", "#9A9A9A")]
MASKED = 3
SUBSTRATE = "#4D4D4D"
PANEL = "#F3F3F3"
TARGET_LAB = (58, 34, -40)
CANDIDATE_LABS = [(57, 30, -36), (52, 20, -45), (64, 10, -20), (45, 38, -12), (70, -5, 10)]


def nk_icon(i, lam):
    """Plausible-looking n, k shapes for the icon only (not data)."""
    x = (lam - 300) / 600
    if i == 0:   # SiO2
        return 1.46 + 0.05 * (1 - x), 0 * x
    if i == 1:   # TiO2
        return 2.3 + 0.8 * np.exp(-6 * x), 0.6 * np.exp(-25 * x)
    if i == 2:   # aSi
        return 4.4 - 0.9 * x + 0.3 * np.exp(-20 * (x - .25) ** 2), 2.2 * np.exp(-6 * x)
    return 0.15 + 0.1 * x, 1.5 + 4.2 * x      # Ag


def box(ax, x, y, w, h, fc=PANEL, ec="none", lw=0.6, r=0.12, z=0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def arrow(ax, x0, y0, x1, y1, lw=0.9, color=S.INK, z=5, ms=7, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0, **kw))


def stack(ax, x, y, w, layers, scale=0.011, z=3, sub_h=0.12):
    """layers: list of (material index, thickness nm), bottom (substrate side) first."""
    ax.add_patch(Rectangle((x, y), w, sub_h, fc=SUBSTRATE, ec="none", zorder=z))
    yy = y + sub_h
    for m, t in layers:
        h = t * scale
        ax.add_patch(Rectangle((x, yy), w, h, fc=MATS[m][1], ec="white", lw=0.3, zorder=z))
        yy += h
    return yy


def swatch(ax, x, y, s, lab, z=3, lw=0.4):
    ax.add_patch(Rectangle((x, y), s, s, fc=S.lab_to_srgb(lab), ec=S.INK, lw=lw, zorder=z))


def header(ax, x, w, title, sub, ytop):
    ax.text(x + w / 2, ytop - 0.22, title, ha="center", va="top", fontweight="bold",
            fontsize=S.FONT_PT)
    if sub:
        ax.text(x + w / 2, ytop - 0.58, sub, ha="center", va="top", fontsize=S.SMALL_PT - 0.5,
                color="#404040")


def mini_axes(ax, x, y, w, h):
    ax.plot([x, x, x + w], [y + h, y, y], color=S.INK, lw=0.5, zorder=4)


def main_figure():
    W, H = 17.0, 5.6
    fig = plt.figure(figsize=(W * S.CM, H * S.CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top, bot = H - 0.08, 0.1
    cols = [(0.08, 2.55), (2.98, 2.6), (6.03, 2.6), (9.08, 2.6), (12.13, 2.6), (15.16, 1.78)]
    heads = [("h_in", "s_in"), ("h_gen", "s_gen"), ("h_sim", "s_sim"),
             ("h_rank", "s_rank"), ("h_ref", "s_ref"), ("h_out", "s_out")]
    for (x, w), (h, s) in zip(cols, heads):
        box(ax, x, bot, w, top - bot)
        header(ax, x, w, LABELS[h], LABELS[s], top)
    ymid = 2.55
    for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
        arrow(ax, x0 + w0 + 0.04, ymid, x1 - 0.04, ymid, lw=1.1, ms=8)

    # ---- inputs ---------------------------------------------------------
    x, w = cols[0]
    lam = np.linspace(300, 900, 80)
    ax.text(x + 0.15, 4.72, LABELS["pool"], fontsize=S.SMALL_PT - 0.5, va="bottom")
    th = 0.52
    for i, (name, c) in enumerate(MATS):
        y0 = 4.6 - (i + 1) * (th + 0.07)
        ax.add_patch(Rectangle((x + 0.15, y0), w - 0.3, th, fc="white", ec="#BBBBBB", lw=0.4, zorder=2))
        ax.add_patch(Rectangle((x + 0.15, y0), 0.16, th, fc=c, ec="none", zorder=3))
        ax.text(x + 0.38, y0 + th / 2, name, va="center", fontsize=S.SMALL_PT - 0.5)
        n, k = nk_icon(i, lam)
        px0, pw = x + 1.2, w - 1.45
        ax.plot(px0 + (lam - 300) / 600 * pw, y0 + 0.06 + n / 6.2 * (th - 0.12), color=S.INK, lw=0.6, zorder=4)
        ax.plot(px0 + (lam - 300) / 600 * pw, y0 + 0.06 + k / 6.2 * (th - 0.12), color=S.INK, lw=0.6, ls=(0, (1.6, 1)), zorder=4)
    ax.text(x + 0.15, 1.42, LABELS["target"], fontsize=S.SMALL_PT - 0.5, va="bottom")
    swatch(ax, x + 0.15, 0.32, 0.95, TARGET_LAB)
    ax.text(x + 1.25, 0.8, LABELS["target_sub"], fontsize=S.SMALL_PT - 0.5, va="center")

    # ---- sample (with constraint gate) ----------------------------------
    x, w = cols[1]
    box(ax, x + 0.3, 3.75, w - 0.6, 0.62, fc="white", ec=S.INK, lw=0.7, r=0.1, z=2)
    ax.text(x + w / 2, 4.06, LABELS["model"], ha="center", va="center", fontsize=S.SMALL_PT)
    # next-layer distribution: 4 materials x a few thickness bars
    bx0, by0, bw = x + 0.3, 1.55, w - 0.6
    rng = np.random.default_rng(3)
    heights = [rng.uniform(0.15, 1.0, 4) * f for f in (0.8, 1.0, 0.6, 0.9)]
    gw = bw / 4
    for m, hs in enumerate(heights):
        for j, hh in enumerate(hs):
            xb = bx0 + m * gw + 0.07 + j * (gw - 0.14) / 4
            masked = m == MASKED
            ax.add_patch(Rectangle((xb, by0), (gw - 0.14) / 4 * 0.8, hh * 0.85,
                                   fc="white" if masked else MATS[m][1],
                                   ec="#9A9A9A" if masked else "none", lw=0.4,
                                   ls=(0, (1.5, 1)) if masked else "-", zorder=3))
    ax.plot([bx0, bx0 + bw], [by0, by0], color=S.INK, lw=0.5, zorder=4)
    ax.text(bx0, by0 - 0.08, LABELS["next"], fontsize=S.SMALL_PT - 1, va="top", color="#404040")
    # gate: a bar closed across the masked material's tokens
    gx = bx0 + MASKED * gw + 0.03
    ax.add_patch(Rectangle((gx, by0 - 0.02), gw - 0.06, 1.0, fc="none", ec=S.INK, lw=0.8,
                           hatch="////", zorder=5))
    ax.text(gx + (gw - 0.06) / 2, by0 + 1.0, "×", ha="center", va="bottom",
            fontsize=S.FONT_PT + 1, fontweight="bold", zorder=6)
    ax.text(gx + (gw - 0.06) / 2, by0 + 1.62, LABELS["gate"], ha="center", va="center",
            fontsize=S.SMALL_PT - 0.5, linespacing=0.95)
    ax.text(bx0 + 1.5 * gw - 0.08, 3.25, LABELS["loop"], fontsize=S.SMALL_PT - 1,
            va="center", ha="right", color="#404040")
    arrow(ax, bx0 + 1.5 * gw, 3.75, bx0 + 1.5 * gw, 2.75, lw=0.7, ms=5)
    # sampled stacks
    cands = [[(2, 40), (1, 60)], [(0, 90), (2, 30), (1, 50)], [(1, 70)],
             [(2, 20), (0, 60), (1, 40), (0, 30)], [(0, 50), (2, 50)]]
    for i, L in enumerate(cands):
        stack(ax, x + 0.3 + i * 0.36, 0.3, 0.26, L, scale=0.0042, sub_h=0.08)
    ax.text(x + 0.3 + 5 * 0.36 + 0.02, 0.4, "… × N", fontsize=S.SMALL_PT - 0.5, va="bottom")

    # ---- simulate ---------------------------------------------------------
    x, w = cols[2]
    stack(ax, x + 0.3, 2.62, 0.5, cands[1], scale=0.008)
    # light rays
    arrow(ax, x + 0.22, 4.55, x + 0.5, 4.14, lw=0.6, ms=5, color="#606060")
    arrow(ax, x + 0.6, 4.14, x + 0.88, 4.55, lw=0.6, ms=5, color="#606060")
    ax.text(x + 0.98, 4.5, LABELS["light"], fontsize=S.SMALL_PT - 1, color="#404040", va="center")
    # reflectance
    px, py, pw, ph = x + 0.95, 1.3, w - 1.15, 1.15
    mini_axes(ax, px, py, pw, ph)
    l = np.linspace(0, 1, 60)
    R = 0.15 + 0.55 * np.exp(-((l - 0.18) / 0.16) ** 2) + 0.35 * np.exp(-((l - 0.95) / 0.12) ** 2)
    ax.plot(px + l * pw, py + R * ph * 0.95, color=S.INK, lw=0.8, zorder=5)
    ax.text(px + 0.05, py + ph + 0.02, LABELS["refl"], ha="left", va="bottom",
            fontsize=S.SMALL_PT - 0.5)
    ax.text(px + pw, py - 0.06, "λ", ha="right", va="top", fontsize=S.SMALL_PT - 0.5)
    arrow(ax, x + 0.55, 2.55, x + 0.85, 2.05, lw=0.6, ms=5)
    # swatches per candidate
    for i, lab in enumerate(CANDIDATE_LABS):
        swatch(ax, x + 0.3 + i * 0.42, 0.35, 0.34, lab)
    arrow(ax, px + pw / 2, py - 0.25, px + pw / 2 - 0.4, 0.8, lw=0.6, ms=5)

    # ---- rank ---------------------------------------------------------------
    x, w = cols[3]
    t = np.array(TARGET_LAB, float)
    de = [np.linalg.norm(np.array(c) - t) for c in CANDIDATE_LABS]   # proxy for ordering only
    order = np.argsort(de)
    ax.text(x + 0.95, 4.0, LABELS["de"], fontsize=S.SMALL_PT - 0.5, va="bottom")
    for r, i in enumerate(order):
        y0 = 3.55 - r * 0.62
        swatch(ax, x + 0.4, y0, 0.42, CANDIDATE_LABS[i])
        ax.add_patch(Rectangle((x + 0.95, y0 + 0.11), de[i] / max(de) * (w - 1.25), 0.2,
                               fc=S.INK if r < 2 else "#B5B5B5", ec="none", zorder=3))
    # top-k bracket
    yk0, yk1 = 3.55 - 0.62 + 0.0, 3.55 + 0.42
    ax.plot([x + 0.28, x + 0.2, x + 0.2, x + 0.28], [yk1, yk1, yk0, yk0], color=S.INK, lw=0.7)
    ax.text(x + 0.14, (yk0 + yk1) / 2, LABELS["topk"], rotation=90, ha="right", va="center",
            fontsize=S.SMALL_PT - 1)
    swatch(ax, x + 0.4, 0.32, 0.42, TARGET_LAB, lw=0.8)
    ax.text(x + 0.95, 0.53, LABELS["target"].lower(), va="center", fontsize=S.SMALL_PT - 0.5)

    # ---- refine -------------------------------------------------------------
    x, w = cols[4]
    sx = x + 0.3
    yy = 2.75 + 0.12
    stack(ax, sx, 2.75, 0.6, cands[1], scale=0.009)
    for m, tk in cands[1]:
        h = tk * 0.009
        ax.annotate("", xy=(sx + 0.75, yy + h), xytext=(sx + 0.75, yy),
                    arrowprops=dict(arrowstyle="<->", lw=0.5, color=S.INK, shrinkA=0, shrinkB=0,
                                    mutation_scale=4))
        yy += h
    ax.text(sx + 0.92, 3.5, LABELS["thick"], fontsize=S.SMALL_PT - 1, va="center",
            linespacing=0.95)
    px, py, pw, ph = x + 0.45, 0.45, w - 0.7, 1.55
    mini_axes(ax, px, py, pw, ph)
    s = np.linspace(0, 1, 50)
    ax.plot(px + s * pw, py + ph * (0.12 + 0.8 * np.exp(-4.5 * s)), color=S.INK, lw=0.8, zorder=5)
    ax.text(px - 0.06, py + ph, LABELS["de"], ha="right", va="top", fontsize=S.SMALL_PT - 0.5)
    ax.text(px + pw, py - 0.06, LABELS["iter"], ha="right", va="top", fontsize=S.SMALL_PT - 0.5)

    # ---- output -------------------------------------------------------------
    x, w = cols[5]
    stack(ax, x + 0.55, 2.7, 0.68, [(0, 84), (2, 33), (1, 47)], scale=0.009)
    ax.text(x + w / 2, 2.5, LABELS["stack"], ha="center", va="top", fontsize=S.SMALL_PT - 0.5)
    sw = 0.62
    sx = x + w / 2 - sw
    ax.add_patch(Rectangle((sx, 0.75), sw, sw, fc=S.lab_to_srgb(TARGET_LAB), ec="none", zorder=3))
    ax.add_patch(Rectangle((sx + sw, 0.75), sw, sw, fc=S.lab_to_srgb((58.4, 33.2, -39.5)),
                           ec="none", zorder=3))
    ax.add_patch(Rectangle((sx, 0.75), 2 * sw, sw, fc="none", ec=S.INK, lw=0.5, zorder=4))
    ax.text(sx + sw / 2, 0.68, LABELS["swatch_t"], ha="center", va="top", fontsize=S.SMALL_PT - 1)
    ax.text(sx + 1.5 * sw, 0.68, LABELS["swatch_r"], ha="center", va="top", fontsize=S.SMALL_PT - 1)
    return fig


def toc_figure():
    W, H = 8.0, 4.0
    fig = plt.figure(figsize=(W * S.CM, H * S.CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    dy = 0.45   # centre the content vertically
    for i, (_, c) in enumerate(MATS[:3]):
        ax.add_patch(Rectangle((0.3 + i * 0.5, 2.4 + dy), 0.4, 0.4, fc=c, ec="none"))
    ax.text(1.0, 2.27 + dy, LABELS["toc_pool"], ha="center", va="top", fontsize=S.FONT_PT)
    swatch(ax, 0.68, 0.65 + dy, 0.65, TARGET_LAB, lw=0.5)
    ax.text(1.0, 0.52 + dy, LABELS["toc_target"], ha="center", va="top", fontsize=S.FONT_PT)
    box(ax, 2.6, 1.4 + dy, 2.05, 1.05, fc="white", ec=S.INK, lw=1.0, r=0.18, z=2)
    ax.text(3.625, 1.925 + dy, LABELS["toc_model"], ha="center", va="center",
            fontsize=S.FONT_PT + 2, fontweight="bold")
    arrow(ax, 1.85, 2.55 + dy, 2.55, 2.12 + dy, lw=1.1, ms=8)
    arrow(ax, 1.48, 1.0 + dy, 2.55, 1.7 + dy, lw=1.1, ms=8)
    arrow(ax, 4.72, 1.925 + dy, 5.4, 1.925 + dy, lw=1.1, ms=8)
    stack(ax, 5.6, 1.08 + dy, 0.95, [(0, 84), (2, 33), (1, 47)], scale=0.0105, sub_h=0.18)
    swatch(ax, 6.8, 1.4 + dy, 0.8, (58.4, 33.2, -39.5), lw=0.5)
    ax.text(6.45, 0.93 + dy, LABELS["toc_out"], ha="center", va="top", fontsize=S.FONT_PT)
    return fig


def main():
    S.apply()
    fig = main_figure()
    S.save(fig, "fig1_pipeline")
    toc = toc_figure()
    S.save_tiff(toc, "toc_graphic")
    toc.savefig(S.OUT_DIR / "toc_graphic.pdf")
    S.write_provenance(
        "Fig 1", "`scripts/figures/fig1_pipeline.py` -> `fig1_pipeline.{pdf,png}` (17 cm) and "
        "`toc_graphic.{tiff,pdf}` (8 x 4 cm, 600 dpi). Schematic, no data. Stage order and "
        "semantics from inference/src/solve.py (generate -> simulate/select -> refine), "
        "constraint gate from generate.py (decode masks ANDed into next-layer logits), "
        "projected DoG from refine.py (refine_optimizer='dog', box projection). "
        "Spectra, stacks and colors are illustrative glyphs.")


if __name__ == "__main__":
    main()
