#!/usr/bin/env python3
"""Fig. 3: model architecture (schematic, no data), for a chemistry audience.

Shapes are read from the code, not typed in:
  src/materials_vocab.py   M_MAX (pool slots), THICKNESSES / NUM_THICKNESSES,
                           MAX_LAYERS, VOCAB_SIZE
  src/material_features.py NUM_LAMBDA (wavelength samples per n and k)
  results/production/current/config.json   d_model, slot_encoder_layers,
                           decoder_layers of the production model
Structure follows src/model.py FlexMaterialCrossAttn:
  each slot's n,k -> MaterialEncoder (same weights for every slot)
  -> slot encoder (self-attention over slots, NO positional embedding on
     slots, so the pool is a set)
  -> decoder: start token = target Lab (+ pool size), then one token per
     layer already placed (the partial stack), reading the slot vectors
  -> pointer head: for every slot, a score for each thickness bin
     (thickness_head on [slot vector, decoder state]) + a stop score
     (eos_head) = M_MAX * NUM_THICKNESSES + 1 choices.
Attention internals are deliberately not drawn. Edit wording in LABELS.

    python scripts/paper_figures/fig3_architecture.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

import _style as S
from src.materials_vocab import M_MAX, NUM_THICKNESSES, THICKNESSES, VOCAB_SIZE, MAX_LAYERS
from src.material_features import NUM_LAMBDA

CFG = json.load(open(S.REPO / "results/production/current/config.json"))
D_MODEL = CFG["d_model"]
SLOT_LAYERS = CFG["slot_encoder_layers"]
DEC_LAYERS = CFG["decoder_layers"]
assert M_MAX * NUM_THICKNESSES + 1 == VOCAB_SIZE == CFG["vocab_size"]

LABELS = {
    "pool": "Material pool",
    "pool_sub": f"up to {M_MAX} slots\nn, k at {NUM_LAMBDA} wavelengths",
    "pad": "empty slot",
    "enc": "Material\nencoder",
    "enc_sub": "same weights\nfor every slot",
    "slotenc": "Slot encoder",
    "slotenc_sub": (f"{SLOT_LAYERS} layers\n\nslots exchange\ninformation\n\n"
                    "no slot order:\nthe pool is a set"),
    "vec": "Slot\nvectors",
    "vec_sub": f"{D_MODEL} numbers\nper slot",
    "target": "Target color\n(L*a*b*)",
    "partial": "Layers placed\nso far",
    "dec": "Decoder",
    "dec_sub": f"{DEC_LAYERS} layer, width {D_MODEL}\nreads the slot vectors",
    "scores": "Scores for the next layer",
    "grid_x": (f"thickness bin ({NUM_THICKNESSES} bins, "
               f"{THICKNESSES[0]}–{THICKNESSES[-1]} nm)"),
    "grid_y": "one row per slot",
    "stop": "stop",
    "total": f"{M_MAX} × {NUM_THICKNESSES} + 1 stop = {VOCAB_SIZE} choices",
    "loop": f"pick one (slot, thickness) and add it to the stack; repeat, up to {MAX_LAYERS} layers",
    "masked": "masked",
    "bus": "decoder state, combined\nwith each slot vector",
}

MATS = [("SiO$_2$", S.OI["sky"]), ("TiO$_2$", S.OI["green"]),
        ("aSi", S.OI["orange"]), ("Ag", "#9A9A9A")]
PANEL = "#F3F3F3"
TARGET_LAB = (58, 34, -40)


def box(ax, x, y, w, h, fc=PANEL, ec="none", lw=0.6, r=0.12, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, lw=0.7, ms=5, color=S.INK, z=6, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0, **kw))


def vdots(ax, x, y):
    ax.plot([x, x, x], [y - 0.12, y, y + 0.12], "o", ms=1.4, color=S.INK, zorder=4)


def nk_icon(i, lam):
    x = (lam - 300) / 600
    if i == 0:
        return 1.46 + 0.05 * (1 - x), 0 * x
    if i == 1:
        return 2.3 + 0.8 * np.exp(-6 * x), 0.6 * np.exp(-25 * x)
    if i == 2:
        return 4.4 - 0.9 * x + 0.3 * np.exp(-20 * (x - .25) ** 2), 2.2 * np.exp(-6 * x)
    return 0.15 + 0.1 * x, 1.5 + 4.2 * x


def main():
    S.apply()
    W, H = 17.0, 7.4
    fig = plt.figure(figsize=(W * S.CM, H * S.CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fs, fss = S.SMALL_PT, S.SMALL_PT - 1

    # rows: 4 materials, an ellipsis row, one padded slot
    row_h, gap = 0.5, 0.14
    rows_y = [6.0 - i * (row_h + gap) for i in range(6)]   # bottom y of each row
    rows_y[5] -= 0.0
    kinds = ["m0", "m1", "m2", "m3", "dots", "pad"]
    lam = np.linspace(300, 900, 60)

    # ---- column A: pool --------------------------------------------------
    xa, wa = 0.15, 2.25
    ax.text(xa, 6.95, LABELS["pool"], fontweight="bold", va="center")
    ax.text(xa, 2.25, LABELS["pool_sub"], fontsize=fss, va="top", color="#404040",
            linespacing=1.05)
    for k, y in zip(kinds, rows_y):
        if k == "dots":
            vdots(ax, xa + wa / 2, y + row_h / 2)
            continue
        if k == "pad":
            ax.add_patch(Rectangle((xa, y), wa, row_h, fc="white", ec="#BBBBBB", lw=0.5,
                                   ls=(0, (2, 1.5)), zorder=2))
            ax.text(xa + wa / 2, y + row_h / 2, LABELS["pad"], ha="center", va="center",
                    fontsize=fss, color="#808080")
            continue
        i = int(k[1])
        ax.add_patch(Rectangle((xa, y), wa, row_h, fc="white", ec="#BBBBBB", lw=0.4, zorder=2))
        ax.add_patch(Rectangle((xa, y), 0.15, row_h, fc=MATS[i][1], ec="none", zorder=3))
        ax.text(xa + 0.22, y + row_h / 2, MATS[i][0], va="center", fontsize=fss + 0.5)
        n, kk = nk_icon(i, lam)
        px, pw = xa + 1.0, wa - 1.1
        for v, ls in ((n, "-"), (kk, (0, (1.6, 1)))):
            ax.plot(px + (lam - 300) / 600 * pw, y + 0.05 + v / 6.2 * (row_h - 0.1),
                    color=S.INK, lw=0.55, ls=ls, zorder=4)

    # ---- column B: shared material encoder ---------------------------------
    xb, wb = 2.85, 0.75
    ax.text(xb + wb / 2, 6.95, LABELS["enc"], ha="center", va="center", fontweight="bold",
            linespacing=0.95)
    for k, y in zip(kinds, rows_y):
        if k == "dots":
            continue
        pad = k == "pad"
        ax.add_patch(Rectangle((xb, y + 0.06), wb, row_h - 0.12, fc="white" if pad else "#DDE6F0",
                               ec="#BBBBBB" if pad else S.INK, lw=0.5,
                               ls=(0, (2, 1.5)) if pad else "-", zorder=3))
        arrow(ax, xa + wa + 0.05, y + row_h / 2, xb - 0.04, y + row_h / 2, ms=4, lw=0.5,
              color="#9A9A9A" if pad else S.INK)
    ax.text(xb + wb / 2, rows_y[-1] - 0.12, LABELS["enc_sub"], ha="center", va="top",
            fontsize=fss, color="#404040", linespacing=1.0)

    # ---- column C: slot encoder ----------------------------------------------
    xc, wc = 4.0, 1.7
    yc0, yc1 = rows_y[-1] - 0.05, rows_y[0] + row_h + 0.05
    box(ax, xc, yc0, wc, yc1 - yc0, fc="white", ec=S.INK, lw=0.8, r=0.15, z=2)
    ax.text(xc + wc / 2, 6.95, LABELS["slotenc"], ha="center", va="center", fontweight="bold")
    ax.text(xc + wc / 2, (yc0 + yc1) / 2, LABELS["slotenc_sub"], ha="center", va="center",
            fontsize=fss + 0.5, linespacing=1.05, zorder=3)
    for k, y in zip(kinds, rows_y):
        if k == "dots":
            continue
        arrow(ax, xb + wb + 0.04, y + row_h / 2, xc - 0.04, y + row_h / 2, ms=4, lw=0.5,
              color="#9A9A9A" if k == "pad" else S.INK)

    # ---- column D: slot vectors ------------------------------------------------
    xd, wd = 6.25, 0.85
    ax.text(xd + wd / 2, 6.95, LABELS["vec"], ha="center", va="center", fontweight="bold",
            linespacing=0.95)
    rng = np.random.default_rng(1)
    for k, y in zip(kinds, rows_y):
        if k == "dots":
            vdots(ax, xd + wd / 2, y + row_h / 2)
            continue
        if k == "pad":   # padded slots are masked out; no vector, no score
            continue
        arrow(ax, xc + wc + 0.04, y + row_h / 2, xd - 0.04, y + row_h / 2, ms=4, lw=0.5)
        vals = rng.uniform(0.15, 0.9, 8)
        for j, v in enumerate(vals):
            ax.add_patch(Rectangle((xd + j * wd / 8, y + 0.1), wd / 8, row_h - 0.2,
                                   fc=plt.cm.Greys(v), ec="none", zorder=3))
        ax.add_patch(Rectangle((xd, y + 0.1), wd, row_h - 0.2, fc="none", ec=S.INK, lw=0.4, zorder=4))

    # ---- column E: score grid (pointer) ------------------------------------------
    xe, we = 8.05, 7.4
    ax.text(xe + we / 2, 6.95, LABELS["scores"], ha="center", va="center", fontweight="bold")
    nb = NUM_THICKNESSES
    cell = we / nb
    tgrid = np.arange(nb)
    peaks = {0: (40, 9, 0.5), 1: (22, 5, 1.0), 2: (12, 4, 0.35), 3: (70, 12, 0.08)}
    for k, y in zip(kinds, rows_y):
        if k == "dots":
            vdots(ax, xe + we / 2, y + row_h / 2)
            continue
        x_from = xe - 0.42 if k == "pad" else xd + wd + 0.04
        arrow(ax, x_from, y + row_h / 2, xe - 0.05, y + row_h / 2, ms=4, lw=0.5)
        if k == "pad":
            ax.add_patch(Rectangle((xe, y + 0.06), we, row_h - 0.12, fc="white", ec="#BBBBBB",
                                   lw=0.5, hatch="////", zorder=3))
            ax.text(xe + we / 2, y + row_h / 2, LABELS["masked"], ha="center", va="center",
                    fontsize=fss, color="#606060", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            continue
        c, s, a = peaks[int(k[1])]
        sc = 0.05 + a * np.exp(-0.5 * ((tgrid - c) / s) ** 2) + 0.04 * rng.random(nb)
        img = np.clip(sc, 0, 1)[None, :]
        ax.imshow(img, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(xe, xe + we, y + 0.06, y + row_h - 0.06), zorder=3)
        ax.add_patch(Rectangle((xe, y + 0.06), we, row_h - 0.12, fc="none", ec=S.INK, lw=0.4, zorder=4))
        ax.add_patch(Rectangle((xe - 0.12, y + 0.06), 0.08, row_h - 0.12, fc=MATS[int(k[1])][1],
                               ec="none", zorder=4))
    # thickness axis
    ya = rows_y[-1] - 0.05
    for t in (2, 50, 100, 150, 200):
        xt = xe + (THICKNESSES.index(t) + 0.5) * cell
        ax.plot([xt, xt], [ya, ya - 0.07], color=S.INK, lw=0.5)
        ax.text(xt, ya - 0.1, str(t), ha="center", va="top", fontsize=fss)
    ax.text(xe + we / 2, ya - 0.36, LABELS["grid_x"], ha="center", va="top", fontsize=fss)
    # stop cell
    xs_ = xe + we + 0.3
    ax.add_patch(Rectangle((xs_, rows_y[1] + 0.06), 0.55, row_h - 0.12,
                           fc=plt.cm.Blues(0.25), ec=S.INK, lw=0.5, zorder=3))
    ax.text(xs_ + 0.275, rows_y[1] + row_h + 0.03, LABELS["stop"], ha="center", va="bottom", fontsize=fss)

    # ---- bottom: decoder ----------------------------------------------------------
    ax.text(xd + wd / 2, rows_y[-1] + row_h / 2, LABELS["vec_sub"], ha="center", va="center",
            fontsize=fss, color="#404040", linespacing=1.0)
    yb = 0.45
    sw = 0.5
    ax.add_patch(Rectangle((0.3, 1.25), sw, sw, fc=S.lab_to_srgb(TARGET_LAB), ec=S.INK, lw=0.5, zorder=3))
    ax.text(0.9, 1.5, LABELS["target"], va="center", fontsize=fss + 0.5, linespacing=1.0)
    sx, sy = 0.3, 0.35
    ax.add_patch(Rectangle((sx, sy), sw, 0.08, fc="#4D4D4D", ec="none", zorder=3))
    ax.add_patch(Rectangle((sx, sy + 0.08), sw, 0.25, fc=MATS[0][1], ec="white", lw=0.3, zorder=3))
    ax.add_patch(Rectangle((sx, sy + 0.33), sw, 0.17, fc=MATS[2][1], ec="white", lw=0.3, zorder=3))
    ax.text(0.9, 0.6, LABELS["partial"], va="center", fontsize=fss + 0.5, linespacing=1.0)
    xdec, wdec, hdec = 4.0, 3.1, 1.15
    box(ax, xdec, yb, wdec, hdec, fc="#DDE6F0", ec=S.INK, lw=0.8, r=0.15, z=2)
    ax.text(xdec + wdec / 2, yb + 0.8, LABELS["dec"], ha="center", va="center", fontweight="bold", zorder=3)
    ax.text(xdec + wdec / 2, yb + 0.36, LABELS["dec_sub"], ha="center", va="center", fontsize=fss,
            linespacing=1.0, zorder=3)
    arrow(ax, 2.3, 1.5, xdec - 0.05, yb + 0.8, ms=5)
    arrow(ax, 2.3, 0.6, xdec - 0.05, yb + 0.35, ms=5)
    # decoder reads the slot vectors
    arrow(ax, xd + wd / 2, rows_y[-1] - 0.05, xd + wd / 2, yb + hdec + 0.03, ms=5)
    # pointer: decoder state joins every slot's path into the score grid
    xbus = xe - 0.42
    ax.plot([xdec + wdec, xbus, xbus], [yb + 0.75, yb + 0.75, rows_y[0] + row_h / 2],
            color=S.OI["vermilion"], lw=0.9, zorder=7, solid_joinstyle="miter")
    for k, y in zip(kinds, rows_y):
        if k != "dots":
            ax.plot(xbus, y + row_h / 2, "o", ms=2.6, color=S.OI["vermilion"], zorder=8)
    ax.text(xbus + 0.08, yb + 0.95, LABELS["bus"], fontsize=fss, color=S.OI["vermilion"],
            va="bottom", ha="left", linespacing=1.0)
    # stop score comes from the decoder alone
    xs_c = xs_ + 0.275
    ax.plot([xbus, xs_c, xs_c], [yb + 0.75, yb + 0.75, rows_y[1] - 0.2], color=S.OI["vermilion"],
            lw=0.9, zorder=7)
    arrow(ax, xs_c, rows_y[1] - 0.25, xs_c, rows_y[1] + 0.04, ms=5, color=S.OI["vermilion"])
    ax.text(xe + we / 2 + 0.6, yb + 0.35, LABELS["total"], ha="center", va="center",
            fontsize=fss + 0.5, fontweight="bold")
    ax.text(xe + we / 2 + 0.6, yb + 0.02, LABELS["loop"], ha="center", va="center", fontsize=fss,
            color="#404040")

    S.save(fig, "fig3_architecture")
    S.write_provenance(
        "Fig 3", "`scripts/paper_figures/fig3_architecture.py` -> `fig3_architecture.{pdf,png}`. "
        "Schematic. Sizes read at run time: src/materials_vocab.py {M_MAX=32, NUM_THICKNESSES=100, "
        "THICKNESSES 2-200 nm, VOCAB_SIZE=3201, MAX_LAYERS=10}, src/material_features.py NUM_LAMBDA=128, "
        "results/production/current/config.json {d_model=352, slot_encoder_layers=6, decoder_layers=1}. "
        "Structure from src/model.py FlexMaterialCrossAttn (shared MaterialEncoder, slot encoder with no "
        "slot positional embedding, decoder over [target Lab + pool size] + placed layers, per-slot "
        "thickness head + EOS head). Score patterns are illustrative.")


if __name__ == "__main__":
    main()
