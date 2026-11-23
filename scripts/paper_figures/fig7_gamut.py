#!/usr/bin/env python3
"""Fig. 7: out-of-distribution targets, the gamut batteries.

Every test-tier target is reachable by construction; these 28 are not
guaranteed to have a solution. One tile per target, three vertical strips:
INDIGO's result | the target | random proposals' result, so each result abuts
the color it was asked for. Dotted lines separate the strips. DeltaE00 of
each run under the tile (INDIGO bold). Rows are the five batteries of
inference/scripts/gamut_eval.py. A dagger marks a target outside sRGB (its
strip is clipped for display; the printed DeltaE00 is the comparison).

Both runs use the paper's search setting (preset "strong": N = 250, top-10
all refined, 100 DoG steps), the production checkpoint, the same pool and
seed. The script stops if the two files disagree on any of that or on the
targets.

    python scripts/paper_figures/fig7_gamut.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import _style as S

RUNS = {"INDIGO": S.REPO / "results/prod_eval/gamut/gamut_strong_dog.json",
        "random": S.REPO / "results/prod_eval/gamut/gamut_strong_dog_random.json"}
SAME = ("checkpoint", "model_sha256", "preset", "optimizer", "seed", "pool_size")

LABELS = {
    "rgb_primaries": "sRGB\nprimaries",
    "lab_l_sweep": "L* sweep",
    "lab_a_sweep": "a* sweep",
    "lab_b_sweep": "b* sweep",
    "chromatic_corners": "Chromatic\ncorners",
    "key_model": "INDIGO",
    "key_target": "target",
    "key_random": "baseline",
    "lab": "({:.0f}, {:.0f}, {:.0f})",
    "gamut": "† target outside sRGB; its strip is clipped for display",
    "lab_head": "Target L*a*b*",
    "de_head": r"Under each tile: $\Delta E_{00}$ of INDIGO (bold) / baseline",
    "key_head": "Each tile:",
}


def load():
    d = {k: json.load(open(p)) for k, p in RUNS.items()}
    m, r = d["INDIGO"], d["random"]
    for k in SAME:
        if m[k] != r[k]:
            sys.exit(f"runs differ on {k}: {m[k]} vs {r[k]}")
    if m["preset"] != "strong" or m.get("proposer") != "model" or r.get("proposer") != "random":
        sys.exit("expected preset strong, proposers model and random")
    for b in m["batteries"]:
        tm = [t["target_lab"] for t in m["batteries"][b]["per_target"]]
        tr = [t["target_lab"] for t in r["batteries"][b]["per_target"]]
        if not np.allclose(tm, tr):
            sys.exit(f"battery {b}: targets differ")
    return m, r


def strip(ax, x, y, w, h, lab):
    if lab is None:
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", hatch="////"))
    else:
        ax.add_patch(Rectangle((x, y), w, h, fc=S.lab_to_srgb(lab), ec="none"))


def main():
    S.apply()
    m, r = load()
    batteries = list(m["batteries"])
    ncol = max(len(m["batteries"][b]["per_target"]) for b in batteries)

    W = 17.0
    x0, tw, gap = 2.0, 1.6, 0.27        # first tile x, tile width, gap
    th, rh = 1.05, 2.05                  # tile height, row pitch
    H = 0.75 + rh * len(batteries) + 0.75
    fig = plt.figure(figsize=(W * S.CM, H * S.CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    any_out = False
    for ri, b in enumerate(batteries):
        ytop = H - 0.75 - ri * rh
        ax.text(0.15, ytop - 0.35 - th / 2, LABELS[b], va="center", ha="left",
                fontweight="bold", linespacing=1.0)
        for ci, (tm, tr) in enumerate(zip(m["batteries"][b]["per_target"],
                                          r["batteries"][b]["per_target"])):
            x = x0 + ci * (tw + gap)
            y = ytop - 0.35 - th
            tgt = tm["target_lab"]
            out = not S.lab_in_srgb(tgt)
            any_out |= out
            w3 = tw / 3
            strip(ax, x, y, w3, th, tm.get("achieved_lab"))
            strip(ax, x + w3, y, w3, th, tgt)
            strip(ax, x + 2 * w3, y, w3, th, tr.get("achieved_lab"))
            ax.add_patch(Rectangle((x, y), tw, th, fc="none", ec=S.INK, lw=0.4))
            for xm in (x + w3, x + 2 * w3):
                ax.plot([xm, xm], [y, y + th], color=S.INK, lw=0.8, ls=(0, (1, 1.2)),
                        dash_capstyle="butt", zorder=4)
            ax.text(x + tw / 2, y + th + 0.06, LABELS["lab"].format(*(v + 0.0 for v in np.round(tgt))) + ("†" if out else ""),
                    ha="center", va="bottom", fontsize=S.SMALL_PT - 1.5, color="#404040")
            dm, dr = tm.get("delta_e"), tr.get("delta_e")
            fm = "–" if dm is None else f"{dm:.1f}"
            fr = "–" if dr is None else f"{dr:.1f}"
            ax.text(x + tw / 2 - 0.05, y - 0.08, fm, ha="right", va="top",
                    fontsize=S.SMALL_PT - 0.5, fontweight="bold")
            ax.text(x + tw / 2, y - 0.08, "/", ha="center", va="top",
                    fontsize=S.SMALL_PT - 0.5, color="#808080")
            ax.text(x + tw / 2 + 0.05, y - 0.08, fr, ha="left", va="top",
                    fontsize=S.SMALL_PT - 0.5)

    # key tile at the top right
    kx, ky, kw, kh = W - 0.15 - 3.3, H - 0.62, 3.3, 0.42
    for j, (k, fc) in enumerate((("key_model", "#E8E8E8"), ("key_target", "#BDBDBD"),
                                 ("key_random", "#E8E8E8"))):
        ax.add_patch(Rectangle((kx + j * kw / 3, ky), kw / 3, kh, fc=fc, ec="none"))
        ax.text(kx + (j + 0.5) * kw / 3, ky + kh / 2, LABELS[k], ha="center", va="center",
                fontsize=S.SMALL_PT - 0.5)
    ax.add_patch(Rectangle((kx, ky), kw, kh, fc="none", ec=S.INK, lw=0.4))
    for j in (1, 2):
        ax.plot([kx + j * kw / 3] * 2, [ky, ky + kh], color=S.INK, lw=0.8, ls=(0, (1, 1.2)))
    ax.text(kx - 0.15, ky + kh / 2, LABELS["key_head"], ha="right", va="center",
            fontsize=S.SMALL_PT - 0.5, color="#404040")
    ax.text(0.15, 0.3, LABELS["de_head"], ha="left", va="center", fontsize=S.SMALL_PT - 0.5,
            color="#404040")
    if any_out:
        ax.text(W - 0.15, 0.3, LABELS["gamut"], ha="right", va="center",
                fontsize=S.SMALL_PT - 0.5, color="#404040")

    S.save(fig, "fig7_gamut")
    so, sr = m["overall"]["stats"], r["overall"]["stats"]
    per_b = "; ".join(f"{b} {m['batteries'][b]['stats']['median_de']:.2f} vs "
                      f"{r['batteries'][b]['stats']['median_de']:.2f}" for b in batteries)
    S.write_provenance(
        "Fig 7", "`scripts/paper_figures/fig7_gamut.py` -> `fig7_gamut.{pdf,png}`. "
        "`results/prod_eval/gamut/gamut_strong_dog.json` (INDIGO) and `gamut_strong_dog_random.json` "
        "(random): batteries.<b>.per_target[].{target_lab, achieved_lab, delta_e}; preset strong "
        f"(N=250, top-10 all refined, 100 DoG steps), checkpoint {m['model_sha256']}, seed {m['seed']}, "
        f"pool {m['pool_size']}; checked identical settings and targets. Caption numbers (not drawn): "
        f"overall median DeltaE00 {so['median_de']:.2f} vs {sr['median_de']:.2f}, DeltaE<2 "
        f"{100 * so['pass_rate_de_lt_2']:.0f}% vs {100 * sr['pass_rate_de_lt_2']:.0f}% (n = {so['n']}); "
        f"per-battery medians INDIGO vs random: {per_b}.")


if __name__ == "__main__":
    main()
