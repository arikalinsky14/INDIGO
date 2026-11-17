#!/usr/bin/env python3
"""Fig. 6: where the proposer matters. Example targets chosen by a fixed rule.

Selection rule: tiers A and B pooled; rows where both runs returned a stack
and the target is a physical color (L* in [0, 100], C* <= 150; 10 of the
1000 headline rows are not, see Fig. 2). For each of the two chromatic
buckets (8 <= C* < 30 and C* >= 30), compute per target the gain
g = DeltaE00(random) - DeltaE00(INDIGO), same row, same pool, same budget,
and take the target whose g is closest to each of GAIN_PERCENTILES
(75th, 90th, 95th, 99th) of g in that bucket.

Each example: two split swatches, target on the left half and the returned
stack's color on the right, for INDIGO and for random proposals, each run's
DeltaE00 under its swatch, and INDIGO's chosen stack drawn to scale
(substrate at the bottom), layers colored by material type as in Fig. 2a.
A dagger marks a target outside the sRGB gamut: its swatch is clipped for
display, so the printed DeltaE00, not the swatch, is the comparison.

    python scripts/paper_figures/fig6_examples.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import _style as S
import _runs as R

GAIN_PERCENTILES = (75, 90, 95, 99)
COLUMNS = tuple((p, f"{p}th percentile\nof gain") for p in GAIN_PERCENTILES)
ROW_BUCKETS = ("8<=C<30", "C>=30")
MODEL, BASE = "INDIGO", "random"
TYPE_COL = {"real": "#4D4D4D", "lorentz": S.OI["purple"], "perturb": S.OI["orange"],
            "interp": S.OI["green"]}
SUBSTRATE = "#BDBDBD"

LABELS = {
    "model": "INDIGO",
    "base": "Random",
    "row_title": "{b}",
    "de": "{v:.2f}",
    "fail": "failed",
    "layers": "{n} layers, {t:.0f} nm",
    "stack": "INDIGO\nstack",
    "real": "Real material",
    "lorentz": "Lorentz oscillator",
    "perturb": "Perturbed real",
    "interp": "Blend of two reals",
    "substrate": "Substrate",
    "gamut": "† target outside sRGB; swatch clipped",
    "split": "Each square: target (left) | result (right)",
    "ylabel": r"$\Delta E_{00}$",
}


def mat_type(name: str) -> str:
    if name.startswith("lorentz"):
        return "lorentz"
    if name.startswith("perturb"):
        return "perturb"
    if name.startswith("interp"):
        return "interp"
    return "real"


def physical(lab) -> bool:
    return 0 <= lab[0] <= 100 and float(np.hypot(lab[1], lab[2])) <= 150


def pick(dm, db):
    """{(bucket, column key): position in the 1000-row arrays}, plus pool sizes."""
    gain = db["de"] - dm["de"]
    phys = np.array([physical(r["target_lab"]) for r in dm["raw"]])
    out, pools = {}, {}
    for b in ROW_BUCKETS:
        ii = np.where((dm["bucket"] == b) & np.isfinite(gain) & phys)[0]
        g = gain[ii]
        pools[b] = len(ii)
        for p in GAIN_PERCENTILES:
            out[(b, p)] = int(ii[np.argmin(np.abs(g - np.percentile(g, p)))])
    return out, pools


def main():
    S.apply()
    h = R.headline()
    dm, db = R.load(MODEL, h), R.load(BASE, h)
    for k, d in ((MODEL, dm), (BASE, db)):
        R.check_against_headline(k, d, h)
    sel, pools = pick(dm, db)

    W, H = 17.0, 8.0
    fig = plt.figure(figsize=(W * S.CM, H * S.CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x0, cw = 1.85, 3.83            # first cell x, cell width
    y_top, rh = 6.75, 2.75          # first row top, row height
    sw = 1.05                       # split swatch: target left half, result right half
    max_t = max(sum(l["thickness_nm"] for l in (dm["raw"][i] or {}).get("layers") or [])
                for i in sel.values())
    scale = 1.75 / max_t            # cm per nm, tallest stack 1.75 cm
    notes = []
    any_out = False

    for c, (_, title) in enumerate(COLUMNS):
        ax.text(x0 + c * cw + 1.45, H - 0.15, title, ha="center", va="top", fontweight="bold",
                linespacing=1.0)
    for r, b in enumerate(ROW_BUCKETS):
        yt = y_top - r * rh
        n_b = pools[b]
        ax.text(0.15, yt - 0.85, f"{R.BUCKET_LABEL[b]}\n(n = {n_b})", ha="left", va="center",
                fontweight="bold", linespacing=1.2)
        for c, (ck, _) in enumerate(COLUMNS):
            i = sel[(b, ck)]
            rm, rb = dm["raw"][i], db["raw"][i]
            xc = x0 + c * cw
            tgt = rm["target_lab"]
            out = not S.lab_in_srgb(tgt)
            any_out = any_out or out
            pairs = [(LABELS["model"], (rm or {}).get("achieved_lab"), dm["de"][i]),
                     (LABELS["base"], (rb or {}).get("achieved_lab"), db["de"][i])]
            for j, (name, lab, de) in enumerate(pairs):
                xs = xc + j * (sw + 0.2)
                ys = yt - 0.3 - sw
                ax.text(xs + sw / 2, yt - 0.22, name, ha="center", va="bottom",
                        fontsize=S.SMALL_PT - 0.5)
                ax.add_patch(Rectangle((xs, ys), sw / 2, sw, fc=S.lab_to_srgb(tgt), ec="none"))
                if lab is None:
                    ax.add_patch(Rectangle((xs + sw / 2, ys), sw / 2, sw, fc="white", ec="none",
                                           hatch="////"))
                else:
                    ax.add_patch(Rectangle((xs + sw / 2, ys), sw / 2, sw,
                                           fc=S.lab_to_srgb(lab), ec="none"))
                ax.add_patch(Rectangle((xs, ys), sw, sw, fc="none", ec=S.INK, lw=0.4))
                txt = LABELS["de"].format(v=de) if np.isfinite(de) else LABELS["fail"]
                ax.text(xs + sw / 2, ys - 0.08, txt + ("†" if out and j == 0 else ""),
                        ha="center", va="top", fontsize=S.SMALL_PT,
                        fontweight="bold" if j == 0 else "normal")
            # INDIGO stack to scale
            layers = rm.get("layers") or []
            sx, sy, sww = xc + 2 * sw + 0.2 + 0.3, yt - 2.2, 0.36
            ax.add_patch(Rectangle((sx, sy - 0.12), sww, 0.12, fc=SUBSTRATE, ec="none"))
            yy = sy
            for L in layers:
                hgt = L["thickness_nm"] * scale
                ax.add_patch(Rectangle((sx, yy), sww, hgt, fc=TYPE_COL[mat_type(L["material"])],
                                       ec="white", lw=0.3))
                yy += hgt
            ax.text(sx + sww / 2, yt - 0.22, LABELS["stack"], ha="center", va="bottom",
                    fontsize=S.SMALL_PT - 1.5, color="#404040", linespacing=0.95)
            tot = sum(L["thickness_nm"] for L in layers)
            ax.text(sx + sww + 0.06, sy, LABELS["layers"].format(n=len(layers), t=tot),
                    rotation=90, ha="left", va="bottom", fontsize=S.SMALL_PT - 1.5, color="#404040")
            notes.append(f"{b} p{ck}: tier {dm['tier'][i]} row {dm['idx'][i]} "
                         f"(INDIGO {dm['de'][i]:.2f}, random {db['de'][i]:.2f})")
    # DeltaE label once, left of the first row's numbers
    ax.text(x0 - 0.12, y_top - 0.3 - sw - 0.08, LABELS["ylabel"], ha="right", va="top",
            fontsize=S.SMALL_PT)

    # legend
    lx, ly = 0.15, 0.3
    for t in ("real", "lorentz", "perturb", "interp", "substrate"):
        col = SUBSTRATE if t == "substrate" else TYPE_COL[t]
        ax.add_patch(Rectangle((lx, ly - 0.08), 0.28, 0.2, fc=col, ec="none"))
        ax.text(lx + 0.36, ly + 0.02, LABELS[t], va="center", fontsize=S.SMALL_PT - 0.5)
        lx += 0.5 + 0.12 * len(LABELS[t])
    ax.text(0.15, ly + 0.5, LABELS["split"], ha="left", va="center",
            fontsize=S.SMALL_PT - 0.5, color="#404040")
    if any_out:
        ax.text(W - 0.15, ly + 0.5, LABELS["gamut"], ha="right", va="center",
                fontsize=S.SMALL_PT - 0.5, color="#404040")

    S.save(fig, "fig6_examples")
    S.write_provenance(
        "Fig 6", "`scripts/paper_figures/fig6_examples.py` -> `fig6_examples.{pdf,png}`. "
        "Rows `results/paper_runs/strong_model` (INDIGO, N=250) and `strong_random` (random, "
        "N=250), tiers A+B pooled, loaded with paper_numbers.load_run. Fields: target_lab, "
        "achieved_lab, delta_e, layers[].{material, thickness_nm}. Selection: per bucket "
        "(8<=C<30, C>=30), physical targets (L* 0-100, C* <= 150) where both runs returned a stack "
        f"(n = {pools['8<=C<30']}, {pools['C>=30']}), gain g = delta_e(random) - delta_e(INDIGO); "
        f"rows nearest the {', '.join(map(str, GAIN_PERCENTILES))}th percentiles of g. Chosen: " + "; ".join(notes) + ". Swatches are Lab -> sRGB "
        "(D65), clipped; dagger = target outside sRGB. Material type from the layer name "
        "(lorentz_*, perturb_*, interp(...), else real).")


if __name__ == "__main__":
    main()
