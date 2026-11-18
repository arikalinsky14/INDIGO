#!/usr/bin/env python3
"""NEUTRON head-to-head on NEUTRON's own reported targets, scored in their physics.

(a) Table 2 pictures. For each picture: the three numbers NEUTRON reports
    (MDN only, PSO random init, NEUTRON; image-mean DeltaE00) as markers, and
    INDIGO in two arms: "fixed" uses the picture's own material set (their
    task exactly), "any" lets INDIGO choose from the pool. For INDIGO, every
    one of the 150 sampled colors is a dot (colors are drawn with probability
    proportional to pixel count, so their plain mean estimates the image
    mean), and the mean with its 95% interval is drawn over them. Log y axis,
    line at DeltaE00 = 2.
(b) Table 1, the chrome designs I-V: NEUTRON's reported DeltaE00 against
    INDIGO's on the same material combination (fixed arm).

Metric: dE_theirs_int_snap, their simulator with whole-nm thicknesses chosen
by rounding against the target, as their paper states ("we constrain the
thicknesses of all layers to be integers"). Means, never medians, because
NEUTRON reports means.

Rows that crashed (no result) are drawn on a "failed" band and EXCLUDED from
the plotted mean, which is then hollow and listed in the provenance line; set
CRASHED_COUNT_AS_FAILURE = True once they are rerun to make that an error.

    python scripts/paper_figures/fig_neutron.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style as S

FILES = {"fixed": S.REPO / "results/neutron/rescore_final_fixed.json",
         "any": S.REPO / "results/neutron/rescore_final_full.json"}
METRIC = "dE_theirs_int_snap"
CRASHED_COUNT_AS_FAILURE = False
FAIL_Y = 60.0
ORDER = ["White Orchard", "Tulip Field", "Great Wall", "Flower (style-transferred)"]

LABELS = {
    "mdn_only": "MDN only (reported)",
    "pso_random_init": "PSO, random init (reported)",
    "neutron": "NEUTRON (reported)",
    "fixed": "INDIGO, their materials",
    "any": "INDIGO, any material",
    "y": r"Image-mean $\Delta E_{00}$ (their physics)",
    "b_y": r"$\Delta E_{00}$ (their physics)",
    "b_x": "Chrome design (Table 1)",
    "fail": "failed",
    "Flower (style-transferred)": "Flower\n(style-transferred)",
}
REPORTED_STYLE = {"mdn_only": ("s", "#BDBDBD"), "pso_random_init": ("D", "#7F7F7F"),
                  "neutron": ("o", S.INK)}
ARM_COL = {"fixed": S.OI["blue"], "any": S.OI["orange"]}


def load():
    d = {k: json.load(open(p)) for k, p in FILES.items()}
    reported = d["fixed"]["reported"]
    if d["any"]["reported"] != reported:
        sys.exit("the two files disagree on NEUTRON's reported numbers")
    pics = {}
    for arm, j in d.items():
        for s, v in j["by_set"].items():
            if not s.startswith("picture:"):
                continue
            name = next(n for n, t in reported["pictures_table2"].items()
                        if t == v["reported_table2"])
            vals = np.array([np.inf if r.get(METRIC) is None else r[METRIC]
                             for r in j["rows"] if r["set"] == s], float)
            pics.setdefault(name, {})[arm] = dict(vals=vals, stat=v[METRIC], n=v["n"])
    cr = {}
    for s, v in d["fixed"]["by_set"].items():
        if s.startswith("cr_table1:"):
            k = s.split(":")[1]
            cr[k] = dict(indigo=v[METRIC]["mean"], reported=reported["cr_table1"][k])
    return reported, pics, cr


def main():
    S.apply()
    reported, pics, cr = load()
    missing = [n for n in reported["pictures_table2"] if n not in pics]
    order = [n for n in ORDER if n in pics]

    fig = plt.figure(figsize=(S.DOUBLE, 7.0 * S.CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], left=0.075, right=0.985, bottom=0.2,
                          top=0.8, wspace=0.28)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    off = {"mdn_only": -0.3, "pso_random_init": -0.18, "neutron": -0.06, "fixed": 0.1,
           "any": 0.3}
    notes = []
    for i, name in enumerate(order):
        t = reported["pictures_table2"][name]
        for k, (mk, col) in REPORTED_STYLE.items():
            ax.plot(i + off[k], t[k], mk, ms=4.5, color=col, mec=S.INK, mew=0.4, zorder=4)
        for arm in ("fixed", "any"):
            if arm not in pics[name]:
                continue
            v = pics[name][arm]["vals"]
            fin = np.isfinite(v)
            n_fail = int((~fin).sum())
            x = i + off[arm] + rng.uniform(-0.055, 0.055, v.size)
            y = np.where(fin, np.clip(v, 0.01, None), FAIL_Y)
            ax.scatter(x, y, s=1.6, lw=0, alpha=0.45, color=ARM_COL[arm], zorder=2,
                       rasterized=True)
            if n_fail and not CRASHED_COUNT_AS_FAILURE:
                m = float(np.mean(v[fin]))
                se = float(np.std(v[fin], ddof=1) / np.sqrt(fin.sum()))
                lo, hi = m - 1.96 * se, m + 1.96 * se
                hollow = True
                notes.append(f"{name} {arm}: {n_fail} crashed row(s) excluded, mean of "
                             f"{int(fin.sum())} = {m:.2f} [{lo:.2f}, {hi:.2f}] (normal approx.)")
            else:
                st = pics[name][arm]["stat"]
                m = st["image_mean_estimate"]["est"]
                lo, hi = st["image_mean_estimate"]["lo"], st["image_mean_estimate"]["hi"]
                hollow = False
            ax.errorbar(i + off[arm], m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4.8,
                        color=ARM_COL[arm], mfc="white" if hollow else ARM_COL[arm],
                        mec=S.INK if not hollow else ARM_COL[arm], mew=0.6 if not hollow else 1.2,
                        elinewidth=0.9, capsize=0, zorder=5)
    ax.axhline(2, color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhspan(FAIL_Y / 1.6, FAIL_Y * 1.6, color="#F2F2F2", lw=0, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.01, FAIL_Y * 1.6)
    ax.set_yticks([0.01, 0.1, 1, 10, FAIL_Y])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", LABELS["fail"]])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(n, n) for n in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel(LABELS["y"])

    # (b) chrome Table 1
    ks = sorted(cr)
    xb = np.arange(len(ks))
    bx.plot(xb - 0.12, [cr[k]["reported"] for k in ks], "o", ms=4.5, color=S.INK, mec=S.INK,
            mew=0.4, zorder=3)
    bx.plot(xb + 0.12, [cr[k]["indigo"] for k in ks], "o", ms=4.5, color=ARM_COL["fixed"],
            mec=S.INK, mew=0.4, zorder=3)
    bx.set_xticks(xb); bx.set_xticklabels(ks)
    bx.set_xlim(-0.6, len(ks) - 0.4)
    bx.set_ylim(0, 2.1)
    bx.axhline(2, color=S.GREY, lw=0.6, ls=(0, (3, 2)))
    bx.set_ylabel(LABELS["b_y"]); bx.set_xlabel(LABELS["b_x"])

    handles = [Line2D([], [], marker=REPORTED_STYLE[k][0], color=REPORTED_STYLE[k][1], mec=S.INK,
                      mew=0.4, ls="none", ms=4.5, label=LABELS[k]) for k in REPORTED_STYLE]
    handles += [Line2D([], [], marker="o", color=ARM_COL[a], mec=S.INK, mew=0.5, ls="none",
                       ms=4.8, label=LABELS[a])
                for a in ("fixed", "any")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 1.0), ncol=3,
               handletextpad=0.3, columnspacing=1.2, labelspacing=0.35)
    for a_, t, x in ((ax, "a", -0.085), (bx, "b", -0.3)):
        a_.text(x, 1.02, t, transform=a_.transAxes, fontweight="bold",
                fontsize=S.FONT_PT + 1, va="bottom")

    S.save(fig, "fig_neutron")
    S.write_provenance(
        "Fig NEUTRON", "`scripts/paper_figures/fig_neutron.py` -> `fig_neutron.{pdf,png}`. "
        "`results/neutron/rescore_final_fixed.json` (fixed: picture's own materials) and "
        "`rescore_final_full.json` (any material): rows[].{set, dE_theirs_int_snap}; by_set."
        "picture:*.dE_theirs_int_snap.image_mean_estimate {est, lo, hi} (colors sampled in proportion "
        "to pixels, so the mean estimates the image mean); by_set.cr_table1:{I..V}.dE_theirs_int_snap."
        "mean. NEUTRON's numbers: reported.{pictures_table2, cr_table1} (their Tables 2 and 1). "
        f"Pictures shown: {', '.join(order)}; in their Table 2 but not run: {', '.join(missing) or 'none'}. "
        + ("; ".join(notes) + "." if notes else "No crashed rows."))
    print(notes, missing)


if __name__ == "__main__":
    main()
