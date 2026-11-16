#!/usr/bin/env python3
"""Fig. 2: material and target diversity.

(a) n and k against wavelength on CANONICAL_LAMBDA_NM: three real
    JaxLayerLumos materials and one draw from each synthetic generator in
    src/synthetic_materials.py. n solid, k dashed. Cut down from the 2x4
    diagnostic in scripts/visualize_synthetic.py (same loader, same
    held-in seed pool, fixed seed).
(b, c) a*b* of the test targets, tier A and tier B as separate panels,
    colored by structure_source. Reads ONLY target metadata (gt_lab,
    structure_source) from results/prod_eval/eval_tier_{a,b}_greedy.json,
    never the model-dependent ciede2000 field. Circles at C* = 8 and 30
    (the paper's chroma buckets).

Targets whose simulated color is not a physical CIELAB color (L* outside
[0, 100] or C* > 150; e.g. a* = -837) are left out and counted in the
provenance line. They are a data-generation issue to chase separately.

    python scripts/figures/fig2_diversity.py [--jll-materials-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style as S
from src.material_features import CANONICAL_LAMBDA_NM, load_jll_directory
from src.synthetic_materials import interpolate_real, parametric_lorentz, perturb_real

TIERS = {"A": "results/prod_eval/eval_tier_a_greedy.json",
         "B": "results/prod_eval/eval_tier_b_greedy.json"}
SEED_LORENTZ = 23   # each generator gets its own fixed RNG so draws are reproducible
SEED_PERTURB = 3
REAL = ["Ag", "TiO2", "aSi"]          # metal, transparent dielectric, absorber
CHROMA_RINGS = (8, 30)
AB_LIM = 130                           # max physical C* in the files is 127.4

LABELS = {
    "x_nk": "Wavelength (nm)",
    "y_nk": "n, k",
    "real": "Real",
    "synth": "Synthetic",
    "n": "n",
    "k": "k",
    "lorentz": "Lorentz oscillator",
    "interp": "Interpolated ({a}–{b})",
    "perturb": "Perturbed ({a})",
    "a": r"$a^*$",
    "b": r"$b^*$",
    "random": "Random stacks",
    "high_chroma_search": "High-chroma search",
    "tier": "Tier {t}",
    "ring": r"$C^*$ = {c}",
}
COL_REAL = [S.OI["black"], S.OI["green"], S.OI["sky"]]
COL_SYN = [S.OI["vermilion"], S.OI["orange"], S.OI["purple"]]
COL_SRC = {"random": S.GREY, "high_chroma_search": S.OI["blue"]}


def find_jll(arg):
    cands = [Path(arg)] if arg else []
    try:
        import jaxlayerlumos
        cands.append(Path(jaxlayerlumos.__file__).parent / "materials")
    except ImportError:
        pass
    cands += [Path("./jaxlayerlumos/materials")]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(f"no JaxLayerLumos materials dir in {cands}; pass --jll-materials-dir")


def materials(jll_dir):
    pool = load_jll_directory(jll_dir)
    reals = [pool[n] for n in REAL]
    lor = parametric_lorentz(np.random.default_rng(SEED_LORENTZ), metallic_prob=0.0)
    interp = interpolate_real(pool["TiO2"], pool["aSi"], weight=0.5)
    pert = perturb_real(pool["TiO2"], np.random.default_rng(SEED_PERTURB), magnitude="large")
    syn = [(LABELS["lorentz"], lor),
           (LABELS["interp"].format(a="TiO$_2$", b="aSi"), interp),
           (LABELS["perturb"].format(a="TiO$_2$"), pert)]
    return reals, syn


def load_targets(path):
    rows = json.load(open(S.REPO / path))["results"]
    lab = np.array([r["gt_lab"] for r in rows], float)
    src = np.array([r["structure_source"] for r in rows])
    c = np.hypot(lab[:, 1], lab[:, 2])
    ok = (lab[:, 0] >= 0) & (lab[:, 0] <= 100) & (c <= 150)
    return lab, src, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jll-materials-dir", default=None)
    a = ap.parse_args()
    S.apply()
    reals, syn = materials(find_jll(a.jll_materials_dir))
    lam = CANONICAL_LAMBDA_NM

    fig = plt.figure(figsize=(S.DOUBLE, 6.2 * S.CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], left=0.06, right=0.99,
                          bottom=0.15, top=0.96, wspace=0.32)
    ax = fig.add_subplot(gs[0])
    names = {"TiO2": "TiO$_2$"}
    handles = [Line2D([], [], color="none", label=LABELS["real"])]
    for m, c in zip(reals, COL_REAL):
        ax.plot(lam, m.n, "-", color=c, lw=1.0)
        ax.plot(lam, m.k, "--", color=c, lw=1.0)
        handles.append(Line2D([], [], color=c, lw=1.0, label=names.get(m.name, m.name)))
    handles.append(Line2D([], [], color="none", label=LABELS["synth"]))
    for (lbl, m), c in zip(syn, COL_SYN):
        ax.plot(lam, m.n, "-", color=c, lw=1.0)
        ax.plot(lam, m.k, "--", color=c, lw=1.0)
        handles.append(Line2D([], [], color=c, lw=1.0, label=lbl))
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(LABELS["x_nk"]); ax.set_ylabel(LABELS["y_nk"])
    leg = ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.4,
                    columnspacing=0.8, labelspacing=0.25, borderaxespad=0.1,
                    fontsize=S.SMALL_PT - 0.5)
    for t in leg.get_texts():
        if t.get_text() in (LABELS["real"], LABELS["synth"]):
            t.set_fontweight("bold")
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color=S.INK, ls="-", label=LABELS["n"]),
                       Line2D([], [], color=S.INK, ls="--", label=LABELS["k"])],
              loc="upper left", ncol=2, handlelength=1.8, borderaxespad=0.1,
              columnspacing=1.0, bbox_to_anchor=(0.0, 0.73), fontsize=S.SMALL_PT - 0.5)
    ymax = max(max(m.n.max(), m.k.max()) for m in reals + [s[1] for s in syn])
    ax.set_ylim(0, ymax * 1.42)

    counts = {}
    th = np.linspace(0, 2 * np.pi, 361)
    for i, (t, path) in enumerate(TIERS.items()):
        bx = fig.add_subplot(gs[i + 1])
        lab, src, ok = load_targets(path)
        counts[t] = dict(total=len(lab), dropped=int((~ok).sum()),
                         **{s: int(((src == s) & ok).sum()) for s in COL_SRC},
                         hc_share_all=float((src == "high_chroma_search").mean()))
        for s in ("random", "high_chroma_search"):
            m = (src == s) & ok
            bx.scatter(lab[m, 1], lab[m, 2], s=1.6, lw=0, alpha=0.55,
                       color=COL_SRC[s], rasterized=True,
                       label=f"{LABELS[s]} (n = {m.sum()})")
        for c in CHROMA_RINGS:
            bx.plot(c * np.cos(th), c * np.sin(th), color=S.INK, lw=0.6, ls=(0, (2, 1.5)))
            ang = np.deg2rad(40)
            bx.text((c + 2) * np.cos(ang), (c + 2) * np.sin(ang), str(c),
                    fontsize=S.SMALL_PT - 1, color=S.INK, ha="left", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        bx.axhline(0, color=S.LIGHT, lw=0.5, zorder=0)
        bx.axvline(0, color=S.LIGHT, lw=0.5, zorder=0)
        bx.set_xlim(-AB_LIM, AB_LIM); bx.set_ylim(-AB_LIM, AB_LIM)
        bx.set_aspect("equal")
        bx.set_xticks([-100, -50, 0, 50, 100]); bx.set_yticks([-100, -50, 0, 50, 100])
        bx.set_xlabel(LABELS["a"]); bx.set_ylabel(LABELS["b"], labelpad=0)
        bx.text(0.03, 0.97, LABELS["tier"].format(t=t), transform=bx.transAxes,
                va="top", fontweight="bold")
        bx.plot([], [], color=S.INK, lw=0.6, ls=(0, (2, 1.5)),
                label=LABELS["ring"].format(c=" and ".join(map(str, CHROMA_RINGS))))
        lg = bx.legend(loc="lower left", markerscale=4, handletextpad=0.1,
                       borderaxespad=0.1, labelspacing=0.2, fontsize=S.SMALL_PT - 1)
        for h in lg.legend_handles:
            h.set_alpha(1)

    for i, ax_ in enumerate(fig.axes):
        ax_.text(-0.2 if i else -0.14, 1.0, "abc"[i], transform=ax_.transAxes,
                 fontweight="bold", fontsize=S.FONT_PT + 1, va="top")

    S.save(fig, "fig2_diversity")
    cA, cB = counts["A"], counts["B"]
    S.write_provenance(
        "Fig 2", "`scripts/figures/fig2_diversity.py` -> `fig2_diversity.{pdf,png}`. "
        "(a) JaxLayerLumos 0.3.3 CSVs (Ag, TiO2, aSi) via src.material_features."
        "load_jll_directory + one draw each of parametric_lorentz (seed 23, dielectric), "
        "interpolate_real(TiO2, aSi, w=0.5), perturb_real(TiO2, large, seed 3), on CANONICAL_LAMBDA_NM "
        "(300-900 nm). (b, c) `results/prod_eval/eval_tier_{a,b}_greedy.json` results[]."
        "{gt_lab, structure_source} only; tiers are SEPARATE PANELS (b = tier A, c = tier B), "
        f"color = source. Tier A n = {cA['total'] - cA['dropped']} plotted of {cA['total']} "
        f"(random {cA['random']}, high-chroma {cA['high_chroma_search']}); tier B n = "
        f"{cB['total'] - cB['dropped']} of {cB['total']} (random {cB['random']}, high-chroma "
        f"{cB['high_chroma_search']}). Dropped as non-physical Lab (L* outside [0,100] or "
        f"C* > 150): {cA['dropped']} (A), {cB['dropped']} (B). High-chroma share of all "
        f"targets: {100 * cA['hc_share_all']:.1f}% (A), {100 * cB['hc_share_all']:.1f}% (B). "
        "Rings at C* = 8 and 30.")
    print(counts)


if __name__ == "__main__":
    main()
