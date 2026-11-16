#!/usr/bin/env python3
"""Fig. 2: material and target diversity.

(a) Material coverage: n against k at the canonical wavelength nearest
    550 nm. A cloud of synthetic materials drawn exactly as training pools
    draw them (create_dataset/src/pool_sampler.py: _sample_synthetic with
    PoolSamplerConfig.synthetic_weights, seeded from the held-in reals of
    split_jll_real), the held-in real materials (in training pools), and the
    held-out reals (HELD_OUT_REAL_MATERIALS, tier B only). The point: the
    synthetic distribution surrounds the real ones, and the held-out
    materials fall inside what training covered.
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

import _style as S
from src.material_features import CANONICAL_LAMBDA_NM, load_jll_directory
from create_dataset.src.pool_sampler import (
    PoolSamplerConfig, _sample_synthetic, split_jll_real,
)

TIERS = {"A": "results/prod_eval/eval_tier_a_greedy.json",
         "B": "results/prod_eval/eval_tier_b_greedy.json"}
SEED_CLOUD = 0
N_CLOUD = 6000
LAMBDA_NM = 550                        # nearest canonical wavelength is used
IDX = int(np.argmin(np.abs(CANONICAL_LAMBDA_NM - LAMBDA_NM)))
GEN_OF = {"perturb_small": "perturb", "perturb_large": "perturb",
          "interpolate_real": "interp", "parametric_lorentz": "lorentz"}
GEN_GROUPS = ["lorentz", "perturb", "interp"]
# Reals labeled on the plot (all are drawn; labels kept sparse to stay legible)
LABEL_IN = ["Ag", "Al", "SiO2", "TiO2", "aSi", "cSi", "Cr"]
# label offsets in points (dx, dy, ha) where the default would collide
LABEL_POS = {"Al": (-3, 2, "right"), "Al2O3": (4, -6, "left"), "Au": (4, -6, "left"),
             "SiO2": (-1, 4, "right"), "ITO": (2, 4, "left"), "TiO2": (2, 4, "left"),
             "Ag": (2, 3, "left")}
K_LINTHRESH = 0.05
CHROMA_RINGS = (8, 30)
AB_LIM = 130                           # max physical C* in the files is 127.4

LABELS = {
    "x_nk": "Refractive index, n",
    "y_nk": "Extinction coefficient, k",
    "wl": "at {wl:.0f} nm",
    "held_in": "Real, in training pools",
    "held_out": "Real, held out (tier B)",
    "lorentz": "Synthetic: Lorentz oscillator",
    "perturb": "Synthetic: perturbed real",
    "interp": "Synthetic: blend of two reals",
    "a": r"$a^*$",
    "b": r"$b^*$",
    "random": "Random stacks",
    "high_chroma_search": "High-chroma search",
    "tier": "Tier {t}",
    "ring": r"$C^*$ = {c}",
}
COL_GEN = {"lorentz": S.OI["purple"], "perturb": S.OI["orange"], "interp": S.OI["green"]}
PRETTY = {"SiO2": "SiO$_2$", "TiO2": "TiO$_2$", "Al2O3": "Al$_2$O$_3$*"}
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


def coverage(jll_dir):
    """Synthetic draws as training draws them, plus held-in / held-out reals."""
    pool = load_jll_directory(jll_dir)
    held_in, held_out = split_jll_real(pool)
    rng = np.random.default_rng(SEED_CLOUD)
    w = PoolSamplerConfig().synthetic_weights
    methods = ["perturb_small", "perturb_large", "interpolate_real", "parametric_lorentz"]
    p = np.asarray(w, float) / np.sum(w)
    pts = {g: [] for g in GEN_GROUPS}
    for _ in range(N_CLOUD):
        # Mirrors pool_sampler._sample_synthetic draw for draw: same method
        # choice, then the same generator call, so the cloud has the
        # training mix.
        state = rng.bit_generator.state
        method = methods[int(rng.choice(4, p=p))]
        rng.bit_generator.state = state
        m = _sample_synthetic(rng, held_in, w)
        g = GEN_OF[method]
        pts[g].append((m.n[IDX], m.k[IDX]))
    return held_in, held_out, {g: np.array(v) for g, v in pts.items()}


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
    held_in, held_out, cloud = coverage(find_jll(a.jll_materials_dir))

    fig = plt.figure(figsize=(S.DOUBLE, 7.2 * S.CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], left=0.06, right=0.99,
                          bottom=0.13, top=0.80, wspace=0.32)
    ax = fig.add_subplot(gs[0])
    for g in GEN_GROUPS:
        xy = cloud[g]
        ax.scatter(xy[:, 0], xy[:, 1], s=1.2, lw=0, alpha=0.35, color=COL_GEN[g],
                   rasterized=True, label=f"{LABELS[g]} (n = {len(xy)})", zorder=2)
    nin = np.array([(m.n[IDX], m.k[IDX]) for m in held_in])
    nout = np.array([(m.n[IDX], m.k[IDX]) for m in held_out])
    ax.scatter(nin[:, 0], nin[:, 1], s=9, color=S.INK, lw=0, zorder=4,
               label=f"{LABELS['held_in']} (n = {len(held_in)})")
    ax.scatter(nout[:, 0], nout[:, 1], s=13, facecolor="white", edgecolor=S.INK, lw=0.8,
               zorder=5, label=f"{LABELS['held_out']} (n = {len(held_out)})")
    ann = [(m, True) for m in held_in if m.name in LABEL_IN] + [(m, False) for m in held_out]
    for m, _ in ann:
        x, y = m.n[IDX], m.k[IDX]
        dx, dy, ha = LABEL_POS.get(m.name, (3, 1.5, "left"))
        ax.annotate(PRETTY.get(m.name, m.name), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=S.SMALL_PT - 1.5, color=S.INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.02", fc="white", ec="none", alpha=0.7))
    ax.set_yscale("symlog", linthresh=K_LINTHRESH, linscale=0.6)
    ax.set_ylim(-0.005, 12)
    ax.set_xlim(0, 5.8)
    ax.set_yticks([0, 0.1, 1, 10]); ax.set_yticklabels(["0", "0.1", "1", "10"])
    ax.set_xlabel(LABELS["x_nk"]); ax.set_ylabel(LABELS["y_nk"])
    ax.text(0.98, 0.02, LABELS["wl"].format(wl=CANONICAL_LAMBDA_NM[IDX]), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=S.SMALL_PT - 1, color="#404040")
    lg = ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=1,
                   handletextpad=0.1, labelspacing=0.12, borderaxespad=0,
                   fontsize=S.SMALL_PT - 1)
    for h in lg.legend_handles:
        h.set_alpha(1)
        h.set_sizes([10])

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
        ax_.text(-0.2 if i else -0.17, 1.0, "abc"[i], transform=ax_.transAxes,
                 fontweight="bold", fontsize=S.FONT_PT + 1, va="top")

    S.save(fig, "fig2_diversity")
    al2o3 = next((m for m in held_out if m.name == "Al2O3"), None)
    cA, cB = counts["A"], counts["B"]
    S.write_provenance(
        "Fig 2", "`scripts/figures/fig2_diversity.py` -> `fig2_diversity.{pdf,png}`. "
        f"(a) n, k at CANONICAL_LAMBDA_NM[{IDX}] = {CANONICAL_LAMBDA_NM[IDX]:.1f} nm. Reals: "
        "JaxLayerLumos 0.3.3 CSVs via load_jll_directory, split by create_dataset/src/pool_sampler.py "
        f"split_jll_real ({len(held_in)} held-in, {len(held_out)} held-out = HELD_OUT_REAL_MATERIALS). "
        f"Cloud: {N_CLOUD} draws of pool_sampler._sample_synthetic with PoolSamplerConfig."
        f"synthetic_weights {tuple(PoolSamplerConfig().synthetic_weights)} on the held-in reals, "
        f"seed {SEED_CLOUD} ({', '.join(f'{g} {len(cloud[g])}' for g in GEN_GROUPS)}). "
        + (f"Al2O3* plotted as loaded (n = {al2o3.n[IDX]:.2f}, k = {al2o3.k[IDX]:.2f}): the upstream "
           "Al2O3-Querry-1985.csv holds metal-like values in the visible. " if al2o3 is not None else "")
        + "(b, c) `results/prod_eval/eval_tier_{a,b}_greedy.json` results[]."
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
