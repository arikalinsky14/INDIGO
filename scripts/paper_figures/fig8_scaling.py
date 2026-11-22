#!/usr/bin/env python3
"""Fig. 8 (Sec. 4.8, "Why search instead of a bigger model").

(a) Pretraining buys little. Greedy median DeltaE00 against pretraining
    budget: the six IsoFLOP budgets (best model and learning rate at each)
    and the production model at its best greedy learning rate. The budget
    whose N* lies on the edge of its curve is unusable for the N* fit, not
    as a loss measurement, so it is drawn (open) rather than dropped.
(b) Search buys a lot. The shipped production model, greedy and then under
    four search settings, tiers A and B. The x axis is ORDERED CATEGORIES,
    not candidate count: refinement changes together with N, so nothing is
    interpolated between positions.
    (a) and (b) share one log y axis and the line at DeltaE00 = 2, so the two
    can be compared by eye. They are separate axes on purpose: (a) is many
    models, (b) is one.
    Datasets differ, as marginal_returns.json's `comparability` records:
    greedy values are on each run's 2,048-row validation slice; search values
    are test rows 0-499. The labels say which is which.
(c) The law that set the production configuration, on validation
    cross-entropy (by_metric["ce"], all six budgets usable). Top: Akima
    IsoFLOP curves with each budget's minimum. Bottom: N* against compute
    with the fitted power law and its 95% band; the star is the production
    configuration, which the law (extrapolated to its budget) predicts.

Inputs, every number read, nothing typed in:
  results/paper/marginal_returns.json
      comparability; study[].{budget_flops, greedy_dE_best, usable};
      production.{budget_flops, n_params, examples, greedy_dE_best_lr.value,
      greedy_dE_shipped_lr.value}; search.<setting>.{a, b}.median
  analyses/scaling/results/tuned/porian_fit_tuned.json
      by_metric.ce.{n_usable, laws.N_star, laws.D_star}
  analyses/scaling/results/tuned/isoflop_tuned.json
      runs[].{n_params, flops, passes, val_loss}: the curves, recomputed with
      src.scaling.porian as plot_porian.py does; stops if the recomputed
      exponent disagrees with the JSON.

    python scripts/paper_figures/fig8_scaling.py
"""
from __future__ import annotations

import collections
import json
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import _style as S
from src.scaling import porian as P

MARGINAL = S.REPO / "results/paper/marginal_returns.json"
FIT = S.REPO / "analyses/scaling/results/tuned/porian_fit_tuned.json"
TUNED = S.REPO / "analyses/scaling/results/tuned/isoflop_tuned.json"
METRIC_C = "ce"
DE_REF = 2.0
SETTING_RE = re.compile(r"(\d+)\s+candidates,\s*top-(\d+),\s*(\d+)\s+steps")

LABELS = {
    "a_x": "Pretraining compute, C (FLOP)",
    "y_de": r"Median $\Delta E_{00}$",
    "a_study": "Best model per budget (greedy)",
    "a_edge": "Budget with N* on its edge",
    "prod": "Production model",
    "b_x": "Search setting (shipped production model)",
    "b_greedy": "No search: greedy,\nvalidation slice",
    "b_tier": "Tier {t}, test rows",
    "b_setting": "{n} ×\ntop-{k}\n{s} steps",
    "none": "none",
    "ref": r"$\Delta E_{00}$ = 2",
    "a_note": "greedy decoding, validation slices",
    "c_x_top": "Parameters, N",
    "c_y_top": "Val. cross-entropy",
    "c_x": "Pretraining compute, C (FLOP)",
    "c_y": r"Compute-optimal size, $N^*$",
    "c_law": r"$N^* \propto C^{{{e:.2f}}}$ [{lo:.2f}, {hi:.2f}]",
}
RAMP = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
PROD_C = S.OI["vermilion"]
TIER_C = {"a": S.OI["blue"], "b": S.OI["orange"]}


def need(d, *keys, src=""):
    """d[k1][k2]..., or exit naming the missing key and file."""
    cur = d
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            sys.exit(f"[fig8] {src}: missing key {'.'.join(map(str, keys[:i + 1]))}")
        cur = cur[k]
    return cur


def load_json(path):
    if not path.is_file():
        sys.exit(f"[fig8] missing file {path.relative_to(S.REPO)}")
    return json.load(open(path))


def search_points(mr):
    """[(label, setting, tier_a_median, tier_b_median)] in increasing search effort."""
    src = MARGINAL.name
    out = []
    for key, v in need(mr, "search", src=src).items():
        m = SETTING_RE.search(key)
        if not m:
            sys.exit(f"[fig8] {src}: cannot parse search setting {key!r}")
        n, k, s = map(int, m.groups())
        out.append(((n, k, s), key, need(v, "a", "median", src=src),
                    need(v, "b", "median", src=src)))
    out.sort(key=lambda t: t[0])
    return out


def sci(x):
    m, e = f"{x:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return rf"$10^{{{int(e)}}}$" if m == "1" else rf"${m}\times10^{{{int(e)}}}$"


def ce_rungs():
    src = load_json(TUNED)
    by_b = collections.defaultdict(list)
    for r in src["runs"]:
        by_b[min(src["budgets"], key=lambda b: abs(b - r["flops"]))].append(r)
    clusters = collections.defaultdict(list)
    for r in src["runs"]:
        clusters[(round(r["flops"], -11), r["n_params"])].append(r["val_loss"])
    noise = P.NoiseModel.from_clusters(list(clusters.values()))
    return [P.fit_rung(b, [r["n_params"] for r in by_b[b]], [r["val_loss"] for r in by_b[b]],
                       d_vals=[r["passes"] for r in by_b[b]], noise=noise,
                       n_boot=P.BOOTSTRAP_ITERS, rng=np.random.default_rng(0))
            for b in sorted(by_b)]


def main():
    S.apply()
    mr, fit = load_json(MARGINAL), load_json(FIT)
    sm, sf = MARGINAL.name, FIT.name
    comp = need(mr, "comparability", "slice", src=sm)
    study = need(mr, "study", src=sm)
    prod = need(mr, "production", src=sm)
    p_c = need(prod, "budget_flops", src=sm)
    p_n, p_d = need(prod, "n_params", src=sm), need(prod, "examples", src=sm)
    p_best = need(prod, "greedy_dE_best_lr", "value", src=sm)
    p_ship = need(prod, "greedy_dE_shipped_lr", "value", src=sm)
    for e in study:
        for k in ("budget_flops", "greedy_dE_best", "usable"):
            need(e, k, src=f"{sm} study[]")
    searches = search_points(mr)
    law = need(fit, "by_metric", METRIC_C, "laws", "N_star", src=sf)
    dlaw = need(fit, "by_metric", METRIC_C, "laws", "D_star", src=sf)
    n_usable = need(fit, "by_metric", METRIC_C, "n_usable", src=sf)

    rungs = ce_rungs()
    good = [r for r in rungs if r.usable]
    if len(good) != n_usable:
        sys.exit(f"[fig8] recomputed {len(good)} usable CE budgets, {sf} says {n_usable}")
    xs = np.array([r.budget for r in good]); ns = np.array([r.n_star_median for r in good])
    sig = np.array([r.log_sigma for r in good])
    n_law = P.power_law_fit(xs, ns, sig)
    if abs(n_law.exponent - law["exponent"]) > 5e-3:
        sys.exit(f"[fig8] recomputed CE exponent {n_law.exponent:.4f} != {sf} {law['exponent']:.4f}")
    pred_n = law["coef"] * p_c ** law["exponent"]
    pred_d = dlaw["coef"] * p_c ** dlaw["exponent"]

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=(S.DOUBLE, 9.0 * S.CM))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], left=0.075, right=0.99,
                             bottom=0.2, top=0.95, wspace=0.32)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.06)
    ax = fig.add_subplot(left[0])
    bx = fig.add_subplot(left[1], sharey=ax)
    right = outer[1].subgridspec(2, 1, hspace=0.75)
    cx_top, cx = fig.add_subplot(right[0]), fig.add_subplot(right[1])

    all_de = ([e["greedy_dE_best"] for e in study] + [p_best, p_ship]
              + [t[2] for t in searches] + [t[3] for t in searches])
    ylo, yhi = min(all_de) / 1.6, max(all_de) * 1.6

    # (a) pretraining buys little
    for e in study:
        ax.plot(e["budget_flops"], e["greedy_dE_best"], "o", ms=4.2, zorder=3,
                color=S.INK if e["usable"] else "white", mec=S.INK, mew=0.8)
    ax.plot(p_c, p_best, "*", ms=8, color=PROD_C, mec=S.INK, mew=0.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(min(e["budget_flops"] for e in study) / 2.5, p_c * 3)
    ax.set_xlabel(LABELS["a_x"]); ax.set_ylabel(LABELS["y_de"])
    ax.text(0.04, 0.04, LABELS["a_note"], transform=ax.transAxes, fontsize=S.SMALL_PT - 0.5,
            color="#404040", va="bottom")
    hd = [Line2D([], [], marker="o", color=S.INK, ls="none", ms=4, label=LABELS["a_study"]),
          Line2D([], [], marker="o", mfc="white", mec=S.INK, mew=0.8, ls="none", ms=4,
                 label=LABELS["a_edge"]),
          Line2D([], [], marker="*", color=PROD_C, mec=S.INK, mew=0.4, ls="none", ms=7,
                 label=LABELS["prod"])]
    ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.0, 0.1), handletextpad=0.2,
              labelspacing=0.25, fontsize=S.SMALL_PT - 0.5)

    # (b) search buys a lot: ordered categories, no lines between positions
    xpos = np.arange(len(searches) + 1)
    bx.plot(0, p_ship, "s", ms=4.2, color=S.GREY, mec=S.INK, mew=0.5, zorder=3)
    for t, col in TIER_C.items():
        vals = [s[2] if t == "a" else s[3] for s in searches]
        bx.plot(xpos[1:] + (-0.09 if t == "a" else 0.09), vals, "o", ms=4.2, color=col,
                mec=S.INK, mew=0.4, zorder=3, ls="none", label=LABELS["b_tier"].format(t=t.upper()))
    bx.set_xticks(xpos)
    bx.set_xticklabels([LABELS["none"]] + [LABELS["b_setting"].format(n=n, k=k, s=s)
                                           for (n, k, s), *_ in searches],
                       fontsize=S.SMALL_PT - 1)
    bx.set_xlim(-0.6, len(searches) + 0.5)
    bx.set_xlabel(LABELS["b_x"])
    plt.setp(bx.get_yticklabels(), visible=False)
    hb = [Line2D([], [], marker="s", color=S.GREY, mec=S.INK, mew=0.5, ls="none", ms=4,
                 label=LABELS["b_greedy"])] + bx.get_legend_handles_labels()[0]
    bx.legend(handles=hb, loc="upper right", handletextpad=0.2, labelspacing=0.3,
              fontsize=S.SMALL_PT - 0.5)

    for a_ in (ax, bx):
        a_.set_yscale("log")
        a_.axhline(DE_REF, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(ax.get_xlim()[1] / 1.15, DE_REF * 1.07, LABELS["ref"], ha="right", va="bottom",
            fontsize=S.SMALL_PT - 1, color="#404040")

    # (c, top) CE IsoFLOP curves
    for i, r in enumerate(rungs):
        c = RAMP[i % len(RAMP)]
        cx_top.plot(r.n_grid, r.y_grid, "--", color=c, lw=0.8, zorder=2)
        cx_top.scatter(r.n_vals, r.y_vals, s=5, color=c, lw=0, zorder=3)
        cx_top.plot(r.n_star_median, r.y_star, "*", ms=5.5, color=c, mec=S.INK, mew=0.35, zorder=4)
    cx_top.set_xscale("log")
    lo_y = min(r.y_vals.min() for r in rungs)
    cx_top.set_ylim(lo_y - 0.02, lo_y + 0.3)
    cx_top.set_xlabel(LABELS["c_x_top"], labelpad=1); cx_top.set_ylabel(LABELS["c_y_top"])

    # (c, bottom) N*(C) with the law, its band, and production
    n_draws = min(200, min(r.samples.size for r in good))
    grid = np.geomspace(xs.min() / 1.6, p_c * 1.6, 60)
    band = [P.power_law_fit(xs, np.array([r.samples[i % r.samples.size] for r in good]),
                            sig)(grid) for i in range(n_draws)]
    blo, bhi = np.percentile(band, [2.5, 97.5], axis=0)
    cx.fill_between(grid, blo, bhi, color=S.LIGHT, lw=0, zorder=1)
    cx.plot(grid, law["coef"] * grid ** law["exponent"], "-", color=S.OI["blue"], lw=1.0, zorder=2,
            label=LABELS["c_law"].format(e=law["exponent"], lo=law["ci_low"], hi=law["ci_high"]))
    cx.errorbar(xs, ns, yerr=[ns - ns * np.exp(-sig), ns * np.exp(sig) - ns], fmt="o", ms=3,
                color=S.OI["blue"], mec="white", mew=0.4, elinewidth=0.6, capsize=0, zorder=3)
    cx.plot(p_c, p_n, "*", ms=8, color=PROD_C, mec=S.INK, mew=0.4, zorder=4, label=LABELS["prod"])
    cx.set_xscale("log"); cx.set_yscale("log")
    cx.set_xlabel(LABELS["c_x"], labelpad=1); cx.set_ylabel(LABELS["c_y"])
    cx.legend(loc="lower right", handletextpad=0.3, labelspacing=0.25, borderaxespad=0.1,
              fontsize=S.SMALL_PT - 1)

    for a_, t, x, y in ((ax, "a", -0.22, 1.06), (bx, "b", -0.04, 1.06), (cx_top, "c", -0.38, 1.14)):
        a_.text(x, y, t, transform=a_.transAxes, fontweight="bold", fontsize=S.FONT_PT + 1,
                va="top")

    S.save(fig, "fig8_scaling")
    sets = "; ".join(f"{key} A {a:.2f} B {b:.2f}" for _, key, a, b in searches)
    S.write_provenance(
        "Fig 8", "`scripts/paper_figures/fig8_scaling.py` -> `fig8_scaling.{pdf,png}`. "
        "(a) `results/paper/marginal_returns.json` study[].{budget_flops, greedy_dE_best, usable} "
        f"({sum(not e['usable'] for e in study)} with usable=false drawn open) and "
        f"production.{{budget_flops, greedy_dE_best_lr.value}} = {p_c:.3g} FLOP, {p_best:.2f}. "
        f"(b) production.greedy_dE_shipped_lr.value = {p_ship:.2f} (no search) and "
        f"search.<setting>.{{a,b}}.median, ordered by (candidates, top-k, steps): {sets}. "
        f"Datasets differ (comparability.slice): {comp} "
        f"(c) `analyses/scaling/results/tuned/porian_fit_tuned.json` by_metric.ce.laws.N_star "
        f"exponent {law['exponent']:.4f} [{law['ci_low']:.3f}, {law['ci_high']:.3f}], r2 "
        f"{law['r2']:.3f}, n_usable {n_usable}; curves from `isoflop_tuned.json` runs[].val_loss "
        "recomputed with src.scaling.porian (seed 0) and checked against that exponent. "
        f"Production star: N = {p_n:,} at C = {p_c:.3g}; laws N_star and D_star predict "
        f"{pred_n / 1e6:.1f}M parameters and {pred_d / 1e6:.1f}M examples (actual "
        f"{p_n / 1e6:.1f}M, {p_d / 1e6:.1f}M).")
    print(f"CE alpha {n_law.exponent:.4f} (json {law['exponent']:.4f}); production predicted "
          f"N {pred_n:,.0f} D {pred_d:,.0f}; actual N {p_n:,} D {p_d:,}")


if __name__ == "__main__":
    main()
