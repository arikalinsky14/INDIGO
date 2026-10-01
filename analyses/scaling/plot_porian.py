"""INDIGO's sweep in the figure grammar of Porian et al. 2024.

Three panels, matching what their `plotting.py` draws:

  A  IsoFLOP curves. Raw points per budget, the Akima interpolation as a dashed
     line, a star at its argmin. Their `isoflop_curves_plot`.
  B  N*(C). Observations with asymmetric error bars from the bootstrap's log
     sigma, a shaded 95% confidence region, the fitted power law, and a
     Chinchilla alpha = 0.5 reference. Their `opt_param_vs_compute_plot`.
  C  The multiplier D*/N*. They report it as its own power law; we never have.
     Chinchilla's is flat at about 20 tokens per parameter.

The x axis is FLOPs by default and service units with --x-axis credits. FLOPs
are what a paper needs and what compares to Chinchilla and Kaplan; credits are
what an allocation holder reads. Selecting credits refits the laws against
credits rather than rescaling a FLOP fit, since the two axes are not
proportional: a rung's cost in SU depends on the throughput of the model that
spends it, not on its FLOP count alone.

    python analyses/scaling/plot_porian.py                     # FLOPs
    python analyses/scaling/plot_porian.py --x-axis credits    # service units
    python analyses/scaling/plot_porian.py --metric low        # a chroma bucket
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scaling import porian as P                      # noqa: E402
from src.scaling.credits import CreditModel              # noqa: E402

# Validated ordinal ramp (one hue, monotone lightness, gaps >= 0.06,
# light end clears the surface). Six steps for six budgets.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#124782", "#0d366b"]
FIT_C, REF = "#eb6834", "#8a8880"
SURF, INK, INK2, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#ebeae5", linewidth=0.8, which="both")
    ax.set_axisbelow(True)


def plain_log_ticks(ax) -> None:
    """Plain decimal labels on a log axis: 1, 2, 5 rather than 10^0.

    Service units span well under a decade here, so every tick is a power of
    ten only in the sense that matplotlib says so. Printing 10^0 for "one
    service unit" is noise on an axis whose whole purpose is being readable to
    someone holding an allocation.
    """
    def fmt(v, _pos):
        if v <= 0:
            return ""
        if v >= 100:
            return f"{v:,.0f}"
        if v >= 10:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:g}"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    for axis in (ax.xaxis,):
        axis.set_major_formatter(FuncFormatter(fmt))
        axis.set_minor_formatter(FuncFormatter(fmt))
    ax.tick_params(axis="x", which="minor", labelsize=8)


def value_of(run, metric):
    return run["val_loss"] if metric == "ce" else run["val_de"][metric]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="analyses/scaling/results/isoflop_fit.json")
    p.add_argument("--metric", default="pooled",
                   choices=["pooled", "low", "mid", "high", "ce"])
    p.add_argument("--x-axis", choices=["flops", "credits"], default="flops")
    p.add_argument("--su-per-gpu-hour", type=float, default=None)
    p.add_argument("--bootstrap-iters", type=int, default=P.BOOTSTRAP_ITERS)
    p.add_argument("--x-pad-left", type=float, default=2.2,
                   help="extra room to the left of the smallest model in "
                        "panel A, as a factor on N. The lowest budget's left "
                        "branch is otherwise flush against the axis.")
    p.add_argument("--x-pad-right", type=float, default=1.25)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    src = json.load(open(a.fit))
    by_b = collections.defaultdict(list)
    for r in src["runs"]:
        by_b[min(src["budgets"], key=lambda b: abs(b - r["flops"]))].append(r)
    budgets = sorted(by_b)

    clusters = collections.defaultdict(list)
    for r in src["runs"]:
        clusters[(round(r["flops"], -11), r["n_params"])].append(value_of(r, a.metric))
    noise = P.NoiseModel.from_clusters(list(clusters.values()))

    rungs = [P.fit_rung(b, [r["n_params"] for r in by_b[b]],
                        [value_of(r, a.metric) for r in by_b[b]],
                        d_vals=[r["passes"] for r in by_b[b]], noise=noise,
                        n_boot=a.bootstrap_iters, rng=np.random.default_rng(0))
             for b in budgets]
    good = [r for r in rungs if r.usable]
    if len(good) < 2:
        sys.exit(f"only {len(good)} usable rungs for metric {a.metric}")

    # A rung with no models below its N* has no descending branch to show, and
    # its minimum rests on the interpolation rather than on data either side.
    thin = [(r.budget, int(np.sum(r.n_vals < r.n_star_median))) for r in good]
    for budget, n_left in thin:
        if n_left < 2:
            print(f"[WARN] C = {budget:.3g} has only {n_left} model(s) below "
                  f"N*, so its left branch is barely sampled. Two more "
                  f"configs below N* at that budget are the cheapest runs in "
                  f"the study.")

    credits = (CreditModel(su_per_gpu_hour=a.su_per_gpu_hour)
               if a.su_per_gpu_hour else CreditModel())
    use_credits = a.x_axis == "credits"
    if use_credits and credits.caveat:
        print(f"[WARN] {credits.caveat}")

    def x_of(rung):
        """The rung's position on the chosen axis.

        For credits this is what the compute-optimal run at that budget costs,
        which is a property of that run rather than of the budget, so the law
        below is refitted against it rather than rescaled from the FLOP fit.
        """
        return credits.from_passes(rung.d_star_median) if use_credits else rung.budget

    xs = np.array([x_of(r) for r in good])
    x_span = float(xs.max() / xs.min())
    # A power law needs decades. The FLOP axis spans 250x; the credit axis can
    # span far less, because wall clock on this cluster tracks examples
    # processed rather than FLOPs (these models are input-bound, not
    # GPU-bound), and D* is nearly constant across the sweep. Fitting an
    # exponent across a narrow span produces a large number that is an artefact
    # of the compression, so say so rather than printing it bare.
    span_warning = (f"x spans only {x_span:.1f}x on this axis; an exponent "
                    f"fitted across that is not identifiable"
                    if x_span < 10 else "")
    if span_warning:
        print(f"[WARN] {span_warning}")
    ns = np.array([r.n_star_median for r in good])
    ds = np.array([r.d_star_median for r in good])
    sig = np.array([r.log_sigma for r in good])
    xlabel = ("compute (Pitt CRC service units)" if use_credits
              else "C (training FLOPs)")

    n_law = P.power_law_fit(xs, ns, sig)
    m_law = P.power_law_fit(xs, ds / ns, sig)
    # Confidence band: refit on bootstrap draw i of every rung.
    n_draws = min(200, min(r.samples.size for r in good))
    grid = np.geomspace(xs.min() / 1.6, xs.max() * 1.6, 60)
    band = []
    for i in range(n_draws):
        ys = np.array([r.samples[i % r.samples.size] for r in good])
        band.append(P.power_law_fit(xs, ys, sig)(grid))
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
    exps = np.array([P.power_law_fit(xs, np.array([r.samples[i % r.samples.size]
                                                   for r in good]), sig).exponent
                     for i in range(n_draws)])

    label = {"pooled": r"$\Delta E_{00}$", "ce": "cross-entropy"}.get(
        a.metric, rf"$\Delta E_{{00}}$ ({a.metric} chroma)")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), facecolor=SURF)
    for ax in axes:
        style(ax)

    # ---- A: IsoFLOP curves ------------------------------------------------
    ax = axes[0]
    for i, r in enumerate(rungs):
        col = RAMP[i % len(RAMP)]
        ax.scatter(r.n_vals, r.y_vals, s=26, color=col, zorder=3,
                   edgecolor=SURF, linewidth=0.8, label=f"C = {r.budget:.2g}")
        ax.plot(r.n_grid, r.y_grid, "--", color=col, linewidth=1.5, zorder=2)
        if r.usable:
            ax.plot([r.n_star_median], [r.y_star], "*", color=col, markersize=15,
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
        else:
            ax.plot([r.n_star], [r.y_star], "x", color=col, markersize=9,
                    markeredgewidth=2, zorder=4)
    ax.set_xscale("log")
    # Breathing room on the left. The lowest budget's descending branch runs
    # right into the spine otherwise, because only one model was sampled below
    # its N*. Padding makes the shape readable; it does not add data, and the
    # real fix is sampling two smaller configs at that budget.
    n_lo = min(r.n_vals.min() for r in rungs)
    n_hi = max(r.n_vals.max() for r in rungs)
    ax.set_xlim(n_lo / a.x_pad_left, n_hi * a.x_pad_right)
    ax.set_xlabel("N (parameters)", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"val {label}", color=INK2, fontsize=9.5)
    ax.set_title("A.  IsoFLOP curves, Akima interpolated", color=INK,
                 fontsize=11.5, pad=10, loc="left")
    ax.plot([], [], "*", color=INK2, markersize=12, markeredgecolor=INK,
            markeredgewidth=0.6, linestyle="none", label=r"$N^*$")
    ax.plot([], [], "x", color=INK2, markersize=8, markeredgewidth=2,
            linestyle="none", label="argmin on the edge (rejected)")
    leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2,
                    handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    lo_y = min(r.y_vals.min() for r in rungs)
    hi_y = max(r.y_vals.max() for r in rungs)
    ax.set_ylim(lo_y - 0.06 * (hi_y - lo_y), hi_y + 0.34 * (hi_y - lo_y))

    # ---- B: N*(C) ---------------------------------------------------------
    ax = axes[1]
    ax.fill_between(grid, lo, hi, color=REF, alpha=0.18, zorder=1,
                    label="95% confidence region")
    ax.errorbar(xs, ns, yerr=[ns - ns * np.exp(-sig), ns * np.exp(sig) - ns],
                fmt="o", color=FIT_C, markersize=7, capsize=5, linewidth=1.4,
                markeredgecolor=SURF, markeredgewidth=1.0, zorder=4,
                label="observations")
    ax.plot(grid, n_law(grid), "--", color=FIT_C, linewidth=2, zorder=3,
            label=rf"$N^* \propto C^{{{n_law.exponent:.2f}}}$")
    anchor = ns[0] * (grid / xs[0]) ** 0.5
    ax.plot(grid, anchor, ls=(0, (5, 3)), color=REF, linewidth=1.8, zorder=2,
            label=r"Chinchilla  $\alpha$ = 0.50")
    ax.set_xscale("log"); ax.set_yscale("log")
    if use_credits:
        plain_log_ticks(ax)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(r"$N^*$ (parameters)", color=INK2, fontsize=9.5)
    ax.set_title(rf"B.  $\alpha$ = {n_law.exponent:+.3f}  "
                 rf"[{np.percentile(exps, 2.5):+.2f}, {np.percentile(exps, 97.5):+.2f}]",
                 color=INK, fontsize=11.5, pad=10, loc="left")
    leg = ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)

    # ---- C: the multiplier ------------------------------------------------
    ax = axes[2]
    mult = ds / ns
    ax.plot(xs, mult, "o", color=FIT_C, markersize=8, markeredgecolor=SURF,
            markeredgewidth=1.1, zorder=3, label="observations")
    ax.plot(grid, m_law(grid), "--", color=FIT_C, linewidth=2, zorder=2,
            label=rf"$D^*/N^* \propto C^{{{m_law.exponent:.2f}}}$")
    ax.plot(grid, np.full_like(grid, mult[0]), ls=(0, (5, 3)), color=REF,
            linewidth=1.8, zorder=1, label="Chinchilla: flat")
    ax.set_xscale("log"); ax.set_yscale("log")
    if use_credits:
        plain_log_ticks(ax)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(r"$D^*/N^*$ (examples per parameter)", color=INK2, fontsize=9.5)
    ax.set_title("C.  Examples per parameter falls with compute", color=INK,
                 fontsize=11.5, pad=10, loc="left")
    leg = ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(f"INDIGO IsoFLOP sweep, estimated as in Porian et al. 2024  "
                 f"({label})", fontsize=13.5, color=INK, x=0.008, ha="left",
                 y=1.035)
    sub = (f"{len(good)} of {len(rungs)} rungs usable. Akima interpolation, "
           f"boundary rejection, seed-noise bootstrap ({a.bootstrap_iters} draws, "
           rf"$\sigma$ = {noise.sigma_lo:.3f} from repeat seeds), "
           f"1/$\\sigma^2$-weighted fit.")
    if use_credits and credits.caveat:
        sub += f"  SU rates are placeholders: {credits.su_per_gpu_hour:g}/GPU-hour."
    if span_warning:
        sub += f"  CAUTION: {span_warning}."
        for ax in (axes[1], axes[2]):
            ax.text(0.5, 0.5, "exponent not identifiable\non this axis",
                    transform=ax.transAxes, fontsize=13, color="#d03b3b",
                    ha="center", va="center", alpha=0.35, rotation=18,
                    zorder=10)
    fig.text(0.008, 0.972, sub, fontsize=9, color=INK2, ha="left")

    out = a.output or (f"analyses/scaling/results/porian_{a.metric}"
                       f"{'_credits' if use_credits else ''}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")
    print(f"  alpha(N*)      {n_law.exponent:+.3f}  r2 {n_law.r2:.3f}")
    print(f"  multiplier     {m_law.exponent:+.3f}  r2 {m_law.r2:.3f}")
    print(f"  usable rungs   {len(good)} of {len(rungs)}")


if __name__ == "__main__":
    main()
