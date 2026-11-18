#!/usr/bin/env python3
"""LR law figure: the tuned learning rate as a plane in (log N, log D, log lr).

Publication re-render of analyses/scaling/results/tuned/lr_surface_3d.png
(analyses/scaling/plot_lr_surface_3d.py): same law file, same points (its
own loader is imported, not copied), same two views. The 2D isoline version
(lr_surface_cliff.png) is deliberately not used.

  law     analyses/scaling/results/lr_law_fit_cliff.json  lr_vs_n_and_d
          {coef, n_exponent, d_exponent, r2, n_points}; configs[] (usable,
          bracket); leave_one_rung_out
  points  analyses/scaling/results/tuned/isoflop_tuned.json (seed 42),
          matched to configs[] by plot_lr_surface_3d.load_points
  check   results/lr_curves/lr_search_ep1_lim39959808_d352_se6_*.json, the
          production-size learning-rate sweep (N = 11.3M, D = 40M), never
          used in the fit: its DeltaE-optimal rate is drawn as a star.

Also writes results/paper/lr_law_numbers.json: every number the LR-law
paragraph quotes, computed here from the same files.

    python scripts/paper_figures/fig_lr_law.py
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import _style as S

sys.path.insert(0, str(S.REPO / "analyses/scaling"))
from plot_lr_surface_3d import load_points  # noqa: E402

LAW = S.REPO / "analyses/scaling/results/lr_law_fit_cliff.json"
STRICT = S.REPO / "analyses/scaling/results/lr_law_fit.json"
TUNED = S.REPO / "analyses/scaling/results/tuned/isoflop_tuned.json"
PROD_SWEEP = "results/lr_curves/lr_search_ep1_lim39959808_d352_se6_bs256_b20.999_*.json"
PROD_META = S.REPO / "results/production/current/PRODUCTION.json"
SHOW_PRODUCTION = True
VIEWS = ((20, -38), (4, -62))      # (elevation, azimuth) of panels a and b

# Sequential single-hue ramp (light = low rate), legible in grayscale.
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0a2a52"]

LABELS = {
    "x": "Parameters, N",
    "y": "Examples seen, D",
    "z": "Tuned learning rate",
    "interior": "Optimum bracketed",
    "cliff": "Bracketed by a diverged rate above",
    "open": "At grid edge (bound, not fitted)",
    "resid": "Residual to the plane",
    "prod": "Production size, held out",
    "plane": r"$\eta^* = {m:.1f}\times10^{{{e}}}\,N^{{{b:.2f}}}\,D^{{{c:.2f}}}$,  $r^2$ = {r2:.2f}",
}


def prod_sweep():
    """Production-size sweep: (lr, val_de, val_loss, diverged) per rate."""
    out = []
    for f in sorted(glob.glob(str(S.REPO / PROD_SWEEP))):
        d = json.load(open(f))
        t = d["results"][0]
        div = not np.isfinite(t["final_val_loss"]) or t["final_val_loss"] > 50
        out.append(dict(lr=d["optimal_lr"], de=t["final_val_de"], ce=t["final_val_loss"],
                        diverged=bool(div), n=d["n_params"]))
    return sorted(out, key=lambda r: r["lr"])


def numbers(fit, strict, pts, sweep, prod):
    law = fit["lr_vs_n_and_d"]
    pred = lambda n, d: law["coef"] * n ** law["n_exponent"] * d ** law["d_exponent"]
    ok = [q for q in pts if q["ok"]]
    res = np.log10([q["lr"] / pred(q["N"], q["D"]) for q in ok])
    loro = fit["leave_one_rung_out"]["rungs"]
    top = next(r for r in loro if r["held_out"].startswith("top"))
    stable = [r for r in sweep if not r["diverged"]]
    best_de = min(stable, key=lambda r: r["de"])
    best_ce = min(stable, key=lambda r: r["ce"])
    shipped = next(r for r in sweep if abs(r["lr"] - prod["lr"]) / prod["lr"] < 1e-3)
    n_prod, d_prod = prod_n(sweep), prod["train_examples"]
    return {
        "source": {"law": str(LAW.relative_to(S.REPO)), "strict_law": str(STRICT.relative_to(S.REPO)),
                   "tuned": str(TUNED.relative_to(S.REPO)), "production_sweep": PROD_SWEEP},
        "law": {"coef": law["coef"], "n_exponent": law["n_exponent"],
                "d_exponent": law["d_exponent"], "r2": law["r2"], "n_points": law["n_points"]},
        "configs": {"total": len(fit["configs"]),
                    "bracket_counts": {b: sum(c.get("bracket") == b for c in fit["configs"])
                                       for b in ("interior", "cliff", "open")}},
        "ranges": {"N": [min(q["N"] for q in ok), max(q["N"] for q in ok)],
                   "D_seen": [min(q["D"] for q in ok), max(q["D"] for q in ok)],
                   "lr": [min(q["lr"] for q in ok), max(q["lr"] for q in ok)]},
        "rms_residual_dex": float(np.sqrt(np.mean(res ** 2))),
        "rms_residual_factor": float(10 ** np.sqrt(np.mean(res ** 2))),
        "n_only_fit": {"exponent": fit["lr_vs_n"]["exponent"], "r2": fit["lr_vs_n"]["r2"]},
        "leave_one_rung_out": {
            "n_exponent_range": [min(r["law_without_it"]["n_exponent"] for r in loro),
                                 max(r["law_without_it"]["n_exponent"] for r in loro)],
            "d_exponent_range": [min(r["law_without_it"]["d_exponent"] for r in loro),
                                 max(r["law_without_it"]["d_exponent"] for r in loro)],
            "top_rung_held_out": {"budget": top["budget"], "n_points": top["n_points"],
                                  "median_measured_over_predicted": top["median_ratio"],
                                  "worst_measured_over_predicted": top["worst_ratio"]}},
        "strict_law": {k: strict["lr_vs_n_and_d"][k] for k in ("n_exponent", "d_exponent", "r2", "n_points")},
        "production_check": {
            "N": n_prod, "D": d_prod,
            "D_over_max_fitted_D": d_prod / max(q["D"] for q in ok),
            "predicted_lr": pred(n_prod, d_prod),
            "best_val_de": {"lr": best_de["lr"], "val_de": best_de["de"]},
            "best_val_ce": {"lr": best_ce["lr"], "val_ce": best_ce["ce"]},
            "shipped": {"lr": shipped["lr"], "val_de": shipped["de"], "val_ce": shipped["ce"],
                        "over_predicted": shipped["lr"] / pred(n_prod, d_prod)},
            "sweep": sweep, "seeds": 1},
    }


def prod_n(sweep):
    return int(sweep[0]["n"])


def main():
    S.apply()
    fit, strict = json.load(open(LAW)), json.load(open(STRICT))
    law = fit["lr_vs_n_and_d"]
    pts = load_points(fit, json.load(open(TUNED)))
    sweep, prod = prod_sweep(), json.load(open(PROD_META))
    nums = numbers(fit, strict, pts, sweep, prod)
    (S.REPO / "results/paper/lr_law_numbers.json").write_text(json.dumps(nums, indent=1) + "\n")

    x = np.log10([q["N"] for q in pts]); y = np.log10([q["D"] for q in pts])
    z = np.log10([q["lr"] for q in pts])
    plane = lambda xx, yy: np.log10(law["coef"]) + law["n_exponent"] * xx + law["d_exponent"] * yy
    zp = plane(x, y)
    pc = nums["production_check"]
    px, py = np.log10(pc["N"]), np.log10(pc["D"])
    pz = np.log10(pc["best_val_de"]["lr"])

    cmap = LinearSegmentedColormap.from_list("lr", RAMP)
    zlo = min(z.min(), pz if SHOW_PRODUCTION else z.min()) - 0.15
    zhi = z.max() + 0.15
    norm = Normalize(vmin=zlo, vmax=zhi)
    gx = np.linspace(x.min() - 0.1, max(x.max(), px) + 0.1, 80)
    gy = np.linspace(y.min() - 0.1, max(y.max(), py) + 0.1, 80)
    GX, GY = np.meshgrid(gx, gy)
    GZ = plane(GX, GY)
    GZ = np.where((GZ >= zlo) & (GZ <= zhi), GZ, np.nan)

    fig = plt.figure(figsize=(S.DOUBLE, 8.6 * S.CM))
    for k, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_axes([0.0 + 0.5 * k, 0.1, 0.5, 0.9], projection="3d")
        ax.set_box_aspect(None, zoom=0.92)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.97, 0.97, 0.97, 1.0)); pane.set_edgecolor("#DDDDDD")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(color="#E3E3E3", linewidth=0.4)
        surf = ax.plot_surface(GX, GY, GZ, facecolors=cmap(norm(GZ)), alpha=0.4, linewidth=0,
                               antialiased=True, shade=False)
        surf.set_rasterized(True)
        ax.plot_wireframe(GX, GY, GZ, rstride=16, cstride=16, color="#555555", linewidth=0.25,
                          alpha=0.35)
        for xi, yi, zi, zpi, q in zip(x, y, z, zp, pts):
            ax.plot([xi, xi], [yi, yi], [zi, zpi], "-", color="#555555", lw=0.5, alpha=0.8)
            col = cmap(norm(zi))
            if q["ok"]:
                ax.scatter(xi, yi, zi, s=16 if q["cliff"] else 13, color=col, edgecolor=S.INK,
                           linewidth=0.4, depthshade=False, marker="^" if q["cliff"] else "o")
            else:
                ax.scatter(xi, yi, zi, s=12, facecolor="white", edgecolor=col, linewidth=1.0,
                           depthshade=False)
        if SHOW_PRODUCTION:
            ax.plot([px, px], [py, py], [pz, plane(px, py)], "-", color=S.OI["vermilion"], lw=0.7)
            ax.scatter(px, py, pz, s=45, marker="*", color=S.OI["vermilion"], edgecolor=S.INK,
                       linewidth=0.4, depthshade=False, zorder=10)
        xt = [t for t in range(int(np.floor(gx.min())), int(np.ceil(gx.max())) + 1)
              if gx.min() <= t <= gx.max()]
        yt = [t for t in range(int(np.floor(gy.min())), int(np.ceil(gy.max())) + 1)
              if gy.min() <= t <= gy.max()]
        ax.set_xticks(xt); ax.set_xticklabels([f"$10^{{{t}}}$" for t in xt])
        ax.set_yticks(yt); ax.set_yticklabels([f"$10^{{{t}}}$" for t in yt])
        zt = [t for t in range(int(np.floor(zlo)), int(np.ceil(zhi)) + 1) if zlo <= t <= zhi]
        ax.set_zticks(zt); ax.set_zticklabels([f"$10^{{{t}}}$" for t in zt])
        ax.tick_params(labelsize=S.SMALL_PT - 1, pad=0)
        ax.set_xlabel(LABELS["x"], labelpad=-4, fontsize=S.SMALL_PT)
        ax.set_ylabel(LABELS["y"], labelpad=-4, fontsize=S.SMALL_PT)
        ax.set_zlabel(LABELS["z"], labelpad=-4, fontsize=S.SMALL_PT)
        ax.set_xlim(gx.min(), gx.max()); ax.set_ylim(gy.min(), gy.max()); ax.set_zlim(zlo, zhi)
        ax.view_init(elev=elev, azim=azim)
        fig.text(0.01 + 0.5 * k, 0.97, "ab"[k], fontweight="bold", fontsize=S.FONT_PT + 1,
                 va="top")

    c = RAMP[3]
    handles = [Line2D([], [], marker="o", color=c, mec=S.INK, mew=0.4, ls="none", ms=4,
                      label=LABELS["interior"]),
               Line2D([], [], marker="^", color=c, mec=S.INK, mew=0.4, ls="none", ms=4.5,
                      label=LABELS["cliff"]),
               Line2D([], [], marker="o", mfc="white", mec=c, mew=1.0, ls="none", ms=4,
                      label=LABELS["open"]),
               Line2D([], [], color="#555555", lw=0.6, label=LABELS["resid"])]
    if SHOW_PRODUCTION:
        handles.append(Line2D([], [], marker="*", color=S.OI["vermilion"], mec=S.INK, mew=0.4,
                              ls="none", ms=7, label=LABELS["prod"]))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.3)
    m, e = f"{law['coef']:.1e}".split("e")
    fig.text(0.99, 0.03, LABELS["plane"].format(m=float(m), e=int(e), b=law["n_exponent"],
                                                c=law["d_exponent"], r2=law["r2"]),
             ha="right", va="bottom", fontsize=S.SMALL_PT)

    S.save(fig, "fig_lr_law")
    S.write_provenance(
        "Fig LR law", "`scripts/paper_figures/fig_lr_law.py` -> `fig_lr_law.{pdf,png}`; numbers "
        "for the paragraph in `results/paper/lr_law_numbers.json`. Plane: "
        "`analyses/scaling/results/lr_law_fit_cliff.json` lr_vs_n_and_d {coef, n_exponent, "
        "d_exponent, r2, n_points} = "
        f"{law['coef']:.4g} N^{law['n_exponent']:.3f} D^{law['d_exponent']:.3f}, r2 {law['r2']:.3f}, "
        f"{law['n_points']} points. Points: configs[] matched to `analyses/scaling/results/tuned/"
        "isoflop_tuned.json` seed-42 runs by analyses/scaling/plot_lr_surface_3d.load_points "
        "(D = examples seen). Star: production-size sweep `results/lr_curves/"
        "lr_search_ep1_lim39959808_d352_se6_bs256_b20.999_*.json`, DeltaE-optimal rate "
        f"{pc['best_val_de']['lr']:.3g}, not in the fit. Same as lr_surface_3d.png except style "
        "and the star.")
    print(json.dumps({k: nums[k] for k in ("law", "rms_residual_dex", "leave_one_rung_out")}, indent=1))
    print(json.dumps({k: v for k, v in pc.items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    main()
