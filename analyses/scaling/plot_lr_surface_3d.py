#!/usr/bin/env python3
"""The tuned learning rate in 3D: N and D on the floor, lr up the z axis.

In log space the joint law lr = a N^b D^c is a plane; it is drawn
translucent, coloured by rate like the points. Each point is a tuned
IsoFLOP point at its own (N, D, lr*): filled where the optimum is
bracketed, a triangle where a diverged rate above closes it, hollow where
it sits at the grid edge (a bound). A thin line from each point to the
plane is its residual. Two views of the same scene.

    python analyses/scaling/plot_lr_surface_3d.py \\
        --law analyses/scaling/results/lr_law_fit_cliff.json \\
        --tuned analyses/scaling/results/tuned/isoflop_tuned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lr_surface import RAMP, SURF, INK, INK2  # noqa: E402


def load_points(fit: dict, tuned: dict) -> list:
    """Every tuned point at the examples it saw (as plot_lr_surface does)."""
    seed42 = [r for r in tuned["runs"] if r["seed"] == 42]
    pts = []
    for c in fit["configs"]:
        n, d = c["n_params"], c["train_examples"]
        match = [r for r in seed42 if r["n_params"] == n
                 and any(abs(r["passes"] - k * d) <= 0.01 * r["passes"] for k in (1, 2, 3))]
        if match:
            pts.append({"N": n, "D": match[0]["passes"], "lr": c["lr_star"],
                        "ok": bool(c["usable"]), "cliff": c.get("bracket") == "cliff"})
    return pts


def log_ticks(lo: float, hi: float) -> list:
    return [k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= k <= hi]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--law", default="analyses/scaling/results/lr_law_fit.json")
    p.add_argument("--tuned", default="analyses/scaling/results/tuned/isoflop_tuned.json")
    p.add_argument("--output", default="analyses/scaling/results/tuned/lr_surface_3d.png")
    a = p.parse_args()

    fit = json.load(open(a.law))
    law = fit["lr_vs_n_and_d"]
    pts = load_points(fit, json.load(open(a.tuned)))
    x = np.log10([q["N"] for q in pts])
    y = np.log10([q["D"] for q in pts])
    z = np.log10([q["lr"] for q in pts])
    plane = lambda xx, yy: (np.log10(law["coef"]) + law["n_exponent"] * xx
                            + law["d_exponent"] * yy)
    zp = plane(x, y)
    n_cliff = sum(q["cliff"] for q in pts)
    n_ok = sum(q["ok"] for q in pts)
    print(f"{len(pts)} points, {n_ok} bracketed ({n_cliff} by divergence); "
          f"rms residual of bracketed points {np.sqrt(np.mean([(zi - zpi) ** 2 for zi, zpi, q in zip(z, zp, pts) if q['ok']])):.3f} dex")

    cmap = LinearSegmentedColormap.from_list("lr", RAMP)
    zlo, zhi = z.min() - 0.15, z.max() + 0.15
    norm = Normalize(vmin=zlo, vmax=zhi)
    gx = np.linspace(x.min() - 0.1, x.max() + 0.1, 120)
    gy = np.linspace(y.min() - 0.1, y.max() + 0.1, 120)
    GX, GY = np.meshgrid(gx, gy)
    GZ = plane(GX, GY)

    fig = plt.figure(figsize=(15, 7.2), facecolor=SURF)
    GZ = np.where((GZ >= zlo) & (GZ <= zhi), GZ, np.nan)
    for k, (elev, azim) in enumerate([(18, -35), (6, -60)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_box_aspect(None, zoom=1.12)
        ax.set_facecolor(SURF)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.97, 0.97, 0.96, 1.0))
            pane.set_edgecolor("#e3e2dd")
        ax.plot_surface(GX, GY, GZ, facecolors=cmap(norm(GZ)), alpha=0.42,
                        linewidth=0, antialiased=True, shade=False)
        ax.plot_wireframe(GX, GY, GZ, rstride=24, cstride=24, color=INK2,
                          linewidth=0.35, alpha=0.35)
        for xi, yi, zi, zpi, q in zip(x, y, z, zp, pts):
            ax.plot([xi, xi], [yi, yi], [zi, zpi], "-", color=INK2,
                    linewidth=0.8, alpha=0.7)
            col = cmap(norm(zi))
            if q["ok"]:
                ax.scatter(xi, yi, zi, s=70 if q["cliff"] else 55, color=col,
                           edgecolor=INK, linewidth=0.7, depthshade=False,
                           marker="^" if q["cliff"] else "o")
            else:
                ax.scatter(xi, yi, zi, s=50, facecolor=SURF, edgecolor=col,
                           linewidth=1.8, depthshade=False)
        for setter, vals, label in ((ax.set_xticks, x, "N, parameters"),
                                    (ax.set_yticks, y, "D, examples seen")):
            ticks = log_ticks(vals.min() - 0.1, vals.max() + 0.1)
            setter(ticks)
        ax.set_xticklabels([f"1e{t}" for t in ax.get_xticks()], fontsize=8, color=INK2)
        ax.set_yticklabels([f"1e{t}" for t in ax.get_yticks()], fontsize=8, color=INK2)
        zt = [v for v in np.arange(np.floor(z.min() * 2) / 2, z.max() + 0.5, 0.5)
              if z.min() - 0.2 <= v <= z.max() + 0.2]
        ax.set_zticks(zt)
        ax.set_zticklabels([f"{10 ** v:.0e}".replace("e-0", "e-") for v in zt],
                           fontsize=8, color=INK2)
        ax.set_xlabel("N, parameters", color=INK2, fontsize=9, labelpad=6)
        ax.set_ylabel("D, examples seen", color=INK2, fontsize=9, labelpad=6)
        ax.set_zlabel("tuned learning rate", color=INK2, fontsize=9, labelpad=8)
        ax.set_xlim(gx.min(), gx.max())
        ax.set_ylim(gy.min(), gy.max())
        ax.set_zlim(zlo, zhi)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(["A.  The plane falls along both N and D", "B.  Edge-on: scatter about the plane"][k],
                     color=INK, fontsize=11.5, loc="left")

    handles = [plt.Line2D([], [], marker="o", color=RAMP[3], markeredgecolor=INK,
                          linestyle="none", label="optimum bracketed"),
               plt.Line2D([], [], marker="^", color=RAMP[3], markeredgecolor=INK,
                          linestyle="none", label="bracketed by a diverged rate above"),
               plt.Line2D([], [], marker="o", markerfacecolor=SURF, markeredgecolor=RAMP[3],
                          markeredgewidth=1.8, linestyle="none",
                          label="best rate at the grid edge (a bound, not fitted)"),
               plt.Line2D([], [], color=INK2, linewidth=0.8, label="residual to the plane")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("The tuned learning rate lies close to one plane in log N, log D, log lr",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.935, f"plane: lr* = {law['coef']:.3g} · N^{law['n_exponent']:.2f} · "
             f"D^{law['d_exponent']:.2f}, fitted on {law['n_points']} bracketed points, "
             f"r² = {law['r2']:.2f}", fontsize=10, color=INK2)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.08, wspace=0.0)
    plt.savefig(out, dpi=160, facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
