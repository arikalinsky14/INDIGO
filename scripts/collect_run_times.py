#!/usr/bin/env python3
"""Measured GPU time per training run, from the lr_grid.sh job logs.

The FLOP axis is what a paper needs; GPU-hours (and Pitt CRC service units,
8 per l40s GPU-hour) are what an allocation holder reads. Rather than convert
one into the other with an assumed constant, this reads what every run
actually took. Each lr_grid.sh log carries the cell's N and D in its banner
and, per learning rate, a `[time] train X min` line.

Only logs written by the current input pipeline are read (they carry the
batch loader's `[loader]` lines); earlier logs ran 6 to 8x slower and say
nothing about what a run costs now.

Two readings per run, because the data is re-read from /ix1 for every rate:

  cold  the first rate a task trains. Its shards come off /ix1 uncached. This
        is what ONE training run costs, so it is the cost basis.
  warm  later rates of the same task, read through the file server's cache,
        up to ~3x faster. Reported as the optimistic bound.

Writes a JSON (every trial, plus examples/s per model size) and a figure:
GPU-hours against FLOPs for every cold run, which is the direct test of
whether FLOPs track GPU time, and throughput against N.

    python scripts/collect_run_times.py --logs job-outputs \\
        --output analyses/scaling/results/tuned/run_times.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEAD = {
    "d_se": re.compile(r"d_model / se\s+(\d+) / (\d+)"),
    "n": re.compile(r"N \(parameters\)\s+(\d+)"),
    "d": re.compile(r"D \(examples\)\s+(\d+) x (\d+) epoch"),
    "seed": re.compile(r"^\s+seed\s+(\d+)", re.M),
    "stage": re.compile(r"STAGE (\S+), cell"),
}
TRIAL = re.compile(r"Training with LR = ([0-9.eE+-]+)")
TIME = re.compile(r"\[time\] train ([0-9.]+) min")


def parse_log(path: Path) -> list:
    text = path.read_text(errors="replace")
    if "[loader]" not in text:
        return []
    m = {k: r.search(text) for k, r in HEAD.items()}
    if not (m["d_se"] and m["n"] and m["d"]):
        return []
    d_model, se = map(int, m["d_se"].groups())
    n = int(m["n"].group(1))
    limit, epochs = map(int, m["d"].groups())
    passes = limit * epochs
    from scripts.lr_edges import arch_for
    from src.scaling.flops import train_flops_per_example
    flops = train_flops_per_example(arch_for(d_model, se)) * passes
    out, lr = [], None
    k = 0
    for line in text.splitlines():
        t = TRIAL.search(line)
        if t:
            lr = float(t.group(1))
            continue
        t = TIME.search(line)
        if t and lr is not None:
            sec = float(t.group(1)) * 60
            if sec <= 0:
                continue
            out.append({"log": path.name, "stage": m["stage"].group(1)
                        if m["stage"] else None, "d_model": d_model, "se": se,
                        "n_params": n, "passes": passes, "flops": flops,
                        "seed": int(m["seed"].group(1)) if m["seed"] else None,
                        "lr": lr, "train_sec": sec, "gpu_hours": sec / 3600,
                        "ex_per_sec": passes / sec, "cold": k == 0})
            k += 1
            lr = None
    return out


def rate_table(trials: list) -> dict:
    """Median examples/s per model size, cold and warm."""
    by = collections.defaultdict(lambda: {"cold": [], "warm": []})
    for t in trials:
        by[t["n_params"]]["cold" if t["cold"] else "warm"].append(t["ex_per_sec"])
    return {str(n): {k: (float(np.median(v)) if v else None)
                     for k, v in d.items()} |
            {"n_cold": len(d["cold"]), "n_warm": len(d["warm"])}
            for n, d in sorted(by.items())}


class RateModel:
    """Examples/s as a function of N: log-log interpolation through the
    per-size medians, held flat beyond the measured sizes."""

    def __init__(self, table: dict, which: str = "cold"):
        pts = sorted((float(n), v[which]) for n, v in table.items()
                     if v.get(which))
        if len(pts) < 2:
            raise ValueError(f"need two model sizes with {which} timings, "
                             f"have {len(pts)}")
        self.which = which
        self.n = np.log([p[0] for p in pts])
        self.r = np.log([p[1] for p in pts])

    def __call__(self, n_params):
        return np.exp(np.interp(np.log(n_params), self.n, self.r))

    def gpu_hours(self, n_params, passes):
        return np.asarray(passes) / self(n_params) / 3600.0

    @classmethod
    def from_file(cls, path: str, which: str = "cold") -> "RateModel":
        return cls(json.load(open(path))["rates"], which)


def plot(trials: list, table: dict, out: Path, su_per_gpu_hour: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    SURF, INK, INK2, FAINT, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#8a8880"
    cold = [t for t in trials if t["cold"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), facecolor=SURF)
    for ax in axes:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(FAINT)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color="#ebeae5", linewidth=0.8, which="both")
        ax.set_axisbelow(True)

    ax = axes[0]
    f = np.array([t["flops"] for t in cold])
    h = np.array([t["gpu_hours"] for t in cold])
    n = np.array([t["n_params"] for t in cold])
    sc = ax.scatter(f, h, c=n, cmap="viridis", norm=LogNorm(), s=30,
                    edgecolor=SURF, linewidth=0.6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("N (parameters)", color=INK2, fontsize=9)
    slope, icpt = np.polyfit(np.log(f), np.log(h), 1)
    resid = np.log(h) - (slope * np.log(f) + icpt)
    g = np.geomspace(f.min() / 1.3, f.max() * 1.3, 40)
    ax.plot(g, np.exp(icpt) * g ** slope, "--", color="#eb6834", linewidth=1.8,
            label=f"fit: GPU-h $\\propto$ C$^{{{slope:.2f}}}$, "
                  f"scatter x{np.exp(resid.std()):.2f}")
    ax.plot(g, np.exp(np.median(np.log(h) - np.log(f))) * g, ls=(0, (5, 3)),
            color=REF, linewidth=1.5, label="proportional (slope 1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("C (training FLOPs)", color=INK2, fontsize=9.5)
    ax.set_ylabel("measured training GPU-hours (first rate, cold read)",
                  color=INK2, fontsize=9.5)
    ax.set_title("A.  Do FLOPs track GPU time?", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    sec = ax.secondary_yaxis("right", functions=(lambda x: x * su_per_gpu_hour,
                                                 lambda x: x / su_per_gpu_hour))
    sec.set_ylabel(f"service units ({su_per_gpu_hour:g} SU per GPU-hour)",
                   color=INK2, fontsize=8.5)
    sec.tick_params(colors=INK2, labelsize=8)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[1]
    for which, col, lab in (("cold", "#2a78d6", "cold (first rate, the cost basis)"),
                            ("warm", "#86b6ef", "warm (later rates, cached)")):
        pts = [t for t in trials if t["cold"] == (which == "cold")]
        ax.scatter([t["n_params"] for t in pts], [t["ex_per_sec"] for t in pts],
                   s=14, color=col, alpha=0.45, edgecolor="none", zorder=2)
        med = [(float(k), v[which]) for k, v in table.items() if v.get(which)]
        if med:
            med.sort()
            ax.plot(*zip(*med), "-o", color=col, markersize=4, linewidth=1.6,
                    zorder=3, label=f"median, {lab}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (parameters)", color=INK2, fontsize=9.5)
    ax.set_ylabel("training examples per second (one l40s)", color=INK2,
                  fontsize=9.5)
    ax.set_title("B.  Throughput by model size", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    leg = ax.legend(fontsize=8, frameon=False, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("Measured GPU time per run (batch-loader pipeline)", fontsize=13,
                 color=INK, x=0.008, ha="left", y=1.02)
    fig.text(0.008, 0.955, f"{len(cold)} runs, {len(trials)} trials, from "
             "lr_grid.sh logs. Training time only: startup and the DeltaE "
             "evaluation are excluded.", fontsize=8.5, color=INK2, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")
    print(f"  GPU-hours ~ C^{slope:.2f} over {len(cold)} cold runs, "
          f"scatter x{np.exp(resid.std()):.2f} (1 sigma)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", default=["job-outputs"],
                   help="directories or files of indigo-lr-grid.*.out logs")
    p.add_argument("--output", default="analyses/scaling/results/tuned/run_times.json")
    p.add_argument("--figure", default=None,
                   help="default: run_times.png beside --output")
    p.add_argument("--su-per-gpu-hour", type=float, default=8.0)
    a = p.parse_args()

    files = []
    for src in a.logs:
        s = Path(src)
        files += ([s] if s.is_file() else
                  [Path(f) for f in sorted(glob.glob(str(s / "indigo-lr-grid.*.out")))])
    trials = [t for f in files for t in parse_log(f)]
    if not trials:
        sys.exit(f"no timed trials in {len(files)} log(s) with [loader] lines")
    table = rate_table(trials)
    print(f"{len(trials)} timed trials from {len(files)} logs")
    print(f"{'N':>11} {'cold ex/s':>10} {'warm ex/s':>10} {'runs':>5}")
    for n, v in table.items():
        c = f"{v['cold']:,.0f}" if v["cold"] else "-"
        w = f"{v['warm']:,.0f}" if v["warm"] else "-"
        print(f"{int(n):>11,} {c:>10} {w:>10} {v['n_cold']:>5}")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "scripts/collect_run_times.py",
               "su_per_gpu_hour": a.su_per_gpu_hour,
               "rates": table, "trials": trials}, open(out, "w"), indent=1)
    print(f"wrote {out}")
    plot(trials, table, Path(a.figure or out.with_suffix(".png")),
         a.su_per_gpu_hour)


if __name__ == "__main__":
    main()
