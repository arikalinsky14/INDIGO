#!/usr/bin/env python3
"""
Plot val / tier-A / tier-B ΔE₀₀ (CIEDE2000) vs training step from the
per-checkpoint JSONs produced by slurms/de_curve.sh.

One --input-dir per split; overlay them on the same axes. `--label` is
optional; falls back to the directory basename.

Emits <output>.png (default: <first input-dir>/de_curve.png).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STEP_RX = re.compile(r"eval_step_(\d+)\.json$")


def _load_dir(indir: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Return (steps, mean, median, p75) sorted by step, skipping unreadable files."""
    rows: List[Tuple[int, float, float, float]] = []
    for path_str in glob.glob(str(indir / "eval_step_*.json")):
        m = _STEP_RX.search(path_str)
        if not m:
            continue
        step = int(m.group(1))
        try:
            payload = json.loads(Path(path_str).read_text())
            metrics = payload.get("metrics", {})
            mean = metrics.get("ciede2000_mean")
            median = metrics.get("ciede2000_median")
            q3 = metrics.get("ciede2000_q3")
        except (OSError, ValueError, KeyError) as exc:
            print(f"[warn] skipping {path_str}: {exc}")
            continue
        if mean is None:
            print(f"[warn] {path_str}: no ciede2000_mean (evaluate.py ran "
                  f"without optical sim?)")
            continue
        rows.append((step, float(mean),
                     float(median) if median is not None else float("nan"),
                     float(q3) if q3 is not None else float("nan")))
    rows.sort()
    steps = [r[0] for r in rows]
    means = [r[1] for r in rows]
    medians = [r[2] for r in rows]
    p75s = [r[3] for r in rows]
    return steps, means, medians, p75s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", action="append", required=True, type=Path,
                    help="A directory of eval_step_*.json files. Repeat for "
                         "each split you want overlaid (val / tier_a / tier_b).")
    ap.add_argument("--label", action="append", default=[],
                    help="Legend label per input-dir, in the same order. "
                         "Falls back to the directory basename if omitted.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG path. Default: <first input-dir>/de_curve.png.")
    ap.add_argument("--title", type=str,
                    default="INDIGO training curve: ΔE₀₀ vs step")
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.input_dir):
        raise SystemExit(
            f"--label count ({len(args.label)}) must match "
            f"--input-dir count ({len(args.input_dir)}) if any labels given")

    fig, (ax_mean, ax_p75) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    n_series = 0
    for i, indir in enumerate(args.input_dir):
        label = (args.label[i] if args.label else indir.name)
        steps, means, medians, p75s = _load_dir(indir)
        if not steps:
            print(f"[warn] no data under {indir} — skipping")
            continue
        n_series += 1
        ax_mean.plot(steps, means, "o-", label=f"{label} (mean)")
        ax_mean.plot(steps, medians, "s--", alpha=0.6,
                     label=f"{label} (median)")
        ax_p75.plot(steps, p75s, "^-", label=f"{label} (p75)")
        # Final-value annotation on mean curve.
        ax_mean.annotate(f"{means[-1]:.2f}",
                         xy=(steps[-1], means[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=8, va="center")

    if n_series == 0:
        raise SystemExit("No input dirs produced any data.")

    ax_mean.set_ylabel("ΔE₀₀ (mean / median)")
    ax_mean.set_title(args.title)
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=8)
    ax_mean.axhline(2.0, color="k", lw=0.5, linestyle=":", alpha=0.5)
    ax_mean.axhline(3.0, color="k", lw=0.5, linestyle=":", alpha=0.5)

    ax_p75.set_xlabel("training step")
    ax_p75.set_ylabel("ΔE₀₀ p75")
    ax_p75.grid(alpha=0.3)
    ax_p75.legend(fontsize=8)

    plt.tight_layout()

    output = args.output or (args.input_dir[0] / "de_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {output}")


if __name__ == "__main__":
    main()
