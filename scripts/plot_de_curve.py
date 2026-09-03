#!/usr/bin/env python3
"""
Plot val / tier-A / tier-B ΔE₀₀ (CIEDE2000) vs training step from the
per-checkpoint JSONs produced by slurms/de_curve.sh.

One --input-dir per split; overlay them on the same axes. `--label` is
optional; falls back to the directory basename.

If the eval JSONs carry a `by_source` block (produced when the parquet
has a `structure_source` column), each split's overall line is
supplemented with per-source lines: high_chroma_search (dashed) and
random (dotted), same colour as the overall line for that split. Old
JSONs without `by_source` render only the overall line.

Emits <output>.png (default: <first input-dir>/de_curve.png).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STEP_RX = re.compile(r"eval_step_(\d+)\.json$")


@dataclass
class Series:
    steps: List[int]
    mean: List[float]
    median: List[float]
    p75: List[float]


def _empty_series() -> Series:
    return Series(steps=[], mean=[], median=[], p75=[])


def _load_dir(indir: Path) -> Tuple[Series, Dict[str, Series]]:
    """Return (overall, per_source_dict) sorted by step.

    per_source_dict is keyed by source name ('high_chroma_search',
    'random', ...); may be empty if none of the JSONs carried a
    by_source block.
    """
    rows_overall: List[Tuple[int, float, float, float]] = []
    rows_by_source: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for path_str in glob.glob(str(indir / "eval_step_*.json")):
        m = _STEP_RX.search(path_str)
        if not m:
            continue
        step = int(m.group(1))
        try:
            payload = json.loads(Path(path_str).read_text())
        except (OSError, ValueError) as exc:
            print(f"[warn] skipping {path_str}: {exc}")
            continue
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        mean = metrics.get("ciede2000_mean")
        if mean is None:
            print(f"[warn] {path_str}: no ciede2000_mean (evaluate.py ran "
                  f"without optical sim?)")
            continue
        median = metrics.get("ciede2000_median")
        q3 = metrics.get("ciede2000_q3")
        rows_overall.append((
            step, float(mean),
            float(median) if median is not None else float("nan"),
            float(q3) if q3 is not None else float("nan"),
        ))
        for src, sm in (metrics.get("by_source") or {}).items():
            src_mean = sm.get("ciede2000_mean")
            if src_mean is None:
                continue
            rows_by_source.setdefault(src, []).append((
                step, float(src_mean),
                float(sm.get("ciede2000_median", "nan")),
                float(sm.get("ciede2000_q3", "nan")),
            ))
    rows_overall.sort()
    overall = Series(
        steps=[r[0] for r in rows_overall],
        mean=[r[1] for r in rows_overall],
        median=[r[2] for r in rows_overall],
        p75=[r[3] for r in rows_overall],
    )
    by_source: Dict[str, Series] = {}
    for src, rows in rows_by_source.items():
        rows.sort()
        by_source[src] = Series(
            steps=[r[0] for r in rows],
            mean=[r[1] for r in rows],
            median=[r[2] for r in rows],
            p75=[r[3] for r in rows],
        )
    return overall, by_source


# Line style per source. Overall uses solid; sources use dashed/dotted.
_SOURCE_LINESTYLE = {
    "high_chroma_search": (0, (4, 2)),   # dashed
    "random":             (0, (1, 2)),   # dotted
}


def _source_label(src: str) -> str:
    return {"high_chroma_search": "HC", "random": "random"}.get(src, src)


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
    ap.add_argument("--no-split-by-source", action="store_true",
                    help="Suppress the per-source dashed/dotted lines even "
                         "when the JSONs carry a by_source block.")
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.input_dir):
        raise SystemExit(
            f"--label count ({len(args.label)}) must match "
            f"--input-dir count ({len(args.input_dir)}) if any labels given")

    fig, (ax_mean, ax_p75) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    n_series = 0
    for i, indir in enumerate(args.input_dir):
        label = (args.label[i] if args.label else indir.name)
        overall, by_source = _load_dir(indir)
        if not overall.steps:
            print(f"[warn] no data under {indir} — skipping")
            continue
        n_series += 1
        color = f"C{i}"

        # Overall lines — solid, bold.
        ax_mean.plot(overall.steps, overall.mean, "o-",
                     color=color, lw=1.8, label=f"{label} · overall (mean)")
        ax_mean.plot(overall.steps, overall.median, "s--",
                     color=color, alpha=0.55, lw=1.0,
                     label=f"{label} · overall (median)")
        ax_p75.plot(overall.steps, overall.p75, "^-",
                    color=color, lw=1.8, label=f"{label} · overall (p75)")

        # Final-value annotation on the overall mean.
        ax_mean.annotate(f"{overall.mean[-1]:.2f}",
                         xy=(overall.steps[-1], overall.mean[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=8, va="center", color=color)

        # Per-source lines — thin, dashed/dotted by source, same colour.
        if not args.no_split_by_source:
            for src, ser in by_source.items():
                if not ser.steps:
                    continue
                ls = _SOURCE_LINESTYLE.get(src, (0, (3, 1, 1, 1)))
                slab = _source_label(src)
                ax_mean.plot(ser.steps, ser.mean, linestyle=ls,
                             color=color, lw=1.1, alpha=0.85,
                             label=f"{label} · {slab} (mean)")
                ax_p75.plot(ser.steps, ser.p75, linestyle=ls,
                            color=color, lw=1.1, alpha=0.85,
                            label=f"{label} · {slab} (p75)")

    if n_series == 0:
        raise SystemExit("No input dirs produced any data.")

    ax_mean.set_ylabel("ΔE₀₀ (mean / median)")
    ax_mean.set_title(args.title)
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=7, ncol=2)
    ax_mean.axhline(2.0, color="k", lw=0.5, linestyle=":", alpha=0.5)
    ax_mean.axhline(3.0, color="k", lw=0.5, linestyle=":", alpha=0.5)

    ax_p75.set_xlabel("training step")
    ax_p75.set_ylabel("ΔE₀₀ p75")
    ax_p75.grid(alpha=0.3)
    ax_p75.legend(fontsize=7, ncol=2)

    plt.tight_layout()

    output = args.output or (args.input_dir[0] / "de_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {output}")


if __name__ == "__main__":
    main()
