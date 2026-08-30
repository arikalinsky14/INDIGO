#!/usr/bin/env python3
"""
Plot train + val loss (top) and LR schedule (bottom) from a run's
history.jsonl, produced by scripts/training.py's per-save hook.

One --history per line-series to overlay (e.g. multiple runs). --label
optional; falls back to the parent directory name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> List[dict]:
    """Parse a history.jsonl into a list of dicts, one per checkpoint."""
    rows: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                print(f"[warn] {path}: skipping malformed line: {exc}")
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", action="append", required=True, type=Path,
                    help="Path to a history.jsonl file. Repeat for each "
                         "run you want overlaid.")
    ap.add_argument("--label", action="append", default=[],
                    help="Legend label per --history, in the same order. "
                         "Falls back to the parent directory name.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG path. Default: "
                         "<first history's parent>/training_curve.png.")
    ap.add_argument("--title", type=str,
                    default="INDIGO training curve")
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.history):
        raise SystemExit(
            f"--label count ({len(args.label)}) must match "
            f"--history count ({len(args.history)}) if any labels given")

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, sharex=True, figsize=(9, 7),
                                          gridspec_kw={"height_ratios": [3, 1]})

    for i, path in enumerate(args.history):
        label = (args.label[i] if args.label else path.parent.name)
        rows = _load(path)
        if not rows:
            print(f"[warn] no rows in {path} — skipping")
            continue
        steps = [r["step"] for r in rows]
        tr = [r.get("train_loss") for r in rows]
        va = [r.get("val_loss") for r in rows]
        lr = [r.get("lr") for r in rows]
        ax_loss.plot(steps, tr, "o-", label=f"{label} train")
        ax_loss.plot(steps, va, "s--", label=f"{label} val", alpha=0.8)
        ax_lr.plot(steps, lr, ".-", label=label, alpha=0.7)
        # Annotate final train/val.
        if tr and tr[-1] is not None:
            ax_loss.annotate(f"tr={tr[-1]:.3f}",
                             xy=(steps[-1], tr[-1]),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=8, va="center")
        if va and va[-1] is not None:
            ax_loss.annotate(f"va={va[-1]:.3f}",
                             xy=(steps[-1], va[-1]),
                             xytext=(4, -10), textcoords="offset points",
                             fontsize=8, va="center", color="tab:orange")

    ax_loss.set_ylabel("loss")
    ax_loss.set_title(args.title)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_lr.set_xlabel("training step")
    ax_lr.set_ylabel("LR")
    ax_lr.grid(alpha=0.3)
    if len(args.history) > 1:
        ax_lr.legend(fontsize=8)

    plt.tight_layout()

    output = args.output or (args.history[0].parent / "training_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {output}")


if __name__ == "__main__":
    main()
