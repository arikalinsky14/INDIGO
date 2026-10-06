"""Join a learning-rate grid that ran as several array tasks into one cell file.

A long cell (stage 2's smallest model on its top curve trains 18.5M examples
per rate, seven rates one after another) would need a ~19 hour job. The SLURM
wrapper therefore splits such a cell's grid into contiguous runs of rates, each
its own array task, writing

    <parts-dir>/lr_search_<tag>_part<k0>-<k1>of<n>.json

where k0..k1 are indices into the cell's n-point grid. Once every index is
present, this writes <out-dir>/lr_search_<tag>.json in exactly the schema
lr_tuning.py writes for an unsplit cell, with the optimum re-selected over all
the rates by lr_tuning.py's own rule. Everything downstream (fit_lr_law.py,
collect_isoflop.py, stage 3's lookup of stage 2's rates) reads that file and
cannot tell the cell was split.

Each part runs the same model, seed, data subset and evaluation slice; only
the rates differ. A split cell is therefore the same experiment as an unsplit
one, give or take the GPU nondeterminism every run has anyway.

Every array task calls this when it finishes, so the last part of a cell to
land writes the cell. It is safe to run any number of times, concurrently, or
by hand: an incomplete cell is reported and left alone, a finished cell file is
never rewritten unless --force, and the write is atomic.

    python scripts/merge_lr_parts.py \
        --parts-dir outputs/lr_search/cross_attn/parts \
        --out-dir outputs/lr_search/cross_attn

Imports no torch, like the rest of the analysis stack.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.materials_vocab import VOCAB_SIZE  # noqa: E402

#: Same screen as lr_tuning.DIVERGENCE_VAL_LOSS: worse than uniform over the
#: vocabulary. Repeated rather than imported because lr_tuning imports torch.
DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)

PART_RE = re.compile(r"^(lr_search_.+)_part(\d+)-(\d+)of(\d+)\.json$")

#: Fields that must agree across the parts of one cell. Anything else that
#: differs (lr_range, the per-part optimum) is expected to.
MUST_MATCH = ("epochs", "seed", "limit_shard_aligned", "selection_metric",
              "limit_de_examples", "head_mode", "n_heads", "d_model",
              "n_layers", "slot_encoder_layers", "decoder_layers", "n_params",
              "batch_size", "beta1", "beta2", "lr_schedule", "weight_decay",
              "grad_clip", "warmup_fraction", "train_examples",
              "val_examples", "limit_examples", "limit_val_examples")


def part_suffix(k0: int, k1: int, n: int) -> str:
    """The suffix the SLURM wrapper passes to lr_tuning.py --output-suffix."""
    return f"_part{k0}-{k1}of{n}"


def _diverged(r: dict) -> bool:
    vl = r.get("best_val_loss")
    return vl is None or not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS


def _score(r: dict, metric: str) -> float:
    """lr_tuning.LRSearchResult.selection_metric, on the saved dict."""
    if _diverged(r):
        return float("inf")
    if metric == "delta_e":
        v = r.get("final_val_de")
        return float("inf") if v is None else v
    return r["best_val_loss"]


def select(results: List[dict], metric: str) -> dict:
    """lr_tuning.lr_tuning's choice of optimum, including both fallbacks."""
    if all(_diverged(r) for r in results):
        return min(results, key=lambda r: r["lr"])
    if metric == "delta_e" and all(r.get("final_val_de") is None
                                   for r in results if not _diverged(r)):
        metric = "val_loss"
    return min(results, key=lambda r: _score(r, metric))


def merge(parts: List[dict]) -> dict:
    """One cell's parts, already known to cover its grid, as one result."""
    base = dict(parts[0])
    for p in parts[1:]:
        for k in MUST_MATCH:
            if p.get(k) != base.get(k):
                raise ValueError(f"parts disagree on {k}: "
                                 f"{base.get(k)!r} vs {p.get(k)!r}")
    results = sorted((r for p in parts for r in p["results"]),
                     key=lambda r: r["lr"])
    best = select(results, base["selection_metric"])
    base.update({
        "optimal_lr": best["lr"],
        "optimal_val_de": best.get("final_val_de"),
        "optimal_val_de_p95": best.get("final_val_de_p95"),
        "optimal_val_de_by_chroma": best.get("final_val_de_by_chroma"),
        "val_loss_would_pick_lr": min(results,
                                      key=lambda r: r["best_val_loss"])["lr"],
        "lr_range": [results[0]["lr"], results[-1]["lr"]],
        "n_lrs": len(results),
        "results": results,
        "merged_from_parts": len(parts),
    })
    return base


def merge_dir(parts_dir: Path, out_dir: Path, force: bool = False,
              verbose: bool = True) -> Dict[str, str]:
    """Merge every complete cell under parts_dir. Returns {cell: status}."""
    groups: Dict[tuple, Dict[int, Path]] = defaultdict(dict)
    for path in sorted(parts_dir.glob("lr_search_*_part*of*.json")):
        m = PART_RE.match(path.name)
        if not m:
            continue
        stem, k0, k1, n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        for k in range(k0, k1 + 1):
            groups[(stem, n)][k] = path
    status: Dict[str, str] = {}
    for (stem, n), have in sorted(groups.items()):
        out = out_dir / f"{stem}.json"
        missing = sorted(set(range(n)) - set(have))
        if missing:
            status[stem] = f"waiting on grid points {missing} of {n}"
        elif out.exists() and not force:
            status[stem] = "already merged"
        else:
            parts = [json.load(open(p)) for p in sorted(set(have.values()))]
            merged = merge(parts)
            if merged["n_lrs"] != n:
                status[stem] = (f"ERROR: parts hold {merged['n_lrs']} rates, "
                                f"grid has {n}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp, out)
            status[stem] = (f"merged {len(parts)} parts -> {out} "
                            f"(optimum {merged['optimal_lr']:.3e})")
        if verbose:
            print(f"[merge] {stem}: {status[stem]}")
    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true",
                   help="rewrite a cell file that already exists")
    a = p.parse_args()
    status = merge_dir(Path(a.parts_dir), Path(a.out_dir), a.force)
    if any(s.startswith("ERROR") for s in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
