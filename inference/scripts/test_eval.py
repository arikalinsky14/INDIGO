#!/usr/bin/env python3
"""
Sweep evaluation of an INDIGO checkpoint against the held-out tier_a and
tier_b test sets.

For each test row:
  - extract the Lab target + the row's per-example material pool
  - run `inference.src.solve.solve()`
  - record achieved Lab + ΔE_00
  - for the first --plot-examples rows: render the composite PNG via
    `inference.src.visualize.render_result`

Output layout (under --output-dir / <checkpoint_tag> /):

    tier_a/
      summary.json     aggregate ΔE distribution + thresholds + runtime
      rows.json        per-row outcomes (compact dicts, for sweeping later)
      examples/        the first --plot-examples Result JSONs + PNGs
    tier_b/
      ... same shape ...

This script reuses the inference pipeline as-is — no model surgery — so
every code path that fires in production also fires here. ΔE numbers in
the summary correspond bit-for-bit to what an end user would see.

Conservatism choices to keep wall time reasonable
-------------------------------------------------
- Default --ensemble-n is 200 (vs production 500). The sequential JLL
  physics is the bottleneck; halving the ensemble is the simplest knob
  with minimal quality loss across so many trials.
- Default --tolerance 0 disables robustness scoring (R_l2 grad + MC) for
  the sweep. We only care about ΔE here; production tolerance behaviour
  is exercised by the demo runs.
- Failures (FeasibilityError, etc.) are recorded and the sweep continues
  — one bad row never aborts an evaluation.

Typical invocation
------------------
    python inference/scripts/test_eval.py \\
        --checkpoint data/checkpoints/<tag>/latest \\
        --test-dir data/test --tiers a b \\
        --limit 500 --plot-examples 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.dataset import FlexThinFilmDataset, find_repo_root
from src.materials_vocab import denormalize_lab, normalize_lab

from inference.src.generate import load_inference_model
from inference.src.schema import (
    InferenceKnobs, InferenceSpec, MaterialEntry,
)
from inference.src.solve import solve
from inference.src.visualize import render_result


# ----------------------------------------------------------------------------
# Conversions: src.dataset.TrainingExample → inference inputs
# ----------------------------------------------------------------------------

def _materialnk_to_entry(m) -> MaterialEntry:
    """Bridge src.material_features.MaterialNK ↔ inference MaterialEntry."""
    return MaterialEntry(
        canonical_name=m.name,
        n=np.asarray(m.n, dtype=np.float32),
        k=np.asarray(m.k, dtype=np.float32),
        source=getattr(m, "source", "test"),
    )


def _row_to_inputs(ex, base_seed: int, row_idx: int, knobs_template: InferenceKnobs
                   ) -> tuple:
    """Returns (pool, spec) ready to hand to solve()."""
    pool = [_materialnk_to_entry(m) for m in ex.pool]
    target_lab_raw = tuple(denormalize_lab(ex.lab))
    target_lab_norm = tuple(float(x) for x in ex.lab.tolist())

    # Per-row seed so each row's ensemble is deterministic AND distinct.
    knobs = InferenceKnobs(
        ensemble_N=knobs_template.ensemble_N,
        temperature=knobs_template.temperature,
        tolerance_pct=knobs_template.tolerance_pct,
        weight_lambda=knobs_template.weight_lambda,
        top_k=knobs_template.top_k,
        refine_max_iters=knobs_template.refine_max_iters,
        refine_step_size=knobs_template.refine_step_size,
        mc_samples=knobs_template.mc_samples,
        seed=base_seed + row_idx,
    )
    spec = InferenceSpec(
        target_lab_raw=target_lab_raw,
        target_lab_normalised=target_lab_norm,
        constraints=[],
        enforce_during=[],
        enforce_post=[],
        knobs=knobs,
        parsed_disclaimer="test_eval: no constraints, full pool from test row",
    )
    return pool, spec


# ----------------------------------------------------------------------------
# Per-tier sweep
# ----------------------------------------------------------------------------

# ΔE bucket thresholds used in the summary. CIE conventional reading:
#  ΔE < 1   imperceptible
#  ΔE < 2   just-perceptible
#  ΔE < 5   noticeable but acceptable
#  ΔE < 10  clearly different (still related)
_DELTA_E_BUCKETS = (1.0, 2.0, 5.0, 10.0)


def _aggregate_delta_e(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ΔE distribution stats + bucket counts. Skips rows that errored."""
    delta_es = [float(r["delta_e"]) for r in rows if "delta_e" in r]
    out: Dict[str, Any] = {
        "n_attempted": len(rows),
        "n_succeeded": len(delta_es),
        "n_failed": len(rows) - len(delta_es),
    }
    if not delta_es:
        out["delta_e"] = None
        out["buckets"] = None
        return out
    arr = np.asarray(delta_es, dtype=np.float64)
    out["delta_e"] = {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p95":    float(np.percentile(arr, 95)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }
    buckets: Dict[str, Any] = {}
    for t in _DELTA_E_BUCKETS:
        below = int((arr < t).sum())
        buckets[f"<{t}"] = {"count": below, "fraction": below / len(arr)}
    out["buckets"] = buckets
    return out


def evaluate_tier(
    tier_label: str,
    test_dir: Path,
    out_tier_dir: Path,
    model: torch.nn.Module,
    model_tag: str,
    model_sha256: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n{'=' * 70}\n[tier_{tier_label}] starting sweep over {test_dir}\n{'=' * 70}",
          flush=True)
    if not test_dir.exists():
        print(f"[tier_{tier_label}] WARNING: {test_dir} does not exist; skipping",
              flush=True)
        return {"tier": tier_label, "skipped": True, "reason": "missing dir"}

    examples_dir = out_tier_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    dataset = FlexThinFilmDataset(
        test_dir, seed=args.seed, split="train",
        verbose=False, streaming=True,
    )

    knobs_template = InferenceKnobs(
        ensemble_N=args.ensemble_n,
        temperature=args.temperature,
        tolerance_pct=args.tolerance,
        weight_lambda=args.weight_lambda,
        top_k=args.top_k,
        refine_max_iters=args.refine_iters,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )

    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    next_log = max(args.log_every, 1)
    for row_idx, ex in enumerate(itertools.islice(dataset, args.limit)):
        pool, spec = _row_to_inputs(ex, args.seed, row_idx, knobs_template)

        try:
            result = solve(
                model=model, pool=pool, spec=spec,
                model_tag=model_tag, model_sha256=model_sha256,
            )
        except Exception as exc:
            rows.append({"row_idx": row_idx,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"[tier_{tier_label}] row {row_idx}: solve() raised "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue

        rec: Dict[str, Any] = {
            "row_idx": row_idx,
            "target_lab": list(spec.target_lab_raw),
            "n_pool": len(pool),
            "ground_truth_n_layers": len(ex.target_slots),
        }
        if result.chosen is None:
            rec["error"] = "; ".join(result.errors) or "no chosen candidate"
            rec["dropped_per_constraint"] = result.ensemble_stats.dropped_per_constraint
        else:
            c = result.chosen
            rec.update({
                "achieved_lab": list(c.achieved_lab),
                "delta_e":      float(c.delta_e),
                "n_layers":     len(c.slot_indices),
                "layers":       [{"material": m, "thickness_nm": float(t)}
                                 for m, t in zip(c.material_names,
                                                 c.thicknesses_nm)],
                "objective":    float(c.objective),
                "refined":      bool(c.refined),
            })
        rows.append(rec)

        # Visualization: only the first N succeed-or-fail Results.
        if row_idx < args.plot_examples:
            ex_out = examples_dir / f"row_{row_idx:04d}"
            ex_out_json = ex_out.with_suffix(".json")
            ex_out_png  = ex_out.with_suffix(".png")
            try:
                result.to_json(path=ex_out_json)
                render_result(result, ex_out_png)
            except Exception as exc:
                print(f"[tier_{tier_label}] row {row_idx}: render failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)

        # Periodic progress + rolling ΔE stat.
        if (row_idx + 1) >= next_log:
            ok = [float(r["delta_e"]) for r in rows if "delta_e" in r]
            de_str = (f"median ΔE = {np.median(ok):.3f}, p95 = {np.percentile(ok, 95):.3f}"
                      if ok else "no successes yet")
            rate = (row_idx + 1) / max(time.time() - t_start, 1e-6)
            print(f"[tier_{tier_label}] {row_idx + 1}/{args.limit} done  "
                  f"{de_str}  ({rate:.2f} rows/s)", flush=True)
            next_log += args.log_every

    runtime = time.time() - t_start
    summary = _aggregate_delta_e(rows)
    summary.update({
        "tier": tier_label,
        "model_tag": model_tag,
        "model_sha256": model_sha256,
        "runtime_seconds": runtime,
        "rows_per_second": len(rows) / max(runtime, 1e-6),
        "args": {
            "ensemble_n": args.ensemble_n,
            "temperature": args.temperature,
            "tolerance": args.tolerance,
            "weight_lambda": args.weight_lambda,
            "top_k": args.top_k,
            "refine_iters": args.refine_iters,
            "mc_samples": args.mc_samples,
            "plot_examples": args.plot_examples,
            "seed": args.seed,
        },
    })

    (out_tier_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_tier_dir / "rows.json").write_text(json.dumps(rows, indent=2))

    print(f"\n[tier_{tier_label}] DONE in {runtime:.1f}s  "
          f"({len(rows)} rows; {summary['n_succeeded']} succeeded)",
          flush=True)
    if summary.get("delta_e"):
        de = summary["delta_e"]
        print(f"[tier_{tier_label}]   ΔE: median={de['median']:.3f}  "
              f"mean={de['mean']:.3f}  p95={de['p95']:.3f}  "
              f"min={de['min']:.3f}  max={de['max']:.3f}", flush=True)
        bs = summary["buckets"]
        line = "  ".join(f"{k}: {v['fraction']*100:.1f}%"
                         for k, v in bs.items())
        print(f"[tier_{tier_label}]   ΔE buckets: {line}", flush=True)

    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="INDIGO test-set sweep evaluation")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to a saved checkpoint dir, e.g. "
                        "data/checkpoints/<tag>/latest")
    p.add_argument("--test-dir", type=str, default=None,
                   help="Root containing tier_a/ and tier_b/ subdirs "
                        "(default: <repo>/data/test).")
    p.add_argument("--tiers", nargs="+", default=["a", "b"],
                   choices=["a", "b"],
                   help="Which tier(s) to evaluate; default both.")
    p.add_argument("--limit", type=int, default=500,
                   help="Max rows per tier (default 500).")
    p.add_argument("--plot-examples", type=int, default=20,
                   help="Number of rows per tier to render as composite PNGs "
                        "(default 20).")
    # Knobs — sweep defaults differ from production for time:
    p.add_argument("--ensemble-n", type=int, default=200,
                   help="Per-row ensemble size (default 200; production uses 500).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="0 disables robustness scoring for the sweep "
                        "(default). Set non-zero to include MC robustness.")
    p.add_argument("--lambda", dest="weight_lambda", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1,
                   help="Only the top candidate is reported by the sweep "
                        "(default 1).")
    p.add_argument("--refine-iters", type=int, default=80)
    p.add_argument("--mc-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Root for evaluation outputs "
                        "(default: <repo>/inference/outputs/test_eval).")
    p.add_argument("--log-every", type=int, default=25,
                   help="Print a rolling ΔE summary every N rows (default 25).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _root

    test_root = Path(args.test_dir) if args.test_dir else repo_root / "data" / "test"
    out_root = (Path(args.output_dir) if args.output_dir
                else repo_root / "inference" / "outputs" / "test_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")
    print(f"[eval] loading model from {args.checkpoint}")
    model, config, sha = load_inference_model(Path(args.checkpoint), device=device)
    model_tag = config.tag()
    print(f"[eval] model tag:    {model_tag}")
    print(f"[eval] model sha256: {sha}")

    # Output gets per-checkpoint partitioning so two runs against different
    # checkpoints land in distinct dirs.
    out_dir = out_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] writing results under: {out_dir}")

    tier_summaries: Dict[str, Any] = {}
    for tier in args.tiers:
        tier_label = tier
        tier_data_dir = test_root / f"tier_{tier_label}"
        tier_out_dir = out_dir / f"tier_{tier_label}"
        summary = evaluate_tier(
            tier_label=tier_label, test_dir=tier_data_dir,
            out_tier_dir=tier_out_dir,
            model=model, model_tag=model_tag, model_sha256=sha,
            args=args,
        )
        tier_summaries[f"tier_{tier_label}"] = summary

    # Top-level summary across all tiers in this run.
    overall = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_tag": model_tag,
        "checkpoint_sha256": sha,
        "tiers": tier_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\n[eval] overall summary written to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
