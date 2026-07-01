#!/usr/bin/env python3
"""
Gamut evaluation for a production INDIGO checkpoint.

Runs `solve()` over a fixed set of test batteries designed to expose
different failure modes of the RGB → thin-film pipeline:

  rgb_primaries       — 8 sRGB cube corners in Lab (black, white,
                        R, G, B, C, M, Y). "Can we hit the common
                        colors users think in?"
  lab_l_sweep         — pure neutral axis at a=b=0, sweeping L in
                        {5, 20, 40, 60, 80, 95}. Isolates lightness.
  lab_a_sweep         — L=50, b=0, a in {-80, -40, 0, 40, 80}. Green
                        ↔ red hue axis, chroma isolated.
  lab_b_sweep         — L=50, a=0, b in {-80, -40, 0, 40, 80}. Blue
                        ↔ yellow.
  chromatic_corners   — L=50, (a,b) in {(±80, ±80)}. Off-axis
                        high-chroma. Some points sit outside the
                        reachable sRGB gamut on purpose to measure
                        graceful degradation.

For every target the script records:
  - target_lab_raw
  - achieved_lab
  - delta_e_00      ← the metric the user cares about
  - n_layers        (how deep the chosen structure ended up)
  - wall_time_s     (per-request; useful for CI budgets)

Aggregate stats per battery: median / mean / p95 / worst ΔE, plus
pass rates at ΔE thresholds {1, 2, 5, 10}. Overall row rolls those
up across batteries.

CLI
---
Minimal:
    python -m inference.scripts.gamut_eval \\
        --checkpoint data/checkpoints/<tag>/latest \\
        --output outputs/gamut_eval.json

A/B the optimizer (runs every target twice — once with DoG, once with
Adam — emits both trajectories side-by-side):
    python -m inference.scripts.gamut_eval \\
        --checkpoint ... --optimizer both --output outputs/gamut_ab.json

The output JSON layout is stable enough for CI regression tracking:
    {
      "checkpoint": "...",
      "preset": "balanced",
      "optimizer": "dog",
      "batteries": {
        "rgb_primaries": {
          "n": 8,
          "stats": {
              "median_de": ..., "mean_de": ..., "p95_de": ...,
              "worst_de": ..., "worst_target": [L,a,b],
              "pass_rate_de_lt_1": ..., "..._lt_2": ..., "..._lt_5": ...,
              "..._lt_10": ...
          },
          "per_target": [{"target_lab": ..., "achieved_lab": ...,
                          "delta_e": ..., "n_layers": ...,
                          "wall_time_s": ...}, ...]
        },
        ...
      },
      "overall": {"n": N, "stats": {...}},
      // When --optimizer both:
      "ab_comparison": {
        "dog":  {"batteries": {...}, "overall": {...}},
        "adam": {"batteries": {...}, "overall": {...}},
      }
    }
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ----------------------------------------------------------------------------
# Test batteries
# ----------------------------------------------------------------------------

def _srgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """sRGB [0..1] → CIE Lab, mirroring the frontend JS conversion so the
    corners we test are exactly the ones a user picks with a colour swatch.
    """
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = lin(r), lin(g), lin(b)
    X = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) * 100.0
    Y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) * 100.0
    Z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) * 100.0
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    delta = 6.0 / 29.0

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > delta ** 3 else t / (3 * delta * delta) + 4.0 / 29.0
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def _rgb_cube_corners() -> List[Tuple[float, float, float]]:
    """Eight sRGB cube corners in Lab: R, G, B, C, M, Y, black, white."""
    corners_rgb = [
        (0.0, 0.0, 0.0),  # black
        (1.0, 1.0, 1.0),  # white
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.0, 1.0),  # magenta
        (1.0, 1.0, 0.0),  # yellow
    ]
    return [_srgb_to_lab(*c) for c in corners_rgb]


def _lab_l_sweep() -> List[Tuple[float, float, float]]:
    return [(L, 0.0, 0.0) for L in (5.0, 20.0, 40.0, 60.0, 80.0, 95.0)]


def _lab_a_sweep() -> List[Tuple[float, float, float]]:
    return [(50.0, a, 0.0) for a in (-80.0, -40.0, 0.0, 40.0, 80.0)]


def _lab_b_sweep() -> List[Tuple[float, float, float]]:
    return [(50.0, 0.0, b) for b in (-80.0, -40.0, 0.0, 40.0, 80.0)]


def _chromatic_corners() -> List[Tuple[float, float, float]]:
    return [
        (50.0, +80.0, +80.0),  # deep orange
        (50.0, +80.0, -80.0),  # magenta-ish
        (50.0, -80.0, +80.0),  # yellow-green
        (50.0, -80.0, -80.0),  # teal
    ]


BATTERIES: Dict[str, List[Tuple[float, float, float]]] = {
    "rgb_primaries":    _rgb_cube_corners(),
    "lab_l_sweep":      _lab_l_sweep(),
    "lab_a_sweep":      _lab_a_sweep(),
    "lab_b_sweep":      _lab_b_sweep(),
    "chromatic_corners": _chromatic_corners(),
}


# ----------------------------------------------------------------------------
# Presets — mirror the frontend's fast / balanced / best knob recipes so the
# eval numbers match what a user would see clicking the same preset in the UI.
# ----------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast":     dict(ensemble_N=100, top_k=1, refine_top_n=1,
                     refine_max_iters=15, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    "balanced": dict(ensemble_N=150, top_k=3, refine_top_n=1,
                     refine_max_iters=25, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    "best":     dict(ensemble_N=500, top_k=5, refine_top_n=3,
                     refine_max_iters=80, tolerance_pct=5.0,
                     weight_lambda=1.0, mc_samples=16, temperature=1.0),
}


# ----------------------------------------------------------------------------
# Per-target runner
# ----------------------------------------------------------------------------

def _run_one(
    solve_fn, model, pool, model_tag, model_sha, device,
    target_lab: Tuple[float, float, float],
    preset_knobs: Dict[str, Any],
    optimizer: str,
    base_seed: int,
) -> Dict[str, Any]:
    from inference.src.schema import InferenceKnobs, InferenceSpec
    from src.materials_vocab import normalize_lab

    knobs = InferenceKnobs(
        seed=base_seed,
        refine_optimizer=optimizer,
        **preset_knobs,
    )
    norm = tuple(float(x) for x in normalize_lab(list(target_lab)).tolist())
    spec = InferenceSpec(
        target_lab_raw=tuple(float(x) for x in target_lab),
        target_lab_normalised=norm,
        constraints=[],
        enforce_during=[], enforce_post=[],
        knobs=knobs,
        parsed_disclaimer=f"gamut_eval target={target_lab}",
    )
    t0 = time.time()
    try:
        result = solve_fn(
            model=model, pool=pool, spec=spec,
            model_tag=model_tag, model_sha256=model_sha, device=device,
        )
    except Exception as exc:
        return {
            "target_lab": list(target_lab),
            "achieved_lab": None,
            "delta_e": None,
            "n_layers": 0,
            "wall_time_s": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    wall = time.time() - t0
    if result.chosen is None:
        return {
            "target_lab": list(target_lab),
            "achieved_lab": None,
            "delta_e": None,
            "n_layers": 0,
            "wall_time_s": wall,
            "error": "; ".join(result.errors) or "no chosen candidate",
        }
    c = result.chosen
    return {
        "target_lab": list(target_lab),
        "achieved_lab": list(c.achieved_lab),
        "delta_e": float(c.delta_e),
        "n_layers": len(c.slot_indices),
        "wall_time_s": wall,
        "error": None,
    }


# ----------------------------------------------------------------------------
# Aggregate stats
# ----------------------------------------------------------------------------

def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    des = [r["delta_e"] for r in rows if r.get("delta_e") is not None]
    if not des:
        return {"n": len(rows), "median_de": None, "mean_de": None,
                "p95_de": None, "worst_de": None, "worst_target": None,
                "pass_rate_de_lt_1": None, "pass_rate_de_lt_2": None,
                "pass_rate_de_lt_5": None, "pass_rate_de_lt_10": None,
                "n_failed": len(rows)}
    des_sorted = sorted(des)
    p95_idx = min(len(des_sorted) - 1, int(round(0.95 * (len(des_sorted) - 1))))
    worst_row = max(
        (r for r in rows if r.get("delta_e") is not None),
        key=lambda r: r["delta_e"],
    )

    def pass_rate(thresh: float) -> float:
        hits = sum(1 for d in des if d < thresh)
        return hits / len(des)

    return {
        "n": len(rows),
        "n_failed": sum(1 for r in rows if r.get("delta_e") is None),
        "median_de": statistics.median(des),
        "mean_de":   statistics.mean(des),
        "p95_de":    des_sorted[p95_idx],
        "worst_de":  worst_row["delta_e"],
        "worst_target": worst_row["target_lab"],
        "pass_rate_de_lt_1":  pass_rate(1.0),
        "pass_rate_de_lt_2":  pass_rate(2.0),
        "pass_rate_de_lt_5":  pass_rate(5.0),
        "pass_rate_de_lt_10": pass_rate(10.0),
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def _run_all_batteries(
    solve_fn, model, pool, model_tag, model_sha, device,
    preset_knobs: Dict[str, Any],
    optimizer: str,
    base_seed: int,
) -> Dict[str, Any]:
    out_batteries: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for name, targets in BATTERIES.items():
        rows = []
        for i, tgt in enumerate(targets):
            per_seed = base_seed + hash(name) % 1_000_000 + i
            row = _run_one(
                solve_fn, model, pool, model_tag, model_sha, device,
                tgt, preset_knobs, optimizer, per_seed,
            )
            rows.append(row)
            de_str = ("—" if row["delta_e"] is None
                      else f"{row['delta_e']:6.2f}")
            print(f"  [{optimizer:>4}][{name:<18}] tgt={tgt!r}  "
                  f"ΔE={de_str}  t={row['wall_time_s']:.1f}s",
                  flush=True)
        out_batteries[name] = {
            "stats": _summarise(rows),
            "per_target": rows,
        }
        all_rows.extend(rows)
    return {
        "batteries": out_batteries,
        "overall": {"stats": _summarise(all_rows)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Checkpoint dir (data/checkpoints/<tag>/latest).")
    p.add_argument("--pool-dir", type=str, default=None,
                   help="JLL materials dir. Defaults to installed package.")
    p.add_argument("--preset", choices=("fast", "balanced", "best"),
                   default="balanced")
    p.add_argument("--optimizer", choices=("dog", "adam", "both"),
                   default="dog",
                   help="Refinement optimizer. 'both' runs every target "
                        "twice for a side-by-side A/B in the output JSON.")
    p.add_argument("--output", required=True, type=str,
                   help="Where to write the aggregated JSON report.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed. Per-battery per-index seeds derive from it.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is technically present.")
    args = p.parse_args()

    # Lazy imports so `--help` is fast.
    from inference.scripts.run_inference import (
        cap_pool_at_m_max, default_jll_materials_dir, load_pool_from_jll,
    )
    from inference.src.generate import load_inference_model
    from inference.src.solve import solve as solve_fn
    from src.materials_vocab import M_MAX

    import torch
    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[gamut_eval] device={device}", flush=True)

    model, config, sha = load_inference_model(
        Path(args.checkpoint), device=device,
    )
    tag = config.tag()
    print(f"[gamut_eval] checkpoint={tag}  sha={sha}", flush=True)

    pool_dir = Path(args.pool_dir) if args.pool_dir else default_jll_materials_dir()
    full = load_pool_from_jll(pool_dir)
    pool = cap_pool_at_m_max(full, M_MAX)
    print(f"[gamut_eval] pool={len(pool)}/{len(full)} materials from {pool_dir}",
          flush=True)

    preset_knobs = PRESETS[args.preset]
    print(f"[gamut_eval] preset={args.preset}  optimizer={args.optimizer}  "
          f"seed={args.seed}", flush=True)

    result_root: Dict[str, Any] = {
        "checkpoint": tag,
        "model_sha256": sha,
        "preset": args.preset,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "pool_size": len(pool),
    }

    if args.optimizer == "both":
        # A/B: run DoG and Adam on the same seeds so per-target rows line up.
        result_root["ab_comparison"] = {}
        for opt in ("dog", "adam"):
            print(f"[gamut_eval] ---- optimizer={opt} ----", flush=True)
            result_root["ab_comparison"][opt] = _run_all_batteries(
                solve_fn, model, pool, tag, sha, device,
                preset_knobs, opt, args.seed,
            )
    else:
        run = _run_all_batteries(
            solve_fn, model, pool, tag, sha, device,
            preset_knobs, args.optimizer, args.seed,
        )
        result_root.update(run)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_root, indent=2))
    print(f"[gamut_eval] wrote {out_path}", flush=True)

    # Human-readable summary at the bottom.
    def _print_overall(label: str, run: Dict[str, Any]) -> None:
        s = run["overall"]["stats"]
        if s.get("median_de") is None:
            print(f"[gamut_eval] {label}: all failed")
            return
        print(f"[gamut_eval] {label}: n={s['n']} "
              f"median ΔE={s['median_de']:.2f} "
              f"mean={s['mean_de']:.2f} "
              f"p95={s['p95_de']:.2f} "
              f"worst={s['worst_de']:.2f} "
              f"pass<5={s['pass_rate_de_lt_5']*100:.0f}% "
              f"pass<10={s['pass_rate_de_lt_10']*100:.0f}%")

    if args.optimizer == "both":
        for opt in ("dog", "adam"):
            _print_overall(f"optimizer={opt}",
                           result_root["ab_comparison"][opt])
    else:
        _print_overall(f"optimizer={args.optimizer}", result_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
