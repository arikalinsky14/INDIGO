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
    # Push the model to its limits for the definitive gamut number. Wall
    # cost per target on L40s: ensemble sim ~60 s, refine 3×120 iters
    # ~90 s, MC 32×5 draws ~20 s → ~2-3 min per target. Full 28-target
    # sweep at OPTIMIZER=both ≈ 2-3 hours.
    "max":      dict(ensemble_N=1000, top_k=5, refine_top_n=3,
                     refine_max_iters=120, tolerance_pct=5.0,
                     weight_lambda=1.0, mc_samples=32, temperature=0.9),
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
# Swatch grid (matplotlib) — one PNG summarising the run
# ----------------------------------------------------------------------------

def _is_in_srgb_gamut(lab: Tuple[float, float, float]) -> bool:
    """True iff the Lab colour has a representative in sRGB without
    clipping. Standalone probe because `lab_to_srgb_int` clips silently
    and never tells the caller — but the swatch's dashed-border marker
    for out-of-gamut targets needs to know.
    """
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    d = 6.0 / 29.0
    finv = lambda t: t ** 3 if t > d else 3.0 * d * d * (t - 4.0 / 29.0)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X, Y, Z = Xn * finv(fx), Yn * finv(fy), Zn * finv(fz)
    rl =  3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z
    gl = -0.9692660 * X +  1.8760108 * Y +  0.0415560 * Z
    bl =  0.0556434 * X + -0.2040259 * Y +  1.0572252 * Z
    return (0.0 <= rl <= 1.0) and (0.0 <= gl <= 1.0) and (0.0 <= bl <= 1.0)


def _readable_ink(rgb01: Tuple[float, float, float]) -> str:
    """Pick black or white text to sit on top of a swatch of colour `rgb01`.
    Uses the WCAG relative-luminance approximation."""
    r, g, b = rgb01
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b1120" if lum > 0.5 else "#f8fafc"


def _draw_swatch_grid(run: Dict[str, Any], out_path: Path, header: str) -> None:
    """Write a PNG showing target vs achieved swatches per battery.

    Layout: five rows (one per battery), each row a horizontal strip of
    (target | achieved) pairs. ΔE_00 printed on the achieved swatch;
    out-of-gamut targets get a dashed outline and are counted in the
    footer disclaimer. Fails silently (returns None) if matplotlib
    isn't importable — the JSON is the source of truth.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        # Reuse the existing Lab→sRGB converter used by the single-result
        # renderer so the swatch colours here match what a user sees on
        # inference/src/visualize.py output (same clipping semantics, same
        # sRGB primaries — no risk of drift).
        from inference.src.visualize import _lab_to_mpl_rgb
    except Exception as exc:
        print(f"[gamut_eval] matplotlib not available; skipping swatch "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return

    batteries = run.get("batteries", {})
    if not batteries:
        return

    # ---- layout constants (points; figure sized in inches at dpi=100) --
    SWATCH_H = 1.0
    SWATCH_W = 1.0
    PAIR_GAP = 0.06
    INTER_PAIR = 0.35
    ROW_H = 1.9
    ROW_TITLE_W = 2.4
    HEADER_H = 0.9
    FOOTER_H = 0.7
    MARGIN = 0.35

    battery_names = list(batteries.keys())
    max_pairs = max(len(batteries[b]["per_target"]) for b in battery_names)
    row_width = ROW_TITLE_W + max_pairs * (
        2 * SWATCH_W + PAIR_GAP + INTER_PAIR
    )
    total_w = row_width + 2 * MARGIN
    total_h = HEADER_H + len(battery_names) * ROW_H + FOOTER_H + 2 * MARGIN

    fig = plt.figure(figsize=(total_w, total_h), dpi=120)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_axis_off()
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")

    # ---- header --------------------------------------------------------
    ax.text(MARGIN, total_h - MARGIN - 0.2, header,
            color="#f8fafc", fontsize=13, weight="bold",
            va="top", ha="left", family="DejaVu Sans")

    # ---- rows ----------------------------------------------------------
    out_of_gamut = 0
    y_top = total_h - MARGIN - HEADER_H
    for row_idx, bname in enumerate(battery_names):
        y = y_top - (row_idx + 1) * ROW_H + 0.15
        # Battery label + summary
        stats = batteries[bname]["stats"]
        median_de = stats.get("median_de")
        worst_de = stats.get("worst_de")
        stats_str = "no data" if median_de is None else (
            f"median ΔE {median_de:.1f}   worst {worst_de:.1f}"
        )
        ax.text(MARGIN + 0.15, y + SWATCH_H + 0.35, bname,
                color="#e2e8f0", fontsize=11, weight="bold",
                va="bottom", ha="left")
        ax.text(MARGIN + 0.15, y + SWATCH_H + 0.05, stats_str,
                color="#94a3b8", fontsize=8.5, va="bottom", ha="left",
                family="DejaVu Sans Mono")

        for i, tgt in enumerate(batteries[bname]["per_target"]):
            x = MARGIN + ROW_TITLE_W + i * (2 * SWATCH_W + PAIR_GAP + INTER_PAIR)
            tgt_lab = tuple(tgt["target_lab"])
            achieved_lab = tgt.get("achieved_lab")
            de = tgt.get("delta_e")

            tgt_rgb = _lab_to_mpl_rgb(tgt_lab)
            tgt_in_gamut = _is_in_srgb_gamut(tgt_lab)
            if not tgt_in_gamut:
                out_of_gamut += 1

            # Target swatch — dashed amber border if out-of-gamut so the
            # user knows the on-screen colour is clipped and doesn't
            # represent the true chroma.
            tgt_rect = Rectangle(
                (x, y), SWATCH_W, SWATCH_H,
                facecolor=tgt_rgb,
                edgecolor="#f59e0b" if not tgt_in_gamut else "#1e293b",
                linewidth=1.5 if not tgt_in_gamut else 0.8,
                linestyle="--" if not tgt_in_gamut else "-",
            )
            ax.add_patch(tgt_rect)

            # Achieved swatch (grey if solve failed).
            if achieved_lab is not None:
                ach_rgb = _lab_to_mpl_rgb(tuple(achieved_lab))
            else:
                ach_rgb = (0.13, 0.16, 0.22)
            ach_rect = Rectangle(
                (x + SWATCH_W + PAIR_GAP, y), SWATCH_W, SWATCH_H,
                facecolor=ach_rgb,
                edgecolor="#1e293b", linewidth=0.8,
            )
            ax.add_patch(ach_rect)

            # ΔE overlaid on the achieved swatch.
            de_text = "fail" if de is None else f"ΔE {de:.1f}"
            ax.text(
                x + SWATCH_W + PAIR_GAP + SWATCH_W / 2.0,
                y + SWATCH_H / 2.0,
                de_text,
                color=_readable_ink(ach_rgb),
                fontsize=9, ha="center", va="center",
                family="DejaVu Sans Mono",
            )

            # Lab caption under the pair
            L, a, bl = tgt_lab
            ax.text(
                x + SWATCH_W + PAIR_GAP / 2.0,
                y - 0.10,
                f"L {L:.0f}  a {a:+.0f}  b {bl:+.0f}",
                color="#64748b", fontsize=7.2,
                ha="center", va="top",
                family="DejaVu Sans Mono",
            )

    # ---- footer + disclaimer ------------------------------------------
    ax.text(
        MARGIN, 0.55,
        f"Legend: [target | achieved] pair per column; ΔE_00 overlaid on "
        f"the achieved swatch. Dashed amber border marks out-of-gamut "
        f"targets ({out_of_gamut} total).",
        color="#94a3b8", fontsize=8.5, va="bottom", ha="left",
    )
    ax.text(
        MARGIN, 0.18,
        "Disclaimer: Lab extremes (chromatic_corners, high-|a|/|b| sweeps) "
        "often lie OUTSIDE the sRGB display gamut. The rendered swatch is "
        "clipped to the nearest displayable colour and may not match the "
        "true target chroma. ΔE_00 is computed in Lab and is unaffected.",
        color="#94a3b8", fontsize=8.5, va="bottom", ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[gamut_eval] wrote {out_path}", flush=True)


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
    p.add_argument("--preset", choices=tuple(PRESETS.keys()),
                   default="balanced")
    p.add_argument("--optimizer", choices=("dog", "adam", "both"),
                   default="dog",
                   help="Refinement optimizer. 'both' runs every target "
                        "twice for a side-by-side A/B in the output JSON.")
    p.add_argument("--output", required=True, type=str,
                   help="Where to write the aggregated JSON report.")
    p.add_argument("--swatch", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also write a companion PNG next to --output "
                        "with a target-vs-achieved swatch grid.")
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

    if args.swatch:
        png_path = out_path.with_suffix(".png")
        header = (f"gamut_eval  ·  {tag[:60]}"
                  f"{'…' if len(tag) > 60 else ''}"
                  f"  ·  preset={args.preset}"
                  f"  ·  optimizer={args.optimizer}")
        if args.optimizer == "both":
            # One PNG per optimizer. Build the name directly —
            # with_suffix('.png') would REPLACE the '.dog'/'.adam' piece
            # treating it as an existing suffix, so both iterations
            # would resolve to the SAME path and the second write would
            # overwrite the first. That was the previous bug that made
            # only the Adam PNG appear on disk.
            for opt in ("dog", "adam"):
                sub_path = out_path.with_name(
                    f"{out_path.stem}.{opt}.png"
                )
                _draw_swatch_grid(
                    result_root["ab_comparison"][opt], sub_path,
                    header.replace(f"optimizer={args.optimizer}",
                                   f"optimizer={opt}"),
                )
        else:
            _draw_swatch_grid(result_root, png_path, header)

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
