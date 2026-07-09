#!/usr/bin/env python3
"""
Render publication-style swatch grids from a gamut_eval JSON.

Standalone counterpart to the inline `--swatch` renderer in
gamut_eval.py. Use this when:

  1. You have a JSON from an earlier run and want to re-render the
     figures without repeating the (expensive) solve() sweep.
  2. The inline renderer only produced one of two expected PNGs (e.g.
     the `both` optimizer run wrote the Adam figure but not the DoG
     figure), and you want to force both.
  3. You want the paper-style look (white background, muted borders,
     serif captions) instead of the dark web-UI aesthetic.

Output paths mirror the JSON:
  input.json  →  input.png                 (single-optimizer JSON)
  input.json  →  input.dog.png, input.adam.png
                                            (JSON contains ab_comparison)

Usage
-----
  python -m inference.scripts.gamut_swatch inference/outputs/gamut_eval/gamut_max_both.json

  # Override the output stem:
  python -m inference.scripts.gamut_swatch <in.json> --out-prefix figures/gamut_max

  # Increase DPI for a print-quality figure:
  python -m inference.scripts.gamut_swatch <in.json> --dpi 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ----------------------------------------------------------------------------
# sRGB gamut probe (dup'd from gamut_eval.py to keep this script standalone;
# the Lab→sRGB conversion itself is imported from inference.src.visualize so
# rendered swatch colours match the single-result renderer exactly).
# ----------------------------------------------------------------------------

def _lab_to_srgb01(lab: Tuple[float, float, float]
                   ) -> Tuple[float, float, float]:
    """Lab → sRGB in [0, 1], clipped to the gamut. Standalone so this
    script has no torch / heavy dependency — matches the conversion
    inference/src/visualize.py's `_lab_to_mpl_rgb` uses, but doesn't
    drag in the training stack. See _is_in_srgb_gamut for the pre-clip
    in-gamut probe used to flag out-of-gamut targets in the figure.
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
    def enc(c: float) -> float:
        c = max(0.0, min(1.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return (enc(rl), enc(gl), enc(bl))


def _is_in_srgb_gamut(lab: Tuple[float, float, float]) -> bool:
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
    r, g, b = rgb01
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.5 else "#ffffff"


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def _draw_paper_swatch_grid(
    run: Dict[str, Any],
    out_path: Path,
    title: str,
    subtitle: str,
    dpi: int = 200,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _lab_to_mpl_rgb = _lab_to_srgb01

    batteries = run.get("batteries", {})
    if not batteries:
        print(f"[gamut_swatch] no batteries in {out_path.name}; skipping",
              file=sys.stderr)
        return

    # ---- typography ---------------------------------------------------
    # Research-paper aesthetic: serif family for the caption/title, a
    # neutral sans for tabular data. Set on the rcParams so labels
    # inherit without per-text plumbing.
    with plt.rc_context({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Nimbus Roman", "Times New Roman"],
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor":    "#111111",
        "axes.labelcolor":   "#111111",
        "text.color":        "#111111",
        "figure.facecolor":  "#ffffff",
        "axes.facecolor":    "#ffffff",
        "savefig.facecolor": "#ffffff",
    }):
        # ---- geometry (inches) ----------------------------------------
        SW = 0.85           # square swatch side
        PAIR_GAP = 0.06     # gap between target + achieved in a pair
        INTER = 0.28        # gap between pairs
        ROW_H = 1.70        # per-battery row height
        TITLE_W = 2.5       # left column for battery name + stats
        HEADER_H = 1.05
        FOOTER_H = 0.70
        MARGIN = 0.45

        battery_names = list(batteries.keys())
        max_pairs = max(
            len(batteries[b].get("per_target", [])) for b in battery_names
        )
        row_w = TITLE_W + max_pairs * (2 * SW + PAIR_GAP + INTER)
        total_w = row_w + 2 * MARGIN
        total_h = HEADER_H + len(battery_names) * ROW_H + FOOTER_H + 2 * MARGIN

        fig = plt.figure(figsize=(total_w, total_h), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, total_w)
        ax.set_ylim(0, total_h)
        ax.set_axis_off()

        # ---- header ---------------------------------------------------
        # No decorative rule — the whitespace + typography carry the
        # section split, matching the plainer paper aesthetic.
        ax.text(MARGIN, total_h - MARGIN - 0.10, title,
                fontsize=14, weight="bold", family="serif",
                va="top", ha="left")
        ax.text(MARGIN, total_h - MARGIN - 0.55, subtitle,
                fontsize=10, style="italic", color="#4b5563",
                family="serif", va="top", ha="left")

        # ---- rows -----------------------------------------------------
        oog_count = 0
        y_top = total_h - MARGIN - HEADER_H
        for row_idx, bname in enumerate(battery_names):
            y = y_top - (row_idx + 1) * ROW_H + 0.35
            stats = batteries[bname].get("stats", {})
            median_de = stats.get("median_de")
            worst_de = stats.get("worst_de")
            n = stats.get("n", 0)

            # Battery name: small-caps-ish emphasis via bold serif.
            ax.text(MARGIN + 0.10, y + SW + 0.30, bname.replace("_", " "),
                    fontsize=10.5, weight="bold", family="serif",
                    va="bottom", ha="left")

            stats_str = "no data" if median_de is None else (
                f"n={n}   med ΔE {median_de:.2f}   worst {worst_de:.2f}"
            )
            ax.text(MARGIN + 0.10, y + SW + 0.05, stats_str,
                    fontsize=8.5, color="#4b5563",
                    family="DejaVu Sans Mono", va="bottom", ha="left")

            for i, tgt in enumerate(batteries[bname].get("per_target", [])):
                x = (MARGIN + TITLE_W
                     + i * (2 * SW + PAIR_GAP + INTER))
                tgt_lab = tuple(tgt["target_lab"])
                achieved_lab = tgt.get("achieved_lab")
                de = tgt.get("delta_e")

                tgt_rgb = _lab_to_mpl_rgb(tgt_lab)
                in_gamut = _is_in_srgb_gamut(tgt_lab)
                if not in_gamut:
                    oog_count += 1

                # Target swatch. Out-of-gamut targets get a hatched
                # overlay (paper convention for "this is not the
                # displayable colour") plus a bold black outline.
                tgt_rect = Rectangle(
                    (x, y), SW, SW,
                    facecolor=tgt_rgb,
                    edgecolor="#111111",
                    linewidth=1.2 if not in_gamut else 0.7,
                )
                ax.add_patch(tgt_rect)
                if not in_gamut:
                    ax.add_patch(Rectangle(
                        (x, y), SW, SW,
                        facecolor="none",
                        edgecolor="#111111",
                        linewidth=0.0,
                        hatch="////",
                    ))

                # Achieved swatch. Neutral grey if solve failed.
                if achieved_lab is not None:
                    ach_rgb = _lab_to_mpl_rgb(tuple(achieved_lab))
                else:
                    ach_rgb = (0.90, 0.90, 0.92)
                ax.add_patch(Rectangle(
                    (x + SW + PAIR_GAP, y), SW, SW,
                    facecolor=ach_rgb,
                    edgecolor="#111111", linewidth=0.7,
                ))

                # ΔE printed BELOW the pair (paper convention: caption
                # under the figure). If solve failed, print 'fail'.
                de_text = "fail" if de is None else f"ΔE = {de:.2f}"
                ax.text(
                    x + SW + PAIR_GAP / 2.0,
                    y - 0.18,
                    de_text,
                    fontsize=8.5, family="serif",
                    ha="center", va="top",
                )
                # Lab coordinates below the ΔE.
                L, a, bl = tgt_lab
                ax.text(
                    x + SW + PAIR_GAP / 2.0,
                    y - 0.42,
                    f"L={L:.0f}  a={a:+.0f}  b={bl:+.0f}",
                    fontsize=6.8, color="#4b5563",
                    family="DejaVu Sans Mono",
                    ha="center", va="top",
                )

        # ---- footer ---------------------------------------------------
        # No decorative rule; the italic disclaimer sits below the last
        # Lab caption with generous whitespace.
        footer = (
            "Fig. Each column shows the target Lab colour (left) and the "
            "sRGB representation of the achieved thin-film reflectance "
            "(right). "
            f"Hatched swatches ({oog_count} total) mark Lab targets that "
            "lie outside the sRGB display gamut — the printed colour is "
            "clipped to the nearest displayable sRGB and does not "
            "represent the true target chroma. ΔE_00 is computed in Lab "
            "and is unaffected by the sRGB clipping."
        )
        ax.text(
            MARGIN, MARGIN + 0.05, footer,
            fontsize=8.0, style="italic", color="#374151",
            family="serif", va="bottom", ha="left",
            wrap=True,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), bbox_inches=None, pad_inches=0)
        plt.close(fig)
        print(f"[gamut_swatch] wrote {out_path}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _header_fields(root: Dict[str, Any]) -> Tuple[str, str]:
    """Title + subtitle strings for the figure header."""
    tag = str(root.get("checkpoint") or "<no tag>")
    preset = str(root.get("preset") or "?")
    seed = root.get("seed")
    pool_size = root.get("pool_size")
    title = f"INDIGO gamut evaluation"
    subtitle = (
        f"checkpoint: {tag[:80]}{'…' if len(tag) > 80 else ''}   "
        f"preset={preset}"
        + (f"   pool={pool_size}" if pool_size is not None else "")
        + (f"   seed={seed}" if seed is not None else "")
    )
    return title, subtitle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", type=str,
                   help="Path to a gamut_eval JSON.")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="Output basename stem. Defaults to the JSON's "
                        "stem in the same directory.")
    p.add_argument("--dpi", type=int, default=200,
                   help="Output DPI. 200 for on-screen preview, "
                        "300+ for print.")
    args = p.parse_args()

    in_path = Path(args.json_path)
    if not in_path.is_file():
        print(f"[gamut_swatch] no such file: {in_path}", file=sys.stderr)
        return 2

    root = json.loads(in_path.read_text())
    stem = (Path(args.out_prefix) if args.out_prefix
            else in_path.with_suffix(""))

    title, subtitle = _header_fields(root)

    # Case 1: single-optimizer JSON (root has 'batteries').
    if "batteries" in root and "ab_comparison" not in root:
        opt = root.get("optimizer", "?")
        _draw_paper_swatch_grid(
            root,
            stem.with_suffix(".png"),
            title,
            subtitle + f"   optimizer={opt}",
            dpi=args.dpi,
        )
        return 0

    # Case 2: A/B JSON (root has 'ab_comparison' with 'dog' and 'adam').
    ab = root.get("ab_comparison") or {}
    if not ab:
        print(f"[gamut_swatch] JSON has neither 'batteries' nor "
              f"'ab_comparison'. Nothing to render.", file=sys.stderr)
        return 2
    for opt in ("dog", "adam"):
        run = ab.get(opt)
        if not run:
            print(f"[gamut_swatch] no '{opt}' block in ab_comparison; "
                  f"skipping", flush=True)
            continue
        # Build the name directly — with_suffix('.png') would REPLACE
        # the '.dog' / '.adam' piece treating it as an existing suffix.
        out_path = stem.with_name(f"{stem.name}.{opt}.png")
        _draw_paper_swatch_grid(
            run, out_path, title,
            subtitle + f"   optimizer={opt}",
            dpi=args.dpi,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
