#!/usr/bin/env python3
"""
k-Nearest Structures by Color
=============================

Scans a customizable subset of the training data, finds the k structures
whose Lab target is closest to a query color (CIEDE2000 distance), and
writes a compact proposal-style figure:

  - LEFT: all k designs' reflectance spectra overlaid on one axis. Each
    line is drawn in the sRGB colour that design actually produces and
    given a distinct linestyle (-, --, -., :, (0,(3,1,1,1))), legend
    "Design 1".."Design k".
  - RIGHT (top row): target swatch + the k design swatches in one row.
  - RIGHT (bottom row): a 2D side-view layer-stack picture aligned
    under each design — bar heights accurate to the stored layer
    thicknesses, each material drawn with a stable (hatch, grey-shade)
    pair so reused materials are visually identifiable across designs.

This conveys inverse-design degeneracy in a single space-efficient
panel — many physically different multilayer structures producing
nearly the same perceived colour.

Examples
--------
    # 8 nearest matches to a saturated red, scanning 100K rows of training data
    python scripts/k_nearest_structures.py \\
        --data-dir data/train \\
        --query-rgb 220 30 30 \\
        --k 8 --scan-rows 100000

    # Query directly in Lab
    python scripts/k_nearest_structures.py \\
        --data-dir data/train \\
        --query-lab 50 40 -20 \\
        --k 12 --scan-rows 250000

    # Tier-A test set instead of training set
    python scripts/k_nearest_structures.py \\
        --data-dir data/test/tier_a \\
        --query-rgb 0 100 200 --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.color_utils import ciede2000, lab_to_srgb_int, srgb_to_lab
from src.dataset import _maybe_json, scan_files
from src.material_features import CANONICAL_LAMBDA_NM, MaterialNK, NUM_LAMBDA
from src.optical_sim import OpticalSimulator, is_available


@dataclass
class Candidate:
    """One row of training data + its Lab target. Pool/structure are kept lazy
    (we only reconstruct n,k for the k matches actually plotted, to save memory)."""
    file_path: str
    row_idx: int
    lab: Tuple[float, float, float]


def scan_candidates(data_dir: Path, max_rows: int, verbose: bool) -> List[Candidate]:
    """Walk the training shards in deterministic order, accumulating Lab targets
    until we have max_rows."""
    files = scan_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No parquet shards found under {data_dir}")
    if verbose:
        print(f"[INFO] {len(files)} shards available; scanning up to {max_rows} rows")

    out: List[Candidate] = []
    for f in files:
        if len(out) >= max_rows:
            break
        try:
            table = pq.read_table(f.path, columns=["lab"])
        except Exception as exc:
            print(f"[WARN] Could not read {f.path}: {exc}")
            continue
        labs = table["lab"].to_pylist()
        for i, raw in enumerate(labs):
            if len(out) >= max_rows:
                break
            try:
                lab = _maybe_json(raw)
                out.append(Candidate(file_path=f.path, row_idx=i,
                                     lab=(float(lab[0]), float(lab[1]), float(lab[2]))))
            except Exception as exc:
                print(f"[WARN] Skipping bad row {i} in {f.path}: {exc}")
        if verbose:
            print(f"[INFO]   scanned {len(out)} / {max_rows} (file: {Path(f.path).name})")
    return out


def reconstruct_full_row(file_path: str, row_idx: int) -> dict:
    """Pull the full row dict for the matched candidates."""
    cols = [
        "lab", "pool_size", "pool_n", "pool_k", "pool_names", "pool_sources",
        "layer_slots", "layer_thicknesses", "num_layers",
    ]
    table = pq.read_table(file_path, columns=cols)
    return {c: table[c][row_idx].as_py() for c in cols}


def row_to_pool_and_structure(
    row: dict,
) -> Tuple[List[MaterialNK], List[int], List[int]]:
    pool_n = _maybe_json(row["pool_n"])
    pool_k = _maybe_json(row["pool_k"])
    pool_names = _maybe_json(row["pool_names"])
    pool_sources = _maybe_json(row["pool_sources"])
    layer_slots = _maybe_json(row["layer_slots"])
    layer_thicknesses = _maybe_json(row["layer_thicknesses"])

    pool: List[MaterialNK] = []
    for s in range(int(row["pool_size"])):
        pool.append(MaterialNK(
            name=str(pool_names[s]),
            n=np.asarray(pool_n[s], dtype=np.float64),
            k=np.asarray(pool_k[s], dtype=np.float64),
            source=str(pool_sources[s]),
        ))
    return pool, [int(s) for s in layer_slots], [int(t) for t in layer_thicknesses]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find k training structures nearest a target color")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory of training shards (e.g. data/train)")

    # Query specification (one or the other)
    p.add_argument("--query-lab", type=float, nargs=3, default=None, metavar=("L", "A", "B"),
                   help="Query color as CIE Lab (L* a* b*). Mutually exclusive with --query-rgb.")
    p.add_argument("--query-rgb", type=int, nargs=3, default=None, metavar=("R", "G", "B"),
                   help="Query color as sRGB ints in [0, 255]. Converted to Lab internally.")

    p.add_argument("--k", type=int, default=5,
                   help="Number of nearest designs to show (default: 5 — the "
                        "proposal figure is built around target + 5 designs; "
                        "k>5 cycles the 5 linestyles and grows the swatch grid)")
    p.add_argument("--scan-rows", type=int, default=100_000,
                   help="How many training rows to scan for candidates (default: 100K)")
    p.add_argument("--output", type=str, default="outputs/k_nearest_structures.png")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if (args.query_lab is None) == (args.query_rgb is None):
        print("[ERROR] Exactly one of --query-lab or --query-rgb must be supplied",
              file=sys.stderr)
        sys.exit(1)

    if args.query_lab is not None:
        query_lab = tuple(args.query_lab)
        query_rgb = lab_to_srgb_int(query_lab)
    else:
        query_rgb = list(args.query_rgb)
        query_lab = srgb_to_lab(query_rgb)

    print(f"[INFO] Query: Lab=({query_lab[0]:.2f}, {query_lab[1]:.2f}, {query_lab[2]:.2f})  "
          f"sRGB={query_rgb}")

    # 1. Scan candidate Lab targets.
    candidates = scan_candidates(Path(args.data_dir), args.scan_rows, args.verbose)
    if len(candidates) < args.k:
        print(f"[ERROR] Scanned only {len(candidates)} candidates, need >= k={args.k}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Scanned {len(candidates)} candidate rows")

    # 2. Compute distances; retain top-k.
    distances = np.fromiter(
        (ciede2000(query_lab, c.lab) for c in candidates),
        dtype=np.float64, count=len(candidates),
    )
    top_idx = np.argsort(distances)[: args.k]
    print(f"[INFO] Top-{args.k} ΔE_00: min={distances[top_idx[0]]:.3f}, "
          f"max={distances[top_idx[-1]]:.3f}, mean={distances[top_idx].mean():.3f}")

    # 3. Reconstruct the matched structures and re-simulate reflectance.
    if not is_available():
        print("[ERROR] jaxlayerlumos required for re-simulating reflectance",
              file=sys.stderr)
        sys.exit(1)

    sim = OpticalSimulator(incidence_angle=0)
    matches = []
    for rank, idx in enumerate(top_idx):
        c = candidates[int(idx)]
        if args.verbose:
            print(f"[INFO]   match #{rank}: ΔE={distances[int(idx)]:.3f}, file={Path(c.file_path).name}, row={c.row_idx}")
        row = reconstruct_full_row(c.file_path, c.row_idx)
        pool, slots, thicks = row_to_pool_and_structure(row)
        reflectance = sim.compute_reflectance(pool, slots, thicks)
        matches.append({
            "rank": rank,
            "lab": c.lab,
            "delta_e": float(distances[int(idx)]),
            "n_layers": int(row["num_layers"]),
            "layer_materials": [pool[s].name for s in slots],
            "layer_thicknesses": thicks,
            "reflectance": np.array(reflectance),
        })

    # 4. Proposal figure: LEFT = all designs' reflectance overlaid (each
    #    line drawn in the colour that design actually produces, with a
    #    distinct linestyle); RIGHT = top row of target + per-design
    #    swatches, bottom row of stylised 2D layer-stack pictures aligned
    #    under each design (blank under target — no structure to show).
    #    Layer heights are accurate to the stored thicknesses; each
    #    material gets a stable (hatch, shade) pair so the eye can spot
    #    when two designs reuse the same material.
    try:
        import matplotlib as mpl
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot")
        return

    n_des = len(matches)
    # User-specified linestyle sequence; cycles if k > 5.
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    # Hatch + grey-shade palette for layers. Sized so 4-5 designs with
    # Poisson(4.5) layers each comfortably show unique materials; cycles
    # if a query happens to involve more than 10 distinct materials.
    _LAYER_PALETTE = [
        ("",       "#dcdcdc"),
        ("///",    "#b8b8b8"),
        ("\\\\\\", "#909090"),
        ("---",    "#cccccc"),
        ("|||",    "#a8a8a8"),
        ("...",    "#d4d4d4"),
        ("xxx",    "#8a8a8a"),
        ("+++",    "#b0b0b0"),
        ("***",    "#c0c0c0"),
        ("ooo",    "#9c9c9c"),
    ]
    # Tighter hatch lines for print legibility.
    mpl.rcParams["hatch.linewidth"] = 0.6

    def _txt_color(srgb):
        b = (0.299 * srgb[0] + 0.587 * srgb[1] + 0.114 * srgb[2]) / 255.0
        return "black" if b > 0.55 else "white"

    # Assign visuals to materials in order of first appearance across the
    # k designs — consistent across panels so a viewer can see, e.g., that
    # design 2 and design 4 share a layer.
    mat_order: List[str] = []
    for m in matches:
        for name in m["layer_materials"]:
            if name not in mat_order:
                mat_order.append(name)
    mat_visual = {
        name: _LAYER_PALETTE[i % len(_LAYER_PALETTE)]
        for i, name in enumerate(mat_order)
    }
    y_max = max(sum(m["layer_thicknesses"]) for m in matches)

    # --- Layout: 2 rows x (1 spec + 1 target + n_des design) cols ---
    n_cols_right = 1 + n_des  # target + designs
    fig = plt.figure(figsize=(3.6 + 1.6 * n_cols_right, 4.4))
    gs = fig.add_gridspec(
        2, 1 + n_cols_right,
        width_ratios=[3.4] + [1.0] * n_cols_right,
        height_ratios=[1.0, 1.25],
        wspace=0.15, hspace=0.18,
    )

    # --- Left (spans both rows): overlaid reflectance spectra ---
    ax = fig.add_subplot(gs[:, 0])
    for i, m in enumerate(matches):
        srgb = lab_to_srgb_int(m["lab"])
        ax.plot(
            CANONICAL_LAMBDA_NM, m["reflectance"],
            color=[c / 255 for c in srgb],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=1.8,
            label=f"Design {m['rank'] + 1}",
        )
    ax.set_xlim(CANONICAL_LAMBDA_NM.min(), CANONICAL_LAMBDA_NM.max())
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    # --- Top right row: target + design swatches ---
    swatch_cells = [("target", query_rgb)] + [
        (f"design {m['rank'] + 1}", lab_to_srgb_int(m["lab"])) for m in matches
    ]
    for col_offset, (label, srgb) in enumerate(swatch_cells):
        sax = fig.add_subplot(gs[0, 1 + col_offset])
        sax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                         facecolor=[v / 255 for v in srgb]))
        sax.set_xlim(0, 1); sax.set_ylim(0, 1)
        sax.set_xticks([]); sax.set_yticks([])
        sax.text(0.5, 0.5, label, ha="center", va="center",
                 fontsize=12, color=_txt_color(srgb))

    # --- Bottom right row: structures (skip column 0 = target) ---
    # Blank cell directly under "target" — nothing to draw there.
    blank_ax = fig.add_subplot(gs[1, 1])
    blank_ax.axis("off")

    for i, m in enumerate(matches):
        stax = fig.add_subplot(gs[1, 2 + i])
        y = 0.0
        for mat, thick in zip(m["layer_materials"], m["layer_thicknesses"]):
            hatch, color = mat_visual[mat]
            stax.bar(
                0.5, thick, bottom=y, width=0.92,
                color=color, hatch=hatch,
                edgecolor="black", linewidth=0.6,
            )
            y += thick
        stax.set_xlim(0, 1)
        stax.set_ylim(0, y_max * 1.04)
        stax.set_xticks([])
        if i == 0:
            stax.set_ylabel("Thickness (nm)", fontsize=9)
            stax.tick_params(axis="y", labelsize=8)
        else:
            stax.set_yticks([])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualization to {output_path}")

    # Also dump a JSON summary alongside.
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "query_lab": list(query_lab),
            "query_rgb_display": list(query_rgb),
            "data_dir": str(args.data_dir),
            "scanned_rows": len(candidates),
            "k": args.k,
            "matches": [
                {
                    "rank": m["rank"],
                    "delta_e_00": m["delta_e"],
                    "lab": list(m["lab"]),
                    "n_layers": m["n_layers"],
                    "layer_materials": m["layer_materials"],
                    "layer_thicknesses": m["layer_thicknesses"],
                }
                for m in matches
            ],
        }, f, indent=2)
    print(f"[INFO] Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
