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
  - RIGHT: the target swatch followed by the k design swatches in a
    grid (target + 5 -> 2x3, matching the proposal layout).

This conveys inverse-design degeneracy — many physically different
multilayer structures producing nearly the same perceived colour —
in a single space-efficient panel.

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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.color_utils import lab_to_srgb_int, srgb_to_lab
from src.dataset import _maybe_json, scan_files
from src.material_features import CANONICAL_LAMBDA_NM, MaterialNK, NUM_LAMBDA
from src.optical_sim import OpticalSimulator, is_available


# CIEDE2000 lifted from scripts/evaluate.py — duplicated here to keep this
# script self-contained and avoid pulling in torch.
def _ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0
    elif abs(dhp) > 180:
        dhp -= 360 if dhp > 180 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hbp = (h1p + h2p) / 2
    if C1p * C2p != 0 and abs(h1p - h2p) > 180:
        hbp += 180 if h1p + h2p < 360 else -180
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dTheta = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    R_C = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    S_L = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    S_C = 1 + 0.045 * Cbp
    S_H = 1 + 0.015 * Cbp * T
    R_T = -math.sin(math.radians(2 * dTheta)) * R_C
    return math.sqrt(
        (dLp / S_L) ** 2 + (dCp / S_C) ** 2 + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


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
        (_ciede2000(query_lab, c.lab) for c in candidates),
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
    #    distinct linestyle); RIGHT = target swatch + per-design swatches
    #    in a compact grid. Conveys inverse-design degeneracy (many
    #    different structures → nearly the same perceived colour) in one
    #    space-efficient panel.
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot")
        return

    import math as _math

    n_des = len(matches)
    # User-specified linestyle sequence; cycles if k > 5.
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    def _txt_color(srgb):
        # Black text on light swatches, white on dark (perceived brightness).
        b = (0.299 * srgb[0] + 0.587 * srgb[1] + 0.114 * srgb[2]) / 255.0
        return "black" if b > 0.55 else "white"

    ncols_sw = 3
    nrows_sw = _math.ceil((n_des + 1) / ncols_sw)

    fig = plt.figure(figsize=(11, 1.6 * nrows_sw + 0.6))
    gs = fig.add_gridspec(
        nrows_sw, 1 + ncols_sw,
        width_ratios=[3.4] + [1.0] * ncols_sw,
        wspace=0.12, hspace=0.18,
    )

    # --- Left: overlaid reflectance spectra ---
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

    # --- Right: target + design swatches in a grid ---
    cells = [("target", query_rgb)] + [
        (f"design {m['rank'] + 1}", lab_to_srgb_int(m["lab"])) for m in matches
    ]
    for idx, (label, srgb) in enumerate(cells):
        r, c = divmod(idx, ncols_sw)
        sax = fig.add_subplot(gs[r, 1 + c])
        sax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                         facecolor=[v / 255 for v in srgb]))
        sax.set_xlim(0, 1)
        sax.set_ylim(0, 1)
        sax.set_xticks([])
        sax.set_yticks([])
        sax.text(0.5, 0.5, label, ha="center", va="center",
                 fontsize=12, color=_txt_color(srgb))
    # Blank any unused trailing grid cells.
    for idx in range(len(cells), nrows_sw * ncols_sw):
        r, c = divmod(idx, ncols_sw)
        fig.add_subplot(gs[r, 1 + c]).axis("off")

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
