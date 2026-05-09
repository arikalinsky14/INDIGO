#!/usr/bin/env python3
"""
Visualize synthetic materials against real JLL materials.

Generates 50 samples from each synthetic strategy and overlays them with
~10 representative real materials on (n vs λ) and (k vs λ) axes. Useful
before committing GPU time to training — catches obviously-pathological
synthetic distributions (e.g. Lorentz oscillators producing nonsensical
hybrids that would never appear at inference).

Output: a 2x3 panel matplotlib figure saved to
        outputs/synthetic_visual_check.png
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.material_features import (
    CANONICAL_LAMBDA_NM,
    MaterialNK,
    load_jll_directory,
)
from src.synthetic_materials import (
    interpolate_real,
    parametric_lorentz,
    perturb_real,
)


_REPRESENTATIVE_REAL = ["Ag", "SiO2", "TiO2", "Si3N4", "aSi", "cSi", "Al", "AZO", "GaP"]


def _generate_perturb(n: int, real: List[MaterialNK], rng: np.random.Generator) -> List[MaterialNK]:
    out = []
    for _ in range(n):
        base = real[int(rng.integers(len(real)))]
        out.append(perturb_real(base, rng))
    return out


def _generate_interp(n: int, real: List[MaterialNK], rng: np.random.Generator) -> List[MaterialNK]:
    if len(real) < 2:
        return []
    out = []
    for _ in range(n):
        i, j = rng.choice(len(real), size=2, replace=False)
        out.append(interpolate_real(real[int(i)], real[int(j)], rng=rng))
    return out


def _generate_lorentz(n: int, rng: np.random.Generator) -> List[MaterialNK]:
    return [parametric_lorentz(rng) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jll-materials-dir", type=str, default=None,
                        help="Path to a JaxLayerLumos materials/ directory")
    parser.add_argument("--output", type=str, default="outputs/synthetic_visual_check.png")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed; cannot produce visualisation.")
        sys.exit(1)

    candidates = []
    if args.jll_materials_dir:
        candidates.append(Path(args.jll_materials_dir))
    candidates += [
        Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials"),
        Path("./jaxlayerlumos/materials"),
    ]
    materials_dir = next((c for c in candidates if c.exists()), None)
    if materials_dir is None:
        print(f"[ERROR] No JaxLayerLumos materials directory found in {candidates}")
        sys.exit(1)

    print(f"[INFO] Loading real materials from {materials_dir}")
    real_pool = load_jll_directory(materials_dir)
    real_held_in = [
        m for name, m in real_pool.items()
        if "-" not in name  # exclude disambiguated duplicates
    ]
    print(f"[INFO] {len(real_held_in)} real materials available for synthesis seeds")

    representatives = [real_pool[n] for n in _REPRESENTATIVE_REAL if n in real_pool]
    print(f"[INFO] {len(representatives)} representative real materials shown for context")

    rng = np.random.default_rng(args.seed)
    perturbed = _generate_perturb(args.num_samples, real_held_in, rng)
    interpolated = _generate_interp(args.num_samples, real_held_in, rng)
    lorentz = _generate_lorentz(args.num_samples, rng)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"Synthetic Materials vs Real ({args.num_samples} samples each)",
        fontsize=14, fontweight="bold",
    )

    panels = [
        ("perturb_real", perturbed, "tab:orange"),
        ("interpolate_real", interpolated, "tab:green"),
        ("parametric_lorentz", lorentz, "tab:purple"),
    ]

    for col, (label, samples, color) in enumerate(panels):
        ax_n = axes[0, col]
        ax_k = axes[1, col]
        for m in samples:
            ax_n.plot(CANONICAL_LAMBDA_NM, m.n, color=color, alpha=0.15, linewidth=0.8)
            ax_k.plot(CANONICAL_LAMBDA_NM, m.k, color=color, alpha=0.15, linewidth=0.8)
        for m in representatives:
            ax_n.plot(CANONICAL_LAMBDA_NM, m.n, "k-", alpha=0.7, linewidth=1.0)
            ax_k.plot(CANONICAL_LAMBDA_NM, m.k, "k-", alpha=0.7, linewidth=1.0)
        ax_n.set_title(f"{label}: n(λ)")
        ax_k.set_title(f"{label}: k(λ)")
        ax_n.set_ylabel("n"); ax_k.set_ylabel("k")
        ax_k.set_xlabel("Wavelength (nm)")
        ax_n.grid(True, alpha=0.3); ax_k.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualisation to {output_path}")


if __name__ == "__main__":
    main()
