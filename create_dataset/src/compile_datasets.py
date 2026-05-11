"""
Compile a parquet shard of (RGB, pool, structure) training rows.

Ported from `chroma-lite/create_dataset/src/compile_datasets.py`. The
text-prompt machinery (Rephraser, procedural_template, OpenAI batch)
and the incorrect-prompts branch have been removed entirely. INDIGO
has no text input.

For each generated structure:
  1. Sample a (structure, RGB) pair via RandomLayerSimulation.
  2. Build a per-row material pool with sample_pool() that includes the
     structure materials at randomized slot indices, plus distractors.
  3. Write the row in the §6.1 schema.

Output directory naming: `layers_NN_angle_AA_substrate_CSi/seed_XXXXX.parquet`
(matches CHROMA-Lite for consistency with `src/dataset.py:scan_files`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.material_features import load_jll_directory

from create_dataset.src.pool_sampler import (
    PoolSamplerConfig,
    sample_pool,
    split_jll_real,
)
from create_dataset.src.random_layer import RandomLayerSimulation


_DEFAULT_JLL_PATHS = [
    Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials"),
    Path("./jaxlayerlumos/materials"),
]


def _find_jll_materials_dir(override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"JLL materials dir not found at {override}")
        return override
    for candidate in _DEFAULT_JLL_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a JaxLayerLumos materials directory. "
        f"Tried: {_DEFAULT_JLL_PATHS}. Pass --jll-materials-dir to override."
    )


def get_output_path(
    output_dir: Path,
    num_layers: int,
    incidence_angle: int,
    structure_seed: int,
) -> Path:
    """Same naming convention as the original CHROMA-Lite pipeline."""
    sub_dir = output_dir / f"layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi"
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir / f"seed_{structure_seed:05d}.parquet"


def build_rows(
    num_structures: int,
    sim: RandomLayerSimulation,
    held_in_real: List,
    pool_config: PoolSamplerConfig,
    pool_rng: np.random.Generator,
    structure_seed: int,
    incidence_angle: int,
    verbose: bool = False,
) -> List[dict]:
    rows: List[dict] = []
    for i in range(num_structures):
        layer_materials, layer_thicknesses, sRGB = sim.sample_structure()

        pool, structure_slot_indices = sample_pool(
            structure_materials_required=layer_materials,
            held_in_real=held_in_real,
            rng=pool_rng,
            config=pool_config,
        )

        rows.append({
            "rgb_R": json.dumps(sRGB),
            "pool_size": len(pool),
            "pool_n": [m.n.tolist() for m in pool],
            "pool_k": [m.k.tolist() for m in pool],
            "pool_names": [m.name for m in pool],
            "pool_sources": [m.source for m in pool],
            "layer_slots": [int(s) for s in structure_slot_indices],
            "layer_thicknesses": [int(t) for t in layer_thicknesses],
            "num_layers": len(layer_materials),
            "structure_seed": structure_seed,
            "incidence_angle": incidence_angle,
        })

        if verbose and (i + 1) % 100 == 0:
            print(f"[INFO] Generated {i + 1}/{num_structures} rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, required=True,
                        help="Number of layers per structure")
    parser.add_argument("--incidence_angle", type=int, required=True,
                        help="Incidence angle in degrees")
    parser.add_argument("--structure_seed", type=int, required=True,
                        help="Seed for the structure-sampling RNG")
    parser.add_argument("--num_structures", type=int, default=10000,
                        help="Number of structures to generate (default: 10000)")
    parser.add_argument("--pool_size_min", type=int, default=4,
                        help="Minimum number of materials in a pool")
    parser.add_argument("--pool_size_max", type=int, default=32,
                        help="Maximum number of materials in a pool")
    parser.add_argument("--p_synthetic", type=float, default=0.8,
                        help="Fraction of distractor pool slots that are synthetic")
    parser.add_argument("--n_synthetic_per_structure", type=int, default=16,
                        help="Synthetic-pool size used when sampling structure layers")
    parser.add_argument("--output_dir", type=str, default="create_dataset/data_prompts",
                        help="Where to write the parquet shard")
    parser.add_argument("--jll_materials_dir", type=str, default=None,
                        help="Override path to the JLL materials/ directory")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jll_dir = _find_jll_materials_dir(
        Path(args.jll_materials_dir) if args.jll_materials_dir else None
    )
    print(f"[INFO] Loading JLL materials from {jll_dir}")
    real_pool = load_jll_directory(jll_dir)
    held_in_real, held_out_real = split_jll_real(real_pool)
    print(
        f"[INFO] held_in: {len(held_in_real)} materials, "
        f"held_out: {len(held_out_real)} materials"
    )

    sim = RandomLayerSimulation(
        held_in_real=held_in_real,
        num_layers=args.num_layers,
        incidence_angle=args.incidence_angle,
        n_synthetic_per_run=args.n_synthetic_per_structure,
        seed=args.structure_seed,
    )

    pool_config = PoolSamplerConfig(
        pool_size_min=args.pool_size_min,
        pool_size_max=args.pool_size_max,
        p_synthetic=args.p_synthetic,
    )
    # Use a derived (but distinct) seed so pool sampling is independent of
    # structure sampling. Hash-based to keep things deterministic.
    pool_rng = np.random.default_rng(
        np.random.SeedSequence([args.structure_seed, 0xC0FFEE]).generate_state(1)[0]
    )

    print(
        f"[INFO] Generating {args.num_structures} structures "
        f"(num_layers={args.num_layers}, angle={args.incidence_angle}, "
        f"seed={args.structure_seed})"
    )
    rows = build_rows(
        num_structures=args.num_structures,
        sim=sim,
        held_in_real=held_in_real,
        pool_config=pool_config,
        pool_rng=pool_rng,
        structure_seed=args.structure_seed,
        incidence_angle=args.incidence_angle,
        verbose=args.verbose,
    )

    output_path = get_output_path(
        Path(args.output_dir),
        args.num_layers,
        args.incidence_angle,
        args.structure_seed,
    )
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Wrote {len(rows)} rows to {output_path} (pid={os.getpid()})")


if __name__ == "__main__":
    main()
