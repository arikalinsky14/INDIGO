#!/usr/bin/env python3
"""
Generate Constraint-Adherence Test Set

Generates 1200 test examples (400 each for 4, 6, and 8 layers) with:
- Natural language prompts (raw procedural templates, no LLM rephrasing)
- Structured constraint annotations for automated checking

Output: create_dataset/data_prompts/test_set.csv

Usage:
    cd create_dataset/src
    python generate_test_set.py
    python generate_test_set.py --output ../data_prompts/test_set.csv --seed 12345
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directories to path for imports
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent.parent))

from random_layer import RandomLayerSimulation
from constraint_aware_template import get_template_with_constraints

# ============================================================================
# Configuration
# ============================================================================

LAYER_COUNTS = [4, 6, 8]
EXAMPLES_PER_LAYER_COUNT = 400
TOTAL_EXAMPLES = len(LAYER_COUNTS) * EXAMPLES_PER_LAYER_COUNT


def generate_structures(num_layers: int, n_examples: int, seed: int):
    """
    Generate random thin-film structures and their RGB colors.

    Uses RandomLayerSimulation to create structures, then computes
    the reflected sRGB color via optical simulation.

    Returns list of dicts with keys: materials, thicknesses, sRGB_R, num_layers
    """
    sim = RandomLayerSimulation(
        num_layers=num_layers,
        incidence_angle=0,
        seed=seed,
    )

    structures = []
    for _ in range(n_examples):
        layer_materials, layer_thicknesses = sim.random_materials_and_thicknesses()

        # Run optical simulation to get reflectance
        n_matrix = sim.create_n_matrix(['Air'] + layer_materials + ['FusedSilica'])
        R_avg, _ = sim.calculate_tr(n_matrix, [0] + list(layer_thicknesses) + [0])

        # Convert reflectance spectrum to sRGB
        import jaxlayerlumos.colors.composite as jll_colors
        import jax.numpy as jnp

        wavelengths_nm = np.array(sim.wavelength_vector) * 1e9
        R_np = np.array(R_avg)

        # Filter to visible range for color computation
        visible = np.logical_and(wavelengths_nm > 360, wavelengths_nm < 830)
        sRGB = jll_colors.spectrum_to_sRGB(
            jnp.array(wavelengths_nm[visible]),
            jnp.array(R_np[visible]),
            use_clipping=True,
        )
        sRGB = np.array(sRGB).flatten()[:3] * 255
        sRGB = np.round(sRGB).astype(int).tolist()

        # Fix AZO naming (RandomLayerSimulation uses 'AZO-Zarei')
        clean_materials = [m.replace('AZO-Zarei', 'AZO') for m in layer_materials]

        structures.append({
            'materials': clean_materials,
            'thicknesses': [int(t) for t in layer_thicknesses],
            'sRGB_R': sRGB,
            'num_layers': num_layers,
        })

    return structures


def main():
    parser = argparse.ArgumentParser(
        description='Generate constraint-adherence test set')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: create_dataset/data_prompts/test_set.csv)')
    parser.add_argument('--seed', type=int, default=99999,
                        help='Base seed for reproducibility')
    args = parser.parse_args()

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _script_dir.parent / 'data_prompts' / 'test_set.csv'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating constraint-adherence test set")
    print(f"  Layer counts: {LAYER_COUNTS}")
    print(f"  Examples per count: {EXAMPLES_PER_LAYER_COUNT}")
    print(f"  Total: {TOTAL_EXAMPLES}")
    print(f"  Output: {output_path}")
    print(f"  Seed: {args.seed}")
    print()

    rows = []

    for num_layers in LAYER_COUNTS:
        # Use a unique seed per layer count for reproducibility
        layer_seed = args.seed + num_layers * 10000
        print(f"[{num_layers} layers] Generating {EXAMPLES_PER_LAYER_COUNT} structures...")

        structures = generate_structures(
            num_layers=num_layers,
            n_examples=EXAMPLES_PER_LAYER_COUNT,
            seed=layer_seed,
        )

        print(f"[{num_layers} layers] Generating templates with constraints...")
        for idx, struct in enumerate(structures):
            # Use a unique prompt seed per example
            prompt_seed = layer_seed + idx

            text, constraints_dict = get_template_with_constraints(
                num_layers=struct['num_layers'],
                materials=struct['materials'],
                thicknesses=struct['thicknesses'],
                incidence_angle=0,
                color_rgb=struct['sRGB_R'],
                seed=prompt_seed,
            )

            rows.append({
                'text': text,
                'materials': json.dumps(struct['materials']),
                'thicknesses': json.dumps(struct['thicknesses']),
                'sRGB_R': json.dumps(struct['sRGB_R']),
                'num_layers': struct['num_layers'],
                'constraints': json.dumps(constraints_dict),
            })

        print(f"[{num_layers} layers] Done. ({len(structures)} examples)")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nWrote {len(df)} examples to {output_path}")
    print(f"  Columns: {list(df.columns)}")

    # Print constraint coverage stats
    n_with_layer = sum(1 for r in rows if json.loads(r['constraints'])['layer_count_type'] != 'none')
    n_with_material = sum(1 for r in rows if json.loads(r['constraints'])['materials_type'] != 'any')
    n_with_additional = sum(1 for r in rows if len(json.loads(r['constraints'])['additional']) > 0)
    avg_additional = np.mean([len(json.loads(r['constraints'])['additional']) for r in rows])

    print(f"\nConstraint coverage:")
    print(f"  Layer count constraints: {n_with_layer}/{len(rows)} ({100*n_with_layer/len(rows):.1f}%)")
    print(f"  Material constraints: {n_with_material}/{len(rows)} ({100*n_with_material/len(rows):.1f}%)")
    print(f"  Additional constraints: {n_with_additional}/{len(rows)} ({100*n_with_additional/len(rows):.1f}%)")
    print(f"  Avg additional constraints per example: {avg_additional:.2f}")


if __name__ == '__main__':
    main()
