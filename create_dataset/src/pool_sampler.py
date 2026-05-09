"""
Material Pool Sampler
=====================

Generates a training pool for a single dataset row. The pool always contains
all materials needed for the structure (so the model has the right answer
available) plus distractors. Slot ordering is randomized per example —
this is the single most important step for breaking material memorization.

Held-out real materials never appear in any training pool. They are reserved
for Tier-B evaluation (see the project README §validation strategy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.material_features import MaterialNK
from src.materials_vocab import M_MAX
from src.synthetic_materials import (
    interpolate_real,
    parametric_lorentz,
    perturb_real,
)


# ============================================================================
# Held-out real materials (excluded from training pools)
# ============================================================================
#
# A reasonable default holdout: ~30% of JLL real materials, picking a mix of
# metal / metal-like / semiconductor / dielectric / transparent-conductor so
# all material classes are represented in the held-out set.

HELD_OUT_REAL_MATERIALS: List[str] = [
    "Au",     # metal
    "TiN",    # metal-like / refractory
    "GaAs",   # semiconductor
    "Ge",     # semiconductor
    "Al2O3",  # dielectric
    "ITO",    # transparent conductor
    "Pt",     # metal
    "ZnO",    # transparent conductor
]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PoolSamplerConfig:
    """Hyperparameters for pool construction."""

    m_max: int = M_MAX
    pool_size_min: int = 4
    pool_size_max: int = M_MAX
    p_synthetic: float = 0.8
    synthetic_weights: Tuple[float, float, float] = (0.4, 0.2, 0.4)
    # synthetic_weights = (perturb_real, interpolate_real, parametric_lorentz)


# ============================================================================
# Real-material partitioning
# ============================================================================


def split_jll_real(
    real_pool: Dict[str, MaterialNK],
) -> Tuple[List[MaterialNK], List[MaterialNK]]:
    """Partition a JLL pool into (held_in, held_out) lists of real materials.

    Disambiguated entries (e.g. "Ag-Rakic-LD-1998") are filtered out so that
    each canonical name (e.g. "Ag") contributes exactly one material to the
    output. This avoids loading the same physical material twice under
    different parameterisations.

    Parameters
    ----------
    real_pool : dict
        Output of `material_features.load_jll_directory` — keyed by both
        bare names and full filename stems.

    Returns
    -------
    held_in : list of MaterialNK
        Bare-name entries not in HELD_OUT_REAL_MATERIALS.
    held_out : list of MaterialNK
        Bare-name entries in HELD_OUT_REAL_MATERIALS.
    """
    held_out_set = set(HELD_OUT_REAL_MATERIALS)
    held_in: List[MaterialNK] = []
    held_out: List[MaterialNK] = []

    for name, material in real_pool.items():
        # Keep only canonical short names (no dashes from disambiguated CSVs).
        if "-" in name:
            continue
        if name in held_out_set:
            held_out.append(material)
        else:
            held_in.append(material)

    return held_in, held_out


# ============================================================================
# Distractor generation
# ============================================================================


def _sample_synthetic(
    rng: np.random.Generator,
    held_in_real: List[MaterialNK],
    weights: Tuple[float, float, float],
) -> MaterialNK:
    """Draw one synthetic material per the given strategy weights."""
    methods = ["perturb_real", "interpolate_real", "parametric_lorentz"]
    p = np.array(weights, dtype=np.float64)
    p = p / p.sum()
    method = methods[int(rng.choice(3, p=p))]

    if method == "perturb_real":
        if not held_in_real:
            raise ValueError("perturb_real requires a non-empty held_in_real pool")
        base = held_in_real[int(rng.integers(len(held_in_real)))]
        return perturb_real(base, rng)

    if method == "interpolate_real":
        if len(held_in_real) < 2:
            raise ValueError("interpolate_real requires at least 2 held-in real materials")
        i, j = rng.choice(len(held_in_real), size=2, replace=False)
        return interpolate_real(held_in_real[int(i)], held_in_real[int(j)], rng=rng)

    return parametric_lorentz(rng)


def sample_distractors(
    n: int,
    held_in_real: List[MaterialNK],
    rng: np.random.Generator,
    p_synthetic: float,
    synthetic_weights: Tuple[float, float, float],
) -> List[MaterialNK]:
    """Sample n distractor materials, mixing synthetic and held-in real."""
    if n <= 0:
        return []

    out: List[MaterialNK] = []
    for _ in range(n):
        if rng.random() < p_synthetic or not held_in_real:
            out.append(_sample_synthetic(rng, held_in_real, synthetic_weights))
        else:
            out.append(held_in_real[int(rng.integers(len(held_in_real)))])
    return out


# ============================================================================
# Top-level pool construction
# ============================================================================


def sample_pool(
    structure_materials_required: List[MaterialNK],
    held_in_real: List[MaterialNK],
    rng: np.random.Generator,
    config: PoolSamplerConfig,
) -> Tuple[List[MaterialNK], List[int]]:
    """Build a (pool, structure_slot_indices) pair for one training row.

    Steps
    -----
    1. Pick pool_size in [max(pool_size_min, n_required), pool_size_max].
    2. Generate (pool_size - n_required) distractor materials.
    3. Concatenate required + distractors → unshuffled pool.
    4. Shuffle slot order with a fresh random permutation.
    5. Compute the slot indices the structure layers should refer to.

    Returns
    -------
    pool : list of MaterialNK, length pool_size
    structure_slot_indices : list of int, length n_required
        Indices into `pool` for each entry in structure_materials_required,
        in the original (deposition) order.
    """
    n_required = len(structure_materials_required)
    if n_required == 0:
        raise ValueError("structure_materials_required must be non-empty")
    if n_required > config.pool_size_max:
        raise ValueError(
            f"need {n_required} materials in pool but pool_size_max={config.pool_size_max}"
        )

    lower = max(config.pool_size_min, n_required)
    upper = config.pool_size_max
    if lower > upper:
        raise ValueError(f"pool size range collapses (lower={lower}, upper={upper})")
    pool_size = int(rng.integers(lower, upper + 1))

    distractors = sample_distractors(
        pool_size - n_required,
        held_in_real,
        rng,
        config.p_synthetic,
        config.synthetic_weights,
    )

    pool_unshuffled = list(structure_materials_required) + distractors
    permutation = rng.permutation(pool_size).tolist()
    pool = [pool_unshuffled[i] for i in permutation]

    # For each required material i, find where it ended up after shuffling.
    # `permutation[k] = i` means slot k of the new pool holds pool_unshuffled[i].
    inverse = [0] * pool_size
    for new_slot, original_idx in enumerate(permutation):
        inverse[original_idx] = new_slot
    structure_slot_indices = inverse[:n_required]

    return pool, structure_slot_indices


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    from src.material_features import load_jll_directory

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        print("[smoke] No JLL materials; skipping.")
        raise SystemExit(0)

    real_pool = load_jll_directory(materials_dir)
    held_in, held_out = split_jll_real(real_pool)
    print(f"[smoke] held_in: {len(held_in)} materials, held_out: {len(held_out)}")
    print(f"[smoke] held_out names: {[m.name for m in held_out]}")

    rng = np.random.default_rng(42)
    config = PoolSamplerConfig(pool_size_min=4, pool_size_max=12)

    # Pretend the structure uses 3 materials drawn from the held-in real pool.
    required = [held_in[i] for i in rng.choice(len(held_in), size=3, replace=False)]
    pool, slots = sample_pool(required, held_in, rng, config)

    print(f"[smoke] required: {[m.name for m in required]}")
    print(f"[smoke] pool size: {len(pool)} (expected in [4, 12])")
    print(f"[smoke] structure slot indices: {slots}")
    for k, slot in enumerate(slots):
        assert pool[slot] is required[k], (
            f"slot {slot} should point to required[{k}] (={required[k].name}), "
            f"got {pool[slot].name}"
        )
    print("[smoke] slot index → required material mapping verified ✓")
    print("[smoke] OK")
