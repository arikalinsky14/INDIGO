"""
Random Layer Sampler
====================

Ported from `chroma-lite/create_dataset/src/random_layer.py`. Differences:

- The fixed 25-material list is gone. Layer materials are drawn from a
  caller-supplied "available" pool (held-in real + a fresh batch of
  synthetic), regenerated per simulation run so that no run sees the
  same synthetic distribution twice.
- The optical-sim call goes through the new `src.optical_sim.OpticalSimulator`
  which accepts MaterialNK objects directly.
- The output is structured as in-memory tuples; serialisation to parquet
  happens in `compile_datasets.py`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.material_features import MaterialNK
from src.optical_sim import OpticalSimulator
from src.synthetic_materials import generate_synthetic_pool


# Thickness grid (nm), unchanged from CHROMA-Lite.
THICKNESS_RANGE_NM = np.arange(5, 201, 5)


class RandomLayerSimulation:
    """Sample structures from a held-in-real + synthetic material pool.

    Parameters
    ----------
    held_in_real : list of MaterialNK
        Real JLL materials that are allowed to appear in training structures.
    num_layers : int
        Number of layers per generated structure.
    incidence_angle : float
        Incidence angle in degrees, passed through to the optical sim.
    n_synthetic_per_run : int
        How many synthetic materials to generate per call to
        `random_materials_and_thicknesses` (drawn fresh each call).
    synthetic_weights : tuple of 3 floats
        Mix of (perturb, interpolate, lorentz) for the synthetic pool.
    seed : int
        Base seed for the simulation RNG.
    """

    def __init__(
        self,
        held_in_real: List[MaterialNK],
        num_layers: int = 4,
        incidence_angle: float = 0,
        n_synthetic_per_run: int = 16,
        synthetic_weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        seed: int = 42,
    ):
        if not held_in_real:
            raise ValueError("held_in_real must be non-empty")
        self.held_in_real = held_in_real
        self.num_layers = int(num_layers)
        self.incidence_angle = float(incidence_angle)
        self.n_synthetic_per_run = int(n_synthetic_per_run)
        self.synthetic_weights = synthetic_weights

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.simulator = OpticalSimulator(incidence_angle=self.incidence_angle)

    def _refresh_available_pool(self) -> List[MaterialNK]:
        """Build a fresh (held-in real + synthetic) pool for one structure."""
        synthetic = generate_synthetic_pool(
            self.held_in_real,
            n_synthetic=self.n_synthetic_per_run,
            rng=self.rng,
            weights=self.synthetic_weights,
        )
        return list(self.held_in_real) + synthetic

    def random_materials_and_thicknesses(
        self,
        available_pool: Optional[List[MaterialNK]] = None,
    ) -> Tuple[List[MaterialNK], List[int]]:
        """Sample `num_layers` materials and thicknesses for one structure.

        If `available_pool` is None, a fresh pool is built from held-in
        real + a new synthetic batch.
        """
        if available_pool is None:
            available_pool = self._refresh_available_pool()

        if len(available_pool) < self.num_layers:
            raise ValueError(
                f"available_pool has {len(available_pool)} materials but "
                f"num_layers={self.num_layers}"
            )

        idxs = self.rng.choice(len(available_pool), size=self.num_layers, replace=False)
        layer_materials = [available_pool[int(i)] for i in idxs]
        layer_thicknesses = [
            int(t) for t in self.rng.choice(THICKNESS_RANGE_NM, size=self.num_layers, replace=True)
        ]
        return layer_materials, layer_thicknesses

    def compute_color(
        self,
        layer_materials: List[MaterialNK],
        layer_thicknesses: List[int],
    ) -> List[int]:
        """Run the optical simulator and return sRGB ints in [0, 255]."""
        slot_indices = list(range(len(layer_materials)))
        return self.simulator.compute_color(
            pool=layer_materials,
            slot_indices=slot_indices,
            thicknesses_nm=layer_thicknesses,
        )

    def sample_structure(
        self,
    ) -> Tuple[List[MaterialNK], List[int], List[int]]:
        """Sample one structure and compute its sRGB.

        Returns
        -------
        layer_materials : list of MaterialNK
        layer_thicknesses : list of int
        sRGB : list of int
        """
        layer_materials, layer_thicknesses = self.random_materials_and_thicknesses()
        sRGB = self.compute_color(layer_materials, layer_thicknesses)
        return layer_materials, layer_thicknesses, sRGB


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    from src.material_features import load_jll_directory
    from create_dataset.src.pool_sampler import split_jll_real

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        print("[smoke] No JLL materials; skipping.")
        raise SystemExit(0)

    real_pool = load_jll_directory(materials_dir)
    held_in, _ = split_jll_real(real_pool)
    print(f"[smoke] held_in: {len(held_in)} materials")

    sim = RandomLayerSimulation(
        held_in_real=held_in,
        num_layers=3,
        incidence_angle=0,
        n_synthetic_per_run=8,
        seed=0,
    )

    materials, thicknesses, sRGB = sim.sample_structure()
    print(f"[smoke] materials: {[m.name for m in materials]}")
    print(f"[smoke] thicknesses: {thicknesses}")
    print(f"[smoke] sRGB: {sRGB}")
    assert all(0 <= c <= 255 for c in sRGB), "sRGB out of range"
    print("[smoke] OK")
