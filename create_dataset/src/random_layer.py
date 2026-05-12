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
- Layer count per structure is sampled from a truncated Poisson distribution
  (`LayerCountConfig`) rather than fixed.
- The output is structured as in-memory tuples; serialisation to parquet
  happens in `compile_datasets.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.material_features import MaterialNK
from src.optical_sim import OpticalSimulator
from src.synthetic_materials import generate_synthetic_pool


# Thickness grid (nm), unchanged from CHROMA-Lite.
THICKNESS_RANGE_NM = np.arange(5, 201, 5)


@dataclass
class LayerCountConfig:
    """Truncated-Poisson layer count sampler.

    For λ=4.5 over [2, 10] the rejection rate is ~6-7%, so rejection
    sampling is cheap.
    """

    lam: float = 4.5
    min_layers: int = 2
    max_layers: int = 10

    def sample(self, rng: np.random.Generator) -> int:
        for _ in range(1000):
            k = int(rng.poisson(lam=self.lam))
            if self.min_layers <= k <= self.max_layers:
                return k
        raise RuntimeError(
            f"Truncated Poisson failed to sample in 1000 tries "
            f"(λ={self.lam}, range=[{self.min_layers}, {self.max_layers}])"
        )


class RandomLayerSimulation:
    """Sample structures from an active-real + synthetic material pool.

    Parameters
    ----------
    held_in_real : list of MaterialNK
        The "active" set of real materials passed through to synthetic
        generators. For training this is the held-in set; for Tier-B
        test set generation it can be flipped to the held-out set
        (`split_jll_real(use_held_out_reals=True)`).
    layer_count : LayerCountConfig, optional
        Variable layer count via truncated Poisson. Mutually exclusive
        with `num_layers`.
    num_layers : int, optional
        Fixed layer count (back-compat). Mutually exclusive with
        `layer_count`.
    incidence_angle : float
        Incidence angle in degrees, passed through to the optical sim.
    n_synthetic_per_run : int
        How many synthetic materials to generate per call to
        `random_materials_and_thicknesses` (drawn fresh each call).
    synthetic_weights : tuple of 4 floats
        Mix of (perturb_small, perturb_large, interpolate_real,
        parametric_lorentz) for the synthetic pool.
    seed : int
        Base seed for the simulation RNG.
    """

    def __init__(
        self,
        held_in_real: List[MaterialNK],
        layer_count: Optional[LayerCountConfig] = None,
        num_layers: Optional[int] = None,
        incidence_angle: float = 0,
        n_synthetic_per_run: int = 16,
        synthetic_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.15, 0.35),
        seed: int = 42,
    ):
        if not held_in_real:
            raise ValueError("held_in_real must be non-empty")
        if layer_count is None and num_layers is None:
            raise ValueError("Provide either layer_count (variable) or num_layers (fixed)")
        if layer_count is not None and num_layers is not None:
            raise ValueError("Provide layer_count OR num_layers, not both")

        self.held_in_real = held_in_real
        self.layer_count = layer_count
        self._fixed_num_layers = int(num_layers) if num_layers is not None else None
        self.incidence_angle = float(incidence_angle)
        self.n_synthetic_per_run = int(n_synthetic_per_run)
        self.synthetic_weights = synthetic_weights

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.simulator = OpticalSimulator(incidence_angle=self.incidence_angle)

    def _sample_layer_count(self) -> int:
        if self.layer_count is not None:
            return self.layer_count.sample(self.rng)
        return self._fixed_num_layers

    def _refresh_available_pool(self) -> List[MaterialNK]:
        """Build a fresh (active-real + synthetic) pool for one structure."""
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
        """Sample materials and thicknesses for one structure.

        Layer count is drawn from the configured distribution (or the
        fixed value if `num_layers` was supplied). If `available_pool`
        is None, a fresh pool is built from active real + a new
        synthetic batch.
        """
        if available_pool is None:
            available_pool = self._refresh_available_pool()

        n_layers = self._sample_layer_count()
        if len(available_pool) < n_layers:
            raise ValueError(
                f"available_pool has {len(available_pool)} materials but "
                f"sampled n_layers={n_layers}"
            )

        idxs = self.rng.choice(len(available_pool), size=n_layers, replace=False)
        layer_materials = [available_pool[int(i)] for i in idxs]
        layer_thicknesses = [
            int(t) for t in self.rng.choice(THICKNESS_RANGE_NM, size=n_layers, replace=True)
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
    from collections import Counter
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

    # Single structure with fixed layer count (back-compat).
    sim_fixed = RandomLayerSimulation(
        held_in_real=held_in, num_layers=3, incidence_angle=0,
        n_synthetic_per_run=8, seed=0,
    )
    materials, thicknesses, sRGB = sim_fixed.sample_structure()
    print(f"[smoke] fixed-count sample: layers={len(materials)}, "
          f"thicknesses={thicknesses}, sRGB={sRGB}")
    assert all(0 <= c <= 255 for c in sRGB), "sRGB out of range"

    # Layer count distribution from truncated Poisson(4.5) over [2, 10].
    sim = RandomLayerSimulation(
        held_in_real=held_in,
        layer_count=LayerCountConfig(lam=4.5, min_layers=2, max_layers=10),
        incidence_angle=0, n_synthetic_per_run=8, seed=1,
    )
    counts = Counter()
    for _ in range(200):
        m, t = sim.random_materials_and_thicknesses()
        counts[len(m)] += 1
    print(f"[smoke] Poisson(4.5)[2,10] histogram over 200 draws: "
          f"{dict(sorted(counts.items()))}")
    for k in counts:
        assert 2 <= k <= 10, f"layer count {k} out of range"
    print("[smoke] OK")
