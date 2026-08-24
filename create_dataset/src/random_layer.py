"""
Random Layer Sampler
====================

Generates a thin-film structure by independently sampling each layer's
material and thickness:

- Number of layers: truncated Poisson via `LayerCountConfig`.
- Per layer: with probability `p_real`, pick uniformly from `held_in_real`;
  otherwise generate a fresh synthetic material via the existing 4-way
  strategy (`generate_synthetic_pool` with `n=1`).
- Thicknesses: uniform over `THICKNESS_RANGE_NM` (5..200 nm in 5 nm steps).

Optical-sim calls go through `src.optical_sim.OpticalSimulator`, which
accepts MaterialNK objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.material_features import MaterialNK
from src.optical_sim import OpticalSimulator
from src.synthetic_materials import generate_synthetic_pool


# Thickness grid (nm). Must match src.materials_vocab.THICKNESSES — the
# vocab token layout depends on every sampled thickness being a legal
# grid point. Grid is 2 nm as of the sensitivity-driven refinement.
THICKNESS_RANGE_NM = np.arange(2, 201, 2)


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
    """Sample structures by independently materializing each layer.

    Parameters
    ----------
    held_in_real : list of MaterialNK
        The "active" set of real materials available to be drawn as a layer
        and used to seed perturb/interpolate synthesis. For training this is
        the held-in set; for Tier-B test set generation this is the held-out
        set (`split_jll_real(use_held_out_reals=True)`).
    layer_count : LayerCountConfig, optional
        Variable layer count via truncated Poisson. Mutually exclusive with
        `num_layers`.
    num_layers : int, optional
        Fixed layer count (back-compat). Mutually exclusive with `layer_count`.
    incidence_angle : float
        Incidence angle in degrees, passed through to the optical sim.
    p_real : float
        Probability that each layer's material is drawn from `held_in_real`.
        The complement is a fresh synthetic material.
    synthetic_weights : tuple of 4 floats
        Mix of (perturb_small, perturb_large, interpolate_real,
        parametric_lorentz) for the synthetic strategies.
    seed : int
        Base seed for the simulation RNG.
    """

    def __init__(
        self,
        held_in_real: List[MaterialNK],
        layer_count: Optional[LayerCountConfig] = None,
        num_layers: Optional[int] = None,
        incidence_angle: float = 0,
        p_real: float = 0.15,
        synthetic_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.15, 0.35),
        seed: int = 42,
    ):
        if not held_in_real:
            raise ValueError("held_in_real must be non-empty")
        if layer_count is None and num_layers is None:
            raise ValueError("Provide either layer_count (variable) or num_layers (fixed)")
        if layer_count is not None and num_layers is not None:
            raise ValueError("Provide layer_count OR num_layers, not both")
        if not (0.0 <= p_real <= 1.0):
            raise ValueError(f"p_real must be in [0, 1], got {p_real}")

        self.held_in_real = held_in_real
        self.layer_count = layer_count
        self._fixed_num_layers = int(num_layers) if num_layers is not None else None
        self.incidence_angle = float(incidence_angle)
        self.p_real = float(p_real)
        self.synthetic_weights = synthetic_weights

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.simulator = OpticalSimulator(incidence_angle=self.incidence_angle)

    def _sample_layer_count(self) -> int:
        if self.layer_count is not None:
            return self.layer_count.sample(self.rng)
        return self._fixed_num_layers

    def _sample_one_material(self) -> MaterialNK:
        """One independent layer-material draw: real with prob p_real, else synthetic."""
        if self.rng.random() < self.p_real:
            return self.held_in_real[int(self.rng.integers(len(self.held_in_real)))]
        return generate_synthetic_pool(
            self.held_in_real,
            n_synthetic=1,
            rng=self.rng,
            weights=self.synthetic_weights,
        )[0]

    def random_materials_and_thicknesses(
        self,
    ) -> Tuple[List[MaterialNK], List[int]]:
        """Sample one structure: independent per-layer material + uniform thickness."""
        n_layers = self._sample_layer_count()
        layer_materials = [self._sample_one_material() for _ in range(n_layers)]
        layer_thicknesses = [
            int(t) for t in self.rng.choice(THICKNESS_RANGE_NM, size=n_layers, replace=True)
        ]
        return layer_materials, layer_thicknesses

    def compute_lab(
        self,
        layer_materials: List[MaterialNK],
        layer_thicknesses: List[int],
    ) -> List[float]:
        """Run the optical simulator and return CIE Lab [L*, a*, b*]."""
        slot_indices = list(range(len(layer_materials)))
        return self.simulator.compute_lab(
            pool=layer_materials,
            slot_indices=slot_indices,
            thicknesses_nm=layer_thicknesses,
        )

    def sample_structure(
        self,
    ) -> Tuple[List[MaterialNK], List[int], List[float]]:
        """Sample one structure and compute its CIE Lab target.

        Returns
        -------
        layer_materials : list of MaterialNK
        layer_thicknesses : list of int
        lab : list of float, shape (3,)
            [L*, a*, b*] in CIE Lab.
        """
        layer_materials, layer_thicknesses = self.random_materials_and_thicknesses()
        lab = self.compute_lab(layer_materials, layer_thicknesses)
        return layer_materials, layer_thicknesses, lab


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

    sim_fixed = RandomLayerSimulation(
        held_in_real=held_in, num_layers=3, incidence_angle=0, p_real=0.15, seed=0,
    )
    materials, thicknesses, lab = sim_fixed.sample_structure()
    print(f"[smoke] fixed-count sample: layers={len(materials)}, "
          f"thicknesses={thicknesses}, "
          f"Lab=[{lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f}]")
    assert 0 <= lab[0] <= 100, f"L* out of range: {lab[0]}"

    sim = RandomLayerSimulation(
        held_in_real=held_in,
        layer_count=LayerCountConfig(lam=4.5, min_layers=2, max_layers=10),
        incidence_angle=0, p_real=0.15, seed=1,
    )
    layer_counts = Counter()
    source_counts = Counter()
    for _ in range(200):
        m, t = sim.random_materials_and_thicknesses()
        layer_counts[len(m)] += 1
        for mat in m:
            source_counts[mat.source] += 1
    print(f"[smoke] Poisson(4.5)[2,10] histogram over 200 draws: "
          f"{dict(sorted(layer_counts.items()))}")
    for k in layer_counts:
        assert 2 <= k <= 10, f"layer count {k} out of range"

    total = sum(source_counts.values())
    print(f"[smoke] structure-layer source distribution over {total} layers:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count} ({100 * count / total:.1f}%)")
    jll_frac = source_counts.get("jaxlayerlumos", 0) / max(total, 1)
    print(f"[smoke] jaxlayerlumos share: {100 * jll_frac:.1f}% (expected ~15%)")
    print("[smoke] OK")
