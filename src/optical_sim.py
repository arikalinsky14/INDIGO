"""
Optical Simulator for Flexible Materials
========================================

Computes reflectance / sRGB color from user-supplied n,k spectra in a
multilayer thin-film stack. This is the canonical optical simulator for
INDIGO — it accepts arbitrary `MaterialNK` objects rather than named
materials in the JaxLayerLumos library, which means synthetic materials
(Lorentz oscillators, perturbed/interpolated reals) drop in transparently.

The wavelength grid in `src.material_features.CANONICAL_FREQ_HZ` is
defined to match the transfer-matrix grid bit-for-bit, so no interpolation
is needed at simulation time.

Usage
-----
    sim = OpticalSimulator(incidence_angle=0)
    rgb = sim.compute_color(pool=[...MaterialNK...],
                             slot_indices=[2, 0, 5],
                             thicknesses_nm=[100, 50, 75])

This module depends on `jaxlayerlumos` only for:
- `stackrt_n_k` (the transfer-matrix calculation)
- `colors.composite.spectrum_to_sRGB` (CIE colour matching)

It does NOT depend on `jaxlayerlumos.utils_materials.get_n_k`, which is
the part hard-coded to the JLL material registry.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.material_features import (
    CANONICAL_FREQ_HZ,
    CANONICAL_LAMBDA_NM,
    NUM_LAMBDA,
    MaterialNK,
)


_JAX_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import scipy.constants as scic
    from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
    import jaxlayerlumos.colors.composite as jll_colors_composite

    _JAX_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def is_available() -> bool:
    return _JAX_AVAILABLE


def get_import_error() -> Optional[str]:
    return _IMPORT_ERROR


# ============================================================================
# Air & substrate (default surroundings — match original CHROMA-Lite)
# ============================================================================

# Air: n=1, k=0 across all wavelengths.
_AIR_N: np.ndarray = np.ones(NUM_LAMBDA, dtype=np.float64)
_AIR_K: np.ndarray = np.zeros(NUM_LAMBDA, dtype=np.float64)

# Fused silica substrate using the Malitson-1965 Sellmeier model evaluated
# on our canonical grid. We compute it once at import time so this module
# does not depend on the JLL CSV registry at all.
def _fused_silica_n() -> np.ndarray:
    """Sellmeier dispersion for fused silica (Malitson 1965, valid 0.21–6.7 μm)."""
    lam_um = CANONICAL_LAMBDA_NM / 1000.0
    lam2 = lam_um**2
    n2 = (
        1.0
        + 0.6961663 * lam2 / (lam2 - 0.0684043**2)
        + 0.4079426 * lam2 / (lam2 - 0.1162414**2)
        + 0.8974794 * lam2 / (lam2 - 9.896161**2)
    )
    return np.sqrt(n2)


_SUBSTRATE_N: np.ndarray = _fused_silica_n()
_SUBSTRATE_K: np.ndarray = np.zeros(NUM_LAMBDA, dtype=np.float64)


# ============================================================================
# sRGB conversion (delegates to JLL's CIE pipeline)
# ============================================================================


def spectrum_to_sRGB(reflectance: np.ndarray) -> List[int]:
    """Convert a reflectance spectrum on the canonical grid to sRGB [0, 255]."""
    if not _JAX_AVAILABLE:
        raise RuntimeError(f"jaxlayerlumos not available: {_IMPORT_ERROR}")
    # Restrict to wavelengths inside CIE colour-matching support (360-830 nm).
    valid = (CANONICAL_LAMBDA_NM > 360) & (CANONICAL_LAMBDA_NM < 830)
    rgb = jll_colors_composite.spectrum_to_sRGB(
        jnp.array(CANONICAL_LAMBDA_NM[valid]),
        jnp.array(reflectance[valid]),
        use_clipping=True,
    )
    rgb_arr = np.array(rgb).flatten()[:3] * 255.0
    return np.round(rgb_arr).astype(int).tolist()


# ============================================================================
# Main simulator
# ============================================================================


class OpticalSimulator:
    """Compute reflectance / sRGB from arbitrary n,k spectra and a layer stack.

    Mirrors `src.optical_sim.OpticalSimulator` from CHROMA-Lite but takes
    a list of `MaterialNK` plus a layer specification (slot indices and
    thicknesses) rather than material names.
    """

    def __init__(self, incidence_angle: float = 0.0):
        if not _JAX_AVAILABLE:
            raise RuntimeError(f"jaxlayerlumos not available: {_IMPORT_ERROR}")
        self.incidence_angle = float(incidence_angle)

    def compute_reflectance(
        self,
        pool: List[MaterialNK],
        slot_indices: List[int],
        thicknesses_nm: List[int],
    ) -> np.ndarray:
        """Reflectance spectrum (TE+TM averaged) on the canonical grid.

        Air is prepended as the superstrate, fused silica as the substrate.
        This matches the dataset-generation conventions in the original
        CHROMA-Lite, so simulator outputs are directly comparable to
        cached training-data RGBs.
        """
        if len(slot_indices) != len(thicknesses_nm):
            raise ValueError("slot_indices and thicknesses_nm must align")
        if len(slot_indices) == 0:
            return np.zeros(NUM_LAMBDA, dtype=np.float64)

        # Build the layer-by-layer (n, k) stack: [Air, layer_0, ..., layer_N-1, Substrate].
        n_per_layer = [_AIR_N]
        k_per_layer = [_AIR_K]
        for slot in slot_indices:
            mat = pool[slot]
            n_per_layer.append(mat.n)
            k_per_layer.append(mat.k)
        n_per_layer.append(_SUBSTRATE_N)
        k_per_layer.append(_SUBSTRATE_K)

        # Assemble n + ik with the layout JLL's stackrt_n_k expects:
        # shape [num_freqs, num_layers], complex.
        n_stack = np.stack(n_per_layer, axis=1)    # [NUM_LAMBDA, num_layers]
        k_stack = np.stack(k_per_layer, axis=1)
        n_complex = jnp.array(n_stack + 1j * k_stack)

        # Thicknesses: prepend 0 (air), append 0 (substrate). Convert to metres.
        d_full_nm = [0] + list(thicknesses_nm) + [0]
        d_jax = jnp.array(d_full_nm, dtype=jnp.float32) * scic.nano

        thetas = jnp.array([self.incidence_angle], dtype=jnp.float32)
        freqs_jax = jnp.array(CANONICAL_FREQ_HZ)
        R_TE, _, R_TM, _ = stackrt_n_k(n_complex, d_jax, freqs_jax, thetas)
        R_avg = (R_TE[0] + R_TM[0]) / 2.0
        return np.array(R_avg)

    def compute_color(
        self,
        pool: List[MaterialNK],
        slot_indices: List[int],
        thicknesses_nm: List[int],
    ) -> List[int]:
        """sRGB [0, 255] for the given (pool, structure)."""
        R = self.compute_reflectance(pool, slot_indices, thicknesses_nm)
        return spectrum_to_sRGB(R)


# ============================================================================
# Smoke test — only runs if jaxlayerlumos is installed
# ============================================================================

if __name__ == "__main__":
    if not _JAX_AVAILABLE:
        print(f"[smoke] jaxlayerlumos not installed ({_IMPORT_ERROR}); skipping.")
        raise SystemExit(0)

    from pathlib import Path
    from src.material_features import load_jll_directory

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    pool_full = load_jll_directory(materials_dir)
    pool = [pool_full["SiO2"], pool_full["Ag"], pool_full["TiO2"]]

    sim = OpticalSimulator(incidence_angle=0)

    # Single-layer Ag test (mirror): expect a reflective near-white.
    rgb_mirror = sim.compute_color(pool=pool, slot_indices=[1], thicknesses_nm=[100])
    print(f"100nm Ag mirror sRGB: {rgb_mirror}")

    # Three-layer SiO2/Ag/TiO2 test.
    rgb_stack = sim.compute_color(
        pool=pool, slot_indices=[0, 1, 2], thicknesses_nm=[100, 30, 75]
    )
    print(f"SiO2(100)/Ag(30)/TiO2(75) sRGB: {rgb_stack}")

    print("[smoke] OK")
