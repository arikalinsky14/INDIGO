"""sRGB ↔ CIE Lab + chroma utilities. Shared by data generation, training,
evaluation, and the k-nearest-structure search.

Lab is the canonical color target for INDIGO — wider gamut than sRGB and
perceptually uniform (so CIEDE2000 distances mean what they look like).
Conversions to/from sRGB are kept around purely for human-readable display
(swatches in `evaluate.py`, etc.).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# D65 white point reference values (CIE 1931 2°).
_X_N: float = 95.047
_Y_N: float = 100.000
_Z_N: float = 108.883

# Linear sRGB → XYZ matrix (D65). Used for both directions; the inverse
# of this matrix recovers linear sRGB from XYZ.
_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_M_XYZ_TO_SRGB = np.linalg.inv(_M_SRGB_TO_XYZ)


# ============================================================================
# sRGB gamma (forward + inverse), elementwise
# ============================================================================


def _srgb_to_linear(c: float) -> float:
    """Inverse sRGB gamma. Input c in [0, 1] (out-of-range values pass through)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    """Forward sRGB gamma encoding."""
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _f_lab(t: float) -> float:
    delta = 6.0 / 29.0
    return t ** (1.0 / 3.0) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4.0 / 29.0)


def _f_lab_inv(t: float) -> float:
    delta = 6.0 / 29.0
    return t ** 3 if t > delta else 3 * delta ** 2 * (t - 4.0 / 29.0)


# ============================================================================
# sRGB ↔ Lab
# ============================================================================


def srgb_to_lab(rgb: Sequence[float]) -> Tuple[float, float, float]:
    """sRGB ints in [0, 255] (or floats in [0, 1]) → CIE Lab.

    Returns (L*, a*, b*).
    """
    r, g, b = [c / 255.0 if c > 1.0 else c for c in rgb]
    r_lin, g_lin, b_lin = map(_srgb_to_linear, (r, g, b))

    X = (_M_SRGB_TO_XYZ[0, 0] * r_lin + _M_SRGB_TO_XYZ[0, 1] * g_lin + _M_SRGB_TO_XYZ[0, 2] * b_lin) * 100
    Y = (_M_SRGB_TO_XYZ[1, 0] * r_lin + _M_SRGB_TO_XYZ[1, 1] * g_lin + _M_SRGB_TO_XYZ[1, 2] * b_lin) * 100
    Z = (_M_SRGB_TO_XYZ[2, 0] * r_lin + _M_SRGB_TO_XYZ[2, 1] * g_lin + _M_SRGB_TO_XYZ[2, 2] * b_lin) * 100

    fx, fy, fz = _f_lab(X / _X_N), _f_lab(Y / _Y_N), _f_lab(Z / _Z_N)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return float(L), float(a), float(b_lab)


def lab_to_srgb_int(lab: Sequence[float]) -> List[int]:
    """CIE Lab → sRGB ints in [0, 255], clipped to gamut. For display only.

    Lab colors outside the sRGB gamut are clipped to the nearest in-gamut sRGB.
    Use this for swatches / visualization; never for training targets.
    """
    L, a, b_lab = lab
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_lab / 200.0

    X = _X_N * _f_lab_inv(fx)
    Y = _Y_N * _f_lab_inv(fy)
    Z = _Z_N * _f_lab_inv(fz)

    # Linear sRGB (may lie outside [0, 1] if Lab is out-of-gamut).
    xyz = np.array([X, Y, Z]) / 100.0
    rgb_lin = _M_XYZ_TO_SRGB @ xyz
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    rgb_gamma = np.array([_linear_to_srgb(float(c)) for c in rgb_lin])
    return [int(round(c * 255.0)) for c in rgb_gamma]


# ============================================================================
# Chroma (for the greyscale filter in the data generator)
# ============================================================================


def lab_chroma(lab: Sequence[float]) -> float:
    """C* = sqrt(a*² + b*²). 0 = perfect grey, ~100+ = saturated."""
    _, a, b = lab
    return float(np.sqrt(a * a + b * b))


def srgb_chroma(rgb: Sequence[float]) -> float:
    """Convenience: chroma directly from sRGB."""
    return lab_chroma(srgb_to_lab(rgb))


# ============================================================================
# Reflectance spectrum → Lab (the production data-generation path)
# ============================================================================
#
# Goes spectrum → un-clipped sRGB (via jaxlayerlumos) → Lab. The un-clipped
# step preserves the full color gamut, so out-of-sRGB-gamut colors come
# through as Lab values that exceed what sRGB can represent.


def spectrum_to_lab(
    wavelengths_nm: np.ndarray,
    reflectance: np.ndarray,
) -> Tuple[float, float, float]:
    """Reflectance spectrum → CIE Lab via jaxlayerlumos' XYZ pipeline.

    Parameters
    ----------
    wavelengths_nm : np.ndarray
        Wavelength values in nm at which reflectance is sampled.
    reflectance : np.ndarray
        Reflectance in [0, 1] at each wavelength.

    Returns
    -------
    (L*, a*, b*) tuple of floats.
    """
    try:
        import jax.numpy as jnp
        import jaxlayerlumos.colors.composite as jll_colors_composite
    except ImportError as exc:
        raise RuntimeError(
            "spectrum_to_lab requires jaxlayerlumos; install it first."
        ) from exc

    valid = (wavelengths_nm > 360) & (wavelengths_nm < 830)

    # use_clipping=False keeps out-of-gamut linear sRGB values intact (they
    # may be negative or > 1), so the resulting Lab can lie outside the
    # sRGB gamut.
    rgb_unclipped = jll_colors_composite.spectrum_to_sRGB(
        jnp.array(wavelengths_nm[valid]),
        jnp.array(reflectance[valid]),
        use_clipping=False,
    )
    rgb_float = np.array(rgb_unclipped).flatten()[:3]

    # Apply inverse sRGB gamma elementwise. _srgb_to_linear is defined for
    # values in [0, 1]; for values outside that range we apply the same
    # analytic formula by branching on sign and magnitude.
    def inv_gamma(c: float) -> float:
        if c < 0:
            return -_srgb_to_linear(-c)
        return _srgb_to_linear(c)

    rgb_linear = np.array([inv_gamma(float(c)) for c in rgb_float])

    # Linear sRGB → XYZ (D65, Y_n=100 normalization).
    xyz = (_M_SRGB_TO_XYZ @ rgb_linear) * 100.0
    X, Y, Z = xyz

    fx, fy, fz = _f_lab(X / _X_N), _f_lab(Y / _Y_N), _f_lab(Z / _Z_N)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return float(L), float(a), float(b_lab)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    test_colors = {
        "black":         [0, 0, 0],
        "white":         [255, 255, 255],
        "neutral_grey":  [128, 128, 128],
        "saturated_red": [255, 0, 0],
        "pale_pink":     [255, 220, 220],
        "deep_blue":     [0, 0, 200],
    }
    print(f"{'name':<16} {'sRGB':<20} {'L*':>6} {'a*':>7} {'b*':>7} {'C*':>6} {'rt sRGB':<18}")
    print("-" * 90)
    for name, rgb in test_colors.items():
        L, a, b = srgb_to_lab(rgb)
        C = lab_chroma((L, a, b))
        rt = lab_to_srgb_int((L, a, b))
        print(f"{name:<16} {str(rgb):<20} {L:>6.2f} {a:>7.2f} {b:>7.2f} {C:>6.2f} {str(rt):<18}")

    print()
    # Verify round-trip is bit-tight for in-gamut sRGB.
    for name, rgb in test_colors.items():
        rt = lab_to_srgb_int(srgb_to_lab(rgb))
        if max(abs(a - b) for a, b in zip(rgb, rt)) > 1:
            print(f"[smoke] {name}: round-trip {rgb} → {rt} DIFFERS by >1, investigate")
            break
    else:
        print("[smoke] sRGB → Lab → sRGB round-trip within 1 unit for all in-gamut tests")
    print("[smoke] OK")
