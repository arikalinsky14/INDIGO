"""sRGB ↔ CIE Lab + chroma utilities. Shared by data generation and evaluation."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# D65 white point reference values (CIE 1931 2°).
_X_N: float = 95.047
_Y_N: float = 100.000
_Z_N: float = 108.883


def _srgb_to_linear(c: float) -> float:
    """Inverse sRGB gamma. Input c in [0, 1]."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _f_lab(t: float) -> float:
    """Lab nonlinearity."""
    delta = 6.0 / 29.0
    return t ** (1.0 / 3.0) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4.0 / 29.0)


def srgb_to_lab(rgb: Sequence[float]) -> Tuple[float, float, float]:
    """sRGB ints in [0, 255] (or floats in [0, 1]) → CIE Lab.

    Returns (L*, a*, b*).
    """
    # Normalise to [0, 1]. A value strictly > 1 is interpreted as 0-255 int.
    r, g, b = [c / 255.0 if c > 1.0 else c for c in rgb]
    r_lin, g_lin, b_lin = map(_srgb_to_linear, (r, g, b))

    # Linear sRGB → XYZ (D65), scaled so Y_n = 100.
    X = (0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin) * 100
    Y = (0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin) * 100
    Z = (0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin) * 100

    fx, fy, fz = _f_lab(X / _X_N), _f_lab(Y / _Y_N), _f_lab(Z / _Z_N)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return float(L), float(a), float(b_lab)


def lab_chroma(lab: Tuple[float, float, float]) -> float:
    """C* = sqrt(a*² + b*²). 0 = perfect grey, ~100+ = saturated."""
    _, a, b = lab
    return float(np.sqrt(a * a + b * b))


def srgb_chroma(rgb: Sequence[float]) -> float:
    """Convenience: chroma directly from sRGB."""
    return lab_chroma(srgb_to_lab(rgb))


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
    print(f"{'name':<16} {'sRGB':<20} {'L*':>6} {'a*':>7} {'b*':>7} {'C*':>6}")
    print("-" * 64)
    for name, rgb in test_colors.items():
        L, a, b = srgb_to_lab(rgb)
        C = lab_chroma((L, a, b))
        print(f"{name:<16} {str(rgb):<20} {L:>6.2f} {a:>7.2f} {b:>7.2f} {C:>6.2f}")
    print("\n[smoke] OK")
