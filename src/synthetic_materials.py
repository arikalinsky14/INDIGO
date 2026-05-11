"""
Synthetic Material Generation
=============================

Generates plausible n,k spectra that look like real materials but aren't any
specific real material. The point is to break the model's incentive to
memorize JaxLayerLumos's ~30 named materials.

Three independent strategies are exposed; the data generator can mix them:

1. `perturb_real`        — Take a real JLL material and apply structured
                           noise (smooth wavelength-dependent rescaling,
                           small additive jitter on n and k).
                           Output spectra are coherent and close to real ones.

2. `interpolate_real`    — Linearly blend two real JLL materials in n,k space.
                           Output spectra are smooth combinations that can
                           plausibly exist (e.g. SiO2-TiO2 mixing for
                           graded-index coatings is a real fabrication
                           technique).

3. `parametric_lorentz`  — Build n,k from Lorentz oscillators + a Drude term.
                           This is the most expressive: with 1-3 oscillators
                           you can reach metals, transparent dielectrics,
                           band-edge absorbers, and everything in between.
                           No JLL data is consulted at all.

All three return MaterialNK instances on the canonical wavelength grid, with
`source` set to a stable string so callers can stratify validation by source
(e.g. evaluate generalization to held-out real materials separately from
generalization to other synthetic distributions).

Physical bounds enforced after generation:
    0.05 ≤ n ≤ 5.5
    0    ≤ k ≤ 10
This is roughly the envelope spanned by the JLL library plus headroom.
Anything outside is clipped (with a warning printed in dev mode).

Usage in a training data pipeline
---------------------------------
    rng = np.random.default_rng(seed)
    real_pool = load_jll_directory(materials_dir)
    real_held_in = {k: v for k, v in real_pool.items() if k in TRAIN_REAL_NAMES}

    for _ in range(num_synthetic):
        method = rng.choice(['perturb_real', 'interpolate_real', 'parametric_lorentz'],
                            p=[0.4, 0.2, 0.4])
        if method == 'perturb_real':
            mat = perturb_real(rng.choice(list(real_held_in.values())), rng)
        elif method == 'interpolate_real':
            a, b = rng.choice(list(real_held_in.values()), size=2, replace=False)
            mat = interpolate_real(a, b, rng=rng)
        else:
            mat = parametric_lorentz(rng)
        synthetic_pool.append(mat)
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from src.material_features import (
    CANONICAL_FREQ_HZ,
    CANONICAL_LAMBDA_NM,
    NUM_LAMBDA,
    MaterialNK,
)


# ============================================================================
# Physical bounds
# ============================================================================

N_MIN: float = 0.05
N_MAX: float = 5.5
K_MIN: float = 0.0
K_MAX: float = 10.0


def _clip_physical(n: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip n,k into the physically realistic envelope. Modifies copies."""
    return np.clip(n, N_MIN, N_MAX), np.clip(k, K_MIN, K_MAX)


# ============================================================================
# Strategy 1: perturb a real material
# ============================================================================


def _smooth_random_curve(
    rng: np.random.Generator,
    num_points: int,
    num_components: int = 4,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Build a smooth wavelength-dependent multiplier curve via low-frequency
    Fourier components. Output is a [num_points] array centred near zero with
    the requested amplitude scale."""
    x = np.linspace(0.0, 1.0, num_points)
    curve = np.zeros(num_points)
    for j in range(1, num_components + 1):
        coef_sin = rng.normal(0.0, 1.0 / j)
        coef_cos = rng.normal(0.0, 1.0 / j)
        curve += coef_sin * np.sin(2.0 * np.pi * j * x)
        curve += coef_cos * np.cos(2.0 * np.pi * j * x)
    # Normalize to roughly the requested amplitude (max abs deviation).
    if np.max(np.abs(curve)) > 1e-8:
        curve = curve / np.max(np.abs(curve)) * amplitude
    return curve


def perturb_real(
    base: MaterialNK,
    rng: np.random.Generator,
    n_scale_amp: float = 0.15,
    k_scale_amp: float = 0.30,
    n_jitter_amp: float = 0.05,
    k_jitter_amp: float = 0.05,
) -> MaterialNK:
    """Apply structured perturbation to a real material's n,k.

    Two perturbations:
    - Multiplicative smooth curve (low-frequency Fourier) — shifts the overall
      shape of the dispersion without introducing high-frequency artefacts.
    - Small white-noise jitter — breaks exact equivalence with the source.

    Defaults are chosen so the output is recognisably 'in the same class' as
    the source (e.g. perturbing Ag still yields a metal-like spectrum) but
    never identical. Tune amplitudes upward to broaden the synthetic family.

    Parameters
    ----------
    base : MaterialNK
        Source material to perturb.
    rng : np.random.Generator
    n_scale_amp, k_scale_amp : float
        Max fractional deviation of the smooth multiplier curve.
    n_jitter_amp, k_jitter_amp : float
        White-noise sigma added on top.
    """
    n_curve = 1.0 + _smooth_random_curve(rng, NUM_LAMBDA, amplitude=n_scale_amp)
    k_curve = 1.0 + _smooth_random_curve(rng, NUM_LAMBDA, amplitude=k_scale_amp)

    n = base.n * n_curve + rng.normal(0.0, n_jitter_amp, size=NUM_LAMBDA)
    k = base.k * k_curve + rng.normal(0.0, k_jitter_amp, size=NUM_LAMBDA)
    n, k = _clip_physical(n, k)

    return MaterialNK(
        name=f"perturb({base.name})",
        n=n,
        k=k,
        source="synthetic_perturb",
    )


# ============================================================================
# Strategy 2: linear interpolation between two real materials
# ============================================================================


def interpolate_real(
    a: MaterialNK,
    b: MaterialNK,
    weight: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> MaterialNK:
    """Convex combination of two materials' n,k.

    This is physically motivated for some real fabrication contexts (graded
    composition, alloy films) and serves as a cheap way to fill in the
    space between known materials.

    Parameters
    ----------
    a, b : MaterialNK
    weight : float, optional
        Mixing weight for `a`. If None, sampled uniformly from (0.1, 0.9)
        to avoid degenerate near-copies of either endpoint.
    rng : np.random.Generator, required if weight is None.
    """
    if weight is None:
        if rng is None:
            raise ValueError("Provide either `weight` or `rng`.")
        weight = float(rng.uniform(0.1, 0.9))

    n = weight * a.n + (1.0 - weight) * b.n
    k = weight * a.k + (1.0 - weight) * b.k
    n, k = _clip_physical(n, k)

    return MaterialNK(
        name=f"interp({a.name},{b.name},w={weight:.2f})",
        n=n,
        k=k,
        source="synthetic_interp",
    )


# ============================================================================
# Strategy 3: parametric Lorentz / Drude model
# ============================================================================
#
# We build the complex permittivity ε(ω) = ε∞ + Σ_j L_j(ω) − D(ω) where:
#
#   Lorentz oscillator j (interband / phonon resonance):
#       L_j(ω) = f_j ω0_j² / (ω0_j² − ω² − i γ_j ω)
#
#   Drude term (free electrons; only enabled for "metallic" materials):
#       D(ω) = ω_p² / (ω² + i ω γ_D)
#
# Then n + ik = sqrt(ε), with the principal-branch convention chosen so
# that k ≥ 0.
#
# Parameter ranges below were tuned by comparing the resulting (n, k) spectra
# to the JaxLayerLumos library; the defaults produce material-like shapes
# spanning low-loss dielectrics through lossy semiconductors and metals.

# Convert canonical frequency grid (Hz) to angular frequency (rad/s)
_OMEGA: np.ndarray = 2.0 * np.pi * CANONICAL_FREQ_HZ


def _eps_to_nk(eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complex permittivity → (n, k) via principal sqrt with k ≥ 0.

    sqrt(ε) has two branches; we pick the one with non-negative imaginary
    part, which is the physical convention for passive (absorbing) media.
    """
    sqrt_eps = np.sqrt(eps)
    # If by branch choice we got k < 0, flip the sign of the whole sqrt.
    flip_mask = sqrt_eps.imag < 0
    sqrt_eps = np.where(flip_mask, -sqrt_eps, sqrt_eps)
    return sqrt_eps.real, sqrt_eps.imag


def parametric_lorentz(
    rng: np.random.Generator,
    metallic_prob: float = 0.25,
    max_oscillators: int = 3,
) -> MaterialNK:
    """Generate a synthetic material from Lorentz (+optional Drude) parameters.

    With probability `metallic_prob`, includes a Drude term so the result is
    a metal. Otherwise the result is a dielectric or semiconductor with one
    or more Lorentz resonances.

    The resonance positions are sampled in ω-space such that they can lie
    anywhere from below the visible (mid-IR) up through the UV. This means
    some oscillators sit outside our 300–900 nm window and only contribute
    a tail — that's intentional, real materials have many resonances and we
    only see the in-band consequences.

    Parameter ranges
    ----------------
    ε∞               ∈ [1.0, 4.0]                 background permittivity
    num_oscillators  ∈ {1, ..., max_oscillators}
    For each oscillator:
        ω0_j         ∈ [0.5×ω_min, 3×ω_max]       resonance frequency
        f_j          ∈ [0.5, 6.0]                 oscillator strength
        γ_j          ∈ [0.02×ω0_j, 0.4×ω0_j]      damping
    Drude (if metallic):
        ω_p          ∈ [0.8×ω_max, 4×ω_max]       plasma frequency
        γ_D          ∈ [0.005×ω_p, 0.05×ω_p]      damping
    """
    omega = _OMEGA
    omega_min = omega.min()
    omega_max = omega.max()

    eps_inf = float(rng.uniform(1.0, 4.0))
    eps = np.full(NUM_LAMBDA, eps_inf, dtype=np.complex128)

    # Lorentz oscillators
    n_osc = int(rng.integers(1, max_oscillators + 1))
    for _ in range(n_osc):
        omega_0 = float(rng.uniform(0.5 * omega_min, 3.0 * omega_max))
        f = float(rng.uniform(0.5, 6.0))
        gamma = float(rng.uniform(0.02 * omega_0, 0.4 * omega_0))
        eps += f * omega_0**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

    # Drude term for metallic materials
    is_metal = bool(rng.random() < metallic_prob)
    if is_metal:
        omega_p = float(rng.uniform(0.8 * omega_max, 4.0 * omega_max))
        gamma_d = float(rng.uniform(0.005 * omega_p, 0.05 * omega_p))
        eps -= omega_p**2 / (omega**2 + 1j * omega * gamma_d)

    n, k = _eps_to_nk(eps)
    n, k = _clip_physical(n, k)

    label = "metal" if is_metal else "dielectric"
    return MaterialNK(
        name=f"lorentz_{label}_n{n_osc}",
        n=n,
        k=k,
        source="synthetic_lorentz",
    )


# ============================================================================
# Top-level dispatcher
# ============================================================================


def generate_synthetic_pool(
    real_pool: List[MaterialNK],
    n_synthetic: int,
    rng: np.random.Generator,
    weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> List[MaterialNK]:
    """Generate `n_synthetic` materials by mixing the three strategies.

    Parameters
    ----------
    real_pool : list of MaterialNK
        Source pool for `perturb_real` and `interpolate_real`. Should be the
        held-IN training subset of JLL materials (held-out reals must NOT
        appear here, otherwise validation leakage).
    n_synthetic : int
    rng : np.random.Generator
    weights : tuple of 3 floats
        Probabilities of (perturb_real, interpolate_real, parametric_lorentz).
        Default favours perturb and parametric (which produce the broadest
        diversity) over interpolate (which can be derivative).
    """
    if not real_pool and (weights[0] > 0 or weights[1] > 0):
        raise ValueError("real_pool empty but real-based strategies have positive weight")

    methods = ["perturb_real", "interpolate_real", "parametric_lorentz"]
    p = np.array(weights, dtype=np.float64)
    p = p / p.sum()

    out: List[MaterialNK] = []
    for _ in range(n_synthetic):
        method = methods[rng.choice(3, p=p)]
        if method == "perturb_real":
            base = real_pool[rng.integers(len(real_pool))]
            out.append(perturb_real(base, rng))
        elif method == "interpolate_real":
            i, j = rng.choice(len(real_pool), size=2, replace=False)
            out.append(interpolate_real(real_pool[i], real_pool[j], rng=rng))
        else:
            out.append(parametric_lorentz(rng))
    return out


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

    real = list(load_jll_directory(materials_dir).values())
    print(f"[smoke] Loaded {len(real)} real materials.")

    rng = np.random.default_rng(42)

    print("\n[smoke] perturb_real on Ag:")
    ag = next(m for m in real if m.name == "Ag")
    for _ in range(3):
        s = perturb_real(ag, rng)
        print(f"  {s.name}: n in [{s.n.min():.3f}, {s.n.max():.3f}], "
              f"k in [{s.k.min():.3f}, {s.k.max():.3f}]")

    print("\n[smoke] interpolate_real(SiO2, TiO2):")
    sio2 = next(m for m in real if m.name == "SiO2")
    tio2 = next(m for m in real if m.name == "TiO2")
    for _ in range(3):
        s = interpolate_real(sio2, tio2, rng=rng)
        print(f"  {s.name}: n in [{s.n.min():.3f}, {s.n.max():.3f}], "
              f"k in [{s.k.min():.3f}, {s.k.max():.3f}]")

    print("\n[smoke] parametric_lorentz random samples:")
    for _ in range(5):
        s = parametric_lorentz(rng)
        print(f"  {s.name}: n in [{s.n.min():.3f}, {s.n.max():.3f}], "
              f"k in [{s.k.min():.3f}, {s.k.max():.3f}]")

    print("\n[smoke] generate_synthetic_pool of 100:")
    pool = generate_synthetic_pool(real, 100, rng)
    sources = [m.source for m in pool]
    from collections import Counter
    print(f"  Source distribution: {Counter(sources)}")
    print(f"  All physical: "
          f"{all(np.all(m.n > 0) and np.all(m.k >= 0) for m in pool)}")
    print("[smoke] OK")
