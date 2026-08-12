"""
Synthetic Material Generation
=============================

Generates plausible n,k spectra that look like real materials but aren't any
specific real material. The point is to break the model's incentive to
memorize JaxLayerLumos's ~30 named materials.

Three independent strategies are exposed; the data generator can mix them:

1. `perturb_real`        — Take a real JLL material and apply two
                           band-limited smooth perturbations (a coarse
                           multiplicative envelope + a finer additive
                           smooth detail; no white noise).
                           Output spectra are smooth and close to real ones.

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
from typing import List, Literal, Optional, Tuple

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
# ε ↔ (n, k) conversions + angular-frequency grid. Shared by every strategy
# that operates in ε-space (all three now do, for KK-consistency).
# ============================================================================

# Angular frequency (rad/s) on the canonical wavelength grid.
_OMEGA: np.ndarray = 2.0 * np.pi * CANONICAL_FREQ_HZ


def _nk_to_eps(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(n, k) → complex ε. ε = (n + ik)² for non-magnetic media (μ_r = 1)."""
    return (n + 1j * k) ** 2


def _eps_to_nk(eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complex permittivity → (n, k) via principal sqrt with k ≥ 0.

    sqrt(ε) has two branches; we pick the one with non-negative imaginary
    part, which is the physical convention for passive (absorbing) media.
    """
    sqrt_eps = np.sqrt(eps)
    flip_mask = sqrt_eps.imag < 0
    sqrt_eps = np.where(flip_mask, -sqrt_eps, sqrt_eps)
    return sqrt_eps.real, sqrt_eps.imag


# ============================================================================
# Strategy 1: perturb a real material
#
# Kramers–Kronig-consistent implementation. The naïve approach — perturb
# n(ω) and k(ω) independently with smooth random curves — breaks causality
# because the KK relation couples ε₁ and ε₂ across ALL frequencies. The
# fix used here follows the linearity property of the Hilbert transform:
# adding a causal correction ε_c(ω) to an already-causal ε_base(ω)
# preserves causality. Correction terms are drawn from the same
# Lorentz-oscillator functional form used in parametric_lorentz
# (§Strategy 3), which is causal by construction (Fourier transform of a
# damped-driven-oscillator EOM, poles in the lower-half ω plane).
# ============================================================================


# Small ε-space perturbation presets. Both quantities are dimensionless
# additions in ε; `eps_inf_shift` shifts the frequency-independent
# background level, `max_oscillator_strength` bounds the peak
# amplitude of each Lorentz correction term. Tune against the JLL
# envelope via scripts/visualize_synthetic.py before shipping the next
# large data-generation run.
_PERTURB_PRESETS = {
    "small": dict(eps_inf_shift=0.15, max_oscillator_strength=0.5,
                  n_correction_terms=1),
    "large": dict(eps_inf_shift=0.35, max_oscillator_strength=1.5,
                  n_correction_terms=2),
}


def perturb_real(
    base: MaterialNK,
    rng: np.random.Generator,
    magnitude: Literal["small", "large"] = "small",
    **overrides,
) -> MaterialNK:
    """Additive causal perturbation of a real material's ε(ω).

    Every operation on the complex permittivity is causality-preserving
    by construction:
      1. Convert the base material's (n, k) → ε.
      2. Add a small frequency-independent real shift to the background
         permittivity. Constants trivially satisfy KK (H[const] = 0).
      3. Add 1–2 low-amplitude Lorentz oscillator terms with strictly
         positive damping (γ > 0). Each term is individually causal;
         sums of causal functions are causal.
      4. Convert back to (n, k) via the physical sqrt branch.

    `magnitude` selects a preset. `small` stays visually close to the
    base; `large` explores further. Preset entries can be overridden
    with kwargs (`eps_inf_shift`, `max_oscillator_strength`,
    `n_correction_terms`).
    """
    params = {**_PERTURB_PRESETS[magnitude], **overrides}

    eps = _nk_to_eps(base.n, base.k)

    # (a) Frequency-independent real background shift. Causality-safe:
    # a constant has zero Hilbert transform, so adding it to ε₁ doesn't
    # require any ε₂ change.
    shift = params["eps_inf_shift"]
    eps = eps + float(rng.uniform(-shift, shift))

    # (b) 1–2 additive Lorentz correction terms. Same functional form
    # as parametric_lorentz. Amplitudes deliberately smaller so the
    # perturbation stays in the "recognisably related to base" regime.
    omega = _OMEGA
    omega_min, omega_max = omega.min(), omega.max()
    n_terms = int(params["n_correction_terms"])
    max_f = float(params["max_oscillator_strength"])
    for _ in range(n_terms):
        omega_0 = float(rng.uniform(0.5 * omega_min, 3.0 * omega_max))
        f = float(rng.uniform(0.0, max_f))
        # γ > 0 required for causality (poles in lower-half ω plane).
        gamma = float(rng.uniform(0.02 * omega_0, 0.4 * omega_0))
        eps += f * omega_0 ** 2 / (omega_0 ** 2 - omega ** 2 - 1j * gamma * omega)

    n, k = _eps_to_nk(eps)
    n, k = _clip_physical(n, k)

    return MaterialNK(
        name=f"perturb_{magnitude}({base.name})",
        n=n,
        k=k,
        source=f"synthetic_perturb_{magnitude}",
    )


# ============================================================================
# Strategy 2: linear interpolation between two real materials
#
# Kramers–Kronig-consistent implementation. The naïve linear mix of
# (n, k) is NOT the same as a linear mix of ε (because ε = (n+ik)² is
# quadratic in n, k); the discrepancy is O(w(1−w)·Δn²) even for smooth
# endpoints. Linear combinations with frequency-independent coefficients
# in ε-space ARE causal by linearity of the Hilbert transform, so we mix
# in ε and convert back.
# ============================================================================


def interpolate_real(
    a: MaterialNK,
    b: MaterialNK,
    weight: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> MaterialNK:
    """Convex combination of two materials in ε-space.

    Physically motivated for graded composition / alloy films; also a
    cheap way to fill in the material space between known endpoints.

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

    # Linear mix of ε with frequency-independent weight → causal by the
    # linearity property of the Hilbert transform, provided a and b are
    # each individually causal (true for tabulated real measured data).
    eps_a = _nk_to_eps(a.n, a.k)
    eps_b = _nk_to_eps(b.n, b.k)
    eps_mix = weight * eps_a + (1.0 - weight) * eps_b
    n, k = _eps_to_nk(eps_mix)
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
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.15, 0.35),
) -> List[MaterialNK]:
    """Generate `n_synthetic` materials by mixing four strategies.

    Parameters
    ----------
    real_pool : list of MaterialNK
        Source pool for `perturb_*` and `interpolate_real`. Should be the
        held-IN training subset of JLL materials (held-out reals must NOT
        appear here, otherwise validation leakage).
    n_synthetic : int
    rng : np.random.Generator
    weights : tuple of 4 floats
        Probabilities of
        (perturb_small, perturb_large, interpolate_real, parametric_lorentz).
    """
    real_dependent_weight = weights[0] + weights[1] + weights[2]
    if not real_pool and real_dependent_weight > 0:
        raise ValueError("real_pool empty but real-based strategies have positive weight")

    methods = ["perturb_small", "perturb_large", "interpolate_real", "parametric_lorentz"]
    p = np.array(weights, dtype=np.float64)
    p = p / p.sum()

    out: List[MaterialNK] = []
    for _ in range(n_synthetic):
        method = methods[int(rng.choice(4, p=p))]
        if method == "perturb_small":
            base = real_pool[int(rng.integers(len(real_pool)))]
            out.append(perturb_real(base, rng, magnitude="small"))
        elif method == "perturb_large":
            base = real_pool[int(rng.integers(len(real_pool)))]
            out.append(perturb_real(base, rng, magnitude="large"))
        elif method == "interpolate_real":
            i, j = rng.choice(len(real_pool), size=2, replace=False)
            out.append(interpolate_real(real_pool[int(i)], real_pool[int(j)], rng=rng))
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

    print("\n[smoke] perturb_real on Ag (small):")
    ag = next(m for m in real if m.name == "Ag")
    for _ in range(3):
        s = perturb_real(ag, rng, magnitude="small")
        print(f"  {s.name}: n in [{s.n.min():.3f}, {s.n.max():.3f}], "
              f"k in [{s.k.min():.3f}, {s.k.max():.3f}]")

    print("\n[smoke] perturb_real on Ag (large):")
    for _ in range(3):
        s = perturb_real(ag, rng, magnitude="large")
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
