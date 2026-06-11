"""
JAX-native differentiable physics chain for INDIGO inference.

  thicknesses_nm (float)  ─►  reflectance  ─►  Lab  ─►  ΔE00 vs target

Every step is `jax.grad`-compatible. The stack geometry matches
`src.optical_sim.OpticalSimulator`:

  [Air, layer_0, ..., layer_{MAX_LAYERS-1}, Fused-silica substrate]

Inactive layers are represented with `thickness = 0` and `layer_mask = False`
— the TMM is identity on a 0-thick layer, so masked layers contribute
nothing to reflectance. Keeping the shape fixed at `MAX_LAYERS+2` is what
keeps the chain well-formed even when the structure has fewer than
MAX_LAYERS active deposits.

A note on `jax.jit` / `jax.vmap`
-------------------------------
`jaxlayerlumos.jaxlayerlumos.stackrt_eps_mu_base` contains
`assert thicknesses[0] == 0` (a precondition that the air superstrate is at
position 0). Python `assert` calls `__bool__` on its argument, and under
`jax.jit` / `jax.vmap` the argument is a Tracer — boolean conversion fails
with `TracerBoolConversionError`. `jax.grad` is fine because grad tracing
keeps concrete values around for the forward pass.

Practical consequence: every candidate's simulation is computed sequentially.
This is fine for the ensemble sizes the inference plan calls for (~500
candidates). If we ever need vmap speed (e.g. larger ensembles, hyperparameter
sweeps), the options are:
  - monkey-patch the JLL assert away at module load time, or
  - reimplement the (very small) normal-incidence TMM in pure JAX.
Both are tracked work; neither blocks the current build.

Why a JAX reimplementation of ΔE00
----------------------------------
`src.color_utils.ciede2000` does not exist yet. `colormath` is numpy-only.
ΔE00 is differentiable almost everywhere (the chroma=0 axis and the
hue-rotation centre at 275° are measure-zero kinks); we add small
denominator epsilons so `jax.grad` never sees a 0/0.

Lab path matches training
-------------------------
For numerical fidelity with the training data, we route through the same
`jaxlayerlumos.colors.composite.spectrum_to_sRGB` that `spectrum_to_lab`
in `src.color_utils` uses — then inverse-gamma → linear sRGB → XYZ → Lab.
Going direct (reflectance → XYZ via CMF integration) would be cleaner but
would introduce a small offset against the labels the model was trained on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Repo root for `src.*` imports.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
from jaxlayerlumos.colors.composite import spectrum_to_sRGB

from src.material_features import CANONICAL_FREQ_HZ, CANONICAL_LAMBDA_NM, NUM_LAMBDA
from src.materials_vocab import MAX_LAYERS, M_MAX

# ----------------------------------------------------------------------------
# Substrate (fused silica, Malitson 1965). Match src/optical_sim.py exactly.
# ----------------------------------------------------------------------------

def _malitson_sellmeier(lambda_nm: np.ndarray) -> np.ndarray:
    """Malitson-1965 Sellmeier for fused silica, returns real n on the grid."""
    L_um = lambda_nm / 1000.0
    L2 = L_um * L_um
    n_sq = (
        1.0
        + 0.6961663 * L2 / (L2 - 0.0684043 ** 2)
        + 0.4079426 * L2 / (L2 - 0.1162414 ** 2)
        + 0.8974794 * L2 / (L2 - 9.896161 ** 2)
    )
    return np.sqrt(n_sq)


_SUBSTRATE_N_NP = _malitson_sellmeier(CANONICAL_LAMBDA_NM).astype(np.float32)
_SUBSTRATE_K_NP = np.zeros(NUM_LAMBDA, dtype=np.float32)
_AIR_N_NP = np.ones(NUM_LAMBDA, dtype=np.float32)
_AIR_K_NP = np.zeros(NUM_LAMBDA, dtype=np.float32)

SUBSTRATE_N = jnp.asarray(_SUBSTRATE_N_NP)
SUBSTRATE_K = jnp.asarray(_SUBSTRATE_K_NP)
AIR_N = jnp.asarray(_AIR_N_NP)
AIR_K = jnp.asarray(_AIR_K_NP)
FREQS_HZ = jnp.asarray(CANONICAL_FREQ_HZ.astype(np.float64))
LAMBDA_NM = jnp.asarray(CANONICAL_LAMBDA_NM.astype(np.float64))


# ----------------------------------------------------------------------------
# Reflectance — fixed-shape JAX path
# ----------------------------------------------------------------------------

def compute_reflectance(
    pool_n: jax.Array,           # [M_MAX, NUM_LAMBDA]  real
    pool_k: jax.Array,           # [M_MAX, NUM_LAMBDA]  real
    slot_indices: jax.Array,     # [MAX_LAYERS]         int — slot used at each layer
    thicknesses_nm: jax.Array,   # [MAX_LAYERS]         float — continuous nm
    layer_mask: jax.Array,       # [MAX_LAYERS]         bool — True for active layers
    incidence_angle: float = 0.0,
) -> jax.Array:                  # [NUM_LAMBDA] reflectance (TE+TM averaged)
    """Reflectance on the canonical wavelength grid.

    Fixed shape: always builds an `MAX_LAYERS+2`-layer stack (Air + slots +
    fused silica). Masked layers get `d=0` so the transfer matrix reduces
    to identity and they don't perturb reflectance.
    """
    # Gather per-layer n,k from pool by slot index. Masked positions index
    # whatever slot (their thickness is zeroed below), so the n,k value
    # there doesn't matter.
    layer_n = pool_n[slot_indices]               # [MAX_LAYERS, NUM_LAMBDA]
    layer_k = pool_k[slot_indices]

    # Air | layers | substrate, in the order JLL expects.
    n_full = jnp.concatenate([AIR_N[None, :], layer_n, SUBSTRATE_N[None, :]], axis=0)
    k_full = jnp.concatenate([AIR_K[None, :], layer_k, SUBSTRATE_K[None, :]], axis=0)
    # JLL wants [NUM_LAMBDA, num_layers], complex.
    n_complex = (n_full + 1j * k_full).T          # [NUM_LAMBDA, MAX_LAYERS+2]

    # Thicknesses in metres. Air and substrate get d=0 (sentinels); active
    # layer thicknesses get the gradient-bearing value, masked ones get 0.
    d_layers_m = thicknesses_nm * layer_mask.astype(thicknesses_nm.dtype) * 1e-9
    d_full = jnp.concatenate([jnp.zeros(1, dtype=d_layers_m.dtype),
                              d_layers_m,
                              jnp.zeros(1, dtype=d_layers_m.dtype)])

    thetas = jnp.array([incidence_angle], dtype=jnp.float32)
    R_TE, _, R_TM, _ = stackrt_n_k(n_complex, d_full, FREQS_HZ, thetas)
    # stackrt_n_k returns [n_angles, n_freqs]; we passed 1 angle.
    return 0.5 * (R_TE[0] + R_TM[0])              # [NUM_LAMBDA]


# ----------------------------------------------------------------------------
# Lab — same path as training (spectrum_to_lab in src/color_utils.py)
# ----------------------------------------------------------------------------

# sRGB ↔ XYZ matrices (D65, Y_n=100). Copied verbatim from src/color_utils.py
# so the round-trip matches.
_M_SRGB_TO_XYZ = jnp.asarray([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=jnp.float32) * 100.0  # because Y_n = 100

_X_N = 95.047
_Y_N = 100.000
_Z_N = 108.883
_LAB_DELTA = 6.0 / 29.0


def _inverse_srgb_gamma(c: jax.Array) -> jax.Array:
    """Companding inverse: sRGB display value → linear sRGB. Matches color_utils."""
    threshold = 0.04045
    linear_low = c / 12.92
    linear_high = jnp.power((jnp.maximum(c, 0.0) + 0.055) / 1.055, 2.4)
    return jnp.where(c <= threshold, linear_low, linear_high)


def _f_lab(t: jax.Array) -> jax.Array:
    """CIE Lab nonlinearity. Differentiable everywhere on t ≥ 0 with a tiny eps."""
    delta3 = _LAB_DELTA ** 3
    return jnp.where(
        t > delta3,
        jnp.cbrt(jnp.maximum(t, 1e-30)),
        t / (3.0 * _LAB_DELTA ** 2) + 4.0 / 29.0,
    )


def reflectance_to_lab(reflectance: jax.Array) -> jax.Array:
    """Reflectance [NUM_LAMBDA] → Lab [3] (L*, a*, b*), all in JAX.

    Matches `src.color_utils.spectrum_to_lab` numerically: same JLL
    spectrum_to_sRGB call, same inverse-gamma, same sRGB→XYZ matrix, same
    f_lab nonlinearity.
    """
    # JLL handles CMF integration + D65 illuminant + sRGB matrix internally.
    # We pass NUM_LAMBDA samples in the CIE-valid range [360, 830] nm; the
    # canonical grid spans 300–900 so we clip rather than mask (clip preserves
    # JAX traceability — np boolean indexing would not).
    valid_lo = 360.0
    valid_hi = 830.0
    # Build a soft "valid" weight that is 1 inside [lo, hi] and 0 outside, with
    # a tiny ramp at the edges for differentiability. (Strict clipping is also
    # safe since the wavelength grid is data, not a traced input.)
    in_band = (LAMBDA_NM >= valid_lo) & (LAMBDA_NM <= valid_hi)
    weights = in_band.astype(reflectance.dtype)

    # JLL returns 3 sRGB display values in [0, 1] (unclipped because
    # use_clipping=False). Multiply by weight to zero out OOB samples.
    sRGB = spectrum_to_sRGB(LAMBDA_NM * weights + (1.0 - weights) * valid_lo,
                            reflectance * weights,
                            use_clipping=False)
    sRGB = jnp.asarray(sRGB).reshape(3)

    # Inverse sRGB gamma → linear sRGB → XYZ (D65, Y_n=100) → Lab.
    linear_rgb = _inverse_srgb_gamma(sRGB)
    XYZ = _M_SRGB_TO_XYZ @ linear_rgb

    fx = _f_lab(XYZ[0] / _X_N)
    fy = _f_lab(XYZ[1] / _Y_N)
    fz = _f_lab(XYZ[2] / _Z_N)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return jnp.stack([L, a, b])


# ----------------------------------------------------------------------------
# ΔE00 (CIEDE2000) — JAX reimplementation
# ----------------------------------------------------------------------------
# Reference: G. Sharma et al., "The CIEDE2000 color-difference formula:
# Implementation notes, supplementary test data, and mathematical
# observations", Color Res. Appl. 30 (2005). Coefficients below match.
#
# Gradient hazards and their fixes:
#  - atan2 at chroma=0 → undefined hue. We add a tiny eps to the chroma so
#    the hue is well-defined but irrelevant for achromatic colors.
#  - Hue-rotation centre at 275° → a sin() peak; smooth, no issue.
#  - Δh wrap at ±180° handled via jnp.where (not differentiable at the
#    branch, but measure-zero).

_EPS = 1e-12


def _deg_to_rad(d: jax.Array) -> jax.Array:
    return d * (jnp.pi / 180.0)


def _rad_to_deg(r: jax.Array) -> jax.Array:
    return r * (180.0 / jnp.pi)


def ciede2000(lab1: jax.Array, lab2: jax.Array,
              kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> jax.Array:
    """ΔE_00 between two Lab triplets. Both inputs are [3] arrays.

    Returns a scalar JAX float. `jax.grad` of this is well-defined except
    on the (measure-zero) wrap branches inside Δh.
    """
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # Step 1: chroma, mean chroma, G factor, a' (Lab a after the G correction).
    C1 = jnp.sqrt(a1 * a1 + b1 * b1)
    C2 = jnp.sqrt(a2 * a2 + b2 * b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - jnp.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    # Step 2: corrected chroma C', corrected hue h' (in degrees).
    C1p = jnp.sqrt(a1p * a1p + b1 * b1)
    C2p = jnp.sqrt(a2p * a2p + b2 * b2)
    # atan2 with an eps guard so the gradient is well-defined at the origin.
    # arctan2 returns (-pi, pi]; convert to [0, 360).
    h1p_rad = jnp.arctan2(b1, a1p + _EPS * jnp.sign(a1p + _EPS))
    h2p_rad = jnp.arctan2(b2, a2p + _EPS * jnp.sign(a2p + _EPS))
    h1p = jnp.where(h1p_rad < 0, h1p_rad + 2.0 * jnp.pi, h1p_rad) * (180.0 / jnp.pi)
    h2p = jnp.where(h2p_rad < 0, h2p_rad + 2.0 * jnp.pi, h2p_rad) * (180.0 / jnp.pi)

    # Step 3: ΔL', ΔC', ΔH' (and Δh' with wrap to [-180, 180]).
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp_raw = h2p - h1p
    # Wrap to (-180, 180]: subtract 360 if > 180, add 360 if <= -180.
    dhp = jnp.where(dhp_raw > 180.0, dhp_raw - 360.0,
                    jnp.where(dhp_raw <= -180.0, dhp_raw + 360.0, dhp_raw))
    # If either chroma is zero, set Δh' = 0 (hue undefined). Use a smooth
    # near-zero check on the product so jax.grad doesn't hit a NaN through
    # the achromatic case.
    chroma_product = C1p * C2p
    dhp = jnp.where(chroma_product < _EPS, 0.0, dhp)

    dHp = 2.0 * jnp.sqrt(jnp.maximum(chroma_product, 0.0)) * jnp.sin(_deg_to_rad(dhp) * 0.5)

    # Step 4: averages.
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    # Mean hue h' — special-case when product is near zero or the |Δh'| > 180.
    hbar_raw_sum = h1p + h2p
    hbar_minus_360 = (hbar_raw_sum + 360.0) * 0.5
    hbar_plain = hbar_raw_sum * 0.5
    abs_diff = jnp.abs(h1p - h2p)
    hbarp_when_chromatic = jnp.where(abs_diff <= 180.0, hbar_plain, hbar_minus_360)
    hbarp = jnp.where(chroma_product < _EPS, h1p + h2p, hbarp_when_chromatic)

    # T factor.
    hbarp_rad = _deg_to_rad(hbarp)
    T = (1.0
         - 0.17 * jnp.cos(hbarp_rad - _deg_to_rad(30.0))
         + 0.24 * jnp.cos(2.0 * hbarp_rad)
         + 0.32 * jnp.cos(3.0 * hbarp_rad + _deg_to_rad(6.0))
         - 0.20 * jnp.cos(4.0 * hbarp_rad - _deg_to_rad(63.0)))

    # Rotation term.
    dTheta = 30.0 * jnp.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    R_C = 2.0 * jnp.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    R_T = -jnp.sin(_deg_to_rad(2.0 * dTheta)) * R_C

    # Lightness, chroma, hue weighting factors.
    Lbarp_minus_50_sq = (Lbarp - 50.0) ** 2
    S_L = 1.0 + (0.015 * Lbarp_minus_50_sq) / jnp.sqrt(20.0 + Lbarp_minus_50_sq)
    S_C = 1.0 + 0.045 * Cbarp
    S_H = 1.0 + 0.015 * Cbarp * T

    dL_term = dLp / (kL * S_L)
    dC_term = dCp / (kC * S_C)
    dH_term = dHp / (kH * S_H)

    return jnp.sqrt(
        dL_term ** 2 + dC_term ** 2 + dH_term ** 2 + R_T * dC_term * dH_term
    )


# ----------------------------------------------------------------------------
# End-to-end objective
# ----------------------------------------------------------------------------

def delta_e_from_thicknesses(
    thicknesses_nm: jax.Array,   # [MAX_LAYERS] float — continuous nm
    pool_n: jax.Array,           # [M_MAX, NUM_LAMBDA]
    pool_k: jax.Array,           # [M_MAX, NUM_LAMBDA]
    slot_indices: jax.Array,     # [MAX_LAYERS] int (static during refine)
    layer_mask: jax.Array,       # [MAX_LAYERS] bool
    target_lab: jax.Array,       # [3]
    incidence_angle: float = 0.0,
) -> jax.Array:
    """Scalar ΔE_00. `jax.grad(... , argnums=0)` gives dΔE/d(thicknesses_nm)."""
    R = compute_reflectance(pool_n, pool_k, slot_indices, thicknesses_nm,
                            layer_mask, incidence_angle)
    lab = reflectance_to_lab(R)
    return ciede2000(lab, target_lab)


# ----------------------------------------------------------------------------
# Layer-mask + structure helpers
# ----------------------------------------------------------------------------

def pad_structure(slot_indices: list, thicknesses_nm: list,
                  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Pad a (slots, thicknesses) tuple of length L ≤ MAX_LAYERS to fixed length.

    Returns three JAX arrays:
      slot_indices  : [MAX_LAYERS] int (0 in padding — safe, paired with mask)
      thicknesses_nm: [MAX_LAYERS] float
      layer_mask    : [MAX_LAYERS] bool — True at the first L positions
    """
    L = len(slot_indices)
    if L > MAX_LAYERS:
        raise ValueError(f"structure has {L} layers, max is {MAX_LAYERS}")
    if len(thicknesses_nm) != L:
        raise ValueError("slot_indices and thicknesses_nm must align")

    slots = np.zeros(MAX_LAYERS, dtype=np.int32)
    thicks = np.zeros(MAX_LAYERS, dtype=np.float32)
    mask = np.zeros(MAX_LAYERS, dtype=bool)
    if L > 0:
        slots[:L] = np.asarray(slot_indices, dtype=np.int32)
        thicks[:L] = np.asarray(thicknesses_nm, dtype=np.float32)
        mask[:L] = True
    return jnp.asarray(slots), jnp.asarray(thicks), jnp.asarray(mask)


def pad_pool_nk(pool) -> Tuple[jax.Array, jax.Array]:
    """Pad a list of MaterialNK to fixed shape [M_MAX, NUM_LAMBDA] each for n, k.

    Slots beyond `len(pool)` are zero-filled. Inference code must take care
    not to point at padded slots — pool_size masking handles this in the
    model; the optical sim is happy with any n,k at a 0-thickness slot.
    """
    if len(pool) > M_MAX:
        raise ValueError(f"pool of {len(pool)} exceeds M_MAX={M_MAX}")
    pool_n_np = np.zeros((M_MAX, NUM_LAMBDA), dtype=np.float32)
    pool_k_np = np.zeros((M_MAX, NUM_LAMBDA), dtype=np.float32)
    for s, mat in enumerate(pool):
        pool_n_np[s, :] = mat.n
        pool_k_np[s, :] = mat.k
    return jnp.asarray(pool_n_np), jnp.asarray(pool_k_np)
