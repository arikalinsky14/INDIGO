"""
Robustness metrics for INDIGO inference.

Two methods, both at the same `tolerance_pct` manufacturing-precision
setting:

  grad_robustness(...)   ─►  R_max, R_l2   (cheap, on ALL candidates)
                              R_max = max_i |∂ΔE/∂tᵢ|·τᵢ·tᵢ      (worst layer)
                              R_l2  = sqrt(Σᵢ (∂ΔE/∂tᵢ·τᵢ·tᵢ)²)   (RMS shift)

  monte_carlo_robustness(...) ─►  p50, p95, worst, K_used
                              K independent uniform jitter draws of size
                              τᵢ·tᵢ per layer; recompute ΔE for each.

The gradient version is the ranking signal during selection. The MC
version is the honest number reported on the top_k candidates. R_l2 is
the default `robustness` field in the Result (per my review note —
linearity-of-expectation gives a more sensible aggregate than max).

The forward + grad timing budget was measured by the sim spike: a
single candidate is ~ms-class on CPU, so K=32 MC samples × top_k=5
candidates is well under a second. Comfortable on any L40s session.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import jax
import jax.numpy as jnp

from inference.src.simulate import (
    delta_e_from_thicknesses, pad_pool_nk, pad_structure,
)
from inference.src.schema import MaterialEntry


# ----------------------------------------------------------------------------
# Result objects
# ----------------------------------------------------------------------------

@dataclass
class GradRobustness:
    r_max: float            # max_i |g_i|·τ_i·t_i — worst single layer
    r_l2: float             # sqrt(Σ (g_i·τ_i·t_i)²) — RMS shift
    per_layer_shift: List[float]  # |g_i|·τ_i·t_i, length = active layers


@dataclass
class MCRobustness:
    p50: float
    p95: float
    worst: float
    K_used: int
    samples: List[float]    # all K perturbed ΔE values (caller may discard)


# ----------------------------------------------------------------------------
# Tolerance helpers
# ----------------------------------------------------------------------------

def _normalise_tolerance(tolerance_pct, L: int) -> np.ndarray:
    """Per-layer tolerance fraction (e.g. 0.05 for 5 %). Accepts scalar or seq.

    Returned shape: [L]. Caller multiplies by `t_i` to get the absolute
    deviation at each layer.
    """
    if np.isscalar(tolerance_pct):
        tau = np.full(L, float(tolerance_pct) / 100.0, dtype=np.float64)
    else:
        tau = np.asarray(tolerance_pct, dtype=np.float64) / 100.0
        if tau.shape != (L,):
            raise ValueError(
                f"per-layer tolerance must be length {L}, got {tau.shape}"
            )
    if (tau < 0).any():
        raise ValueError("tolerance values must be non-negative")
    return tau


# ----------------------------------------------------------------------------
# Gradient predicted shift (run on every candidate during selection)
# ----------------------------------------------------------------------------

def grad_robustness(
    pool: List[MaterialEntry],
    slot_indices: List[int],
    thicknesses_nm: List[float],
    target_lab: Tuple[float, float, float],
    tolerance_pct: float = 5.0,
    incidence_angle: float = 0.0,
) -> GradRobustness:
    """Linearised manufacturing-tolerance prediction.

    Computes ∂ΔE/∂tᵢ at the nominal structure and propagates the per-layer
    tolerance through it. Both R_max (worst layer) and R_l2 (RMS) are
    returned — R_l2 is the recommended default for ranking; R_max is kept
    for back-compat with the original plan.

    Uses `jax.grad` (no jit / vmap, see simulate.py docstring on the JLL
    assert that blocks them). Per-candidate cost is the same as the timing
    in the sim spike: a few ms.
    """
    L = len(slot_indices)
    if L != len(thicknesses_nm):
        raise ValueError("slot_indices and thicknesses_nm must align")
    if L == 0:
        return GradRobustness(0.0, 0.0, [])

    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    target = jnp.asarray(target_lab, dtype=jnp.float64)
    tau = _normalise_tolerance(tolerance_pct, L)

    def f(t):
        return delta_e_from_thicknesses(
            t, pool_n, pool_k, slots, mask, target, incidence_angle,
        )

    grad = np.asarray(jax.grad(f)(thicks))            # [MAX_LAYERS]
    grad_active = grad[:L]
    t_active = np.asarray(thicknesses_nm, dtype=np.float64)
    shift = np.abs(grad_active) * tau * t_active      # [L] predicted ΔE shift
    r_max = float(np.max(shift)) if L > 0 else 0.0
    r_l2 = float(np.sqrt(np.sum(shift ** 2)))
    return GradRobustness(
        r_max=r_max, r_l2=r_l2, per_layer_shift=shift.tolist(),
    )


# ----------------------------------------------------------------------------
# Monte-Carlo robustness (run only on the top_k after selection)
# ----------------------------------------------------------------------------

def monte_carlo_robustness(
    pool: List[MaterialEntry],
    slot_indices: List[int],
    thicknesses_nm: List[float],
    target_lab: Tuple[float, float, float],
    tolerance_pct: float = 5.0,
    K: int = 32,
    seed: int = 0,
    incidence_angle: float = 0.0,
) -> MCRobustness:
    """K independent uniform perturbations; report ΔE percentiles.

    Each draw: tᵢ ← tᵢ · (1 + uᵢ), uᵢ ~ U(−τᵢ, +τᵢ). The simulator runs
    K times sequentially because the JLL assert blocks vmap. K=32 keeps
    per-candidate cost ≲ 100 ms; top_k=5 ⇒ ≲ 500 ms total — modest.

    `seed` is mixed with the sample index so the K perturbations are
    deterministic across reruns of the same candidate.
    """
    L = len(slot_indices)
    if L != len(thicknesses_nm):
        raise ValueError("slot_indices and thicknesses_nm must align")
    if L == 0 or K <= 0:
        return MCRobustness(0.0, 0.0, 0.0, 0, [])

    pool_n, pool_k = pad_pool_nk(pool)
    slots, _, mask = pad_structure(slot_indices, thicknesses_nm)
    target = jnp.asarray(target_lab, dtype=jnp.float64)
    tau = _normalise_tolerance(tolerance_pct, L)
    t_active = np.asarray(thicknesses_nm, dtype=np.float64)

    rng = np.random.default_rng(seed)
    samples: List[float] = []
    for k in range(K):
        u = rng.uniform(-tau, tau, size=L)                # [L]
        t_perturbed = t_active * (1.0 + u)                # [L]
        t_perturbed = np.clip(t_perturbed, 1e-6, None)    # never negative
        # Pad back up to MAX_LAYERS
        t_full = np.asarray(slots).astype(np.float64) * 0.0  # [MAX_LAYERS]
        t_full[:L] = t_perturbed
        de = float(delta_e_from_thicknesses(
            jnp.asarray(t_full), pool_n, pool_k, slots, mask, target,
            incidence_angle,
        ))
        samples.append(de)

    arr = np.asarray(samples, dtype=np.float64)
    return MCRobustness(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        worst=float(arr.max()),
        K_used=K,
        samples=arr.tolist(),
    )


# ----------------------------------------------------------------------------
# Smoke test — runs only when JLL + JAX are present
# ----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Light end-to-end on a small synthetic pool.

    Skipped automatically if dependencies aren't installed (e.g. dev sandbox).
    """
    try:
        pool = [
            MaterialEntry(canonical_name=f"m{i}",
                          n=np.linspace(1.2 + 0.3 * i, 1.5 + 0.3 * i, 128).astype(np.float32),
                          k=np.full(128, 0.01 * i, dtype=np.float32))
            for i in range(3)
        ]
        slots = [0, 1, 2]
        thicks = [50.0, 100.0, 30.0]
        target = (60.0, 5.0, -3.0)

        g = grad_robustness(pool, slots, thicks, target, tolerance_pct=5.0)
        print(f"[sensitivity] grad: r_max={g.r_max:.4f}  r_l2={g.r_l2:.4f}  "
              f"per-layer={[round(s, 4) for s in g.per_layer_shift]}")
        assert g.r_max >= 0 and g.r_l2 >= 0
        assert len(g.per_layer_shift) == len(slots)

        mc = monte_carlo_robustness(
            pool, slots, thicks, target, tolerance_pct=5.0, K=8, seed=42,
        )
        print(f"[sensitivity] MC:   p50={mc.p50:.4f}  p95={mc.p95:.4f}  "
              f"worst={mc.worst:.4f}  K={mc.K_used}")
        assert mc.K_used == 8
        assert mc.worst >= mc.p95 >= mc.p50 >= 0
    except ImportError as exc:
        print(f"[sensitivity] smoke skipped (missing dep): {exc}")
        return
    print("[sensitivity] smoke OK")


if __name__ == "__main__":
    _smoke_test()
