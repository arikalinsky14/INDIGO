"""Shared thickness-optimizer primitives.

Two callers need the same box-projected gradient-descent loop over a
thickness vector:

  1. `inference/src/refine.py` — refines the top-k INFERRED thicknesses
     against the user's target Lab colour at solve time.
  2. `create_dataset/src/high_chroma_search.py` — refines a candidate
     structure's thicknesses toward a saturated Lab target during
     DATA GENERATION, so we can enrich the training distribution with
     high-chroma rows.

Both flows want the same numerical primitives:

  _project     — box clamp to per-layer (min, max) bounds.
  _adam_step   — hand-rolled Adam step (no torch/optax dep).
  _dog_step +  — parameter-free Distance-over-Gradients step
  DoGState +     (Ivgi, Hinder, Carmon, ICML 2023).
  _dog_init

Everything here is pure NumPy — no torch, no JAX. Callers wire the
gradient computation themselves (via JAX autograd on their preferred
simulator) and hand ndarray gradients into these steps.

Keeping this module in `src/` (the shared model/utility layer per the
README's directory map) rather than `inference/src/` avoids
inference-layer imports leaking into `create_dataset/` — the data-gen
worker only depends on the optimizer math, not on the inference stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================================
# Box projection
# ============================================================================

def _project(t: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Clamp `t` elementwise into [mins, maxs]. Modifies a copy."""
    return np.minimum(np.maximum(t, mins), maxs)


# ============================================================================
# Adam
# ============================================================================

def _adam_step(t: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray,
               step_idx: int, lr: float,
               beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Adam step. Returns (t_new, m_new, v_new).

    `lr` is the raw learning rate — no schedule, no per-parameter scale
    knob. Kept alongside DoG for A/B benchmarking.
    """
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    m_hat = m / (1.0 - beta1 ** step_idx)
    v_hat = v / (1.0 - beta2 ** step_idx)
    t_new = t - lr * m_hat / (np.sqrt(v_hat) + eps)
    return t_new, m, v


# ============================================================================
# DoG — Distance over Gradients (Ivgi, Hinder, Carmon 2023)
#
# Parameter-free step size:
#     eta_t = r_t / sqrt(G_t + eps)
#     r_t   = max(r_{t-1}, ||x_t - x_0||)     max distance from init
#     G_t   = G_{t-1} + ||g_t||^2             cumulative squared-grad norm
#
# `r_0 = r_eps` is a tiny anchoring radius that just breaks the
# first-step degeneracy. See _dog_init's docstring for calibration
# notes specific to our thickness-scale problem.
# ============================================================================

@dataclass
class DoGState:
    """Iteration state carried between DoG steps.

    Vanilla DoG — scalars aggregated across the whole thickness vector.
    Per-layer L-DoG adds a hyperparameter (the "reference layer") that
    hasn't been necessary in practice for our dozen-parameter vectors.
    """
    x0: np.ndarray                       # starting iterate (for r_t)
    r_max: float                         # running max ‖x_t - x_0‖
    g_squared_sum: float                 # sum of ‖g_s‖² up to now


def _dog_init(x0: np.ndarray, r_eps_scale: float = 1e-3) -> DoGState:
    """Initial DoG state.

    The paper's default `r_eps = 1e-6 * (1 + ||x_0||)` is calibrated for
    deep-learning-scale problems (millions of params, gradients O(10^3)).
    For our thickness vector (≤ 10 params, gradients O(1)) that default
    keeps the effective step negligible for the entire iteration budget —
    the iterate never moves far enough for r_t to grow past the anchor.

    Numerical sweep on the toy loss ||x - x*||² with x_0 ≈ 100 nm:
      scale=1e-6 →  loss=2.4e3 after 50 iters (essentially no progress)
      scale=1e-4 →  3.06
      scale=1e-3 →  1.5e-4                                  ← default
      scale=1e-2 →  3.5e-13

    1e-3 leaves headroom on more curved landscapes while still making
    meaningful early progress. Exposed as an argument so downstream
    A/B scripts can sweep it per battery.
    """
    r_eps = r_eps_scale * (1.0 + float(np.linalg.norm(x0)))
    return DoGState(x0=x0.copy(), r_max=r_eps, g_squared_sum=0.0)


def _dog_step(t: np.ndarray, g: np.ndarray, state: DoGState,
              eps: float = 1e-12,
              ) -> Tuple[np.ndarray, DoGState]:
    """One DoG step. Returns (t_new, state_new)."""
    # Update r_t BEFORE stepping: it's the max distance seen so far.
    dist = float(np.linalg.norm(t - state.x0))
    r_t = max(state.r_max, dist)
    g_sq_sum = state.g_squared_sum + float(np.dot(g, g))
    eta = r_t / np.sqrt(g_sq_sum + eps)
    t_new = t - eta * g
    return t_new, DoGState(
        x0=state.x0, r_max=r_t, g_squared_sum=g_sq_sum,
    )
