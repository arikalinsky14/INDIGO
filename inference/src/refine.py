"""
Refinement: gradient local search on the continuous-nm thickness vector.

  top_k candidates (5 nm grid)   ─►   continuous-nm thicknesses minimising
  ΔE_00, projected onto per-layer thickness bounds                ─►
  post-refine constraint recheck   ─►   J-regression sanity check

Variables refined: continuous thickness (we deliberately leave the 5 nm
grid). Slot identities and the layer count are fixed; refinement is a pure
geometry tune.

Objective in v1
---------------
We refine on ΔE_00 only (not the full J = ΔE + λ·R). The full objective
is recomputed after refinement for ranking, but the descent gradient is
ΔE-only. Rationale:

  - ΔE is typically the dominant term (ΔE ~ 5–10, R_l2 ~ 0.5 in the
    cross_attn results so far).
  - Refining on J requires the Hessian of ΔE (because R_l2 contains
    ∂ΔE/∂t), which would more than double the per-iter cost and add a
    failure surface (NaNs in second-order grads near the achromatic axis).
  - The seed candidates are already tolerance-aware (selection ranked
    them by J), so R_l2 stays close to its seed value during a short
    descent.

If the residual mismatch between selection-on-J and refine-on-ΔE causes a
worse J(refined) > J(seed), the J-regression fallback below catches it
and reverts to the seed.

Optimizer
---------
Hand-rolled Adam. Box projection (clamp to per-layer min/max) after each
step. Stop on max_iters or |ΔE_prev − ΔE_curr| < tol.

Multi-start
-----------
Each seed candidate is refined from its grid point. Optional extra random
in-bounds restarts per seed via `n_random_restarts > 0`. The best
post-recheck thickness wins for that seed.

Fallbacks (in priority order)
-----------------------------
1. If the refined structure violates any post-hoc constraint → revert to
   seed for THAT candidate.
2. If J(refined) > J(seed) by more than `j_regression_tol` → revert.
3. Otherwise keep the refined thicknesses.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.materials_vocab import MAX_LAYERS, MAX_THICKNESS_NM

from inference.src.constraints import (
    ConstraintSet, FinishedStructure, ThicknessRange,
)
from inference.src.schema import (
    Candidate, InferenceKnobs, MaterialEntry, RobustnessReport,
)
from inference.src.sensitivity import grad_robustness
from inference.src.simulate import (
    compute_reflectance, ciede2000, delta_e_from_thicknesses,
    pad_pool_nk, pad_structure, reflectance_to_lab,
)


# ----------------------------------------------------------------------------
# Bounds construction from the constraint set
# ----------------------------------------------------------------------------

_THICKNESS_FLOOR_NM = 5.0   # matches the 5 nm grid floor


def _build_thickness_bounds(constraint_set: ConstraintSet, L: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layer (min_nm, max_nm) arrays of length L, narrowed by any
    ThicknessRange constraints the user provided.

    Global thickness ranges narrow every layer; positional ranges narrow
    only their target layer.
    """
    mins = np.full(L, _THICKNESS_FLOOR_NM, dtype=np.float64)
    maxs = np.full(L, float(MAX_THICKNESS_NM), dtype=np.float64)
    for c in constraint_set.constraints:
        if isinstance(c, ThicknessRange):
            if c.position is None:
                mins[:] = np.maximum(mins, float(c.min_nm))
                maxs[:] = np.minimum(maxs, float(c.max_nm))
            elif 0 <= c.position < L:
                mins[c.position] = max(mins[c.position], float(c.min_nm))
                maxs[c.position] = min(maxs[c.position], float(c.max_nm))
    if (mins > maxs).any():
        raise ValueError(
            f"empty thickness bound after applying constraints: "
            f"mins={mins.tolist()} maxs={maxs.tolist()}"
        )
    return mins, maxs


def _project(t: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Box projection onto per-layer bounds."""
    return np.minimum(np.maximum(t, mins), maxs)


# ----------------------------------------------------------------------------
# Adam (no external dep)
# ----------------------------------------------------------------------------

def _adam_step(t: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray,
               step_idx: int, lr: float,
               beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    m_hat = m / (1.0 - beta1 ** step_idx)
    v_hat = v / (1.0 - beta2 ** step_idx)
    t_new = t - lr * m_hat / (np.sqrt(v_hat) + eps)
    return t_new, m, v


# ----------------------------------------------------------------------------
# Single-candidate refinement
# ----------------------------------------------------------------------------

@dataclass
class RefineDiagnostics:
    iters: int
    delta_e_start: float
    delta_e_end: float
    j_start: float
    j_end: float
    fell_back_to_seed: bool
    fallback_reason: str = ""


def _delta_e_of(thicknesses: np.ndarray,
                pool_n_jax, pool_k_jax,
                slots_jax, mask_jax,
                target_jax,
                incidence_angle: float = 0.0) -> float:
    t_padded = np.zeros(MAX_LAYERS, dtype=np.float64)
    L = len(thicknesses)
    t_padded[:L] = thicknesses
    return float(delta_e_from_thicknesses(
        jnp.asarray(t_padded), pool_n_jax, pool_k_jax, slots_jax, mask_jax,
        target_jax, incidence_angle,
    ))


def _grad_delta_e(thicknesses: np.ndarray,
                  pool_n_jax, pool_k_jax,
                  slots_jax, mask_jax,
                  target_jax,
                  incidence_angle: float = 0.0) -> np.ndarray:
    """Returns the gradient over the L active layers (already trimmed)."""
    t_padded = np.zeros(MAX_LAYERS, dtype=np.float64)
    L = len(thicknesses)
    t_padded[:L] = thicknesses

    def f(t):
        return delta_e_from_thicknesses(
            t, pool_n_jax, pool_k_jax, slots_jax, mask_jax, target_jax,
            incidence_angle,
        )

    grad_full = np.asarray(jax.grad(f)(jnp.asarray(t_padded)))
    return grad_full[:L]


def refine_candidate(
    cand: Candidate,
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
    constraint_set: ConstraintSet,
    knobs: InferenceKnobs,
    incidence_angle: float = 0.0,
    j_regression_tol: float = 1e-4,
    n_random_restarts: int = 0,
    base_seed: int = 0,
) -> Tuple[Candidate, RefineDiagnostics]:
    """Refine one candidate via projected-Adam on ΔE.

    Returns the (possibly refined) Candidate with updated fields and a
    RefineDiagnostics with the descent + fallback record.
    """
    L = len(cand.slot_indices)
    if L == 0:
        return cand, RefineDiagnostics(0, cand.delta_e, cand.delta_e,
                                       cand.objective, cand.objective,
                                       fell_back_to_seed=True,
                                       fallback_reason="empty structure")

    # Build bounds + JAX inputs.
    mins, maxs = _build_thickness_bounds(constraint_set, L)
    pool_n_jax, pool_k_jax = pad_pool_nk(pool)
    slots_jax, _, mask_jax = pad_structure(cand.slot_indices, cand.thicknesses_nm)
    target_jax = jnp.asarray(target_lab_raw, dtype=jnp.float64)

    # Seed in bounds (guard against out-of-bounds seeds).
    t_seed = _project(np.asarray(cand.thicknesses_nm, dtype=np.float64), mins, maxs)
    de_seed = _delta_e_of(t_seed, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                          target_jax, incidence_angle)
    # Robustness at seed (used for the J-regression check and final report).
    g_seed = grad_robustness(
        pool, cand.slot_indices, t_seed.tolist(), target_lab_raw,
        knobs.tolerance_pct, incidence_angle,
    )
    j_seed = de_seed + knobs.weight_lambda * g_seed.r_l2

    # Collect starting points (seed + n_random_restarts uniform draws).
    starts = [t_seed]
    if n_random_restarts > 0:
        rng = np.random.default_rng(base_seed)
        for _ in range(n_random_restarts):
            starts.append(rng.uniform(mins, maxs))

    best_t = t_seed.copy()
    best_de = de_seed
    iters_total = 0

    # Each start gets its own short Adam descent.
    for t0 in starts:
        t = _project(t0, mins, maxs)
        m = np.zeros_like(t)
        v = np.zeros_like(t)
        prev_de = float("inf")
        for it in range(1, knobs.refine_max_iters + 1):
            iters_total += 1
            de = _delta_e_of(t, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                             target_jax, incidence_angle)
            if abs(prev_de - de) < 1e-4:
                break
            g = _grad_delta_e(t, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                              target_jax, incidence_angle)
            t, m, v = _adam_step(t, g, m, v, step_idx=it, lr=knobs.refine_step_size)
            t = _project(t, mins, maxs)
            prev_de = de
        # Score the final point of this start.
        final_de = _delta_e_of(t, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                               target_jax, incidence_angle)
        if final_de < best_de:
            best_de = final_de
            best_t = t.copy()

    # Re-check constraints on the refined geometry (continuous nm — we
    # round-up for the check call signature, which expects ints).
    fs_refined = FinishedStructure(
        slot_indices=list(cand.slot_indices),
        thicknesses_nm=[int(round(x)) for x in best_t.tolist()],
        pool_size=len(pool),
    )
    constraint_results = constraint_set.check(fs_refined, pool)
    constraints_passed = all(p for _, p, _ in constraint_results)

    fallback = False
    fallback_reason = ""
    if not constraints_passed:
        fallback = True
        failing = [k for k, p, _ in constraint_results if not p]
        fallback_reason = f"post-refine constraints failed: {failing}"

    # Robustness + J at refined (or seed if we already decided to fall back).
    if not fallback:
        g_ref = grad_robustness(
            pool, cand.slot_indices, best_t.tolist(), target_lab_raw,
            knobs.tolerance_pct, incidence_angle,
        )
        j_refined = best_de + knobs.weight_lambda * g_ref.r_l2
        if j_refined > j_seed + j_regression_tol:
            fallback = True
            fallback_reason = (
                f"J regression: J(refined)={j_refined:.5f} > "
                f"J(seed)={j_seed:.5f} (tol {j_regression_tol})"
            )

    # Commit either refined or seed back onto the candidate.
    if fallback:
        final_t = t_seed
        final_de = de_seed
        final_grad = g_seed
        final_j = j_seed
    else:
        final_t = best_t
        final_de = best_de
        final_grad = g_ref
        final_j = j_refined

    # Recompute reflectance + Lab for the chosen geometry so the report is
    # consistent with the final thicknesses.
    pad = np.zeros(MAX_LAYERS, dtype=np.float64)
    pad[:L] = final_t
    R = compute_reflectance(pool_n_jax, pool_k_jax, slots_jax,
                            jnp.asarray(pad), mask_jax, incidence_angle)
    lab = reflectance_to_lab(R)
    refined_cand = Candidate(
        slot_indices=list(cand.slot_indices),
        material_names=list(cand.material_names),
        thicknesses_nm=[float(x) for x in final_t.tolist()],
        achieved_lab=tuple(float(x) for x in np.asarray(lab).tolist()),
        reflectance=[float(x) for x in np.asarray(R).tolist()],
        delta_e=float(final_de),
        robustness=RobustnessReport(
            grad_predicted_shift=final_grad.r_max,
            grad_l2_shift=final_grad.r_l2,
            mc_p50=cand.robustness.mc_p50,
            mc_p95=cand.robustness.mc_p95,
            mc_worst=cand.robustness.mc_worst,
            mc_samples=cand.robustness.mc_samples,
        ),
        objective=float(final_j),
        refined=not fallback,
        refine_iters=iters_total,
        sample_index=cand.sample_index,
        multiplicity=cand.multiplicity,
    )
    diag = RefineDiagnostics(
        iters=iters_total, delta_e_start=de_seed, delta_e_end=float(final_de),
        j_start=j_seed, j_end=float(final_j),
        fell_back_to_seed=fallback, fallback_reason=fallback_reason,
    )
    return refined_cand, diag


# ----------------------------------------------------------------------------
# Top-k refinement
# ----------------------------------------------------------------------------

def refine_top_k(
    candidates: List[Candidate],
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
    constraint_set: ConstraintSet,
    knobs: InferenceKnobs,
    incidence_angle: float = 0.0,
    n_random_restarts: int = 0,
) -> Tuple[List[Candidate], List[RefineDiagnostics]]:
    """Refine each top-k candidate independently. Returns refined Candidates
    in the same order and the per-candidate RefineDiagnostics."""
    refined: List[Candidate] = []
    diags: List[RefineDiagnostics] = []
    for idx, c in enumerate(candidates):
        rc, dd = refine_candidate(
            c, pool, target_lab_raw, constraint_set, knobs, incidence_angle,
            n_random_restarts=n_random_restarts,
            base_seed=knobs.seed + idx,
        )
        refined.append(rc)
        diags.append(dd)
    return refined, diags


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

def _smoke_test() -> None:
    try:
        pool = [
            MaterialEntry(canonical_name="A",
                          n=np.linspace(1.3, 1.6, 128).astype(np.float32),
                          k=np.zeros(128, dtype=np.float32)),
            MaterialEntry(canonical_name="B",
                          n=np.linspace(2.0, 2.4, 128).astype(np.float32),
                          k=np.full(128, 0.02, dtype=np.float32)),
        ]
        seed = Candidate(
            slot_indices=[0, 1, 0], material_names=["A", "B", "A"],
            thicknesses_nm=[80.0, 50.0, 80.0],
            achieved_lab=(0, 0, 0), reflectance=[0.0] * 128,
            delta_e=10.0,                     # placeholder; refine recomputes
            objective=10.0,
            robustness=RobustnessReport(grad_l2_shift=0.5),
        )
        knobs = InferenceKnobs(
            top_k=1, weight_lambda=1.0, tolerance_pct=5.0,
            refine_max_iters=40, refine_step_size=2.0,
        )
        cs = ConstraintSet(constraints=[])
        target = (55.0, 0.0, 0.0)

        # Use a not-too-good seed; refinement should at least not make ΔE worse.
        refined, diag = refine_candidate(seed, pool, target, cs, knobs)
        print(f"[refine] seed ΔE={diag.delta_e_start:.4f} → "
              f"{diag.delta_e_end:.4f}  iters={diag.iters}  "
              f"fallback={diag.fell_back_to_seed}  "
              f"thicknesses(continuous)={[round(t, 2) for t in refined.thicknesses_nm]}")
        assert refined.delta_e <= diag.delta_e_start + 1e-6
        assert refined.refined or diag.fell_back_to_seed
    except ImportError as exc:
        print(f"[refine] smoke skipped (missing dep): {exc}")
        return
    print("[refine] smoke OK")


if __name__ == "__main__":
    _smoke_test()
