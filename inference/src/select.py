"""
Selection: sim + score + post-hoc filter + top-k for INDIGO inference.

  candidates from generate.py    ─►   simulate ΔE + grad R   ─►
  weighted objective J = ΔE + λ·R   ─►   post-hoc constraints  ─►
  top-k by J (tie-break by R)         ─►   refine.py / report

The orchestrator (solve.py) is the caller; this module is intentionally
narrow — it owns the math of selection and nothing else.

Feasibility error
-----------------
Empty feasible set after post-hoc filtering raises `FeasibilityError`.
The Result envelope catches it and writes `chosen=None`, populated
`errors`, and the per-constraint drop counts so the GUI can show "this
many of N were dropped by X" without re-running anything.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from inference.src.constraints import ConstraintSet, FinishedStructure
from inference.src.schema import (
    Candidate, ConstraintCheck, InferenceKnobs, MaterialEntry,
    RobustnessReport,
)
from inference.src.sensitivity import grad_robustness
from inference.src.simulate import (
    compute_reflectance, pad_pool_nk, pad_structure, reflectance_to_lab,
    ciede2000,
)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------

class FeasibilityError(RuntimeError):
    """No candidate satisfied all post-hoc constraints.

    Carries diagnostic fields the orchestrator drops into the failure-mode
    Result so the GUI can render a coherent error page.
    """

    def __init__(self, message: str, n_sampled: int, n_unique: int,
                 dropped_per_constraint: dict):
        super().__init__(message)
        self.n_sampled = n_sampled
        self.n_unique = n_unique
        self.dropped_per_constraint = dropped_per_constraint


# ----------------------------------------------------------------------------
# Simulate-and-score loop
# ----------------------------------------------------------------------------

def simulate_and_score(
    candidates: List[Candidate],
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
    knobs: InferenceKnobs,
    incidence_angle: float = 0.0,
    on_progress=None,
) -> List[Candidate]:
    """In-place fill of `achieved_lab`, `reflectance`, `delta_e`, `objective`,
    and `robustness` on every Candidate.

    Robustness uses R_l2 (the default I pushed for in the review). R_max is
    populated alongside it on the dataclass for completeness.

    Sequential per-candidate — `jax.vmap` is blocked by the JLL assert
    (see simulate.py docstring). For N=500 this is the dominant compute
    in the inference call; at ~10 ms/candidate it's still under a single
    user-visible second.
    """
    if not candidates:
        return []
    pool_n_jax, pool_k_jax = pad_pool_nk(pool)
    target_jax = jnp.asarray(target_lab_raw, dtype=jnp.float64)

    total = len(candidates)
    report_every = max(1, total // 20)  # ~20 updates over the loop
    for cand_idx, cand in enumerate(candidates, start=1):
        L = len(cand.slot_indices)
        if L == 0:
            continue  # generate.py already drops these; defensive.
        # Reflectance + Lab + ΔE via the JAX path.
        slots_jax, thicks_jax, mask_jax = pad_structure(
            cand.slot_indices, cand.thicknesses_nm,
        )
        R = compute_reflectance(
            pool_n_jax, pool_k_jax, slots_jax, thicks_jax, mask_jax,
            incidence_angle,
        )
        lab = reflectance_to_lab(R)
        de = float(ciede2000(lab, target_jax))
        cand.reflectance = [float(x) for x in np.asarray(R).tolist()]
        cand.achieved_lab = tuple(float(x) for x in np.asarray(lab).tolist())
        cand.delta_e = de

        # Robustness only if tolerance > 0 (else cost is wasted).
        if knobs.tolerance_pct > 0:
            g = grad_robustness(
                pool, cand.slot_indices, cand.thicknesses_nm,
                target_lab_raw, knobs.tolerance_pct, incidence_angle,
            )
            cand.robustness = RobustnessReport(
                grad_predicted_shift=g.r_max,
                grad_l2_shift=g.r_l2,
            )
            r_for_objective = g.r_l2
        else:
            cand.robustness = RobustnessReport(
                grad_predicted_shift=0.0, grad_l2_shift=0.0,
            )
            r_for_objective = 0.0

        cand.objective = de + knobs.weight_lambda * r_for_objective

        if on_progress is not None and (
            cand_idx % report_every == 0 or cand_idx == total
        ):
            try:
                on_progress("simulate", cand_idx, total, {})
            except Exception:
                pass

    return candidates


# ----------------------------------------------------------------------------
# Feasibility filter (post-hoc check + per-constraint drop counts)
# ----------------------------------------------------------------------------

def filter_feasible(
    candidates: List[Candidate],
    pool: List[MaterialEntry],
    constraint_set: ConstraintSet,
) -> Tuple[List[Candidate], dict]:
    """Apply every post-hoc check; return (feasible_list, drop_count_by_kind).

    Constraints enforced during decoding will always pass here by
    construction; constraints in the post-set actually filter. We still call
    every check so the GUI can confirm "all decoded-time constraints held".
    """
    drop_counts: dict = {}
    feasible: List[Candidate] = []
    for c in candidates:
        L = len(c.slot_indices)
        if L == 0:
            drop_counts["empty_structure"] = drop_counts.get("empty_structure", 0) + 1
            continue
        fs = FinishedStructure(
            slot_indices=list(c.slot_indices),
            thicknesses_nm=[int(round(t)) for t in c.thicknesses_nm],
            pool_size=len(pool),
        )
        results = constraint_set.check(fs, pool)
        all_pass = True
        for kind, passed, _detail in results:
            if not passed:
                drop_counts[kind] = drop_counts.get(kind, 0) + 1
                all_pass = False
                break  # one fail → drop; no need to count multiple kinds per cand
        if all_pass:
            feasible.append(c)
    return feasible, drop_counts


# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------

def top_k_by_objective(candidates: List[Candidate], top_k: int) -> List[Candidate]:
    """Sort ascending by `objective`, then by `robustness.grad_l2_shift`
    (tie-break: lower-robustness wins), then by negative `multiplicity`
    (more frequent samples win the next tie).
    """
    if top_k <= 0:
        return []
    return sorted(
        candidates,
        key=lambda c: (
            c.objective,
            c.robustness.grad_l2_shift,
            -c.multiplicity,
            c.sample_index,
        ),
    )[:top_k]


# ----------------------------------------------------------------------------
# All-in-one helper used by solve.py
# ----------------------------------------------------------------------------

def select_top_k(
    candidates: List[Candidate],
    pool: List[MaterialEntry],
    target_lab_raw: Tuple[float, float, float],
    constraint_set: ConstraintSet,
    knobs: InferenceKnobs,
    incidence_angle: float = 0.0,
    on_progress=None,
) -> Tuple[List[Candidate], dict, List[ConstraintCheck]]:
    """End-to-end selection step: sim + filter + rank + top_k.

    Returns
    -------
    chosen_top_k : ordered list, length ≤ knobs.top_k.
    drop_counts  : per-constraint count of post-hoc drops (for EnsembleStats).
    constraints_report : ConstraintCheck for every constraint applied to the
                         best-scoring candidate; intended for the Result.
    """
    # 1. Simulate and score every unique candidate.
    simulate_and_score(candidates, pool, target_lab_raw, knobs, incidence_angle,
                       on_progress=on_progress)

    # 2. Post-hoc filter.
    feasible, drop_counts = filter_feasible(candidates, pool, constraint_set)

    if not feasible:
        n_sampled = sum(c.multiplicity for c in candidates)
        n_unique = len(candidates)
        raise FeasibilityError(
            f"0 / {n_unique} unique candidates ({n_sampled} samples) satisfied "
            f"all constraints. Rerun with a larger --ensemble-N, or relax the "
            f"most-violated constraint.",
            n_sampled=n_sampled,
            n_unique=n_unique,
            dropped_per_constraint=drop_counts,
        )

    # 3. Rank by J, take top_k.
    top = top_k_by_objective(feasible, knobs.top_k)

    # 4. Build a constraint report for the best candidate (the orchestrator
    # may also want one for the chosen post-refine; that lives in solve.py).
    if not top:
        return [], drop_counts, []
    fs_best = FinishedStructure(
        slot_indices=list(top[0].slot_indices),
        thicknesses_nm=[int(round(t)) for t in top[0].thicknesses_nm],
        pool_size=len(pool),
    )
    rep = [ConstraintCheck(kind=k, passed=p, detail=d)
           for k, p, d in constraint_set.check(fs_best, pool)]

    return top, drop_counts, rep


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Toy candidates + a hand-rolled pool; checks ranking + feasibility path.

    Uses synthetic materials so the simulator runs without external data.
    """
    try:
        # Build a small pool + a small candidate list and rank.
        pool = [
            MaterialEntry(canonical_name="A",
                          n=np.linspace(1.4, 1.6, 128).astype(np.float32),
                          k=np.zeros(128, dtype=np.float32)),
            MaterialEntry(canonical_name="B",
                          n=np.linspace(2.0, 2.4, 128).astype(np.float32),
                          k=np.full(128, 0.05, dtype=np.float32)),
        ]
        # Three candidates: two feasible, one with empty structure (drops).
        cands = [
            Candidate(slot_indices=[0, 1], material_names=["A", "B"],
                      thicknesses_nm=[50.0, 100.0], achieved_lab=(0, 0, 0),
                      reflectance=[0.0] * 128, delta_e=float("nan"),
                      sample_index=1, multiplicity=3),
            Candidate(slot_indices=[1, 0], material_names=["B", "A"],
                      thicknesses_nm=[100.0, 50.0], achieved_lab=(0, 0, 0),
                      reflectance=[0.0] * 128, delta_e=float("nan"),
                      sample_index=2, multiplicity=1),
            Candidate(slot_indices=[], material_names=[],
                      thicknesses_nm=[], achieved_lab=(0, 0, 0),
                      reflectance=[0.0] * 128, delta_e=float("nan"),
                      sample_index=3, multiplicity=1),
        ]
        knobs = InferenceKnobs(top_k=2, weight_lambda=1.0, tolerance_pct=5.0)
        target = (50.0, 0.0, 0.0)
        cs = ConstraintSet(constraints=[])

        top, drops, rep = select_top_k(cands, pool, target, cs, knobs)
        # The empty structure should have been dropped.
        assert drops.get("empty_structure", 0) == 1
        # Two feasible → at most 2 in top.
        assert len(top) <= 2
        # Each chosen candidate has a real ΔE + objective set.
        for c in top:
            assert np.isfinite(c.delta_e)
            assert np.isfinite(c.objective)
            assert c.robustness.grad_l2_shift >= 0
        # Top sorted ascending by objective.
        objectives = [c.objective for c in top]
        assert objectives == sorted(objectives)
        print(f"[select] sim+score+rank OK; top objectives = "
              f"{[round(o, 4) for o in objectives]}")

        # FeasibilityError path: drop every candidate with a constraint that
        # fails on everything.
        from inference.src.constraints import AllowedSubset
        cs_strict = ConstraintSet(constraints=[
            AllowedSubset(allowed_names=("nonexistent",)),
        ])
        try:
            select_top_k(cands[:2], pool, target, cs_strict, knobs)
            assert False, "expected FeasibilityError"
        except FeasibilityError as exc:
            assert exc.n_unique == 2
            assert exc.dropped_per_constraint.get("allowed_subset", 0) == 2
            print(f"[select] FeasibilityError path OK; drops="
                  f"{exc.dropped_per_constraint}")
    except ImportError as exc:
        print(f"[select] smoke skipped (missing dep): {exc}")
        return
    print("[select] smoke OK")


if __name__ == "__main__":
    _smoke_test()
