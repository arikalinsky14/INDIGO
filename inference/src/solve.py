"""
INDIGO inference orchestrator.

  solve(model, pool, spec, knobs=None) -> Result

This is the pure-function boundary the GUI / CLI will call. Every other
module in `inference/src/` is composed inside this function; no module
imports `solve.py`.

Pipeline
--------
  1. generate.py        → N candidates (deduped, decode-time feasible)
  2. select.py          → sim + grad robustness + post-hoc filter → top_k
                          • FeasibilityError on empty feasible set → failure Result
  3. refine.py          → projected Adam on continuous nm, multi-start opt
  4. sensitivity.py     → MC robustness on the (refined) top_k
  5. re-rank by post-refine J (refine may have changed ordering)
  6. schema.py          → Result envelope with chosen + alternatives +
                          constraints_report + ensemble_stats + provenance

Conservatism choices retained
-----------------------------
  - All N candidates fit in one model forward by default (chunk_size=None).
    Bump `knobs` if you ever need to spread memory.
  - MC robustness runs on top_k only — K=32 × 5 ≈ 160 sequential sims per
    inference call. < 1 s on the cluster.
  - solve() takes a pre-loaded model so the caller controls device + load
    cost; we don't re-load weights here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from inference.src.constraints import ConstraintSet
from inference.src.generate import (
    GenerationConfig, generate_ensemble, load_inference_model,
)
from inference.src.refine import refine_top_k
from inference.src.schema import (
    Candidate, ConstraintCheck, EnsembleStats, InferenceKnobs, InferenceSpec,
    MaterialEntry, Provenance, Result, RobustnessReport, pool_fingerprint,
)
from inference.src.select import FeasibilityError, select_top_k, top_k_by_objective
from inference.src.sensitivity import grad_robustness, monte_carlo_robustness


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------

def solve(
    model: torch.nn.Module,
    pool: List[MaterialEntry],
    spec: InferenceSpec,
    *,
    model_tag: str = "",
    model_sha256: str = "",
    incidence_angle: float = 0.0,
    n_random_restarts: int = 0,
    device: Optional[torch.device] = None,
) -> Result:
    """Run the full inference pipeline. Returns a Result either way:
    success → `chosen` populated; failure → `chosen=None`, `errors[...]`."""
    knobs = spec.knobs
    constraint_set = ConstraintSet(constraints=list(spec.constraints))

    pool_fp = pool_fingerprint(pool)
    prov = Provenance.capture(
        model_tag=model_tag, model_sha256=model_sha256,
        pool_fp=pool_fp, seed=knobs.seed,
    )

    # 1. Generate.
    gen_cfg = GenerationConfig(
        ensemble_N=knobs.ensemble_N,
        temperature=knobs.temperature,
        base_seed=knobs.seed,
    )
    try:
        candidates, ens_stats = generate_ensemble(
            model=model, pool=pool, target_lab_raw=spec.target_lab_raw,
            constraint_set=constraint_set, cfg=gen_cfg, device=device,
        )
    except Exception as exc:
        return _failure_result(
            spec, prov, errors=[f"generate failed: {type(exc).__name__}: {exc}"],
        )

    # 2. Simulate + score + post-hoc filter + top_k.
    try:
        top, drop_counts, constraints_report = select_top_k(
            candidates, pool, spec.target_lab_raw, constraint_set, knobs,
            incidence_angle,
        )
    except FeasibilityError as exc:
        # Failure envelope: chosen=None, drop counts attached so the GUI can
        # render "0 / N feasible after constraint X dropped K".
        ens_stats.n_feasible = 0
        ens_stats.dropped_per_constraint = exc.dropped_per_constraint
        return _failure_result(
            spec, prov, ens_stats=ens_stats,
            errors=[str(exc)],
        )

    ens_stats.n_feasible = len(candidates) - sum(drop_counts.values())
    ens_stats.dropped_per_constraint = drop_counts

    # 3. Refine.
    refined, _diag = refine_top_k(
        candidates=top, pool=pool, target_lab_raw=spec.target_lab_raw,
        constraint_set=constraint_set, knobs=knobs,
        incidence_angle=incidence_angle, n_random_restarts=n_random_restarts,
    )
    ens_stats.n_refined = sum(1 for c in refined if c.refined)

    # 4. MC robustness on the (refined) top_k. Honest robustness numbers for
    # the report; the gradient version was used as the ranking signal.
    if knobs.tolerance_pct > 0 and knobs.mc_samples > 0:
        for idx, c in enumerate(refined):
            mc = monte_carlo_robustness(
                pool, c.slot_indices, c.thicknesses_nm,
                spec.target_lab_raw, knobs.tolerance_pct,
                K=knobs.mc_samples, seed=knobs.seed + 1000 + idx,
                incidence_angle=incidence_angle,
            )
            c.robustness = RobustnessReport(
                grad_predicted_shift=c.robustness.grad_predicted_shift,
                grad_l2_shift=c.robustness.grad_l2_shift,
                mc_p50=mc.p50, mc_p95=mc.p95, mc_worst=mc.worst,
                mc_samples=mc.K_used,
            )

    # 5. Re-rank in case refinement changed J ordering.
    final_ordered = top_k_by_objective(refined, knobs.top_k)
    ens_stats.n_returned = len(final_ordered)

    # 6. Result.
    if not final_ordered:
        return _failure_result(
            spec, prov, ens_stats=ens_stats,
            errors=["empty top-k after refinement"],
            constraints_report=constraints_report,
        )

    return Result(
        spec_echo=spec,
        chosen=final_ordered[0],
        alternatives=final_ordered[1:],
        constraints_report=constraints_report,
        ensemble_stats=ens_stats,
        provenance=prov,
        errors=[],
    )


# ----------------------------------------------------------------------------
# Convenience: load model + solve
# ----------------------------------------------------------------------------

def solve_from_checkpoint(
    checkpoint_dir: Path,
    pool: List[MaterialEntry],
    spec: InferenceSpec,
    *,
    incidence_angle: float = 0.0,
    n_random_restarts: int = 0,
    device: Optional[torch.device] = None,
) -> Result:
    """Loads model.pt + config.json from disk, then `solve()`. The model_tag
    + model_sha256 are recorded in Provenance automatically.
    """
    model, config, sha = load_inference_model(checkpoint_dir, device=device)
    return solve(
        model=model, pool=pool, spec=spec,
        model_tag=config.tag(), model_sha256=sha,
        incidence_angle=incidence_angle,
        n_random_restarts=n_random_restarts, device=device,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _failure_result(
    spec: InferenceSpec, prov: Provenance,
    *,
    ens_stats: Optional[EnsembleStats] = None,
    errors: Optional[List[str]] = None,
    constraints_report: Optional[List[ConstraintCheck]] = None,
) -> Result:
    return Result(
        spec_echo=spec,
        chosen=None,
        alternatives=[],
        constraints_report=constraints_report or [],
        ensemble_stats=ens_stats or EnsembleStats(),
        provenance=prov,
        errors=errors or [],
    )
