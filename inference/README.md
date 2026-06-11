# INDIGO inference branch

Status: end-to-end pipeline runnable via CLI on a saved cross_attn
checkpoint. LLM-based prompt parsing (`parse.py`) is the last remaining
module — for now the CLI takes a structured constraints JSON file directly.

## What's here

```
inference/
├── src/
│   ├── simulate.py     JAX-native compute_reflectance + reflectance_to_lab
│   │                    + ciede2000 + delta_e_from_thicknesses.
│   ├── schema.py       InferenceSpec, Candidate, Result, Provenance,
│   │                    RobustnessReport, ConstraintCheck, EnsembleStats.
│   │                    JSON round-trip; pool fingerprinting.
│   ├── constraints.py  8 constraint kinds (AllowedSubset, LayerIdentity,
│   │                    AdjacentForbidden, ThicknessRange, LayerCount,
│   │                    OrderingBefore, TotalThickness, Symmetry) with
│   │                    unified `check + decode_mask`; ConstraintSet
│   │                    composes them and partitions during/post.
│   ├── generate.py     Batched constrained ensemble decoder over a single
│   │                    model forward per step; per-replica torch.Generator
│   │                    for reproducibility; strict dedup.
│   ├── sensitivity.py  grad_robustness (R_max + R_l2) and
│   │                    monte_carlo_robustness (p50 / p95 / worst).
│   ├── select.py       simulate_and_score, filter_feasible, top_k_by_objective,
│   │                    select_top_k all-in-one. FeasibilityError on empty.
│   ├── refine.py       Projected Adam on continuous nm, multi-start,
│   │                    post-refine constraint recheck + J-regression fallback.
│   └── solve.py        Orchestrator. Pure function:
│                        solve(model, pool, spec) -> Result.
├── scripts/
│   ├── sim_spike.py    De-risker for simulate.py.
│   └── run_inference.py CLI: pool from JSON or JLL dir, target Lab, optional
│                        constraints JSON, knobs as flags.
└── outputs/             Persisted Result JSONs.
```

## What's left

1. `parse.py` — forced-JSON LLM prompt → InferenceSpec with the three
   validation gates (schema, semantic, physical-sense).
2. End-to-end run on the cluster with the real cross_attn checkpoint.
3. (later) Eval harness, GUI.

## Known constraint: no `jax.jit` / `jax.vmap` on the physics chain

`jaxlayerlumos.jaxlayerlumos.stackrt_eps_mu_base` contains
`assert thicknesses[0] == 0`. Under `jax.jit` / `jax.vmap` the argument is
a Tracer and the `__bool__` call inside `assert` raises
`TracerBoolConversionError`. `jax.grad` is fine because grad tracing
keeps concrete values around.

Consequence: every candidate is simulated **sequentially**. For the
planned ensemble of N=500 this is ~1–3 s of simulation per inference
call — acceptable. If we ever need vmap speed, the fix is either to
monkey-patch the JLL assert away at module load or to reimplement the
(small) normal-incidence TMM in pure JAX. Both are tracked work; neither
blocks the current build.

## Conventions

- **Numerical fidelity with training.** `reflectance_to_lab` routes
  through `jaxlayerlumos.colors.composite.spectrum_to_sRGB` — the same
  path as `src/color_utils.py:spectrum_to_lab` — so achieved Lab from
  inference matches training labels to ~0.05 per channel.
- **Fixed shapes everywhere.** Pool is always padded to `M_MAX=32`;
  structures always to `MAX_LAYERS=10` with a boolean layer mask. Lets
  `vmap` ride over the ensemble without dynamic shapes.
- **D65/2°, normal incidence, air/fused-silica.** Conditions are fixed
  in the simulator and never exposed as user knobs. They match the
  training dataset convention so reflectance↔color↔Lab is consistent.
- **float64 in the spike.** `jax_enable_x64` is on so the finite-
  difference vs `jax.grad` check has room to breathe; production code
  may run float32 for speed once the physics is trusted.

## Re-running the spike

```bash
python inference/scripts/sim_spike.py
```

If anything fails, do NOT continue building constraints/generate against
a broken physics chain — fix the spike first.
