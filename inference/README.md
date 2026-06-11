# INDIGO inference branch

Status: scaffolding + JAX physics chain landed. Constraints / generate /
refine / parse / solve still to come.

## What's here

```
inference/
├── src/
│   ├── simulate.py      JAX-native compute_reflectance + reflectance_to_lab
│   │                    + ciede2000 + delta_e_from_thicknesses. Fixed-shape
│   │                    [MAX_LAYERS+2] stack with masked layers for vmap.
│   └── schema.py        InferenceSpec, Candidate, Result, Provenance,
│                        RobustnessReport. JSON round-trip for the GUI
│                        contract. Pool fingerprinting.
├── scripts/
│   └── sim_spike.py     De-risker. Checks (1) reflectance vs numpy sim,
│                        (2) Lab vs src.color_utils.spectrum_to_lab, (3)
│                        ΔE_00 vs reference numpy port, (4) jax.grad vs
│                        centered FD, (5) jit/grad pipeline, (6) schema
│                        round-trip. Run after every change to simulate.py.
└── outputs/             Persisted Result JSONs (gitignored downstream).
```

## What's not here yet

Per the implementation plan, in build order:

1. `constraints.py` — 8 checks + decode-mask interface
2. `generate.py` — batched constrained ensemble decoding (with **slot
   encoder cached once per pool**)
3. `select.py` — weighted objective `J = ΔE + λ·R`, feasibility error
4. `sensitivity.py` — gradient predicted shift + Monte-Carlo top-k
5. `refine.py` — gradient local search, multi-start, post-refine recheck
6. `parse.py` — forced-JSON LLM prompt → InferenceSpec with validation gates
7. `solve.py` — orchestrator
8. `scripts/run_inference.py` — CLI

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
