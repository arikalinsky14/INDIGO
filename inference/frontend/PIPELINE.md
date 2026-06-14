# INDIGO inference — pipeline reference

How `solve(model, pool, spec) → Result` works end to end. The frontend's
"how it works ⓘ" drawer shows a condensed version of this same content
inline next to the advanced knobs. This file is the long form.

The pure-function boundary is `inference.src.solve.solve()`. Every stage
below corresponds to one module in `inference/src/`. No shared global
state; the orchestrator is the only thing that knows about the others.

## Stages, in execution order

### 1. (Prompt mode only) Parse free-text → InferenceSpec

Module: `inference/src/parse.py`.

If you used the Prompt tab, the LLM (`gpt-4o-mini` by default; takes
`$OPENAI_API_KEY` or a per-request key from the GUI) is called with a
**forced JSON schema** that lists the 8 supported constraint kinds. The
response is run through three validation gates:

1. **Schema** — handled by the API's strict mode.
2. **Semantic** — every material name in the response must exist in the
   pool you supplied; layer positions live in `[0, MAX_LAYERS)`;
   thicknesses on the 5 nm grid; pool ≤ `M_MAX`.
3. **Physical** — `layer_count.min ≤ max`, `total_thickness.max_total_nm
   ≥ 5`, no empty `allowed_subset`, distinct `name_a / name_b` in
   `ordering_before`, …

Any gate failure raises `ParseError` carrying the gate name and an
actionable message — the server surfaces it as `HTTP 400`.

In Structured mode, this stage is skipped — your Lab + constraints JSON
become an `InferenceSpec` directly.

### 2. Encode pool

Module: `inference/src/generate.py` (`encode_pool`).

The pool is padded to `[M_MAX, 2, NUM_LAMBDA] = [32, 2, 128]` (n + k per
slot on the canonical wavelength grid), masked, and fed through the
shared **material encoder** — the same encoder used at training time, so
slots are permutation-equivariant. Runs once per request; negligible
cost.

### 3. Sample ensemble  
Knobs: **Ensemble N**, **Temperature**

Module: `inference/src/generate.py` (`generate_ensemble`).

The cross-attn model decodes `N` candidate structures **in one batched
forward pass**, conditioning on the target Lab + pool encoding. At each
decoding step:

- Per-replica `torch.Generator` (seeded from `base_seed × 7919 + r`) so
  every replica is deterministic.
- The slot-validity mask the model already produces is AND'd with the
  active **decode masks** from the constraint set (`allowed_subset`,
  `layer_identity`, `adjacent_forbidden`, `thickness_range`,
  `layer_count`, `total_thickness`, `ordering_before`, `symmetry`) and a
  universal **no-same-material-adjacent** rule. Forbidden tokens get
  `-inf` before sampling.
- Sample at `Temperature` (1.0 = trained distribution; <1 = sharper; >1
  = more diverse).

After the run, candidates are de-duplicated by exact `(slot_indices,
thicknesses)` tuple.

**Cost:** linear in `N`. Model forward is cheap; the dominant cost is
later (next stages).

### 4. Simulate candidates

Module: `inference/src/simulate.py` (`compute_reflectance →
reflectance_to_lab → ciede2000`).

For every unique candidate the JLL transfer-matrix physics computes
reflectance over the canonical 128-point frequency grid, then converts
to Lab via the same path training labels used, then computes ΔE_00 vs
the target.

**This is the dominant CPU cost.** The chain is sequential per
candidate — `jax.jit` / `jax.vmap` are blocked by an `assert
thicknesses[0] == 0` inside `jaxlayerlumos.stackrt_eps_mu_base` (the
constraint is documented in `simulate.py`'s docstring). On a typical
laptop CPU you can budget ~30–80 ms per candidate.

### 5. Select top-k  
Knobs: **Top-k**, **Tolerance %**, **Weight λ**

Module: `inference/src/select.py` + `inference/src/sensitivity.py`.

Each candidate is scored

```
J(t) = ΔE_00(t) + λ · R_l2(t)
```

where `R_l2 = ‖∂ΔE/∂t · τ · t‖₂` is the predicted color shift under a
manufacturing tolerance of `τ %` per layer (gradient computed via
`jax.grad`). If **Tolerance % = 0** the robustness term is skipped
entirely — that's the right setting for a pure-color demo and roughly
halves the wall clock.

The top **Top-k** candidates by J are kept; the rest are discarded as
alternatives. If post-hoc filters drop every candidate, `solve()` raises
`FeasibilityError` and the failure-mode `Result` envelope is returned
with `chosen: null` and `errors[]`.

### 6. Refine top-k  
Knobs: **Refine top-N**, **Refine iters**

Module: `inference/src/refine.py`.

Each of the first **Refine top-N** candidates (≤ Top-k) gets a projected
**Adam** local search over continuous thickness within constraint
bounds. The objective is ΔE_00 (the orchestrator recomputes the full J
after refinement); each iteration is one forward sim + one `jax.grad`
through the JLL physics. Up to **Refine iters** steps, terminates early
when ΔE stops moving.

Falls back to the unrefined seed if either (a) refinement violates a
post-hoc constraint or (b) `J(refined) > J(seed)` by more than a small
tolerance — the un-refined geometry can never lose against itself.

**This is the slowest single stage on CPU.** Cost ≈ `Refine top-N ×
Refine iters × (forward + grad)` ≈ `Refine top-N × Refine iters × 200
ms` on a typical CPU. With **Refine top-N = 1** and **Refine iters =
25** that's about 5 s; with **Refine top-N = 3** and **Refine iters =
80** you're at the 1-minute mark.

### 7. Finalise  
Knobs: **MC samples**

Module: `inference/src/sensitivity.py` (`monte_carlo_robustness`) +
`inference/src/solve.py`.

If **MC samples > 0**, K Monte-Carlo thickness perturbations resample
ΔE on each refined candidate to compute `p50 / p95 / worst` robustness
numbers. **MC samples = 0** skips this entirely.

Candidates are then re-ranked by J (refinement may have shifted the
order), the chosen pick wraps in a `Result` envelope along with all
alternatives, ensemble stats (sampled / unique / feasible / refined /
returned), and a Provenance receipt (model tag, sha256, pool
fingerprint, seed, library versions, timestamp). Returned to the UI.

## Hyperparameter quick reference

| Knob | Default | Affects | Typical range | Speed impact |
|---|---|---|---|---|
| **Ensemble N** | 150 | diversity, best-of accuracy | 50 – 2000 | linear; dominant when refinement is small |
| **Top-k** | 3 | how many alternatives are returned | 1 – 10 | small |
| **Refine top-N** | 1 | which of top_k get refined | 0 – top_k | **multiplicative** with refine_iters |
| **Refine iters** | 25 | Adam steps per refined candidate | 0 – 200 | dominant when refine_top_n ≥ 1 |
| **Temperature** | 1.0 | sampling sharpness | 0.5 – 1.5 | negligible |
| **Tolerance %** | 0 | manufacturing precision | 0 – 10 | enables robustness scoring (slower) |
| **Weight λ** | 1.0 | ΔE vs R_l2 trade-off (in J) | 0 – 2 | negligible |
| **MC samples** | 0 | robustness validation draws on refined top-k | 0 – 64 | linear with refine_top_n |
| **Seed** | 42 | deterministic reproduction | any int | none |
| **OpenAI API key** | — | LLM parse identity | sk-… | none |

## Speed presets (CPU laptop)

| Preset | Knobs | Approx wall | When to use |
|---|---|---|---|
| **fast** | N=100, k=1, refine_top_n=1, refine_iters=15, tol=0, mc=0 | 5–10 s | Live demos, click-and-show |
| **balanced** (default) | N=150, k=3, refine_top_n=1, refine_iters=25, tol=0, mc=0 | 15–25 s | Most exploratory use |
| **best** | N=500, k=5, refine_top_n=3, refine_iters=80, tol=5, mc=16 | 2–5 min | Quoting a final structure for fabrication |

## If a single solve is taking forever

In rough order of likely cause:

1. **`Refine top-N × Refine iters` is large.** Drop both. Each refine
   step is the worst per-unit cost in the pipeline because it does a
   forward + grad through the JLL physics with no JIT.
2. **First solve after server startup pays the JAX cold-start.** The
   server runs a tiny dummy solve on startup specifically to absorb this
   for you; pass `--no-prewarm` to opt out (NOT recommended for
   demos).
3. **MC samples > 0 multiplies refinement cost.** Set MC samples = 0 if
   you don't need robustness numbers.
4. **Tolerance > 0** turns on per-candidate gradient calls in the
   selection stage. Turn it off (Tolerance = 0) when you only care about
   nominal ΔE.
5. **N is huge.** Each candidate costs a sequential physics pass. Drop N
   to 100–200 for laptop CPU.

## Server-side flags

- `--cpu` — force CPU even if CUDA is technically present. Required on
  viz / non-GPU nodes where cuDNN isn't installed.
- `--no-prewarm` — skip the dummy-solve cold-start absorber. Server
  starts ~10–30 s faster but the first user request pays that cost
  instead.
