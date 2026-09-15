# CLAUDE.md — INDIGO project notes for future Claude sessions

Repo: **INDIGO** (Pitt CRC). Flexible-material RGB/Lab → thin-film-stack
generative model. Autoregressive decoder predicts (slot, thickness) per
layer; the material pool varies per example (encoder is pool-agnostic).

## Current state (Sept 15, 2026)

**Val (greedy ΔE₀₀ on 1000 val examples):**

| Model | val_loss_de | Notes |
|---|---|---|
| Pretrain (`prod_3ep_bs512_lr6e-5/step_13000`) | ~8.55 | 3 epochs CE, val-optimal step |
| K=3 slot finetune, no sim-feedback | 8.02 | LR=1e-5, CE=0.1, const LR, unfrozen encoder |
| **K=3 slot finetune + sim-feedback** | **7.92** | Same recipe + `SIM_FEEDBACK=1`; step 1250 of 1665 — current champion |
| hier M=4 × N=3 + simfb + prefix-aug 0.20 (Sept 14) | 8.06 | worse than K=3+simfb; fell into flat-target pathology |

**Inference-time (ensemble N=200 + real-sim select) on 500 rows × 2 tiers:**

| Model | tier_a med / p95 | tier_b med / p95 |
|---|---|---|
| Pretrain (no simfb) | 0.717 / 3.735 | 0.535 / 2.726 |
| K=3 slot finetune (no simfb) | 0.658 / 3.423 | 0.564 / 3.008 |
| K=3 slot + simfb, SIM_FEEDBACK=1 | 0.710 / 3.438 | **0.509** / 2.836 |

Bucket rates (<1.0, <2.0, <5.0, <10.0) all within ~1 percentage point
of each other on tier_a. **The 200-replica ensemble washes out
val_de differences of 0.6 units — measurement problem** (see Sept 15
findings below).

**The bottleneck is high-chroma / edge-of-gamut colors** — the ~5% p95+
tail. Median already crushes the target. HC=0.30 finetune data was
generated specifically to help this tail.

## Key architectural pieces (know these first)

### Finetune pipeline — `src/de_finetune.py` + `scripts/finetune_de.py`

Three loss modes selected by `TOPK_MODE`:
- `slot` (default) — top-K over per-slot max-thickness scores; each
  candidate uses its argmax thickness. Best empirically.
- `joint` — top-K over the flat (slot × thickness) grid. Sept 10:
  **concentrates on 1-2 slots' neighbor thicknesses → uniform loss →
  no learning.** Kept but don't use.
- `hierarchical` — top-K slots × top-N thicknesses per slot (K·N
  candidates). Untested at scale. `THICKNESS_TOPN` sets N.

Also:
- `EPSILON_START/END` — ε-exploration: floor(K·ε) random candidates,
  rest top-K. Anneals linearly.
- `CE_LOSS_WEIGHT` — CE anchor to GT tokens. **0.1 is the sweet spot.**
- `LR_SCHEDULE={cosine, constant}` — constant holds val_de near peak;
  cosine peaks then degrades ("cosine death").
- `SIM_FEEDBACK` — see below.
- Best-checkpoint autosave to `<SAVE_DIR>/best/` on every val
  improvement. Look at that dir, not `latest/`.

### Sim-feedback residual (Sept 13 architectural add)

- **Model** (`src/model.py:FlexMaterialCrossAttn`): new `residual_proj:
  Linear(3, d_model)` **zero-initialised**. Optional `residual_labs`
  kwarg to `forward()` — shape `[B, SEQ_LEN, 3]` in normalised-Lab
  space. When None or all-zeros: forward is bit-identical to
  pre-residual model. Pretrained checkpoints load with
  `strict=False`; missing residual_proj weights init to zero.
- **Training** (`_compute_partial_residuals` in `src/de_finetune.py`):
  for each example at each position k>0, real-sim the first k GT
  layers and compute residual = target − partial_sim (normalized).
  Cost: ~25% overhead on top of the top-K sims.
- **Inference** (Sept 14 plumbing): `generate_ensemble` in
  `inference/src/generate.py` maintains a `residual_labs_all[N,
  SEQ_LEN, 3]` cache. At step k>0 it fills position k for each
  replica from prefix sim. `sim_feedback` flag threads through
  `solve()` → `test_eval.py` → SLURM env `SIM_FEEDBACK=1`.
- **Loader** (`load_inference_model`): uses `strict=False`. Any
  pre-sim-feedback checkpoint (including the production pretrain
  baseline) loads cleanly; `residual_proj.{weight,bias}` init to zero
  and the residual path is a bit-identical no-op. Fixed Sept 14 after
  the first eval attempt hit `Missing key(s) in state_dict: "residual_proj.*"`.

**Only checkpoints finetuned with `--sim-feedback` have non-zero
residual weights.** Running inference with `SIM_FEEDBACK=1` on a
pretrain-only checkpoint is a no-op (zero-init residual_proj).

### Prefix augmentation for the residual (Sept 14)

**Why**: at training, residual = target − sim(GT_prefix). At val /
inference, residual = target − sim(model_prefix). Model_prefix drifts
on hard examples, so the residual distribution the model sees at
deployment is systematically noisier than what it trained on. Prefix
aug narrows the gap by perturbing GT prefix thicknesses before sim'ing
the residual — teaches the model to consume a noisy residual channel.

**Design** (`src/de_finetune.py:_compute_partial_residuals`): per
example, per prefix layer independently, with prob `PREFIX_AUG_PROB`,
multiply thickness by Uniform(1-`scale`, 1+`scale`). Perturbations are
coherent across k (one draw per layer, used at every prefix depth
that touches that layer) so growing prefixes see a consistent noisy
trajectory. Materials are NOT swapped — material errors produce
residuals too large / off-distribution to be useful supervision.

**Model input tokens and top-K target are unchanged.** Only the
residual-sim input is noisy. This is the cheapest useful form of
train↔inference alignment; the more principled scheduled-sampling /
DAgger version would also perturb the tokens the model sees, but
that breaks CE loss and adds bookkeeping.

**Knobs** (env vars): `PREFIX_AUG_PROB` (default 0.0, off),
`PREFIX_AUG_THICKNESS_SCALE` (default 0.15). Only active when
`SIM_FEEDBACK=1`. No effect on wall clock.

### Inference-time search — `inference/src/`

- `generate.py:generate_ensemble` — broadcasts one (Lab, pool) across
  N replicas, each temperature-samples with constraint masks. 200 is
  test default; production 500.
- `solve.py:solve()` — full pipeline: generate → real-sim → select
  top-K by objective → refine (thickness gradient) → optional MC.
- `test_eval.py` — sweeps `data/test/tier_{a,b}` rows, reports ΔE
  distribution + threshold pass-rates.

## Session file map

**Design docs:**
- `analyses/de_finetune/README.md` — full ΔE finetune design
- `analyses/de_finetune/GRADIENT_FLOW_EXPLAINER.md` — 6-piece walk of
  how gradients flow through the STE + sim pipeline

**SLURM wrappers:** all in `slurms/`
- `finetune_de.sh` — the workhorse; env vars listed in header. Look
  at the multi-block usage examples at the bottom for recipe patterns.
- `test_eval.sh` — `CHECKPOINT=... sbatch slurms/test_eval.sh`. **Do
  NOT set `--qos=short`** (caps at 3h regardless of `--time`); we
  removed it Sept 14.
- `thickness_distribution_plot.sh` — faceted P(thickness | winning
  slot) plot; 32 examples in ~1 min.
- `ste_projection_quality.sh` — diagnostic that measured why the
  original STE finetune was harmful (~41% top-1 at model anchors).

**Analyses:**
- `analyses/de_finetune/ste_projection_quality.py` — the diagnostic
  that killed STE mode and motivated top-K real-sim.
- `analyses/de_finetune/thickness_distribution_plot.py` — the
  bimodality visualizer.

**Checkpoints on cluster** (under
`/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/`):
- `prod_3ep_bs512_lr6e-5/step_13000/` — production pretrain, use as
  `PRETRAINED_CHECKPOINT` for all finetune runs
- `finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best/` — pre-sim-feedback
  best (val_de 8.02)
- `finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best/` —
  sim-feedback best (val_de 7.92); the current champion

## Journey notes (why we ended up here)

1. **STE finetune** (original) was net-harmful. `ste_projection_quality.py`
   showed the projected-gradient onto non-winning materials has only ~41%
   top-1 accuracy at model-argmax anchors. All LRs degraded val_de.
2. **CE anchor** (λ ∈ {0.1, 1, 10, 100}) held things stable but ΔE
   contribution was net-zero. λ=100 (~pure CE) matched λ=10.
3. **C1 top-K real-sim listwise loss** replaced STE linearization.
   Sim K candidates per position, loss = CE(softmax(logits[topK]),
   softmax(-β·ΔE_real)). This is when things started working.
4. **Unfrozen encoder (Experiment B)** unlocked ~0.15 val_de vs frozen.
5. **Constant LR** avoided cosine-death degradation past mid-training.
6. **Joint mode failed** because top-K concentrates on 1-2 slots'
   neighbor thicknesses → target dist is flat → no rank signal.
7. **Sim-feedback residual** finally broke through the ~8.0 plateau
   → 7.92 (Sept 13). Small but real, and beat all previous variants.
8. **Hierarchical M=4×N=3 also failed at β=1** (Sept 14 run) for
   the same "flat target" reason as joint mode: with 12 candidates
   whose ΔE spans only a few units, softmax(-1·ΔE) is nearly
   uniform. Diagnosed Sept 15 via new flat-target metrics
   (target_entropy near log(12), argmin_hit near 1/12). Fix under
   test: bump β to 5-10 to sharpen the target distribution.

## Sept 15 findings (what we learned overnight)

### 1. Hierarchical M=4 × N=3 fell into the flat-target pathology

The hier + simfb + prefix-aug 0.20 run (job 3946535) hit val_de=8.06
peak at step 1000, worse than the K=3 slot + simfb baseline (7.92)
and degraded past step 1000 (8.42, 8.50).

**Diagnostic signature**: `loss_topk ≈ 2.4` throughout (log(12) =
2.485 → uniform). `argmin_hit ≈ 0.12` (1/12 = 0.083 → random). The
model is not learning to rank the 12 candidates.

**Root cause hypothesis**: the target distribution `softmax(-β·ΔE)`
with β=1 is nearly uniform because the 12 candidates within any
position have small ΔE spread relative to β. The K=3 slot mode
worked because 3 slot-diverse candidates have wider ΔE spread (each
material fundamentally different) and log(3)=1.10 is easier to
beat.

**Fix to test**: bump `SIM_TARGET_BETA` from 1.0 to 5-10, keeping
everything else identical. β=5 concentrates ~80% of the target mass
on the top-2 candidates and gives the model something to rank
toward. **Flat-target diagnostics wired Sept 15** (see below) will
tell us whether the fix worked at step 1, not step 1000.

### 2. Ensemble decoder is the measurement bottleneck at inference

val_de spans 7.92 → 8.55 across our checkpoints (0.63 unit gap), but
at N=200 ensemble the tier_a medians span 0.658 → 0.717 (~0.06 unit,
= noise). **The 200-replica real-sim selector is too generous** —
with that many diverse samples and real-sim scoring, model quality
barely matters. We can't distinguish models with N=200.

**Fix to test**: rerun test_eval with `ENSEMBLE_N=20` (or even 10).
If model quality matters, tier_a medians should widen and we'll see
which model actually helps. Bonus: 10× faster.

### 3. Sim-feedback at inference: winning on tier_b, losing on tier_a

The SIM_FEEDBACK=1 test won on tier_b median (0.509 vs 0.564 non-simfb
finetune) but LOST on tier_a median (0.710 vs 0.658). Consistent with
the "residual-distribution-shift" hypothesis: training saw
sim(GT_prefix), inference sees sim(model_prefix), which differs
systematically — the residual channel becomes noise on hard examples.

**Fix to test**: run K=3 slot + simfb WITH prefix-aug (the whole point
of prefix-aug — this was mixed into the hier run and masked by the
flat-target problem, so we still don't know if prefix-aug helps).

### 4. Flat-target diagnostics (added Sept 15)

`_topK_sim_loss_for_example` now logs three per-position metrics that
identify the pathology from step 1:

- `target_entropy` — H(softmax(-β·ΔE)); log(K) = uniform, 0 = peaked
- `target_max_prob` — max target weight; 1/K = uniform, 1 = peaked
- `topk_delta_e_range` — (max ΔE − min ΔE) across candidates; small
  range means no β can peak the target (candidates are too similar)

Watch the training log — new fields `tgt_H`, `tgt_max`, `dE_rng`.
Guideline: if `tgt_max < 2/K`, β is too low OR the candidate set is
too correlated.

Also: **ε-exploration is now wired into hierarchical mode** (was
future work Sept 13). Setting `EPSILON_START>0` in hierarchical picks
floor(K·ε) random slots per position, each still getting its own
top-N thickness.

## Open threads / suggested next work

### Immediate — parallel experiments to launch (Sept 15)

Fires four SLURM jobs in parallel; total ~29 GPU-hours to answer the
four questions above.

**P1** (finetune, ~5h): **prefix-aug on K=3 slot + simfb** — clean
single-variable test of prefix aug on the winning recipe.
**P2** (finetune, ~12h): **hier M=4×N=3 + simfb + prefix-aug + β=5**
— unstick the flat-target pathology.
**P3** (finetune, ~12h): **hier M=4×N=3 + simfb + prefix-aug + β=10**
— aggressive β, in case β=5 isn't enough.
**P4** (test_eval, ~1h × 3): **ensemble N=20 on the 3 existing
checkpoints** — unmask the model quality differences the N=200
ensemble is hiding.

Copy-paste commands are in Common Invocations.

### Post-P1-P4 decisions (what to do based on results)

- If P4 shows N=20 spreads the medians: **run all future test_evals at
  N=20** as a discrimination probe, keeping N=200 for the "final
  product" number.
- If P2/P3 loss_topk drops well below log(12) AND val_de beats 7.92:
  hierarchical is unlocked → try M=6 × N=3 next.
- If P2/P3 fixes loss_topk but val_de still plateaus at 8: candidate
  diversity isn't the bottleneck — try ε-exploration on top of the
  fixed recipe.
- If P1 beats 7.92: prefix-aug alone is the win; sim-feedback +
  prefix-aug becomes the new baseline for all future work.

### Longer-term ideas

- **EMA / weight averaging** across recent-best checkpoints to smooth
  the peak-then-oscillate pattern.
- **Curriculum by chroma magnitude**: train easy → hard so the model
  builds representations before hitting the hard tail.
- **Larger dataset**: 213k might not be enough. Scale to 500k or 1M
  once the recipe is settled.
- **Batched-vmap partial-residual sim** to cut the current 25% wall
  overhead of `_compute_partial_residuals`.
- **DAgger / scheduled sampling**: during training, sometimes feed
  the model its own generated prefix (no-grad rollout) and use the
  resulting sim residual. Directly closes the train↔inference gap
  (cleaner than prefix-aug's mild inconsistency) but more complex.

## Cluster / environment gotchas

- **Login-node**: don't run heavy Python there; even a model forward
  can trigger resource kills. Use SLURM even for small analyses.
- **`--qos=short` caps at 3h** regardless of `--time`. Removed from
  `test_eval.sh` Sept 14. Other slurms use `qos=short` intentionally
  (they finish under 3h).
- **QoS default may need to be raised for multi-day runs.** If a submit
  is rejected, try `#SBATCH --qos=long` or contact CRC.
- **JAX_PLATFORMS=cpu** is set in most slurms because the sim path
  isn't GPU-amenable (physics is unrolled small-matrix ops); GPU stays
  free for torch.
- **Working dir**: `$HOME/Github/INDIGO`; data lives at
  `/ix1/ohinder/ajk245/Github/INDIGO/data/`.

## Git workflow (this session's pattern)

Develop on `claude/build-chroma-indigo-LYNpa`, then merge to `main`
per user request. Always push both. Commit message trailer:
```
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
Claude-Session: <session URL>
```
The trailer is auto-inserted from the session's attribution config.

## Common invocations (copy-paste ready)

### Sept 15 parallel experiments (P1-P4)

**P1 — prefix-aug on K=3 slot + simfb (clean single-variable test):**
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb_paug20 \
    FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=3 CE_LOSS_WEIGHT=0.1 \
    TOPK_MODE=slot LR_SCHEDULE=constant SIM_FEEDBACK=1 \
    PREFIX_AUG_PROB=0.20 PREFIX_AUG_THICKNESS_SCALE=0.15 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=05:00:00 slurms/finetune_de.sh
```

**P2 — hier M=4×N=3 + simfb + prefix-aug + β=5 (unstick flat target):**
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_hier4x3_ce0p1_lr1e5_213k_const_simfb_paug20_beta5 \
    FREEZE_ENCODER=0 LR=1e-5 CE_LOSS_WEIGHT=0.1 LR_SCHEDULE=constant \
    TOPK_MODE=hierarchical REAL_SIM_TOPK=4 THICKNESS_TOPN=3 \
    SIM_TARGET_BETA=5.0 \
    SIM_FEEDBACK=1 PREFIX_AUG_PROB=0.20 PREFIX_AUG_THICKNESS_SCALE=0.15 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=12:00:00 slurms/finetune_de.sh
```

**P3 — hier M=4×N=3 + simfb + prefix-aug + β=10 (aggressive β):**
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_hier4x3_ce0p1_lr1e5_213k_const_simfb_paug20_beta10 \
    FREEZE_ENCODER=0 LR=1e-5 CE_LOSS_WEIGHT=0.1 LR_SCHEDULE=constant \
    TOPK_MODE=hierarchical REAL_SIM_TOPK=4 THICKNESS_TOPN=3 \
    SIM_TARGET_BETA=10.0 \
    SIM_FEEDBACK=1 PREFIX_AUG_PROB=0.20 PREFIX_AUG_THICKNESS_SCALE=0.15 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=12:00:00 slurms/finetune_de.sh
```

**P4 — small-ensemble test_evals (unmask model differences):** fire
all three in parallel. Each ~1h at N=20. Use a distinct
`OUTPUT_DIR_SUFFIX` so results don't collide with the N=200 runs.

```bash
# P4a: pretrain baseline @ N=20
CHECKPOINT=data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    LIMIT=500 ENSEMBLE_N=20 TEMPERATURE=1.0 \
    OUTPUT_DIR=inference/outputs/test_eval_n20 \
    sbatch --time=03:00:00 slurms/test_eval.sh

# P4b: K=3 slot no simfb @ N=20
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best \
    LIMIT=500 ENSEMBLE_N=20 TEMPERATURE=1.0 \
    OUTPUT_DIR=inference/outputs/test_eval_n20 \
    sbatch --time=03:00:00 slurms/test_eval.sh

# P4c: K=3 slot + simfb, SIM_FEEDBACK=1 @ N=20
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best \
    LIMIT=500 ENSEMBLE_N=20 TEMPERATURE=1.0 SIM_FEEDBACK=1 \
    OUTPUT_DIR=inference/outputs/test_eval_n20 \
    sbatch --time=03:00:00 slurms/test_eval.sh
```

### Standing recipes

**Finetune (winning K=3 slot + simfb baseline, Sept 13):**
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb \
    FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=3 CE_LOSS_WEIGHT=0.1 \
    TOPK_MODE=slot LR_SCHEDULE=constant SIM_FEEDBACK=1 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=05:00:00 slurms/finetune_de.sh
```

**Test eval (final-product N=200 setting):**
```bash
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best \
    LIMIT=500 ENSEMBLE_N=200 TEMPERATURE=1.0 SIM_FEEDBACK=1 \
    OUTPUT_DIR=inference/outputs/test_eval \
    sbatch slurms/test_eval.sh
```

**Thickness distribution plot:**
```bash
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best \
    OUTPUT_PATH=analyses/de_finetune/results/thickness_dist_simfb.png \
    sbatch slurms/thickness_distribution_plot.sh
```
