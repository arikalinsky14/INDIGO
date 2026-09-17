# CLAUDE.md — INDIGO project notes for future Claude sessions

## ⚠️ DIRECTION CHANGE — Sept 17, 2026 — read this first

Per research advisor guidance, the paper is refocusing on **pretrain
understanding**, not the ΔE finetune line of work. As of Sept 17:

- **All finetune experiments have been moved off `main`.** The code,
  analyses, SLURM wrappers, checkpoints, and diagnostics for the
  Sept 8 – Sept 16 finetune push live on the `finetune-experiment`
  branch. Checkout that branch to run any of it or rebuild on it.
- **`main` is now clean pretrain territory.** Only the `test_eval.sh`
  `--qos=short` removal (a general cluster fix that helps pretrain
  eval too) was carried forward from the finetune session.
- **This CLAUDE.md's finetune notes are preserved for reference below**
  — the numbers, journey, and dead-ends are worth keeping so we
  don't relitigate them if the finetune direction is ever revived.
  But the referenced files (`src/de_finetune.py`, `scripts/finetune_de.py`,
  `analyses/de_finetune/*`, `slurms/finetune_de.sh`, etc.) do NOT exist
  on `main`. To read/run them: `git checkout finetune-experiment`.

**For new work on `main`**: focus is pretrain — training curves,
ΔE evaluation of the pretrain checkpoints, model-analysis diagnostics,
paper-figure generation, dataset-quality checks. Anything ΔE-finetune-
adjacent goes on `finetune-experiment`.

---

## Finetune experiment notes (archived Sept 17, 2026 — for reference only)

The section below is the CLAUDE.md as it stood at the end of the
finetune push (Sept 16). Kept in place so future sessions have the
full picture of what was tried and learned. Files referenced here
live on the `finetune-experiment` branch.

---


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

**Inference-time (ensemble + real-sim select) on 500 rows × 2 tiers:**

| Model | N=200 tier_a med / p95 | H_slot | H_thick | N=20 tier_a med / p95 |
|---|---|---|---|---|
| Pretrain (no simfb) | 0.717 / 3.735 | 2.30 | 3.49 | 1.552 / 6.756 |
| K=3 slot finetune (no simfb) | 0.658 / 3.423 | 2.62 | 4.30 | 1.652 / 6.422 |
| K=3 slot + simfb, SIM_FEEDBACK=1 | 0.710 / 3.438 | 2.63 | 4.31 | 1.545 / 7.114 |

Val_de spread = 0.63 units. Ensemble N=200 tier_a spread = 0.06 units.
Ensemble N=20 tier_a spread = 0.10 units. `frac_unique = 1.000`
across all three (every one of 200 samples is a unique candidate).
**The ensemble decoder is largely model-agnostic in its current
form** — see Sept 15 finding #1. Note that log(pool=15)=2.71, so
finetune samples are at 97% of uniform vs pretrain at 85%.

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

## Sept 15 findings (what we learned overnight and today)

### 1. Ensemble decoder is largely model-agnostic (the primary finding)

val_de spans **7.92 → 8.55** across our checkpoints (0.63 unit gap).
At N=200 ensemble the tier_a medians span **0.658 → 0.717** (0.06
unit). Reducing to N=20 didn't help: medians spanned **1.545 → 1.652**
(0.10 unit). At both N the spread is a small fraction of the val_de
gap.

**Interpretation**: the ensemble decoder finds low-ΔE candidates in
the sampling *tail*; finetune moves the *mode*. Different things.
Because the pretrain-era model already samples with high entropy
(slot_ent ≈ log(pool_size), thick_ent ≈ log(NUM_THICKNESSES)),
temperature=1.0 sampling produces ~200 diverse candidates that the
real-sim reranker can pick from — model quality barely matters as
long as sampling is diverse. The training objective (make greedy
val_de lower) and the deployed metric (ensemble ΔE) are only weakly
correlated.

**What this means for the roadmap**: driving val_de below 7.92 by any
of the standard tricks (bigger K, prefix aug, β tuning, longer runs)
looks unlikely to move the ensemble inference metric that matters.
The interesting question shifts from "how do we lower val_de?" to
"can we make the *ensemble* itself better?"

### 2. β sharpening for hierarchical FAILED

Sept 15 P2/P3 (β=5, β=10) tested my Sept 14 flat-target hypothesis:

| Run | val_de best | tgt_max | tgt_H | argmin_hit |
|---|---|---|---|---|
| hier β=1 | 8.06 | ~0.09 | ~2.4 | 0.12 |
| hier β=5 | 8.20 | ~0.49 | ~1.3 | 0.12 |
| hier β=10 | 8.21 | ~0.51 | ~1.2 | 0.12 |

The `tgt_max` jumped from ~1/K (uniform) to ~0.5 (mass on top-2)
exactly as expected. So β sharpening DID make the target peaked.
But `argmin_hit` didn't move (still random-over-12) and val_de got
**worse** by 0.15 units. Sharpening made the model overcommit to a
specific (slot, thick) that didn't generalize.

**Revised understanding**: With a flat target, gradient spreads
across all K candidates weighted by their ΔE ordering — a smooth
learning signal. With a peaked target, the model force-pushes toward
the argmin candidate and collapses. In hierarchical mode, β=1 is
apparently the best of the bad options.

### 3. Sim-feedback at inference: winning on tier_b, losing on tier_a

The N=200 SIM_FEEDBACK=1 test won on tier_b median (0.509 vs 0.564
non-simfb finetune) but LOST on tier_a median (0.710 vs 0.658).
Consistent with the "residual-distribution-shift" hypothesis:
training saw sim(GT_prefix), inference sees sim(model_prefix). But
given the primary finding above, the whole SIM_FEEDBACK=1 vs =0
difference is within ensemble noise anyway.

### 4. Diagnostics added Sept 15

**Training-side** (`_topK_sim_loss_for_example`):
- `target_entropy` — H(softmax(-β·ΔE)); log(K) = uniform, 0 = peaked
- `target_max_prob` — max target weight; 1/K = uniform, 1 = peaked
- `topk_delta_e_range` — (max ΔE − min ΔE) across candidates
- Printed in the log as `tgt_H`, `tgt_max`, `dE_rng`

**Inference-side** (`generate_ensemble` + `test_eval.py`):
- `mean_slot_entropy_by_pos` — per-position empirical H of slots
  actually sampled across the N replicas
- `mean_thick_entropy_by_pos` — same for thicknesses
- `fraction_unique` — unique-after-dedup / N; near 1 = highly diverse
  sampling, near 1/K = degenerate
- Aggregated in `summary.json["sampling"]` and printed in the tier
  summary line as `H_slot`, `H_thick`, `frac_unique`

These are the ONLY way to tell if our finetune is reducing the
sampling diversity that ensemble inference depends on. Watch them
across the P1-P4 checkpoints.

**Also**: ε-exploration is wired into hierarchical mode (was future
work Sept 13). Setting `EPSILON_START>0` in hierarchical picks
floor(K·ε) random slots per position, each still getting its own
top-N thickness.

### Sept 16 addendum: sampling-entropy hypothesis was reversed

P7 (N=200 with the new sampling diagnostic) shows the OPPOSITE of
what I predicted. I hypothesized finetune might be *reducing*
sampling entropy (concentrating the model's proposal distribution
and hurting the ensemble). Actual:

| Checkpoint | H_slot | H_thick | tier_a med |
|---|---|---|---|
| Pretrain | 2.296 | 3.487 | 0.717 |
| K=3 slot finetune | 2.625 | 4.305 | 0.658 |
| K=3 slot + simfb | 2.626 | 4.306 | 0.710 |

Finetune INCREASES sampling entropy (85% → 97% of log(pool_size)).
And the two finetunes are essentially identical proposers — H_slot
matches to 3 decimals despite differing training objectives and a
0.10 val_de gap between them. This explains why their ensemble ΔE
is within noise of each other: they produce virtually the same
sampling distribution.

Implications:
- The "sampling collapse" concern from Sept 15 is dead. The current
  top-K real-sim loss with β=1 spreads probability across candidates
  because the flat-ish target puts non-negligible gradient on
  multiple candidates per step — the model raises multiple logits
  rather than concentrating.
- The finetune's val_de improvements come from making argmax pick
  match GT better, without concentrating overall sampling mass.
  Two proposers can have identical sampling distributions but
  different argmax picks.
- Suggests "make model a BETTER PROPOSER" is not "make it more
  peaked" — the pretrain is more peaked but samples worse
  candidates. What we'd need is peaked ON THE RIGHT CANDIDATES,
  which is exactly what best-of-N-in-training would optimize.

### 5. Neighbor-mode ε-exploration (added Sept 15)

**Motivation**: uniform ε-random draws sample slots from the pool
tail — candidates the model already discriminates against as
obviously bad. The learning signal comes from candidates the model
is AMBIGUOUS about (its 4th-8th ranked slots), not garbage.

**Design** (`_topK_sim_loss_for_example`, applies to slot and
hierarchical modes): with `epsilon_neighbor_m = M > 0`, restrict
ε-random draws to the M non-top-K slots with the HIGHEST model
logits (uniform draw within that pool). With `M = 0` (default),
old uniform-over-pool behavior.

**Knob**: `EPSILON_NEIGHBOR_M` env var, `--epsilon-neighbor-m` CLI
arg. Sensible starting value: `M = 2 · REAL_SIM_TOPK` (gives the
model ~2× more "next-best" candidates than the top-K itself). No
effect when `EPSILON_START = EPSILON_END = 0`.

**Why this may help the ensemble-agnosticism finding**: if finetune
is concentrating the model's sampling distribution (verifiable via
the sampling-entropy diagnostic in P6/P7), neighbor-mode training
may keep it peaked-but-not-collapsed — the model learns nuanced
ranking over plausible candidates, keeping sampling diversity where
it matters. If sampling entropy is already high across all
checkpoints, neighbor-mode is a moderate-expected-win but low-risk
addition.

**Sept 16 activation footgun (fixed)**: the original ε formula was
`K_random = int(K_eff * ε)` — floor rounding. With K=3 and ε=0.20,
that's `int(0.6) = 0`. The P8 run configured ε=0.20 + neighbor M=6
and got val_de=7.89@step500 (nominally beating 7.92) BUT with
K_random=0 — neighbor mode never activated. The "champion" was
seed variance, not a real neighbor-mode result.

Sept 16 fix: `K_random = int(round(K_eff * ε))` (round-to-nearest,
so ε=0.20 at K=3 now gives K_random=1). AND when neighbor mode is
on with any ε > 0, force K_random ≥ 1 so the intent is always
honored. Both changes to `_topK_sim_loss_for_example`. Backward
compatibility notes:
  - Old K=3, ε=0.15 → 0 random (unchanged, rounds down)
  - Old K=3, ε=0.20 → 0 random → **new: 1 random** (round to nearest)
  - Old K=3, ε=0.34 → 1 random (unchanged)
  - K=5, ε=0.15 → old 0, new 1 (round to nearest)

To engage neighbor mode reliably: use ε ≥ 0.34 at K=3, or ε ≥ 0.20
at K=5. Or just rely on the "≥ 1 when neighbor is on" clamp.

## Open threads / suggested next work

The primary Sept 15 finding (ensemble decoder is model-agnostic)
changes the strategy. Instead of chasing val_de improvements, we
need to understand what actually moves ensemble ΔE.

### Immediate diagnostic runs (Sept 15 followup)

**P5-P7** — cheap test_evals that isolate model quality from ensemble
brute-force. All three use existing code + new sampling-entropy
diagnostics (added Sept 15). Small wall times, run in parallel.

**P5** (test_eval, ~15 min × 3): **greedy inference (TEMPERATURE=0.01,
ENSEMBLE_N=1)** on the same 3 checkpoints as P4. Does the 0.63-unit
val_de gap actually show up in inference ΔE when ensemble effects are
turned off? If yes, model quality matters at pure greedy but the
ensemble washes it out. If no, the val_de improvements are illusory.

**P6** (test_eval, ~30 min × 3): **N=5 ensemble at TEMPERATURE=1.0**.
Halfway between greedy and N=20. Complete the ΔE(N) scaling picture.

**P7** (test_eval, ~1h × 3): **N=200 at TEMPERATURE=1.0** re-runs
with the new sampling-entropy diagnostics enabled — completes the
per-checkpoint entropy profile. (The N=200 runs from Sept 14 didn't
have these diagnostics.)

Copy-paste in Common Invocations.

### Interpretation guide for P5-P7 results

- **If P5 medians spread by ~0.6 units**: model matters at greedy;
  the ensemble is the equalizer. Next: reduce ensemble reliance —
  ideas include (i) fewer replicas but higher-quality proposal
  (nucleus-p, learned temperature), (ii) train the model to be a
  BETTER PROPOSER for the ensemble rather than a better greedy
  predictor, (iii) reduce N and use the savings to sim more candidates
  per position.
- **If P5 medians spread by <0.1 units**: greedy val_de is a noisy
  proxy, model quality has never been the bottleneck. Radical
  rethink needed — maybe the physics search IS the whole product,
  and the model should be replaced with a much smaller distribution
  (fixed uniform over "likely" materials + pool-conditioned thickness
  distribution).
- **If P6/P7 entropies differ across checkpoints**: finetune is
  changing sampling diversity, which explains the tier_a/tier_b
  split (simfb wins tier_b but loses tier_a). Then: constrained
  finetune that preserves entropy might be the direction.
- **If entropies are all ~equal at ~log(pool_size)**: finetune is
  NOT reducing sampling diversity — the model is a near-uniform
  proposer regardless of training. Then: pushing entropy DOWN on
  correct picks (making it a better proposer) is the direction.

### The pending P1 run (K=3 slot + simfb + prefix-aug)

Not yet in the .out set we received. If val_de comes out ≤ 7.92,
prefix-aug on the winning recipe is confirmed. But given the primary
finding, even a val_de win won't matter for ensemble inference.
Still worth running because it's the cleanest signal on whether
prefix-aug alone helps the residual signal.

### Longer-term ideas (unchanged from Sept 14)

- **Best-of-N-in-training**: sample K candidates from the model, sim
  each, backprop to increase the probability of the best one. This
  is exactly the ensemble decoder as a training objective — should
  align train and deploy metrics.
- **EMA / weight averaging** across recent-best checkpoints.
- **Curriculum by chroma magnitude**.
- **Larger dataset** (500k-1M).
- **Batched-vmap partial-residual sim** to cut the 25% simfb overhead.
- **DAgger / scheduled sampling**: sometimes feed the model its own
  generated prefix during training, use the resulting sim residual.
  Directly closes the train↔inference residual-distribution gap.

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

### P8v2 — K=3 slot + simfb + neighbor-mode ε (Sept 16, K_random fix)

The Sept 15 P8 run configured EPSILON_START=0.20 with K=3, but
floor(3 · 0.20) = 0 meant K_random=0 and neighbor mode never
activated. The reported val_de=7.89 was seed variance, not a real
result. Fixed Sept 16 by switching K_random to round-to-nearest AND
forcing K_random ≥ 1 whenever neighbor mode is on. This re-run
GUARANTEES 1 random slot per position from the neighbor pool.

Recipe below uses K=3 with ε=0.34 (unambiguously K_random=1 even
without the new clamp), so the run is reproducible even if the
clamp is reverted.

```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb_eps34_nbr6 \
    FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=3 CE_LOSS_WEIGHT=0.1 \
    TOPK_MODE=slot LR_SCHEDULE=constant SIM_FEEDBACK=1 \
    EPSILON_START=0.34 EPSILON_END=0.34 EPSILON_DECAY_FRACTION=1.0 \
    EPSILON_NEIGHBOR_M=6 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=05:00:00 slurms/finetune_de.sh
```

Alternate at K=5 (more candidates per position, ε=0.20 → 1 random):
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot5_ce0p1_lr1e5_213k_const_simfb_eps20_nbr10 \
    FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=5 CE_LOSS_WEIGHT=0.1 \
    TOPK_MODE=slot LR_SCHEDULE=constant SIM_FEEDBACK=1 \
    EPSILON_START=0.20 EPSILON_END=0.20 EPSILON_DECAY_FRACTION=1.0 \
    EPSILON_NEIGHBOR_M=10 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=06:00:00 slurms/finetune_de.sh
```

### P9 — gamut eval on pretrain + best simfb checkpoint (Sept 15)

Runs the 28-target gamut battery (sRGB corners, L/a/b sweeps,
chromatic corners) on both checkpoints. Distinct from test_eval:
gamut eval hits worst-case edge-of-gamut colors that expose the
p95+ tail directly.

```bash
# Pretrain baseline
CHECKPOINT=data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    PRESET=balanced OPTIMIZER=dog \
    OUTPUT_NAME=gamut_pretrain_balanced_dog.json \
    sbatch slurms/gamut_eval.sh

# Best simfb checkpoint — with SIM_FEEDBACK=1 to unlock the residual
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best \
    PRESET=balanced OPTIMIZER=dog SIM_FEEDBACK=1 \
    OUTPUT_NAME=gamut_simfb_balanced_dog.json \
    sbatch slurms/gamut_eval.sh
```

Bumps to `PRESET=best` (~1.5h) or `PRESET=max` (~3h) trade time for
tighter numbers per target. The `dog` optimizer is the current
production default; add `OPTIMIZER=both` to also run Adam.

### Sept 15 followup: P5-P7 (isolate model quality from ensemble)

**P5 — greedy inference on all 3 checkpoints, N=1 TEMP=0.01:**
```bash
for CKPT in \
    data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best \
    data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best
do
  CHECKPOINT=$CKPT \
      LIMIT=500 ENSEMBLE_N=1 TEMPERATURE=0.01 \
      OUTPUT_DIR=inference/outputs/test_eval_greedy \
      sbatch --time=01:30:00 slurms/test_eval.sh
done
# For the simfb checkpoint add SIM_FEEDBACK=1 on that one command:
CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best \
    LIMIT=500 ENSEMBLE_N=1 TEMPERATURE=0.01 SIM_FEEDBACK=1 \
    OUTPUT_DIR=inference/outputs/test_eval_greedy_simfb \
    sbatch --time=01:30:00 slurms/test_eval.sh
```

**P6 — N=5 tiny ensemble:**
```bash
# Same triplet with ENSEMBLE_N=5, output to test_eval_n5. Estimated ~30 min each.
CHECKPOINT=data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    LIMIT=500 ENSEMBLE_N=5 TEMPERATURE=1.0 \
    OUTPUT_DIR=inference/outputs/test_eval_n5 \
    sbatch --time=02:00:00 slurms/test_eval.sh
# (repeat for the other two checkpoints)
```

**P7 — N=200 re-runs with the new sampling entropy diagnostic:**
```bash
# Re-run N=200 so summary.json now includes the "sampling" block with
# H_slot, H_thick, frac_unique per checkpoint.
CHECKPOINT=data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    LIMIT=500 ENSEMBLE_N=200 TEMPERATURE=1.0 \
    OUTPUT_DIR=inference/outputs/test_eval_n200_v2 \
    sbatch slurms/test_eval.sh
# (repeat for the other two checkpoints, + SIM_FEEDBACK=1 on the simfb one)
```

### Earlier Sept 15 experiments (P1-P4)

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
