# CLAUDE.md — INDIGO project notes for future Claude sessions

Repo: **INDIGO** (Pitt CRC). Flexible-material RGB/Lab → thin-film-stack
generative model. Autoregressive decoder predicts (slot, thickness) per
layer; the material pool varies per example (encoder is pool-agnostic).

## Current state (Sept 14, 2026)

**Numbers (val_loss_de = mean ΔE₀₀ at greedy pick on 500 val examples):**

| Model | val_loss_de | Notes |
|---|---|---|
| Pretrain (`prod_3ep_bs512_lr6e-5/step_13000`) | ~8.55 | 3 epochs CE, val-optimal step |
| Best finetune, no sim-feedback | 8.02 | K=3 slot, LR=1e-5, CE=0.1, const LR, unfrozen encoder |
| **Best finetune, WITH sim-feedback** | **7.92** | Same recipe + `SIM_FEEDBACK=1`; step 1250 of 1665 |

**Inference-time (ensemble decode + real-sim select) on test set:**
Median ΔE ≈ **0.65**, p95 ≈ **3.3** — for tier_a partial (350/500).
Full 500×2-tier eval is pending resubmit (see Open Threads below).

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

**Only checkpoints finetuned with `--sim-feedback` have non-zero
residual weights.** Running inference with `SIM_FEEDBACK=1` on a
pretrain-only checkpoint is a no-op (zero-init residual_proj).

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
6. **Joint mode failed** because top-K concentrates on 1-2 slots.
   Hierarchical M×N is the proper thickness-training design (not yet
   run at scale).
7. **Sim-feedback residual** finally broke through the ~8.0 plateau
   → 7.92 (Sept 13). Small but real, and beat all previous variants.

## Open threads / suggested next work

### Immediate

1. **Rerun test_eval on our two best checkpoints** (12h wall now,
   `--qos=short` removed):
   - `finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best` — pretrain
     comparison baseline (no sim_feedback needed)
   - `finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb/best` — **run
     with `SIM_FEEDBACK=1`** to unlock the residual signal at inference
     (this is the whole point of the plumbing added Sept 14)
2. **Compare summary.json** — sim-feedback should show measurable
   improvement on the p95+ high-chroma tail if the residual signal
   generalizes.

### The Big Open Question (user flagged for next session)

**Sim-feedback trajectory oscillates and peaks early — how do we
extend the advantage?** All our finetune runs share a pattern: val_de
drops sharply in the first 300-600 steps, then oscillates around a
noisy plateau for the rest. The sim-feedback run hit val_de=7.99 at
step 250 (already best) and 7.92 at step 1250 — genuinely better peak
but same oscillation pattern.

**User's own hypothesis to explore:** more simulations per example
during finetune (larger K, or hierarchical M×N) might give the
residual-conditioned model richer per-position choice sets to learn
from. The intuition: residual conditioning changes what "good pick"
looks like; if the top-K set is still narrow (K=3 slot), we're not
giving the model enough opportunity to rerank.

**Other ideas worth trying:**

- **Hierarchical M×N + sim-feedback**: pair the two additions. M=3,
  N=3 gives 9 candidates per position with distinct (slot, thickness)
  pairs; combined with residual conditioning the model gets both a
  richer choice set AND state feedback per step.
- **ε-exploration + sim-feedback**: the residual tells the model
  "you're off by X"; random exploration might surface candidates the
  model wouldn't ordinarily consider that better match the residual.
- **Different LR for residual_proj**: the base model may be
  over-training while residual_proj is under-training. Split LR groups
  in the optimizer.
- **EMA / weight averaging across recent-best checkpoints** to smooth
  the oscillation and capture "the average of the peak region."
- **Curriculum by chroma magnitude**: train easy → hard so the model
  builds representations before hitting the hard tail.
- **Larger dataset**: 213k examples might just not be enough for the
  residual signal to fully develop. Scale to 500k or 1M once we know
  the recipe works.
- **Longer training + best-checkpoint retention**: we already have
  best-checkpoint save; a longer run at winning recipe might find a
  deeper trough somewhere later than step 1250.
- **Data augmentation of prefixes**: during training, perturb GT
  prefix layers (small thickness jitter or occasional wrong-material
  swap) so the residual distribution the model sees at training is
  wider — closer to what inference will produce.

**Design constraint to keep in mind**: the residual at inference is
computed from the model's OWN partial prefix, not GT. During training
we compute it from GT prefix. Distribution shift is real. Prefix
augmentation might close that gap.

### Also-nice-to-have

- Plumb sim-feedback through `inference/src/generate.py` was done
  Sept 14; not yet exercised at scale. First test_eval with
  `SIM_FEEDBACK=1` will validate the plumbing end-to-end.
- Consider MOVING the `_compute_partial_residuals` loop to
  batched-vmap sim. Current per-example Python loop is fine but adds
  25% wall clock; a batched JAX call could nearly eliminate it.

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

**Finetune (winning recipe with sim-feedback):**
```bash
PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
    SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const_simfb \
    FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=3 CE_LOSS_WEIGHT=0.1 \
    TOPK_MODE=slot LR_SCHEDULE=constant SIM_FEEDBACK=1 \
    EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
    NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
    sbatch --time=05:00:00 slurms/finetune_de.sh
```

**Inference eval with sim-feedback:**
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
