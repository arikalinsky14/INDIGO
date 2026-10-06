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

## ⚠️ CURRENT WORK — Oct 1, 2026 — compute-optimal scaling study

The live line of work on `main`. A 48-run IsoFLOP sweep is **finished and
analysed**; the optimizer tuning that should have preceded it is **set up but
not yet run**. Read `analyses/scaling/README.md` for the design and
`analyses/scaling/METHOD_DIFFS.md` for the audit against Porian et al. 2024.

### What is settled

| | |
|---|---|
| N* vs compute | C^0.92 on ΔE₀₀, C^0.81 on CE (Chinchilla: 0.50) |
| in effective parameters | C^0.98 and C^0.87 |
| examples per parameter | D*/N* ∝ C^-0.78 (Chinchilla: flat) |
| both metrics turn at | C = 2.75e15, best pooled ΔE 10.04 |
| usable rungs | 5 of 6 (top rung's argmin is on the boundary) |
| epoch-ceiling theory | refuted: 47 of 48 runs under one epoch |
| wall clock | tracks examples, not FLOPs (p = 0.06 for a size term) |

`scripts/fit_scaling_porian.py` regenerates all of it from
`analyses/scaling/results/isoflop_fit.json`. The estimator is a port of the
authors' released code, not a reading of the paper: Akima interpolation with
boundary rejection, a seed-noise bootstrap whose median is the observation, and
a 1/σ²-weighted power law. Switching to it took the pooled interval from
[0.53, 1.42] to [0.81, 1.00].

### What is NOT settled, and the plan

The learning rate the sweep ran on rests on three measurements, **none of which
bracketed its own optimum**. Batch size is fixed by VRAM (256, not tuned, and
note the sweep ran 256 while production may run 512). AdamW β₂ has always been
torch's 0.999; Porian's data puts it at 0.95 at batch 256.

Run the stages in `slurms/lr_grid.sh` in order. Each gates the next.

**0. `STAGE=probe`** — one cell, one LR, <1 GPU-h. Measures examples per
second. Run it ALONE, nothing else of yours queued, or it measures contention.
Two decisions ride on the number, and both are the point of the stage:

- **Is the rest affordable?** The first stage-1 submission ran at a median of
  204 ex/s where the sweep reaches 2171 on the same shards. Feed the answer
  back as `EXAMPLES_PER_SEC=` before sizing anything.
- **How many IsoFLOP curves does stage 2 tune in full, and how many are
  projected from the fitted law?** Each added rung improves the conditioning of
  the (N, M) fit and roughly doubles the cost:

  | rungs | cells | GPU-h @2171 | GPU-h @204 | cond | sd(b) | sd(c) | sizes extrap. |
  |---|---|---|---|---|---|---|---|
  | 2 | 12 | 31.3 | 245.3 | 630 | 0.100 | 0.054 | 13 of 24 |
  | **3** | **18** | **56.7** | **471.3** | **401** | **0.050** | **0.031** | **9 of 24** |
  | 4 | 24 | 103.7 | 927.6 | 303 | 0.032 | 0.021 | 5 of 24 |

  sd(b), sd(c) are the spreads of the recovered N and M exponents over 2000
  synthetic draws at 0.10 of noise in log-lr units; the last column counts the
  sweep's distinct model sizes falling above the largest size tuned, so the
  fourth curve buys coverage as well as conditioning. At 2171 it tightens the N
  exponent 1.6x and halves the extrapolated sizes for twice the compute, a
  judgement call worth making on the measured rate; at 204 three curves already
  costs 471 GPU-h and four is out of reach. Pass the answer as `RUNGS=`.

**1. `STAGE=1`** — β₂ ∈ {0.95, 0.99, 0.999} at both ends of the ladder, three
LRs each at prior/4, prior, prior×4 (stage 1 sets `LR_SPAN=4`; the default 30
would put two of three points so far off the optimum that they measure
divergence), 6 cells, ~15 GPU-h at 2171 ex/s. Three rates rather than one because
β₂ and the LR interact; three rather than seven because the question is whether
the β₂ *ranking* is stable, not where the LR optimum is. If the two ends
disagree, β₂ interacts with scale and the sequential staging below does not
hold: stop and reconsider rather than carrying a wrong constant forward.

**Stage 1 result (job 4126638, Oct 2).** Median ΔE₀₀ on the 2,048-example slice,
one seed (42), identical init, data order and eval slice across β₂:

| | β₂ | prior/4 | prior | 4×prior |
|---|---|---|---|---|
| d40/se1, N = 100k, D = 3.45M, prior 2.28e-3 | 0.95 | 14.65 | 12.98 | diverged |
| | **0.99** | **13.99** | **12.40** | diverged |
| | 0.999 | 15.46 | 12.89 | diverged |
| d288/se5, N = 6.6M, D = 6.14M, prior 2.12e-4 | 0.95 | 12.32 | **10.33** | diverged |
| | **0.99** | **11.47** | 10.67 | diverged |
| | 0.999 | 11.72 | 11.61 | diverged |

- **0.999, torch's default and everything INDIGO has run, is never best** and
  trails by 1.3 ΔE at the large end's prior. **Retracted Oct 4**: runs are not
  reproducible to that precision. The same configuration rerun in the β₂ study
  (d40, β₂ 0.999, seed 42, prior LR) came in 1.19 ΔE better than in stage 1.
- **The ends disagree on the winner, narrowly.** Best over LR: 0.99 at the small
  end (0.59 ahead of 0.95), 0.95 at the large end (0.33 ahead of 0.99). Both
  gaps are at or under the 0.52 seed σ, though the comparison is paired (same
  init, data and eval slice), so its own noise is smaller than that.
- **0.99 minimises the worst case**: its largest loss to the per-end winner is
  0.33 ΔE, against 0.59 for 0.95 and 1.28 for 0.999. Stage 2's models
  (81k to 3.4M) sit nearer the small end, where 0.99 won at both rates.
- **The old LR law is close at both ends**: the prior beat prior/4 in all six
  cells and 4×prior diverged in all six. Stage 2's grid is set from this:
  prior/8 to 5×prior (`LR_SPAN_DOWN=8`, `LR_SPAN_UP=5`), 1.85x steps.
- **Throughput with two cells sharing a node**: 3,150 to 3,420 ex/s
  (harmonic means), above the probe's solo 2,033. Stage 2 at three curves
  prices at 43 GPU-h at 3,150.
- Stage 1 results belong in `outputs/lr_search/stage1/`, not in the directory
  `fit_lr_law.py` reads: three rates cannot bracket, and the fit collapses β₂.

**`STAGE=beta2`: the defensible β₂ study, run before stage 2.** Stage 1
cannot pick a β₂: one seed, and only two stable LRs per cell, so no β₂ had a
bracketed LR optimum. Run through `scripts/fit_beta2.py`'s own decision rule,
its pooled 0.95-vs-0.99 interval is ±5.8 ΔE. The study:

| | |
|---|---|
| β₂ | 0.9, 0.95, 0.98, 0.99, 0.999 (even in log(1−β₂); brackets 0.95 to 0.99 on both sides) |
| sizes | N* of the smallest, middle and largest usable rung (100k, 1.0M, 6.6M), at 1.1×D* |
| seeds | 42, 43, 44 (each changes init, data order and validation slice) |
| LRs | 7 per cell, √2 steps, prior/2.83 to 2.83×prior |
| cost | 45 cells, ~156 GPU-h at 3,150 ex/s; `--array=0-14` is one full seed, ~52 |

`scripts/fit_beta2.py` compares each β₂ at its own Akima-interpolated LR
optimum, paired within (size, seed) blocks; runs a two-way ANOVA for the β₂
effect and the size × β₂ interaction; reports per-size and pooled paired 95%
intervals against the winner; fits a quadratic in log10(1−β₂) for a continuous
β₂* with a parametric-bootstrap interval; and bootstraps the 2,048 examples
paired (per-example ΔE is now saved) for the evaluation noise alone. **The
decision rule is fixed in its docstring before any data**: an interaction
p < 0.05 means per-size β₂; otherwise the pooled winner, with every β₂ whose
interval includes zero reported as indistinguishable; a winner at the end of
the β₂ grid is not bracketed. `tests/test_fit_beta2.py` checks it recovers a
planted optimum and reports no effect when none is planted. **Calibrated on
200 simulated studies** of this exact design (seed σ 0.35, a planted optimum at
0.97): the pooled β₂* 95% interval covered 0.97 in 189/200 (94.5%), a false
size × β₂ interaction appeared in 9/200 (4.5%), and a β₂ effect with none
planted in 8/200 (4.0%). `results/beta2_method_check_synthetic.png` is the
figure on one such synthetic study; `results/beta2_pilot_stage1.png` is stage 1
run through the same analysis, labelled as the pilot it is.

**Run-to-run noise is large, and the calibration above understated it.** GPU
nondeterminism sends identical starts down different paths: the same seed,
size, β₂, data and LR, trained in stage 1 and again here, differed by up to
1.19 ΔE (sd of a single run ~0.36 from four such pairs), and scatter around a
smooth fit of each cell's LR curve is ~0.96. Pairing by seed still removes the
shared start and evaluation slice, but not the trajectory. Recalibrated at
per-run noise 0.6 and 0.9, 100 studies per condition:

| estimator | coverage of β₂* | false β₂ effect | winner under no effect | power, weak effect |
|---|---|---|---|---|
| **Akima** (pre-registered) | 93 to 95% | 3 to 5% | uniform, 16 to 22% each | 0.87 at noise 0.9 |
| quadratic in log LR | 91 to 96% | 7% | **28 to 31% to 0.9** | 0.99 |

The quadratic extrapolates optimistically when a β₂'s LR optimum sits near the
grid edge, which favours 0.9 even with no effect, so **Akima stays the primary
estimator**; `--estimator quadratic` is a robustness check. The choice was made
on this simulation, not on which β₂ either favours.

**Open LR edges get closed, `STAGE=beta2x`.** A cell whose LR optimum ran off
an edge that did not diverge has only an upper-bound tuned ΔE, which tilts the
comparison against that β₂. `fit_beta2.py` lists them (`extend: up/down`), and
`STAGE=beta2x` trains two more rates past the edge (4× and 5.66× the prior, or
the mirror below) into `outputs/lr_search/beta2_ext/`, which the analysis
merges into the same cells. In the interim data (23 of 45 cells, Oct 4) five
cells were open, all for the leading β₂ values; about 7 GPU-h to close.

**Result, 54 cells (Oct 6): 0.9995 added, open edges closed, 0.999 stands.**
`results/beta2_fit_oct6_54cells.json`. The 45-cell grid plus `BETA2_VALUES=0.9995`
(9 cells) and the `STAGE=beta2x` extensions merged in.

| | |
|---|---|
| β₂ effect | F(5, 30) = 4.74, **p = 0.0026** |
| size × β₂ | p = 0.21: one β₂ for all sizes |
| winner | **0.999**, now bracketed on both sides |
| vs 0.9995 | +0.01 ΔE [−0.21, +0.24], p = 0.90: a tie |
| vs 0.99, 0.95 | +0.13, +0.27: indistinguishable |
| vs 0.98, 0.9 | +0.52 (p = 0.021), +0.65 (p = 0.015); Holm 0.086, 0.077 |
| continuous β₂* | pooled: no interior minimum (40% of draws interior, CI [0.9968, 0.9995]); low chroma 0.9988 [0.9969, 0.9994], high chroma 0.9982 [0.9958, 0.9994] |
| per size | 0.999 at 100k, 0.99 at 1M, 0.95 at 6.6M, each within noise |

Reading: a broad plateau from 0.99 to 0.9995, with the bottom near 0.998 to
0.9995; below 0.99 ΔE rises. **Stage 2 runs at β₂ = 0.999.** Six LR edges are
still open, all on 0.9995 cells (up: d40 s42/s43, d120 s43/s44; down: d288
s43/s44). Closing them can only lower 0.9995's tuned ΔE, which could move the
nominal winner from 0.999 to 0.9995 but not off the plateau, so it is optional
(`STAGE=beta2x`, ~8 GPU-h). Panel E also shows the LR optimum at ~0.5 to 0.7×
the prior at 6.6M and 1.2 to 2× at 100k for β₂ ≥ 0.99: the old LR law is too
shallow in N, which is what stage 2 refits.

**Result, all 45 cells (Oct 5), under the pre-registered rule.** Superseded by the 54-cell result above.
`results/beta2_fit_oct5_45cells.json`.

| | |
|---|---|
| β₂ effect | F(4, 24) = 4.6, **p = 0.007**: β₂ matters |
| size × β₂ | p = 0.14: no evidence the best β₂ moves with size, so one value |
| winner | **0.999**, pooled and in every chroma bucket (low, mid, high) |
| vs 0.99 | +0.13 ΔE, p = 0.46: indistinguishable |
| vs 0.95 | +0.27 ΔE, p = 0.32: indistinguishable |
| vs 0.98 | +0.52 ΔE, p = 0.021 (Holm over 4: 0.064) |
| vs 0.9 | +0.65 ΔE, p = 0.016 (Holm over 4: 0.065) |
| per size | 0.999 at 100k, 0.99 at 1M, 0.95 at 6.6M (each within noise of 0.999) |
| continuous β₂* | no interior minimum: the trend still falls toward higher β₂ |

Stage 1's suggestion that 0.95 or 0.99 beat 0.999 was noise; the replicated,
LR-tuned study reverses it. **Two things stand between this and a claim of
optimality, both named by the rule:** 0.999 is the end of the grid, so it is
not bracketed (`BETA2_VALUES=0.9995` adds 9 cells, ~31 GPU-h), and 16 cells
did not bracket their LR optimum, 11 of them on an open edge
(`STAGE=beta2x`, ~16 GPU-h). Until then the defensible statement is: lowering
β₂ from 0.999 does not help, and 0.9 and 0.98 are worse at nominal 95%.
0.999 is also what the first sweep ran, so carrying it into stage 2 leaves
the learning rate as the only change between the two IsoFLOP figures.

**Interim, 23 of 45 cells (Oct 4), superseded by the above.** No β₂ is distinguishable from the
leader; Akima leads with 0.99, the quadratic with 0.999, as expected when
nothing is resolved. Three complete blocks; no size yet has the two seeds the
ANOVA needs.

**Analysis runs through SLURM**, since python cannot run on the login nodes.
`slurms/analyze.sh` runs any analysis script on smp and zips everything it
wrote (plus `INCLUDE=<dir>`) into `job-outputs/analysis_<job>.zip`:

```bash
INCLUDE=outputs/lr_search/beta2/cross_attn sbatch slurms/analyze.sh scripts/fit_beta2.py
# -> analyses/scaling/results/beta2_fit.json, beta2.png, beta2.pdf
STAGE=beta2x bash slurms/lr_grid.sh --list     # then sbatch it, then re-run the above
```

**The validation set is now read once per cell.** `lr_tuning.py` used to build
each trial's validation loader on the streaming dataset, so every trial re-read
10,000 shard-aligned rows spread over ~3,600 shards, near half the corpus.
Two small d40 cells (β₂ 0.9 and 0.95, seed 43) hit an 8-hour wall about 100
steps from the end of their seventh rate. The ΔE slice and the token-weighted
CE are unchanged, so cells already finished stay valid and paired. Each trial
now logs `[time] train / val / dE`.

**2. `STAGE=2`** — **the LR search runs inside the IsoFLOP test.** Every model
on the `RUNGS` lowest curves is tuned directly; at the default 3 that is 18
cells, ~57 GPU-h at 2171 ex/s.
Not one representative point per rung: the whole curve, because the curve is
what the parabola is fitted through, and a point whose LR was extrapolated
moves the minimum as surely as one trained wrong.

Each point carries its own (N, M), and that is what makes the law identifiable
without a separate experiment. Within ONE rung C is fixed, so M = C/(kN²) and
log M = const − 2 log N: the columns are collinear (corr −0.9998) and only the
combination b − 2c is recoverable. A second rung shifts the intercept and
separates them. At three rungs the design matrix has condition number 401, and
on synthetic data with 0.10 of noise in log-lr units it recovers both exponents
to ±0.05 and ±0.03. Two rungs would do at a pinch (cond 630, ±0.10 and ±0.05).
A fourth is the live question the probe decides, per the table in stage 0: it
is a genuine improvement rather than a rounding one, and it costs twice.

So the multiplier axis comes free from the IsoFLOP geometry. It needs no
fractional-scoring trick and no constant learning rate to get it, which means
the law is measured under the schedule the sweep actually trains with. This is
the one place we knowingly diverge from Porian et al., and it is forced: they
tune at a constant multiplier (20.0 to 21.1 while parameters vary 42x), so a
law in N alone is the right object for them. Ours cannot be, because D\*/N\*
runs 29.3 to 0.86 across our budgets.

Then fit the law:

```bash
python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
    --coverage-from analyses/scaling/results/isoflop_fit.json
```

It writes `analyses/scaling/results/lr_law_fit.json`, lr(N, D) = a N^b D^c,
which `configs.py` picks up automatically; until it exists everything falls
back to the old three-point law and the dry run says which is in force. Any
cell whose optimum lands on a grid endpoint is **discarded**, so widen
`LR_SPAN` for those and re-run them. The coverage report lists every sweep size
and whether the law interpolates or extrapolates to it: 9 of 24 are
extrapolated at three tuned curves, 5 at four, and zero is not reachable at any
affordable number of rungs.

**`STAGE=check`**: the extrapolation check, 1 cell, ~6 GPU-h, run BEFORE
stage 3 spends anything. Tune the compute-optimal point of the highest usable
rung in full and compare what the law predicted against what that point wanted.
It writes to `outputs/lr_search/check/`, so the law is never fitted on the point
that tests it. If the ratio is far from 1, the law does not reach: tune one more
rung directly (`RUNGS` + 1) rather than spend stage 3 on rates that are guesses.

**`STAGE=3`**: finish the IsoFLOP, 30 cells, ~61 GPU-h at 2171 ex/s. Every
(N, D) point of the remaining curves, repeat seeds included, trained once at the
learning rate the law gives for that point's own N and D. **The learning rate is
the only extrapolated quantity**: β₂ is stage 1's winner and the grid is the
first sweep's, point for point. It also runs the lower rungs' repeat seeds, at
the rate stage 2 picked for their seed-42 twin, so every seed cluster sits at
one learning rate. The default includes the 2.5e16 rung (8 cells, ~28 GPU-h).
`--list` prices it before stage 2 exists; it refuses to run until the law does.

**Stages 2 and 3 together are the final IsoFLOP figure.** Stage 2's winning
trial at each lower-rung point is that point's run; nothing is re-run.

```bash
python scripts/collect_isoflop.py --beta2 <winner> --rungs 3
python scripts/fit_scaling_porian.py \
    --fit analyses/scaling/results/isoflop_tuned.json \
    --output analyses/scaling/results/porian_fit_tuned.json
python analyses/scaling/plot_porian.py \
    --fit analyses/scaling/results/isoflop_tuned.json \
    --output analyses/scaling/results/porian_pooled_tuned.png
```

The collector writes the same schema as `isoflop_fit.json`, so the estimator
and plots are reused unchanged; fed the first sweep's own values it reproduces
that sweep's exponents exactly. It refuses any point not run like the rest
(shard alignment, β₂, epochs, seed) unless `--allow-mixed`. One asymmetry is
recorded, not hidden: a stage-2 point is the best of seven noisy ΔE readings
and a stage-3 point is one reading, so the lower rungs sit slightly optimistic.
That shifts whole rungs, not points within one, so N*(C) is unaffected; each run
carries its `stage` for cross-boundary comparisons.

**Every stage now passes `--limit-shard-aligned`**, as the sweep always did.
Tuning cells did not: a 3.4M-example subset scattered over all 8,000 shards made
each trial stream the whole 40M-row corpus, an I/O amplification of about
40M/D. That predicts 11.6x at the first stage-1 submission's D against 10.6x
observed (204 vs 2171 ex/s). **Confirmed by the probe, Oct 1**: the same cell
(d120/se4, D = 614,400) ran at ~340 ex/s and timed out without shard alignment
(job 4125615), and at a median of 2156, harmonic mean 2033, with it (job
4125957). Everything is now sized at 2033, so the tables' 2171 column is the one
that applies, give or take 7%. The probe trains at the prior LR and writes to
`outputs/lr_search/probe/`, outside what the LR fit reads.

### Hard-won lessons, do not relearn these

- **At D = 614,400 nothing learns.** Accuracy sits at the EOS base rate
  (1/5.5 = 0.182) and ΔE is scatter from 25 to 37. The sweep's own run at
  N = 81k needed D = 4.2M to reach ΔE 12.4. Tune at D*, not at the historical
  614,400.
- **Tuning cells ran ~10x slower than sweep runs.** Most likely cause: no
  `--limit-shard-aligned`, so every trial streamed the whole corpus (now fixed).
  Contention was the earlier guess. With the validation set also read once per
  cell (Oct 4), six cells run side by side fine: arrays default to `%6`
  (`THROTTLE=` to change what `--list` suggests). Raise or lower a running
  array with `scontrol -M gpu update JobId=<id> ArrayTaskThrottle=<n>`.
- **Any run that goes on an IsoFLOP passes `--limit-shard-aligned`.** It decides
  which shards the training subset and the ΔE slice are drawn from; points drawn
  differently do not belong on one curve.
- **A 3-point LR grid can never bracket**: its only interior point is the
  second and second-to-last at once. Use ≥5, centred per cell on the prior.
- **The analysis stack must import without torch.** `tests/` enforces it.
  `flops.ArchSpec` is the stand-in for `ModelConfig`; never import `src.model`
  in an analysis script or a SLURM heredoc.
- **Every SLURM script uses the cluster's own two lines**, `module load
  python/pytorch_251_311_cu124` then `source "$HOME/envs/llm-env/bin/activate"`.
  Never invent an activation, never hide its failure behind `|| true`.
- **C is not 6ND here.** `src/scaling/flops.py` computes it analytically;
  C/(N·D) falls from 306 to 218 across the ladder.

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
