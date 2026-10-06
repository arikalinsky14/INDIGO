# Compute-Optimal Scaling Study

IsoFLOP study that asks **what (N, D) split minimises ΔE₀₀ at a fixed
compute budget** for the INDIGO pretrain, and — the part that is not in
the literature — **whether that optimal split changes with chroma
difficulty**.

Method is Chinchilla's Approach 2 with the corrections from
[Porian et al. 2024](https://arxiv.org/abs/2406.19146) built in from the
start rather than retrofitted. This README is the hand-off document: what
the study asks, why it is set up the way it is, what the first attempt got
wrong, and where every artefact lives.

**Status, Oct 1 2026.** The 48-run IsoFLOP sweep is finished and analysed.
The optimizer tuning that should have preceded it is set up and not yet run.

| | pooled ΔE₀₀ | cross-entropy |
|---|---|---|
| N\* vs compute, α | **+0.923** [0.814, 0.999] | +0.813 [0.730, 0.878] |
| same, in effective parameters | +0.983 [0.866, 1.063] | +0.866 [0.777, 0.934] |
| examples per parameter, D\*/N\* | ∝ C^**−0.783** | ∝ C^−0.572 |
| usable rungs, of 6 | 5 | 4 |

Chinchilla is α = 0.50 with a flat multiplier. Both metrics turn upward at
C = 2.75e15, where the best pooled ΔE is 10.04. Full numbers in *Results of
the finished sweep* below; the audit against the authors' own code is
`METHOD_DIFFS.md`; the live plan is *What is not settled* below and
`CLAUDE.md`.

---

## What we are trying to answer

1. **For a fixed compute budget C, what model size N\* minimises ΔE?**
   Interpolate (log N, val_de) at each budget, read off the argmin, then fit
   a power law N\*(C) ∝ C^α across budgets. Same for D\*(C) ∝ C^β. The
   interpolant is an Akima spline, not a parabola, for the reason in
   `METHOD_DIFFS.md` item 1; `fit_scaling.py` keeps the parabola and
   `fit_scaling_porian.py` is the one to quote.

2. **Is the production checkpoint the right size for its budget?**
   Early indications are no, by a large factor. Sweep v1's best run reached
   greedy val_de 9.56 with **0.97M parameters**, within ~1 ΔE of production
   at 18–72× the parameters and ~19–70× the compute. If α confirms that,
   the practical finding is that INDIGO has been over-parameterised and the
   same quality is reachable far cheaper.

3. **Does α differ across chroma buckets?** If saturated colours have a
   different compute-optimal split than near-greys, the frontier is
   chroma-dependent and a single scaling law is the wrong object. This is
   the novel contribution, and it is also the most fragile result — see
   *Statistical power* below.

Prior evidence that (3) is worth asking rather than a fishing expedition:
on the production run, high-chroma targets sit ~45% worse than random ones
and their p75 tail degrades over training while the median stays flat.

---

## Why ΔE₀₀ and never cross-entropy

**CE and ΔE are decoupled on INDIGO.** This is verified, not assumed, and
visible inside a single production run: its CE-optimal and ΔE-optimal
checkpoints are different steps. Fitting CE would answer a different
question than the one being asked, so `fit_scaling.py` reads `val_de` and
treats `val_loss` purely as a recorded diagnostic. There is deliberately
no option to fit on CE.

CE still earns its place in exactly one role: **divergence screening.**
ΔE cannot detect divergence — during LR tuning a diverged d512 model
(val_loss 4570) and a healthy d128 model reported *identical* val_de of
28.6454, because a model emitting garbage and a model emitting nothing
both score the same distance from the target. So any run whose val_loss
exceeds `2·log(VOCAB_SIZE) = 16.14` is excluded before fitting.

The metric reported is the **median**, not the mean: the ΔE distribution
has a heavy high-chroma tail and the mean tracks the tail rather than
typical performance.

---

## The three Porian corrections

| # | Correction | Where it lives |
|---|---|---|
| 1 | Last-layer FLOP accounting, including embedding and output head | `src/scaling/flops.py` |
| 2 | Warmup as a fraction of total compute, not fixed steps | `scripts/training.py` (`--warmup-fraction`) |
| 3 | Per-scale LR re-tuning | `scripts/lr_tuning.py`, fitted law in `src/scaling/configs.py` |

Correction 1 is not cosmetic here. The usual `C = 6ND` shortcut is wrong
for this architecture by a factor of **35–50×**: measured train FLOPs per
(parameter, example) run **212–306**, and at the production config 226.5.
The reason is that the head and embeddings dominate at small width — they
are **71% of forward FLOPs at d_model=64**, falling to 18% at d_model=1024
— so an N-sweep that ignores them mis-assigns compute in a way that varies
systematically along the very axis being swept. Every budget in this study
is computed analytically from `flops.py`, whose parameter counts are
bit-identical to `build_model` across seven configs.

Correction 2 turned out to already be satisfied; `warmup_steps_for_flop_fraction()`
documents that.

---

## Why the grid looks the way it does (and what v1 got wrong)

**Sweep v1 (Sept 22) completed all 20 configs and could not be fitted.**
Three of four pooled parabolas came back with a ≤ 0, and no chroma bucket
had the three usable rungs a power law needs. The fitter was right to
refuse; the grid was measuring the wrong thing.

### The aspect-ratio confound

v1 quantised `d_model` to multiples of 64, which makes the smallest width
step a 3.3× jump in parameters. Depth was therefore the only knob fine
enough to hit a target N, so within a budget *larger N* nearly always meant
*deeper at the same width*. At C=1e14, four of five points were d_model=64
at depths 2, 3, 4 and 7 — an aspect ratio sweeping from 32 down to 9.1.

So the sweep measured how badly a narrow-deep model trains, not capacity.
Within-budget Spearman ρ between log N and val_de:

| budget | ρ(log N, val_de) | reading |
|---|---|---|
| 1.0e14 | **+0.70** | bigger is monotonically worse |
| 3.7e14 | **+0.50** | same |
| 1.4e15 | **+0.20** | same, weaker |

A monotone or concave series has no interior minimum, which is precisely
what the refusals reported. Corroborating it: the per-budget *winners* sat
at aspect 32, 64, 42.7 and 16, while the *losers* were the extremes, 9.1
and 320.

### The fix

`achievable_sizes()` now admits only aspect ratios in **[28, 72]** and,
where several shapes share an `n_params`, keeps the one nearest **48**. So
moving along the ladder moves capacity, not geometry. `HEAD_DIM` drops from
64 to 32, which halves the width quantum so depth no longer has to absorb
the residual; `n_heads` is exactly parameter- and FLOP-neutral, so this
cannot perturb the compute axis. It does differ from production
(head_dim 64), and internal consistency across the ladder is the right
trade there, since every point is compared to other points in the ladder.

`DEFAULT_SPAN` also went from **2.8× to 10×** in N. 2.8× cannot show
curvature; Chinchilla's own IsoFLOP slices span more than an order of
magnitude.

### Verification that the fix works

Replaying a *known* α through the new grid recovers it exactly at zero
noise and holds all four chroma buckets at 0.25–0.5 ΔE of jitter, where
the old geometry lost the low bucket to extrapolation outright and could
not separate the rest. `fit_scaling.py` itself was separately validated
against synthetic ground truth: pooled α recovered +0.5046 (true 0.500),
mid +0.4996 (0.500), low +0.4316 (0.420).

---

## The wall-clock floor, and why QoS is the binding constraint

This is the least obvious structural fact in the study, and it sets the
ceiling on what can be measured.

**At fixed C, a smaller model needs *more* passes** (D = C / FLOPs per
example). So the wall clock sets a **floor** on N, not a ceiling. That
floor grows like C, while N\* grows only like √C — so they must cross, and
beyond the crossing every size that fits the time limit already sits above
the optimum and the rung goes one-sided.

| QoS | top usable budget | lever arm |
|---|---|---|
| 3h (`short`) | 4e15 | 1.6 decades |
| 12h | 2.5e16 | 2.4 decades |
| 24h | 1.2e17 | 3.1 decades (production scale) |

The same arithmetic has a sharper consequence for question (2) above:
**under `qos=short` a 0.97M-parameter model tops out at 8.8M passes, which
is less than one epoch of the 10M-example corpus.** The regime that
produced the surprising v1 result — small model, lots of data — is exactly
the regime the 3h cap forbids. Hence the separate fixed-N data ladder.

---

## The fixed-N data ladder

`build_data_ladder()` holds one shape (d128/se3, 0.97M params — the exact
v1 standout) and sweeps D log-spaced. **This is not an IsoFLOP rung and
must not be fitted as one**: every point is its own budget, so there is no
parabola in log N to fit. It answers a different question — does a ~1M
model saturate, or keep improving with data?

It deliberately inverts `build_grid`'s epoch arithmetic. `build_grid`
holds steps exact and raises epochs until one divides, which keeps C on
its rung to the example. Here that rule defeats the experiment: at 42.5M
passes it lands on 8 epochs over a 5.3M subset when the whole question is
what *more data* does. Holding epochs at the minimum and rounding steps
instead gives 5 passes over 8.5M (85% of the corpus, against 53%), for a
sub-0.5% shift in D that no conclusion here turns on.

| wall cap | data span | deepest |
|---|---|---|
| 3h | 9× | 0.88 epochs |
| 12h | 43× | 4.26 epochs |

4.26 epochs is inside the 8 the epoch-ceiling probe examined.

---

## Calibration: the wall model

`estimate_wall_sec()` is **refit on real elapsed times**, not estimated:

```
elapsed_sec = 1712 + passes / 1985 + ΔE eval
```

from least squares over sweep v1's 20 completed runs. The original
constants (5224 ex/s from a d512 step-timing probe, 480 s startup) were
optimistic end to end and v1 ran up to **2.05×** its predicted wall. Two
things the step-timing figure missed: fixed startup is nearly half an hour
once module load, torch and JAX imports, the shard scan, the validation
read and the simulator preflight are counted; and these small models are
input-bound rather than GPU-bound, so they run *slower* per example than
the much larger probe did.

The rate used is the **p10 across configs (1301 ex/s), not the median**,
because measured throughput ranged 1162–2820 ex/s under node contention
and a grid sized on the median would put its slowest configs over the
wall. At p10, v1's worst run would have come in at 1.08× its prediction.
`WALL_MARGIN` is 0.80 on top of that.

A timed-out config is worse than one never attempted: it silently removes
a point from its parabola and biases the fitted minimum, and the
epoch-boundary-only resume cannot recover it because most configs are a
single epoch.

`wall_model_residuals()` exists so a finished sweep's elapsed times can be
fed back and the constants re-checked before the next one is sized. v1's
model was wrong by 2× and nothing caught it until the runs were in.

**That feedback has now been done**, on 90 checkpoint intervals from the
finished sweep: the measured rate is **2171 ex/s**, so 1301 is conservative by
about 1.7x, and the model needs no size term at all. See *Wall clock tracks
examples, not FLOPs* below. The 1301 constant is left in place because it is
conservative in the safe direction; a timed-out config costs a point on a
parabola, an early finish costs nothing.

**Tuning cells are a different regime and the sweep's rate does not transfer.**
Six concurrent `lr_grid.sh` cells on the same shards measured a median of 204
ex/s, 6x under the planner, and every cell hit its wall. Size tuning arrays
from a measured rate and throttle them (`%6` since the validation-read fix).

**Throttle concurrency with `%N`.** v1 ran all 20 tasks at once against the
same parquet shards on shared `/ix1` and measured 246–2856 ex/s. These jobs
compete for reads, so throttling should make configs finish early rather
than late.

---

## Interpreting the output

`fit_scaling.py` refuses rather than reporting a plausible-looking number,
because every degenerate case here yields a finite exponent and a wrong α
is the entire deliverable:

- **Parabola with a ≤ 0 in log N** has no interior minimum. Reporting
  −b/2a anyway yields a meaningless N\*. Refused.
- **N\* outside the sampled range** is extrapolation from a fit that never
  bracketed its own optimum. Reported, and excluded from the power law
  unless `--allow-extrapolated`.
- **Fewer than 3 sizes per budget** cannot determine a parabola; **fewer
  than 3 budgets** makes α an artifact of two points. Both refused.
- **α + β ≈ 1 by construction**, since C ∼ N·D. This is an arithmetic
  check on the budget bookkeeping, *not* an empirical result. A large
  deviation means something is wrong upstream.

α is reported with a bootstrap CI over budgets. At four budgets a point
estimate alone would be over-claimed.

### A metric that looks broken and is not

`val_acc` sits at **0.173–0.176 in every run** regardless of model size or
compute, which reads as "nothing learned" and has been misdiagnosed twice.
It is the **EOS base rate**. A token is a (slot, thickness) pair out of
3201 and one token per structure is EOS; structures average 4.5 layers, so
EOS is 1/5.5 = 0.182 of scored tokens, and a model that has learned only
where structures end already scores ~0.18. Both loss functions now return
`n_correct_non_eos` / `n_tokens_non_eos` and training logs `acc_noEOS`
alongside `val_acc`. Read that one.

### Statistical power

The chroma-conditioned exponents are the weakest link. Replaying the grid
with 0.25 ΔE of jitter and no repeats lost a bucket entirely. Two
mitigations: `--limit-de-examples` is 2048 (≈680 per bucket, up from v1's
512/≈170), and `--repeat-seed` re-runs each rung's middle size under a
second init. Those pairs are the only error bar on the fit — v1 had no
repeats, so when its top rung's spread fell to 1.04 ΔE there was no way to
call it signal or noise.

**If the chroma comparison reports "does not separate the buckets", read
that as a power limit, not as evidence the frontier is chroma-independent.**

---

## Chroma buckets

`C* = √(a*² + b*²)` via `lab_chroma` — the repo's own definition, not
|a| + |b|. Edges at **(20, 50)**, lower edge inclusive:

| bucket | C* |
|---|---|
| low | < 20 |
| mid | 20 ≤ C* < 50 |
| high | ≥ 50 |

---

## File map

**Grid and planning**
- `src/scaling/configs.py` — the grid, the measured constants, the LR law,
  the wall model, the data ladder. Single source of truth: `SweepConfig`
  carries `n_heads` and `seed` so the planner and the emitted training
  command cannot drift (they did in v1, where two call sites each
  open-coded `d_model // 64`).
- `src/scaling/flops.py` — exact parameter counts and analytic FLOPs.
  Includes `test_no_grad_undercount_regression()`, which locks down a
  `FlopCounterMode` blind spot that silently reports zero FLOPs for
  `slot_encoder` under eval + `no_grad` together.
- `scripts/scaling_sweep.py` — dispatcher. `--dry-run`, `--n-configs`,
  `--emit-config N`, `--data-ladder`.

**Evaluation and fitting**
- `src/delta_e_eval.py` — the shared ΔE evaluator, matched to
  `scripts/evaluate.py`'s loop to 0.000e+00.
- `src/color_utils.py` — `ciede2000`, `lab_diff_ciede2000`.
- `scripts/fit_scaling.py` — the two-step IsoFLOP fit, pooled and
  per-chroma.

**SLURM** (all wrappers in `slurms/`)
- `scaling_sweep.sh` — job array, one task per config. Also runs the data
  ladder via `DATA_LADDER=1`.
- `scaling_fit.sh` — `MODE=dry-run` to inspect the grid, `MODE=fit` to fit.
  CPU only.

**Upstream groundwork**
- `scripts/epoch_ceiling_probe.py` — measured how many passes over the
  corpus are safe. Verdict: **no detectable degradation up to 8 epochs** at
  all three sizes tested, largest change +1.190 against a paired seed-noise
  floor of 5.125 ΔE. That is explicitly a **non-detection with limited
  power**, not a verified safe depth.
- `scripts/lr_tuning.py` — per-scale LR, selected on ΔE with the CE
  divergence screen. Also the worker for every tuning stage: it exposes
  `--beta1/--beta2`, `--lr-schedule`, `--eval-fractions`, and it checks for an
  existing result file **before** loading data or touching the GPU, so
  re-submitting an array is free.

**The Porian-faithful stack (added Oct 1)**
- `src/scaling/porian.py`: the port of the authors' released analysis code.
  `akima_argmin`, `fit_rung` with the seed-noise bootstrap, `power_law_fit` and
  `bootstrap_power_law`, `saturating_fit`, and `nested_hparam_optimum` with the
  `on_edge` bracketing test.
- `scripts/fit_scaling_porian.py`: the CLI, writes `results/porian_fit.json`.
  **This is the one to quote**; `fit_scaling.py` keeps the parabola for
  continuity with wave 1.
- `analyses/scaling/plot_porian.py`: their three panels (IsoFLOP curves, N\*(C),
  multiplier). `--x-axis {flops,credits}`; the credit axis prints plain numbers
  on its log ticks, and `--x-pad-left` widens the low-compute end so the
  bottom curve shows its left descent.
- `scripts/lr_grid_cells.py` + `slurms/lr_grid.sh`: the staged tuning. Cells
  are derived from the sweep's own grid, never written by hand.
- `scripts/fit_lr_law.py`: the 2-D law in (N, M), with the coverage report
  that names every sweep size the law interpolates to and every one it reaches
  past.
- `analyses/scaling/plot_lr_sweep.py`: per-cell LR curves, starred green when
  the optimum is bracketed and red when it is on an endpoint.
- `scripts/fit_wall_model.py` + `slurms/fit_wall_model.sh`: seconds per
  example against forward cost per example, with the permutation test that
  decided there is no size term.
- `METHOD_DIFFS.md`: the line-by-line audit against the authors' released
  code. Read it before changing the estimator.

**Torch-free requirement.** The whole analysis stack must import without
torch, so it can run on an SMP node or a login shell. `flops.ArchSpec` is the
duck-type stand-in for `ModelConfig`, and `tests/` enforces the rule. Never
import `src.model` from an analysis script or a SLURM heredoc; this broke three
times in one session.

---

## Running it

The grid is sized against the wall cap rather than hoping to fit it, so
inspect before submitting:

```bash
MODE=dry-run sbatch slurms/scaling_fit.sh
```

Rungs are **independent parabolas** and the power law simply gains a point
per rung, so they can go in waves into the same `OUT_ROOT`, re-fitting
after each, with nothing wasted on an early stop. The three cheapest rungs
fit `qos=short` and give one decade with no long-QoS exposure:

```bash
COMMON='DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train
        MAX_WALL_HOURS=12 OUT_ROOT=data/checkpoints/scaling_sweep_12h'
BUD='1e14 3.02e14 9.1e14 2.75e15 8.29e15 2.5e16'

env $COMMON BUDGETS="$BUD" sbatch --array=0-20%8  slurms/scaling_sweep.sh
env $COMMON BUDGETS="$BUD" sbatch --qos=long --time=06:00:00 --array=21-27%6 slurms/scaling_sweep.sh
env $COMMON BUDGETS="$BUD" sbatch --qos=long --time=09:00:00 --array=28-34%6 slurms/scaling_sweep.sh
env $COMMON BUDGETS="$BUD" sbatch --qos=long --time=12:00:00 --array=35-41%6 slurms/scaling_sweep.sh
```

### Operational traps, all of which have bitten

- **The array index is a position in the grid.** Every wave must pass
  identical `BUDGETS`/`SPAN`/`POINTS`/`REPEAT_SEED`, or the indices shift
  and a later wave trains a different config into an earlier one's save
  directory.
- **The batch script is copied at submission; the Python it calls is not.**
  `scaling_sweep.py --emit-config` runs at job start from the submit
  directory, so editing `src/scaling/configs.py` while tasks are pending
  changes what those tasks train. Freeze the grid files until the queue
  drains.
- **Save directories are derived from the config, not the job ID.** Two
  queued jobs for the same config write the same `model.pt` and both
  *append* to the same `history.jsonl`. Cancel duplicates before they run.
- **This is a multi-cluster Slurm.** `scancel` needs `-M gpu` just as
  `squeue` does; without it it targets the default cluster, fails, and
  leaves the job queued.
- **Use a fresh `OUT_ROOT` per sweep.** `fit_scaling.py` clusters runs by
  measured compute, so runs from two different sweeps in one root can
  overlap in C and be silently merged. Mixing two runs' outputs is what
  produced a bogus 21.6 ΔE floor in the epoch-ceiling analysis.
- **The data ladder needs its own `OUT_ROOT`.** Its points are not IsoFLOP
  rungs and `fit_scaling.py` must not see them.

---

## Wave-1 results (Sept 23-24) and what they changed

**Superseded on the headline numbers by the finished sweep, two sections
down.** Kept because the three corrections it forced are still in force, and
because the data-ladder and corpus-sizing findings in it are not superseded by
anything.

**The geometry fix worked.** All 12 parabolas now open upward with R^2
0.77-0.98, against v1's three-of-four refusals. N\* is inside the sampled
range at 11 of 12. Real IsoFLOP curves exist.

Pooled alpha = **+0.71, 95% interval [+0.52, +0.98]**, over three rungs and
one decade. Notably above Chinchilla's 0.5, but only marginally excluding it.

The **low-chroma bucket is the best-determined exponent in the study**:
alpha = **+0.714, [+0.564, +0.788]**, with **0% of noise draws refused**.
Low-chroma DeltaE is the least noisy signal, so its parabolas survive
perturbation where the others do not (mid refuses 38% of draws, high 60%).
If one number is quoted from wave 1, it should be that one, not the pooled.

Three corrections came out of reading it:

1. **The reported CI was wrong by ~150x.** `fit_power_law`'s bootstrap
   resamples the (log C, log N\*) points and propagates none of the val_de
   uncertainty each N\* was derived from, so it printed +-0.003. Measured
   seed sigma from the repeat pairs is 2.83 / 0.65 / 0.51 dE by rung, which
   gives +-0.23. Fixed: `montecarlo_exponent()` now perturbs every run's
   val_de and refits, and that interval is what the chroma test uses.

2. **The chroma-conditioned frontier is NOT established.** The job printed
   "spread EXCEEDS the widest CI", but that was the broken CI. Spread
   low-vs-mid is 0.082 against +-0.23 intervals. Exactly the power limit this
   README predicted; read it as such, not as evidence of independence.

3. **alpha is fragile to non-learning points.** Dropping the one point worse
   than random (C=1e14, d128/se4, val_de 30.86 against ~28.6 for a model
   emitting nothing usable) moves alpha 0.735 -> 0.908; dropping everything
   over 20 refuses. Those points sit at the high-N end of the lowest rung
   where a config gets too few steps to learn at all, which a parabola cannot
   distinguish from "past N\*".

### The data ladder overturned the epoch ceiling

Not saturation. **Degradation.**

| epochs | val_de | vs minimum |
|---|---|---|
| 0.45 | **10.82** | minimum |
| 0.95 | 11.43 | +0.61 (~1.2 sigma) |
| 2.01 | 12.50 | +1.68 (~3.3 sigma) |
| 4.26 | 12.99 | +2.17 (~4.3 sigma) |

Corroborated twice independently: val_loss in the same runs bottoms near one
epoch and rises after (6.053 at step 34k to 6.14 at step 160k), so it is not
a dE-only artifact; and **production itself** trained 3 epochs but its
dE-optimal checkpoint was step 13000 of ~58,600, i.e. **0.67 epochs**.

`EPOCH_CEILING` therefore went from 8.0 to **1.0**, and `build_grid` now
enforces it instead of merely flagging it. The probe's non-detection was a
power failure: a ~4.9 dE floor cannot see a 2.2 dE effect.

**This makes the corpus, not the QoS, the binding constraint.** The top
budget that can still straddle N\* collapses to **1.92e15 at any QoS** --
1.3 decades. A longer wall clock no longer buys anything, because the limit
is now how many unique examples exist. Reaching production's ~1e17 with
points below N\* would need roughly 13 epochs, so about a 10-100x larger
corpus. That is a data-generation question, not a scheduling one.

**Waves 2-4 as designed are invalid** (2.75e15, 8.29e15 and 2.5e16 all sit
above 1.92e15, with 2, 3 and 5 of 7 configs past one epoch). Do not submit
them.

### The confound is closed: it is repetition, not the schedule

DeltaE trajectories recovered from the ladder's own saved checkpoints
(`scripts/de_trajectory.py`, ~1 GPU-hour, no retraining) settle it. Three
checks, all against the cosine-death hypothesis:

1. **Both runs peak at 35-38% of their own training**, while the cosine LR is
   still high. Cosine death peaks late, in the decay tail. This is the
   opposite shape.
2. **Degradation is monotone from 2.1 to 4.2 epochs while the LR is
   decaying.** A decaying LR steadily making things worse is not the schedule
   failing.
3. **At matched pass counts, finishing with a fully decayed LR is not
   systematically better than passing through mid-flight at high LR** (gaps
   of +1.1, +0.2, -0.3, -0.0 dE on identical data, about 2 sigma either way).
   If decay were doing the work, finished would win every time.

The 5-epoch run against a 0.51 dE seed sigma:

| epochs | val_de | vs best |
|---|---|---|
| 0.56 | 11.726 | 1.8 sigma (plateau) |
| 1.08 | 11.603 | 1.6 sigma (plateau) |
| **1.59** | **10.792** | best |
| 2.10 | 12.249 | 2.9 sigma, significant |
| 3.64 | 13.000 | 4.3 sigma |
| 4.15 | 12.974 | 4.3 sigma |

So the optimum is a broad plateau from roughly 0.5 to 1.6 epochs, and
`EPOCH_CEILING` is **1.5**, between the deepest depth not significantly worse
than the best (1.59) and the first that is (2.10). The corpus really is the
binding constraint, and more data is the lever:

| corpus | generation GPU-h | top budget @12h | decades |
|---|---|---|---|
| 10M (current) | -- | 3.52e15 | 1.5 |
| 20M | ~96 | 1.52e16 | 2.2 |
| **40M** | **~192** | **2.87e16** | **2.5** |
| 100M | ~480 | 2.87e16 | 2.5 |

4x the corpus reaches production's own budget. Beyond ~40M the wall clock
takes over again and more data buys nothing, so 40M is the target.

### The 40M corpus (generated Sept 24-30)

Done. `data/train` is 8,000 shards, ids 0-7999, no gaps, 40,000,000 rows,
every shard at its expected count, train split 39,980,000. Generated at
`high_chroma_prob=0.2` throughout, verified uniform against the per-shard
sidecars before the extension began -- the top-level `run_manifest.json` is
overwritten by each run and described only shards 1800-1999, so it could not
be trusted for that.

Measured cost, which the script header had wrong by 30x (it quoted the
random-path figure): the search path is 8.106 s/row against 0.135 s/row
random, so at p=0.2 the average is 1.729 s/row = **2.40 core-hours per
5000-row shard**. Sizing `--time` from the old number killed all 30 tasks of
the first extension array at their 6h limit. No shard was corrupted by those
kills: each is ~2.4 h of computation followed by a write of seconds, so the
window for a kill to land mid-write is ~0.1%.

What it bought, and the reason it was worth 14,400 core-hours:

| rung | at the old 10M corpus | at 40M |
|---|---|---|
| 8.29e15 | 1 below / 5 above | **3 / 3** |
| 2.50e16 | 0 / 6, ONE-SIDED | **3 / 3** |

All six rungs now bracket their own optimum, where the top two previously
could not be fitted at all.

### Two other findings

- **head_dim is a confound against v1.** v1's standout, d128/se3 at val_de
  9.565, ran `--n-heads 2` (head_dim 64). The ladder's d128/se3 ran
  `--n-heads 4` (head_dim 32) and got 10.820 at comparable D. Same
  parameters and FLOPs, different model. The 9.565 has not been reproduced.
- **`acc_noEOS` is 0.001-0.005 everywhere.** These models essentially never
  get an exact (slot, thickness) pair right, yet reach val_de ~11 against
  ~28.6 for a model emitting nothing usable. Being one thickness bin off
  costs little in dE, so approximate correctness is what is being learned.
  The metric is working; it is just showing that exact-token accuracy is the
  wrong lens on this task.

---

## Results of the finished sweep (48 runs, Porian estimator)

Regenerate every number here with

```bash
python scripts/fit_scaling_porian.py          # -> results/porian_fit.json
python analyses/scaling/plot_porian.py        # -> results/porian_pooled.png
```

The estimator is a port of the authors' released code, not a reading of the
paper: Akima interpolation in log-log with boundary rejection, a seed-noise
bootstrap whose **median** is the observation, sigma inflated by
`n_boot / n_valid`, and a 1/sigma^2-weighted power law in step 2. Weighted and
unweighted exponents are both reported and agree to within 0.01 on every
bucket except low chroma.

### The headline exponents

| bucket | usable rungs | alpha (weighted) | 95% interval | r^2 | D\*/N\* exponent | alpha in N_eff |
|---|---|---|---|---|---|---|
| **pooled ΔE** | 5 of 6 | **+0.923** | [+0.814, +0.999] | 0.997 | **−0.783** | +0.983 |
| low chroma | 5 | +0.646 | [+0.336, +0.818] | 0.921 | −0.241 | +0.688 |
| mid chroma | 5 | +0.936 | [+0.818, +1.014] | 0.999 | −0.807 | +0.996 |
| high chroma | 5 | +0.823 | [+0.573, +0.955] | 0.956 | −0.586 | +0.875 |
| cross-entropy | 4 | +0.813 | [+0.730, +0.878] | 0.996 | −0.572 | +0.866 |

Chinchilla is alpha = 0.50 with a multiplier exponent of ~0. INDIGO is roughly
twice the exponent on N and strongly negative on the multiplier: **examples
per parameter falls as compute grows**, from D\*/N\* = 29.3 at C = 1e14 to 0.86
at C = 8.3e15. Those are the same fact stated twice, since alpha + beta = 1 by
construction.

The last column is the exponent in **effective parameters**,
`N_eff = forward_flops_per_example / 2`, the units in which C = 6·N_eff·D is
an identity so alpha + beta is forced to exactly 1 rather than landing at 1.06.
`N_eff ~ 98.9 N^0.940` with r^2 = 0.9999. Quote the parameter-count version as
the headline, because that is a model size and N_eff is not, and always say
which one is being quoted.

### Per-rung detail, pooled ΔE

| budget C | points | N\* (median of draws) | log sigma | valid draws | D\* | best ΔE at this budget | usable |
|---|---|---|---|---|---|---|---|
| 1.00e14 | 8 | 1.07e5 | 0.273 | 65% | 3.14e6 | 12.39 | yes |
| 3.02e14 | 8 | 3.35e5 | 0.224 | 97% | 3.22e6 | 10.84 | yes |
| 9.10e14 | 8 | 1.08e6 | 0.346 | 95% | 3.24e6 | 10.46 | yes |
| **2.75e15** | 8 | 2.56e6 | 0.155 | 100% | 4.51e6 | **10.04** | yes |
| 8.29e15 | 8 | 6.51e6 | 0.221 | 97% | 5.58e6 | 10.75 | yes |
| 2.50e16 | 8 | n/a | n/a | n/a | n/a | 11.40 | **no, argmin on the boundary** |

Every rung brackets its own optimum except the top one, whose interpolated
argmin sits on the largest model sampled. That rejection is the estimator
working as intended, and it is why alpha rests on five points rather than six.

**Both metrics turn upward at C = 2.75e15.** Pooled ΔE goes 12.39, 10.84,
10.46, **10.04**, 10.75, 11.40 across the six budgets, and CE does the same
thing two decimal places down (6.144, 6.075, 6.042, **6.023**, 6.030, 6.035).
A U shape in the best-achievable value at each budget is not what a scaling
study expects to find, and it is the single most important open question in
the study. The two candidate explanations are an un-tuned learning rate at the
top (the law is extrapolated there, and it was never bracketed anywhere) and
something real about the data or the task. Resolving it is what the optimizer
tuning below is for.

### Akima against the parabola, on the same 48 runs

| | parabola (`fit_scaling.py`) | Akima (`fit_scaling_porian.py`) |
|---|---|---|
| usable rungs | 4 of 6 | **5 of 6** |
| pooled alpha | +0.937 | +0.923 |
| 95% interval | [+0.53, +1.42] | **[+0.81, +1.00]** |
| noise-propagated fit | **4000 of 4000 draws refused** | survives |

The point estimate barely moves. Its credibility moves enormously. The rescued
rung is the bottom one, C = 1e14, whose parabola vertex fell below the smallest
model sampled; the top rung is still rejected, now for the defensible reason
that its argmin genuinely sits on the edge.

### Chroma: suggestive, not established

Low chroma at +0.646 against mid at +0.936 is the comparison of interest, and
the intervals [+0.336, +0.818] and [+0.818, +1.014] **abut rather than
separate**, touching at 0.818. Treat that as a power limit, not as a result in
either direction, and note that the buckets are not independent observations:
they are the same 48 runs with the ΔE distribution split three ways.

Low chroma is still the best-behaved bucket in a different sense, as it was in
wave 1: its seed noise is 0.388 ΔE against 1.625 for high chroma, and it is
the only bucket whose top rung survives (N\* = 6.67e6 at C = 2.5e16).

### The saturating fit: not identifiable, and that is informative

`saturating_fit` fits `L(C) = logaddexp(a − alpha·log C, e)`, whose `exp(e)` is
the floor no further compute buys past. It is the direct quantitative test of
the pool-floor hypothesis.

It does not identify. The form is monotonically decreasing, our curve turns
upward at 2.75e15, and the four budgets on the descending branch cannot
constrain three parameters: pooled returns alpha 0.671 at rmse 0.313 and
`identified = False` rather than a floor of zero dressed up as a measurement.
High chroma has only three usable descending budgets and does not fit at all.
**Pinning the floor needs more budgets below 2.75e15**, which are the cheapest
runs in the study.

### Wall clock tracks examples, not FLOPs

`scripts/fit_wall_model.py` fits seconds per example against forward cost per
example over 90 checkpoint intervals from the finished sweep.

| | |
|---|---|
| forward cost per example, spanned | 34x |
| throughput, spanned | 1.9x |
| correlation of log(s/example) with log F | r = +0.59 |
| fitted size coefficient | 5.7e-13 s/FLOP, **p = 0.33** |

The size term does not survive a permutation test once the observations are
clustered by configuration (p = 0.058 treating intervals as independent, 0.33
cluster-aware, and the cluster-aware one is the honest test). Throughput is
not monotone in model size: 2451 ex/s at F = 4.9e7 against 1469 at F = 9.5e7
and 2051 at F = 1.2e8, and repeats of one size scatter as much as the sizes
differ.

So these models are input-bound, as `estimate_wall_sec` assumes, and three
things follow. **Wall clock tracks examples. Service units track examples. The
credit axis of `plot_porian.py --x-axis credits` is a restatement of D.** A
cost-optimal frontier distinct from the compute-optimal one does not exist
here; the cost-optimal choice is simply the largest N the wall cap allows at
the D you want. One side finding: measured throughput is **2171 ex/s** against
the **1301** the planner assumes, so the ladder is sized conservatively by
about 1.7x and configs finish early.

The credit axis converts at **8 SU per L40S-hour**, the `l40s` billing weight
in CRC's published table (CRC bills the max of cores × weight, GB × memory
weight and GPUs × weight, times walltime; memory weight on `l40s` is 0). Run
time is priced at the measured 2171 ex/s, not the planner's conservative 1301,
because a charge is linear in time and the p10 rate overstated every credit by
23 to 28%. The compute-optimal run costs 7.3 SU at C = 1e14 rising to 9.8 SU
at 8.3e15. Over half of the low-rung figure is the fixed 1712 s startup, so cost
is affine in D, not proportional, and the five usable rungs span only 1.3x.

Still `su_rates_confirmed: false`: the table lists the weight "per CPU/GPU",
and if cores on `l40s` carry it too, `--cpus-per-task=8` bills 64 SU per hour,
not 8. Read `billing=` from `sacct -X -M gpu -j <job> --format=AllocTRES%60`
on a finished sweep job before any credit figure leaves the group.

### Epoch repetition is not why the top rung fails

`EPOCH_CEILING` is 1.5, measured, and it is **not binding on this sweep**: 47
of 48 runs are under one epoch and the deepest is 1.45. So the top rung's
one-sidedness cannot be blamed on data repetition. It is the wall-clock floor
from the section above, and the fact that the learning rate at those sizes is
extrapolated from a law that was never bracketed.

### Two sanity checks that look like failures and are not

- **`val_acc` 0.173 to 0.176 everywhere** is the EOS base rate, 1/5.5 = 0.182.
  Read `acc_noEOS`.
- **`acc_noEOS` 0.001 to 0.005 everywhere.** These models essentially never get
  an exact (slot, thickness) pair right, yet reach ΔE ~10 against ~28.6 for a
  model emitting nothing usable. Being one thickness bin off costs little in
  ΔE, so approximate correctness is what is being learned.

---

## What is not settled, and the plan

**The learning rate the sweep ran on rests on three measurements, none of
which bracketed its own optimum.** Under the authors' own `on_edge` test all
three fail:

| grid | points | span | optimum landed | bracketed? |
|---|---|---|---|---|
| 1e-4 to 3e-3 (d128) | 3 | 30x | middle point | no |
| 1e-4 to 3e-3 (d256) | 3 | 30x | middle point | no |
| 1e-4 to 5e-4 (d512) | 4 | 5x | lower bound | no |

A three-point grid can never bracket, because its only interior point is the
second and the second-to-last point at once. The deployed law
`lr(N) = 1.573 N^-0.567` is fitted through those three. Batch size is fixed at
256 by VRAM and deliberately not swept (`METHOD_DIFFS.md` item A). AdamW beta2
has always been torch's 0.999; Porian's own data puts it at 0.95 at batch 256
and reports that tuning it matters most at low batch size, which is the regime
we train in.

Run the stages in `slurms/lr_grid.sh` in order. Each gates the next, and the
cell list for each comes from `scripts/lr_grid_cells.py`, derived from the
sweep's own grid rather than written down by hand.

**Stage `probe`.** 1 cell, 1 LR, <1 GPU-h, run alone. It measures examples per
second, and that number settles two things.

*First, whether the rest is affordable.* The first stage-1 submission ran at a
median of 204 ex/s where the sweep reaches 2171 on the same shards, so the gap
is contention rather than a floor, and which holds is the difference between 20
GPU-hours and 200.

*Second, how many IsoFLOP curves stage 2 tunes in full against how many are
projected from the fitted law.* Each added rung improves the conditioning of
the (N, M) design matrix and roughly doubles the cost:

| rungs tuned | cells | GPU-h @2171 | GPU-h @204 | cond | sd(b) | sd(c) | sweep sizes extrapolated |
|---|---|---|---|---|---|---|---|
| 2 | 12 | 31.3 | 245.3 | 630 | 0.100 | 0.054 | 13 of 24 |
| **3** (default) | **18** | **56.7** | **471.3** | **401** | **0.050** | **0.031** | **9 of 24** |
| 4 | 24 | 103.7 | 927.6 | 303 | 0.032 | 0.021 | 5 of 24 |

GPU-hours count everything a cell pays: one 1712 s startup, then per learning
rate a training pass, a CE validation pass and a DeltaE eval. Pricing the
training pass alone, as an earlier version did, under-stated stage 2 by about a
third. sd(b) and sd(c) are the spreads of the recovered N and M exponents over 2000
synthetic draws at 0.10 of noise in log-lr units. The last column is the number
of the sweep's 24 distinct model sizes that fall above the largest size tuned,
so the fourth curve buys **coverage** as well as conditioning, and coverage is
the stronger argument of the two: an extrapolated size is a size whose learning
rate is a guess.

At 2171 ex/s the fourth curve tightens the N exponent by 1.6x and halves the
extrapolated sizes for twice the compute, which is a real judgement call; at
204 ex/s three curves already costs 471 GPU-hours and four is out of reach, so
the answer is three or nothing. Price both before committing:

```bash
STAGE=2 RUNGS=3 EXAMPLES_PER_SEC=<measured> bash slurms/lr_grid.sh --list
STAGE=2 RUNGS=4 EXAMPLES_PER_SEC=<measured> bash slurms/lr_grid.sh --list
```

**Stage 1.** AdamW beta2 in {0.95, 0.99, 0.999} at both ends of the ladder, three
LRs each at prior/4, prior and prior×4, 6 cells, ~15 GPU-h at 2171 ex/s. Three
rates rather than one because beta2 and the LR interact and a single fixed rate picks whichever beta2 suits
it; three rather than seven because the question is whether the beta2 *ranking*
is stable, not where the LR optimum is. **If the two ends disagree, beta2
interacts with scale, the sequential staging does not hold, and the right move
is to stop rather than carry a wrong constant forward.**

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

**Input pipeline rebuilt, same data (Oct 6). Run `STAGE=speed` before stage 2.**
Same-FLOP runs took different wall time because the GPU sat at 0.03% to 4% of
peak: per example, DataLoader workers spent ~85% of their time turning
spectra into Python floats and back (`.as_py()`, per-material `np.asarray`,
per-material featurize) and under 10% reading parquet, and the training loop
synced the GPU every step (`.item()` twice, plus `valid.any()` and boolean
indexing in `compute_loss_packed`). Changed:

- `src/dataset.py`: each shard is decoded column-wise (`_shard_examples_vectorized`),
  spectra stay in numpy, `pool_features` is computed once per shard, the
  MaterialNK pool is built only if something reads it (`_LazyPool`). Any shard
  whose layout it cannot verify goes through the old per-row decode.
  `INDIGO_LEGACY_DECODE=1` forces the old decode everywhere.
- `scripts/training.py`: collate fills preallocated batch tensors; the loop
  keeps loss sums on the GPU (float64, read at log/checkpoint/epoch only),
  copies batches with `non_blocking`, and logs `data_wait=` (share of time
  blocked on the DataLoader: near 0 means not input-bound).
- Validation rows are cached once per (seed, limit, corpus) as one parquet
  under `cache/splits/` (`load_split_cached`), decoded by the same function;
  `slurms/prepare_split_cache.sh` builds it on smp so no GPU task pays the
  ~3,600-shard read. Startup phases are logged as `[startup]`.
- `PREFETCH=4` (was 1). `NUM_WORKERS` stays 6: the worker count decides batch
  composition and order, so changing it changes the data order.

**Measured (STAGE=speed, job 4230033, Oct 6).** Each task alone on an l40s:

| | before | N = 81k | N = 2.5M |
|---|---|---|---|
| throughput, harmonic mean | 2,033 ex/s | 11,942 | 16,602 |
| `data_wait` | not measured | 64% | 25% |
| startup | ~28 min | 0.8 min | 0.5 min |
| ΔE eval per LR | 146 s assumed | 3.2 min | |

Building the validation cache on smp took 41.6 min (job 24273491), which is
what every GPU task used to spend. Stage 2 now prices at ~25 GPU-h at the
measured rates and 33.8 at the planner's conservative 8,000 ex/s (was 111);
its longest cell is ~3.4 to 4.9 h, so nothing splits. Still input-bound at the
bottom of the ladder (64% waiting on data at 81k): wall time tracks FLOPs
more than it did, not fully. The ΔE eval, single-example greedy decoding plus
the CPU simulator, is now about a third of stage 2's cost.

**Batch-native loader and five curves (Oct 6, overnight; on the branch, not
yet on main).**

- `src/batch_stream.py`: DataLoader workers build each packed batch straight
  from parquet columns, no TrainingExample per row. Handed to the DataLoader
  with `batch_size=None`, so torch's worker-to-shard assignment and
  round-robin are untouched. Batches are bit-identical to the old decode +
  old collate + old DataLoader at 0, 1, 3, 6 and 9 workers, including
  batches spanning shards, each worker's partial last batch, an old-layout
  shard and the same ValueErrors. Planted bugs (float16 structure values,
  dropped partial batches) fail the tests. The pre-Oct-6 `lr_tuning.py` and
  the new one, 6 workers, wrote identical results. `INDIGO_BATCH_LOADER=0`
  reverts to the per-example loader.
- Per worker, collate is 4x cheaper (0.009 vs ~0.035 ms per row), but on
  synthetic shards end to end it is no faster: what is left is the parquet
  read, 0.05 ms per row warm and 0.24 cold. On CRC every shard is read cold
  from /ix1 at ~28 KB per row, so the remaining input limit is probably I/O
  and decompression, which no loader change touches. Each worker now prints
  `[loader] ... read X s (MB, MB/s), decode, collate` at the end of its
  stream, so the next speed run says which. If it is read-bound, the levers
  are the data format (float32 spectra, fewer bytes per row) or caching a
  cell's subset on node-local disk for its 7 learning rates.
- **RUNGS=5** is the default in `lr_grid.sh`, `lr_grid_cells.py` and
  `collect_isoflop.py`. Stage 2 was submitted as RUNGS=4 tasks 0-23 plus
  `STAGE=2 RUNGS=5 --array=24-30` (tasks 0-23 are the same cells either way).
- `STAGE=check` now targets the lowest rung stage 2 does not tune; at
  RUNGS=5 that is 2.5e16, whose minimum fell on its edge, so the check uses
  its best seed-42 point (d416/se7, 17.8M, D = 6.4M, ~2 GPU-h).
- `fit_lr_law.py --coverage-from ...` adds a leave-one-rung-out check: fit
  lr(N, D) without a rung, predict its tuned optima. The held-out top rung is
  the extrapolation test (`tests/test_lr_law_loro.py` plants a law and a law
  that bends).
- `STAGE=speed` is six sizes (81k to 17.8M) at 2,457,600 examples each, for
  rate(N) in the credit cost model, ~1.1 GPU-h. Run with
  `LIMIT_DE_EXAMPLES=0`: throughput needs no DeltaE eval.

**Closing stage 2's open edges (Oct 7): `STAGE=2x` and `STAGE=2edge`.**
Both are decided from results on disk (`scripts/lr_edges.py`) and freeze
their task list at `--list` time (`outputs/lr_search/tasklists/<stage>.txt`),
since their own results change what they would list.

- `2x`: a cell is open when, over its NON-diverged rates, the Akima argmin
  is outside the second and second-to-last rates (porian `tuned_optimum`, the
  fit's own rule). Best at the bottom: two rates below. Best at the top with
  nothing above: two above. Best stable rate just under a diverged one: one
  rate at their geometric midpoint. Written as `<cell>_ext<k>.json` beside the
  cell; `fit_lr_law.py` reads them as more trials of the same (N, D), and
  `collect_isoflop.py` re-selects the point's best trial over all of them.
- `fit_lr_law.py` now drops diverged trials (as lr_tuning.py's selection
  always did), so "best just under a diverged rate" counts as unbracketed.
  In the 22 finished cells, 10 chose 2.7x the prior, the second-highest
  rate; they are open under this rule if 5x diverged.
- `2edge`: for each tuned rung whose pooled-DeltaE minimum is its smallest or
  largest model (from the stage-2 preview's porian_fit_stage2.json), two more
  ladder sizes beyond that end, >= 1.25x apart, at the rung's FLOPs, full
  stage-2 grid. Found again later by their FLOPs (`discover_cells`), so the
  collector puts them on their rung and a second round goes further.
- **The 1e14 rung cannot be bracketed on the left.** Its minimum is at 81k,
  the smallest model, and below that the shape-bounded ladder (aspect 28 to
  72) has only d28/se1 at 72.6k: the parameter count floors on ~50k of fixed
  encoder parameters. It stays an edge rung, excluded by the estimator.

**Nothing finished is invalidated**, and new runs are comparable to the beta2
study and the first sweep: `tests/test_fast_data_path.py` checks bit-for-bit
equality of every example, every batch, the batch order through 6 workers
(old prefetch 1 vs new 4), the validation cache, and the trained weights and
losses against a verbatim copy of the old loop; old and new `lr_tuning.py`
end to end on CPU wrote identical results files. The only value that can
differ is the logged training accuracy, in the last bit. On synthetic shards
the workers are ~2.6x faster end to end (~5x steady state); real shards and
CRC CPUs will differ, which is what `STAGE=speed` measures (smallest, middle,
largest model, 614,400 examples each). Not done, deliberately: batched ΔE
generation (146 s per trial, but it would change eval numerics) and
`torch.compile` / fused AdamW (faster kernels, not bit-identical).

**Long cells run as several array tasks (Oct 6).** `lr_tuning.py` trains a
cell's rates one after another, so stage 2's d96/se3 cell at D = 18.5M would
take ~18.5 h. Any stage-2 or check cell longer than `MAX_TASK_HOURS` (6) is cut
into contiguous runs of its rates, each its own array task writing to
`<OUTPUT_DIR>/parts/`; the last part to land merges the cell into its usual
file (`scripts/merge_lr_parts.py`, which re-selects the optimum by
`lr_tuning.py`'s own rule, so nothing downstream can tell). Same rates, model,
seed and data; 32 tasks instead of 24, +3.8 GPU-h of startup (3%), longest
task 6 h instead of 18.5 h. Task numbering depends on `MAX_TASK_HOURS`, so
change it only between submissions. If a cell's parts all finished but its
file is missing (a part died after training), run the merge by hand through
`slurms/analyze.sh scripts/merge_lr_parts.py --parts-dir <dir>/parts --out-dir <dir>`.

**Why the expensive points stay (Oct 6).** The costly cells are the small-N,
high-D left ends of each curve (time tracks examples, not FLOPs). They were
kept: the lowest curves' minima sit at or next to their left edge, tuning may
move minima further left (panel E: small models wanted 1.2 to 2x the old
prior, large ones 0.5 to 0.7x), and they are the only tuned points at high M
for moderate N, which stage 3's leftmost points need the law to reach.

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

**Stage 2. The LR search runs inside the IsoFLOP test.** Every model on
the `RUNGS` lowest curves is tuned directly: at the default 3 that is 18 cells,
~57 GPU-h. Not one representative point per rung, the whole curve, because the
curve is what the interpolant runs through and a point whose LR was
extrapolated moves the argmin as surely as one trained wrong.

Each point carries its own (N, M), and that is what makes a law in both
variables identifiable with no separate experiment. Within one rung C is fixed,
so M = C/(kN^2) and log M = const − 2 log N: the columns are collinear (corr
−0.9998, cond 9591) and only the combination b − 2c is recoverable. A second
rung shifts the intercept and separates them.

So the multiplier axis comes free from the IsoFLOP geometry. It needs no
fractional-scoring trick and no constant learning rate to get it, which means
the law is measured under the cosine schedule the sweep actually trains with.
This is the one place the study knowingly diverges from Porian et al., and it
is forced: they tune at a constant multiplier (M = 20.0 to 21.1 while
parameters vary 42x), so a law in N alone is the right object for them. Ours
cannot be, because D\*/N\* runs 29.3 to 0.86.

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

**Stage `check`**: the extrapolation check, 1 cell, ~6 GPU-h, run BEFORE
stage 3 spends anything. Tune the compute-optimal point of the highest usable
rung in full and compare what the law predicted against what that point wanted.
It writes to `outputs/lr_search/check/`, so the law is never fitted on the point
that tests it. If the ratio is far from 1, the law does not reach: tune one more
rung directly (`RUNGS` + 1) rather than spend stage 3 on rates that are guesses.

**Stage 3**: finish the IsoFLOP, 30 cells, ~61 GPU-h at 2171 ex/s. Every
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

Where this study **does** do better than the paper it copies is at the small
end. They fit their LR law over a window and extrapolate *above* it, which is
reasonable when the configurations of interest are the large ones. INDIGO's
ladder runs the other way: the smallest rungs, at 0.08M parameters, are both
furthest from where anyone normally tunes and the most LR-sensitive, and they
anchor the low-compute end of every IsoFLOP fit. Stage 2 tunes them densely and
directly (`METHOD_DIFFS.md` item F).

### What the first stage-1 submission cost, so it is not repeated

Six cells, all six hit the six-hour wall after two to four of their seven
learning rates. Two separate problems.

**Throughput 6x below the planner's assumption.** 227 logged step samples: min
13, median 204, max 395 ex/s, against 1301 assumed and 2171 measured on the
sweep's own runs. Same sizes or smaller, so it is not capacity. Six array tasks
were streaming the same shards at once and these runs are input-bound.
`--rate` now defaults to the measured 204 and the table warns when a cell needs
more than 60% of its wall.

**At D = 614,400 nothing learns, so the metric cannot rank learning rates.**
Token accuracy sat at 0.164 to 0.183, the EOS base rate, and training loss
stayed near 6.5 from first step to last. The ΔE values that completed ran 24.8
to 37.4, non-monotone, on a metric whose seed noise is 0.5: scatter around a
model that has not learned, not an optimum. The sweep's own run at N = 81k
needed D = 4.2M to reach ΔE 12.4. **Tune at D\*, not at the historical
614,400.**

---

## Known limitations

Ordered by how much they threaten a published number.

1. **The learning rate was never tuned at any of these scales.** Every run in
   the sweep took its LR from a law fitted through three unbracketed
   measurements. This is the one limitation that could move alpha itself, and
   it is also the leading suspect for the upturn at 2.75e15. It is what the
   staged tuning above exists to close, and nothing in the results section
   should be published before it runs.
2. **Alpha rests on five rungs over 83x in compute** (1e14 to 8.29e15), against
   Porian's twelve over 2048x, and production sits at ~1e17, two decades past
   the top. Reaching production scale from this fit is extrapolation. Rungs
   *below* 1e14 are the cheapest in the study and would serve both this and
   limitation 3.
3. **The irreducible floor is not identified.** Only four budgets sit on the
   descending branch, which cannot constrain a three-parameter saturating form.
4. **Chroma is suggestive, not established.** Low at +0.646 and mid at +0.936
   have intervals that abut at 0.818, and the buckets are not independent
   observations. A null here is a power limit, not evidence of independence.
5. **One seed per point**, apart from one repeat per rung. Those pairs are the
   only error bar in the study, and they are what the noise model is calibrated
   from. Init variance is real: two identical d512 LR sweeps once gave val_de
   41.30 against 24.33 before seeding was fixed.
6. **head_dim 32 differs from production's 64**, deliberately, for width
   granularity. Internal consistency across the ladder is the right trade, but
   it is a real architectural difference between the ladder and the checkpoint
   the ladder is used to judge, and it already cost one comparison: v1's
   standout d128/se3 at val_de 9.565 ran head_dim 64 and has not been
   reproduced at head_dim 32 (10.820 at comparable D).
7. **Aspect ratio is a band, not a policy.** 28 to 72, a 2.6x spread and not
   monotone in width, against Porian's 1.6x drifting monotonically. Much
   tighter than v1's disaster, looser than theirs, and the residual variation
   is still a confound with N.
8. **Production's architecture is unconfirmed here.** Whether prod is 17.5M or
   69.6M parameters changes the headline ratio from 18x to 72x. Resolve with
   `cat data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000/config.json`.
9. **Batch size 256 may not match production's 512.** All 48 runs used 256 and
   the optimal LR depends on it, so raising the constant means re-tuning and
   re-running: existing runs would not be comparable to new ones.
10. **The credit axis converts at 8 SU per L40S-hour, core weight unconfirmed.**
    The GPU weight is CRC's published figure; whether the 8 cores each job
    requests are also billed at 8 (64 SU per hour) is not. One `sacct` on a
    finished job settles it.
