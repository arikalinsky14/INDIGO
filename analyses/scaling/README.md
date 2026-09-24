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

---

## What we are trying to answer

1. **For a fixed compute budget C, what model size N\* minimises ΔE?**
   Fit a parabola to (log N, val_de) at each budget, read off its minimum,
   then fit a power law N\*(C) ∝ C^α across budgets. Same for D\*(C) ∝ C^β.

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
  divergence screen.

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

**The geometry fix worked.** All 12 parabolas now open upward with R^2
0.77-0.98, against v1's three-of-four refusals. N\* is inside the sampled
range at 11 of 12. Real IsoFLOP curves exist.

Pooled alpha = **+0.71, 95% interval [+0.52, +0.98]**, over three rungs and
one decade. Notably above Chinchilla's 0.5, but only marginally excluding it.

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

### The open confound, and the cheap experiment that settles it

Every ladder point shared one LR (the law is a function of N alone) and one
cosine schedule. So "more passes hurt" and "this LR schedule degrades over
long horizons" are not yet separated. CLAUDE.md records cosine death on the
finetune line, and production's optimum at 0.67 epochs is equally consistent
with either. Two deep-end runs with constant or re-tuned LR (~15 GPU-hours)
decide it, and the answer moves the epoch ceiling and with it the whole
reachable budget range.

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

## Known limitations

1. **Lever arm.** Under `qos=short` the ladder reaches 4e15, which is 1.6
   decades and well short of production's ~1e17. α is fitted over a
   shorter range than is ideal, and extrapolating to production scale is
   an extrapolation.
2. **Chroma power.** See *Statistical power*. A null result on the
   chroma-conditioned frontier is probably a power limit.
3. **One seed per point**, apart from one repeat per rung. Init variance
   is real: two identical d512 LR sweeps once gave val_de 41.30 versus
   24.33 before seeding was fixed.
4. **The epoch ceiling is a non-detection**, not a verified safe depth.
5. **Production's architecture is unconfirmed here.** Whether prod is
   17.5M or 69.6M parameters changes the headline ratio from 18× to 72×.
   Resolve with
   `cat data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000/config.json`.
6. **head_dim 32 differs from production's 64.** Deliberate, for width
   granularity, but it is an architectural difference between the ladder
   and the checkpoint the ladder is used to judge.
