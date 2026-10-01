# INDIGO vs Porian et al. 2024: a method audit

Porian, Wortsman, Jitsev, Schmidt and Carmon, *Resolving Discrepancies in
Compute-Optimal Scaling of Language Models*, NeurIPS 2024 (arXiv:2406.19146).

This is a line-by-line comparison against the authors' released analysis code
(`github.com/formll/resolving-scaling-law-discrepancies`), not against the
paper text, and the numbers attributed to them below were read out of their
published `experiment_results.pickle.xz` rather than quoted. Everything marked
**fixed** is implemented in `src/scaling/porian.py`, `scripts/fit_scaling_porian.py`,
`scripts/fit_lr_law.py` and `slurms/lr_grid.sh`.

---

## What the sweeps actually are

|  | Porian et al. | INDIGO |
|---|---|---|
| FLOP budgets | 12, spaced 2x, spanning 2048x | 6, spaced ~3x, spanning 250x |
| Model sizes per budget | varies | 8 |
| Runs in the study | 975 | 48 |
| Aspect ratio (width/depth) | 32.0 to 50.5, 1.6x, rising monotonically with width | band 28 to 72, 2.6x |
| Hyperparameter sweep | **566 runs** at 7 model sizes | **3 runs** at 3 model sizes |
| Learning rates per size | 6 to 7, spanning 64x (7.5e-4 to 4.8e-2) | 3 to 4, spanning 5x to 30x |
| Batch sizes per size | 5 to 7, spanning 64x (8 to 512) | **1** (fixed at 256) |
| AdamW beta2 values | **3** (0.95, 0.99, 0.999) | **1** (default) |
| Sizes the tuned law is applied to | 16 | 9 |
| LR schedule in the tuned arm | **constant, no decay** | cosine |
| Metric | validation cross-entropy | ΔE00 median (CE is decoupled on INDIGO) |

Their hyperparameter sweep is 64 to 112 runs *per model size*. Ours was three
runs in total. That single row explains most of what follows.

---

## How the tuned laws reach the sweep

`src/scaling/configs.py` loads `analyses/scaling/results/lr_law_fit.json` when
it exists and falls back to the old three-point law when it does not, so
nothing about the existing sweep moves until the grid has actually been run.
The dry run says which law is in force. Set `TUNED_LAWS_PATH` to point
elsewhere, or at a path that does not exist to force the fallback.

## Differences that are now fixed

### 1. Parabola vs Akima spline  **(the big one)**

They interpolate loss vs N with `Akima1DInterpolator` in log-log space over a
geometric grid of `(n_points - 1) * 25` samples and take N* as its argmin. We
fitted a quadratic in log N.

A parabola *forces* an interior minimum. Fed a flat or monotone series it still
returns a confident N*, and the only defence is to check afterwards whether the
vertex landed inside the sampled range. A spline simply puts its argmin on the
boundary, which is then rejected.

Measured effect on our own 48 runs, pooled ΔE:

| | parabola (ours) | Akima (theirs) |
|---|---|---|
| usable rungs | 4 of 6 | **5 of 6** |
| alpha | +0.937 | +0.923 |
| 95% CI | [+0.53, +1.42] | **[+0.81, +1.00]** |
| r^2 | - | 0.997 |
| noise-propagated fit | **all 4000 draws refused** | survives |

The headline number barely moves. Its credibility moves enormously: the
exponent goes from not identifiable under measured seed noise to tight. The
rescued rung is the bottom one, C = 1e14, whose parabola vertex fell below the
smallest model sampled. The top rung, C = 2.5e16, is still rejected, and now
for a defensible reason: its interpolated argmin genuinely sits on the edge.

### 2. Seed-noise bootstrap, and the median is the observation

Their `vectorized_interp_with_seed_noise` redraws every loss 1000 times with
calibrated noise, re-interpolates, and keeps the draws whose argmin stays
interior. The reported N* is the **median of those draws**, sigma is their
log-space std, and a rung is dropped only when **fewer than half** survive.
Sigma is floored at a fraction of the grid step and then **inflated by
`n_boot / n_valid`**, so a rung that barely survives is down-weighted rather
than silently trusted.

Ours refused the whole fit if the Monte Carlo was unstable, which threw away
usable information. Theirs degrades gracefully.

Our noise is calibrated differently by necessity: their sigma interpolates with
the loss level (0.002 to 0.05 across CE 3 to 7), which is meaningless for ΔE.
`NoiseModel.from_clusters` keeps their shape but takes endpoints from our six
repeat-seed clusters, which support a pooled constant (sigma = 0.517 on pooled
ΔE) and show no clean trend with the metric value.

### 3. The step-2 fit is weighted by 1/sigma^2

Theirs is. Ours was unweighted, so the least identified rungs pulled the
exponent as hard as the best ones. Both are now reported side by side.

### 4. Interpolated hyperparameter optima, not the best grid point

Their `minimize_with_interp` takes the LR optimum as the Akima argmin in
log-log space, not the winning grid point. With a coarse grid the two differ a
lot, and only the interpolated version makes the next item meaningful.

### 5. The `on_edge` bracketing test

They flag any optimum falling outside the second and second-to-last grid
points: such a sweep did not measure an optimum, it hit a wall.

Applied to INDIGO's three historical LR measurements, **all three fail**:

| grid | points | span | result | bracketed? |
|---|---|---|---|---|
| 1e-4 to 3e-3 (d128) | 3 | 30x | middle point | no |
| 1e-4 to 3e-3 (d256) | 3 | 30x | middle point | no |
| 1e-4 to 5e-4 (d512) | 4 | 5x | lower bound | no |

A three-point grid can never bracket: its only interior point *is* the second
and second-to-last point at once. So the deployed law
`lr(N) = 1.573 * N^-0.567` rests on three measurements of which none would
qualify under their test. `scripts/fit_lr_law.py` now applies it and excludes
failures from the fit rather than averaging them in.

### 6. The LR law is fitted over a window, then extrapolated

They fit only over `2.5e7 < params < 1.1e8` and extrapolate above it, which
keeps one endpoint from setting the slope and leaves larger configs as a
held-out check. Ours fitted across the entire range, which is how a single
unbracketed boundary value came to determine the exponent. `--min-params`,
`--max-params` and the held-out extrapolation report now implement this.

### 7. The multiplier D*/N* gets its own power law

They fit it; we never reported it. It is the Chinchilla headline quantity (the
famous ~20 tokens per parameter, an exponent of ~0). Ours:

| bucket | multiplier exponent |
|---|---|
| pooled | **-0.783** |
| low chroma | -0.241 |
| mid chroma | -0.807 |
| high chroma | -0.586 |
| cross-entropy | -0.572 |

Examples per parameter falls steeply with compute on INDIGO, where Chinchilla's
is flat. That is the same fact as alpha ~ 0.9 versus 0.5, stated in the units
the field actually argues about.

### 8. Loss is fitted with an irreducible-floor term

Their `fit_loss_with_saturation` fits `L(C) = logaddexp(a - alpha*log C, e)`,
whose `exp(e)` is the floor no further compute buys past. For INDIGO that is
the direct quantitative test of the pool-floor hypothesis, so it is now
implemented as `saturating_fit`.

**Result: not identifiable, and that is informative.** The model is
monotonically decreasing, our curve turns upward at C = 2.75e15, and the four
budgets on the descending branch cannot separate a three-parameter saturating
model. `saturating_fit` reports `identified = False` rather than returning a
floor of zero as though it were measured. Pinning the floor needs more budgets
*below* 2.75e15, which are the cheapest runs in the study.

---

## Differences that remain open

### A. Batch size  **(deliberately not adopted)**

They sweep 5 to 7 batch sizes per model size and minimise over that axis. We do
not, and will not: on INDIGO batch size is a VRAM decision rather than a tuning
one. These models are input-bound rather than GPU-bound, so the largest batch
the card holds maximises throughput and nothing is traded away. Their optimum
moves with scale because theirs is a real tuning axis; ours is pinned by
hardware.

What matters instead is that ONE value is used everywhere, since the optimal
learning rate depends on it. `src/scaling/configs.py:DEFAULT_BATCH_SIZE` is the
single source of truth and `slurms/lr_grid.sh` reads it rather than carrying
its own default, so a law tuned at one batch size can never be applied at
another.

**Open question for the sweep itself.** All 48 runs used **256**. If production
runs 512, the sweep is not matched to production, and raising the constant
means re-tuning and re-running: existing runs would not be comparable to new
ones. `SWEEP_BATCH_SIZE` overrides it, and the sweep refuses to dispatch into
an output root that already holds runs.

### B. AdamW beta2  **(adopted; awaiting the run)**

They sweep {0.95, 0.99, 0.999} and report that tuning beta2 is **essential at
lower batch sizes**. INDIGO ran torch's default of 0.999, the top of that
range, at a batch size that sits at the small end of their grid: exactly the
regime they flag.

Neither `lr_tuning.py` nor `training.py` exposed the betas at all. Both do now,
beta2 is a swept axis, and `beta2_for()` supplies the tuned value to the sweep.
Stage 1 of `slurms/lr_grid.sh` runs it at the smallest and largest rungs, which
answers both whether it matters and whether its optimum moves with scale.

### C. Constant LR versus cosine decay

Their tuned arm uses `decay = const`, and one of the paper's findings is that
careful LR decay is not essential to the scaling law. INDIGO uses cosine. Not
wrong, but it is an uncontrolled difference, and it interacts with the missing
D term: under cosine the schedule's shape depends on total steps, so the same
LR means something different at different D. Under a constant LR it does not.

### D. No D term in the LR law  **(the one place we knowingly diverge)**

Theirs is arguably not a "law in N" at all: they tune at the configurations
they then use, so whatever D dependence exists is absorbed into each tuned
optimum. We extrapolated one N-only law to configurations never tuned at,
spanning 9x in D at fixed N.

They can get away with a law in N because they tune at a CONSTANT multiplier:
M = 20.0 to 21.1 while parameters vary 42x. INDIGO cannot, because D*/N* runs
29.3 to 0.86 across our budgets.

The fix is to put the learning-rate search INSIDE the IsoFLOP test. Stage 2 of
`slurms/lr_grid.sh` tunes every model on the lowest few curves, so each tuned
cell carries its own (N, M) and the 2-D fit in `scripts/fit_lr_law.py` is
identifiable from the sweep's own geometry. Within one rung C is fixed, so
log M = const - 2 log N and the columns are collinear (corr -0.9998, cond
9591); a second rung shifts the intercept and separates them, three give cond
401 and recover both exponents to 0.05 and 0.03 under 0.10 of log-lr noise.

Two consequences worth stating. The multiplier axis needs no fractional-scoring
trick and no constant learning rate, so the law is measured under the cosine
schedule the sweep actually trains with. And `tuned_lr_for(n, d)` prefers the
two-dimensional law over the one-dimensional one wherever the dataset size is
known.

### E. Budget count and spacing

12 rungs over 2048x versus 6 over 250x. Our exponent rests on five points.
Adding rungs *below* 1e14 is cheap and would serve both this and the
saturating-floor fit in item 8.

### F. Coverage of the small rungs  **(better than theirs at the small end, worse at the large)**

They fit the learning-rate law over a window of model sizes and extrapolate
ABOVE it. That is reasonable when the configurations you care about are the
large ones. INDIGO's ladder runs the other way: the smallest rungs, at 0.08M
parameters, are both the furthest from where anyone normally tunes and the most
sensitive to learning rate, and they anchor the low-compute end of every
IsoFLOP fit.

So stage 2 tunes the low rungs DENSELY and completely rather than sampling one
point per rung, and `scripts/fit_lr_law.py --coverage-from` lists every distinct
model size the sweep trains and says whether the fitted law interpolates or
reaches past the tuned range, and by how far.

Zero extrapolated sizes is not reachable at any affordable number of rungs, and
the earlier version of this file claimed otherwise. Tuning the lowest three
curves covers 15 of the sweep's 24 distinct sizes and leaves 9 above the tuned
range; four curves leaves 5. That trade is the second objective of the
throughput probe, and stage 3 is the check on whatever is left extrapolated.

The learning-rate grid is also centred per cell on the prior rather than being
one fixed window, because small models want much higher learning rates than
large ones and a single window wide enough for both resolves neither.

### G. Aspect ratio policy

Theirs drifts 32 to 50 monotonically with width, a 1.6x spread. Ours is a band
of 28 to 72, a 2.6x spread, not monotone. Tighter than sweep v1's disaster, but
looser than theirs, and the residual variation is still a confound with N.

### H. Checkpoint selection

They checkpoint at pre-specified FLOP values and evaluate there. We take the
final checkpoint of each run. Ours is the stricter reading of an IsoFLOP point
and should stay, but it is a difference.

### I. Metric

They fit validation cross-entropy. We fit ΔE00 median, deliberately, because
CE and ΔE are decoupled on INDIGO. This is correct for us, but it means our
"loss" fits are not directly comparable to theirs, and it forced the noise
recalibration in item 2.

---

## What we do that they do not

**Last-layer FLOP accounting is a much larger correction here.** Their
correction #1 concerns a vocabulary projection that matters only at small N.
INDIGO's factorised pointer head runs per (position, slot), 352 times per
example, and the slot encoder runs over 32 pool slots rather than the decoder
sequence. `src/scaling/flops.py` computes C = 3 * F(N) * D analytically, and
over this ladder C/(N*D) falls from 306 to 218, so a per-token count would be
off by 36x to 51x with the error itself drifting 1.4x across the grid.

**Effective parameters, and making C = 6ND exact.** The parameter count does
not satisfy a constant-k compute law here, so `alpha + beta` lands near 1.06
rather than on 1.

Their `params` column is `flops_per_token / 6`, but since
`flops_per_token = 6 * params_active_precise` for their architecture, that is an
identity and their headline N is a plain parameter count. Effective parameters
appear only in their attention-accounting variant, `eff_params_att`, a
supplementary figure worth 1.10x to 1.22x across their ladder. They report both
in a case where the two barely differ.

Ours differ by 36x to 49x. `src/scaling/flops.py:effective_params` defines
`N_eff = forward_flops_per_example / 2`, which makes C = 6 N_eff D an identity
and forces `alpha + beta = 1`, and the fit reports the exponent in both units:
pooled alpha is +0.923 in parameters, +0.983 in effective parameters
(`N_eff ~ 98.9 * N^0.940`, r^2 = 0.9999). Keep the parameter count as the
headline, since it is a model size and the effective one is not, and always say
which is being quoted.

**Whether wall clock tracks either of them is a separate question**, and it is
now measured. `scripts/fit_wall_model.py` fits seconds per example against
forward cost per example over 90 checkpoint intervals from the finished sweep:

| | |
|---|---|
| forward cost per example, spanned | 34x |
| throughput, spanned | 1.9x |
| correlation of log(s/example) with log(F) | r = +0.59 |
| fitted size coefficient | 5.7e-13 s/FLOP, **p = 0.33** |

The size term does not survive a permutation test, throughput is not monotone
in model size (2451 ex/s at F = 4.9e7 against 1469 at F = 9.5e7 and 2051 at
F = 1.2e8), and repeats of one size scatter as much as the sizes differ. These
models are input-bound, as `estimate_wall_sec` assumes: **wall clock tracks
examples, service units track examples, and the credit axis is a restatement of
D.** A cost-optimal frontier separate from the compute-optimal one does not
exist here; the cost-optimal choice is simply the largest N the wall cap allows
at the D you want.

One actionable side finding: measured throughput is about 2171 ex/s against the
1301 the planner assumes, so the ladder is sized conservatively by roughly 1.7x
and configs finish early.

**The chroma stratification is ours.** Nothing in their method fits separate
laws per difficulty stratum, and it is where INDIGO's most interpretable result
lives: low chroma is the bucket with the cleanest law.

---

## Stage 1, first submission: what it cost and what it showed

Job 4123703, six cells, submitted at the sizing in this file's earlier version.
Every cell hit the six-hour wall after completing two to four of its seven
learning rates. Two separate problems, both worth recording before the next
attempt.

**Throughput was 6x below the planner's assumption.** 227 logged step samples
give min 13, median 204, max 395 examples per second, against the 1301
`EXAMPLES_PER_SEC` assumes and the 2171 the wall-model fit measured on the
sweep's own runs. The models here are the same size or smaller, so this is not
capacity: six array tasks were streaming the same shards at once, and these
runs are input-bound. `scripts/lr_grid_cells.py --rate` now defaults to the
measured 204 and the table warns when a cell needs more than 60% of its wall.

**At D = 614,400 nothing learns, so the metric cannot rank learning rates.**
Token accuracy sat at 0.164 to 0.183 across every trial, which is the EOS base
rate (1/5.5 = 0.182), and training loss stayed near 6.5 from first step to
last. The DeltaE values that did complete:

| N | beta2 | lr | val DeltaE |
|---|---|---|---|
| 100,493 | 0.95 | 7.6e-5 | 28.8 |
| 100,493 | 0.95 | 2.4e-4 | 32.4 |
| 100,493 | 0.95 | 7.4e-4 | 24.8 |
| 100,493 | 0.99 | 7.6e-5 | 30.6 |
| 100,493 | 0.999 | 7.6e-5 | 37.4 |
| 6,606,789 | 0.95 | 7.1e-6 | 27.0 |
| 6,606,789 | 0.95 | 2.2e-5 | 26.6 |

Non-monotone, no bowl, and spanning 25 to 37 on a metric whose seed noise is
0.5. That is scatter around a model that has not learned, not an optimum. The
tuning budget has to be large enough that DeltaE responds to the learning rate
at all, which is why stage 2 tunes at each rung's own D* (3.4M to 6.1M
examples) rather than at the historical 614,400.

The two problems compound: stage 2 as now designed costs 56.7 GPU-hours at the
sweep's 2171 ex/s and 471.3 at the measured 204. Resolve the throughput first.
The sweep reached 2171 ex/s on the same data, so 204 is contention rather than
a floor, and a single unthrottled cell is the measurement that settles it. That
same number also decides whether three or four IsoFLOP curves get tuned, the
rest being projected: see the stage table in `slurms/lr_grid.sh`.
