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

### A. Batch size  **(now implemented, awaiting the run)**

They sweep 5 to 7 batch sizes per model size, spanning 64x, and minimise over
that axis before fitting. INDIGO fixed `batch_size = 256` everywhere and never
looked, so a moving optimum was a scale-dependent handicap of unknown sign.

`nested_hparam_optimum` now does their nested minimisation: beta2 collapses
first, then the learning rate within each batch size, then the batch size
across them, carrying the inner optimum out by interpolation rather than
re-reading a grid cell. `scripts/fit_lr_law.py` fits `bs(N)` alongside `lr(N)`,
and `batch_size_for()` in `src/scaling/configs.py` supplies it to the sweep.

Critically, D is held fixed **in examples** while batch size varies, so steps
adjust and every cell sees the same data. That matches their design: within a
model size their token budget is constant to within 1% across all 64 to 112
cells. Varying batch size at fixed *steps* instead would change D and confound
the two axes.

### B. AdamW beta2  **(now implemented, awaiting the run)**

They sweep {0.95, 0.99, 0.999} and report that tuning beta2 is **essential at
lower batch sizes**. INDIGO ran torch's default of 0.999, the top of that
range, at batch 256, which is the small end of their grid: exactly the regime
they flag.

Neither `lr_tuning.py` nor `training.py` exposed the betas at all. Both do now,
the grid sweeps beta2 as its own axis, and `beta2_for()` supplies the tuned
value. Stage 1 of `slurms/lr_grid.sh` exists to answer whether it moves: if it
is flat, stages 2 and 3 drop to one value and get three times cheaper.

### C. Constant LR versus cosine decay

Their tuned arm uses `decay = const`, and one of the paper's findings is that
careful LR decay is not essential to the scaling law. INDIGO uses cosine. Not
wrong, but it is an uncontrolled difference, and it interacts with the missing
D term: under cosine the schedule's shape depends on total steps, so the same
LR means something different at different D. Under a constant LR it does not.

### D. No D term in the LR law

Covered at length in the meeting doc. Theirs is arguably not a "law in N" at
all: they tune at the configurations they then use, so whatever D dependence
exists is absorbed into each tuned optimum. We extrapolate one N-only law to
configurations never tuned at, spanning 9x in D at fixed N. Stage 3 of `slurms/lr_grid.sh` plus the 2-D fit in `scripts/fit_lr_law.py`
measures the exponent, and `tuned_lr_for(n, d)` prefers the two-dimensional law
over the one-dimensional one wherever the dataset size is known.

### E. Budget count and spacing

12 rungs over 2048x versus 6 over 250x. Our exponent rests on five points.
Adding rungs *below* 1e14 is cheap and would serve both this and the
saturating-floor fit in item 8.

### F. Aspect ratio policy

Theirs drifts 32 to 50 monotonically with width, a 1.6x spread. Ours is a band
of 28 to 72, a 2.6x spread, not monotone. Tighter than sweep v1's disaster, but
looser than theirs, and the residual variation is still a confound with N.

### G. Checkpoint selection

They checkpoint at pre-specified FLOP values and evaluate there. We take the
final checkpoint of each run. Ours is the stricter reading of an IsoFLOP point
and should stay, but it is a difference.

### H. Metric

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

**The chroma stratification is ours.** Nothing in their method fits separate
laws per difficulty stratum, and it is where INDIGO's most interpretable result
lives: low chroma is the bucket with the cleanest law.
