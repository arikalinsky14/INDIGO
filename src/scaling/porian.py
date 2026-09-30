"""Porian et al. 2024 (arXiv:2406.19146) IsoFLOP estimator, ported to INDIGO.

This is a faithful port of the analysis in the authors' released code
(github.com/formll/resolving-scaling-law-discrepancies, `analysis.py`), not a
reimplementation from the paper text. Where their choice and ours differed,
theirs is implemented here and the difference is noted in
`analyses/scaling/METHOD_DIFFS.md`.

What they do per IsoFLOP rung, and we did not
---------------------------------------------
1. **Akima spline, not a parabola.** Loss vs N is interpolated with
   `Akima1DInterpolator` in log-log space on a geometric grid of
   `(n_points - 1) * 25` samples, and N* is the argmin of that curve. A
   parabola *forces* an interior minimum and reports a confident N* even for a
   monotone series; the spline simply puts its argmin on the boundary, which is
   then rejected. Our flat top rungs are exactly the case this distinguishes.

2. **Boundary rejection, not extrapolation.** A rung is usable only if the
   argmin index is neither 0 nor `interp_num - 1`.

3. **Seed-noise bootstrap, and the median is the observation.** They redraw the
   losses `bootstrap_iters` times with calibrated noise, re-interpolate, and
   keep the draws whose argmin stays interior. The reported N* is the *median*
   of those draws rather than the noiseless point estimate, sigma is their
   log-space std, and a rung is dropped only if fewer than half the draws
   survive. Sigma is floored at `min_std_factor * log(grid step)` and then
   inflated by `n_boot / n_valid`, so a rung that barely survives is
   down-weighted rather than silently trusted.

4. **The step-2 fit is weighted by 1/sigma^2.** Ours was unweighted, which let
   the least identified rungs pull the exponent as hard as the best ones.

5. **They fit the multiplier too.** D*/N* (their `multiplier`) gets its own
   power law. It is the quantity Chinchilla is famous for (~20 tokens/param),
   and it is not derivable from the N* exponent alone once the FLOP model is
   not proportional to N.

6. **They fit loss with a saturation term**, `L(C) = logaddexp(a - alpha*log C,
   e)`, whose `exp(e)` is an irreducible floor. For INDIGO that term is the
   direct quantitative test of the "there is a DeltaE floor" hypothesis, so it
   is exposed here as `saturating_fit`.

Noise calibration
-----------------
Their `get_noise_for_loss` interpolates sigma log-linearly between two loss
thresholds, calibrated for LM cross-entropy (sigma 0.002 to 0.05 over loss 3 to
7). DeltaE_00 lives on a different scale entirely, so `NoiseModel` keeps their
*shape* but takes its endpoints from measured repeat-seed clusters. Use
`NoiseModel.from_clusters` and pass the runs; do not hardcode their numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit

INTERP_MULTIPLIER = 25          # their `interp_num_multiplier`
BOOTSTRAP_ITERS = 1000          # their `bootstrap_iters`
MIN_STD_FACTOR = 0.33           # their `min_std_factor`
MIN_VALID_FRACTION = 0.5        # their "< bootstrap_iters // 2 -> give up"


# ---------------------------------------------------------------- noise ----
@dataclass(frozen=True)
class NoiseModel:
    """Per-observation sigma as a log-linear function of the metric value.

    Mirrors `get_noise_for_loss`: below `lo_at` sigma is `sigma_lo`, above
    `hi_at` it is `sigma_hi`, and in between it interpolates linearly in
    log-log. With `sigma_lo == sigma_hi` this degenerates to constant noise,
    which is what INDIGO's own repeat seeds support.
    """
    sigma_lo: float
    sigma_hi: float
    lo_at: float
    hi_at: float

    def __call__(self, value: np.ndarray) -> np.ndarray:
        v = np.asarray(value, dtype=float)
        if self.sigma_lo == self.sigma_hi:
            return np.full(v.shape, self.sigma_lo)
        lv = np.log(np.clip(v, 1e-12, None))
        return np.exp(np.interp(lv,
                                [np.log(self.lo_at), np.log(self.hi_at)],
                                [np.log(self.sigma_lo), np.log(self.sigma_hi)]))

    @classmethod
    def from_clusters(cls, clusters: Sequence[Sequence[float]],
                      constant: bool = True) -> "NoiseModel":
        """Calibrate from repeat-seed clusters (one inner sequence per config).

        Two-seed clusters use |a - b| / sqrt(2); three or more use the sample
        std. With `constant=True` (the default, and what six INDIGO clusters
        actually support) every observation gets the pooled median sigma.
        """
        means, sigmas = [], []
        for c in clusters:
            v = np.asarray([x for x in c if np.isfinite(x)], dtype=float)
            if v.size < 2:
                continue
            s = abs(v[0] - v[1]) / np.sqrt(2) if v.size == 2 else float(v.std(ddof=1))
            means.append(float(v.mean()))
            sigmas.append(s)
        if not sigmas:
            raise ValueError("no repeat-seed cluster had two or more runs")
        med = float(np.median(sigmas))
        if constant or len(sigmas) < 4:
            return cls(med, med, 1.0, 10.0)
        order = np.argsort(means)
        m, s = np.asarray(means)[order], np.asarray(sigmas)[order]
        k = max(1, len(m) // 3)
        return cls(float(np.median(s[:k])), float(np.median(s[-k:])),
                   float(np.median(m[:k])), float(np.median(m[-k:])))


# ------------------------------------------------------------ isoflop ----
@dataclass
class RungFit:
    """One IsoFLOP rung. `usable` is False when the argmin sat on the edge."""
    budget: float
    n_vals: np.ndarray
    y_vals: np.ndarray
    n_grid: np.ndarray
    y_grid: np.ndarray
    star_index: int
    n_star: float
    y_star: float
    on_edge: bool
    d_vals: Optional[np.ndarray] = None
    n_star_median: Optional[float] = None
    log_sigma: Optional[float] = None
    valid_fraction: Optional[float] = None
    samples: Optional[np.ndarray] = None

    @property
    def usable(self) -> bool:
        return (not self.on_edge) and self.log_sigma is not None

    def d_of(self, n) -> np.ndarray:
        """D at a given N on this rung, by log-log interpolation of its points.

        Along a rung C is fixed and D = C / (3 * F(N)), a deterministic
        function of N, so this is interpolation rather than inference. It is
        NOT C / (6 * N): INDIGO's forward cost per example is not proportional
        to N (see src/scaling/flops.py), so the 6ND shortcut Porian et al. can
        use for transformers does not hold here.
        """
        if self.d_vals is None:
            raise ValueError("this rung carries no D values")
        order = np.argsort(self.n_vals)
        return np.exp(np.interp(np.log(np.asarray(n, dtype=float)),
                                np.log(self.n_vals[order]),
                                np.log(self.d_vals[order])))

    @property
    def d_star_median(self) -> float:
        return float(self.d_of(self.n_star_median))


def akima_argmin(x: Sequence[float], y: Sequence[float],
                 interp_multiplier: int = INTERP_MULTIPLIER
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Their `interpolation`: Akima in log-log over a geometric grid.

    Duplicate x are collapsed to their minimum y first, matching their
    `groupby_action='min'`; Akima requires strictly increasing x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep: Dict[float, float] = {}
    for xi, yi in zip(x, y):
        if xi not in keep or yi < keep[xi]:
            keep[xi] = yi
    xs = np.array(sorted(keep))
    ys = np.array([keep[v] for v in xs])
    if xs.size < 3:
        raise ValueError(f"Akima needs at least 3 distinct x, got {xs.size}")
    grid = np.geomspace(xs.min(), xs.max(), (xs.size - 1) * interp_multiplier)
    interp = Akima1DInterpolator(np.log(xs), np.log(ys))
    y_grid = np.exp(interp(np.log(grid)))
    return grid, y_grid, int(np.argmin(y_grid))


def fit_rung(budget: float, n_vals: Sequence[float], y_vals: Sequence[float],
             d_vals: Optional[Sequence[float]] = None,
             noise: Optional[NoiseModel] = None,
             n_boot: int = BOOTSTRAP_ITERS,
             interp_multiplier: int = INTERP_MULTIPLIER,
             min_std_factor: float = MIN_STD_FACTOR,
             rng: Optional[np.random.Generator] = None) -> RungFit:
    """One rung, with their bootstrap. Returns a RungFit even when unusable."""
    grid, y_grid, idx = akima_argmin(n_vals, y_vals, interp_multiplier)
    fit = RungFit(budget=budget, n_vals=np.asarray(n_vals, float),
                  y_vals=np.asarray(y_vals, float),
                  n_grid=grid, y_grid=y_grid,
                  star_index=idx, n_star=float(grid[idx]),
                  y_star=float(y_grid[idx]),
                  on_edge=(idx == 0 or idx == grid.size - 1),
                  d_vals=None if d_vals is None else np.asarray(d_vals, float))
    if noise is None or fit.on_edge:
        return fit

    rng = rng or np.random.default_rng(0)
    keep: Dict[float, float] = {}
    for xi, yi in zip(np.asarray(n_vals, float), np.asarray(y_vals, float)):
        if xi not in keep or yi < keep[xi]:
            keep[xi] = yi
    xs = np.array(sorted(keep))
    ys = np.array([keep[v] for v in xs])
    lx, lg = np.log(xs), np.log(grid)
    sigma = noise(ys)

    stars: List[float] = []
    for draw in ys + sigma * rng.standard_normal((n_boot, ys.size)):
        if np.any(draw <= 0):
            continue
        yb = Akima1DInterpolator(lx, np.log(draw))(lg)
        j = int(np.argmin(yb))
        if j != 0 and j != grid.size - 1:
            stars.append(float(grid[j]))

    fit.valid_fraction = len(stars) / n_boot
    if fit.valid_fraction < MIN_VALID_FRACTION:
        return fit

    arr = np.asarray(stars)
    # Their sigma: log-space std, floored at a fraction of the grid step, then
    # inflated by the share of draws that fell off the edge.
    s = float(np.std(np.log(arr)))
    floor = min_std_factor * interp_multiplier * float(np.log(grid[1] / grid[0]))
    fit.log_sigma = max(s, floor) / fit.valid_fraction
    fit.n_star_median = float(np.median(arr))     # their `bs_median_as_obs`
    fit.samples = arr
    return fit


# ---------------------------------------------------------- power laws ----
@dataclass
class PowerLaw:
    exponent: float
    coef: float
    r2: float
    n_points: int
    weighted: bool
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def __call__(self, x):
        return self.coef * np.asarray(x, dtype=float) ** self.exponent


def power_law_fit(x: Sequence[float], y: Sequence[float],
                  sigma: Optional[Sequence[float]] = None) -> PowerLaw:
    """Their `power_law_fit`: OLS in log-log, optionally 1/sigma^2 weighted."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    w = None
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        if np.all(np.isfinite(s)) and np.all(s > 0):
            w = 1.0 / s ** 2
    if lx.size < 2:
        raise ValueError("power law needs at least two points")
    W = np.ones_like(lx) if w is None else w
    X = np.vstack([lx, np.ones_like(lx)]).T
    A = X.T @ (W[:, None] * X)
    b = X.T @ (W * ly)
    slope, intercept = np.linalg.solve(A, b)
    pred = slope * lx + intercept
    ss_res = float(np.sum(W * (ly - pred) ** 2))
    ss_tot = float(np.sum(W * (ly - np.average(ly, weights=W)) ** 2))
    return PowerLaw(exponent=float(slope), coef=float(np.exp(intercept)),
                    r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    n_points=int(lx.size), weighted=w is not None)


def bootstrap_power_law(rungs: Sequence[RungFit], value: str = "n",
                        n_draws: int = 200, weighted: bool = True,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[PowerLaw, np.ndarray]:
    """Refit the law on bootstrap index i of every rung, as they do.

    `value` is 'n' (N*) or 'multiplier' (D*/N*). Returns the point fit with a
    percentile CI attached, plus the raw exponent draws.
    """
    good = [r for r in rungs if r.usable]
    if len(good) < 2:
        raise ValueError(f"need two usable rungs, got {len(good)}")
    rng = rng or np.random.default_rng(0)
    budgets = np.array([r.budget for r in good])
    sig = np.array([r.log_sigma for r in good])

    def series(pick) -> np.ndarray:
        return np.array([pick(r) for r in good])

    if value == "n":
        obs = series(lambda r: r.n_star_median)
        draw = lambda r, i: r.samples[i % r.samples.size]
    elif value == "multiplier":
        obs = series(lambda r: r.d_star_median / r.n_star_median)
        draw = lambda r, i: r.d_of(r.samples[i % r.samples.size]) / r.samples[i % r.samples.size]
    else:
        raise ValueError(f"unknown value {value!r}")

    point = power_law_fit(budgets, obs, sig if weighted else None)
    exps = []
    for i in range(n_draws):
        ys = np.array([draw(r, i) for r in good])
        exps.append(power_law_fit(budgets, ys, sig if weighted else None).exponent)
    exps = np.asarray(exps)
    point.ci_low, point.ci_high = (float(np.percentile(exps, 2.5)),
                                   float(np.percentile(exps, 97.5)))
    return point, exps


# ------------------------------------------------------- saturating fit ----
@dataclass
class SaturatingFit:
    """`L(C) = logaddexp(a - alpha*log C, e)`: a power law plus a floor.

    `identified` is False when the optimiser drives the floor to zero or its
    standard error cannot be estimated. That is a real outcome, not an error:
    with only a handful of budgets on the descending branch, a three-parameter
    saturating model is not identifiable, and reporting `floor = 0` as though
    it were measured would be worse than reporting nothing.
    """
    a: float
    alpha: float
    floor: float
    rmse: float
    n_points: int
    identified: bool
    log_floor_se: Optional[float] = None
    note: str = ""

    def __call__(self, c):
        return np.logaddexp(self.a - self.alpha * np.log(np.asarray(c, float)),
                            np.log(max(self.floor, 1e-12)))


def saturating_fit(budgets: Sequence[float], values: Sequence[float],
                   descending_only: bool = True,
                   min_points: int = 4) -> SaturatingFit:
    """Their `fit_loss_with_saturation`, in the metric's own units.

    `floor` is the irreducible component: the value no further compute buys
    past. For DeltaE that is the direct quantitative test of the pool-floor
    hypothesis.

    The model is monotonically decreasing, so with `descending_only` the fit
    uses the budgets up to and including the minimum. INDIGO's curve turns
    upward afterwards and no saturating power law can describe that; silently
    fitting the whole range would hide the turn rather than measure it.
    """
    c = np.asarray(budgets, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(c)
    c, v = c[order], v[order]
    note = ""
    if descending_only:
        k = int(np.argmin(v)) + 1
        if k < c.size:
            note = (f"fit on the {k} descending budgets; the curve turns upward "
                    f"at C = {c[k - 1]:.3g} and the saturating form cannot "
                    f"represent that")
        c, v = c[:k], v[:k]
    if c.size < min_points:
        return SaturatingFit(float("nan"), float("nan"), float("nan"),
                             float("nan"), int(c.size), False, None,
                             f"only {c.size} usable budgets, need {min_points}")

    def model(cc, a, alpha, e):
        return np.logaddexp(a - alpha * np.log(cc), e)

    p0 = (float(np.log(v.max()) + 0.5 * np.log(c.min())), 0.05,
          float(np.log(max(v.min() * 0.8, 1e-6))))
    try:
        popt, pcov = curve_fit(model, c, v, p0=p0, maxfev=400_000)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        return SaturatingFit(float("nan"), float("nan"), float("nan"),
                             float("nan"), int(c.size), False, None,
                             f"{type(exc).__name__}: {exc}")
    floor = float(np.exp(popt[2]))
    resid = v - model(c, *popt)
    se = None
    if np.all(np.isfinite(pcov)):
        se = float(np.sqrt(np.diag(pcov))[2])
    # Not identified if the floor collapsed toward zero relative to the data,
    # or the curvature needed to pin it is below the noise.
    identified = bool(se is not None and np.isfinite(se)
                      and floor > 0.01 * float(v.min()))
    if not identified and not note:
        note = ("floor not identified: too few budgets on the descending "
                "branch to separate the power law from its asymptote")
    elif not identified:
        note += "; floor itself not identified"
    return SaturatingFit(a=float(popt[0]), alpha=float(popt[1]), floor=floor,
                         rmse=float(np.sqrt(np.mean(resid ** 2))),
                         n_points=int(c.size), identified=identified,
                         log_floor_se=se, note=note)


# --------------------------------------------------- hyperparameter laws ----
def tuned_optimum(grid_values: Sequence[float], losses: Sequence[float],
                  interp_multiplier: int = INTERP_MULTIPLIER
                  ) -> Tuple[float, bool]:
    """Their `minimize_with_interp`, for one config's hyperparameter sweep.

    Returns (optimum, on_edge). `on_edge` is True when the interpolated argmin
    falls outside the second and second-to-last *grid* points, which is their
    test for "this sweep did not bracket its own optimum" and the exact
    condition INDIGO's d512 learning-rate measurement failed.
    """
    x = np.asarray(grid_values, dtype=float)
    y = np.asarray(losses, dtype=float)
    keep: Dict[float, float] = {}
    for xi, yi in zip(x, y):
        if xi not in keep or yi < keep[xi]:
            keep[xi] = yi
    xs = np.array(sorted(keep))
    ys = np.array([keep[v] for v in xs])
    if xs.size < 3:
        # Two points cannot bracket anything; report the better and say so.
        return float(xs[int(np.argmin(ys))]), True
    grid, y_grid, idx = akima_argmin(xs, ys, interp_multiplier)
    best = float(grid[idx])
    return best, bool(best < xs[1] or best > xs[-2])
