"""
IsoFLOP sweep grid: which (N, D) to run at each compute budget
==============================================================

Phase 0.2. Turns a list of compute budgets into concrete, runnable configs,
and refuses to emit ones that cannot finish.

What sets the grid
------------------
1. N*(C) prior. Chinchilla's 20 tokens/param, converted to examples via the
   ~5.5 supervised targets per example (mean structure length + EOS), and
   through THIS architecture's FLOP model rather than 6ND -- INDIGO spends
   212-306 train FLOPs per (param, example), not 6 (see src/scaling/flops.py).
   The prior only has to put the bracket near the minimum; the sweep measures
   where it actually is.

2. Achievable N is quantised. N comes from (d_model, slot_encoder_layers),
   with d_model a multiple of 64 so n_heads = d_model/64 divides it evenly.
   Targets snap to the nearest achievable point, so realised multiples drift
   from the nominal bracket -- what matters is that the sampled Ns straddle
   the minimum, not that they hit exact ratios.

3. Learning rate per scale (Porian correction #3), from a power law fitted to
   the measured LR sweeps rather than a fresh sweep per config. CAVEAT: that
   law rests on three points, two of which landed on the same value because
   the 3-point grid was coarse, so the exponent is weakly determined. It is
   still far better than one global LR -- 6e-5 (tuned for the 69.6M
   production model) left every probe model untrained.

4. Wall-clock feasibility. --qos=short enforces 3h whatever --time says. At
   the measured 5224 ex/s and 1.75s per shard read, that caps a config at
   ~16M example-passes. The low-N corner of a rung needs the most data, so it
   is what limits the top budget -- and a config that cannot finish is worse
   than one not attempted, because it silently removes a point from the
   parabola.

Because of (4) the reachable top budget is well below the production run's
~1e17. Raising it needs a longer QoS (each config is then hours, not minutes)
or a faster data path; neither is a code change here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.model import ModelConfig
from src.scaling.flops import (
    n_params,
    train_flops_per_example,
    tokens_per_example,
    DEFAULT_MEAN_LAYERS,
)

# ---------------------------------------------------------------------------
# Measured constants. Every one of these came off a real run; none is guessed.
# ---------------------------------------------------------------------------

# End-to-end throughput and overhead, REFIT on the 20 completed sweep-v1
# runs (Sept 22, job 3998482) by least squares of elapsed vs passes:
#
#     elapsed_sec = 1712 + passes / 1985 + delta_e_eval
#
# The v1 constants came from probe run 3's steady-state step timing at
# d_model=512 (5224 ex/s) plus an estimated 480 s of startup. Both were badly
# optimistic end to end: v1 configs ran up to 2.05x their predicted wall, and
# the sweep only survived because the grid was conservative for other reasons.
# Two things the step-timing figure left out -- fixed startup is nearly half
# an hour once module load, torch and JAX imports, the shard scan, the
# validation read and the simulator preflight are counted, and these small
# models are input-bound rather than GPU-bound, so they run SLOWER per example
# than the much larger d512 probe did.
#
# The rate below is the 10th-percentile per-config figure, not the median:
# measured throughput ranged 1162-2820 ex/s across the 20 runs (node
# contention), and a grid sized on the median would put its slowest configs
# over the wall. At p10 the worst v1 run would have come in at 1.08x its
# prediction instead of 2.05x.
EXAMPLES_PER_SEC = 1301.0
STARTUP_SEC = 1712.0
ROWS_PER_SHARD = 5000
# Shard-read cost is no longer a separate term: the empirical slope above was
# fitted on shard-aligned runs and already contains it. Keeping both would
# double-count.
SEC_PER_SHARD_READ = 0.0

# DeltaE eval, remeasured on sweep v1 at the real setting: median 146 s for
# 2048 examples. Every config pays this exactly once (scripts/scaling_sweep.py
# makes the epoch-boundary eval unreachable so only the forced "final" one
# runs). This is the one v1 estimate that held up.
SEC_PER_DE_EXAMPLE = 146.0 / 2048.0

# Examples scored per DeltaE eval. 512 splits into only ~170 per chroma
# bucket, and a median over 170 is noisy enough that a synthetic replay of
# this exact grid with 0.25 dE of jitter lost the low bucket entirely and
# could no longer separate the buckets' exponents -- which is the study's
# novel claim. 2048 gives ~680 per bucket for ~168 s per config, which the
# wall budget absorbs comfortably now that the eval runs once.
DE_EXAMPLES = 2048
# --qos=short enforces this regardless of --time (CLAUDE.md).
QOS_SHORT_SEC = 3 * 3600
# 0.80, down from v1's 0.85. EXAMPLES_PER_SEC is already the p10 across 20
# runs, but the slowest run of v1 was another 12% under even that, so a config
# planned at the cap could still overrun. The margin covers that tail: a
# timed-out config is not merely wasted, it deletes a point from its parabola
# and biases the fitted minimum, and the epoch-boundary-only resume cannot
# recover it because most configs are a single epoch.
WALL_MARGIN = 0.80

CORPUS_EXAMPLES = 9_997_312

# Epoch ceiling. The probe found NO DETECTABLE degradation up to 8 passes at
# any size tested, against a ~4.9 dE noise floor. That is a non-detection, not
# a verified safe depth: a smaller effect would have been invisible. 8 is used
# because it is what was actually examined.
EPOCH_CEILING = 8.0

# Measured LR optima (scripts/lr_tuning.py, selected on DeltaE, diverged
# trials excluded). d512's value is the one that survived the divergence
# screen; its grid neighbour at 5.48e-4 blew up to val_loss 4570.
MEASURED_LR: List[Tuple[int, float]] = [
    (767_013, 5.477e-4),     # d_model=128, slot_encoder_layers=2
    (2_869_029, 5.477e-4),   # d_model=256, slot_encoder_layers=2
    (17_506_597, 1.0e-4),    # d_model=512, slot_encoder_layers=4
]

# Sweep v1 (Sept 22) used a multiplicative bracket of (0.6 ... 1.7) x the
# Chinchilla prior. That span is 2.8x in N, and every one of its 20 IsoFLOP
# points came back either monotone or concave: three of four pooled parabolas
# had a <= 0 and the study could not fit a single exponent. Two causes, both
# fixed below.
#
#   (a) 2.8x is too narrow to show curvature. Chinchilla's own IsoFLOP slices
#       span more than an order of magnitude. DEFAULT_SPAN is now 10x.
#   (b) Worse, N was confounded with SHAPE. See ASPECT_* below.
DEFAULT_SPAN = 10.0
DEFAULT_POINTS = 6
DEFAULT_BATCH_SIZE = 256

# Aspect ratio (d_model / slot_encoder_layers), and the reason v1 failed.
#
# v1 quantised d_model to multiples of 64, which makes the smallest width
# steps 3.3x and 2.2x in N. Depth was therefore the only knob fine enough to
# hit a target N, so within a budget "larger N" nearly always meant "deeper at
# the SAME width": at C=1e14 four of five points were d_model=64 with depth
# 2, 3, 4, 7, an aspect ratio sweeping 32 down to 9.1.
#
# The sweep measured that, not capacity. Within-budget Spearman rho between
# log N and val_de came out +0.70, +0.50, +0.20 at the three lower budgets:
# bigger models were WORSE, monotonically, because they were narrower-per-
# layer, not because they were past N*. An IsoFLOP parabola cannot survive
# that; a monotone or concave series has no interior minimum to report.
#
# Fix: hold aspect ratio ~constant so width and depth grow together, which is
# standard practice for scaling ladders, and let head_dim drop to 32 so width
# moves in fine enough steps that depth does not have to absorb the residual.
# The band is centred where this sweep's best configs actually landed: the
# per-budget winners had aspect 32, 64, 42.7, 16, while the per-budget losers
# were the extremes (9.1 and 320).
ASPECT_MIN = 28.0
ASPECT_MAX = 72.0
ASPECT_TARGET = 48.0

# Head dim 32, not 64. n_heads is exactly param- and FLOP-neutral
# (src/scaling/flops.py), so this cannot perturb the compute axis, and it
# halves the width quantum: d_model steps by 32 instead of 64, which is what
# lets the ladder hold aspect ratio fixed. Note this differs from production
# (d_model=512, head_dim 64); internal consistency across the ladder matters
# more here than matching one checkpoint, since every point is compared to
# other points in the same ladder.
HEAD_DIM = 32


def fit_lr_law(points: Sequence[Tuple[int, float]] = tuple(MEASURED_LR)
               ) -> Tuple[float, float]:
    """Fit log(lr) = a + b*log(N). Returns (a, b)."""
    n = np.log(np.array([p[0] for p in points], dtype=float))
    lr = np.log(np.array([p[1] for p in points], dtype=float))
    b, a = np.polyfit(n, lr, 1)
    return float(a), float(b)


def lr_for(n: float, law: Optional[Tuple[float, float]] = None) -> float:
    a, b = law if law is not None else fit_lr_law()
    return float(math.exp(a) * n ** b)


def n_heads_for(d_model: int) -> int:
    """Head count under the ladder's fixed-head-dim policy.

    Single source of truth: SweepConfig carries the result, so the planner and
    the emitted training command cannot drift apart on model shape. They did
    drift in v1, where two call sites each open-coded d_model // 64.
    """
    return max(1, d_model // HEAD_DIM)


def achievable_sizes(
    d_models: Sequence[int] = tuple(HEAD_DIM * k for k in range(1, 33)),
    depths: Sequence[int] = tuple(range(1, 13)),
    decoder_layers: int = 1,
    aspect_min: float = ASPECT_MIN,
    aspect_max: float = ASPECT_MAX,
    aspect_target: float = ASPECT_TARGET,
) -> List[Tuple[int, int, int]]:
    """The (n_params, d_model, slot_encoder_layers) ladder, at ~fixed shape.

    Only shapes whose aspect ratio falls in [aspect_min, aspect_max] are
    admissible, so moving along this ladder moves capacity rather than
    geometry. Where several shapes land on the same n_params, the one nearest
    aspect_target wins, which keeps the ladder as close to iso-shape as the
    integer grid allows.
    """
    best: dict = {}
    for d in d_models:
        for sel in depths:
            aspect = d / sel
            if not aspect_min <= aspect <= aspect_max:
                continue
            cfg = ModelConfig(head_mode="cross_attn", d_model=d,
                              n_heads=n_heads_for(d), slot_encoder_layers=sel,
                              decoder_layers=decoder_layers)
            n = n_params(cfg)
            score = abs(math.log(aspect / aspect_target))
            if n not in best or score < best[n][0]:
                best[n] = (score, d, sel)
    return sorted((n, d, sel) for n, (_, d, sel) in best.items())


def chinchilla_n_star(budget: float,
                      mean_layers: float = DEFAULT_MEAN_LAYERS) -> float:
    """N* under the 20-tokens-per-param prior, in THIS FLOP model.

    C = k(N) * N * D with k(N) ~ 222 train FLOPs per (param, example), and
    D = rho * N with rho = 20 / tokens_per_example. Only a starting bracket.
    """
    rho = 20.0 / tokens_per_example(mean_layers)
    return math.sqrt(budget / (222.0 * rho))


@dataclass
class SweepConfig:
    budget: float
    target_n: float
    n_params: int
    d_model: int
    slot_encoder_layers: int
    n_heads: int
    lr: float
    batch_size: int
    passes: int
    steps: int
    epochs: int
    limit_examples: int
    epochs_over_corpus: float
    est_wall_sec: float
    nominal_multiple: float
    seed: Optional[int] = None

    @property
    def aspect(self) -> float:
        return self.d_model / self.slot_encoder_layers

    @property
    def name(self) -> str:
        return (f"sweep_C{self.budget:.2e}_d{self.d_model}"
                f"_se{self.slot_encoder_layers}".replace("+", ""))

    @property
    def realised_multiple(self) -> float:
        return self.n_params / self.target_n * self.nominal_multiple

    @property
    def fits_qos_short(self) -> bool:
        return self.fits_wall(QOS_SHORT_SEC * WALL_MARGIN)

    def fits_wall(self, cap_sec: Optional[float] = None) -> bool:
        """Does this config finish inside `cap_sec`? Defaults to --qos=short.

        Separate from fits_qos_short so a longer QoS does not have to be
        smuggled past a hardcoded 3h check: the dispatcher refuses on the cap
        it was actually given.
        """
        if cap_sec is None:
            cap_sec = QOS_SHORT_SEC * WALL_MARGIN
        return self.est_wall_sec <= cap_sec

    @property
    def within_epoch_ceiling(self) -> bool:
        return self.epochs_over_corpus <= EPOCH_CEILING


def estimate_wall_sec(passes: int, de_examples: int = DE_EXAMPLES) -> float:
    """Wall clock for one config: startup + training + shard I/O + one DeltaE eval.

    The DeltaE term is small but not ignorable, and leaving it out would let a
    bigger --limit-de-examples silently eat the feasibility margin instead of
    tightening the grid.
    """
    return (STARTUP_SEC + passes / EXAMPLES_PER_SEC
            + (passes / ROWS_PER_SHARD) * SEC_PER_SHARD_READ
            + de_examples * SEC_PER_DE_EXAMPLE)


def wall_model_residuals(observations: Sequence[Tuple[int, float]]
                         ) -> List[Tuple[int, float, float, float]]:
    """(passes, actual_sec) -> (passes, actual, predicted, ratio), for auditing.

    v1's wall model was wrong by 2x and nothing caught it until the runs were
    already in. Feeding a finished sweep's elapsed times back through this is
    how the constants above get re-checked before the next one is sized.
    """
    return [(p, a, estimate_wall_sec(p), a / estimate_wall_sec(p))
            for p, a in observations]


def feasible_n_window(
    budget: float,
    sizes: Sequence[Tuple[int, int, int]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    wall_cap_sec: Optional[float] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Smallest and largest ladder N whose run fits the wall cap at this budget.

    At fixed C, a SMALLER model needs MORE passes (D = C / flops_per_example),
    so the wall constraint bites at the low-N end and the high-N end is free.
    v1 got this backwards in effect: it pinned the bracket to the Chinchilla
    prior and then declared budgets above 5.5e15 unreachable, when in fact the
    ladder simply has to start higher at higher C.
    """
    cap = wall_cap_sec if wall_cap_sec is not None else QOS_SHORT_SEC * WALL_MARGIN
    lo = hi = None
    for n, d, sel in sizes:
        cfg = ModelConfig(head_mode="cross_attn", d_model=d,
                          n_heads=n_heads_for(d), slot_encoder_layers=sel,
                          decoder_layers=1, batch_size=batch_size)
        passes = budget / train_flops_per_example(cfg)
        if passes < batch_size:
            continue                      # fewer than one step
        if estimate_wall_sec(int(passes)) <= cap:
            if lo is None:
                lo = n
            hi = n
    return lo, hi


def build_grid(
    budgets: Sequence[float],
    span: float = DEFAULT_SPAN,
    points: int = DEFAULT_POINTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    corpus: int = CORPUS_EXAMPLES,
    lr_law: Optional[Tuple[float, float]] = None,
    wall_cap_sec: Optional[float] = None,
    repeat_seed: Optional[int] = None,
) -> List[SweepConfig]:
    """One IsoFLOP rung per budget, `points` sizes spanning `span` x in N.

    The window is centred on the Chinchilla prior when it can be, and slid
    (never narrowed) to stay inside the wall-clock window otherwise, so every
    rung keeps the full span needed for curvature to be visible.

    `repeat_seed` re-runs each rung's middle size under a second seed. Those
    pairs are the only way to put an error bar on a fitted N*: v1 had no
    repeats, so when its per-budget val_de spread fell to 1.04 units at the
    top rung there was no way to tell signal from eval noise.
    """
    law = lr_law or fit_lr_law()
    sizes = achievable_sizes()
    grid: List[SweepConfig] = []

    def make(budget: float, target: float, n: int, d: int, sel: int,
             mult: float, seed: Optional[int]) -> SweepConfig:
        cfg = ModelConfig(head_mode="cross_attn", d_model=d,
                          n_heads=n_heads_for(d), slot_encoder_layers=sel,
                          decoder_layers=1, batch_size=batch_size)
        per_ex = train_flops_per_example(cfg)
        steps = max(1, int(round(budget / per_ex)) // batch_size)
        passes = steps * batch_size          # realise the rounding

        # training.py derives total_steps as steps_per_epoch * epochs, with
        # steps_per_epoch = limit_examples / batch_size. To land on exactly
        # `steps`, split the passes into whole epochs over a subset small
        # enough to fit the corpus, then size the subset so the product comes
        # back to `steps`. Epochs must divide steps exactly or C drifts.
        epochs = max(1, math.ceil(passes / corpus))
        while steps % epochs and epochs < steps:
            epochs += 1
        return SweepConfig(
            budget=budget, target_n=target, n_params=n, d_model=d,
            slot_encoder_layers=sel, n_heads=n_heads_for(d),
            lr=lr_for(n, law), batch_size=batch_size, passes=passes,
            steps=steps, epochs=epochs,
            limit_examples=(steps // epochs) * batch_size,
            epochs_over_corpus=passes / corpus,
            est_wall_sec=estimate_wall_sec(passes),
            nominal_multiple=mult, seed=seed,
        )

    for budget in budgets:
        floor_n, ceil_n = feasible_n_window(budget, sizes, batch_size, wall_cap_sec)
        if floor_n is None:
            continue                      # no size at this budget fits the wall
        n_star = chinchilla_n_star(budget)
        lo = n_star / math.sqrt(span)
        hi = n_star * math.sqrt(span)
        # Slide, do not shrink: a narrowed rung is what killed v1.
        if lo < floor_n:
            lo, hi = float(floor_n), float(floor_n) * span
        if hi > ceil_n:
            hi, lo = float(ceil_n), float(ceil_n) / span
        lo = max(lo, float(floor_n))

        chosen: List[Tuple[int, int, int, float]] = []
        seen = set()
        for k in range(points):
            frac = k / (points - 1) if points > 1 else 0.5
            target = lo * (hi / lo) ** frac
            in_window = [t for t in sizes if floor_n <= t[0] <= ceil_n]
            if not in_window:
                continue
            n, d, sel = min(in_window,
                            key=lambda t: abs(math.log(t[0] / target)))
            if n in seen:
                continue                  # two targets snapped to one size
            seen.add(n)
            chosen.append((n, d, sel, target / n_star))

        for n, d, sel, mult in chosen:
            grid.append(make(budget, mult * n_star, n, d, sel, mult, None))
        if repeat_seed is not None and chosen:
            n, d, sel, mult = chosen[len(chosen) // 2]
            grid.append(make(budget, mult * n_star, n, d, sel, mult, repeat_seed))
    return grid


def build_data_ladder(
    d_model: int = 128,
    slot_encoder_layers: int = 3,
    points: int = 6,
    min_passes: int = 1_000_000,
    max_passes: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    corpus: int = CORPUS_EXAMPLES,
    lr_law: Optional[Tuple[float, float]] = None,
    wall_cap_sec: Optional[float] = None,
) -> List[SweepConfig]:
    """One model shape, D swept log-spaced: does a SMALL model keep improving?

    This is not an IsoFLOP rung and must not be fitted as one -- every point
    is its own budget, so a parabola in log N does not exist here. It answers
    a different question, the one sweep v1 raised by accident: its best run
    was d128/se3 at 0.97M params reaching val_de 9.56, within ~1 dE of the
    production checkpoint at 18-72x the parameters and ~19-70x the compute.
    Whether that is a saturation point or still climbing decides whether the
    model or the data is the binding constraint on this task.

    The IsoFLOP grid cannot answer it, because at fixed C a smaller model
    needs MORE passes and so the wall clock sets a floor on N: under
    --qos=short a 0.97M model tops out at 8.8M passes, which is less than one
    epoch of the corpus. The default here is that shape, so the ladder starts
    from the exact configuration that produced the result.
    """
    law = lr_law or fit_lr_law()
    cap = wall_cap_sec if wall_cap_sec is not None else QOS_SHORT_SEC * WALL_MARGIN
    cfg0 = ModelConfig(head_mode="cross_attn", d_model=d_model,
                       n_heads=n_heads_for(d_model),
                       slot_encoder_layers=slot_encoder_layers,
                       decoder_layers=1, batch_size=batch_size)
    per_ex = train_flops_per_example(cfg0)
    n = n_params(cfg0)

    if max_passes is None:
        # Largest D whose wall still fits, then held under the repeat depth
        # the epoch-ceiling probe actually examined.
        budget_sec = cap - STARTUP_SEC - DE_EXAMPLES * SEC_PER_DE_EXAMPLE
        max_passes = int(budget_sec * EXAMPLES_PER_SEC)
    max_passes = min(max_passes, int(EPOCH_CEILING * corpus))
    if max_passes <= min_passes:
        return []

    out: List[SweepConfig] = []
    seen = set()
    for k in range(points):
        frac = k / (points - 1) if points > 1 else 1.0
        passes = int(min_passes * (max_passes / min_passes) ** frac)
        steps = max(1, passes // batch_size)

        # Unlike build_grid, hold EPOCHS at the minimum and round STEPS to a
        # multiple of it, rather than holding steps exact and raising epochs
        # until one divides. build_grid's rule keeps C on its rung to the
        # example, which an IsoFLOP point needs. Here it would defeat the
        # experiment: at 42.5M passes it lands on 8 epochs over a 5.3M subset,
        # when the question is what MORE DATA does. Minimum epochs over the
        # largest subset that fits gives 5 passes over 8.5M instead, and the
        # cost is a sub-0.5% shift in D, which no conclusion here turns on.
        epochs = max(1, math.ceil(steps * batch_size / corpus))
        steps = max(epochs, round(steps / epochs) * epochs)
        passes = steps * batch_size
        if steps in seen:
            continue
        seen.add(steps)
        out.append(SweepConfig(
            budget=per_ex * passes, target_n=float(n), n_params=n,
            d_model=d_model, slot_encoder_layers=slot_encoder_layers,
            n_heads=n_heads_for(d_model), lr=lr_for(n, law),
            batch_size=batch_size, passes=passes, steps=steps, epochs=epochs,
            limit_examples=(steps // epochs) * batch_size,
            epochs_over_corpus=passes / corpus,
            est_wall_sec=estimate_wall_sec(passes), nominal_multiple=1.0,
        ))
    return out


def describe_data_ladder(ladder: Sequence[SweepConfig],
                         corpus: int = CORPUS_EXAMPLES) -> str:
    if not ladder:
        return "data ladder is empty: max_passes <= min_passes at this wall cap."
    c0 = ladder[0]
    lines = [
        f"FIXED-N DATA LADDER  d_model={c0.d_model} "
        f"slot_encoder_layers={c0.slot_encoder_layers}  "
        f"N={c0.n_params:,}  lr={c0.lr:.3e}",
        "",
        "Not an IsoFLOP rung. One shape, more data, to see whether a ~1M-param",
        "model saturates or keeps improving. Fit nothing in log N from this.",
        "",
        f"{'D (passes)':>13} {'steps':>9} {'epochs':>7} {'implied C':>11} {'wall':>7}",
        "-" * 54,
    ]
    for c in ladder:
        lines.append(f"{c.passes:>13,} {c.steps:>9,} {c.epochs_over_corpus:>7.2f} "
                     f"{c.budget:>11.2e} {c.est_wall_sec / 3600:>6.2f}h")
    lines += [
        "-" * 54,
        f"{len(ladder)} runs, {sum(c.est_wall_sec for c in ladder) / 3600:.1f} "
        f"GPU-hours, {ladder[-1].passes / ladder[0].passes:.0f}x data span, "
        f"deepest {ladder[-1].epochs_over_corpus:.2f} epochs "
        f"(probe examined up to {EPOCH_CEILING:.0f})",
    ]
    return "\n".join(lines)


def max_feasible_budget(span: float = DEFAULT_SPAN,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        wall_cap_sec: Optional[float] = None) -> float:
    """Largest budget whose feasible N window is still `span` wide.

    Not the same question v1 asked. v1 pinned the rung to the Chinchilla prior
    and asked when that prior's LOW corner stopped fitting, which answered
    5.5e15. With a sliding window the rung simply starts higher at higher C,
    so what actually runs out is SPAN: eventually the smallest model that fits
    the wall clock is within `span` of the largest useful one, and the rung can
    no longer show curvature.
    """
    sizes = achievable_sizes()
    lo, hi = 1e12, 1e19
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        floor_n, ceil_n = feasible_n_window(mid, sizes, batch_size, wall_cap_sec)
        ok = (floor_n is not None and ceil_n is not None
              and ceil_n >= floor_n * span)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


def describe(grid: Sequence[SweepConfig], corpus: int = CORPUS_EXAMPLES,
             wall_cap_sec: Optional[float] = None) -> str:
    a, b = fit_lr_law()
    lines = [
        f"LR law: lr(N) = {math.exp(a):.3e} * N^{b:.4f}   "
        f"(fit to {len(MEASURED_LR)} measured points; exponent weakly "
        f"determined -- see module docstring)",
        "",
        f"{'config':<30} {'N':>11} {'asp':>5} {'lr':>9} {'D (passes)':>12} "
        f"{'steps':>8} {'ep':>6} {'wall':>7} {'mult':>6}",
        "-" * 108,
    ]
    infeasible = []
    for c in grid:
        flag = ""
        if not c.fits_wall(wall_cap_sec):
            flag += "  OVER-WALL"
            infeasible.append(c)
        if not c.within_epoch_ceiling:
            flag += "  OVER-EPOCH-CEILING"
        tag = c.name + (f"_s{c.seed}" if c.seed is not None else "")
        lines.append(
            f"{tag:<30} {c.n_params:>11,} {c.aspect:>5.1f} {c.lr:>9.2e} "
            f"{c.passes:>12,} {c.steps:>8,} {c.epochs_over_corpus:>6.2f} "
            f"{c.est_wall_sec / 3600:>6.2f}h {c.realised_multiple:>6.2f}{flag}")
    lines.append("-" * 108)

    by_budget: Dict[float, List[SweepConfig]] = {}
    for c in grid:
        by_budget.setdefault(c.budget, []).append(c)
    lines.append("")
    lines.append("per-budget coverage (the parabola needs the minimum "
                 "straddled, and needs SPAN to show curvature):")
    for budget, group in sorted(by_budget.items()):
        ns = sorted({c.n_params for c in group})
        star = chinchilla_n_star(budget)
        below = sum(1 for n in ns if n < star)
        asp = sorted(c.aspect for c in group)
        lines.append(
            f"  C={budget:.2e}: N* prior {star / 1e6:6.2f}M, sampled "
            f"{len(ns)} sizes {ns[0] / 1e6:.2f}-{ns[-1] / 1e6:.2f}M "
            f"= {ns[-1] / ns[0]:.1f}x span ({below} below / {len(ns) - below} above), "
            f"aspect {asp[0]:.0f}-{asp[-1]:.0f}"
            + ("   <-- ONE-SIDED, minimum not bracketed"
               if below == 0 or below == len(ns) else ""))

    lines.append("")
    lines.append("realised-vs-requested check (steps_per_epoch * epochs must "
                 "equal steps, and limit_examples must fit the corpus):")
    bad = 0
    for c in grid:
        realised = (c.limit_examples // c.batch_size) * c.epochs
        ok = realised == c.steps and c.limit_examples <= corpus
        if not ok:
            bad += 1
            lines.append(f"  MISMATCH {c.name}: realised {realised} steps vs "
                         f"{c.steps} requested, limit={c.limit_examples:,}")
    lines.append(f"  {len(grid) - bad}/{len(grid)} configs realise their "
                 f"requested budget exactly"
                 + ("" if bad == 0 else f"   <-- {bad} MISMATCHED"))

    total_wall = sum(c.est_wall_sec for c in grid)
    lines += [
        "",
        f"{len(grid)} configs, {total_wall / 3600:.1f} GPU-hours total, "
        f"longest single config {max(c.est_wall_sec for c in grid) / 3600:.2f}h",
        f"largest budget whose feasible N window still spans "
        f"{DEFAULT_SPAN:.0f}x: {max_feasible_budget():.2e}",
    ]
    if infeasible:
        lines.append(
            f"\n{len(infeasible)} config(s) EXCEED the 3h --qos=short cap. A "
            f"config that cannot finish silently removes a point from its "
            f"parabola, so either lower the top budget, reduce --span, "
            f"or obtain a longer QoS.")
    return "\n".join(lines)


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="IsoFLOP sweep grid")
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[1e14, 3.7e14, 1.4e15, 5e15])
    p.add_argument("--span", type=float, default=DEFAULT_SPAN)
    p.add_argument("--points", type=int, default=DEFAULT_POINTS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--repeat-seed", type=int, default=None)
    p.add_argument("--max-wall-hours", type=float, default=None,
                   help="override the 3h --qos=short cap when a longer QoS is available")
    args = p.parse_args()
    cap = args.max_wall_hours * 3600 * WALL_MARGIN if args.max_wall_hours else None
    print(describe(build_grid(args.budgets, span=args.span, points=args.points,
                              batch_size=args.batch_size,
                              wall_cap_sec=cap, repeat_seed=args.repeat_seed)))


if __name__ == "__main__":
    _main()
