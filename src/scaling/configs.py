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

# Steady-state throughput, probe run 3 (d_model=512, bs=256, shard-aligned):
# 47-49 ms/step.
EXAMPLES_PER_SEC = 5224.0
# Cost of pulling one ~140MB parquet shard, from the epoch-start excess in
# arm 11 (16 shards, ~28s over steady state).
SEC_PER_SHARD_READ = 1.75
ROWS_PER_SHARD = 5000
# scan_files (cached) + one validation-split read.
STARTUP_SEC = 480.0

# DeltaE eval, from the probe's "[Val] ... (NNs)" lines: 512 examples took
# 34-42 s across d128/d256/d512. 42/512 is the conservative end, and d512 is
# larger than anything in this grid, so it bounds us. Every sweep config pays
# this exactly once (scripts/scaling_sweep.py makes the epoch-boundary eval
# unreachable so only the forced "final" one runs).
SEC_PER_DE_EXAMPLE = 42.0 / 512.0

# Examples scored per DeltaE eval. 512 splits into only ~170 per chroma
# bucket, and a median over 170 is noisy enough that a synthetic replay of
# this exact grid with 0.25 dE of jitter lost the low bucket entirely and
# could no longer separate the buckets' exponents -- which is the study's
# novel claim. 2048 gives ~680 per bucket for ~168 s per config, which the
# wall budget absorbs comfortably now that the eval runs once.
DE_EXAMPLES = 2048
# --qos=short enforces this regardless of --time (CLAUDE.md).
QOS_SHORT_SEC = 3 * 3600
WALL_MARGIN = 0.85

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

DEFAULT_BRACKET: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.3, 1.7)
DEFAULT_BATCH_SIZE = 256


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


def achievable_sizes(
    d_models: Sequence[int] = tuple(64 * k for k in range(1, 17)),
    depths: Sequence[int] = tuple(range(1, 9)),
    decoder_layers: int = 1,
) -> List[Tuple[int, int, int]]:
    """Every (n_params, d_model, slot_encoder_layers) the grid can request.

    d_model is a multiple of 64 so n_heads = d_model/64 always divides it
    (head dim stays 64, the literature-standard value; n_heads is exactly
    param- and FLOP-neutral, so this choice cannot perturb the compute axis).
    """
    out = {}
    for d in d_models:
        for sel in depths:
            cfg = ModelConfig(head_mode="cross_attn", d_model=d,
                              n_heads=max(1, d // 64), slot_encoder_layers=sel,
                              decoder_layers=decoder_layers)
            out[n_params(cfg)] = (d, sel)
    return sorted((n, d, sel) for n, (d, sel) in out.items())


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
    lr: float
    batch_size: int
    passes: int
    steps: int
    epochs: int
    limit_examples: int
    epochs_over_corpus: float
    est_wall_sec: float
    nominal_multiple: float

    @property
    def name(self) -> str:
        return (f"sweep_C{self.budget:.2e}_d{self.d_model}"
                f"_se{self.slot_encoder_layers}".replace("+", ""))

    @property
    def realised_multiple(self) -> float:
        return self.n_params / self.target_n * self.nominal_multiple

    @property
    def fits_qos_short(self) -> bool:
        return self.est_wall_sec <= QOS_SHORT_SEC * WALL_MARGIN

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


def build_grid(
    budgets: Sequence[float],
    bracket: Sequence[float] = DEFAULT_BRACKET,
    batch_size: int = DEFAULT_BATCH_SIZE,
    corpus: int = CORPUS_EXAMPLES,
    lr_law: Optional[Tuple[float, float]] = None,
) -> List[SweepConfig]:
    law = lr_law or fit_lr_law()
    sizes = achievable_sizes()
    grid: List[SweepConfig] = []
    for budget in budgets:
        n_star = chinchilla_n_star(budget)
        seen = set()
        for mult in bracket:
            target = mult * n_star
            n, d, sel = min(sizes, key=lambda s: abs(math.log(s[0]) - math.log(target)))
            if (budget, n) in seen:
                continue          # two multiples snapped to the same size
            seen.add((budget, n))
            cfg = ModelConfig(head_mode="cross_attn", d_model=d,
                              n_heads=max(1, d // 64), slot_encoder_layers=sel,
                              decoder_layers=1, batch_size=batch_size)
            per_ex = train_flops_per_example(cfg)
            passes = int(round(budget / per_ex))
            steps = max(1, passes // batch_size)
            passes = steps * batch_size          # realise the rounding

            # training.py derives total_steps as steps_per_epoch * epochs,
            # with steps_per_epoch = limit_examples / batch_size. To land on
            # exactly `steps`, split the passes into whole epochs over a
            # subset small enough to fit the corpus, then size the subset so
            # the product comes back to `steps`. Epochs must divide steps
            # exactly or the realised budget drifts off C.
            epochs = max(1, math.ceil(passes / corpus))
            while steps % epochs and epochs < steps:
                epochs += 1
            limit_examples = (steps // epochs) * batch_size

            grid.append(SweepConfig(
                budget=budget, target_n=target, n_params=n, d_model=d,
                slot_encoder_layers=sel, lr=lr_for(n, law),
                batch_size=batch_size, passes=passes, steps=steps,
                epochs=epochs, limit_examples=limit_examples,
                epochs_over_corpus=passes / corpus,
                est_wall_sec=estimate_wall_sec(passes),
                nominal_multiple=mult,
            ))
    return grid


def max_feasible_budget(bracket_min: float = min(DEFAULT_BRACKET),
                        batch_size: int = DEFAULT_BATCH_SIZE) -> float:
    """Largest budget whose LOW-N corner still fits --qos=short.

    The smallest model of a rung consumes the most data, so it is what limits
    the rung. Solved numerically against the same wall model the grid uses.
    """
    lo, hi = 1e12, 1e18
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        n_star = chinchilla_n_star(mid)
        cfg = ModelConfig(head_mode="cross_attn", d_model=256, n_heads=4,
                          slot_encoder_layers=2, decoder_layers=1,
                          batch_size=batch_size)
        # Approximate per-example cost at the corner via the ~222 factor.
        passes = mid / (222.0 * bracket_min * n_star)
        if estimate_wall_sec(int(passes)) <= QOS_SHORT_SEC * WALL_MARGIN:
            lo = mid
        else:
            hi = mid
    return lo


def describe(grid: Sequence[SweepConfig], corpus: int = CORPUS_EXAMPLES) -> str:
    a, b = fit_lr_law()
    lines = [
        f"LR law: lr(N) = {math.exp(a):.3e} * N^{b:.4f}   "
        f"(fit to {len(MEASURED_LR)} measured points; exponent weakly "
        f"determined -- see module docstring)",
        "",
        f"{'config':<30} {'N':>11} {'lr':>9} {'D (passes)':>12} {'steps':>8} "
        f"{'ep':>6} {'wall':>7} {'mult':>6}",
        "-" * 100,
    ]
    infeasible = []
    for c in grid:
        flag = ""
        if not c.fits_qos_short:
            flag += "  OVER-3h"
            infeasible.append(c)
        if not c.within_epoch_ceiling:
            flag += "  OVER-EPOCH-CEILING"
        lines.append(
            f"{c.name:<30} {c.n_params:>11,} {c.lr:>9.2e} {c.passes:>12,} "
            f"{c.steps:>8,} {c.epochs_over_corpus:>6.2f} "
            f"{c.est_wall_sec / 3600:>6.2f}h {c.realised_multiple:>6.2f}{flag}")
    lines.append("-" * 100)

    by_budget: Dict[float, List[SweepConfig]] = {}
    for c in grid:
        by_budget.setdefault(c.budget, []).append(c)
    lines.append("")
    lines.append("per-budget bracket coverage (the parabola needs the minimum "
                 "straddled):")
    for budget, group in sorted(by_budget.items()):
        ns = sorted(c.n_params for c in group)
        star = chinchilla_n_star(budget)
        below = sum(1 for n in ns if n < star)
        lines.append(
            f"  C={budget:.2e}: N* prior {star / 1e6:6.2f}M, sampled "
            f"{len(ns)} sizes {ns[0] / 1e6:.2f}-{ns[-1] / 1e6:.2f}M "
            f"({below} below / {len(ns) - below} above)"
            + ("   <-- ONE-SIDED, minimum not bracketed" if below == 0 or below == len(ns) else ""))

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
        f"max budget whose low-N corner still fits --qos=short: "
        f"{max_feasible_budget():.2e}",
    ]
    if infeasible:
        lines.append(
            f"\n{len(infeasible)} config(s) EXCEED the 3h --qos=short cap. A "
            f"config that cannot finish silently removes a point from its "
            f"parabola, so either lower the top budget, narrow the bracket, "
            f"or obtain a longer QoS.")
    return "\n".join(lines)


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="IsoFLOP sweep grid")
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[1e14, 3.7e14, 1.4e15, 5e15])
    p.add_argument("--bracket", type=float, nargs="+", default=list(DEFAULT_BRACKET))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = p.parse_args()
    print(describe(build_grid(args.budgets, args.bracket, args.batch_size)))


if __name__ == "__main__":
    _main()
