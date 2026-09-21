#!/usr/bin/env python3
"""
IsoFLOP two-step fit: N*(C) ~ C^alpha and D*(C) ~ C^beta
=========================================================

Step 1: for each compute budget C_i, fit a parabola to (log N, val_de) over
        the model sizes run at that budget, and read off the minimum ->
        N*(C_i), D*(C_i).
Step 2: fit a power law across budgets: log N* = alpha * log C + const.

Same two-step structure as Chinchilla's Approach 2, with the corrections from
Porian et al. 2024 (arXiv:2406.19146) built in upstream of this script:
last-layer FLOP accounting (src/scaling/flops.py), warmup as a fraction of
total compute (scripts/training.py), and per-scale LR re-tuning
(scripts/lr_tuning.py, which selects on DeltaE).

The metric is DeltaE_00, never cross-entropy
-------------------------------------------
CE and DeltaE are decoupled on INDIGO -- verified, and visible even within a
single production run, whose CE-optimal and DeltaE-optimal checkpoints are
different steps. Fitting CE would answer a different question than the one
the study is asking, so this script reads val_de and treats val_loss purely
as a recorded diagnostic. There is deliberately no option to fit on CE.

Chroma-conditioned exponents (the novel part)
---------------------------------------------
Beyond the pooled fit, the same two-step procedure runs independently on each
chroma bucket (C* = sqrt(a*^2+b*^2): low < 20, 20 <= mid < 50, high >= 50).
If alpha differs across buckets, the compute-optimal frontier depends on
chroma difficulty -- i.e. the optimal (N, D) split for saturated colors is
not the optimal split for near-greys. Prior evidence that this is a real
effect rather than a fishing expedition: on the production run, high-chroma
targets sit ~45% worse than random ones (mean DeltaE ~24 vs ~16.5) and their
p75 tail degrades over training while the median stays flat.

Why this script is so insistent about refusing to fit
-----------------------------------------------------
Every degenerate case here produces a plausible-looking exponent rather than
an error, and a wrong alpha is the entire deliverable. So:

  * A parabola with a <= 0 in log N has no interior minimum. Reporting
    -b/(2a) anyway yields a finite, utterly meaningless N*. Refused.
  * An N* outside the sampled range is an extrapolation from a fit that
    never bracketed its own optimum. Reported, and excluded from the
    power-law fit unless --allow-extrapolated.
  * Fewer than 3 sizes per budget cannot determine a parabola, and fewer
    than 3 budgets make alpha an artifact of 2 points. Both refused.
  * alpha + beta must come to ~1 by construction, since C ~ N*D. It is an
    arithmetic check, NOT an empirical result -- a large deviation means the
    budget bookkeeping is wrong somewhere. Reported as a diagnostic.

Uncertainty is not optional at 4 budgets. alpha is reported with a
bootstrap CI over budgets; a point estimate alone would be over-claimed.

Inputs
------
A root directory of run subdirectories, each containing the `config.json`
and `history.jsonl` that scripts/training.py writes. N comes from
config.json through src/scaling/flops.py (so the fit and the sweep planner
agree by construction) and C from the same calculator times the observed
example-passes -- not from the directory name, so a mislabelled run cannot
quietly shift a budget.

Usage
-----
    python scripts/fit_scaling.py --runs-root data/checkpoints/scaling_sweep
    python scripts/fit_scaling.py --runs-root ... --plot
    python scripts/fit_scaling.py --runs-root ... --metric-selection best
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.materials_vocab import VOCAB_SIZE
from src.model import ModelConfig
from src.scaling.flops import (
    n_params,
    train_flops_per_example,
    tokens_per_example,
)

CHROMA_BUCKETS = ("low", "mid", "high")
POOLED = "pooled"

# A run ending above this cross-entropy has diverged, not learned -- it is
# worse than predicting uniformly at random over the vocabulary.
#
# This has to be screened on CE, because DeltaE cannot see it. A diverged
# model still emits structures the simulator can colour, and the resulting
# DeltaE lands squarely in the range an untrained model produces: in the LR
# sweeps, a 0.77M and a 17.5M model that had both blown up (val_loss 1426 and
# inf) reported the SAME val_de of 28.6454. Such a point dropped into an
# IsoFLOP parabola would drag its minimum with a number that means nothing.
DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)


# ============================================================================
# Loading
# ============================================================================


@dataclass
class Run:
    path: Path
    config: ModelConfig
    n_params: int
    steps: int
    passes: int
    flops: float
    val_de: Dict[str, Optional[float]]      # bucket (or POOLED) -> DeltaE
    val_loss: Optional[float]
    epoch_estimate: Optional[float]

    @property
    def name(self) -> str:
        return self.path.name


def _read_history(path: Path, selection: str) -> Optional[dict]:
    """Pick the row a run is summarised by.

    'final' (default) takes the last row carrying a scored val_de, matching
    Chinchilla/Porian, who summarise a run by the loss at the end of a
    COMPLETED schedule. 'best' takes the minimum val_de over the run, which
    is a different experiment (it bakes in early stopping) and will bias
    every point optimistically -- offered for diagnosis, not for the
    headline fit.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("val_de_median") is not None:
                rows.append(row)
    if not rows:
        return None
    if selection == "best":
        return min(rows, key=lambda r: r["val_de_median"])
    return max(rows, key=lambda r: r.get("step", 0))


def load_runs(root: Path, selection: str, corpus: Optional[int]) -> Tuple[List[Run], List[str]]:
    runs: List[Run] = []
    skipped: List[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = d / "config.json"
        # training.py writes config.json inside each checkpoint subdir, so
        # fall back to any of them; they are identical within a run.
        if not cfg_path.exists():
            candidates = sorted(d.glob("*/config.json"))
            if not candidates:
                skipped.append(f"{d.name}: no config.json")
                continue
            cfg_path = candidates[0]
        hist = d / "history.jsonl"
        if not hist.exists():
            skipped.append(f"{d.name}: no history.jsonl")
            continue
        row = _read_history(hist, selection)
        if row is None:
            skipped.append(f"{d.name}: no scored val_de in history")
            continue

        with open(cfg_path) as f:
            config = ModelConfig.from_dict(json.load(f))
        steps = int(row.get("step", 0))
        passes = steps * config.batch_size
        if passes <= 0:
            skipped.append(f"{d.name}: step=0, nothing trained")
            continue

        vl = row.get("val_loss")
        if vl is not None and (not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS):
            skipped.append(
                f"{d.name}: DIVERGED (val_loss={vl:.4g} > "
                f"{DIVERGENCE_VAL_LOSS:.1f}); its val_de is meaningless")
            continue

        by_chroma = row.get("val_de_by_chroma") or {}
        val_de: Dict[str, Optional[float]] = {POOLED: row["val_de_median"]}
        for b in CHROMA_BUCKETS:
            st = by_chroma.get(b) or {}
            val_de[b] = st.get("median")

        runs.append(Run(
            path=d, config=config, n_params=n_params(config),
            steps=steps, passes=passes,
            flops=train_flops_per_example(config) * passes,
            val_de=val_de, val_loss=row.get("val_loss"),
            epoch_estimate=(passes / corpus) if corpus else None,
        ))
    return runs, skipped


def group_by_budget(runs: List[Run], rel_tol: float) -> Dict[float, List[Run]]:
    """Cluster runs into budgets by relative FLOP proximity.

    Budgets are recovered from the measured compute rather than trusted from
    a label, so a run that stopped early (preemption, a TIME_LIMIT kill)
    lands in its own cluster and is visible, instead of silently corrupting
    the rung it was supposed to belong to.
    """
    groups: Dict[float, List[Run]] = {}
    for r in sorted(runs, key=lambda r: r.flops):
        placed = False
        for key in groups:
            if abs(math.log(r.flops) - math.log(key)) < rel_tol:
                groups[key].append(r)
                placed = True
                break
        if not placed:
            groups[r.flops] = [r]
    # Re-key each cluster by its geometric-mean budget.
    return {
        float(np.exp(np.mean([math.log(x.flops) for x in group]))): group
        for group in groups.values()
    }


# ============================================================================
# Step 1: per-budget parabola
# ============================================================================


@dataclass
class IsoFlopFit:
    budget: float
    bucket: str
    n_values: List[int]
    de_values: List[float]
    coeffs: Optional[Tuple[float, float, float]] = None
    n_star: Optional[float] = None
    d_star: Optional[float] = None
    de_at_min: Optional[float] = None
    extrapolated: bool = False
    status: str = "ok"
    r_squared: Optional[float] = None

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.n_star is not None


def fit_isoflop(budget: float, runs: List[Run], bucket: str,
                min_points: int = 3) -> IsoFlopFit:
    pairs = [(r.n_params, r.val_de.get(bucket), r) for r in runs]
    pairs = [(n, de, r) for n, de, r in pairs if de is not None]
    ns = [n for n, _, _ in pairs]
    des = [de for _, de, _ in pairs]
    fit = IsoFlopFit(budget=budget, bucket=bucket, n_values=ns, de_values=des)

    if len(pairs) < min_points:
        fit.status = (f"too few points ({len(pairs)} < {min_points}); a "
                      f"parabola is undetermined")
        return fit

    x = np.log(np.array(ns, dtype=float))
    y = np.array(des, dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    fit.coeffs = (float(a), float(b), float(c))

    resid = y - np.polyval([a, b, c], x)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    fit.r_squared = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else None

    if a <= 0:
        # Opens downward (or is a straight line): the vertex is a MAXIMUM, or
        # there is no vertex at all. -b/(2a) would still evaluate to a
        # finite, meaningless number, so refuse it rather than report it.
        fit.status = (f"parabola opens downward (a={a:.3e} <= 0): no interior "
                      f"minimum, so N* is not identified at this budget")
        return fit

    log_n_star = -b / (2.0 * a)
    n_star = float(np.exp(log_n_star))
    fit.n_star = n_star
    fit.de_at_min = float(np.polyval([a, b, c], log_n_star))

    if not (min(ns) <= n_star <= max(ns)):
        fit.extrapolated = True
        fit.status = "ok"  # usable, but flagged

    # D* from the budget identity, using the same FLOP model the sweep used.
    ref = min(runs, key=lambda r: abs(r.n_params - n_star))
    scaled = ModelConfig.from_dict({**ref.config.to_dict()})
    per_ex = train_flops_per_example(scaled) * (n_star / ref.n_params)
    fit.d_star = float(budget / per_ex) if per_ex > 0 else None
    return fit


# ============================================================================
# Step 2: power law across budgets
# ============================================================================


@dataclass
class PowerLaw:
    bucket: str
    exponent: Optional[float] = None
    intercept: Optional[float] = None
    n_budgets: int = 0
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    log_rms_residual: Optional[float] = None
    status: str = "ok"


def fit_power_law(budgets: Sequence[float], values: Sequence[float],
                  bucket: str, n_boot: int = 2000,
                  min_budgets: int = 3, seed: int = 0) -> PowerLaw:
    pl = PowerLaw(bucket=bucket, n_budgets=len(budgets))
    if len(budgets) < min_budgets:
        pl.status = (f"only {len(budgets)} budget(s) with a usable N*; need "
                     f">= {min_budgets} for an exponent that is not an "
                     f"artifact of too few points")
        return pl

    x = np.log(np.asarray(budgets, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pl.exponent = float(slope)
    pl.intercept = float(intercept)
    resid = y - (slope * x + intercept)
    pl.log_rms_residual = float(np.sqrt(np.mean(resid ** 2)))

    # Bootstrap over budgets. With 4 rungs the CI is wide and that is the
    # honest answer -- a bare point estimate would imply precision we do
    # not have.
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2:
            continue
        s, _ = np.polyfit(x[idx], y[idx], 1)
        slopes.append(s)
    if len(slopes) >= 50:
        pl.ci_low = float(np.percentile(slopes, 2.5))
        pl.ci_high = float(np.percentile(slopes, 97.5))
    return pl


# ============================================================================
# Reporting
# ============================================================================


def report(fits: Dict[str, List[IsoFlopFit]],
           laws_n: Dict[str, PowerLaw], laws_d: Dict[str, PowerLaw]) -> None:
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        blist = fits.get(bucket, [])
        if not blist:
            continue
        label = "POOLED (all chroma)" if bucket == POOLED else f"CHROMA = {bucket}"
        print(f"\n{'=' * 84}\nSTEP 1 -- IsoFLOP parabolas: {label}\n{'=' * 84}")
        print(f"  {'budget C':>11} {'pts':>4} {'N* (M)':>10} {'D* (Mex)':>10} "
              f"{'dE at min':>10} {'R^2':>7}  note")
        for f in sorted(blist, key=lambda f: f.budget):
            if f.usable:
                note = "EXTRAPOLATED (N* outside sampled range)" if f.extrapolated else ""
                print(f"  {f.budget:>11.3e} {len(f.n_values):>4} "
                      f"{f.n_star / 1e6:>10.3f} "
                      f"{(f.d_star or float('nan')) / 1e6:>10.2f} "
                      f"{f.de_at_min:>10.4f} "
                      f"{(f.r_squared if f.r_squared is not None else float('nan')):>7.4f}  {note}")
            else:
                print(f"  {f.budget:>11.3e} {len(f.n_values):>4} {'--':>10} "
                      f"{'--':>10} {'--':>10} {'--':>7}  REFUSED: {f.status}")

    print(f"\n{'=' * 84}\nSTEP 2 -- power laws\n{'=' * 84}")
    print(f"  {'bucket':>8} {'quantity':>4}  {'exponent':>9} {'95% CI':>20} "
          f"{'rungs':>6} {'log-RMS':>8}  status")
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        for tag, laws in (("N*", laws_n), ("D*", laws_d)):
            pl = laws.get(bucket)
            if pl is None:
                continue
            if pl.exponent is None:
                print(f"  {bucket:>8} {tag:>4}  {'--':>9} {'--':>20} "
                      f"{pl.n_budgets:>6} {'--':>8}  REFUSED: {pl.status}")
                continue
            ci = (f"[{pl.ci_low:+.3f}, {pl.ci_high:+.3f}]"
                  if pl.ci_low is not None else "n/a")
            print(f"  {bucket:>8} {tag:>4}  {pl.exponent:>+9.4f} {ci:>20} "
                  f"{pl.n_budgets:>6} {pl.log_rms_residual:>8.4f}  ok")

    # alpha + beta is an identity, not a result. Say so, and use the
    # deviation as a bookkeeping check.
    print(f"\n{'-' * 84}")
    print("consistency: alpha + beta should be ~1 BY CONSTRUCTION (C ~ N*D).")
    print("This is an arithmetic check on the budget bookkeeping, not a finding.")
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        a_, b_ = laws_n.get(bucket), laws_d.get(bucket)
        if a_ and b_ and a_.exponent is not None and b_.exponent is not None:
            tot = a_.exponent + b_.exponent
            flag = "" if abs(tot - 1.0) < 0.05 else "  <-- CHECK THE BUDGET ACCOUNTING"
            print(f"  {bucket:>8}: alpha={a_.exponent:+.4f} beta={b_.exponent:+.4f} "
                  f"sum={tot:.4f}{flag}")

    # The novel claim lives here, so state what would and would not support it.
    pooled = laws_n.get(POOLED)
    bucket_laws = {b: laws_n[b] for b in CHROMA_BUCKETS
                   if b in laws_n and laws_n[b].exponent is not None}
    print(f"\n{'=' * 84}\nCHROMA-CONDITIONED FRONTIER\n{'=' * 84}")
    if len(bucket_laws) < 2:
        print("  Not enough buckets with a usable exponent to compare.")
    else:
        spread = max(p.exponent for p in bucket_laws.values()) - \
                 min(p.exponent for p in bucket_laws.values())
        print("  alpha by bucket: " +
              ", ".join(f"{b}={p.exponent:+.4f}" for b, p in bucket_laws.items()))
        print(f"  spread: {spread:.4f}")
        widest = max((p.ci_high - p.ci_low) for p in bucket_laws.values()
                     if p.ci_low is not None) if any(
            p.ci_low is not None for p in bucket_laws.values()) else None
        if widest is not None:
            print(f"  widest single-bucket 95% CI: {widest:.4f}")
            if spread > widest:
                print("  -> spread EXCEEDS the widest CI: the buckets plausibly "
                      "have different exponents.")
            else:
                print("  -> spread is WITHIN the widest CI: these data do not "
                      "separate the buckets. Do not claim a chroma-dependent\n"
                      "     frontier on this evidence; more rungs or more "
                      "examples per bucket are needed.")
        if pooled and pooled.exponent is not None:
            print(f"  (pooled alpha for reference: {pooled.exponent:+.4f})")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fit IsoFLOP N*(C) ~ C^alpha and D*(C) ~ C^beta on val_de.")
    p.add_argument("--runs-root", type=str, required=True,
                   help="Directory of scaling-sweep run subdirectories.")
    p.add_argument("--metric-selection", type=str, default="final",
                   choices=["final", "best"],
                   help="Summarise each run by its FINAL val_de (default, "
                        "matches Chinchilla/Porian on a completed schedule) "
                        "or its BEST (bakes in early stopping; diagnostic "
                        "only).")
    p.add_argument("--budget-rel-tol", type=float, default=0.15,
                   help="Runs within this log-distance in FLOPs are treated "
                        "as the same budget (default 0.15 ~ +/-16%%).")
    p.add_argument("--min-sizes-per-budget", type=int, default=3)
    p.add_argument("--min-budgets", type=int, default=3)
    p.add_argument("--allow-extrapolated", action="store_true",
                   help="Include budgets whose N* fell outside the sampled "
                        "range in the power-law fit. Off by default: such a "
                        "parabola never bracketed its own optimum.")
    p.add_argument("--corpus-examples", type=int, default=None,
                   help="Corpus size, to report each run's epoch count "
                        "(repeats are a known threat to the fit).")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        raise SystemExit(f"--runs-root {root} is not a directory")

    runs, skipped = load_runs(root, args.metric_selection, args.corpus_examples)
    print(f"loaded {len(runs)} runs from {root}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print(f"   - {s}")
    if not runs:
        raise SystemExit("no usable runs")

    groups = group_by_budget(runs, args.budget_rel_tol)
    print(f"\n{len(groups)} budget cluster(s) "
          f"(runs grouped by MEASURED compute, not by name):")
    for budget, group in sorted(groups.items()):
        eps = [r.epoch_estimate for r in group if r.epoch_estimate]
        ep_s = f"  epochs {min(eps):.2f}-{max(eps):.2f}" if eps else ""
        print(f"  C={budget:.3e}: {len(group)} sizes "
              f"N={min(r.n_params for r in group) / 1e6:.2f}-"
              f"{max(r.n_params for r in group) / 1e6:.2f}M{ep_s}")
        if eps and max(eps) > 4:
            print(f"     WARNING: up to {max(eps):.1f} passes over the corpus; "
                  f"Chinchilla's model assumes fresh data. See "
                  f"scripts/epoch_ceiling_probe.py.")

    fits: Dict[str, List[IsoFlopFit]] = {}
    laws_n: Dict[str, PowerLaw] = {}
    laws_d: Dict[str, PowerLaw] = {}
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        blist = [fit_isoflop(b, g, bucket, args.min_sizes_per_budget)
                 for b, g in sorted(groups.items())]
        fits[bucket] = blist
        good = [f for f in blist if f.usable and
                (args.allow_extrapolated or not f.extrapolated)]
        laws_n[bucket] = fit_power_law([f.budget for f in good],
                                       [f.n_star for f in good], bucket,
                                       min_budgets=args.min_budgets)
        d_ok = [f for f in good if f.d_star]
        laws_d[bucket] = fit_power_law([f.budget for f in d_ok],
                                       [f.d_star for f in d_ok], bucket,
                                       min_budgets=args.min_budgets)

    report(fits, laws_n, laws_d)

    out = Path(args.output) if args.output else root / "scaling_fit.json"
    payload = {
        "metric": "val_de_median",
        "metric_note": "DeltaE_00. CE is not fit: it is decoupled from DeltaE "
                       "on INDIGO.",
        "metric_selection": args.metric_selection,
        "n_runs": len(runs),
        "skipped": skipped,
        "budgets": sorted(groups),
        "fits": {
            b: [{
                "budget": f.budget, "n_points": len(f.n_values),
                "n_values": f.n_values, "de_values": f.de_values,
                "coeffs": f.coeffs, "n_star": f.n_star, "d_star": f.d_star,
                "de_at_min": f.de_at_min, "r_squared": f.r_squared,
                "extrapolated": f.extrapolated, "status": f.status,
            } for f in fits[b]] for b in fits
        },
        "power_laws": {
            b: {
                "N_star": vars(laws_n[b]),
                "D_star": vars(laws_d[b]),
            } for b in laws_n
        },
        "runs": [{
            "name": r.name, "n_params": r.n_params, "steps": r.steps,
            "passes": r.passes, "flops": r.flops, "val_de": r.val_de,
            "val_loss": r.val_loss, "epochs": r.epoch_estimate,
        } for r in runs],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[INFO] Fit written to {out}")

    if args.plot:
        _plot(fits, laws_n, root, out.with_suffix(".png"))


def _plot(fits, laws_n, root: Path, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for f in sorted(fits[POOLED], key=lambda f: f.budget):
        if not f.n_values:
            continue
        ns = np.array(f.n_values, dtype=float)
        ax.plot(ns, f.de_values, "o", label=f"C={f.budget:.1e}")
        if f.coeffs:
            grid = np.geomspace(ns.min(), ns.max(), 100)
            ax.plot(grid, np.polyval(list(f.coeffs), np.log(grid)), "-", alpha=0.5)
        if f.usable:
            ax.axvline(f.n_star, ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("N (parameters)")
    ax.set_ylabel(r"val $\Delta E_{00}$ (median)")
    ax.set_title("Step 1: IsoFLOP curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        good = [f for f in fits.get(bucket, []) if f.usable]
        if len(good) < 2:
            continue
        good.sort(key=lambda f: f.budget)
        cs = np.array([f.budget for f in good])
        ns = np.array([f.n_star for f in good])
        pl = laws_n.get(bucket)
        fitted = pl is not None and pl.exponent is not None
        # A refused fit must not be drawn as if it were a result: dash the
        # line and say so in the legend. Extrapolated N* (parabolas that
        # never bracketed their own minimum) get hollow markers, since they
        # are excluded from the fit by default.
        lbl = (f"{bucket} (a={pl.exponent:+.3f})" if fitted
               else f"{bucket} (FIT REFUSED)")
        line, = ax.plot(cs, ns, "-" if fitted else "--",
                        alpha=1.0 if fitted else 0.45, label=lbl, zorder=2)
        colour = line.get_color()
        solid = [(f.budget, f.n_star) for f in good if not f.extrapolated]
        hollow = [(f.budget, f.n_star) for f in good if f.extrapolated]
        if solid:
            ax.plot([c for c, _ in solid], [n for _, n in solid], "o",
                    color=colour, zorder=3)
        if hollow:
            ax.plot([c for c, _ in hollow], [n for _, n in hollow], "o",
                    mfc="none", mec=colour, mew=1.6, zorder=3)
    ax.plot([], [], "o", mfc="none", mec="0.35", mew=1.6,
            label="extrapolated N* (excluded)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C (FLOPs)")
    ax.set_ylabel("N* (parameters)")
    ax.set_title(r"Step 2: $N^*(C) \propto C^{\alpha}$, per chroma bucket")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to {path}")


if __name__ == "__main__":
    main()
