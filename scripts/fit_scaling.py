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


def measure_seed_noise(groups: Dict[float, List[Run]], bucket: str
                       ) -> Dict[float, Optional[float]]:
    """Per-budget per-point sigma in val_de, from repeat-seed pairs.

    A repeat arm is the SAME (N, D, C) under a different init, so the two
    runs differ only by seed and eval draw. For two independent draws
    Var(delta) = 2*sigma^2, hence sigma = |delta| / sqrt(2).

    Budgets with no repeat arm inherit the nearest budget's sigma in log C,
    which is better than dropping them: seed noise falls steeply with compute
    (measured 2.83 / 0.65 / 0.51 dE at 1e14 / 3e14 / 9.1e14), so a rung with
    no pair of its own is still far better served by its neighbour's figure
    than by pretending it has none.
    """
    out: Dict[float, Optional[float]] = {}
    for budget, runs in groups.items():
        by_cfg: Dict[tuple, List[float]] = {}
        for r in runs:
            de = r.val_de.get(bucket)
            if de is None:
                continue
            key = (r.config.d_model, r.config.slot_encoder_layers, r.passes)
            by_cfg.setdefault(key, []).append(de)
        deltas = [abs(v[0] - v[1]) for v in by_cfg.values() if len(v) >= 2]
        out[budget] = (float(np.mean(deltas)) / math.sqrt(2.0)
                       if deltas else None)
    known = {b: s for b, s in out.items() if s is not None}
    if known:
        for b, s in out.items():
            if s is None:
                near = min(known, key=lambda k: abs(math.log(k) - math.log(b)))
                out[b] = known[near]
    return out


def montecarlo_exponent(groups: Dict[float, List[Run]], bucket: str,
                        sigma: Dict[float, Optional[float]],
                        min_points: int, min_budgets: int,
                        allow_extrapolated: bool,
                        n_draws: int = 4000, seed: int = 0) -> dict:
    """alpha interval that propagates val_de noise into the parabola fits.

    fit_power_law's bootstrap resamples the (log C, log N*) points. With three
    rungs that is nearly degenerate, and it carries NO uncertainty from the
    val_de values each N* was derived from -- so it reported +-0.003 on the
    Sept 23 wave-1 fit when the real figure, measured below, is about +-0.23.
    An interval wrong by two orders of magnitude is worse than none, because
    the chroma-frontier comparison is a test of one spread against it.

    Each draw perturbs every run's val_de by its rung's sigma, refits every
    parabola, and refits the power law. Draws where any rung refuses (a <= 0,
    or N* outside the sampled range) are counted, not silently dropped: a high
    refusal rate means the exponent is not robustly identified at all, which
    is itself the finding.
    """
    if not sigma or all(v is None for v in sigma.values()):
        return {"status": "no repeat-seed pairs, so no noise estimate"}
    rng = np.random.default_rng(seed)
    alphas: List[float] = []
    refused = 0
    for _ in range(n_draws):
        ns, cs = [], []
        ok = True
        for budget, runs in sorted(groups.items()):
            s = sigma.get(budget) or 0.0
            shifted = []
            for r in runs:
                de = r.val_de.get(bucket)
                if de is None:
                    continue
                shifted.append(Run(
                    path=r.path, config=r.config, n_params=r.n_params,
                    steps=r.steps, passes=r.passes, flops=r.flops,
                    val_de={bucket: de + rng.normal(0.0, s)},
                    val_loss=r.val_loss, epoch_estimate=r.epoch_estimate))
            f = fit_isoflop(budget, shifted, bucket, min_points)
            if not f.usable or (f.extrapolated and not allow_extrapolated):
                ok = False
                break
            ns.append(math.log(f.n_star))
            cs.append(math.log(f.budget))
        if not ok or len(ns) < min_budgets:
            refused += 1
            continue
        alphas.append(float(np.polyfit(cs, ns, 1)[0]))
    if len(alphas) < 50:
        return {"status": f"{refused}/{n_draws} draws refused; alpha is not "
                          f"robustly identified under the measured noise",
                "refused_fraction": refused / n_draws}
    alphas.sort()
    return {
        "status": "ok",
        "median": float(np.median(alphas)),
        "ci_low": float(np.percentile(alphas, 2.5)),
        "ci_high": float(np.percentile(alphas, 97.5)),
        "refused_fraction": refused / n_draws,
        "sigma_by_budget": {f"{b:.3e}": sigma.get(b) for b in sorted(groups)},
    }


# ============================================================================
# Reporting
# ============================================================================


def report(fits: Dict[str, List[IsoFlopFit]],
           laws_n: Dict[str, PowerLaw], laws_d: Dict[str, PowerLaw],
           mc: Optional[Dict[str, dict]] = None) -> None:
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
    # ---- noise-propagated intervals -------------------------------------
    if mc:
        print(f"\n{'-' * 84}")
        print("alpha with val_de NOISE PROPAGATED (the interval to quote)")
        print("Each draw perturbs every run's val_de by its rung's measured")
        print("seed sigma, refits every parabola, and refits the power law.")
        print("The bootstrap CI above resamples the (log C, log N*) points only")
        print("and carries none of that, so it is far too tight to test anything.")
        any_sigma = next((m.get("sigma_by_budget") for m in mc.values()
                          if m.get("sigma_by_budget")), None)
        if any_sigma:
            print("  per-rung sigma from repeat-seed pairs: "
                  + ", ".join(f"C={k}: {v:.2f} dE"
                              for k, v in any_sigma.items() if v is not None))
        print(f"\n    {'bucket':>7} {'alpha':>8} {'95% interval':>22} "
              f"{'width':>7} {'refused':>8}")
        for b in [POOLED, *CHROMA_BUCKETS]:
            m = mc.get(b) or {}
            if m.get("status") != "ok":
                print(f"    {b:>7} {'--':>8} {'--':>22} {'--':>7} "
                      f"{m.get('refused_fraction', float('nan')):>7.0%}"
                      f"   {m.get('status', 'not run')}")
                continue
            interval = f"[{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]"
            print(f"    {b:>7} {m['median']:>+8.3f} {interval:>22} "
                  f"{m['ci_high'] - m['ci_low']:>7.3f} "
                  f"{m['refused_fraction']:>7.0%}")
        print("\n  A high refused fraction means the exponent is not robustly")
        print("  identified under the noise the runs actually exhibit.")

    print(f"\n{'=' * 84}\nCHROMA-CONDITIONED FRONTIER\n{'=' * 84}")
    if len(bucket_laws) < 2:
        print("  Not enough buckets with a usable exponent to compare.")
    else:
        spread = max(p.exponent for p in bucket_laws.values()) - \
                 min(p.exponent for p in bucket_laws.values())
        print("  alpha by bucket: " +
              ", ".join(f"{b}={p.exponent:+.4f}" for b, p in bucket_laws.items()))
        print(f"  spread: {spread:.4f}")

        # The test MUST use the noise-propagated width. Using the
        # resample-only CI declared a chroma-dependent frontier on the Sept 23
        # wave-1 data purely because that CI was ~150x too tight.
        widths = [mc[b]["ci_high"] - mc[b]["ci_low"] for b in bucket_laws
                  if mc and (mc.get(b) or {}).get("status") == "ok"] if mc else []
        if widths:
            widest = max(widths)
            print(f"  widest noise-propagated 95% interval: {widest:.4f}")
            if spread > widest:
                print("  -> spread EXCEEDS it: the buckets plausibly have "
                      "different exponents.")
            else:
                print("  -> spread is WITHIN it: these data DO NOT separate the "
                      "buckets.")
                print("     Do not claim a chroma-dependent frontier on this "
                      "evidence. More rungs,")
                print("     more examples per bucket, or more seeds are needed. "
                      "A null here is a")
                print("     power limit, not evidence that the frontier is "
                      "chroma-independent.")
        else:
            print("  No noise-propagated interval available (no repeat-seed "
                  "pairs), so no test.")
            print("  Add --repeat-seed arms before comparing buckets; the "
                  "resample-only CI is")
            print("  far too tight to support a comparison.")
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
    p.add_argument("--n-noise-draws", type=int, default=4000,
                   help="Monte Carlo draws for the noise-propagated alpha "
                        "interval; 0 disables it")
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

    # Noise-propagated intervals. These are the ones to quote; the bootstrap
    # inside fit_power_law resamples budget points only.
    mc: Dict[str, dict] = {}
    for bucket in [POOLED, *CHROMA_BUCKETS]:
        sigma = measure_seed_noise(groups, bucket)
        mc[bucket] = montecarlo_exponent(
            groups, bucket, sigma,
            min_points=args.min_sizes_per_budget,
            min_budgets=args.min_budgets,
            allow_extrapolated=args.allow_extrapolated,
            n_draws=args.n_noise_draws)

    report(fits, laws_n, laws_d, mc)

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
                # The interval to quote. N_star.ci_* resamples budget points
                # only and propagates no val_de uncertainty.
                "N_star_noise_propagated": mc.get(b),
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
        # Reuse the points' own colour for this rung's curve and N* marker.
        # Letting each ax.plot draw the next colour in the cycle put every
        # rung's parabola in a DIFFERENT colour from its own points, which
        # made the figure unreadable: the C=1e14 dots came out blue with an
        # orange curve.
        line, = ax.plot(ns, f.de_values, "o", label=f"C={f.budget:.1e}")
        colour = line.get_color()
        if f.coeffs:
            grid = np.geomspace(ns.min(), ns.max(), 100)
            ax.plot(grid, np.polyval(list(f.coeffs), np.log(grid)), "-",
                    color=colour, alpha=0.6)
        if f.usable:
            ax.axvline(f.n_star, ls=":", color=colour, alpha=0.5)
            ax.plot([f.n_star], [f.de_at_min], "*", color=colour,
                    markersize=13, markeredgecolor="0.2", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("N (parameters)")
    ax.set_ylabel(r"val $\Delta E_{00}$ (median)")
    ax.set_title(r"Step 1: IsoFLOP curves  (star = fitted $N^*$ per budget)")
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
