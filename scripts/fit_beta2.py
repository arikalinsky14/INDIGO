#!/usr/bin/env python3
"""Which AdamW beta2? A paired, seed-replicated, LR-tuned comparison.

Reads the cells `slurms/lr_grid.sh STAGE=beta2` writes: every beta2 in
{0.9, 0.95, 0.98, 0.99, 0.999} at three model sizes spanning the ladder, three
seeds each, seven learning rates per cell. Writes a JSON verdict, a text
report, and the figure.

What makes the comparison defensible
------------------------------------
1. **Each beta2 at its own learning rate.** beta2 and the LR interact (stage 1
   saw 0.95 lose at prior/4 and win at the prior), so a comparison at one
   shared rate picks whichever beta2 suits that rate. Every cell's optimum is
   taken Porian-style, as the argmin of an Akima interpolation of DeltaE
   against log LR, and flagged when it is not bracketed by the grid.
2. **Paired by seed.** Within one (size, seed) every beta2 shares the
   initialisation, the data order and the 2,048 validation examples, so their
   differences cancel everything but beta2. Seeds are blocks.
3. **Two noise sources, measured separately.** Seed-to-seed spread (training
   noise) gives the t-intervals and the ANOVA; a paired bootstrap over the
   2,048 examples (same examples, both arms) gives the evaluation noise.
4. **Interaction tested, not assumed.** A two-way ANOVA with seed blocks asks
   whether the best beta2 moves with model size before one value is carried
   across the ladder.

Decision rule, fixed before the data
------------------------------------
* If the size x beta2 interaction is significant (p < 0.05), beta2 depends on
  scale: report the per-size winners and use `configs.beta2_for`'s step
  function rather than one constant.
* Otherwise choose the beta2 with the lowest pooled, block-centred tuned
  DeltaE. It is "distinguishable" from an alternative when the pooled paired
  95% interval of their difference excludes zero; alternatives that are not
  form the indistinguishable set, and the report says so rather than
  overstating a winner.
* A winner at an end of the beta2 grid (0.9 or 0.999) is not bracketed: the
  grid gets extended before any claim.
* Cells whose LR optimum sits on the edge of their grid are listed; their tuned
  DeltaE is an upper bound, which can only make that beta2 look worse.

    python scripts/fit_beta2.py --results-dir outputs/lr_search/beta2/cross_attn
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.materials_vocab import VOCAB_SIZE                       # noqa: E402
from src.scaling.porian import akima_argmin, tuned_optimum      # noqa: E402

DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)   # as lr_tuning.py
METRICS = ("pooled", "low", "mid", "high")
ALPHA = 0.05


# ------------------------------------------------------------------ loading --
def load_cells(results_dirs: Sequence[str]) -> List[dict]:
    """One record per (size, beta2, seed): its learning-rate curve.

    A cell may be spread over several directories: the main seven-rate grid,
    plus the extra rates `STAGE=beta2x` adds past an unbracketed edge. Files
    for the same (size, beta2, seed) are merged into one curve; the prior is
    taken from the file with the most rates, the centred main grid.
    """
    cells = []
    for root in results_dirs:
        if not Path(root).is_dir():
            continue
        for path in sorted(glob.glob(str(Path(root) / "lr_search_*.json"))):
            d = json.load(open(path))
            trials = sorted(d.get("results", []), key=lambda t: t["lr"])
            if not trials:
                continue
            lrs = np.array([t["lr"] for t in trials], dtype=float)
            # The grid is centred on the prior in log space, so its geometric
            # mean IS the prior; normalising by it puts every size on one axis.
            prior = float(np.exp(np.mean(np.log(lrs))))
            curve = {m: [] for m in METRICS}
            diverged, per_ex, buckets = [], [], None
            for t in trials:
                vl = t.get("best_val_loss")
                div = (vl is None or not math.isfinite(vl)
                       or vl > DIVERGENCE_VAL_LOSS)
                diverged.append(div)
                curve["pooled"].append(t.get("final_val_de"))
                by_c = t.get("final_val_de_by_chroma") or {}
                for b in ("low", "mid", "high"):
                    curve[b].append((by_c.get(b) or {}).get("median"))
                de_res = t.get("de_result") or {}
                per_ex.append(de_res.get("per_example_de"))
                buckets = buckets or de_res.get("per_example_bucket")
            cells.append({
                "file": Path(path).name,
                "size": (int(d["d_model"]), int(d["slot_encoder_layers"])),
                "n_params": int(d["n_params"]),
                "D": int(d.get("train_examples", 0)) * int(d.get("epochs", 1)),
                "beta2": float(d["beta2"]), "seed": int(d.get("seed", 42)),
                "lrs": lrs, "prior": prior, "diverged": np.array(diverged),
                "curve": curve, "per_example": per_ex, "buckets": buckets,
            })
    return _merge(cells)


def _merge(cells: List[dict]) -> List[dict]:
    """Fold records of one (size, beta2, seed) into a single LR curve."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["size"], c["beta2"], c["seed"])].append(c)
    out = []
    for parts in groups.values():
        if len(parts) == 1:
            out.append(parts[0])
            continue
        main = max(parts, key=lambda c: len(c["lrs"]))
        rows = {}
        for c in parts:
            for i, lr in enumerate(c["lrs"]):
                key = round(float(np.log(lr)), 6)
                if key in rows:            # same rate twice: keep the first
                    continue
                rows[key] = (float(lr), bool(c["diverged"][i]),
                             {m: c["curve"][m][i] for m in METRICS},
                             c["per_example"][i])
        order = sorted(rows.values(), key=lambda r: r[0])
        out.append({**main,
                    "file": "+".join(c["file"] for c in parts),
                    "lrs": np.array([r[0] for r in order]),
                    "diverged": np.array([r[1] for r in order]),
                    "curve": {m: [r[2][m] for r in order] for m in METRICS},
                    "per_example": [r[3] for r in order]})
    return out


#: How a cell's LR curve becomes one number. "akima" is Porian's: the minimum
#: of an Akima interpolant through the points. "quadratic" fits a parabola in
#: log LR through every stable point and takes its minimum. Set by --estimator.
ESTIMATOR = "akima"


def tune(cell: dict, metric: str = "pooled") -> dict:
    """The cell's optimum over learning rate.

    `value` is the estimator's minimum (see ESTIMATOR) when enough rates were
    stable, otherwise the best stable grid point. `on_edge` is Porian's
    bracketing test, the optimum outside the second and second-to-last stable
    rates; `edge` says which side.

    Why there are two estimators. INDIGO's runs are not reproducible to the
    bit: the same seed, size, beta2, data and LR, trained twice (stage 1 and
    the beta2 study), differed by up to 1.2 DeltaE, single-run sd ~0.36. GPU
    nondeterminism sends identical starts down different paths. With noise
    that size on every point, an interpolant through the points finds the
    luckiest one, and the minimum of seven noisy values is biased low by
    roughly a standard deviation. A parabola through all of them averages the
    noise instead. Which one serves the beta2 decision better is settled by
    simulation at that noise level, not by which beta2 it favours.
    """
    ok = [(lr, v) for lr, v, div in zip(cell["lrs"], cell["curve"][metric],
                                       cell["diverged"])
          if not div and v is not None and np.isfinite(v)]
    out = {"n_stable": len(ok), "n_diverged": int(cell["diverged"].sum())}
    if not ok:
        return {**out, "value": None, "lr": None, "on_edge": True,
                "edge": "all diverged", "grid_value": None, "grid_lr": None}
    xs = np.array([p[0] for p in ok])
    ys = np.array([p[1] for p in ok])
    j = int(np.argmin(ys))
    out.update(grid_value=float(ys[j]), grid_lr=float(xs[j]))
    if len(ok) < 3:
        return {**out, "value": float(ys[j]), "lr": float(xs[j]),
                "on_edge": True, "edge": "fewer than 3 stable rates"}
    if ESTIMATOR == "quadratic":
        return {**out, **_quadratic_min(cell, xs, ys)}
    lr_star, on_edge = tuned_optimum(xs, ys)
    grid, y_grid, idx = akima_argmin(xs, ys)
    edge = ""
    if on_edge:
        if lr_star < xs[1]:
            edge = "low"
        else:
            # Above the second-highest stable rate. If a higher rate diverged,
            # the optimum is still capped from above, by instability.
            higher_div = bool(cell["diverged"][cell["lrs"] > xs[-1]].any())
            edge = "high, capped by divergence" if higher_div else "high"
    return {**out, "value": float(y_grid[idx]), "lr": float(lr_star),
            "on_edge": bool(on_edge), "edge": edge}


def _quadratic_min(cell: dict, xs: np.ndarray, ys: np.ndarray) -> dict:
    """Minimum of a parabola in log2(LR) through every stable point.

    Needs four points, so the fit has a residual. A parabola that opens down,
    or whose vertex falls outside the second and second-to-last stable rates,
    is not bracketed: its value is then the fitted curve's lowest point inside
    the tested range, and the cell is flagged like Porian's on_edge.
    """
    if xs.size < 4:
        j = int(np.argmin(ys))
        return {"value": float(ys[j]), "lr": float(xs[j]), "on_edge": True,
                "edge": "fewer than 4 stable rates"}
    lx = np.log2(xs)
    a, b, c = np.polyfit(lx, ys, 2)
    fine = np.linspace(lx.min(), lx.max(), 400)
    fit = a * fine ** 2 + b * fine + c
    k = int(np.argmin(fit))
    x_star = fine[k]
    on_edge = not (a > 0 and lx[1] <= -b / (2 * a) <= lx[-2])
    edge = ""
    if on_edge:
        if a <= 0:
            edge = "no interior minimum"
        elif x_star < lx[1]:
            edge = "low"
        else:
            higher_div = bool(cell["diverged"][cell["lrs"] > xs[-1]].any())
            edge = "high, capped by divergence" if higher_div else "high"
    return {"value": float(fit[k]), "lr": float(2 ** x_star),
            "on_edge": bool(on_edge), "edge": edge}


# ----------------------------------------------------------------- analysis --
def block_matrix(cells: List[dict], metric: str
                 ) -> Tuple[List[float], Dict[tuple, Dict[int, np.ndarray]], dict]:
    """size -> seed -> tuned values over the beta2 grid, complete blocks only."""
    betas = sorted({c["beta2"] for c in cells})
    tuned = {}
    for c in cells:
        tuned[(c["size"], c["seed"], c["beta2"])] = tune(c, metric)
    blocks: Dict[tuple, Dict[int, np.ndarray]] = defaultdict(dict)
    for size in sorted({c["size"] for c in cells}):
        for seed in sorted({c["seed"] for c in cells if c["size"] == size}):
            row = [tuned.get((size, seed, b), {}).get("value") for b in betas]
            if all(v is not None for v in row):
                blocks[size][seed] = np.array(row, dtype=float)
    return betas, dict(blocks), tuned


def t_interval(diffs: np.ndarray) -> Tuple[float, float, float]:
    """Mean and 95% t-interval; (mean, nan, nan) with one observation."""
    m = float(np.mean(diffs))
    if diffs.size < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(1 - ALPHA / 2, diffs.size - 1) * np.std(diffs, ddof=1) \
        / math.sqrt(diffs.size)
    return m, m - h, m + h


def anova(blocks: Dict[tuple, Dict[int, np.ndarray]]) -> dict:
    """Two-way ANOVA with seed blocks nested in size.

    y[size, seed, beta2] = block(size, seed) + beta2 + size:beta2 + error. With
    one observation per cell the error is the beta2 x seed interaction within
    each size, which is exactly the seed-to-seed inconsistency of the beta2
    effect, the noise any claim about beta2 has to beat.
    """
    sizes = list(blocks)
    seeds_per = {s: list(blocks[s]) for s in sizes}
    if not sizes or min(len(v) for v in seeds_per.values()) < 2:
        return {"status": "needs at least two seeds per size"}
    k = len(next(iter(blocks[sizes[0]].values())))
    Y = {s: np.vstack([blocks[s][sd] for sd in seeds_per[s]]) for s in sizes}
    # Centre within block: removes block means, leaves beta2 + interaction + e.
    C = {s: Y[s] - Y[s].mean(axis=1, keepdims=True) for s in sizes}
    allC = np.vstack([C[s] for s in sizes])
    grand_beta = allC.mean(axis=0)
    ss_beta = allC.shape[0] * float(np.sum(grand_beta ** 2))
    ss_int = sum(C[s].shape[0] * float(np.sum((C[s].mean(axis=0) - grand_beta) ** 2))
                 for s in sizes)
    ss_err = sum(float(np.sum((C[s] - C[s].mean(axis=0)) ** 2)) for s in sizes)
    df_beta = k - 1
    df_int = (len(sizes) - 1) * (k - 1)
    df_err = sum((C[s].shape[0] - 1) * (k - 1) for s in sizes)
    ms_err = ss_err / df_err
    f_beta = (ss_beta / df_beta) / ms_err
    out = {"df_beta2": df_beta, "df_error": df_err, "ms_error": ms_err,
           "sigma_error": math.sqrt(ms_err),
           "F_beta2": f_beta, "p_beta2": float(stats.f.sf(f_beta, df_beta, df_err))}
    if len(sizes) > 1:
        f_int = (ss_int / df_int) / ms_err
        out.update(df_interaction=df_int, F_interaction=f_int,
                   p_interaction=float(stats.f.sf(f_int, df_int, df_err)))
    return out


def quad_optimum(betas: Sequence[float], centred: np.ndarray, sigma: Optional[float],
                 n_seeds: int, rng: np.random.Generator, n_draws: int = 4000
                 ) -> dict:
    """Continuous optimum of a quadratic in x = log10(1 - beta2).

    `centred` is the seed-averaged, block-centred curve. The interval is a
    parametric bootstrap: each draw adds noise of the residual sigma (scaled
    for an average over `n_seeds`) and refits. A draw with no interior minimum
    counts against the estimate rather than being dropped.
    """
    x = np.log10(1 - np.asarray(betas))
    if x.size < 4:
        # Three points fit a parabola exactly, so its vertex has no error
        # estimate at all. Refuse rather than draw a confident-looking line.
        return {"beta2_star": None,
                "status": f"needs at least 4 beta2 values, have {x.size}"}
    a, b, c = np.polyfit(x, centred, 2)
    res = {"coef": [float(a), float(b), float(c)]}
    lo, hi = x.min(), x.max()
    if a > 0 and lo <= -b / (2 * a) <= hi:
        xs = -b / (2 * a)
        res["beta2_star"] = float(1 - 10 ** xs)
    else:
        res["beta2_star"] = None
    if sigma and n_seeds:
        s = sigma / math.sqrt(n_seeds)
        stars, interior = [], 0
        for _ in range(n_draws):
            a2, b2, _ = np.polyfit(x, centred + rng.normal(0, s, x.size), 2)
            if a2 > 0 and lo <= -b2 / (2 * a2) <= hi:
                interior += 1
                stars.append(-b2 / (2 * a2))
        res["interior_fraction"] = interior / n_draws
        if stars:
            q = np.percentile(stars, [2.5, 97.5])
            # x is log10(1 - beta2): a LARGER x is a SMALLER beta2.
            res["beta2_ci"] = [float(1 - 10 ** q[1]), float(1 - 10 ** q[0])]
    return res


def example_bootstrap(cells: List[dict], winner: float, n_boot: int,
                      rng: np.random.Generator) -> List[dict]:
    """Evaluation noise alone: paired bootstrap over the validation examples.

    Within one (size, seed) every beta2 scored the same examples, so resample
    example indices once and take each arm's median on the same draw. Uses
    each cell's best stable grid rate, the run that actually exists.
    """
    by = defaultdict(dict)
    for c in cells:
        t = tune(c)
        if t["grid_lr"] is None:
            continue
        i = int(np.where(c["lrs"] == t["grid_lr"])[0][0])
        pe = c["per_example"][i]
        if pe is not None:
            by[(c["size"], c["seed"])][c["beta2"]] = np.array(
                [np.nan if v is None else v for v in pe], dtype=float)
    out = []
    for (size, seed), arms in sorted(by.items()):
        if winner not in arms:
            continue
        w = arms[winner]
        for b, alt in sorted(arms.items()):
            if b == winner or alt.size != w.size:
                continue
            idx = rng.integers(0, w.size, size=(n_boot, w.size))
            d = np.nanmedian(alt[idx], axis=1) - np.nanmedian(w[idx], axis=1)
            out.append({"size": list(size), "seed": seed, "beta2": b,
                        "median_diff": float(np.nanmedian(alt) - np.nanmedian(w)),
                        "ci": [float(np.percentile(d, 2.5)),
                               float(np.percentile(d, 97.5))],
                        "p_alt_better": float(np.mean(d < 0))})
    return out


def analyse(cells: List[dict], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {"n_cells": len(cells), "by_metric": {}}
    for metric in METRICS:
        betas, blocks, tuned = block_matrix(cells, metric)
        if not blocks:
            continue
        sizes = sorted(blocks)
        m = {"beta2_grid": betas, "per_size": {}, "anova": anova(blocks)}
        sigma = m["anova"].get("sigma_error")
        all_rows = []
        for size in sizes:
            Y = np.vstack(list(blocks[size].values()))
            C = Y - Y.mean(axis=1, keepdims=True)
            all_rows.append(C)
            mean = C.mean(axis=0)
            w = int(np.argmin(mean))
            pair = {}
            for j, b in enumerate(betas):
                if j == w:
                    continue
                d = Y[:, j] - Y[:, w]
                mu, lo, hi = t_interval(d)
                entry = {"mean": mu, "ci_this_size_only": [lo, hi],
                         "seeds_winner_better": int(np.sum(d > 0)),
                         "n_seeds": int(d.size)}
                # Primary interval: the ANOVA's pooled error. Three seeds alone
                # leave 2 degrees of freedom (t = 4.30); the residual pools the
                # same seed-by-beta2 noise over every size (24 df at 3 x 3 x 5).
                if sigma and d.size:
                    h = (stats.t.ppf(1 - ALPHA / 2, m["anova"]["df_error"])
                         * sigma * math.sqrt(2 / d.size))
                    entry["ci"] = [mu - h, mu + h]
                else:
                    entry["ci"] = [lo, hi]
                pair[str(b)] = entry
            n_params = next(c["n_params"] for c in cells if c["size"] == size)
            m["per_size"][f"d{size[0]}/se{size[1]}"] = {
                "n_params": n_params, "seeds": sorted(blocks[size]),
                "tuned": Y.tolist(), "centred_mean": mean.tolist(),
                "winner": betas[w], "vs_winner": pair,
                "quadratic": quad_optimum(betas, mean, sigma, Y.shape[0], rng),
            }
        P = np.vstack(all_rows)
        pmean = P.mean(axis=0)
        w = int(np.argmin(pmean))
        pooled_pair, indist = {}, []
        for j, b in enumerate(betas):
            if j == w:
                continue
            mu, lo, hi = t_interval(P[:, j] - P[:, w])
            pooled_pair[str(b)] = {"mean": mu, "ci": [lo, hi]}
            if not (lo > 0):
                indist.append(b)
        m["pooled"] = {"centred_mean": pmean.tolist(), "winner": betas[w],
                       "vs_winner": pooled_pair, "n_blocks": int(P.shape[0]),
                       "indistinguishable_from_winner": indist,
                       "quadratic": quad_optimum(
                           betas, pmean, sigma, P.shape[0], rng)}
        # "extend" says which way STAGE=beta2x should add rates. Not for an
        # edge capped by divergence: the next rate up already blew up, so the
        # optimum is bounded, and a higher one would only diverge again.
        m["lr_edge_cells"] = [
            {"size": list(k[0]), "seed": k[1], "beta2": k[2], "edge": v["edge"],
             "extend": {"high": "up", "low": "down"}.get(v["edge"])}
            for k, v in sorted(tuned.items()) if v.get("on_edge")]
        result["by_metric"][metric] = m

    pooled = result["by_metric"].get("pooled")
    if pooled:
        result["example_bootstrap"] = example_bootstrap(
            cells, pooled["pooled"]["winner"], n_boot, rng)
        result["decision"] = decide(pooled)
    return result


def decide(m: dict) -> dict:
    """The pre-registered rule from the module docstring, applied."""
    betas, a = m["beta2_grid"], m["anova"]
    w = m["pooled"]["winner"]
    notes = []
    if "p_interaction" in a and a["p_interaction"] < ALPHA:
        verdict = "per-size"
        notes.append(f"size x beta2 interaction p = {a['p_interaction']:.3g}: "
                     "the best beta2 moves with scale; use per-size values")
    else:
        verdict = "single"
    if w in (min(betas), max(betas)):
        notes.append(f"winner {w} is at the end of the grid: not bracketed, "
                     "extend the grid before claiming it")
    indist = m["pooled"]["indistinguishable_from_winner"]
    if "p_beta2" in a and a["p_beta2"] >= ALPHA:
        notes.append(f"no beta2 effect detected at all (p = {a['p_beta2']:.3g})")
    if m["lr_edge_cells"]:
        notes.append(f"{len(m['lr_edge_cells'])} cell(s) did not bracket their "
                     "LR optimum; their tuned DeltaE is an upper bound")
    return {"verdict": verdict, "beta2": w,
            "per_size": {k: v["winner"] for k, v in m["per_size"].items()},
            "indistinguishable_from_winner": indist, "notes": notes}


# ------------------------------------------------------------------- report --
def report(result: dict) -> str:
    lines = []
    m = result["by_metric"].get("pooled")
    if not m:
        return "no complete blocks: nothing to compare"
    betas = m["beta2_grid"]
    a = m["anova"]
    lines.append(f"beta2 grid {betas}; {result['n_cells']} cells; "
                 f"{m['pooled']['n_blocks']} complete (size, seed) blocks")
    if "F_beta2" in a:
        lines.append(f"ANOVA  beta2: F({a['df_beta2']},{a['df_error']}) = "
                     f"{a['F_beta2']:.2f}, p = {a['p_beta2']:.3g}")
        if "F_interaction" in a:
            lines.append(f"       size x beta2: F({a['df_interaction']},"
                         f"{a['df_error']}) = {a['F_interaction']:.2f}, "
                         f"p = {a['p_interaction']:.3g}")
        lines.append(f"       residual sigma {a['sigma_error']:.3f} DeltaE")
    else:
        lines.append(f"ANOVA  {a.get('status')}")
    for name, s in m["per_size"].items():
        lines.append(f"\n{name} (N = {s['n_params']:,}), seeds {s['seeds']}: "
                     f"winner {s['winner']}")
        for b, p in s["vs_winner"].items():
            ci = ("" if math.isnan(p["ci"][0])
                  else f"  [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]")
            lines.append(f"   {b:>6} - winner  {p['mean']:+.2f}{ci}   winner "
                         f"better in {p['seeds_winner_better']}/{p['n_seeds']} seeds")
        q = s["quadratic"]
        if q.get("beta2_star") is not None:
            ci = q.get("beta2_ci")
            lines.append(f"   quadratic optimum beta2* = {q['beta2_star']:.4f}"
                         + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""))
    p = m["pooled"]
    lines.append(f"\nPOOLED winner {p['winner']}")
    q = p["quadratic"]
    if q.get("beta2_star") is not None:
        ci = q.get("beta2_ci")
        lines.append(f"   quadratic optimum beta2* = {q['beta2_star']:.4f}"
                     + (f" [{ci[0]:.4f}, {ci[1]:.4f}], interior in "
                        f"{q['interior_fraction']:.0%} of draws" if ci else ""))
    else:
        lines.append("   continuous optimum: " + q.get(
            "status", "the quadratic has no interior minimum on this grid"))
    for b, d in p["vs_winner"].items():
        ci = ("" if math.isnan(d["ci"][0])
              else f"  [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]")
        lines.append(f"   {b:>6} - winner  {d['mean']:+.2f}{ci}")
    for metric in ("low", "mid", "high"):
        mm = result["by_metric"].get(metric)
        if mm:
            lines.append(f"   {metric:>4}-chroma winner {mm['pooled']['winner']}")
    eb = result.get("example_bootstrap") or []
    if eb:
        lines.append("\nevaluation noise (paired bootstrap over examples):")
        for e in eb:
            lines.append(f"   d{e['size'][0]}/se{e['size'][1]} s{e['seed']} "
                         f"{e['beta2']:>6}: {e['median_diff']:+.2f} "
                         f"[{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]")
    d = result["decision"]
    lines.append(f"\nDECISION: {d['verdict']}, beta2 = {d['beta2']}; "
                 f"per size {d['per_size']}")
    if d["indistinguishable_from_winner"]:
        lines.append(f"   not distinguishable from the winner at 95%: "
                     f"{d['indistinguishable_from_winner']}")
    for n in d["notes"]:
        lines.append(f"   note: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------- plot --
#: Ordinal ramp for beta2 (one hue, light = short horizon, dark = long),
#: validated with the dataviz skill's validator (--ordinal, light surface).
BETA_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
#: Model sizes: categorical, validated all-pairs. Aqua sits under 3:1 on the
#: surface, so sizes also carry marker shapes and direct labels.
SIZE_COLORS = ("#eb6834", "#1baf7a", "#4a3aa7")
SIZE_MARKERS = ("o", "s", "D")
#: Display only: a run at or above this DeltaE learned nothing usable (a model
#: emitting nothing useful scores ~28.6), even if its CE did not trip the
#: divergence screen. Drawn as a failure at the top of its panel so one such
#: point does not flatten every curve. The analysis keeps its value.
FAILED_DE = 25.0
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def plot(cells: List[dict], result: dict, out: Path, title_note: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = result["by_metric"]["pooled"]
    betas = m["beta2_grid"]
    sizes = sorted({c["size"] for c in cells})
    ramp = {b: BETA_RAMP[round(i * (len(BETA_RAMP) - 1) / max(1, len(betas) - 1))]
            for i, b in enumerate(betas)}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK})
    fig = plt.figure(figsize=(15, 9.2), facecolor=SURF)
    gs = fig.add_gridspec(2, max(3, len(sizes)), hspace=0.42, wspace=0.28)

    def style(ax):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Row 1: the LR curve of every beta2, per size. Seed mean as the line,
    # individual seeds as faint points, divergence as an x at the top.
    for k, size in enumerate(sizes):
        ax = fig.add_subplot(gs[0, k])
        style(ax)
        sc = [c for c in cells if c["size"] == size]
        shown = [v for c in sc for v, dv in zip(c["curve"]["pooled"], c["diverged"])
                 if v is not None and not dv and v < FAILED_DE]
        top, bot = max(shown), min(shown)
        ytop = top + 0.12 * (top - bot + 1e-9)
        for b in betas:
            cb = [c for c in sc if c["beta2"] == b]
            if not cb:
                continue
            rel = defaultdict(list)
            for c in cb:
                for lr, v, dv in zip(c["lrs"] / c["prior"], c["curve"]["pooled"],
                                     c["diverged"]):
                    key = round(float(np.log2(lr)) * 4) / 4
                    failed = dv or v is None or v >= FAILED_DE
                    rel[key].append(None if failed else v)
            xs = sorted(rel)
            mean_x, mean_y = [], []
            for x in xs:
                vals = [v for v in rel[x] if v is not None]
                if len(vals) == len(rel[x]) and vals:
                    mean_x.append(2 ** x)
                    mean_y.append(float(np.mean(vals)))
                    ax.scatter([2 ** x] * len(vals), vals, s=10, color=ramp[b],
                               alpha=0.35, linewidths=0, zorder=2)
                else:
                    ax.scatter([2 ** x], [ytop], marker="x", s=30,
                               color=ramp[b], linewidths=1.5, zorder=3)
            ax.plot(mean_x, mean_y, "-o", color=ramp[b], linewidth=2,
                    markersize=4, label=f"β₂ = {b:g}", zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_ylim(bot - 0.08 * (top - bot + 1e-9), ytop + 0.06 * (top - bot + 1e-9))
        n = sc[0]["n_params"]
        ax.set_title(f"{'ABC'[k]}.  N = {n / 1e6:.2g}M  (d{size[0]}/se{size[1]}),"
                     f"  D = {sc[0]['D'] / 1e6:.2f}M", loc="left", fontsize=10)
        ax.set_xlabel("learning rate / prior")
        if k == 0:
            ax.set_ylabel("val ΔE₀₀ (median, mean over seeds)")
            ax.legend(frameon=False, fontsize=8, loc="upper center")
        ax.text(0.99, 0.02, "× at top: diverged or failed", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=MUTED)

    x_of = lambda b: 1 - b
    xt = [x_of(b) for b in betas]

    # D: block-centred tuned DeltaE against beta2, per size and pooled.
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    sig = m["anova"].get("sigma_error")
    for k, (name, s) in enumerate(m["per_size"].items()):
        mean = np.array(s["centred_mean"])
        ns = len(s["seeds"])
        col, mk = SIZE_COLORS[k % 3], SIZE_MARKERS[k % 3]
        if sig and ns > 1:
            h = stats.t.ppf(0.975, m["anova"]["df_error"]) * sig / math.sqrt(ns)
            ax.errorbar(xt, mean, yerr=h, color=col, marker=mk, markersize=5,
                        linewidth=1.5, capsize=2, label=f"N = {s['n_params'] / 1e6:.2g}M")
        else:
            ax.plot(xt, mean, color=col, marker=mk, markersize=5, linewidth=1.5,
                    label=f"N = {s['n_params'] / 1e6:.2g}M")
    p = m["pooled"]
    ax.plot(xt, p["centred_mean"], color=INK, linewidth=2.4, marker="o",
            markersize=6, label="pooled", zorder=5)
    q = p["quadratic"]
    if q.get("beta2_star") is not None:
        xx = np.geomspace(min(xt), max(xt), 200)
        a, b_, c = q["coef"]
        ax.plot(xx, a * np.log10(xx) ** 2 + b_ * np.log10(xx) + c, ":",
                color=INK2, linewidth=1.2)
        if q.get("beta2_ci"):
            ax.axvspan(x_of(q["beta2_ci"][1]), x_of(q["beta2_ci"][0]),
                       color=GRID, alpha=0.7, zorder=0)
        ax.axvline(x_of(q["beta2_star"]), color=INK2, linewidth=1, linestyle="--")
        ax.text(x_of(q["beta2_star"]), ax.get_ylim()[1], f" β₂* ≈ {q['beta2_star']:.3f}",
                va="top", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.minorticks_off()
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("AdamW β₂  (log scale in 1 − β₂)")
    ax.set_ylabel("tuned ΔE₀₀, centred per (size, seed)")
    ax.set_title("D.  Each β₂ at its own best learning rate", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # E: where the LR optimum sits for each beta2.
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    for k, size in enumerate(sizes):
        ys, edge = [], []
        for b in betas:
            tt = [(tune(c), c["prior"]) for c in cells
                  if c["size"] == size and c["beta2"] == b]
            rel = [t["lr"] / pr for t, pr in tt if t["lr"]]
            ys.append(float(np.exp(np.mean(np.log(rel)))) if rel else np.nan)
            edge.append(any(t["on_edge"] for t, _ in tt))
        n = next(c["n_params"] for c in cells if c["size"] == size)
        col, mk = SIZE_COLORS[k % 3], SIZE_MARKERS[k % 3]
        ax.plot(xt, ys, color=col, linewidth=1.5, label=f"N = {n / 1e6:.2g}M")
        for x_, y_, e_ in zip(xt, ys, edge):
            ax.plot([x_], [y_], marker=mk, markersize=6, color=col,
                    markerfacecolor=SURF if e_ else col, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    lo_e, hi_e = ax.get_ylim()
    ticks = [2.0 ** k for k in np.arange(np.floor(np.log2(lo_e) * 2) / 2,
                                         np.log2(hi_e) + 0.01, 0.5)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2g}×" for t in ticks])
    ax.invert_xaxis()
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.minorticks_off()
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_xlabel("AdamW β₂")
    ax.set_ylabel("optimal LR / prior (geometric mean over seeds)")
    ax.set_title("E.  Where each β₂'s LR optimum sits", loc="left", fontsize=10)
    ax.text(1.0, -0.17, "hollow: not bracketed by the LR grid",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color=MUTED)
    ax.legend(frameon=False, fontsize=8)

    # F: forest plot, every alternative minus the winner, per size and pooled.
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    rows, labels = [], []
    groups = list(m["per_size"].items()) + [("pooled", p)]
    for gi, (name, s) in enumerate(groups):
        for b, d in s["vs_winner"].items():
            rows.append((gi, d, name))
            labels.append(f"{b} vs {s['winner']}")
    ypos = np.arange(len(rows))[::-1]
    for y, (gi, d, name) in zip(ypos, rows):
        col = INK if name == "pooled" else SIZE_COLORS[gi % 3]
        mk = "o" if name == "pooled" else SIZE_MARKERS[gi % 3]
        lo, hi = d["ci"]
        if not math.isnan(lo):
            ax.plot([lo, hi], [y, y], color=col, linewidth=2)
        ax.plot([d["mean"]], [y], marker=mk, color=col, markersize=5)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("ΔE₀₀ difference (positive: the winner is better)")
    ax.set_title("F.  Every alternative against the winner, 95% CI", loc="left",
                 fontsize=10)
    # Group labels on the right edge, in ink.
    start = 0
    for gi, (name, s) in enumerate(groups):
        cnt = len(s["vs_winner"])
        if cnt:
            mid = ypos[start:start + cnt].mean()
            lab = name if name == "pooled" else f"N = {s['n_params'] / 1e6:.2g}M"
            ax.text(1.01, mid, lab, transform=ax.get_yaxis_transform(),
                    fontsize=8, color=INK2, va="center")
        start += cnt

    d = result["decision"]
    a = m["anova"]
    stat = (f"β₂ effect F({a['df_beta2']},{a['df_error']}) = {a['F_beta2']:.1f}, "
            f"p = {a['p_beta2']:.2g}" if "F_beta2" in a else a.get("status", ""))
    if "p_interaction" in a:
        stat += f";  size × β₂ p = {a['p_interaction']:.2g}"
    fig.suptitle(f"AdamW β₂ for INDIGO: pooled winner β₂ = {d['beta2']:g}"
                 + (f"   ({title_note})" if title_note else ""),
                 x=0.06, ha="left", fontsize=13, color=INK, y=0.995)
    n_lr = int(np.median([len(c["lrs"]) for c in cells]))
    fig.text(0.06, 0.955, stat + "   ·   " + f"{p['n_blocks']} paired (size, seed) "
             f"blocks; {n_lr} learning rates per β₂ per block", fontsize=9,
             color=INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", nargs="+",
                   default=["outputs/lr_search/beta2/cross_attn",
                            "outputs/lr_search/beta2_ext/cross_attn"],
                   help="every directory holding cells; a cell spread over "
                        "several (the main grid plus beta2x's extra rates) is "
                        "merged. Missing directories are skipped.")
    p.add_argument("--output", default="analyses/scaling/results/beta2_fit.json")
    p.add_argument("--figure", default="analyses/scaling/results/beta2.png")
    p.add_argument("--title-note", default="")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--estimator", choices=("akima", "quadratic"),
                   default="akima",
                   help="how a cell's LR curve becomes one number; see tune()")
    a = p.parse_args()
    global ESTIMATOR
    ESTIMATOR = a.estimator

    cells = load_cells(a.results_dir)
    if not cells:
        sys.exit(f"no lr_search_*.json under {a.results_dir}")
    result = analyse(cells, a.n_boot, a.seed)
    text = report(result)
    print(text)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({**result, "report": text, "sources": a.results_dir},
              open(a.output, "w"), indent=1)
    plot(cells, result, Path(a.figure), a.title_note)
    print(f"\n[INFO] {a.output}\n[INFO] {a.figure} (+ .pdf)")


if __name__ == "__main__":
    main()
