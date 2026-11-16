#!/usr/bin/env python3
"""The paper's headline numbers, from test_eval.py runs.

For each run (a folder of test_eval.py output, possibly split into row
chunks): the share of test rows under DeltaE00 1, 2, 3 and 5, and the median
DeltaE00, per tier (A, B, and both pooled), overall and by target chroma
C* = sqrt(a*^2 + b*^2) in buckets < 8, 8 to 30, >= 30. Every number carries a
95% bootstrap interval (rows resampled with replacement, 2000 draws,
percentile; the pooled tier resamples within each tier).

A row that failed (solve() raised, or returned no candidate) counts as a
failure at every threshold and as +inf in the median: it is never dropped.
A row missing from the output altogether (a chunk that never ran) counts the
same way and is reported as missing, so an incomplete run cannot look better.

Comparisons (--compare A:B) are paired: the same rows under both runs,
resampled together, giving A minus B with its interval.

Each run's settings are read from its summary.json files, which must agree
across chunks (ROW_OFFSET aside) and are written into the output.

    python scripts/paper_numbers.py \\
        --run random=outputs/paper/strong_random \\
        --run INDIGO=outputs/paper/strong_model \\
        --run INDIGO_N50=outputs/paper/budget50_model \\
        --compare INDIGO:random --compare INDIGO_N50:random
    # -> results/paper/headline.json and results/paper/headline.md

Needs numpy only (no torch).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
BUCKETS = (("C<8", 0.0, 8.0), ("8<=C<30", 8.0, 30.0), ("C>=30", 30.0, math.inf))
TIERS = ("a", "b")
# Settings that must match across a run's chunks (and that define the run).
SETTING_KEYS = ("ensemble_n", "top_k", "refine_iters", "proposer", "seed",
                "temperature", "tolerance", "weight_lambda", "mc_samples")


def load_run(folder: Path) -> Tuple[Dict[str, Dict[int, dict]], dict]:
    """({tier: {row_idx: row}}, settings) for every chunk under folder."""
    rows: Dict[str, Dict[int, dict]] = {t: {} for t in TIERS}
    settings: Optional[dict] = None
    sha = set()
    found = 0
    for tier in TIERS:
        for rp in sorted(glob.glob(str(folder / "**" / f"tier_{tier}" / "rows.json"),
                                   recursive=True)):
            found += 1
            sp = Path(rp).with_name("summary.json")
            if sp.is_file():
                summ = json.load(open(sp))
                args = summ.get("args", {})
                # Runs from before the proposer was recorded used the model.
                s = {k: args.get(k, "model" if k == "proposer" else None)
                     for k in SETTING_KEYS}
                if settings is None:
                    settings = s
                elif s != settings:
                    diff = {k: (settings[k], s[k]) for k in SETTING_KEYS if settings[k] != s[k]}
                    sys.exit(f"[paper_numbers] {folder}: chunks disagree on {diff} ({sp})")
                if summ.get("model_sha256"):
                    sha.add(summ["model_sha256"])
            for r in json.load(open(rp)):
                i = int(r["row_idx"])
                old = rows[tier].get(i)
                if old is not None and old.get("delta_e") != r.get("delta_e"):
                    sys.exit(f"[paper_numbers] {folder}: tier {tier} row {i} appears twice "
                             f"with different results ({old.get('delta_e')} vs {r.get('delta_e')})")
                rows[tier][i] = r
    if not found:
        sys.exit(f"[paper_numbers] no tier_*/rows.json under {folder}")
    if len(sha) > 1:
        sys.exit(f"[paper_numbers] {folder}: chunks scored different checkpoints {sorted(sha)}")
    return rows, {**(settings or {}), "model_sha256": next(iter(sha), None)}


def chroma(lab) -> float:
    return float(math.hypot(lab[1], lab[2]))


def bucket_of(c: Optional[float]) -> Optional[str]:
    if c is None:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= c < hi:
            return name
    return None


def table(rows: Dict[int, dict], expect: List[int], targets: Dict[int, list]):
    """(delta_e array with inf for failed or missing, chroma bucket per row,
    n_failed, n_missing), over the expected rows in order."""
    de, bk = [], []
    n_failed = n_missing = 0
    for i in expect:
        r = rows.get(i)
        if r is None:
            n_missing += 1
            de.append(math.inf)
        elif r.get("delta_e") is None or not math.isfinite(float(r["delta_e"])):
            n_failed += 1
            de.append(math.inf)
        else:
            de.append(float(r["delta_e"]))
        lab = (r or {}).get("target_lab") or targets.get(i)
        bk.append(bucket_of(chroma(lab)) if lab else None)
    return np.array(de), np.array(bk, dtype=object), n_failed, n_missing


def summarize(vals: np.ndarray, boots: np.ndarray) -> dict:
    """Point estimate and 95% interval for each statistic. vals [n],
    boots [draws, n] resampled copies (strata already combined)."""
    out = {}
    for t in THRESHOLDS:
        est = float(np.mean(vals < t))
        bs = np.mean(boots < t, axis=1)
        out[f"pass_lt_{t:g}"] = {"est": est, "lo": float(np.percentile(bs, 2.5)),
                                 "hi": float(np.percentile(bs, 97.5))}
    med = float(np.median(vals))
    bm = np.median(boots, axis=1)
    out["median"] = {"est": med, "lo": float(np.percentile(bm, 2.5)),
                     "hi": float(np.percentile(bm, 97.5))}
    return out


def resample(strata: List[np.ndarray], rng, draws: int) -> List[np.ndarray]:
    """Bootstrap indices per stratum: [draws, n_s] each."""
    return [rng.integers(0, len(s), size=(draws, len(s))) for s in strata]


def stats_for(strata: List[np.ndarray], idx: List[np.ndarray]) -> dict:
    vals = np.concatenate(strata)
    boots = np.concatenate([s[i] for s, i in zip(strata, idx)], axis=1)
    return {"n": int(len(vals)), **summarize(vals, boots)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="NAME=FOLDER, a test_eval.py output root (chunks found below it)")
    p.add_argument("--compare", action="append", default=[],
                   help="A:B, a paired comparison A minus B on the same rows")
    p.add_argument("--rows", type=int, default=500,
                   help="rows expected per tier, 0..ROWS-1 (default 500)")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--out", default="results/paper/headline.json")
    a = p.parse_args()

    runs = {}
    for spec in a.run:
        name, _, folder = spec.partition("=")
        if not folder:
            p.error(f"--run needs NAME=FOLDER, got {spec!r}")
        runs[name] = load_run(Path(folder))
        runs[name][1]["folder"] = folder
    expect = list(range(a.rows))
    # A failed or missing row's target, from whichever run has it, so it still
    # falls in its chroma bucket.
    targets = {t: {} for t in TIERS}
    for rows, _ in runs.values():
        for t in TIERS:
            for i, r in rows[t].items():
                if r.get("target_lab"):
                    targets[t].setdefault(i, r["target_lab"])

    tab = {name: {t: table(rows[t], expect, targets[t]) for t in TIERS}
           for name, (rows, _) in runs.items()}
    rng = np.random.default_rng(a.seed)
    # One set of bootstrap indices per tier, shared by every run: the rows are
    # the same, so runs are resampled identically and comparisons are paired.
    idx = {t: rng.integers(0, a.rows, size=(a.draws, a.rows)) for t in TIERS}

    out = {"thresholds": list(THRESHOLDS), "chroma_buckets": [b[0] for b in BUCKETS],
           "rows_per_tier": a.rows, "bootstrap": {"draws": a.draws, "seed": a.seed,
                                                  "interval": "95% percentile"},
           "failed_rows": "counted as failures (DeltaE = inf), never dropped",
           "runs": {}, "comparisons": {}}
    for name, (rows, settings) in runs.items():
        res = {"settings": settings, "tiers": {}}
        for tier_name, tiers in (("A", ["a"]), ("B", ["b"]), ("A+B", ["a", "b"])):
            de = [tab[name][t][0] for t in tiers]
            bk = [tab[name][t][1] for t in tiers]
            ent = {"all": stats_for(de, [idx[t] for t in tiers]),
                   "n_failed": sum(tab[name][t][2] for t in tiers),
                   "n_missing": sum(tab[name][t][3] for t in tiers),
                   "by_chroma": {}}
            for bname, _, _ in BUCKETS:
                masks = [b == bname for b in bk]
                if not any(m.any() for m in masks):
                    continue
                sub = [d[m] for d, m in zip(de, masks)]
                sidx = [rng.integers(0, max(int(m.sum()), 1), size=(a.draws, int(m.sum())))
                        for m in masks]
                ent["by_chroma"][bname] = stats_for(sub, sidx)
            res["tiers"][tier_name] = ent
        out["runs"][name] = res

    for spec in a.compare:
        A, _, B = spec.partition(":")
        if A not in runs or B not in runs:
            p.error(f"--compare {spec}: both names must be given with --run")
        comp = {}
        for tier_name, tiers in (("A", ["a"]), ("B", ["b"]), ("A+B", ["a", "b"])):
            da = np.concatenate([tab[A][t][0] for t in tiers])
            db = np.concatenate([tab[B][t][0] for t in tiers])
            ii = np.concatenate([idx[t] + k * a.rows for k, t in enumerate(tiers)], axis=1)
            ent = {}
            for t in THRESHOLDS:
                diff = np.mean(da < t) - np.mean(db < t)
                bs = np.mean(da[ii] < t, axis=1) - np.mean(db[ii] < t, axis=1)
                ent[f"pass_lt_{t:g}"] = {"est": float(diff),
                                         "lo": float(np.percentile(bs, 2.5)),
                                         "hi": float(np.percentile(bs, 97.5))}
            ent["rows_better"] = float(np.mean(da < db))
            ent["rows_worse"] = float(np.mean(da > db))
            comp[tier_name] = ent
        out["comparisons"][f"{A} - {B}"] = comp

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_finite(out), open(outp, "w"), indent=2)
    md = outp.with_suffix(".md")
    md.write_text(markdown(out))
    print(f"wrote {outp} and {md}")
    for name, res in out["runs"].items():
        for tier_name, ent in res["tiers"].items():
            if ent["n_missing"]:
                print(f"[WARN] {name} tier {tier_name}: {ent['n_missing']} row(s) missing, "
                      "counted as failures")


def _finite(x):
    """Strict JSON: an infinite median (most rows failed) is written as null."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def pct(c: dict) -> str:
    return f"{100 * c['est']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def num(c: dict) -> str:
    f = lambda x: "fail" if not math.isfinite(x) else f"{x:.2f}"
    return f"{f(c['est'])} [{f(c['lo'])}, {f(c['hi'])}]"


def markdown(out: dict) -> str:
    th = out["thresholds"]
    head = ("| run | n | " + " | ".join(f"ΔE<{t:g} (%)" for t in th)
            + " | median ΔE | failed | missing |\n|" + "---|" * (len(th) + 5) + "\n")
    lines = ["# Headline numbers", "",
             f"95% bootstrap intervals ({out['bootstrap']['draws']} draws, rows resampled). "
             "Failed and missing rows count as failures.", ""]
    settings = []
    for name, res in out["runs"].items():
        s = res["settings"]
        settings.append(f"- **{name}**: proposer {s.get('proposer')}, N = {s.get('ensemble_n')}, "
                        f"top-k {s.get('top_k')}, {s.get('refine_iters')} refinement steps, "
                        f"seed {s.get('seed')}, checkpoint {s.get('model_sha256')}")
    lines += ["## Runs", ""] + settings + [""]
    for tier_name in ("A", "B", "A+B"):
        lines += [f"## Tier {tier_name}", "", head.rstrip("\n")]
        for name, res in out["runs"].items():
            ent = res["tiers"][tier_name]
            c = ent["all"]
            lines.append(f"| {name} | {c['n']} | " + " | ".join(pct(c[f'pass_lt_{t:g}']) for t in th)
                         + f" | {num(c['median'])} | {ent['n_failed']} | {ent['n_missing']} |")
        lines.append("")
        lines += [f"### Tier {tier_name} by target chroma", "",
                  "| run | chroma | n | " + " | ".join(f"ΔE<{t:g} (%)" for t in th)
                  + " | median ΔE |", "|" + "---|" * (len(th) + 4)]
        for name, res in out["runs"].items():
            for b, c in res["tiers"][tier_name]["by_chroma"].items():
                lines.append(f"| {name} | {b} | {c['n']} | "
                             + " | ".join(pct(c[f'pass_lt_{t:g}']) for t in th)
                             + f" | {num(c['median'])} |")
        lines.append("")
    if out["comparisons"]:
        lines += ["## Paired comparisons (same rows, percentage points)", "",
                  "| comparison | tier | " + " | ".join(f"Δ pass<{t:g}" for t in th)
                  + " | rows better | rows worse |", "|" + "---|" * (len(th) + 4)]
        for key, comp in out["comparisons"].items():
            for tier_name, ent in comp.items():
                lines.append(f"| {key} | {tier_name} | "
                             + " | ".join(f"{100 * ent[f'pass_lt_{t:g}']['est']:+.1f} "
                                          f"[{100 * ent[f'pass_lt_{t:g}']['lo']:+.1f}, "
                                          f"{100 * ent[f'pass_lt_{t:g}']['hi']:+.1f}]" for t in th)
                             + f" | {100 * ent['rows_better']:.0f}% | {100 * ent['rows_worse']:.0f}% |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
