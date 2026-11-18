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
A row missing from a tier the run did score (a chunk that never ran) counts
the same way and is reported as missing, so an incomplete run cannot look
better. A tier the run never scored is reported as not run.

Tier B by held-out real material. The held-out reals are
create_dataset/src/pool_sampler.py's HELD_OUT_REAL_MATERIALS, the list tier B
was generated from (generate_test_sets.py passes --use-held-out-reals). With
--test-dir, the list is checked against the shards themselves: every real
material in a tier B pool must be on it, no tier A pool may hold one, and
every tier B manifest must say use_held_out_reals; any failure stops the
script. Each tier B row then falls in:
  (a) its pool holds at least one held-out real material
  (b) its chosen stack uses at least one (the strictest generalization test;
      it conditions on what the search chose and on the row succeeding, so
      its failed rows are those of (a), reported beside it)
  (c) its pool holds none
The pool comes from the row itself (test_eval.py records it since Nov 16) or,
for older runs, from the shard row with the same target Lab. A row whose pool
or stack cannot be traced is counted and left out of (a), (b), (c), never
guessed.

Comparisons (--compare A:B) are paired: the same rows under both runs,
resampled together, giving A minus B with its interval, overall and per
chroma bucket.

Each run's settings are read from its summary.json files, which must agree
across chunks (ROW_OFFSET aside); they and the folder every number came from
are written into the output.

    python scripts/paper_numbers.py \\
        --run random=outputs/paper/strong_random \\
        --run INDIGO=outputs/paper/strong_model \\
        --run INDIGO_N50=outputs/paper/budget50_model \\
        --compare INDIGO:random --compare INDIGO_N50:random --test-dir data/test
    # -> results/paper/headline.json and results/paper/headline.md

Needs numpy (and pyarrow for --test-dir); no torch.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
BUCKETS = (("C<8", 0.0, 8.0), ("8<=C<30", 8.0, 30.0), ("C>=30", 30.0, math.inf))
TIERS = ("a", "b")
TIER_SETS = (("A", ("a",)), ("B", ("b",)), ("A+B", ("a", "b")))
# Settings that must match across a run's chunks (and that define the run).
SETTING_KEYS = ("ensemble_n", "top_k", "refine_iters", "proposer", "seed",
                "temperature", "tolerance", "weight_lambda", "mc_samples",
                "layer_count", "split")
HELD_OUT_SOURCE = "create_dataset/src/pool_sampler.py:HELD_OUT_REAL_MATERIALS"


def held_out_list() -> List[str]:
    from create_dataset.src.pool_sampler import HELD_OUT_REAL_MATERIALS
    return list(HELD_OUT_REAL_MATERIALS)


SAMPLER_BUG = "out of range [0, 3201)"


def is_sampler_bug(r: dict) -> bool:
    """The generate crash fixed Nov 18 (inference/src/generate.py:_sample_token):
    a float32 rounding draw past the vocabulary, not a search outcome."""
    return SAMPLER_BUG in str(r.get("error") or "")


def load_run(folder: Path):
    """({tier: {row_idx: row}}, settings, [rows.json files]) over all chunks."""
    rows: Dict[str, Dict[int, dict]] = {t: {} for t in TIERS}
    settings: Optional[dict] = None
    sha = set()
    files = []
    replaced = []
    for tier in TIERS:
        for rp in sorted(glob.glob(str(folder / "**" / f"tier_{tier}" / "rows.json"),
                                   recursive=True)):
            files.append(rp)
            sp = Path(rp).with_name("summary.json")
            if sp.is_file():
                summ = json.load(open(sp))
                args = summ.get("args", {})
                # Runs from before these were recorded used the model, free
                # layer count and the train split.
                default = {"proposer": "model", "layer_count": 0, "split": "train"}
                s = {k: args.get(k, default.get(k)) for k in SETTING_KEYS}
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
                    # A row rerun after the sampler's token-3201 crash (fixed
                    # Nov 18) replaces the crashed copy; nothing else may.
                    if is_sampler_bug(old) and not r.get("error"):
                        replaced.append((tier, i))
                    elif is_sampler_bug(r) and not old.get("error"):
                        replaced.append((tier, i))
                        continue
                    else:
                        sys.exit(f"[paper_numbers] {folder}: tier {tier} row {i} appears twice "
                                 f"with different results ({old.get('delta_e')} vs "
                                 f"{r.get('delta_e')})")
                rows[tier][i] = r
    if not files:
        sys.exit(f"[paper_numbers] no tier_*/rows.json under {folder}")
    if len(sha) > 1:
        sys.exit(f"[paper_numbers] {folder}: chunks scored different checkpoints {sorted(sha)}")
    return rows, {**(settings or {}), "model_sha256": next(iter(sha), None),
                  "rerun_after_sampler_bug": sorted(replaced)}, files


def chroma(lab) -> float:
    return float(math.hypot(lab[1], lab[2]))


def bucket_of(c: Optional[float]) -> Optional[str]:
    if c is None:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= c < hi:
            return name
    return None


def de_of(r: Optional[dict]) -> float:
    if r is None or r.get("delta_e") is None or not math.isfinite(float(r["delta_e"])):
        return math.inf
    return float(r["delta_e"])


def table(rows: Dict[int, dict], expect: List[int], targets: Dict[int, list]):
    """(delta_e array with inf for failed or missing, chroma bucket per row,
    n_failed, n_missing), over the expected rows in order."""
    de, bk = [], []
    n_failed = n_missing = 0
    for i in expect:
        r = rows.get(i)
        if r is None:
            n_missing += 1
        elif not math.isfinite(de_of(r)):
            n_failed += 1
        de.append(de_of(r))
        lab = (r or {}).get("target_lab") or targets.get(i)
        bk.append(bucket_of(chroma(lab)) if lab else None)
    return np.array(de), np.array(bk, dtype=object), n_failed, n_missing


def summarize(vals: np.ndarray, boots: np.ndarray) -> dict:
    """Point estimate and 95% interval for each statistic. vals [n],
    boots [draws, n] resampled copies (strata already combined)."""
    out = {}
    for t in THRESHOLDS:
        bs = np.mean(boots < t, axis=1)
        out[f"pass_lt_{t:g}"] = {"est": float(np.mean(vals < t)),
                                 "lo": float(np.percentile(bs, 2.5)),
                                 "hi": float(np.percentile(bs, 97.5))}
    bm = np.median(boots, axis=1)
    out["median"] = {"est": float(np.median(vals)), "lo": float(np.percentile(bm, 2.5)),
                     "hi": float(np.percentile(bm, 97.5))}
    return out


def stats_for(strata: List[np.ndarray], idx: List[np.ndarray]) -> dict:
    vals = np.concatenate(strata)
    boots = np.concatenate([s[i] for s, i in zip(strata, idx)], axis=1)
    return {"n": int(len(vals)), **summarize(vals, boots)}


def stats_independent(vals: np.ndarray, rng, draws: int) -> Optional[dict]:
    if len(vals) == 0:
        return None
    return stats_for([vals], [rng.integers(0, len(vals), size=(draws, len(vals)))])


# --------------------------------------------------------------------------
# Tier B pools: from the rows, or from the shards by target Lab
# --------------------------------------------------------------------------

class ShardPools:
    """Pool names and sources of every test row, keyed by target Lab, read
    straight from the shards (no dataset-loader ordering assumed)."""

    def __init__(self, test_dir: Path):
        import pyarrow.parquet as pq
        self.index: Dict[str, Dict[tuple, list]] = {t: {} for t in TIERS}
        self.reals: Dict[str, set] = {t: set() for t in TIERS}
        self.manifest_flags: Dict[str, list] = {t: [] for t in TIERS}
        self.n_rows: Dict[str, int] = {t: 0 for t in TIERS}
        for t in TIERS:
            for sp in sorted(glob.glob(str(test_dir / f"tier_{t}" / "**" / "shard_*.parquet"),
                                       recursive=True)):
                tab = pq.read_table(sp, columns=["lab", "pool_names", "pool_sources"]).to_pydict()
                for lab, names, srcs in zip(tab["lab"], tab["pool_names"], tab["pool_sources"]):
                    lab = np.asarray(json.loads(lab) if isinstance(lab, str) else lab, float)
                    self.index[t].setdefault(tuple(np.round(lab, 2)), []).append(
                        (lab, list(names), list(srcs)))
                    self.n_rows[t] += 1
                    for n, s in zip(names, srcs):
                        if not _synthetic(n):
                            self.reals[t].add(n)
                mf = Path(sp).with_suffix(".manifest.json")
                if mf.is_file():
                    self.manifest_flags[t].append(bool(json.load(open(mf)).get("use_held_out_reals")))

    def lookup(self, tier: str, lab) -> Optional[Tuple[list, list]]:
        lab = np.asarray(lab, float)
        hits = []
        key = tuple(np.round(lab, 2))
        # Neighbouring keys too, in case rounding falls on a boundary.
        for dk in [(0, 0, 0)] + [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                                 for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]:
            k2 = tuple(round(a + 0.01 * d, 2) for a, d in zip(key, dk))
            for cand in self.index[tier].get(k2, []):
                if np.max(np.abs(cand[0] - lab)) < 2e-3:
                    hits.append(cand)
        pools = {(tuple(h[1]), tuple(h[2])) for h in hits}
        if len(pools) != 1:
            return None           # not found, or two rows share a target with different pools
        names, srcs = next(iter(pools))
        return list(names), list(srcs)


def _synthetic(name: str) -> bool:
    """Synthetic materials are named for how they were made: perturb_*(X),
    interp(X,Y,w=..), lorentz_*. Real ones carry the bare material name."""
    return "(" in name or name.startswith("lorentz")


def verify_held_out(shards: ShardPools, held: List[str]) -> dict:
    """The held-out list against the test data itself."""
    tb, ta = shards.reals["b"], shards.reals["a"]
    res = {"tier_b_rows": shards.n_rows["b"], "tier_a_rows": shards.n_rows["a"],
           "real_materials_in_tier_b_pools": sorted(tb),
           "real_materials_in_tier_a_pools": sorted(ta),
           "tier_b_manifests_use_held_out_reals": shards.manifest_flags["b"]}
    problems = []
    if shards.n_rows["b"] == 0:
        problems.append("no tier B shards found")
    if tb - set(held):
        problems.append(f"tier B pools hold reals not on the list: {sorted(tb - set(held))}")
    if ta & set(held):
        problems.append(f"tier A pools hold held-out reals: {sorted(ta & set(held))}")
    if shards.manifest_flags["b"] and not all(shards.manifest_flags["b"]):
        problems.append("a tier B manifest says use_held_out_reals is false")
    res["verified"] = not problems
    res["problems"] = problems
    return res


def tier_b_split(rows: Dict[int, dict], expect: List[int], held: set,
                 shards: Optional[ShardPools]) -> dict:
    """Per row: group memberships, or why the row could not be traced."""
    groups = {"a": [], "b": [], "c": []}
    untr_pool, untr_stack, failed_in_a = [], [], 0
    pool_from = {"row": 0, "shards": 0}
    for i in expect:
        r = rows.get(i)
        names = (r or {}).get("pool_names")
        if names:
            pool_from["row"] += 1
        elif r is not None and r.get("target_lab") is not None and shards is not None:
            hit = shards.lookup("b", r["target_lab"])
            if hit:
                names = hit[0]
                pool_from["shards"] += 1
        if not names:
            untr_pool.append(i)
            continue
        has = any(n in held for n in names)
        groups["a" if has else "c"].append(i)
        if has and not math.isfinite(de_of(r)):
            failed_in_a += 1
        layers = (r or {}).get("layers") or []
        if layers and math.isfinite(de_of(r)):
            used = [l["material"] for l in layers]
            if not all(u in names for u in used):
                untr_stack.append(i)
            elif any(u in held for u in used):
                groups["b"].append(i)
    return {"groups": groups, "untraceable_pool": untr_pool,
            "untraceable_stack": untr_stack, "failed_in_a": failed_in_a,
            "pool_from": pool_from}


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="NAME=FOLDER, a test_eval.py output root (chunks found below it)")
    p.add_argument("--compare", action="append", default=[],
                   help="A:B, a paired comparison A minus B on the same rows")
    p.add_argument("--rows", type=int, default=500,
                   help="rows expected per scored tier, 0..ROWS-1 (default 500)")
    p.add_argument("--test-dir", default=None,
                   help="test set root (tier_a/, tier_b/ shards): verifies the held-out "
                        "list and traces pools for runs that did not record them")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--out", default="results/paper/headline.json")
    a = p.parse_args()

    held = held_out_list()
    shards = None
    held_check = {"verified": False, "problems": ["no --test-dir given"
                                                  ]}
    if a.test_dir:
        shards = ShardPools(Path(a.test_dir))
        held_check = verify_held_out(shards, held)
        if not held_check["verified"]:
            sys.exit(f"[paper_numbers] held-out list does not match the test data: "
                     f"{held_check['problems']}")

    runs = {}
    for spec in a.run:
        name, _, folder = spec.partition("=")
        if not folder:
            p.error(f"--run needs NAME=FOLDER, got {spec!r}")
        rows, settings, files = load_run(Path(folder))
        runs[name] = {"rows": rows, "settings": settings, "folder": folder, "files": files}
    expect = list(range(a.rows))
    # A failed or missing row's target, from whichever run has it, so it still
    # falls in its chroma bucket.
    targets = {t: {} for t in TIERS}
    for run in runs.values():
        for t in TIERS:
            for i, r in run["rows"][t].items():
                if r.get("target_lab"):
                    targets[t].setdefault(i, r["target_lab"])

    scored = {n: [t for t in TIERS if run["rows"][t]] for n, run in runs.items()}
    tab = {n: {t: table(run["rows"][t], expect, targets[t]) for t in scored[n]}
           for n, run in runs.items()}
    rng = np.random.default_rng(a.seed)
    # One set of bootstrap indices per tier, shared by every run: the rows are
    # the same, so runs are resampled identically and comparisons are paired.
    idx = {t: rng.integers(0, a.rows, size=(a.draws, a.rows)) for t in TIERS}

    out = {"thresholds": list(THRESHOLDS), "chroma_buckets": [b[0] for b in BUCKETS],
           "rows_per_tier": a.rows, "bootstrap": {"draws": a.draws, "seed": a.seed,
                                                  "interval": "95% percentile"},
           "failed_rows": "counted as failures (DeltaE = inf), never dropped",
           "held_out_reals": {"list": held, "source": HELD_OUT_SOURCE, "check": held_check},
           "runs": {}, "comparisons": {}}
    for name, run in runs.items():
        res = {"settings": run["settings"], "folder": run["folder"],
               "rows_files": run["files"], "tiers": {}}
        for tier_name, tiers in TIER_SETS:
            if not all(t in scored[name] for t in tiers):
                res["tiers"][tier_name] = {"not_run": True}
                continue
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
        if "b" in scored[name]:
            sp = tier_b_split(run["rows"]["b"], expect, set(held), shards)
            de_b = {i: de_of(run["rows"]["b"].get(i)) for i in expect}
            res["tier_b_held_out"] = {
                "groups": {
                    "a_pool_has_held_out": stats_independent(
                        np.array([de_b[i] for i in sp["groups"]["a"]]), rng, a.draws),
                    "b_stack_uses_held_out": stats_independent(
                        np.array([de_b[i] for i in sp["groups"]["b"]]), rng, a.draws),
                    "c_pool_has_none": stats_independent(
                        np.array([de_b[i] for i in sp["groups"]["c"]]), rng, a.draws)},
                "n_failed_in_a": sp["failed_in_a"],
                "untraceable_pool": len(sp["untraceable_pool"]),
                "untraceable_stack": len(sp["untraceable_stack"]),
                "untraceable_rows": {"pool": sp["untraceable_pool"][:50],
                                     "stack": sp["untraceable_stack"][:50]},
                "pool_from": sp["pool_from"],
                "note": "(b) conditions on the chosen stack and on success; read it "
                        "beside (a) and its failed rows"}
        out["runs"][name] = res

    for spec in a.compare:
        A, _, B = spec.partition(":")
        if A not in runs or B not in runs:
            p.error(f"--compare {spec}: both names must be given with --run")
        comp = {"folders": {A: runs[A]["folder"], B: runs[B]["folder"]}}
        for tier_name, tiers in TIER_SETS:
            if not all(t in scored[A] and t in scored[B] for t in tiers):
                continue
            da = np.concatenate([tab[A][t][0] for t in tiers])
            db = np.concatenate([tab[B][t][0] for t in tiers])
            bk = np.concatenate([tab[A][t][1] for t in tiers])
            ii = np.concatenate([idx[t] + k * a.rows for k, t in enumerate(tiers)], axis=1)
            ent = paired(da, db, ii)
            ent["rows_better"] = float(np.mean(da < db))
            ent["rows_worse"] = float(np.mean(da > db))
            ent["by_chroma"] = {}
            for bname, _, _ in BUCKETS:
                m = bk == bname
                if not m.any():
                    continue
                # Paired within the bucket, stratified by tier.
                jj = np.concatenate(
                    [rng.choice(np.flatnonzero(m[k * a.rows:(k + 1) * a.rows]) + k * a.rows,
                                size=(a.draws, int(m[k * a.rows:(k + 1) * a.rows].sum())))
                     for k in range(len(tiers)) if m[k * a.rows:(k + 1) * a.rows].any()], axis=1)
                ent["by_chroma"][bname] = {"n": int(m.sum()), **paired(da[m], db[m], None,
                                                                         boot=(da[jj], db[jj]))}
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
            if ent.get("n_missing"):
                print(f"[WARN] {name} tier {tier_name}: {ent['n_missing']} row(s) missing, "
                      "counted as failures")
        hb = res.get("tier_b_held_out")
        if hb and (hb["untraceable_pool"] or hb["untraceable_stack"]):
            print(f"[WARN] {name} tier B: {hb['untraceable_pool']} row(s) with no traceable "
                  f"pool, {hb['untraceable_stack']} with an untraceable stack; left out of "
                  "the held-out split")


def paired(da, db, ii, boot=None) -> dict:
    """A minus B pass rates with paired bootstrap intervals."""
    ba, bb = boot if boot is not None else (da[ii], db[ii])
    ent = {}
    for t in THRESHOLDS:
        bs = np.mean(ba < t, axis=1) - np.mean(bb < t, axis=1)
        ent[f"pass_lt_{t:g}"] = {"est": float(np.mean(da < t) - np.mean(db < t)),
                                 "lo": float(np.percentile(bs, 2.5)),
                                 "hi": float(np.percentile(bs, 97.5))}
    return ent


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
    f = lambda x: "fail" if x is None or not math.isfinite(x) else f"{x:.2f}"
    return f"{f(c['est'])} [{f(c['lo'])}, {f(c['hi'])}]"


def pp(c: dict) -> str:
    return (f"{100 * c['est']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]")


def markdown(out: dict) -> str:
    th = out["thresholds"]
    cols = " | ".join(f"ΔE<{t:g} (%)" for t in th)
    head = (f"| run | n | {cols} | median ΔE | failed | missing |\n|" + "---|" * (len(th) + 5))
    lines = ["# Headline numbers", "",
             f"95% bootstrap intervals ({out['bootstrap']['draws']} draws, rows resampled). "
             "Failed and missing rows count as failures.", "", "## Runs", ""]
    for name, res in out["runs"].items():
        s = res["settings"]
        lines.append(f"- **{name}** (`{res['folder']}`): proposer {s.get('proposer')}, "
                     f"N = {s.get('ensemble_n')}, top-k {s.get('top_k')}, "
                     f"{s.get('refine_iters')} refinement steps, seed {s.get('seed')}, "
                     f"layers {s.get('layer_count') or 'free'}, checkpoint {s.get('model_sha256')}"
                     + (f"; rows rerun after the sampler crash fixed Nov 18: "
                        + ", ".join(f"{t.upper()}{i}" for t, i in s["rerun_after_sampler_bug"])
                        if s.get("rerun_after_sampler_bug") else ""))
    lines.append("")
    for tier_name, _ in TIER_SETS:
        body = []
        for name, res in out["runs"].items():
            ent = res["tiers"][tier_name]
            if ent.get("not_run"):
                continue
            c = ent["all"]
            body.append(f"| {name} | {c['n']} | " + " | ".join(pct(c[f'pass_lt_{t:g}']) for t in th)
                        + f" | {num(c['median'])} | {ent['n_failed']} | {ent['n_missing']} |")
        if not body:
            continue
        lines += [f"## Tier {tier_name}", "", head] + body + [""]
        lines += [f"### Tier {tier_name} by target chroma", "",
                  f"| run | chroma | n | {cols} | median ΔE |", "|" + "---|" * (len(th) + 4)]
        for name, res in out["runs"].items():
            for b, c in res["tiers"][tier_name].get("by_chroma", {}).items():
                lines.append(f"| {name} | {b} | {c['n']} | "
                             + " | ".join(pct(c[f'pass_lt_{t:g}']) for t in th)
                             + f" | {num(c['median'])} |")
        lines.append("")
    hb_runs = {n: r["tier_b_held_out"] for n, r in out["runs"].items() if "tier_b_held_out" in r}
    if hb_runs:
        ho = out["held_out_reals"]
        lines += ["## Tier B by held-out real material", "",
                  f"Held-out reals ({ho['source']}): {', '.join(ho['list'])}. "
                  + ("Checked against the test shards: tier B pools hold only these reals, "
                     "tier A pools none." if ho["check"].get("verified")
                     else f"Not checked against the data: {'; '.join(ho['check']['problems'])}."),
                  "", "(a) pool has a held-out real; (b) chosen stack uses one, the strictest "
                  "test, conditioned on the search's choice and on success; (c) pool has none.",
                  "", f"| run | group | n | {cols} | median ΔE |", "|" + "---|" * (len(th) + 4)]
        labels = {"a_pool_has_held_out": "(a) pool has", "b_stack_uses_held_out": "(b) stack uses",
                  "c_pool_has_none": "(c) none in pool"}
        for name, hb in hb_runs.items():
            for g, lab in labels.items():
                c = hb["groups"][g]
                if c is None:
                    lines.append(f"| {name} | {lab} | 0 | " + " | ".join("" for _ in th) + " | |")
                    continue
                lines.append(f"| {name} | {lab} | {c['n']} | "
                             + " | ".join(pct(c[f'pass_lt_{t:g}']) for t in th)
                             + f" | {num(c['median'])} |")
        lines.append("")
        for name, hb in hb_runs.items():
            lines.append(f"- {name}: {hb['n_failed_in_a']} failed row(s) in (a); untraceable: "
                         f"{hb['untraceable_pool']} pool, {hb['untraceable_stack']} stack; pools "
                         f"from rows {hb['pool_from']['row']}, from shards {hb['pool_from']['shards']}")
        lines.append("")
    if out["comparisons"]:
        lines += ["## Paired comparisons (same rows, percentage points)", "",
                  "| comparison | tier | " + " | ".join(f"Δ pass<{t:g}" for t in th)
                  + " | rows better | rows worse |", "|" + "---|" * (len(th) + 4)]
        for key, comp in out["comparisons"].items():
            for tier_name, ent in comp.items():
                if tier_name == "folders":
                    continue
                lines.append(f"| {key} | {tier_name} | "
                             + " | ".join(pp(ent[f'pass_lt_{t:g}']) for t in th)
                             + f" | {100 * ent['rows_better']:.0f}% | {100 * ent['rows_worse']:.0f}% |")
        lines += ["", "### Gap at ΔE < 2 by target chroma (percentage points)", "",
                  "| comparison | tier | " + " | ".join(b[0] for b in BUCKETS) + " |",
                  "|" + "---|" * (len(BUCKETS) + 2)]
        for key, comp in out["comparisons"].items():
            for tier_name, ent in comp.items():
                if tier_name == "folders":
                    continue
                cells = []
                for b, _, _ in BUCKETS:
                    c = ent["by_chroma"].get(b)
                    cells.append(f"{pp(c['pass_lt_2'])} (n={c['n']})" if c else "")
                lines.append(f"| {key} | {tier_name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
