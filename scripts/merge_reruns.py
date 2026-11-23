#!/usr/bin/env python3
"""Fold sampler-crash reruns into their original chunks, so no row index of a
run appears twice; and check that none does.

Seven rows crashed on the token-3201 sampler bug (fixed Nov 18, 1113878) and
were rerun one row at a time into `<run>/rerun_*` folders
(slurms/rerun_crashed_rows.sh). Both copies then sat in the run: a reader who
loaded every rows.json got the crashed one as well (tier B row 173 counted as
a failure: 484 instead of 485 rows under DeltaE 2).

    python scripts/merge_reruns.py results/paper_runs/strong_model ...   # merge
    python scripts/merge_reruns.py --check results/paper_runs results/neutron_final_runs

Merge, per rerun row: the original chunk's row must be a sampler crash and
the rerun must have succeeded, else nothing is touched. The rerun row replaces
it in place, carrying `rerun_of_sampler_crash` (rerun folder, the error it
replaces, the rerun's model tag and checkpoint hash, which must match the
run's). The chunk's tier summary is recomputed with test_eval.py's own
aggregation (inference/src/eval_stats.py), its old counts kept under
`before_rerun_merge`. The rerun folder is moved, not deleted, to
`<run's parent>/_merged_reruns/<run>/`, outside every run. Idempotent.

--check: every immediate subfolder of each root is a run; exit 1 if any
(tier, row_idx) appears in more than one rows.json of a run. sync_results.sh
runs it before committing. No torch.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CRASH = "out of range [0, 3201)"


def tier_files(run: Path):
    """{(tier, row_idx): [rows.json path, ...]} over the whole run."""
    seen = defaultdict(list)
    for rp in sorted(glob.glob(str(run / "**" / "tier_*" / "rows.json"), recursive=True)):
        tier = Path(rp).parent.name
        for r in json.load(open(rp)):
            seen[(tier, int(r["row_idx"]))].append(rp)
    return seen


def check(roots) -> int:
    bad = 0
    for root in map(Path, roots):
        runs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith("_")] \
            if root.is_dir() else []
        for run in runs:
            dups = {k: v for k, v in tier_files(run).items() if len(v) > 1}
            for (tier, i), paths in sorted(dups.items()):
                bad += 1
                print(f"[FAIL] {run}: {tier} row {i} in {len(paths)} files: " + ", ".join(paths))
    if not bad:
        print(f"[check] no duplicated row in {', '.join(map(str, roots))}")
    return 1 if bad else 0


def merge(run: Path, dry: bool) -> int:
    from inference.src.eval_stats import aggregate_delta_e   # numpy; --check needs none
    reruns = sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("rerun_"))
    if not reruns:
        print(f"[merge] {run}: no rerun folders")
        return 0
    plan = []          # (rerun dir, tier, row, original rows.json)
    for rd in reruns:
        for rp in glob.glob(str(rd / "**" / "tier_*" / "rows.json"), recursive=True):
            tier = Path(rp).parent.name
            rsum = json.load(open(Path(rp).with_name("summary.json")))
            for new in json.load(open(rp)):
                i = int(new["row_idx"])
                if new.get("error") or new.get("delta_e") is None:
                    sys.exit(f"[merge] {rp}: rerun of row {i} did not succeed; nothing merged")
                hits = []
                for op in glob.glob(str(run / "**" / tier / "rows.json"), recursive=True):
                    if Path(op).is_relative_to(rd):
                        continue
                    hits += [(op, r) for r in json.load(open(op)) if int(r["row_idx"]) == i]
                if len(hits) != 1:
                    sys.exit(f"[merge] {run}: {tier} row {i} found {len(hits)} times outside "
                             f"the rerun; expected 1. Nothing merged.")
                op, old = hits[0]
                if CRASH not in str(old.get("error") or ""):
                    sys.exit(f"[merge] {op}: row {i} is not a sampler crash "
                             f"({old.get('error')!r}); refusing to replace it")
                osum = json.load(open(Path(op).with_name("summary.json")))
                if rsum.get("model_sha256") != osum.get("model_sha256"):
                    sys.exit(f"[merge] {rp}: checkpoint {rsum.get('model_sha256')} differs from "
                             f"the run's {osum.get('model_sha256')}")
                plan.append((rd, tier, i, op, old, new, rsum))
    for rd, tier, i, op, old, new, rsum in plan:
        print(f"[merge] {run.name}: {tier} row {i}: {Path(op).relative_to(run)} <- "
              f"{rd.name} (dE {new['delta_e']:.4f})")
    if dry:
        return 0
    for rd, tier, i, op, old, new, rsum in plan:
        rows = json.load(open(op))
        merged = dict(new)
        merged["rerun_of_sampler_crash"] = {
            "rerun_folder": f"_merged_reruns/{run.name}/{rd.name}",
            "replaced_error": old.get("error"),
            "rerun_model_tag": rsum.get("model_tag"),
            "rerun_model_sha256": rsum.get("model_sha256")}
        rows = [merged if int(r["row_idx"]) == i else r for r in rows]
        Path(op).write_text(json.dumps(rows, indent=2))
        sp = Path(op).with_name("summary.json")
        summ = json.load(open(sp))
        before = {k: summ.get(k) for k in ("n_attempted", "n_succeeded", "n_failed",
                                            "delta_e", "buckets")}
        summ.setdefault("before_rerun_merge", before)
        summ.update(aggregate_delta_e(rows))
        summ["merged_reruns"] = sorted(set(summ.get("merged_reruns", []) + [i]))
        sp.write_text(json.dumps(summ, indent=2))
        # the chunk's run-level summary keeps a copy of the tier summary
        top = sp.parent.parent / "summary.json"
        if top.is_file():
            t = json.load(open(top))
            if isinstance(t.get("tiers"), dict) and tier in t["tiers"]:
                t["tiers"][tier] = {**t["tiers"][tier], **{k: summ[k] for k in (
                    "n_attempted", "n_succeeded", "n_failed", "delta_e", "buckets",
                    "merged_reruns")}}
                top.write_text(json.dumps(t, indent=2))
    for rd in sorted({p[0] for p in plan}):
        dest = run.parent / "_merged_reruns" / run.name / rd.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rd), str(dest))
        print(f"[merge] moved {rd} -> {dest}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--check", action="store_true", help="only check for duplicated rows")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.check:
        return check(a.paths)
    rc = max(merge(Path(x), a.dry_run) for x in a.paths)
    if a.dry_run:
        return rc
    return max(rc, check(sorted({str(Path(x).parent) for x in a.paths})))


if __name__ == "__main__":
    sys.exit(main())
