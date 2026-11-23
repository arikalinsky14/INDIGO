#!/usr/bin/env python3
"""Add the checkpoint hash to greedy evaluate.py outputs written before
evaluate.py recorded it (Nov 18), using only evidence from the runs themselves.

For each eval JSON (default outputs/prod_eval/eval_{val,tier_a,tier_b}_greedy.json)
it finds the job log that wrote it (job-outputs/indigo-eval.*.out:
"[INFO] Results saved to <file>") and reads, from that same log, the
checkpoint it loaded ("[INFO] Loading model from <dir>") and the config tag
("[INFO] Config tag: ..."). It then hashes that checkpoint's model.pt as it
is on disk now. The file is stamped only if
  - exactly one log wrote it (a later rerun to the same path is ambiguous),
  - the log's config tag equals the file's metrics.config_tag, and
  - the checkpoint's model.pt still exists, so its hash can be computed;
and it records which log and directory the hash came from. Nothing else in
the file changes, and no number is recomputed. A file that fails any test is
reported and left alone: rerun it with slurms/evaluate.sh (evaluate.py now
writes model_sha256 itself).

    python scripts/stamp_greedy_evals.py            # on CRC, from the repo
    python scripts/stamp_greedy_evals.py --dry-run

Caveat it cannot remove: the hash is of model.pt as it is today. The
production weights are never overwritten (promote_production.py refuses), so
for them it is the file the job read.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", nargs="+", default=[
        f"outputs/prod_eval/eval_{t}_greedy.json" for t in ("val", "tier_a", "tier_b")])
    ap.add_argument("--logs", default="job-outputs/indigo-eval.*.out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    logs = {}
    for lp in sorted(glob.glob(str(REPO / a.logs))):
        text = Path(lp).read_text(errors="replace")
        saved = re.findall(r"\[INFO\] Results saved to (\S+)", text)
        ck = re.findall(r"\[INFO\] Loading model from (\S+)", text)
        tag = re.findall(r"\[INFO\] Config tag: (\S+)", text)
        for out in saved:
            logs.setdefault(Path(out).name, []).append((lp, ck, tag))

    bad = 0
    for ep in a.evals:
        f = REPO / ep
        if not f.is_file():
            print(f"[skip] {ep}: not here")
            continue
        d = json.load(open(f))
        m = d["metrics"]
        if m.get("model_sha256"):
            print(f"[ok]   {ep}: already carries model_sha256 {m['model_sha256']}")
            continue
        hits = logs.get(f.name, [])
        why = None
        if len(hits) != 1:
            why = f"{len(hits)} logs wrote a file named {f.name}; need exactly one"
        else:
            lp, ck, tag = hits[0]
            if len(set(ck)) != 1 or len(set(tag)) != 1:
                why = f"{lp}: loaded {len(set(ck))} checkpoints / {len(set(tag))} tags"
            elif tag[0] != m.get("config_tag"):
                why = f"{lp}: config tag {tag[0]} != file's {m.get('config_tag')}"
            else:
                ckdir = Path(ck[0])
                ckdir = ckdir if ckdir.is_absolute() else REPO / ckdir
                if not (ckdir / "model.pt").is_file():
                    why = f"{lp}: {ckdir}/model.pt is gone; cannot hash it"
        if why:
            bad += 1
            print(f"[FAIL] {ep}: {why}. Left unchanged; rerun it with slurms/evaluate.sh.")
            continue
        full = sha256(ckdir / "model.pt")
        print(f"[stamp] {ep}: model_sha256 {full[:16]} from {ckdir} (log {Path(lp).name})")
        if a.dry_run:
            continue
        m["model_sha256"] = full[:16]
        m["model_sha256_full"] = full
        m["checkpoint"] = str(ckdir.resolve())
        m["model_sha256_evidence"] = (
            f"stamped Nov 2026 by scripts/stamp_greedy_evals.py: log {Path(lp).name} wrote this "
            f"file after loading {ck[0]} (config tag matches); sha256 of that model.pt")
        f.write_text(json.dumps(d, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
