#!/usr/bin/env python3
"""Rebuild every result and figure of the lite summary in one job.

    sbatch slurms/analyze.sh scripts/compile_results.py

Run after any new tuning results land (e.g. the STAGE=2x lower-rate runs).
In order, each step non-fatal after the first:

1. stage2_preview.py --final --rungs 6 --refit-law: the IsoFLOP collection
   (extension trials merged, each point re-selected), its fits and figures,
   and the strict learning-rate law.
2. fit_lr_law.py --diverged-brackets: the law with divergence closing
   top-edge optima, into lr_law_fit_cliff.json.
3. The CE-selected collection (collect_isoflop.py --select-by val_loss).
4. collect_confirm.py: the seed-45 retrains, two-seed mean, pipeline.
5. Figures: learning-rate surface (2D and 3D, cliff law), loss against
   learning rate at the four top-curve points.
6. A check: every point whose stage-2 winning rate no longer matches the
   rate its seed-45 retrain used. Those retrains, and their pipeline score,
   are stale until rerun at the new rate.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = "analyses/scaling/results"
TOP = ["d160/se4@2.5e16", "d256/se4@2.5e16", "d352/se6@2.5e16", "d416/se7@2.5e16"]


def run(*args: str, required: bool = False) -> bool:
    print(f"\n$ {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=REPO)
    if r.returncode:
        if required:
            raise SystemExit(f"failed: {args[0]}")
        print(f"[WARN] {args[0]} failed (exit {r.returncode}); continuing")
    return r.returncode == 0


def stale_retrains() -> None:
    tuned = REPO / RES / "tuned/isoflop_tuned.json"
    conf = REPO / RES / "confirm/isoflop_confirm.json"
    if not (tuned.is_file() and conf.is_file()):
        print("[WARN] stale check skipped: collections missing")
        return
    key = lambda r: (re.search(r"_d(\d+)_se(\d+)_", r["name"]).groups(), r["passes"])
    won = {key(r): r["lr"] for r in json.load(open(tuned))["runs"] if r["seed"] == 42}
    moved = []
    for r in json.load(open(conf))["runs"]:
        k = key(r)
        if k in won and abs(won[k] / r["lr"] - 1) > 0.01:
            moved.append((k, r["lr"], won[k]))
    print(f"\n== retrains at a rate that is no longer the winner: {len(moved)}")
    for ((d, se), D), old, new in moved:
        print(f"   d{d}/se{se}  D={D:,}  retrained at {old:.3e}, now wins at {new:.3e}")
    if moved:
        print("   Rerun these with STAGE=confirm (and score them with eval_pipeline.sh)"
              " before using the two-seed mean or pipeline figures for them.")


def main() -> None:
    run("scripts/stage2_preview.py", "--final", "--rungs", "6", "--refit-law",
        required=True)
    run("scripts/fit_lr_law.py", "--results-dir", "outputs/lr_search/cross_attn",
        "--coverage-from", f"{RES}/isoflop_fit.json", "--diverged-brackets",
        "--output", f"{RES}/lr_law_fit_cliff.json")
    run("scripts/collect_isoflop.py", "--beta2", "0.999", "--rungs", "6",
        "--select-by", "val_loss", "--output", f"{RES}/tuned/isoflop_tuned_ce.json")
    run("scripts/collect_confirm.py")
    run("analyses/scaling/plot_lr_surface.py", "--law", f"{RES}/lr_law_fit_cliff.json",
        "--output", f"{RES}/tuned/lr_surface_cliff.png")
    run("analyses/scaling/plot_lr_surface_3d.py", "--law", f"{RES}/lr_law_fit_cliff.json",
        "--output", f"{RES}/tuned/lr_surface_3d.png")
    run("analyses/scaling/plot_lr_curves.py", "--results-dir",
        "outputs/lr_search/cross_attn", "--confirm-dir", f"{RES}/confirm",
        "--select", *TOP, "--output", f"{RES}/tuned/lr_curves_top.png")
    stale_retrains()
    print("\n[INFO] done; send job-outputs/analysis_<job>.zip")


if __name__ == "__main__":
    main()
