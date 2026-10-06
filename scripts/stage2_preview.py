#!/usr/bin/env python3
"""Stage 2's IsoFLOP curves and learning-rate law, before stage 3 exists.

One command for slurms/analyze.sh, which runs it on smp and zips what it
writes:

    sbatch slurms/analyze.sh scripts/stage2_preview.py

It runs, in order:

1. scripts/collect_isoflop.py: stage 2's winning trial at every point of the
   tuned rungs, in isoflop_fit.json's schema. Stage 3's points are reported
   missing, which is expected here.
2. scripts/fit_scaling_porian.py and analyses/scaling/plot_porian.py on those
   points. Stage 2 trains seed 42 only, so there are no repeat seeds to
   calibrate the bootstrap's noise; it is borrowed from the first sweep's
   repeat seeds (--noise-from) and the figure says so. The first sweep's own
   figure is drawn alongside for comparison.
3. scripts/fit_lr_law.py on stage 2's cells: the law lr(N, D) that stage 3
   and STAGE=check use, with the leave-one-rung-out check. This one is not a
   preview: it writes analyses/scaling/results/lr_law_fit.json, which
   src/scaling/configs.py picks up.

Everything lands in analyses/scaling/results/stage2/ except the law.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beta2", default="0.999")
    p.add_argument("--rungs", default="5")
    p.add_argument("--stage2-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--sweep", default="analyses/scaling/results/isoflop_fit.json",
                   help="the first sweep: noise source and comparison")
    p.add_argument("--out-dir", default="analyses/scaling/results/stage2")
    a = p.parse_args()

    out = Path(a.out_dir)
    pts = out / "isoflop_stage2.json"
    run("scripts/collect_isoflop.py", "--beta2", a.beta2, "--rungs", a.rungs,
        "--stage2-dir", a.stage2_dir, "--output", str(pts))
    run("scripts/fit_scaling_porian.py", "--fit", str(pts),
        "--noise-from", a.sweep, "--output", str(out / "porian_fit_stage2.json"))
    run("analyses/scaling/plot_porian.py", "--fit", str(pts),
        "--noise-from", a.sweep, "--output", str(out / "isoflop_stage2.png"),
        "--title", f"Stage 2 IsoFLOP, rungs 1-{a.rungs} at tuned learning "
                   f"rates (seed 42; noise from the first sweep)")
    run("analyses/scaling/plot_porian.py", "--fit", a.sweep,
        "--output", str(out / "isoflop_first_sweep.png"),
        "--title", "First sweep, old learning-rate law (for comparison)")
    run("scripts/fit_lr_law.py", "--results-dir", a.stage2_dir,
        "--coverage-from", a.sweep)
    print(f"\n[INFO] figures and fits in {out}/; the law in "
          f"analyses/scaling/results/lr_law_fit.json")


if __name__ == "__main__":
    main()
