#!/usr/bin/env python3
"""The confirmation runs (STAGE=confirm) as IsoFLOP points, on two metrics.

Every tuned IsoFLOP point was retrained once at the rate it won with, on a new
seed, with its weights kept. This reads those runs and writes two files in
isoflop_fit.json's schema, so fit_scaling_porian.py and plot_porian.py read
them unchanged:

  isoflop_confirm.json   CE and greedy DeltaE of the retrained points. Nothing
                         was selected on these runs, so unlike stage 2's
                         best-of-7 values they carry no winner's-curse bias.
  isoflop_pipeline.json  The same points scored by the light inference
                         pipeline (slurms/eval_pipeline.sh: best of N sampled
                         stacks, a few refinement steps), as val_de.pooled,
                         with each test tier alongside. Points without a
                         pipeline score yet are left out and listed.

The confirmation runs are one seed, so fit them with --noise-from the tuned
collection, whose repeat seeds calibrate the bootstrap.

    python scripts/collect_confirm.py \\
        --confirm-dir outputs/isoflop_confirm/cross_attn \\
        --out-dir analyses/scaling/results/confirm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_isoflop import to_run                         # noqa: E402
from lr_edges import CELL_RE, arch_for                     # noqa: E402
from src.scaling.flops import n_params, train_flops_per_example  # noqa: E402


def pipeline_scores(pipe_dir: Path) -> dict:
    """{'pooled': median over every row of both tiers, 'tier_a', 'tier_b'}
    from test_eval.py's rows.json files under one checkpoint's output, or {}."""
    out, pooled = {}, []
    for tier in ("a", "b"):
        rows_files = sorted(pipe_dir.glob(f"*/tier_{tier}/rows.json"))
        if not rows_files:
            continue
        rows = json.load(open(rows_files[-1]))
        de = [float(r["delta_e"]) for r in rows if "delta_e" in r]
        if de:
            out[f"tier_{tier}"] = float(np.median(de))
            pooled += de
    if pooled:
        out["pooled"] = float(np.median(pooled))
        out["n_rows"] = len(pooled)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confirm-dir", default="outputs/isoflop_confirm/cross_attn")
    p.add_argument("--pipeline-dir", default=None,
                   help="default: <confirm-dir>/pipeline")
    p.add_argument("--fit", default="analyses/scaling/results/porian_fit.json",
                   help="for the IsoFLOP budgets")
    p.add_argument("--out-dir", default="analyses/scaling/results/confirm")
    a = p.parse_args()

    cdir = Path(a.confirm_dir)
    pdir = Path(a.pipeline_dir) if a.pipeline_dir else cdir / "pipeline"
    budgets = sorted(json.load(open(a.fit))["budgets"])
    runs, pipe_runs, problems = [], [], []
    for path in sorted(cdir.glob("lr_search_*.json")):
        m = CELL_RE.match(path.name)
        if not m:
            continue
        ep, lim, d, se, _bs, b2, s = m.groups()
        arch = arch_for(int(d), int(se))
        flops = train_flops_per_example(arch) * int(lim) * int(ep)
        budget = min(budgets, key=lambda b: abs(np.log(flops / b)))
        c = {"d_model": int(d), "se": int(se), "limit": int(lim),
             "epochs": int(ep), "beta2": float(b2), "seed": int(s or 42),
             "n_params": int(n_params(arch)), "budget": budget}
        res = json.load(open(path))
        run, why = to_run(c, res, "confirm", allow_mixed=False)
        if why:
            problems.append(f"{path.name}: {why}")
            continue
        run["name"] = run["name"].replace("tuned_", "confirm_")
        runs.append(run)
        tag = path.name[len("lr_search_"):-len(".json")]
        scores = pipeline_scores(pdir / tag / f"lr{run['lr']:.6e}")
        if "pooled" in scores:
            pipe_runs.append(dict(run, name=run["name"].replace("confirm_", "pipeline_"),
                                  val_de={"pooled": scores["pooled"]},
                                  pipeline=scores))
        else:
            problems.append(f"{path.name}: no pipeline score yet")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": "scripts/collect_confirm.py", "budgets": budgets,
            "confirm_dir": str(cdir)}
    json.dump(dict(meta, n_runs=len(runs), runs=runs, excluded=problems),
              open(out / "isoflop_confirm.json", "w"), indent=1)
    json.dump(dict(meta, n_runs=len(pipe_runs), runs=pipe_runs,
                   metric="light inference pipeline DeltaE, both test tiers"),
              open(out / "isoflop_pipeline.json", "w"), indent=1)
    per = {b: sum(1 for r in runs if r["flops"] and
                  min(budgets, key=lambda x: abs(np.log(r["flops"] / x))) == b)
           for b in budgets}
    print(f"{len(runs)} confirmation runs, {len(pipe_runs)} with a pipeline score")
    for b in budgets:
        print(f"  C = {b:.3g}: {per[b]} points")
    for line in problems:
        print(f"  [WARN] {line}")
    print(f"wrote {out}/isoflop_confirm.json and isoflop_pipeline.json")


if __name__ == "__main__":
    main()
