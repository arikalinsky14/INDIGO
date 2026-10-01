#!/usr/bin/env python3
"""Assemble the final IsoFLOP from tuning stages 2 and 3.

The final IsoFLOP figure is the sweep's (N, D) grid re-run at tuned
hyperparameters, and no single job produces it:

  * the lowest RUNGS curves come from STAGE 2, where every point was trained at
    seven learning rates. The trial lr_tuning.py selected (lowest DeltaE among
    the non-diverged) IS that point's run; the other six are tuning evidence.
  * the remaining curves come from STAGE 3, one run per point at the rate the
    fitted law gives for that point's (N, D).
  * repeat seeds come from STAGE 3 on every rung: at the stage-2 rate on the
    lower rungs, at the law's rate above.

This writes those runs in the schema `scripts/fit_scaling_porian.py` and
`analyses/scaling/plot_porian.py` already read (the "runs" and "budgets" of
isoflop_fit.json), so the estimator, bootstrap and plots are reused unchanged.

Comparability is enforced, not assumed. Every point must come from
lr_tuning.py with --limit-shard-aligned, the stage-1 beta2 winner, and the
epochs and seed its cell asked for. A stage-2 result from before the shard
alignment change reads a differently drawn training subset and DeltaE slice
than a stage-3 run would, so mixing them is refused unless --allow-mixed.

One known asymmetry is recorded rather than hidden: a stage-2 point is the
minimum of seven noisy DeltaE readings, a stage-3 point is one reading, so the
lower rungs sit slightly optimistic relative to the upper ones. It shifts
whole rungs, not points within a rung, so N*(C) is unaffected; a curve-level
comparison across the boundary should keep it in mind. Each run carries its
"stage" so the effect can be checked against the repeat seeds.

    python scripts/collect_isoflop.py --beta2 0.99 --rungs 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.lr_grid_cells import cells_for                      # noqa: E402
from src.materials_vocab import VOCAB_SIZE                        # noqa: E402
from src.scaling.configs import (CORPUS_EXAMPLES,                 # noqa: E402
                                 DEFAULT_BATCH_SIZE, n_heads_for)
from src.scaling.flops import ArchSpec, train_flops_per_example  # noqa: E402

#: Same threshold lr_tuning.py and fit_scaling.py use.
DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)
CHROMA_BUCKETS = ("low", "mid", "high")


def result_path(root: Path, c: dict) -> Path:
    """The file lr_tuning.py writes for this cell (see its tag)."""
    seed = "" if c["seed"] == 42 else f"_s{c['seed']}"
    return root / (f"lr_search_ep{c['epochs']}_lim{c['limit']}"
                   f"_d{c['d_model']}_se{c['se']}_bs{DEFAULT_BATCH_SIZE}"
                   f"_b2{c['beta2']:g}{seed}.json")


def to_run(c: dict, d: dict, stage: str, allow_mixed: bool
           ) -> tuple[Optional[dict], Optional[str]]:
    """One isoflop_fit-schema run from a cell and its results file."""
    if not d.get("limit_shard_aligned") and not allow_mixed:
        return None, ("not shard-aligned: run before lr_tuning.py took "
                      "--limit-shard-aligned, so its data and DeltaE slice are "
                      "not drawn like the other points'. Re-run it, or pass "
                      "--allow-mixed")
    for key, want in (("beta2", c["beta2"]), ("epochs", c["epochs"]),
                      ("seed", c["seed"])):
        got = d.get(key, 42 if key == "seed" else None)
        if got is not None and float(got) != float(want):
            return None, f"{key} is {got}, the cell asked for {want}"

    trials = d.get("results", [])
    pick = [t for t in trials if t["lr"] == d.get("optimal_lr")] or trials[:1]
    if not pick:
        return None, "results file has no trials"
    t = pick[0]
    vl = t.get("best_val_loss")
    if vl is None or not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS:
        return None, f"diverged (val_loss {vl})"
    if t.get("final_val_de") is None:
        return None, "no DeltaE recorded"

    by_c = t.get("final_val_de_by_chroma") or {}
    arch = ArchSpec(d_model=c["d_model"], n_heads=n_heads_for(c["d_model"]),
                    head_mode="cross_attn", slot_encoder_layers=c["se"],
                    decoder_layers=1)
    n_train = int(d.get("train_examples", c["limit"]))
    passes = n_train * c["epochs"]
    return {
        "name": f"tuned_C{c['budget']:.2e}_d{c['d_model']}_se{c['se']}"
                f"_s{c['seed']}",
        "n_params": c["n_params"],
        "steps": c["epochs"] * math.ceil(n_train / DEFAULT_BATCH_SIZE),
        "passes": passes,
        "flops": train_flops_per_example(arch) * passes,
        "val_de": {"pooled": t["final_val_de"],
                   **{b: (by_c.get(b) or {}).get("median")
                      for b in CHROMA_BUCKETS}},
        "val_loss": t.get("final_val_loss"),
        "epochs": passes / CORPUS_EXAMPLES,
        "lr": t["lr"], "beta2": c["beta2"], "seed": c["seed"],
        "stage": stage,
    }, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="analyses/scaling/results/porian_fit.json",
                   help="the sweep fit whose (N, D) grid stages 2 and 3 re-ran")
    p.add_argument("--stage2-dir", default="outputs/lr_search/cross_attn")
    p.add_argument("--stage3-dir", default="outputs/isoflop_tuned/cross_attn")
    p.add_argument("--beta2", type=float, required=True,
                   help="stage 1's winner, the BETA2_WINNER stages 2 and 3 ran")
    p.add_argument("--rungs", type=int, default=3,
                   help="the RUNGS stage 2 ran with")
    p.add_argument("--output",
                   default="analyses/scaling/results/isoflop_tuned.json")
    p.add_argument("--allow-mixed", action="store_true",
                   help="accept results not run with --limit-shard-aligned")
    a = p.parse_args()

    s2, s3 = Path(a.stage2_dir), Path(a.stage3_dir)
    expected = ([(c, s2, "2") for c in cells_for(2, a.fit, a.beta2, a.rungs)]
                + [(c, s3, "3") for c in cells_for(3, a.fit, a.beta2, a.rungs,
                                                   a.stage2_dir)])
    runs, problems = [], []
    for c, root, stage in expected:
        path = result_path(root, c)
        label = (f"stage {stage}  C={c['budget']:.2e}  d{c['d_model']}/"
                 f"se{c['se']}  s{c['seed']}")
        if not path.is_file():
            problems.append(f"{label}: missing ({path})")
            continue
        run, why = to_run(c, json.load(open(path)), stage, a.allow_mixed)
        if why:
            problems.append(f"{label}: {why}")
        else:
            runs.append(run)

    budgets = sorted(json.load(open(a.fit))["budgets"])
    want, got = defaultdict(int), defaultdict(int)
    for c, _, _ in expected:
        want[c["budget"]] += 1
    for r in runs:
        got[min(budgets, key=lambda b: abs(b - r["flops"]))] += 1
    print(f"{'budget':>10} {'points':>8}  source")
    for i, b in enumerate(budgets):
        src = ("stage 2 (+ stage 3 repeats)" if i < a.rungs
               else "stage 3, law learning rate")
        print(f"{b:>10.3g} {got[b]:>3} / {want[b]:<3} {src}")
    if problems:
        print(f"\n[WARN] {len(problems)} point(s) not included:")
        for line in problems:
            print(f"   {line}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "scripts/collect_isoflop.py", "grid_from": a.fit,
               "beta2": a.beta2, "rungs_tuned": a.rungs, "budgets": budgets,
               "n_runs": len(runs), "excluded": problems, "runs": runs},
              open(a.output, "w"), indent=1)
    print(f"\n[INFO] {len(runs)} runs written to {a.output}. Fit and plot with:")
    print(f"  python scripts/fit_scaling_porian.py --fit {a.output} \\")
    print(f"      --output analyses/scaling/results/porian_fit_tuned.json")
    print(f"  python analyses/scaling/plot_porian.py --fit {a.output} \\")
    print(f"      --output analyses/scaling/results/porian_pooled_tuned.png")


if __name__ == "__main__":
    main()
