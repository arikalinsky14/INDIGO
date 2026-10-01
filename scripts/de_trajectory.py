#!/usr/bin/env python3
"""
Recover the DeltaE trajectory of a finished run from its saved checkpoints.
===========================================================================

Why this exists
---------------
The scaling sweep records exactly ONE DeltaE value per run, at the final step
(scripts/scaling_sweep.py sets --de-every past the end so only training.py's
forced "final" eval fires). That was a deliberate wall-clock trade, and it
turned out to cost the one measurement needed to interpret the fixed-N data
ladder.

The ladder showed val_de rising with more passes: 10.82 at 0.45 epochs, then
11.43, 12.50, 12.99 at 0.95, 2.01 and 4.26. Two explanations fit that equally
well from endpoint values alone:

  (a) REPETITION. Passes beyond ~1 epoch genuinely hurt, so the corpus is the
      binding constraint and reaching production-scale budgets needs roughly
      4x more data.
  (b) SCHEDULE. Every ladder point ran one cosine cycle over its own horizon
      at one base LR (the LR law is a function of N alone). A long cosine
      horizon can peak early and decay badly -- CLAUDE.md records exactly
      this as "cosine death" on the finetune line, and production's own
      DeltaE optimum sat at step 13000 of ~58,600, i.e. 0.67 epochs.

Those imply very different next steps (generate data, versus fix the
schedule), so the distinction is worth settling before spending either way.

The discriminator
-----------------
Endpoints cannot separate them; trajectories can, and the checkpoints to
build them already exist because the sweep saved every 2000 steps.

  * If val_de peaks EARLY and decays while the LR is still high (mid-cosine),
    the cause is what the model is being fed -- repetition.
  * If val_de tracks the LR decay, falling only as the schedule winds down and
    degrading in the tail, the cause is the schedule.
  * The sharpest version compares the SAME pass count across two runs: the
    ladder's 9.49M-pass run FINISHED there with a fully decayed LR, while the
    42.5M-pass run merely passes through 9.49M mid-flight at high LR. A large
    gap at equal passes is a schedule effect by construction, since both saw
    identical data.

This costs no retraining. It is a forward pass plus an optical sim per
checkpoint, which is why --every-nth exists: at 2000-step saves a 166k-step
run has 83 checkpoints, and DeltaE eval runs about 146 s per 2048 examples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import torch

from src.dataset import FlexThinFilmDataset
from src.delta_e_eval import OPTICAL_SIM_AVAILABLE, evaluate_delta_e
from src.model import ModelConfig, build_model
from src.optical_sim import OpticalSimulator


def find_checkpoints(run_dir: Path) -> List[tuple]:
    """(step, dir) for every step_* checkpoint, plus final, sorted by step."""
    out = []
    for d in run_dir.iterdir():
        if not d.is_dir() or not (d / "model.pt").exists():
            continue
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m:
            out.append((int(m.group(1)), d))
        elif d.name == "final":
            meta = d / "meta.json"
            step = json.load(open(meta))["step"] if meta.exists() else None
            if step is not None:
                out.append((int(step), d))
    # 'final' and the last step_* can be the same step; keep one.
    seen, uniq = set(), []
    for step, d in sorted(out):
        if step in seen:
            continue
        seen.add(step)
        uniq.append((step, d))
    return uniq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="a single run directory containing step_* checkpoints")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--every-nth", type=int, default=1,
                   help="evaluate every Nth checkpoint (cost is linear in this)")
    p.add_argument("--limit-de-examples", type=int, default=2048)
    p.add_argument("--limit-val-examples", type=int, default=2000)
    p.add_argument("--corpus-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="passes = step * batch_size; defaults to the config's")
    p.add_argument("--seed", type=int, default=42,
                   help="must match the run's seed so the eval slice matches")
    p.add_argument("--limit-shard-aligned", action="store_true", default=True)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if not OPTICAL_SIM_AVAILABLE:
        raise SystemExit("no optical simulator, so no DeltaE can be computed")

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir {run_dir} is not a directory")
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no step_* checkpoints with model.pt under {run_dir}")
    # Always keep the LAST checkpoint. Plain [::n] slicing drops it whenever
    # the count is not a multiple of n, and it silently did: the 39-checkpoint
    # run reported its "final" as step 62,000 of 78,494, so the trajectory
    # stopped short of the endpoint the ladder actually recorded.
    if args.every_nth > 1:
        thinned = ckpts[::args.every_nth]
        if thinned[-1] is not ckpts[-1]:
            thinned.append(ckpts[-1])
        ckpts = thinned

    corpus = args.corpus_examples
    if corpus is None:
        from src.scaling.configs import CORPUS_EXAMPLES
        corpus = CORPUS_EXAMPLES

    cfg_path = ckpts[0][1] / "config.json"
    config = ModelConfig.from_dict(json.load(open(cfg_path)))
    batch_size = args.batch_size or config.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"run:          {run_dir}")
    print(f"checkpoints:  {len(ckpts)} (every {args.every_nth}th), "
          f"steps {ckpts[0][0]:,} to {ckpts[-1][0]:,}")
    print(f"model:        d_model={config.d_model} heads={config.n_heads} "
          f"slot_layers={config.slot_encoder_layers}")
    print(f"device:       {device}")
    print(f"dE examples:  {args.limit_de_examples}\n", flush=True)

    # Read the eval slice ONCE and reuse it for every checkpoint, so the
    # trajectory is a like-for-like comparison and not a sequence of different
    # example draws. Built exactly as scripts/training.py builds its own, with
    # the same split, seed and shard-alignment, so a trajectory point is
    # directly comparable to that run's recorded val_de.
    examples = list(FlexThinFilmDataset(
        Path(args.data_dir), seed=args.seed, split="validation", verbose=False,
        limit_examples=max(args.limit_val_examples, args.limit_de_examples),
        limit_shard_aligned=args.limit_shard_aligned, streaming=args.streaming))
    if not examples:
        raise SystemExit("validation split yielded no examples")
    # One simulator for every checkpoint: jaxlayerlumos re-traces per stack
    # depth, so reusing it keeps that cache warm across the whole trajectory.
    simulator = OpticalSimulator(incidence_angle=0)

    print(f"{'step':>10} {'passes':>13} {'epochs':>7} {'val_de':>8} "
          f"{'low':>7} {'mid':>7} {'high':>7} {'n':>6}")
    print("-" * 74)
    rows = []
    best = None
    for step, d in ckpts:
        model = build_model(config)
        model.load_state_dict(
            torch.load(d / "model.pt", map_location=device, weights_only=True),
            strict=False)
        model.to(device).eval()
        res = evaluate_delta_e(model, examples, device,
                               limit=args.limit_de_examples,
                               simulator=simulator)
        if not res.get("n_scored"):
            print(f"{step:>10,} {'':>13} {'':>7} {'no valid generations':>8}")
            continue
        passes = step * batch_size
        bc = res.get("by_chroma", {})
        row = {"step": step, "passes": passes, "epochs": passes / corpus,
               "val_de_median": res["delta_e_median"],
               "val_de_p75": res.get("delta_e_p75"),
               "n_scored": res["n_scored"],
               "by_chroma": {b: (bc.get(b) or {}).get("median")
                             for b in ("low", "mid", "high")}}
        rows.append(row)
        if best is None or row["val_de_median"] < best["val_de_median"]:
            best = row
        print(f"{step:>10,} {passes:>13,} {passes/corpus:>7.2f} "
              f"{row['val_de_median']:>8.3f} "
              + " ".join(f"{(row['by_chroma'][b] or float('nan')):>7.2f}"
                         for b in ("low", "mid", "high"))
              + f" {row['n_scored']:>6}", flush=True)

    if not rows:
        raise SystemExit("no checkpoint produced a scored DeltaE")

    print("-" * 74)
    last = rows[-1]
    print(f"\nbest      : val_de {best['val_de_median']:.3f} at step "
          f"{best['step']:,} ({best['epochs']:.2f} epochs)")
    print(f"final     : val_de {last['val_de_median']:.3f} at step "
          f"{last['step']:,} ({last['epochs']:.2f} epochs)")
    print(f"regression: {last['val_de_median'] - best['val_de_median']:+.3f} dE "
          f"from the best checkpoint to the last")
    frac = best["step"] / last["step"] if last["step"] else float("nan")
    print(f"peak at   : {frac:.0%} of the way through training")
    print("\nHow to read that last number:")
    print("  Peaking EARLY (well under ~70%) while the cosine LR is still high")
    print("  points at what the model is being fed -- repetition -- because the")
    print("  schedule has barely started decaying. Peaking LATE, in the decay")
    print("  tail, points at the schedule. Compare against the same pass count")
    print("  in a SHORTER run, which reached it with a fully decayed LR.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"run": str(run_dir), "batch_size": batch_size,
                   "corpus_examples": corpus,
                   "model": {"d_model": config.d_model,
                             "n_heads": config.n_heads,
                             "slot_encoder_layers": config.slot_encoder_layers},
                   "best": best, "final": last, "trajectory": rows},
                  open(out, "w"), indent=2)
        print(f"\n[INFO] Written to {out}")


if __name__ == "__main__":
    main()
