#!/usr/bin/env python3
"""Does INDIGO's wall clock track examples, FLOPs, or both?

The wall model in `src/scaling/configs.py` has NO model-size term: it predicts
elapsed time from the example count alone, on the grounds that these models are
input-bound rather than GPU-bound. That assumption is load-bearing in two
places, and it has never been tested against the sweep's own elapsed times:

  * it is how every config in the ladder is sized against the wall cap, and
  * it is why the service-unit axis compresses the sweep to 1.4x while the FLOP
    axis spans 250x, which is the whole reason credits and FLOPs disagree.

Circumstantial evidence says it is mostly right. Forward cost per example spans
164x across the ladder, so a GPU-bound pipeline would show a 164x throughput
spread; sweep v1 measured 12x, and attributed that to shared-filesystem
contention. But 12x is not 1x either, and the difference between "wall ~ D" and
"wall ~ D + small FLOP term" decides whether a cost-optimal frontier exists
separately from the compute-optimal one.

This fits both terms:

    elapsed = startup + D / rate + c * F(N) * D

and reports how much each explains. Feed it the sweep's finished runs.

    python scripts/fit_wall_model.py --runs-root data/checkpoints/scaling_sweep_12h
    python scripts/fit_wall_model.py --observations obs.csv   # passes,n_params,seconds
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaling.flops import forward_flops_per_example                # noqa: E402
from src.scaling.configs import (EXAMPLES_PER_SEC, STARTUP_SEC,        # noqa: E402
                                 estimate_wall_sec, n_heads_for)


class _Arch:
    """The handful of fields src/scaling/flops.py reads off a model config.

    Standing in for ModelConfig keeps this script off src/model.py, which
    imports torch. The point of a wall-clock audit is that it runs anywhere,
    including a login node with no environment loaded.
    """
    feature_mode = "raw_spectrum"
    encoder_hidden = 128
    encoder_out = 64
    head_mode = "cross_attn"
    n_layers = 8
    decoder_layers = 1
    batch_size = None
    vocab_size = None

    def __init__(self, d_model: int, slot_encoder_layers: int):
        self.d_model = int(d_model)
        self.slot_encoder_layers = int(slot_encoder_layers)
        self.n_heads = n_heads_for(self.d_model)


def forward_cost(d_model: int, slot_encoder_layers: int) -> float:
    return forward_flops_per_example(_Arch(d_model, slot_encoder_layers))


def _parse_utc(stamp: str) -> float:
    import calendar
    import time as _t
    return calendar.timegm(_t.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def load_from_runs(root: Path, batch_size: int) -> List[Tuple[float, float, float]]:
    """(examples, forward_flops_per_example, seconds) from each run's history.

    training.py writes history.jsonl, one entry per checkpoint, carrying step
    and wall_time_utc. Consecutive entries give an interval of known length in
    examples and in seconds, so one run yields several observations rather than
    one, and the startup cost drops out of every interval. The architecture
    comes from the config.json a checkpoint saves alongside.
    """
    out: List[Tuple[float, float, float]] = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        hist = run / "history.jsonl"
        if not hist.is_file():
            continue
        cfgs = sorted(run.glob("*/config.json")) or sorted(run.glob("config.json"))
        if not cfgs:
            continue
        try:
            cfg = json.load(open(cfgs[0]))
            rows = [json.loads(ln) for ln in open(hist) if ln.strip()]
        except (OSError, ValueError):
            continue
        d_model = cfg.get("d_model")
        se = cfg.get("slot_encoder_layers")
        if d_model is None or se is None:
            continue
        f = forward_cost(d_model, se)
        rows = [r for r in rows if r.get("step") and r.get("wall_time_utc")]
        rows.sort(key=lambda r: r["step"])
        for a, b in zip(rows, rows[1:]):
            dt = _parse_utc(b["wall_time_utc"]) - _parse_utc(a["wall_time_utc"])
            dn = (b["step"] - a["step"]) * batch_size
            # Intervals containing a DeltaE eval are not training time.
            dt -= float(b.get("val_de_seconds") or 0.0)
            if dt > 0 and dn > 0:
                out.append((float(dn), f, float(dt)))
    return out


def load_from_csv(path: Path) -> List[Tuple[float, float, float]]:
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append((float(row["passes"]),
                        forward_cost(int(row["d_model"]),
                                     int(row["slot_encoder_layers"])),
                        float(row["seconds"])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs-root", type=Path,
                   help="directory of finished sweep runs, each with "
                        "history.jsonl and a checkpoint's config.json")
    g.add_argument("--observations", type=Path,
                   help="CSV with columns passes,d_model,slot_encoder_layers,seconds")
    p.add_argument("--batch-size", type=int, default=None,
                   help="examples per step, for converting steps to examples. "
                        "Defaults to configs.DEFAULT_BATCH_SIZE.")
    a = p.parse_args()

    if a.batch_size is None:
        from src.scaling.configs import DEFAULT_BATCH_SIZE
        a.batch_size = DEFAULT_BATCH_SIZE

    obs = (load_from_runs(a.runs_root, a.batch_size) if a.runs_root
           else load_from_csv(a.observations))
    if len(obs) < 4:
        sys.exit(f"found {len(obs)} usable intervals, need at least 4. "
                 f"If reading run directories, each run needs history.jsonl "
                 f"with two or more entries carrying step and wall_time_utc, "
                 f"plus a config.json with d_model and slot_encoder_layers.")

    D = np.array([o[0] for o in obs])
    F = np.array([o[1] for o in obs])
    T = np.array([o[2] for o in obs])

    print(f"{len(obs)} intervals. examples per interval span {D.max() / D.min():.0f}x, "
          f"forward cost per example spans {F.max() / F.min():.0f}x.\n")

    # Current model: no size term, constants frozen.
    pred0 = np.array([estimate_wall_sec(int(d)) for d in D])
    rms0 = float(np.sqrt(np.mean((T / pred0 - 1) ** 2)))
    print(f"current model (no size term, as shipped):")
    print(f"   rate {EXAMPLES_PER_SEC:.0f} ex/s, startup {STARTUP_SEC:.0f} s")
    print(f"   RMS relative error {rms0:.3f},  worst "
          f"{np.max(np.abs(T / pred0 - 1)):.3f}")

    # Fit seconds PER EXAMPLE against forward cost per example:
    #
    #     dt / dn = 1 / rate + c * F(N)
    #
    # rather than total time against (D, F*D). Checkpoints land every
    # save_every STEPS, so every interval holds the same number of examples
    # and a regression on total time has almost no leverage on D: the
    # intercept absorbs it and the rate comes out meaningless. Per-example
    # time has no such problem, and it is the quantity the wall model is
    # really asserting is constant.
    spe = T / D                                     # seconds per example
    A = np.vstack([np.ones_like(F), F]).T
    scale = np.linalg.norm(A, axis=0)
    coef, *_ = np.linalg.lstsq(A / scale, spe, rcond=None)
    inv_rate, c_flop = coef / scale
    resid = spe - (inv_rate + c_flop * F)
    ss = float(np.sum((spe - spe.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan")

    print(f"\nmeasured throughput by model size:")
    print(f"   {'F per example':>15} {'runs':>5} {'ex/s':>9}")
    for f in sorted(set(F.tolist())):
        sel = F == f
        print(f"   {f:>15.3e} {int(sel.sum()):>5} {1 / np.mean(spe[sel]):>9.0f}")

    print(f"\nseconds per example = 1/rate + c * F")
    print(f"   rate      {1 / inv_rate if inv_rate > 0 else float('nan'):>12.0f} ex/s"
          f"   (shipped model assumes {EXAMPLES_PER_SEC:.0f})")
    print(f"   c         {c_flop:>12.3e} s per FLOP")
    print(f"   R2        {r2:>12.4f}")

    share = (c_flop * F) / np.maximum(spe, 1e-12)
    print(f"\nshare of time per example attributable to model size: "
          f"{share.min():.1%} to {share.max():.1%}")
    thr = 1 / spe
    print(f"throughput spread across the ladder: {thr.max() / thr.min():.1f}x "
          f"(forward cost spans {F.max() / F.min():.0f}x)")

    if c_flop <= 0 or share.max() < 0.10:
        print("\n=> The size term earns nothing. Wall clock tracks examples, so "
              "service units do too, and the cost-optimal choice is simply "
              "'make N as large as the wall cap allows at the D you want'. The "
              "credit axis really is a restatement of D.")
    else:
        print("\n=> The size term is real. The credit axis is NOT a pure "
              "restatement of D, and the cost-optimal allocation has to be "
              "solved rather than read off. Put the FLOP term into "
              "estimate_wall_sec before sizing another sweep.")


if __name__ == "__main__":
    main()
