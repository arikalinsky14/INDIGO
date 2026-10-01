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
from src.model import ModelConfig                              # noqa: E402
from src.scaling.flops import forward_flops_per_example        # noqa: E402
from src.scaling.configs import (EXAMPLES_PER_SEC, STARTUP_SEC,  # noqa: E402
                                 estimate_wall_sec, n_heads_for)


def forward_cost(d_model: int, slot_encoder_layers: int) -> float:
    return forward_flops_per_example(ModelConfig(
        feature_mode="raw_spectrum", encoder_hidden=128, encoder_out=64,
        encoder_dropout=0.1, d_model=d_model, n_layers=8, dropout=0.1,
        head_mode="cross_attn", n_heads=n_heads_for(d_model),
        slot_encoder_layers=slot_encoder_layers, decoder_layers=1))


def load_from_runs(root: Path) -> List[Tuple[float, float, float]]:
    """(passes, forward_flops_per_example, elapsed_sec) from run histories."""
    out = []
    for hist in sorted(root.glob("*/history.json")):
        try:
            h = json.load(open(hist))
        except (OSError, ValueError):
            continue
        elapsed = h.get("elapsed_sec") or h.get("wall_sec")
        passes = h.get("passes") or h.get("examples_seen")
        d_model, se = h.get("d_model"), h.get("slot_encoder_layers")
        if not (elapsed and passes and d_model and se):
            continue
        out.append((float(passes), forward_cost(int(d_model), int(se)),
                    float(elapsed)))
    return out


def load_from_csv(path: Path) -> List[Tuple[float, float, float]]:
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            d_model, se = int(row["d_model"]), int(row["slot_encoder_layers"])
            out.append((float(row["passes"]), forward_cost(d_model, se),
                        float(row["seconds"])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs-root", type=Path,
                   help="directory of finished sweep runs with history.json")
    g.add_argument("--observations", type=Path,
                   help="CSV with columns passes,d_model,slot_encoder_layers,seconds")
    a = p.parse_args()

    obs = (load_from_runs(a.runs_root) if a.runs_root
           else load_from_csv(a.observations))
    if len(obs) < 4:
        sys.exit(f"found {len(obs)} usable observations, need at least 4. "
                 f"If reading run directories, check that history.json carries "
                 f"elapsed_sec, passes, d_model and slot_encoder_layers.")

    D = np.array([o[0] for o in obs])
    F = np.array([o[1] for o in obs])
    T = np.array([o[2] for o in obs])

    print(f"{len(obs)} runs. D spans {D.max() / D.min():.0f}x, "
          f"forward cost per example spans {F.max() / F.min():.0f}x.\n")

    # Current model: no size term, constants frozen.
    pred0 = np.array([estimate_wall_sec(int(d)) for d in D])
    rms0 = float(np.sqrt(np.mean((T / pred0 - 1) ** 2)))
    print(f"current model (no size term, as shipped):")
    print(f"   rate {EXAMPLES_PER_SEC:.0f} ex/s, startup {STARTUP_SEC:.0f} s")
    print(f"   RMS relative error {rms0:.3f},  worst "
          f"{np.max(np.abs(T / pred0 - 1)):.3f}")

    def lstsq_scaled(cols: List[np.ndarray]) -> np.ndarray:
        """Least squares with the columns normalised first.

        The design matrix spans 1 (intercept) to F*D ~ 1e16, a condition
        number at the edge of float64, and an unscaled solve silently
        truncates a singular value: it returned a zero startup and the wrong
        rate on data generated from a known model. Scaling each column to unit
        norm and undoing it afterwards fixes that.
        """
        A = np.vstack(cols).T
        scale = np.linalg.norm(A, axis=0)
        scale[scale == 0] = 1.0
        coef, *_ = np.linalg.lstsq(A / scale, T, rcond=None)
        return coef / scale

    ones = np.ones_like(D)
    c1 = lstsq_scaled([ones, D])                      # T = s + D / r
    r1 = T - (c1[0] + c1[1] * D)
    c2 = lstsq_scaled([ones, D, F * D])               # ... + c * F * D
    r2 = T - (c2[0] + c2[1] * D + c2[2] * F * D)
    ss = float(np.sum((T - T.mean()) ** 2))

    def rate(b: float) -> str:
        return f"{1 / b:7.0f}" if b > 0 else "      -"

    print(f"\nrefit, examples only:    startup {c1[0]:8.0f} s, "
          f"{rate(c1[1])} ex/s,  R2 {1 - np.sum(r1 ** 2) / ss:.4f}")
    print(f"refit, examples + FLOPs: startup {c2[0]:8.0f} s, "
          f"{rate(c2[1])} ex/s, FLOP term {c2[2]:.3e} s per FLOP,  "
          f"R2 {1 - np.sum(r2 ** 2) / ss:.4f}")

    total2 = c2[0] + c2[1] * D + c2[2] * F * D
    share = (c2[2] * F * D) / np.maximum(total2, 1e-9)
    print(f"\nshare of predicted time attributable to the FLOP term: "
          f"{share.min():.1%} to {share.max():.1%}")
    gain = (np.sum(r1 ** 2) - np.sum(r2 ** 2)) / max(np.sum(r1 ** 2), 1e-9)
    print(f"adding the FLOP term removes {gain:.1%} of the residual variance.")
    if c2[2] <= 0 or share.max() < 0.10:
        print("\n=> The size term earns nothing. Wall clock tracks examples, so "
              "service units do too, and a cost-optimal frontier is simply "
              "'make N as large as the wall cap allows at the D you want'.")
    else:
        print("\n=> The size term is real. The credit axis is NOT a pure "
              "restatement of D, and the cost-optimal allocation has to be "
              "solved rather than read off. Update EXAMPLES_PER_SEC and add "
              "the FLOP term to estimate_wall_sec before sizing another sweep.")


if __name__ == "__main__":
    main()
