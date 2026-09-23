#!/usr/bin/env python3
"""
Fixed-N data ladder: does a ~1M-parameter model saturate, or keep improving?
============================================================================

Companion to scripts/fit_scaling.py, for the runs that script must NOT see.

The data ladder holds one model shape and sweeps D. Every point is therefore
its own compute budget, so there is no parabola in log N to fit and
fit_scaling.py would read each run as a one-point budget and refuse. The
question here is different and simpler: plot val_de against D at fixed N and
say whether the curve has flattened.

Why it matters: sweep v1's best run reached greedy val_de 9.56 with 0.97M
parameters, within ~1 dE of the production checkpoint at 18-72x the
parameters. If that point is still descending in D, the binding constraint on
INDIGO is data rather than capacity, and the production model is
over-parameterised by a wide margin. If it has flattened, 9.56 is close to
what this capacity can do and the remaining gap really is capacity.

What "saturated" means here
---------------------------
A power law val_de = A * D^(-p) + E is the natural form but needs an
irreducible floor E that 6 points cannot determine jointly with A and p. So
this script does not fit one. It reports two things a reader can act on:

  * the local slope d(log val_de) / d(log D) between consecutive points,
    which goes to zero on saturation, and
  * the improvement over the LAST doubling of D, against the run-to-run
    noise the caller supplies.

Saturation is declared only when the final slope is flat AND the last
doubling's gain is under the noise floor. Both, because a noisy pair of
points can fake either one alone.

Noise
-----
--noise-de is the paired run-to-run spread in val_de at fixed config, which
the IsoFLOP sweep's --repeat-seed arms measure. It has no default on purpose:
calling a curve flat is a statement about noise, and inventing a figure for
it is how the epoch-ceiling probe first reported a ceiling that did not
exist. Without it this script reports slopes and refuses to call saturation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CHROMA_BUCKETS = ("low", "mid", "high")
POOLED = "pooled"

# Same screen as fit_scaling.py: dE cannot detect divergence, so a diverged
# run reports a plausible-looking dE that must be excluded on CE instead.
VOCAB_SIZE = 3201
DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)


class Point:
    def __init__(self, path: Path, steps: int, passes: int,
                 val_de: Dict[str, Optional[float]], val_loss: Optional[float],
                 epochs: Optional[float], n_params: Optional[int]):
        self.path = path
        self.steps = steps
        self.passes = passes
        self.val_de = val_de
        self.val_loss = val_loss
        self.epochs = epochs
        self.n_params = n_params


def _read_last_scored(hist: Path) -> Optional[dict]:
    """Last history row carrying a real val_de_median.

    Matches training.py's contract: a val_de_* distribution key is present
    only when it holds a real number, so presence means usable.
    """
    best = None
    with open(hist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("val_de_median") is not None:
                best = row
    return best


def load_points(root: Path, corpus: Optional[int]) -> tuple[List[Point], List[str]]:
    points: List[Point] = []
    skipped: List[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        hist = d / "history.jsonl"
        if not hist.exists():
            skipped.append(f"{d.name}: no history.jsonl")
            continue
        row = _read_last_scored(hist)
        if row is None:
            skipped.append(f"{d.name}: no scored val_de in history")
            continue
        vl = row.get("val_loss")
        if vl is not None and (not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS):
            skipped.append(f"{d.name}: DIVERGED (val_loss={vl:.4g} > "
                           f"{DIVERGENCE_VAL_LOSS:.1f}); its val_de is meaningless")
            continue

        cfg_path = d / "config.json"
        if not cfg_path.exists():
            cands = sorted(d.glob("*/config.json"))
            cfg_path = cands[0] if cands else None
        n_params = batch_size = None
        if cfg_path is not None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from src.model import ModelConfig
                from src.scaling.flops import n_params as count_params
                cfg = ModelConfig.from_dict(json.load(open(cfg_path)))
                n_params = count_params(cfg)
                batch_size = cfg.batch_size
            except Exception:
                pass

        steps = int(row.get("step", 0))
        passes = steps * (batch_size or 256)
        if passes <= 0:
            skipped.append(f"{d.name}: step=0, nothing trained")
            continue

        by_chroma = row.get("val_de_by_chroma") or {}
        val_de: Dict[str, Optional[float]] = {POOLED: row["val_de_median"]}
        for b in CHROMA_BUCKETS:
            val_de[b] = (by_chroma.get(b) or {}).get("median")

        points.append(Point(d, steps, passes, val_de, vl,
                            (passes / corpus) if corpus else None, n_params))
    points.sort(key=lambda p: p.passes)
    return points, skipped


def local_slopes(points: Sequence[Point], bucket: str) -> List[Optional[float]]:
    """d(log val_de) / d(log D) between consecutive points. 0 = flat."""
    out: List[Optional[float]] = [None]
    for a, b in zip(points, points[1:]):
        va, vb = a.val_de.get(bucket), b.val_de.get(bucket)
        if not va or not vb or va <= 0 or vb <= 0 or a.passes <= 0 or b.passes <= 0:
            out.append(None)
            continue
        dlogd = math.log(b.passes) - math.log(a.passes)
        out.append((math.log(vb) - math.log(va)) / dlogd if dlogd else None)
    return out


def last_doubling_gain(points: Sequence[Point], bucket: str) -> Optional[float]:
    """val_de improvement over the final factor-2 of D, interpolated in log D."""
    scored = [p for p in points if p.val_de.get(bucket)]
    if len(scored) < 2:
        return None
    last = scored[-1]
    target = last.passes / 2.0
    if scored[0].passes > target:
        return None                       # ladder does not span a doubling
    lo = max((p for p in scored if p.passes <= target), key=lambda p: p.passes)
    hi = min((p for p in scored if p.passes >= target), key=lambda p: p.passes)
    if lo is hi:
        ref = lo.val_de[bucket]
    else:
        w = ((math.log(target) - math.log(lo.passes))
             / (math.log(hi.passes) - math.log(lo.passes)))
        ref = lo.val_de[bucket] + w * (hi.val_de[bucket] - lo.val_de[bucket])
    return ref - last.val_de[bucket]      # positive = still improving


def report(points: Sequence[Point], skipped: Sequence[str],
           noise: Optional[float], flat_slope: float) -> dict:
    W = 92
    print("=" * W)
    print("FIXED-N DATA LADDER")
    print("=" * W)
    if not points:
        print("\nNo usable runs. Nothing to report.")
        for s in skipped:
            print(f"  - {s}")
        return {"points": [], "skipped": list(skipped)}

    ns = {p.n_params for p in points if p.n_params}
    if len(ns) > 1:
        print(f"\n!! WARNING: {len(ns)} DIFFERENT model sizes present "
              f"({', '.join(f'{n:,}' for n in sorted(ns))}).")
        print("   A data ladder must hold N fixed. This root probably contains")
        print("   runs from another sweep; the slopes below are not a data-scaling")
        print("   curve. Use a fresh OUT_ROOT per experiment.")
    elif ns:
        print(f"\nN = {ns.pop():,} parameters, held fixed across {len(points)} points")

    print(f"\n{'D (passes)':>13} {'epochs':>7} {'val_de':>8} {'slope':>8} "
          f"{'low':>7} {'mid':>7} {'high':>7}")
    print("-" * W)
    sl = local_slopes(points, POOLED)
    for p, s in zip(points, sl):
        ep = f"{p.epochs:>7.2f}" if p.epochs is not None else f"{'--':>7}"
        de = p.val_de.get(POOLED)
        print(f"{p.passes:>13,} {ep} {de:>8.3f} "
              f"{(f'{s:+.3f}' if s is not None else '--'):>8} "
              + " ".join(f"{(p.val_de.get(b) or float('nan')):>7.2f}"
                         for b in CHROMA_BUCKETS))
    print("-" * W)
    print("slope = d(log val_de)/d(log D). More negative = still improving fast.")
    print("0 = flat. Positive = getting worse with more data.")

    out = {"points": [{"path": p.path.name, "passes": p.passes,
                       "steps": p.steps, "epochs": p.epochs,
                       "n_params": p.n_params, "val_de": p.val_de,
                       "val_loss": p.val_loss} for p in points],
           "skipped": list(skipped), "buckets": {}}

    print("\n" + "=" * W)
    print("SATURATION VERDICT")
    print("=" * W)
    if noise is None:
        print("\nNo --noise-de supplied, so no verdict. Slopes are above.")
        print("Pass the paired run-to-run spread from the IsoFLOP sweep's")
        print("--repeat-seed arms. Calling a curve flat is a claim about noise,")
        print("and inventing a noise figure is how the epoch-ceiling probe first")
        print("reported a ceiling that did not exist.")
    else:
        print(f"\nnoise floor: {noise:.3f} dE (paired, caller-supplied)")
        print(f"flat-slope threshold: |slope| < {flat_slope:.3f}\n")

    for b in (POOLED,) + CHROMA_BUCKETS:
        s = [x for x in local_slopes(points, b) if x is not None]
        gain = last_doubling_gain(points, b)
        rec: dict = {"final_slope": s[-1] if s else None,
                     "last_doubling_gain": gain}
        if noise is not None and s and gain is not None:
            flat = abs(s[-1]) < flat_slope
            small = gain < noise
            if flat and small:
                v = "SATURATED"
            elif not flat and gain >= noise:
                v = "STILL IMPROVING"
            else:
                v = "AMBIGUOUS"
            rec["verdict"] = v
            print(f"  {b:>7}: final slope {s[-1]:+.3f}, last doubling "
                  f"{gain:+.3f} dE vs noise {noise:.3f}  -> {v}")
        out["buckets"][b] = rec

    if noise is not None:
        print("\nSATURATED needs BOTH a flat final slope and a last-doubling gain")
        print("under the noise floor; a noisy pair of points can fake either one")
        print("alone. AMBIGUOUS means exactly that -- do not read it as either.")

    if skipped:
        print("\n" + "-" * W)
        print(f"{len(skipped)} run(s) not used:")
        for s in skipped:
            print(f"  - {s}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True,
                   help="directory of data-ladder run dirs (NOT the IsoFLOP root)")
    p.add_argument("--noise-de", type=float, default=None,
                   help="paired run-to-run val_de spread; without it, no verdict")
    p.add_argument("--flat-slope", type=float, default=0.05,
                   help="|d log dE / d log D| below which the curve counts as flat")
    p.add_argument("--corpus-examples", type=int, default=None,
                   help="for the epochs column; defaults to configs.CORPUS_EXAMPLES")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        raise SystemExit(f"--runs-root {root} is not a directory")

    corpus = args.corpus_examples
    if corpus is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from src.scaling.configs import CORPUS_EXAMPLES
            corpus = CORPUS_EXAMPLES
        except Exception:
            corpus = None

    points, skipped = load_points(root, corpus)
    result = report(points, skipped, args.noise_de, args.flat_slope)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(out, "w"), indent=2)
        print(f"\n[INFO] Written to {out}")


if __name__ == "__main__":
    main()
