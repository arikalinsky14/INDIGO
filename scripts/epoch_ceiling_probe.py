#!/usr/bin/env python3
"""
Epoch-ceiling probe: how many passes over the corpus can we afford?
===================================================================

The compute-optimal scaling sweep has to run models SMALLER than the
Chinchilla-optimal N at each budget -- that is what an IsoFLOP curve is. At
fixed C a smaller model burns fewer FLOPs per example, so the same budget
buys it proportionally MORE data, and the low-N corners of each rung end up
demanding several passes over the corpus.

With ~10M examples on disk, the top rungs need 4-14 passes. Chinchilla's loss
model assumes fresh data, so repeats threaten the fit. The usual answer is
"~4 epochs is nearly free" (Muennighoff et al. 2023, arXiv:2305.16264), but
that is a language-model result on a token-level objective and there is no
reason to assume it transfers to a 5.5-token thin-film task with a factorised
pointer head. Worse, the production run's own DeltaE curve shows the p75 TAIL
degrading across epochs 2-3 while the median stays flat -- so the damage may
not even show up in the headline metric.

So: measure it instead of importing it.

Why the bias matters
--------------------
At fixed C the low-N points run the most epochs. If they overfit, their
val_de is inflated, the fitted IsoFLOP parabola's minimum shifts toward
larger N, and alpha in N*(C) ~ C^alpha comes out biased HIGH -- we would
wrongly conclude "spend on parameters, not data". That is precisely the class
of methodological artifact Porian et al. exists to catch.

There is an opposing effect: small models have less capacity to memorise, so
they should tolerate repeats better. The two push against each other with no
clear net sign, which is why this probe sweeps model size as well as repeat
depth. A single-N measurement would give a ceiling that is wrong for the rest
of the grid.

Design
------
Each arm holds TOTAL STEPS and TOTAL EXAMPLE-PASSES exactly fixed, and varies
only how much distinct data those passes draw from:

    corpus(e) = batch_size * total_steps / e        for e in repeat depths

so steps_per_epoch = corpus/batch_size = total_steps/e, and
total_steps = steps_per_epoch * e is identical across arms. Every arm
therefore sees the same number of gradient updates, the same number of
example-passes, the same compute C, and the same LR schedule shape (warmup is
a fraction of total steps, so it lines up too). The ONLY difference is data
diversity. `--limit-examples` truncates a seeded permutation, so the smaller
corpora are strict nested subsets of the larger ones.

`total_steps` must be divisible by every repeat depth, or the arms drift apart
on step count; the script enforces this rather than silently rounding.

What to read off it
-------------------
Per arm: final val_de median / p75 / p95, the per-chroma breakdown, and the
train-minus-val CE gap. The gap separates the two failure modes -- if val_de
degrades AND the gap widens, it is memorisation; if val_de degrades with a
flat gap, it is an optimisation or LR-schedule artifact and repeats are not
the cause.

The ceiling is the largest e at which val_de (and p75) are still within noise
of the e=1 arm, per model size. That number sets the sweep's top rung.

Usage
-----
    # 1. See the grid and confirm C is matched across arms
    python scripts/epoch_ceiling_probe.py --dry-run

    # 2. Emit submit commands (default), or dispatch them
    python scripts/epoch_ceiling_probe.py --data-dir /path/to/train
    python scripts/epoch_ceiling_probe.py --data-dir /path/to/train --dispatch

    # 3. Once the runs finish
    python scripts/epoch_ceiling_probe.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.model import ModelConfig
from src.scaling.flops import (
    n_params,
    train_flops,
    train_flops_per_example,
    slot_encoder_depth,
)

# Model sizes to probe. Spans the range the sweep actually uses, so the
# capacity-dependence of repeat tolerance is visible. n_heads follows
# d_head = 64 (n_heads = d_model / 64), which is FLOP- and param-neutral.
DEFAULT_SIZES: List[Tuple[int, int]] = [
    (128, 2),    # ~0.77M params -- the sweep's small end
    (256, 2),    # ~2.9M
    (512, 4),    # ~17.5M -- near the Chinchilla-optimal N of the top rungs
]
DEFAULT_DEPTHS: List[int] = [1, 2, 4, 8]


@dataclass
class Arm:
    d_model: int
    slot_encoder_layers: int
    epochs: int
    corpus: int
    total_steps: int
    batch_size: int
    n_params: int
    flops: float
    name: str

    @property
    def passes(self) -> int:
        return self.total_steps * self.batch_size


def build_config(d_model: int, sel: int, batch_size: int, lr: float) -> ModelConfig:
    return ModelConfig(
        head_mode="cross_attn",
        d_model=d_model,
        n_heads=max(1, d_model // 64),
        slot_encoder_layers=sel,
        decoder_layers=1,
        batch_size=batch_size,
        learning_rate=lr,
    )


def build_arms(sizes: List[Tuple[int, int]], depths: List[int],
               total_steps: int, batch_size: int, lr: float) -> List[Arm]:
    lcm = 1
    for e in depths:
        lcm = lcm * e // math.gcd(lcm, e)
    if total_steps % lcm != 0:
        raise SystemExit(
            f"--total-steps {total_steps} is not divisible by {lcm} (the LCM of "
            f"repeat depths {depths}). Arms would end up with different step "
            f"counts, which breaks the fixed-C comparison this probe depends on. "
            f"Use a multiple of {lcm}, e.g. {lcm * max(1, total_steps // lcm)}."
        )

    arms: List[Arm] = []
    for d_model, sel in sizes:
        cfg = build_config(d_model, sel, batch_size, lr)
        N = n_params(cfg)
        for e in depths:
            corpus = batch_size * total_steps // e
            arms.append(Arm(
                d_model=d_model, slot_encoder_layers=sel, epochs=e,
                corpus=corpus, total_steps=total_steps, batch_size=batch_size,
                n_params=N,
                flops=train_flops(cfg, total_steps * batch_size),
                name=f"probe_d{d_model}_se{sel}_ep{e}_corpus{corpus}",
            ))
    return arms


def print_grid(arms: List[Arm]) -> None:
    print(f"{'arm':<34} {'N':>11} {'corpus':>10} {'epochs':>7} "
          f"{'steps':>7} {'passes':>10} {'C (FLOPs)':>12}")
    print("-" * 96)
    for a in arms:
        print(f"{a.name:<34} {a.n_params:>11,} {a.corpus:>10,} {a.epochs:>7} "
              f"{a.total_steps:>7} {a.passes:>10,} {a.flops:>12.4e}")
    print("-" * 96)

    # The whole design rests on C being identical within a model size; assert it.
    print("\nfixed-C check (must be identical within each model size):")
    ok = True
    by_size: Dict[int, List[Arm]] = {}
    for a in arms:
        by_size.setdefault(a.n_params, []).append(a)
    for N, group in by_size.items():
        cs = {f"{a.flops:.6e}" for a in group}
        steps = {a.total_steps for a in group}
        passes = {a.passes for a in group}
        status = "OK" if len(cs) == 1 and len(steps) == 1 and len(passes) == 1 else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  N={N:>11,}: C={cs.pop() if len(cs)==1 else cs}  "
              f"steps={steps}  passes={passes}  [{status}]")
    total = sum(a.flops for a in arms)
    print(f"\ntotal probe compute: {total:.4e} FLOPs")
    # The production run measured 460-717 examples/s on one L40s (4-6%
    # utilisation), so 460 is the conservative floor. These arms use much
    # smaller models, which will be faster if the pipeline is FLOP-bound and
    # the same speed if it is dataloader-bound -- so 460 stays a safe floor.
    passes_per_arm = arms[0].passes
    for rate, label in ((460, "conservative"), (717, "optimistic")):
        per_arm = passes_per_arm / rate
        print(f"  at {rate:>3} ex/s ({label:<12}): "
              f"{per_arm / 60:>5.1f} min/arm, "
              f"{len(arms) * per_arm / 3600:>5.2f} GPU-hours total")
    train_s = passes_per_arm / 460
    print(f"\nper-arm wall budget (conservative 460 ex/s):")
    print(f"  training                {train_s / 60:>6.1f} min")
    print(f"  DeltaE evals (~4)       {4 * 90 / 60:>6.1f} min   (512 ex at ~150ms each + JAX warmup)")
    print(f"  CE val + ckpt + startup {3.0:>6.1f} min")
    need = train_s / 60 + 6.0 + 3.0
    print(f"  ---------------------------------")
    print(f"  estimated need          {need:>6.1f} min")
    print(f"  x1.5 safety budget      {1.5 * need:>6.1f} min  <-- set --time at or above this")
    if not ok:
        raise SystemExit("fixed-C check FAILED -- refusing to emit commands.")


def emit_commands(arms: List[Arm], data_dir: str, out_root: str, lr: float,
                  de_examples: int, val_examples: int,
                  save_every: int, use_slurm: bool,
                  num_workers: int = 4) -> List[List[str]]:
    """Build one submit command per arm.

    SLURM mode drives `slurms/training.sh`, which is configured by env vars.
    Local mode invokes `scripts/training.py` directly, which takes CLI flags --
    the two are NOT interchangeable, so each is constructed separately rather
    than sharing a prefix.
    """
    cmds = []
    for a in arms:
        save_dir = f"{out_root}/{a.name}"
        n_heads = max(1, a.d_model // 64)
        # One DeltaE eval per arm, at the end. The final save always forces
        # one, and --no-de-on-epoch-end suppresses the per-epoch ones -- so
        # the e=8 arm pays the same eval cost as the e=1 arm instead of 8x,
        # which both saves wall time and keeps the arms' costs symmetric.
        de_every = a.total_steps

        if use_slurm:
            env = {
                "DATA_DIR": data_dir,
                "SAVE_DIR": save_dir,
                "HEAD_MODE": "cross_attn",
                "D_MODEL": str(a.d_model),
                "N_HEADS": str(n_heads),
                "SLOT_ENCODER_LAYERS": str(a.slot_encoder_layers),
                "DECODER_LAYERS": "1",
                "LR": str(lr),
                "BATCH_SIZE": str(a.batch_size),
                "EPOCHS": str(a.epochs),
                "LIMIT_EXAMPLES": str(a.corpus),
                # Every arm limits to a small slice of the corpus, which is
                # exactly the case where a scattered limit re-reads
                # everything each epoch.
                "LIMIT_SHARD_ALIGNED": "1",
                "LIMIT_VAL_EXAMPLES": str(val_examples),
                "LIMIT_DE_EXAMPLES": str(de_examples),
                "DE_EVERY": str(de_every),
                "DE_ON_EPOCH_END": "0",
                "SAVE_EVERY": str(save_every),
                "NUM_WORKERS": str(num_workers),
            }
            cmds.append([f"{k}={v}" for k, v in env.items()]
                        + ["sbatch", "slurms/training.sh"])
        else:
            cmds.append([
                "python", "scripts/training.py",
                "--data-dir", data_dir,
                "--save-dir", save_dir,
                "--head-mode", "cross_attn",
                "--d-model", str(a.d_model),
                "--n-heads", str(n_heads),
                "--slot-encoder-layers", str(a.slot_encoder_layers),
                "--decoder-layers", "1",
                "--lr", str(lr),
                "--batch-size", str(a.batch_size),
                "--epochs", str(a.epochs),
                "--limit-examples", str(a.corpus),
                "--limit-shard-aligned",
                "--limit-val-examples", str(val_examples),
                "--limit-de-examples", str(de_examples),
                "--de-every", str(de_every),
                "--no-de-on-epoch-end",
                "--save-every", str(save_every),
                "--num-workers", str(num_workers),
            ])
    return cmds


def _final_row(history: Path) -> Optional[dict]:
    """Last history row that carries a scored val_de."""
    best = None
    with open(history) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("val_de_median") is not None:
                if best is None or row["step"] >= best["step"]:
                    best = row
    return best


def diagnose(out_root: Path, total_steps: int, batch_size: int) -> None:
    """Recover the REAL throughput from partial (even timed-out) runs.

    Every history.jsonl row carries `step` and `wall_time_utc`, so a run that
    died against its time limit still says exactly how fast it was going.
    That is the number needed to re-size the probe -- guessing a second time
    would just burn another allocation.
    """
    from datetime import datetime
    dirs = sorted(p for p in out_root.glob("probe_d*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no probe_* directories under {out_root}")

    print("=" * 96)
    print("THROUGHPUT DIAGNOSIS (from partial runs)")
    print("=" * 96)
    print(f"  {'arm':<34} {'reached':>8} {'of':>7} {'%':>6} {'ex/s':>8} "
          f"{'dE evals':>9} {'dE s':>8}")
    rates = []
    for d in dirs:
        hist = d / "history.jsonl"
        if not hist.exists():
            print(f"  {d.name:<34} {'no history.jsonl':>40}")
            continue
        rows = []
        for line in hist.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if len(rows) < 2:
            reached = rows[0]["step"] if rows else 0
            print(f"  {d.name:<34} {reached:>8} {total_steps:>7} "
                  f"{'--':>6} {'(need >=2 rows)':>18}")
            continue
        t0 = datetime.strptime(rows[0]["wall_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.strptime(rows[-1]["wall_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
        span = (t1 - t0).total_seconds()
        dsteps = rows[-1]["step"] - rows[0]["step"]
        # Subtract the DeltaE evals, which are measured separately, to get a
        # clean training rate.
        de_s = sum(r.get("val_de_seconds") or 0.0 for r in rows[1:])
        n_de = sum(1 for r in rows if r.get("val_de_seconds"))
        train_s = max(span - de_s, 1e-9)
        rate = dsteps * batch_size / train_s if dsteps > 0 else 0.0
        if rate > 0:
            rates.append(rate)
        reached = rows[-1]["step"]
        print(f"  {d.name:<34} {reached:>8} {total_steps:>7} "
              f"{100.0 * reached / total_steps:>5.1f}% {rate:>8.1f} "
              f"{n_de:>9} {de_s:>8.0f}")

    if not rates:
        print("\nNo arm produced two timestamped rows; cannot estimate a rate.")
        return
    import statistics
    slow, med = min(rates), statistics.median(rates)
    passes = total_steps * batch_size
    print(f"\n  measured training throughput: slowest {slow:.1f} ex/s, "
          f"median {med:.1f} ex/s")
    print(f"  (the estimate this probe was sized with was 460 ex/s -- "
          f"{460 / slow:.1f}x optimistic vs the slowest arm)")
    print(f"\n  to finish {passes:,} passes at the SLOWEST observed rate:")
    train_min = passes / slow / 60
    de_min = 3 * 90 / 60
    need = train_min + de_min + 3
    print(f"    training {train_min:>7.1f} min + DeltaE ~{de_min:.0f} min "
          f"+ overhead 3 min = {need:.0f} min")
    print(f"    x1.5 safety budget: {1.5 * need:.0f} min "
          f"({1.5 * need / 60:.2f} h)")
    if 1.5 * need > 180:
        print(f"\n  WARNING: that exceeds the 3h cap that --qos=short enforces.")
        fits = int(180 / 1.5 * 60 * slow / batch_size)
        fits -= fits % 8
        print(f"  Options: (a) drop --qos=short for a longer QoS, or")
        print(f"           (b) reduce TOTAL_STEPS to ~{fits} so one arm fits "
              f"in 3h with the same 1.5x margin.")
    print("=" * 96)


def analyze(out_root: Path) -> None:
    dirs = sorted(p for p in out_root.glob("probe_d*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no probe_* directories under {out_root}")

    rows = []
    skipped: List[str] = []
    for d in dirs:
        hist = d / "history.jsonl"
        if not hist.exists():
            skipped.append(f"{d.name} (no history.jsonl -- did the run fail?)")
            continue
        row = _final_row(hist)
        if row is None:
            skipped.append(f"{d.name} (ran, but no scored val_de -- every "
                           f"generation invalid)")
            continue
        parts = d.name.split("_")
        rows.append({
            "d_model": int(parts[1][1:]),
            "sel": int(parts[2][2:]),
            "epochs": int(parts[3][2:]),
            "corpus": int(parts[4][6:]),
            "step": row["step"],
            "val_de": row["val_de_median"],
            "p75": row.get("val_de_p75"),
            "p95": row.get("val_de_p95"),
            "train_loss": row.get("train_loss"),
            "val_loss": row.get("val_loss"),
            "by_chroma": row.get("val_de_by_chroma") or {},
        })

    if not rows:
        raise SystemExit("no analysable arms found")

    by_model: Dict[Tuple[int, int], List[dict]] = {}
    for r in rows:
        by_model.setdefault((r["d_model"], r["sel"]), []).append(r)

    print("=" * 100)
    print("EPOCH-CEILING PROBE")
    print("=" * 100)
    print("\nAll arms within a model size share total steps, example-passes and C.")
    print("Only corpus diversity differs. Deltas are vs the e=1 arm.\n")

    ceilings = {}
    for (d_model, sel), group in sorted(by_model.items()):
        group.sort(key=lambda r: r["epochs"])
        base = next((r for r in group if r["epochs"] == 1), None)
        print(f"--- d_model={d_model} slot_encoder_layers={sel} ---")
        print(f"  {'epochs':>7} {'corpus':>10} {'val_de':>9} {'d_val_de':>9} "
              f"{'p75':>8} {'d_p75':>8} {'p95':>8} {'CE gap':>8} {'d_gap':>8}")
        base_gap = (base["val_loss"] - base["train_loss"]) if base and \
            base.get("val_loss") is not None and base.get("train_loss") is not None else None
        for r in group:
            gap = (r["val_loss"] - r["train_loss"]) \
                if r.get("val_loss") is not None and r.get("train_loss") is not None else None
            d_de = f"{r['val_de'] - base['val_de']:+.3f}" if base else "-"
            d_p75 = (f"{r['p75'] - base['p75']:+.3f}"
                     if base and r.get("p75") is not None and base.get("p75") is not None else "-")
            d_gap = f"{gap - base_gap:+.4f}" if gap is not None and base_gap is not None else "-"
            print(f"  {r['epochs']:>7} {r['corpus']:>10,} {r['val_de']:>9.3f} {d_de:>9} "
                  f"{(r['p75'] if r['p75'] is not None else float('nan')):>8.3f} {d_p75:>8} "
                  f"{(r['p95'] if r['p95'] is not None else float('nan')):>8.3f} "
                  f"{(gap if gap is not None else float('nan')):>8.4f} {d_gap:>8}")

        # Ceiling: largest e whose val_de AND p75 are still within tolerance
        # of e=1. Requires BOTH an e=1 baseline and at least one deeper arm --
        # a "ceiling" derived from a lone surviving arm is vacuous, and must
        # not be reported as if it were a measurement.
        others = [r for r in group if r["epochs"] != 1]
        if base is None:
            print("  -> NO CEILING: the e=1 baseline arm is missing, so there "
                  "is nothing to compare against.\n")
        elif not others:
            print("  -> NO CEILING: only the e=1 arm produced a scorable "
                  "val_de, so no repeat depth was actually tested. This is "
                  "not evidence that the ceiling is 1 epoch.\n")
        else:
            tol = 0.02 * abs(base["val_de"]) if base["val_de"] else 0.0
            ceiling = 1
            for r in others:
                de_ok = (r["val_de"] - base["val_de"]) <= tol
                p75_ok = (base.get("p75") is None or r.get("p75") is None or
                          (r["p75"] - base["p75"]) <= 0.02 * abs(base["p75"]))
                if de_ok and p75_ok:
                    ceiling = max(ceiling, r["epochs"])
            ceilings[(d_model, sel)] = ceiling
            tested = sorted(r["epochs"] for r in others)
            print(f"  -> ceiling (val_de and p75 within 2% of e=1): "
                  f"{ceiling} epoch(s)   [depths tested: {tested}]\n")

    # Per-chroma view: the tail is where repeats were expected to bite first.
    print("--- per-chroma val_de median by repeat depth ---")
    print(f"  {'d_model':>8} {'epochs':>7} {'low':>9} {'mid':>9} {'high':>9}")
    for (d_model, sel), group in sorted(by_model.items()):
        for r in sorted(group, key=lambda r: r["epochs"]):
            vals = []
            for b in ("low", "mid", "high"):
                st = r["by_chroma"].get(b) or {}
                m = st.get("median")
                vals.append(f"{m:.3f}" if m is not None else "-")
            print(f"  {d_model:>8} {r['epochs']:>7} {vals[0]:>9} {vals[1]:>9} {vals[2]:>9}")

    print(f"\n{'=' * 100}")
    if skipped:
        print(f"{len(skipped)} of {len(dirs)} arms produced no usable val_de:")
        for name in skipped:
            print(f"   - {name}")
        print("\nArms with no scored val_de usually mean the probe is mis-sized:"
              "\ntoo few steps, so the model still emits EOS immediately and"
              "\ngenerates nothing to simulate. Raise --total-steps (or --lr)"
              "\nuntil every arm generates valid structures before trusting any"
              "\nceiling from this run.\n")
    if not ceilings:
        print("NO CEILING COULD BE MEASURED.")
        print("No model size had both an e=1 baseline and at least one deeper")
        print("arm with a scorable val_de. Do NOT infer a ceiling from this run;")
        print("re-run with more steps per arm.")
        print("=" * 100)
        return

    overall = min(ceilings.values())
    print("CEILING BY MODEL SIZE: " +
          ", ".join(f"d{d}/se{s}={c}" for (d, s), c in sorted(ceilings.items())))
    print(f"BINDING CEILING (smallest across sizes): {overall} epoch(s)")
    if len(ceilings) < len(by_model):
        print(f"\nCAUTION: only {len(ceilings)} of {len(by_model)} model sizes "
              f"yielded a ceiling; the binding value may change once the rest "
              f"are measured.")
    print("\nThe sweep's top rung must be chosen so that its lowest-N corner")
    print("stays at or under this many passes over the corpus. If the ceiling")
    print("varies with model size, use the ceiling of the SMALLEST model in")
    print("the grid -- that is the corner which consumes the most data.")
    print("=" * 100)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure how many corpus passes the model tolerates, at matched compute.")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Training parquet dir (required unless --dry-run/--analyze).")
    p.add_argument("--out-root", type=str,
                   default="data/checkpoints/epoch_ceiling_probe",
                   help="Parent dir for the per-arm checkpoint dirs.")
    p.add_argument("--total-steps", type=int, default=2400,
                   help="Gradient steps per arm, identical across arms. Must be "
                        "divisible by the LCM of --depths (default 2400: 8|2400).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=6e-5,
                   help="Held fixed across arms: this probe isolates the repeat "
                        "effect, so LR must not co-vary. Defaults to the "
                        "production LR.")
    p.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS,
                   help="Repeat depths (epochs) to probe.")
    p.add_argument("--sizes", type=str, nargs="+", default=None,
                   help="Model sizes as d_model:slot_encoder_layers, "
                        "e.g. 128:2 256:2 512:4")
    p.add_argument("--limit-de-examples", type=int, default=512,
                   help="Examples per DeltaE eval. Larger than the training "
                        "default because these are the probe's only outputs.")
    p.add_argument("--limit-val-examples", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=600)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid and the fixed-C check, then stop.")
    p.add_argument("--dispatch", action="store_true",
                   help="Actually submit the arms via sbatch.")
    p.add_argument("--local", action="store_true",
                   help="Emit plain `python scripts/training.py` commands "
                        "instead of sbatch.")
    p.add_argument("--analyze", action="store_true",
                   help="Read the finished arms and report the ceiling.")
    p.add_argument("--diagnose", action="store_true",
                   help="Recover the real throughput from partial or "
                        "timed-out runs, and re-size the probe from it.")
    # -- SLURM job-array support. The grid lives here, in one place; the
    # SLURM wrapper just asks for arm $SLURM_ARRAY_TASK_ID and runs it.
    p.add_argument("--n-arms", action="store_true",
                   help="Print the arm count and exit (for --array=0-N).")
    p.add_argument("--emit-arm", type=int, default=None,
                   help="Print the training.py CLI flags for this arm index "
                        "and exit. Used by slurms/epoch_ceiling_probe.sh.")
    args = p.parse_args()

    if args.diagnose:
        diagnose(Path(args.out_root), args.total_steps, args.batch_size)
        return

    if args.analyze:
        analyze(Path(args.out_root))
        return

    sizes = DEFAULT_SIZES
    if args.sizes:
        sizes = []
        for spec in args.sizes:
            d, sel = spec.split(":")
            sizes.append((int(d), int(sel)))

    arms = build_arms(sizes, args.depths, args.total_steps, args.batch_size, args.lr)

    if args.n_arms:
        print(len(arms))
        return

    if args.emit_arm is not None:
        if not 0 <= args.emit_arm < len(arms):
            raise SystemExit(f"--emit-arm {args.emit_arm} out of range "
                             f"(0..{len(arms) - 1})")
        if not args.data_dir:
            raise SystemExit("--emit-arm requires --data-dir")
        cmds = emit_commands(arms, args.data_dir, args.out_root, args.lr,
                             args.limit_de_examples, args.limit_val_examples,
                             args.save_every, use_slurm=False,
                             num_workers=args.num_workers)
        # Strip the leading "python scripts/training.py" -- the wrapper adds it.
        print(" ".join(cmds[args.emit_arm][2:]))
        return

    print_grid(arms)

    if args.dry_run:
        return
    if not args.data_dir:
        raise SystemExit("--data-dir is required unless --dry-run or --analyze")

    cmds = emit_commands(arms, args.data_dir, args.out_root, args.lr,
                         args.limit_de_examples, args.limit_val_examples,
                         args.save_every, use_slurm=not args.local,
                         num_workers=args.num_workers)
    print(f"\n{len(cmds)} arms:\n")
    for c in cmds:
        print("  " + " ".join(c))

    if args.dispatch:
        import os
        print(f"\nDispatching {len(cmds)} arms...")
        for c in cmds:
            # SLURM commands carry a KEY=VALUE prefix; local ones do not.
            n_env = 0
            while n_env < len(c) and "=" in c[n_env] and not c[n_env].startswith("-"):
                n_env += 1
            env = os.environ.copy()
            for pair in c[:n_env]:
                k, v = pair.split("=", 1)
                env[k] = v
            r = subprocess.run(c[n_env:], env=env, capture_output=True, text=True)
            out = (r.stdout.strip() or r.stderr.strip() or "").splitlines()
            print(f"  [{'ok' if r.returncode == 0 else 'FAIL'}] "
                  f"{out[-1] if out else '(no output)'}")
    else:
        print("\n(Add --dispatch to submit, or pipe these to sh.)")


if __name__ == "__main__":
    main()
