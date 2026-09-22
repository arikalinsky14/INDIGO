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
DEFAULT_SIZES: List[Tuple[int, int, Optional[float]]] = [
    # (d_model, slot_encoder_layers, lr). lr=None falls back to --lr; set it
    # per size from scripts/lr_tuning.py, because a single global LR tuned on
    # the 69.6M production model leaves these far smaller models untrained.
    (128, 2, None),    # ~0.77M params -- the sweep's small end
    (256, 2, None),    # ~2.9M
    (512, 4, None),    # ~17.5M -- near the Chinchilla-optimal N of the top rungs
]
DEFAULT_DEPTHS: List[int] = [1, 2, 4, 8]


@dataclass
class Arm:
    d_model: int
    slot_encoder_layers: int
    lr: float
    epochs: int
    seed: int
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


def build_arms(sizes: List[Tuple[int, int, Optional[float]]], depths: List[int],
               total_steps: int, batch_size: int, lr: float,
               seeds: Optional[List[int]] = None) -> List[Arm]:
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
    for spec in sizes:
        d_model, sel = spec[0], spec[1]
        # LR is per SIZE, not global: 6e-5 was tuned for the 69.6M production
        # model and is far too low for a 0.8M one. It stays FIXED across the
        # depths of a given size, which is what isolates the repeat effect.
        arm_lr = spec[2] if len(spec) > 2 and spec[2] else lr
        cfg = build_config(d_model, sel, batch_size, arm_lr)
        N = n_params(cfg)
        for e in depths:
            corpus = batch_size * total_steps // e
            for seed in (seeds or [42]):
                arms.append(Arm(
                    d_model=d_model, slot_encoder_layers=sel, lr=arm_lr,
                    epochs=e, seed=seed,
                    corpus=corpus, total_steps=total_steps,
                    batch_size=batch_size, n_params=N,
                    flops=train_flops(cfg, total_steps * batch_size),
                    name=(f"probe_d{d_model}_se{sel}_ep{e}"
                          f"_corpus{corpus}_s{seed}"),
                ))
    return arms


def print_grid(arms: List[Arm]) -> None:
    print(f"{'arm':<42} {'N':>11} {'lr':>9} {'corpus':>10} {'ep':>4} "
          f"{'seed':>5} {'passes':>10} {'C (FLOPs)':>12}")
    print("-" * 110)
    for a in arms:
        print(f"{a.name:<42} {a.n_params:>11,} {a.lr:>9.2e} {a.corpus:>10,} "
              f"{a.epochs:>4} {a.seed:>5} {a.passes:>10,} {a.flops:>12.4e}")
    print("-" * 110)

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
    # Wall-time model built from what the probe runs ACTUALLY measured, not
    # from the production run. The 460 ex/s originally used here came from
    # prod (d_model=1024, bs=512) and was ~11x pessimistic: with
    # --limit-shard-aligned the probe arms held 47-49 ms/step at bs=256,
    # i.e. ~5200 ex/s.
    #
    # Three terms, because they scale differently:
    #   training     steps x 49ms            -- the cheap part
    #   shard reads  (passes / rows_per_shard) x 1.75s
    #                Each epoch re-reads the shards its corpus lives on.
    #                n_shards(e) x e = passes/rows_per_shard, so this is the
    #                SAME for every depth -- which is what keeps the arms'
    #                costs symmetric. 1.75s/shard is measured: arm 11's
    #                epoch starts ran ~28s over steady state for 16 shards.
    #   startup      one cached scan_files + one validation-split read
    MS_PER_STEP = 49.0
    SEC_PER_SHARD_READ = 1.75
    ROWS_PER_SHARD = 5000
    STARTUP_MIN = 8.0
    DE_EVAL_MIN = 1.0

    train_min = arms[0].total_steps * MS_PER_STEP / 1000 / 60
    shard_reads = arms[0].passes / ROWS_PER_SHARD
    shard_min = shard_reads * SEC_PER_SHARD_READ / 60
    need = train_min + shard_min + STARTUP_MIN + DE_EVAL_MIN

    print(f"\nper-arm wall budget (from measured rates, not extrapolated):")
    print(f"  training ({arms[0].total_steps} steps at {MS_PER_STEP:.0f}ms) "
          f"{train_min:>8.1f} min")
    print(f"  shard reads ({shard_reads:.0f} at {SEC_PER_SHARD_READ}s)      "
          f"{shard_min:>8.1f} min   (equal across depths by construction)")
    print(f"  startup (scan + val read)        {STARTUP_MIN:>8.1f} min")
    print(f"  DeltaE eval (1 per arm)          {DE_EVAL_MIN:>8.1f} min")
    print(f"  ------------------------------------------")
    print(f"  estimated need                   {need:>8.1f} min")
    print(f"  x1.5 safety budget               {1.5 * need:>8.1f} min")
    print(f"\n  {len(arms)} arms -> {len(arms) * need / 60:.1f} GPU-hours "
          f"(wall time is per-arm; arms run in parallel as the scheduler allows)")
    if 1.5 * need > 180:
        print(f"\n  WARNING: exceeds the 3h --qos=short enforces. Lower "
              f"TOTAL_STEPS (keep it divisible by the LCM of the depths).")
    print(f"\ntotal probe compute: {sum(a.flops for a in arms):.4e} FLOPs")

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
                "LR": str(a.lr),
                "BATCH_SIZE": str(a.batch_size),
                "EPOCHS": str(a.epochs),
                "LIMIT_EXAMPLES": str(a.corpus),
                # Every arm limits to a small slice of the corpus, which is
                # exactly the case where a scattered limit re-reads
                # everything each epoch.
                "LIMIT_SHARD_ALIGNED": "1",
                # Explicit rather than inherited: the non-streaming path
                # materialises a whole epoch in memory and OOMs the
                # large-corpus arms.
                "STREAMING": "1",
                "LIMIT_VAL_EXAMPLES": str(val_examples),
                "LIMIT_DE_EXAMPLES": str(de_examples),
                "DE_EVERY": str(de_every),
                "DE_ON_EPOCH_END": "0",
                "SAVE_EVERY": str(save_every),
                "NUM_WORKERS": str(num_workers),
                "SEED": str(a.seed),
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
                "--lr", str(a.lr),
                "--batch-size", str(a.batch_size),
                "--epochs", str(a.epochs),
                "--limit-examples", str(a.corpus),
                "--limit-shard-aligned",
                # REQUIRED. training.py defaults --streaming to False, and
                # the non-streaming path materialises the whole epoch in
                # memory (~40KB/example). At the e=1 corpus of 2.46M that is
                # ~98GB against a 64G allocation, which OOM-killed all six
                # e=1 arms in run 3 -- and e=1 is the baseline every ceiling
                # is measured against. The SLURM env path inherits
                # STREAMING=1 from slurms/training.sh, but these are direct
                # training.py CLI args and inherit nothing.
                "--streaming",
                "--limit-val-examples", str(val_examples),
                "--limit-de-examples", str(de_examples),
                "--de-every", str(de_every),
                "--no-de-on-epoch-end",
                "--save-every", str(save_every),
                "--num-workers", str(num_workers),
                "--seed", str(a.seed),
            ])
    # Guard the two failure modes that cost a full submission each: a missing
    # --streaming OOMs the large-corpus arms, and a missing
    # --limit-shard-aligned re-reads the whole corpus every epoch.
    for cmd in cmds:
        joined = " ".join(cmd)
        if use_slurm:
            need = ("STREAMING=1", "LIMIT_SHARD_ALIGNED=1")
        else:
            need = ("--streaming", "--limit-shard-aligned")
        for token in need:
            if token not in joined:
                raise SystemExit(
                    f"internal error: emitted arm command is missing {token!r}. "
                    f"Refusing to submit -- this silently OOMs or thrashes I/O.\n"
                    f"  {joined}")
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


def analyze(out_root: Path, expected_corpus: Optional[Dict[int, int]] = None
             ) -> None:
    """Report the epoch ceiling.

    `expected_corpus` maps depth -> corpus for the grid being analysed. Any
    directory whose corpus does not match is EXCLUDED and named.

    This filter is not optional bookkeeping. Successive probe runs write into
    the same OUT_ROOT, and their corpora overlap -- a 2400-step run's e=1
    corpus (614,400) is a 9600-step run's e=4 corpus. Without the filter,
    `analyze` globbed every probe_* directory and averaged runs together. It
    did exactly that once: it mixed a 2400-step run whose models never
    trained (val_de 24-32) with a 9600-step run whose models did (val_de
    8-15), reported the resulting 21.6 dE gap as the seed noise floor, and
    declared a binding ceiling of 1 epoch off the back of it. Every number in
    that report was an artifact.

    Older directories predating seeds also carry no _sNN suffix, so they were
    silently assigned seed 42 and averaged into the real seed-42 cell.
    """
    dirs = sorted(p for p in out_root.glob("probe_d*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no probe_* directories under {out_root}")

    if expected_corpus:
        print("[grid] analysing only runs matching: " +
              ", ".join(f"e{e}={c:,}" for e, c in sorted(expected_corpus.items())))
    else:
        print("[grid] WARNING: no grid given, so every probe_* directory under "
              "this root is included. If more than one probe run has written "
              "here, their results will be averaged together and the output "
              "will be meaningless. Pass --total-steps/--batch-size/--depths.")
    rows = []
    skipped: List[str] = []      # ran, but produced nothing usable
    excluded: List[str] = []     # belongs to a different grid -- not a failure
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
        try:
            d_epochs = int(parts[3][2:])
            d_corpus = int(parts[4][6:])
        except (IndexError, ValueError):
            skipped.append(f"{d.name}: unparseable name")
            continue
        if expected_corpus is not None:
            want = expected_corpus.get(d_epochs)
            if want is None:
                excluded.append(f"{d.name}: depth {d_epochs} not in this grid")
                continue
            if d_corpus != want:
                excluded.append(
                    f"{d.name}: corpus {d_corpus:,} != {want:,} "
                    f"(expected for e={d_epochs})")
                continue
        rows.append({
            "d_model": int(parts[1][1:]),
            "sel": int(parts[2][2:]),
            "epochs": int(parts[3][2:]),
            "corpus": int(parts[4][6:]),
            "seed": int(parts[5][1:]) if len(parts) > 5 and parts[5].startswith("s")
                    else 42,
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

    # Noise floor, computed PAIRED.
    #
    # The seed sets model init, and init turns out to be a large SYSTEMATIC
    # offset rather than symmetric noise: in run 3, seed 43 was worse than
    # seed 42 in 9 of 9 cells, by up to 5.04 dE. Comparing raw per-cell means
    # therefore buries the depth effect under an offset that affects every
    # depth of a seed equally.
    #
    # Pairing removes it. Within one seed, depth deltas are far more
    # reproducible than the raw values: d128's ep8-ep2 delta came out -0.393
    # and -0.458 on the two seeds, against a 2.2 dE raw offset between them.
    # So the ceiling is decided on the mean PAIRED delta, and the resolution
    # limit is how much the paired deltas DISAGREE across seeds -- not how
    # much the raw values do.
    import statistics

    def paired_deltas(group: List[dict], baseline_epochs: int,
                      key: str = "val_de") -> Dict[int, List[float]]:
        """Per depth, one delta-vs-baseline per seed."""
        by_seed: Dict[int, Dict[int, float]] = {}
        for r in group:
            if r.get(key) is not None:
                by_seed.setdefault(r["seed"], {})[r["epochs"]] = r[key]
        out: Dict[int, List[float]] = {}
        for seed, depths in by_seed.items():
            if baseline_epochs not in depths:
                continue
            base = depths[baseline_epochs]
            for e, v in depths.items():
                if e != baseline_epochs:
                    out.setdefault(e, []).append(v - base)
        return out

    def floor_for(key: str) -> Tuple[Optional[float], int]:
        """(worst paired cross-seed disagreement, number of comparisons)."""
        out = []
        for (dm, sel), grp in by_model.items():
            avail = sorted({r["epochs"] for r in grp})
            if not avail:
                continue
            for e, deltas in paired_deltas(grp, min(avail), key).items():
                if len(deltas) > 1:
                    out.append(max(deltas) - min(deltas))
        return (max(out) if out else None), len(out)

    noise_floor, n_comparisons = floor_for("val_de")
    # p75 needs its OWN measured floor. It previously kept a hardcoded 2%
    # gate while val_de moved onto the measured one, and that mismatch
    # decided the ceiling on its own: with the real data every val_de effect
    # sat inside the 5.125 dE floor, yet p75 deltas of 1.2-2.5 were compared
    # against a 0.43-0.51 gate and failed, producing ceilings of 4/2/1 and a
    # binding ceiling of 1 epoch. That is the most restrictive answer
    # possible, from a threshold that was never a variance estimate.
    noise_floor_p75, _ = floor_for("p75")

    raw_spreads = []
    for (dm, sel), group in by_model.items():
        cells: Dict[int, List[float]] = {}
        for r in group:
            cells.setdefault(r["epochs"], []).append(r["val_de"])
        raw_spreads += [max(v) - min(v) for v in cells.values() if len(v) > 1]

    if noise_floor is not None:
        print(f"[noise] paired: worst cross-seed disagreement in a "
              f"depth-vs-baseline delta = {noise_floor:.3f} dE "
              f"({n_comparisons} comparisons).")
        if raw_spreads:
            print(f"        unpaired, for contrast: worst raw within-cell "
                  f"spread = {max(raw_spreads):.3f} dE.")
        if noise_floor_p75 is not None:
            print(f"        p75 floor (same paired method): "
                  f"{noise_floor_p75:.3f} dE.")
        print("        Depth differences smaller than the paired figure are "
              "not resolvable.\n")
    else:
        print("[noise] NO SEED REPEATS in this run, so there is no variance "
              "estimate and any ceiling below is provisional. Re-run with "
              "--seeds 42 43 to get one.\n")

    print("=" * 100)
    print("EPOCH-CEILING PROBE")
    print("=" * 100)
    print("\nAll arms within a model size share total steps, example-passes and C.")
    print("Only corpus diversity differs. Deltas are vs the e=1 arm.\n")

    ceilings = {}
    for (d_model, sel), group in sorted(by_model.items()):
        # One entry per depth: mean over seeds, with the spread carried along.
        cells: Dict[int, List[dict]] = {}
        for r in group:
            cells.setdefault(r["epochs"], []).append(r)
        group = []
        for e, reps in sorted(cells.items()):
            merged = dict(reps[0])
            merged["n_seeds"] = len(reps)
            merged["val_de"] = statistics.fmean(r["val_de"] for r in reps)
            merged["seed_spread"] = (max(r["val_de"] for r in reps)
                                     - min(r["val_de"] for r in reps)
                                     if len(reps) > 1 else None)
            for k in ("p75", "p95"):
                vals = [r[k] for r in reps if r.get(k) is not None]
                merged[k] = statistics.fmean(vals) if vals else None
            group.append(merged)
        base = next((r for r in group if r["epochs"] == 1), None)
        print(f"--- d_model={d_model} slot_encoder_layers={sel} ---")
        print(f"  {'epochs':>7} {'corpus':>10} {'val_de':>9} {'d_val_de':>9} "
              f"{'p75':>8} {'d_p75':>8} {'p95':>8} {'CE gap':>8} {'d_gap':>8} "
              f"{'seeds':>8}")
        base_gap = (base["val_loss"] - base["train_loss"]) if base and \
            base.get("val_loss") is not None and base.get("train_loss") is not None else None
        for r in group:
            gap = (r["val_loss"] - r["train_loss"]) \
                if r.get("val_loss") is not None and r.get("train_loss") is not None else None
            d_de = f"{r['val_de'] - base['val_de']:+.3f}" if base else "-"
            d_p75 = (f"{r['p75'] - base['p75']:+.3f}"
                     if base and r.get("p75") is not None and base.get("p75") is not None else "-")
            d_gap = f"{gap - base_gap:+.4f}" if gap is not None and base_gap is not None else "-"
            sp = r.get("seed_spread")
            sp_s = f"+-{sp / 2:.2f}" if sp is not None else f"n={r.get('n_seeds', 1)}"
            print(f"  {r['epochs']:>7} {r['corpus']:>10,} {r['val_de']:>9.3f} {d_de:>9} "
                  f"{(r['p75'] if r['p75'] is not None else float('nan')):>8.3f} {d_p75:>8} "
                  f"{(r['p95'] if r['p95'] is not None else float('nan')):>8.3f} "
                  f"{(gap if gap is not None else float('nan')):>8.4f} {d_gap:>8} "
                  f"{sp_s:>8}")

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
            # Tolerance = measured seed spread where available, else a 2%
            # placeholder that is explicitly NOT a variance estimate.
            tol = (noise_floor if noise_floor is not None
                   else 0.02 * abs(base["val_de"]) if base["val_de"] else 0.0)
            ceiling = 1
            for r in others:
                de_ok = (r["val_de"] - base["val_de"]) <= tol
                tol_p75 = (noise_floor_p75 if noise_floor_p75 is not None
                           else 0.02 * abs(base.get("p75") or 0.0))
                p75_ok = (base.get("p75") is None or r.get("p75") is None or
                          (r["p75"] - base["p75"]) <= tol_p75)
                if de_ok and p75_ok:
                    ceiling = max(ceiling, r["epochs"])
            ceilings[(d_model, sel)] = ceiling
            tested = sorted(r["epochs"] for r in others)
            basis = (f"measured floors val_de {noise_floor:.3f} / "
                     f"p75 {noise_floor_p75:.3f}"
                     if noise_floor is not None and noise_floor_p75 is not None
                     else "2% placeholder (NO seed data)")
            biggest = max((r["val_de"] - base["val_de"]) for r in others)
            if noise_floor is not None and biggest <= noise_floor:
                print(f"  -> NO DEGRADATION DETECTED up to {max(tested)} "
                      f"epoch(s). The largest val_de change ({biggest:+.3f}) "
                      f"is inside the {noise_floor:.3f} noise floor, so this "
                      f"is a NON-DETECTION, not a verified safe depth: any "
                      f"effect smaller than the floor would be invisible "
                      f"here.\n     [depths tested: {tested}; "
                      f"tolerance = {basis}]\n")
            else:
                print(f"  -> ceiling: {ceiling} epoch(s)   "
                      f"[depths tested: {tested}; tolerance = {basis}]\n")

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
    if excluded:
        print(f"{len(excluded)} of {len(dirs)} directories belong to a "
              f"DIFFERENT grid and were excluded (this is correct, not a "
              f"failure -- they are earlier probe runs sharing this OUT_ROOT):")
        for name in excluded:
            print(f"   - {name}")
        print()
    if skipped:
        print(f"{len(skipped)} of {len(dirs)} arms RAN but produced no usable "
              f"val_de:")
        for name in skipped:
            print(f"   - {name}")
        print("\nThat usually means the probe is mis-sized: too few steps, so"
              "\nthe model still emits EOS immediately and generates nothing to"
              "\nsimulate. Raise --total-steps (or --lr) until every arm"
              "\ngenerates valid structures before trusting any ceiling.\n")
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
                   help="Model sizes as d_model:slot_encoder_layers[:lr], "
                        "e.g. 128:2:1e-3 256:2:5e-4 512:4:2e-4. The optional "
                        "third field sets a PER-SIZE learning rate (from "
                        "scripts/lr_tuning.py); it falls back to --lr. LR is "
                        "held fixed across the depths of a size, which is "
                        "what isolates the repeat effect.")
    p.add_argument("--limit-de-examples", type=int, default=512,
                   help="Examples per DeltaE eval. Larger than the training "
                        "default because these are the probe's only outputs.")
    p.add_argument("--limit-val-examples", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=600)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds to repeat every (size, depth) cell at. More "
                        "than one is what makes a measured ceiling "
                        "believable: without a variance estimate there is no "
                        "way to tell a real degradation from init noise, and "
                        "that noise was measured at 52%% of the mean before "
                        "seeding was fixed. Two seeds doubles the arm count.")
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
        expected = {e: args.batch_size * args.total_steps // e
                    for e in args.depths}
        analyze(Path(args.out_root), expected)
        return

    sizes = DEFAULT_SIZES
    if args.sizes:
        sizes = []
        for spec in args.sizes:
            parts = spec.split(":")
            if len(parts) not in (2, 3):
                raise SystemExit(f"--sizes entry {spec!r} must be "
                                 f"d_model:slot_encoder_layers[:lr]")
            sizes.append((int(parts[0]), int(parts[1]),
                          float(parts[2]) if len(parts) == 3 else None))

    arms = build_arms(sizes, args.depths, args.total_steps, args.batch_size,
                      args.lr, args.seeds)

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
