#!/usr/bin/env python3
"""
Dispatch the IsoFLOP sweep
==========================

Phase 0.3. Turns the grid in `src/scaling/configs.py` into one training run
per config, and hands `scripts/fit_scaling.py` a directory it can fit.

The grid lives in configs.py, not here: this module only renders it into
commands, so the submitted sweep and the analysed sweep cannot disagree about
what the grid was.

Every emitted command carries --streaming and --limit-shard-aligned, and that
is asserted before anything is submitted. Both have already cost a full
submission apiece:

  * without --streaming, training.py materialises an entire epoch in memory
    (~40KB/example) and the large-corpus configs are OOM-killed;
  * without --limit-shard-aligned, a corpus smaller than the full dataset is
    scattered across every shard, so each epoch re-reads the whole thing.

Usage
-----
    python scripts/scaling_sweep.py --dry-run
    python scripts/scaling_sweep.py --n-configs
    python scripts/scaling_sweep.py --emit-config 7 --data-dir ... --out-root ...
    python scripts/scaling_sweep.py --data-dir ... --dispatch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.scaling.configs import (
    CORPUS_EXAMPLES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_POINTS,
    DEFAULT_SPAN,
    DE_EXAMPLES,
    WALL_MARGIN,
    SweepConfig,
    build_grid,
    describe,
)

# Top rung is set by where a rung can still STRADDLE the prior N*. Under
# --qos=short the wall-clock floor rises faster in C than N* does (floor ~ C,
# N* ~ sqrt(C)), so past ~4e15 every feasible size sits above the optimum and
# the parabola is one-sided. A longer QoS moves this a long way: 12h reaches
# 2.5e16 and 24h reaches production scale. See slurms/scaling_sweep.sh.
DEFAULT_BUDGETS = [1e14, 4e14, 1.4e15, 4e15]


def emit_command(cfg: SweepConfig, data_dir: str, out_root: str,
                 seed: int, de_examples: int, val_examples: int,
                 num_workers: int, save_every: int,
                 use_slurm: bool) -> List[str]:
    # cfg.seed overrides the sweep-wide seed: a repeat arm is the SAME
    # (budget, N, D) point under a different init, which is what gives the
    # fit an error bar rather than a second grid point.
    seed = cfg.seed if cfg.seed is not None else seed
    save_dir = f"{out_root}/{cfg.name}_s{seed}"
    n_heads = cfg.n_heads
    # DeltaE at the end only. The fit reads the final value, and an eval per
    # save tick would be a large fraction of the shorter configs.
    #
    # steps + 1, not steps: training.py's "final" save already forces a
    # DeltaE eval, and cfg.steps would ALSO fire at the epoch-boundary save,
    # which lands on the same global_step. That duplicate buys nothing (same
    # weights, same eval slice) and spends wall clock the 3h QoS does not
    # have to spare. Made unreachable so exactly one eval runs per config.
    de_every = cfg.steps + 1

    if use_slurm:
        env = {
            "DATA_DIR": data_dir, "SAVE_DIR": save_dir,
            "HEAD_MODE": "cross_attn", "D_MODEL": str(cfg.d_model),
            "N_HEADS": str(n_heads),
            "SLOT_ENCODER_LAYERS": str(cfg.slot_encoder_layers),
            "DECODER_LAYERS": "1", "LR": f"{cfg.lr:.6e}",
            "BATCH_SIZE": str(cfg.batch_size), "EPOCHS": str(cfg.epochs),
            "LIMIT_EXAMPLES": str(cfg.limit_examples),
            "LIMIT_SHARD_ALIGNED": "1", "STREAMING": "1",
            "LIMIT_VAL_EXAMPLES": str(val_examples),
            "LIMIT_DE_EXAMPLES": str(de_examples),
            "DE_EVERY": str(de_every), "DE_ON_EPOCH_END": "0",
            "SAVE_EVERY": str(save_every), "NUM_WORKERS": str(num_workers),
            "SEED": str(seed),
        }
        return [f"{k}={v}" for k, v in env.items()] + ["sbatch", "slurms/training.sh"]

    return [
        "python", "scripts/training.py",
        "--data-dir", data_dir, "--save-dir", save_dir,
        "--head-mode", "cross_attn", "--d-model", str(cfg.d_model),
        "--n-heads", str(n_heads),
        "--slot-encoder-layers", str(cfg.slot_encoder_layers),
        "--decoder-layers", "1", "--lr", f"{cfg.lr:.6e}",
        "--batch-size", str(cfg.batch_size), "--epochs", str(cfg.epochs),
        "--limit-examples", str(cfg.limit_examples),
        "--limit-shard-aligned", "--streaming",
        "--limit-val-examples", str(val_examples),
        "--limit-de-examples", str(de_examples),
        "--de-every", str(de_every), "--no-de-on-epoch-end",
        "--save-every", str(save_every), "--num-workers", str(num_workers),
        "--seed", str(seed),
    ]


def build_commands(grid: Sequence[SweepConfig], **kw) -> List[List[str]]:
    cmds = [emit_command(c, **kw) for c in grid]
    use_slurm = kw["use_slurm"]
    required = (("STREAMING=1", "LIMIT_SHARD_ALIGNED=1") if use_slurm
                else ("--streaming", "--limit-shard-aligned"))
    for cmd in cmds:
        joined = " ".join(cmd)
        for token in required:
            if token not in joined:
                raise SystemExit(
                    f"internal error: emitted command missing {token!r}; "
                    f"refusing to submit (this OOMs or thrashes I/O).\n  {joined}")
    return cmds


def main() -> None:
    p = argparse.ArgumentParser(description="Dispatch the IsoFLOP sweep")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-root", type=str,
                   default="data/checkpoints/scaling_sweep")
    p.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    p.add_argument("--span", type=float, default=DEFAULT_SPAN,
                   help="ratio of largest to smallest N within a budget")
    p.add_argument("--points", type=int, default=DEFAULT_POINTS,
                   help="sizes per budget")
    p.add_argument("--repeat-seed", type=int, default=None,
                   help="re-run each rung's middle size under this second seed")
    p.add_argument("--max-wall-hours", type=float, default=None,
                   help="override the 3h --qos=short cap when a longer QoS is available")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--corpus", type=int, default=CORPUS_EXAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-de-examples", type=int, default=DE_EXAMPLES)
    p.add_argument("--limit-val-examples", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-configs", action="store_true")
    p.add_argument("--emit-config", type=int, default=None)
    p.add_argument("--local", action="store_true")
    p.add_argument("--dispatch", action="store_true")
    args = p.parse_args()

    cap = (args.max_wall_hours * 3600 * WALL_MARGIN
           if args.max_wall_hours else None)
    grid = build_grid(args.budgets, span=args.span, points=args.points,
                      batch_size=args.batch_size, corpus=args.corpus,
                      wall_cap_sec=cap, repeat_seed=args.repeat_seed)

    if args.n_configs:
        print(len(grid))
        return

    kw = dict(data_dir=args.data_dir or "", out_root=args.out_root,
              seed=args.seed, de_examples=args.limit_de_examples,
              val_examples=args.limit_val_examples,
              num_workers=args.num_workers, save_every=args.save_every,
              use_slurm=not args.local)

    if args.emit_config is not None:
        if not 0 <= args.emit_config < len(grid):
            raise SystemExit(f"--emit-config {args.emit_config} out of range "
                             f"(0..{len(grid) - 1})")
        if not args.data_dir:
            raise SystemExit("--emit-config requires --data-dir")
        # Always the LOCAL form here: the SLURM wrapper passes these straight
        # to training.py, so it needs CLI flags, not env assignments. Strip
        # the leading "python scripts/training.py", which the wrapper adds.
        local_kw = dict(kw, use_slurm=False)
        print(" ".join(build_commands(grid, **local_kw)[args.emit_config][2:]))
        return

    print(describe(grid, args.corpus))
    infeasible = [c for c in grid if not c.fits_qos_short]
    if infeasible:
        print(f"\nREFUSING to dispatch: {len(infeasible)} config(s) cannot "
              f"finish inside --qos=short. Lower the top budget or narrow the "
              f"bracket.")
        if args.dispatch:
            raise SystemExit(1)
    if args.dry_run:
        return
    if not args.data_dir:
        raise SystemExit("--data-dir is required unless --dry-run/--n-configs")

    cmds = build_commands(grid, **kw)
    print(f"\n{len(cmds)} configs:\n")
    for c in cmds:
        print("  " + " ".join(c))
    if args.dispatch:
        print(f"\nDispatching {len(cmds)}...")
        for c in cmds:
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
        print("\n(Add --dispatch to submit.)")


if __name__ == "__main__":
    main()
