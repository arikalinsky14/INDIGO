#!/usr/bin/env python3
"""
Generate N shards of the INDIGO training dataset.

Each shard is fully determined by its shard_id, so re-running with the same
arguments produces identical data, and extending the dataset is as simple
as picking up where the previous run left off.

Examples
--------
# 2M training rows, 5000 rows/shard → 400 shards
python scripts/generate_dataset.py --total-rows 2000000 --rows-per-shard 5000 \\
    --start-shard-id 0 --output-dir data/train

# Extend with 500k more rows without touching existing shards
python scripts/generate_dataset.py --total-rows 500000 --rows-per-shard 5000 \\
    --start-shard-id 400 --output-dir data/train
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-rows", type=int, required=True,
                   help="Total rows to generate across all shards.")
    p.add_argument("--rows-per-shard", type=int, default=5000)
    p.add_argument("--start-shard-id", type=int, default=0,
                   help="First shard id (lets you extend an existing dataset)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--incidence-angle", type=int, default=0)
    p.add_argument("--layer-lambda", type=float, default=4.5)
    p.add_argument("--layer-min", type=int, default=2)
    p.add_argument("--layer-max", type=int, default=10)
    p.add_argument("--greyscale-threshold", type=float, default=8.0)
    p.add_argument("--greyscale-keep-prob", type=float, default=0.2)
    p.add_argument("--p-real", type=float, default=0.15,
                   help="Per material: probability of pulling from held-in real "
                        "instead of generating a fresh synthetic (default: 0.15)")
    p.add_argument("--pool-size-min", type=int, default=4)
    p.add_argument("--pool-size-max", type=int, default=32)
    p.add_argument("--use-held-out-reals", action="store_true",
                   help="Tier-B test set: use the held-out real materials instead")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip shards whose parquet already exists")
    p.add_argument("--jll-materials-dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true",
                   help="Pass --verbose to compile_datasets for per-shard progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_shards = (args.total_rows + args.rows_per_shard - 1) // args.rows_per_shard
    last_shard_rows = args.total_rows - (n_shards - 1) * args.rows_per_shard

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": args.total_rows,
        "rows_per_shard": args.rows_per_shard,
        "n_shards": n_shards,
        "shard_id_range": [args.start_shard_id, args.start_shard_id + n_shards - 1],
        "args": vars(args),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(top_manifest, indent=2))
    print(f"[INFO] Wrote run manifest to {output_dir / 'run_manifest.json'}")
    print(f"[INFO] Will generate {n_shards} shards "
          f"(ids {args.start_shard_id}..{args.start_shard_id + n_shards - 1})")

    for i in range(n_shards):
        shard_id = args.start_shard_id + i
        rows = args.rows_per_shard if i < n_shards - 1 else last_shard_rows

        shard_path = (
            output_dir
            / f"angle_{args.incidence_angle:02d}_substrate_CSi"
            / f"shard_{shard_id:05d}.parquet"
        )
        if args.skip_existing and shard_path.exists():
            print(f"[skip] {shard_path}")
            continue

        cmd = [
            sys.executable, "create_dataset/src/compile_datasets.py",
            "--target-rows", str(rows),
            "--shard-id", str(shard_id),
            "--incidence-angle", str(args.incidence_angle),
            "--layer-lambda", str(args.layer_lambda),
            "--layer-min", str(args.layer_min),
            "--layer-max", str(args.layer_max),
            "--greyscale-threshold", str(args.greyscale_threshold),
            "--greyscale-keep-prob", str(args.greyscale_keep_prob),
            "--p-real", str(args.p_real),
            "--pool-size-min", str(args.pool_size_min),
            "--pool-size-max", str(args.pool_size_max),
            "--output-dir", str(output_dir),
        ]
        if args.use_held_out_reals:
            cmd.append("--use-held-out-reals")
        if args.jll_materials_dir:
            cmd += ["--jll-materials-dir", args.jll_materials_dir]
        if args.verbose:
            cmd.append("--verbose")

        print(f"[run] shard {shard_id} ({i + 1}/{n_shards}): {rows} rows")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Shard {shard_id} failed; aborting.", file=sys.stderr)
            sys.exit(1)

    print(f"[done] {n_shards} shards written to {output_dir}")


if __name__ == "__main__":
    main()
