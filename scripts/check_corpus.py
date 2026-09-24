#!/usr/bin/env python3
"""
Check a generated corpus on disk: integrity, size, and room to extend it.
=========================================================================

Run this BEFORE extending a corpus and AFTER the extension finishes.

Integrity
---------
create_dataset/src/compile_datasets.py writes shards with a plain
`df.to_parquet(output_path)` -- straight to the final path, with no
write-to-temp-then-rename. A worker killed mid-write (TIME_LIMIT on a chunked
run, a full filesystem, a node failure) therefore leaves a TRUNCATED parquet
at the real path, and `--skip-existing` only tests whether that path exists.
So a corrupt shard is skipped forever by every later resume, and the corpus
silently carries a hole.

Two signals catch it, and both are cheap:

  * The manifest sidecar is written AFTER the parquet, so a parquet with no
    matching .manifest.json is a write that did not finish.
  * The parquet footer carries num_rows without reading any row groups, so a
    short or unreadable shard is detectable in milliseconds per file.

Anything flagged should be deleted (parquet AND sidecar) and regenerated: the
same START_SHARD_ID run will recreate exactly those ids, since --skip-existing
skips the shards that are fine.

Space
-----
Projects what extending by N shards would cost, from the MEASURED mean shard
size of this corpus rather than an assumed one, and compares that against the
free space actually reported for the filesystem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--extend-shards", type=int, default=0,
                   help="project the cost of adding this many more shards")
    p.add_argument("--expect-rows-per-shard", type=int, default=5000)
    p.add_argument("--skip-footers", action="store_true",
                   help="skip the per-file row-count read (sidecar check only)")
    args = p.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"--data-dir {root} is not a directory")

    shards = sorted(root.rglob("shard_*.parquet"))
    if not shards:
        raise SystemExit(f"no shard_*.parquet under {root}")

    total_bytes = sum(s.stat().st_size for s in shards)
    mean_bytes = total_bytes / len(shards)

    print("=" * 78)
    print(f"CORPUS  {root}")
    print("=" * 78)
    print(f"  shards on disk : {len(shards):,}")
    print(f"  size on disk   : {human(total_bytes)}")
    print(f"  mean per shard : {human(mean_bytes)}")
    ids = []
    for s in shards:
        try:
            ids.append(int(s.stem.split("_")[-1]))
        except ValueError:
            pass
    if ids:
        ids.sort()
        print(f"  shard id range : {ids[0]} to {ids[-1]}")
        missing = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
        if missing:
            print(f"  GAPS           : {len(missing)} missing id(s), "
                  f"first few {missing[:10]}")
        else:
            print(f"  gaps           : none")

    # ---- integrity -------------------------------------------------------
    print("\nINTEGRITY")
    no_sidecar = [s for s in shards
                  if not s.with_suffix(".manifest.json").exists()]
    if no_sidecar:
        print(f"  {len(no_sidecar)} parquet(s) with NO manifest sidecar -- the "
              f"sidecar is written after\n  the parquet, so these are "
              f"unfinished writes. Delete and regenerate:")
        for s in no_sidecar[:10]:
            print(f"    {s}")
        if len(no_sidecar) > 10:
            print(f"    ... and {len(no_sidecar) - 10} more")
    else:
        print("  every parquet has its manifest sidecar")

    bad = []
    if not args.skip_footers:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("  (pyarrow unavailable, skipping the row-count check)")
        else:
            rows = 0
            for s in shards:
                try:
                    n = pq.ParquetFile(s).metadata.num_rows
                except Exception as exc:                      # truncated footer
                    bad.append((s, f"unreadable: {exc}"))
                    continue
                rows += n
                if n != args.expect_rows_per_shard:
                    bad.append((s, f"{n} rows, expected "
                                   f"{args.expect_rows_per_shard}"))
            print(f"  total rows     : {rows:,}")
            if bad:
                print(f"  {len(bad)} shard(s) short or unreadable:")
                for s, why in bad[:10]:
                    print(f"    {s.name}: {why}")
                if len(bad) > 10:
                    print(f"    ... and {len(bad) - 10} more")
            else:
                print("  every shard reads and has the expected row count")

    # ---- space -----------------------------------------------------------
    usage = shutil.disk_usage(root)
    print(f"\nFILESYSTEM  {root}")
    print(f"  total {human(usage.total)}   used {human(usage.used)}   "
          f"free {human(usage.free)}")
    print("  NOTE: this is the filesystem, not your QUOTA. On a shared /ix1")
    print("        allocation the quota usually binds first -- check it with")
    print("        crc-quota (or your site's equivalent) before trusting this.")

    if args.extend_shards:
        need = args.extend_shards * mean_bytes
        print(f"\nEXTENSION  +{args.extend_shards:,} shards")
        print(f"  projected size : {human(need)}   "
              f"(at this corpus's measured {human(mean_bytes)}/shard)")
        print(f"  corpus after   : {human(total_bytes + need)} "
              f"in {len(shards) + args.extend_shards:,} shards")
        headroom = usage.free - need
        if headroom < 0:
            print(f"  VERDICT        : WILL NOT FIT -- short by {human(-headroom)}")
        elif headroom < need * 0.2:
            print(f"  VERDICT        : fits, but only {human(headroom)} to spare "
                  f"(<20% margin)")
        else:
            print(f"  VERDICT        : fits, {human(headroom)} to spare")
        print("  A full filesystem mid-generation truncates the shard being")
        print("  written, and --skip-existing will then skip it forever. Leave")
        print("  margin, and re-run this script afterwards.")

    sys.exit(1 if (no_sidecar or bad) else 0)


if __name__ == "__main__":
    main()
