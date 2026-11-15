#!/usr/bin/env python3
"""Copy the production model's weights to one fixed, documented place.

    data/checkpoints/production/
        README.md                     what this folder is (written here)
        current -> <version>/         the model to use; load this path
        <version>/                    one folder per promoted model
            model.pt, config.json, meta.json   copied from the training run
            PRODUCTION.json           where it came from and why it was chosen

The training run's own weights stay where lr_tuning.py saved them
(outputs/lr_curves/<head>/weights/<cell tag>/lr<rate>/): this is a copy, so
every path the study and the eval outputs recorded stays valid. A version is
never overwritten; promoting another model adds a folder and moves `current`.

Two ways in:

  # One weights folder, chosen by hand:
  python scripts/promote_production.py \\
      --weights outputs/lr_curves/cross_attn/weights/ep1_lim39959808_d352_se6_bs256_b20.999_r5/lr3.104000e-04

  # A production learning-rate curve, chosen by the production rule (the
  # highest rate that did not diverge). slurms/lr_grid.sh calls this after
  # every STAGE=lrcurve task on an explicit-D cell; it promotes only once the
  # rule can be applied: a diverged rate exists and the grid step just below
  # the lowest diverged rate has finished.
  python scripts/promote_production.py \\
      --weights-root outputs/lr_curves/cross_attn/weights \\
      --cell ep1_lim39959808_d352_se6_bs256_b20.999

Needs no torch (the analysis-stack rule): it reads meta.json and hashes bytes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_edges import DIVERGENCE_VAL_LOSS  # noqa: E402

DEFAULT_DEST = "data/checkpoints/production"
FILES = ("model.pt", "config.json", "meta.json")
RULE = "highest learning rate that did not diverge"

README = """# Production model

`current/` is the INDIGO production model. Load it like any checkpoint:

    CHECKPOINT=data/checkpoints/production/current sbatch slurms/gamut_eval.sh

Every other folder here is one promoted model, never overwritten. Its
`PRODUCTION.json` says which training run it was copied from, the rule that
chose it, its metrics and the sha256 of `model.pt` (the first 16 characters are
the `model_sha256` every eval output records).

Written by scripts/promote_production.py; see that script to promote another.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_meta(wdir: Path) -> dict:
    return json.load(open(wdir / "meta.json"))


def is_diverged(meta: dict) -> bool:
    vl = meta.get("final_val_loss")
    return vl is None or not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS


def cell_trials(root: Path, cell: str) -> List[Tuple[float, Path, dict]]:
    """Every saved rate of one cell: the cell's own folder and its curve-tag
    siblings (cell_r0, cell_lo0, ...), but not a repeat seed's (cell_s43_...)."""
    pat = re.compile(re.escape(cell) + r"(_(?!s\d)[a-z]+\d+)?")
    out = []
    for tag_dir in sorted(root.iterdir()):
        if not (tag_dir.is_dir() and pat.fullmatch(tag_dir.name)):
            continue
        for wdir in sorted(tag_dir.glob("lr*")):
            if all((wdir / f).is_file() for f in FILES):
                meta = read_meta(wdir)
                out.append((float(meta["lr"]), wdir, meta))
    return sorted(out, key=lambda t: t[0])


def choose_by_rule(trials) -> Tuple[Optional[Path], str]:
    """(chosen folder, reason). None until the rule can be applied."""
    if not trials:
        return None, "no finished rates"
    div = [lr for lr, _, m in trials if is_diverged(m)]
    if not div:
        return None, (f"no diverged rate yet among {len(trials)} finished; the "
                      "highest stable rate is not known to be the edge")
    lowest_div = min(div)
    stable = [(lr, w) for lr, w, m in trials if not is_diverged(m) and lr < lowest_div]
    if not stable:
        return None, "every finished rate below the first divergence is missing"
    lr, wdir = stable[-1]
    # The rate one grid step under the lowest diverged one must be present: a
    # gap means a task has not finished and a higher stable rate may exist.
    rates = sorted({t[0] for t in trials})
    steps = [b / a for a, b in zip(rates, rates[1:]) if a < lowest_div]
    step = min(steps) if steps else None
    if step and lowest_div / lr > step * 1.05:
        return None, (f"gap: best stable rate {lr:.3e} is {lowest_div / lr:.2f}x under "
                      f"the first diverged rate {lowest_div:.3e} (grid step {step:.2f}x)")
    return wdir, f"{RULE}: {lr:.4e} (lowest diverged {lowest_div:.4e})"


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def promote(wdir: Path, dest: Path, reason: str, dry_run: bool) -> None:
    meta = read_meta(wdir)
    if is_diverged(meta):
        sys.exit(f"[promote] refusing: {wdir} diverged (val loss {meta.get('final_val_loss')})")
    digest = sha256(wdir / "model.pt")
    version = f"{wdir.parent.name}__lr{float(meta['lr']):.4e}"
    vdir = dest / version
    current = dest / "current"
    if current.exists() and current.resolve() == vdir.resolve():
        print(f"[promote] {vdir} is already current; nothing to do")
        return
    print(f"[promote] {wdir}\n       -> {vdir}\n       ({reason})")
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    if vdir.exists():
        # Same source promoted before (and later superseded): reuse it only if
        # the bytes match, never overwrite.
        if sha256(vdir / "model.pt") != digest:
            sys.exit(f"[promote] {vdir} exists with different weights; not touching it")
    else:
        tmp = dest / f".{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for f in FILES:
            shutil.copy2(wdir / f, tmp / f)
        if sha256(tmp / "model.pt") != digest:
            sys.exit("[promote] copy does not match the source; nothing promoted")
        json.dump({
            "source": str(wdir.resolve()),
            "chosen_by": reason,
            "lr": meta["lr"],
            "seed": meta.get("seed"),
            "epochs": meta.get("epochs"),
            "train_examples": meta.get("train_examples"),
            "final_val_loss": meta.get("final_val_loss"),
            "final_val_de": meta.get("final_val_de"),
            "model_sha256": digest,
            "model_sha256_16": digest[:16],
            "promoted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        }, open(tmp / "PRODUCTION.json", "w"), indent=2)
        os.replace(tmp, vdir)
    # Swap `current` atomically, so a reader never sees it missing.
    link_tmp = dest / ".current.tmp"
    if link_tmp.is_symlink() or link_tmp.exists():
        link_tmp.unlink()
    link_tmp.symlink_to(version)
    os.replace(link_tmp, current)
    (dest / "README.md").write_text(README)
    print(f"[promote] current -> {version}  (sha256 {digest[:16]})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", help="one weights folder (model.pt, config.json, meta.json)")
    p.add_argument("--weights-root", help="lr_tuning.py's --save-weights-dir")
    p.add_argument("--cell", help="cell tag without its curve suffix, e.g. "
                                  "ep1_lim39959808_d352_se6_bs256_b20.999")
    p.add_argument("--dest", default=DEFAULT_DEST)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if bool(a.weights) == bool(a.cell):
        p.error("give --weights, or --weights-root with --cell")
    if a.weights:
        wdir = Path(a.weights)
        if not all((wdir / f).is_file() for f in FILES):
            sys.exit(f"[promote] {wdir} is missing one of {FILES}")
        promote(wdir, Path(a.dest), "chosen by hand", a.dry_run)
        return
    if not a.weights_root:
        p.error("--cell needs --weights-root")
    trials = cell_trials(Path(a.weights_root), a.cell)
    for lr, wdir, meta in trials:
        print(f"  {lr:.4e}  CE {meta.get('final_val_loss'):.4f}"
              f"{'  diverged' if is_diverged(meta) else ''}  {wdir.parent.name}")
    wdir, reason = choose_by_rule(trials)
    if wdir is None:
        print(f"[promote] not yet: {reason}")
        return
    promote(wdir, Path(a.dest), reason, a.dry_run)


if __name__ == "__main__":
    main()
