#!/usr/bin/env python3
"""Fail when a production checkpoint's PRODUCTION.json and the config.json
beside it disagree on the hyperparameters a reader would quote.

Checked: learning_rate (PRODUCTION.json "lr"), batch_size, d_model,
slot_encoder_layers, decoder_layers, n_heads. A field missing from either file
is a failure too, so the check can never pass by having nothing to compare.

Why (Nov 18): lr_tuning.py saved config.json with ModelConfig's default
learning_rate 4.42e-5 and batch_size 64 instead of the run's 3.104e-4 and 256,
and ModelConfig.tag() put the defaults into every eval directory name.

    python scripts/check_production_config.py                 # every production folder found
    python scripts/check_production_config.py <dir> [<dir>...]

Folders searched by default: results/production/*/ and
data/checkpoints/production/*/ (each one holding a PRODUCTION.json).
Exit 0 when all agree, 1 otherwise. Needs no torch.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
# (PRODUCTION.json key, config.json key)
FIELDS = (("lr", "learning_rate"), ("batch_size", "batch_size"), ("d_model", "d_model"),
          ("slot_encoder_layers", "slot_encoder_layers"),
          ("decoder_layers", "decoder_layers"), ("n_heads", "n_heads"))


def problems(folder: Path) -> List[str]:
    prod_p, cfg_p = folder / "PRODUCTION.json", folder / "config.json"
    if not cfg_p.is_file():
        return [f"{folder}: no config.json beside PRODUCTION.json"]
    prod, cfg = json.load(open(prod_p)), json.load(open(cfg_p))
    out = []
    for pk, ck in FIELDS:
        if pk not in prod or ck not in cfg:
            out.append(f"{folder}: {pk!r} missing from "
                       + ("PRODUCTION.json" if pk not in prod else "config.json"))
            continue
        a, b = prod[pk], cfg[ck]
        same = (math.isclose(float(a), float(b), rel_tol=1e-6) if pk == "lr"
                else int(a) == int(b))
        if not same:
            out.append(f"{folder}: PRODUCTION.json {pk}={a} but config.json {ck}={b}")
    return out


def default_folders() -> List[Path]:
    found = []
    for root in (REPO / "results" / "production", REPO / "data" / "checkpoints" / "production"):
        if root.is_dir():
            found += sorted(p.parent for p in root.glob("*/PRODUCTION.json"))
    return found


def main(argv: List[str]) -> int:
    folders = [Path(a) for a in argv] or default_folders()
    if not folders:
        print("[check] no production folder found")
        return 1
    bad = [msg for f in folders for msg in problems(f)]
    for f in folders:
        print(f"[check] {f}")
    for msg in bad:
        print(f"[FAIL] {msg}")
    if not bad:
        print(f"[check] PRODUCTION.json and config.json agree in {len(folders)} folder(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
