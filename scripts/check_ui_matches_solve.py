#!/usr/bin/env python3
"""The demo's Paper preset runs the paper's search: check it end to end.

One target, two runs on the same checkpoint, seed and pool:
  1. through the demo server's HTTP API, the request the UI sends when the
     Paper preset is selected (POST /api/solve with preset "strong", the
     UI's default material selection, no constraints);
  2. a direct call to inference.src.solve.solve() with the knobs from
     inference/src/presets.py "strong", which are test_eval.py's paper
     settings (N 250, top-10 all refined, 100 DoG steps, tolerance 0, no MC).
The chosen stack, its thicknesses, achieved Lab and DeltaE00 must agree to
floating-point noise; any difference is a failure (exit 1).

    # on CRC (GPU or CPU), a tier A test row as the target:
    python scripts/check_ui_matches_solve.py --row 0
    # faster smoke test of the plumbing:
    python scripts/check_ui_matches_solve.py --row 0 --preset fast
    # no test shards at hand:
    python scripts/check_ui_matches_solve.py --lab 60 5 -8 --checkpoint DIR

The pool is the server's own default selection (/api/pool deduped_default),
because the UI can only offer the materials the server holds; a tier A row's
own pool includes synthetic materials the UI cannot express without uploads.
What is checked is the search, which is what the presets decide.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "inference" / "frontend"))

TOL_DE = 1e-6       # DeltaE00
TOL_LAB = 1e-5      # Lab components
TOL_NM = 1e-4       # thickness, nm


def tier_a_target(test_dir: Path, row: int, seed: int):
    import itertools
    from src.dataset import FlexThinFilmDataset
    from src.materials_vocab import denormalize_lab
    ds = FlexThinFilmDataset(test_dir / "tier_a", seed=seed, split="train",
                             verbose=False, streaming=True)
    ex = next(itertools.islice(ds, row, row + 1))
    return [float(x) for x in denormalize_lab(ex.lab)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(REPO / "data/checkpoints/production/current"))
    p.add_argument("--test-dir", default=str(REPO / "data/test"))
    p.add_argument("--row", type=int, default=0, help="tier A row (test_eval.py order)")
    p.add_argument("--lab", type=float, nargs=3, default=None, help="target Lab instead of a row")
    p.add_argument("--preset", default="strong")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()

    import server
    from fastapi.testclient import TestClient
    from inference.src.presets import knobs_for
    from inference.src.schema import InferenceKnobs, InferenceSpec
    from inference.src.solve import solve
    from src.materials_vocab import normalize_lab

    lab = a.lab if a.lab else tier_a_target(Path(a.test_dir), a.row, a.seed)
    print(f"[check] target Lab {lab}  preset {a.preset}  seed {a.seed}")

    server._startup(Path(a.checkpoint), None, force_cpu=a.cpu, prewarm=False)
    client = TestClient(server._make_app())
    subset = client.get("/api/pool").json()["deduped_default"]

    # 1. what the UI sends
    r = client.post("/api/solve", json={"target_lab": lab, "preset": a.preset,
                                        "knobs": {"seed": a.seed},
                                        "pool_subset": subset})
    if r.status_code != 200:
        print(f"[FAIL] server returned {r.status_code}: {r.text[:500]}")
        return 1
    ui = r.json()["chosen"]

    # 2. solve() directly, the way the evaluation scripts call it
    pool = server._effective_pool(subset_names=subset, custom=[])
    knobs = InferenceKnobs(seed=a.seed, **knobs_for(a.preset))
    spec = InferenceSpec(
        target_lab_raw=tuple(lab),
        target_lab_normalised=tuple(float(x) for x in normalize_lab(list(lab)).tolist()),
        constraints=[], enforce_during=[], enforce_post=[], knobs=knobs,
        parsed_disclaimer="check_ui_matches_solve")
    direct = solve(model=server._MODEL, pool=pool, spec=spec, model_tag=server._MODEL_TAG,
                   model_sha256=server._MODEL_SHA, device=server._DEVICE).chosen

    bad = []
    if list(ui["material_names"]) != list(direct.material_names):
        bad.append(f"materials {ui['material_names']} vs {direct.material_names}")
    elif any(abs(x - y) > TOL_NM for x, y in zip(ui["thicknesses_nm"], direct.thicknesses_nm)):
        bad.append(f"thicknesses {ui['thicknesses_nm']} vs {direct.thicknesses_nm}")
    if any(abs(x - y) > TOL_LAB for x, y in zip(ui["achieved_lab"], direct.achieved_lab)):
        bad.append(f"achieved Lab {ui['achieved_lab']} vs {direct.achieved_lab}")
    if not math.isclose(ui["delta_e"], direct.delta_e, abs_tol=TOL_DE):
        bad.append(f"DeltaE {ui['delta_e']} vs {direct.delta_e}")

    print(f"[check] UI     : dE {ui['delta_e']:.6f}  {list(zip(ui['material_names'], [round(t, 3) for t in ui['thicknesses_nm']]))}")
    print(f"[check] solve(): dE {direct.delta_e:.6f}  {list(zip(direct.material_names, [round(t, 3) for t in direct.thicknesses_nm]))}")
    print(f"[check] checkpoint sha {server._MODEL_SHA}  knobs {knobs_for(a.preset)}")
    for b in bad:
        print(f"[FAIL] {b}")
    if not bad:
        print("[check] the demo's preset reproduces solve() to floating-point noise")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
