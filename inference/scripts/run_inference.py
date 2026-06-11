#!/usr/bin/env python3
"""
INDIGO inference CLI.

Minimal wrapper over `inference.src.solve.solve_from_checkpoint` for
command-line use. Future LLM-based prompt parsing lives in
`inference/src/parse.py`; this CLI takes a structured spec file directly.

Examples
--------
    # Greyscale target, default knobs, JLL pool from the trained library
    python inference/scripts/run_inference.py \\
        --checkpoint data/checkpoints/<tag>/latest \\
        --target-lab 60 5 -8 \\
        --pool-dir src/_jll_materials

    # With a JSON pool file and a JSON constraints file
    python inference/scripts/run_inference.py \\
        --checkpoint data/checkpoints/<tag>/latest \\
        --target-lab 70 0 0 \\
        --pool inference/example_pool.json \\
        --constraints inference/example_constraints.json \\
        --top-k 5 --ensemble-n 500 --tolerance 5.0 --lambda 1.0 \\
        --output inference/outputs/run1.json

The output Result JSON is the same envelope the GUI consumes — success
runs carry `chosen`, failure runs carry `errors` and `chosen=null`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.material_features import load_jll_directory, NUM_LAMBDA
from src.materials_vocab import normalize_lab

from inference.src.schema import (
    InferenceKnobs, InferenceSpec, MaterialEntry,
)
from inference.src.solve import solve_from_checkpoint


# ----------------------------------------------------------------------------
# Pool / spec loading
# ----------------------------------------------------------------------------

def load_pool_from_json(path: Path) -> List[MaterialEntry]:
    """JSON: [{"canonical_name": "...", "n": [...], "k": [...], "source": "..."}].

    `n` and `k` must each be NUM_LAMBDA-long lists on the canonical grid.
    """
    data = json.loads(Path(path).read_text())
    pool: List[MaterialEntry] = []
    for entry in data:
        n = np.asarray(entry["n"], dtype=np.float32)
        k = np.asarray(entry["k"], dtype=np.float32)
        if n.shape != (NUM_LAMBDA,) or k.shape != (NUM_LAMBDA,):
            raise ValueError(
                f"material {entry.get('canonical_name', '?')}: n/k must be "
                f"length {NUM_LAMBDA}, got n={n.shape} k={k.shape}"
            )
        pool.append(MaterialEntry(
            canonical_name=str(entry["canonical_name"]),
            n=n, k=k, source=str(entry.get("source", "user")),
        ))
    return pool


def load_pool_from_jll(directory: Path) -> List[MaterialEntry]:
    """Load every JLL CSV in `directory`. Sorted by canonical name."""
    mnk = load_jll_directory(directory)
    out: List[MaterialEntry] = []
    for name in sorted(mnk):
        m = mnk[name]
        out.append(MaterialEntry(
            canonical_name=m.name, n=m.n.astype(np.float32),
            k=m.k.astype(np.float32), source=m.source,
        ))
    return out


def build_spec(
    target_lab: List[float],
    constraints_path: Path | None,
    knobs: InferenceKnobs,
) -> InferenceSpec:
    """Assemble an InferenceSpec from the CLI args + an optional constraint
    JSON file (the LLM parse target).

    Constraints JSON schema (one of):

        {"kind": "allowed_subset", "allowed_names": ["Ag", "SiO2"]}
        {"kind": "layer_identity", "position": 0, "material_name": "Ag"}
        {"kind": "adjacent_forbidden",
         "forbidden_pairs": [["Ag", "TiO2"]]}
        {"kind": "thickness_range", "min_nm": 20, "max_nm": 150,
         "position": null}
        {"kind": "layer_count", "min_layers": 2, "max_layers": 6}
        {"kind": "ordering_before", "name_a": "Ag", "name_b": "SiO2"}
        {"kind": "total_thickness", "max_total_nm": 600,
         "markovian_decode": true}
        {"kind": "symmetry", "match_thickness": true}
    """
    target = tuple(float(x) for x in target_lab)
    norm = normalize_lab(list(target))
    constraints = []
    if constraints_path is not None and Path(constraints_path).exists():
        data = json.loads(Path(constraints_path).read_text())
        if isinstance(data, dict):
            data = [data]
        constraints = _build_constraints_from_json(data)
    return InferenceSpec(
        target_lab_raw=target,
        target_lab_normalised=tuple(float(x) for x in norm.tolist()),
        constraints=constraints,
        enforce_during=[],   # populated by ConstraintSet.split_during_post()
        enforce_post=[],
        knobs=knobs,
        parsed_disclaimer="parsed from CLI args + constraints JSON",
    )


def _build_constraints_from_json(items: list) -> list:
    from inference.src.constraints import (
        AdjacentForbidden, AllowedSubset, LayerCount, LayerIdentity,
        OrderingBefore, Symmetry, ThicknessRange, TotalThickness,
    )
    kind_map = {
        "allowed_subset": lambda d: AllowedSubset(
            allowed_names=tuple(d["allowed_names"]),
        ),
        "layer_identity": lambda d: LayerIdentity(
            position=int(d["position"]),
            material_name=str(d["material_name"]),
        ),
        "adjacent_forbidden": lambda d: AdjacentForbidden(
            forbidden_pairs=tuple(tuple(p) for p in d["forbidden_pairs"]),
        ),
        "thickness_range": lambda d: ThicknessRange(
            min_nm=int(d.get("min_nm", 5)),
            max_nm=int(d.get("max_nm", 200)),
            position=d.get("position"),
        ),
        "layer_count": lambda d: LayerCount(
            min_layers=int(d.get("min_layers", 1)),
            max_layers=int(d.get("max_layers", 10)),
        ),
        "ordering_before": lambda d: OrderingBefore(
            name_a=str(d["name_a"]), name_b=str(d["name_b"]),
        ),
        "total_thickness": lambda d: TotalThickness(
            max_total_nm=int(d["max_total_nm"]),
            markovian_decode=bool(d.get("markovian_decode", True)),
        ),
        "symmetry": lambda d: Symmetry(
            match_thickness=bool(d.get("match_thickness", True)),
        ),
    }
    out = []
    for item in items:
        kind = item.get("kind")
        if kind not in kind_map:
            raise ValueError(f"unknown constraint kind {kind!r}")
        out.append(kind_map[kind](item))
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="INDIGO inference (cross_attn checkpoint)")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to a saved checkpoint dir, e.g. "
                        "data/checkpoints/<tag>/latest or .../final")
    p.add_argument("--target-lab", nargs=3, required=True, type=float,
                   metavar=("L", "A", "B"),
                   help="Target color in CIE Lab (raw, not normalised)")
    pool_grp = p.add_mutually_exclusive_group(required=True)
    pool_grp.add_argument("--pool", type=str,
                          help="JSON pool file (list of materials with n,k)")
    pool_grp.add_argument("--pool-dir", type=str,
                          help="Directory of JLL CSV files to load as pool")
    p.add_argument("--constraints", type=str, default=None,
                   help="JSON file: list of constraint dicts (see solve.py "
                        "docstring for schema)")
    # Knobs
    p.add_argument("--ensemble-n", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="Manufacturing tolerance (%%) — 0 disables robustness")
    p.add_argument("--lambda", dest="weight_lambda", type=float, default=1.0,
                   help="Robustness weight in J = ΔE + λ·R_l2")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--refine-iters", type=int, default=100)
    p.add_argument("--refine-step", type=float, default=1.0,
                   help="Adam initial step size (nm)")
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-restarts", type=int, default=0,
                   help="Random in-bounds restarts per top-k seed (multi-start)")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: inference/outputs/result_<seed>.json)")
    args = p.parse_args()

    # Load pool.
    if args.pool:
        pool = load_pool_from_json(Path(args.pool))
    else:
        pool = load_pool_from_jll(Path(args.pool_dir))
    print(f"[run] pool: {len(pool)} materials, fingerprint "
          f"{__import__('inference.src.schema', fromlist=['pool_fingerprint']).pool_fingerprint(pool)}")

    # Build spec + knobs.
    knobs = InferenceKnobs(
        ensemble_N=args.ensemble_n,
        temperature=args.temperature,
        tolerance_pct=args.tolerance,
        weight_lambda=args.weight_lambda,
        top_k=args.top_k,
        refine_max_iters=args.refine_iters,
        refine_step_size=args.refine_step,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )
    spec = build_spec(
        target_lab=args.target_lab,
        constraints_path=Path(args.constraints) if args.constraints else None,
        knobs=knobs,
    )
    print(f"[run] target Lab raw={spec.target_lab_raw}  "
          f"normalised={spec.target_lab_normalised}")
    print(f"[run] constraints: {[c.kind for c in spec.constraints] or '(none)'}")

    # Solve.
    result = solve_from_checkpoint(
        checkpoint_dir=Path(args.checkpoint),
        pool=pool, spec=spec,
        n_random_restarts=args.random_restarts,
    )

    # Persist + summary.
    out_path = (Path(args.output) if args.output
                else _root / "inference" / "outputs" / f"result_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(path=out_path)
    print(f"[run] result written to {out_path}")
    _print_summary(result)
    return 0 if result.chosen is not None else 1


def _print_summary(r) -> None:
    s = r.ensemble_stats
    print()
    print("=" * 60)
    if r.chosen is None:
        print("FAILURE")
        for e in r.errors:
            print(f"  {e}")
    else:
        c = r.chosen
        print(f"CHOSEN  ΔE={c.delta_e:.4f}  "
              f"R_l2={c.robustness.grad_l2_shift:.4f}  "
              f"J={c.objective:.4f}  "
              f"refined={c.refined}")
        print(f"        layers ({len(c.slot_indices)}):  "
              + ", ".join(f"{m}@{t:.1f}nm" for m, t in
                          zip(c.material_names, c.thicknesses_nm)))
        print(f"        achieved Lab = "
              f"{tuple(round(x, 2) for x in c.achieved_lab)}")
        if c.robustness.mc_samples > 0:
            print(f"        MC robustness  p50={c.robustness.mc_p50:.4f}  "
                  f"p95={c.robustness.mc_p95:.4f}  "
                  f"worst={c.robustness.mc_worst:.4f}  "
                  f"K={c.robustness.mc_samples}")
        if r.alternatives:
            print(f"        + {len(r.alternatives)} alternatives, "
                  f"top-3 J = "
                  + ", ".join(f"{a.objective:.4f}" for a in r.alternatives[:3]))
    print()
    print(f"Ensemble: N={s.n_sampled} → unique {s.n_unique_after_dedup} → "
          f"feasible {s.n_feasible} → refined {s.n_refined} → "
          f"returned {s.n_returned}")
    if s.dropped_per_constraint:
        print(f"  drops: {s.dropped_per_constraint}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
