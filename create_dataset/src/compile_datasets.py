"""
Compile a parquet shard of (RGB, pool, structure) training rows.

Each shard is fully determined by its `shard_id`: the structure RNG, pool
RNG, and grey-accept RNG are all derived from `SeedSequence(shard_id)`,
so re-running with the same arguments yields bit-identical data.

Differences from the original CHROMA-Lite pipeline:
- Layer count per structure is drawn from a truncated Poisson distribution.
- Low-chroma (greyscale) structures are kept with probability
  `greyscale_keep_prob` and otherwise rejected and resampled.
- The text-prompt machinery (Rephraser, procedural_template) and the
  incorrect-prompts branch are gone — INDIGO has no text input.
- Output directories drop the `layers_NN` prefix because layer count is
  now per-row; the new layout is `angle_AA_substrate_CSi/shard_XXXXX.parquet`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make `src.*` importable when invoked directly (e.g. by SLURM workers that
# don't set PYTHONPATH).
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

from src.color_utils import lab_chroma
from src.material_features import MaterialNK, load_jll_directory

from create_dataset.src.pool_sampler import (
    PoolSamplerConfig,
    sample_pool,
    split_jll_real,
)
from create_dataset.src.random_layer import (
    LayerCountConfig,
    RandomLayerSimulation,
)


_DEFAULT_JLL_PATHS = [
    Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials"),
    Path("./jaxlayerlumos/materials"),
]

# Magic constants XOR'd into the structure seed to derive each auxiliary
# RNG seed. Any fixed values work; these are just easy to spot in logs.
# Two new ones for the high-chroma-search path — kept distinct from
# _GREY_RNG_MAGIC so grey rejection stays reproducible even if the
# search RNGs change semantics.
_GREY_RNG_MAGIC = 0xABCDEF
_HIGH_CHROMA_GATE_RNG_MAGIC = 0xCAFE01
_HIGH_CHROMA_TARGET_RNG_MAGIC = 0xCAFE02
_HIGH_CHROMA_SEARCH_RNG_MAGIC = 0xCAFE03


def _find_jll_materials_dir(override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"JLL materials dir not found at {override}")
        return override

    candidates: List[Path] = []

    # Auto-locate the installed jaxlayerlumos package's materials dir. Works
    # in any environment where `pip install jaxlayerlumos` succeeded, so
    # nothing needs to be passed on the CLI in normal use.
    try:
        import jaxlayerlumos
        candidates.append(Path(jaxlayerlumos.__file__).parent / "materials")
    except ImportError:
        pass

    # Fallbacks for source checkouts / dev environments.
    candidates.extend(_DEFAULT_JLL_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a JaxLayerLumos materials directory. "
        f"Tried: {candidates}. Pass --jll-materials-dir to override."
    )


def derive_seeds(shard_id: int) -> Tuple[int, int]:
    """Deterministic (structure_seed, pool_seed) from a shard id."""
    base = np.random.SeedSequence(shard_id)
    structure_seed, pool_seed = base.generate_state(2).tolist()
    return int(structure_seed), int(pool_seed)


def get_output_path(output_dir: Path, incidence_angle: int, shard_id: int) -> Path:
    """Single-deck layout: one directory per (angle, substrate), one parquet per shard."""
    sub_dir = output_dir / f"angle_{incidence_angle:02d}_substrate_CSi"
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir / f"shard_{shard_id:05d}.parquet"


def _get_git_hash() -> Optional[str]:
    """Best-effort current git commit; None if unavailable."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
        return result.decode().strip()
    except Exception:
        return None


def _accept_row(
    lab: List[float],
    rng: np.random.Generator,
    chroma_threshold: float,
    greyscale_keep_prob: float,
) -> Tuple[bool, float]:
    """Reject most low-chroma (greyscale) structures.

    Returns (accept, chroma). If chroma >= threshold the row is always
    accepted; otherwise it is accepted with probability greyscale_keep_prob.
    """
    chroma = lab_chroma(lab)
    if chroma >= chroma_threshold:
        return True, chroma
    return (rng.random() < greyscale_keep_prob), chroma


def build_rows(
    target_rows: int,
    sim: RandomLayerSimulation,
    held_in_real: List[MaterialNK],
    pool_config: PoolSamplerConfig,
    pool_rng: np.random.Generator,
    accept_rng: np.random.Generator,
    structure_seed: int,
    incidence_angle: int,
    greyscale_chroma_threshold: float,
    greyscale_keep_prob: float,
    max_attempts_factor: float = 4.0,
    verbose: bool = False,
    # High-chroma-search path (optional; default off keeps behaviour
    # byte-identical to the pre-feature codebase).
    high_chroma_prob: float = 0.0,
    high_chroma_gate_rng: Optional[np.random.Generator] = None,
    high_chroma_target_rng: Optional[np.random.Generator] = None,
    high_chroma_search_rng: Optional[np.random.Generator] = None,
    high_chroma_target_cfg: Optional["HighChromaTargetConfig"] = None,
    high_chroma_search_cfg: Optional["HighChromaSearchConfig"] = None,
) -> Tuple[List[dict], Dict[str, float]]:
    """Sample structures until `target_rows` rows have been accepted.

    The grey-filter accept-rng must be independent of the simulation RNG
    so structure samples are not skipped due to coupled randomness.

    When `high_chroma_prob > 0`, each attempt flips a coin; on heads the
    row's structure is produced by the directed high-chroma-search path
    (see create_dataset/src/high_chroma_search.py) instead of the
    default undirected `sim.sample_structure()`. The greyscale filter
    still runs — but a search-path row targets `chroma_min ≥ 60` by
    construction, so it clears the default threshold of 8 on essentially
    the first try and doesn't need any special-case handling downstream.
    """
    rows: List[dict] = []
    n_attempts = 0
    n_grey_rejected = 0
    n_high_chroma_attempted = 0
    n_high_chroma_accepted = 0
    max_attempts = int(target_rows * max_attempts_factor)

    # Late import so the (heavy) JAX imports in high_chroma_search only
    # load when someone actually asks for high_chroma_prob > 0.
    if high_chroma_prob > 0.0:
        from create_dataset.src.high_chroma_search import (
            HighChromaSearchConfig, HighChromaTargetConfig,
            sample_high_chroma_target_lab, search_structure_for_target,
        )
        assert high_chroma_gate_rng is not None
        assert high_chroma_target_rng is not None
        assert high_chroma_search_rng is not None
        target_cfg = high_chroma_target_cfg or HighChromaTargetConfig()
        search_cfg = high_chroma_search_cfg or HighChromaSearchConfig()

    while len(rows) < target_rows and n_attempts < max_attempts:
        n_attempts += 1

        # Route this attempt through the search path with probability
        # high_chroma_prob; else the existing undirected sampler.
        use_search = (
            high_chroma_prob > 0.0
            and high_chroma_gate_rng.random() < high_chroma_prob
        )
        if use_search:
            n_high_chroma_attempted += 1
            target_lab = sample_high_chroma_target_lab(
                high_chroma_target_rng, target_cfg,
            )
            (layer_materials, layer_thicknesses, lab,
             thicknesses_raw_nm) = search_structure_for_target(
                sim, target_lab, search_cfg, high_chroma_search_rng,
            )
            structure_source = "high_chroma_search"
        else:
            layer_materials, layer_thicknesses, lab = sim.sample_structure()
            thicknesses_raw_nm = None
            target_lab = None
            structure_source = "random"

        accept, chroma = _accept_row(
            lab, accept_rng, greyscale_chroma_threshold, greyscale_keep_prob
        )
        if not accept:
            n_grey_rejected += 1
            continue

        pool, structure_slot_indices = sample_pool(
            structure_materials_required=layer_materials,
            held_in_real=held_in_real,
            rng=pool_rng,
            config=pool_config,
        )

        row = {
            "lab": json.dumps([float(c) for c in lab]),
            "pool_size": len(pool),
            "pool_n": [m.n.tolist() for m in pool],
            "pool_k": [m.k.tolist() for m in pool],
            "pool_names": [m.name for m in pool],
            "pool_sources": [m.source for m in pool],
            "layer_slots": [int(s) for s in structure_slot_indices],
            "layer_thicknesses": [int(t) for t in layer_thicknesses],
            "num_layers": len(layer_materials),
            "structure_seed": structure_seed,
            "incidence_angle": incidence_angle,
            "target_chroma": float(chroma),
            # Additive, backward-compat schema columns:
            "structure_source": structure_source,
            "layer_thicknesses_raw_nm": (
                [float(x) for x in thicknesses_raw_nm]
                if thicknesses_raw_nm is not None else None
            ),
            "search_target_lab": (
                [float(x) for x in target_lab]
                if target_lab is not None else None
            ),
        }
        rows.append(row)
        if use_search:
            n_high_chroma_accepted += 1

        if verbose and len(rows) % 500 == 0:
            print(
                f"[INFO] {len(rows)}/{target_rows} rows "
                f"({n_grey_rejected} grey rejected, {n_attempts} attempts, "
                f"{n_high_chroma_accepted} high-chroma accepted / "
                f"{n_high_chroma_attempted} attempted)"
            )

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Hit max_attempts={max_attempts} before reaching target_rows={target_rows}. "
            f"Got {len(rows)} rows. Increase max_attempts_factor or check grey rejection rate."
        )

    rejection_stats = {
        "n_attempts": n_attempts,
        "n_grey_rejected": n_grey_rejected,
        "grey_rejection_rate": n_grey_rejected / max(n_attempts, 1),
        "n_high_chroma_attempted": n_high_chroma_attempted,
        "n_high_chroma_accepted": n_high_chroma_accepted,
        "high_chroma_acceptance_rate": (
            n_high_chroma_accepted / max(n_high_chroma_attempted, 1)
        ),
    }
    return rows, rejection_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-rows", type=int, required=True,
                        help="Number of rows to accept into the shard")
    parser.add_argument("--shard-id", type=int, required=True,
                        help="Integer shard identifier; seeds derived from this")
    parser.add_argument("--incidence-angle", type=int, required=True,
                        help="Incidence angle in degrees")

    # Layer-count distribution.
    parser.add_argument("--layer-lambda", type=float, default=4.5,
                        help="Poisson λ for layer count (default: 4.5)")
    parser.add_argument("--layer-min", type=int, default=2)
    parser.add_argument("--layer-max", type=int, default=10)

    # Greyscale filter.
    parser.add_argument("--greyscale-threshold", type=float, default=8.0,
                        help="Lab chroma C* below which a row is greyscale-filtered")
    parser.add_argument("--greyscale-keep-prob", type=float, default=0.2,
                        help="Probability of keeping a greyscale row")

    # High-chroma-search path (opt-in; --high-chroma-prob 0 keeps behaviour
    # byte-identical to the pre-feature codebase). Target production value
    # is ~0.2 per §3 of the integration spec — ~20% of rows go through the
    # directed two-stage search targeting a saturated Lab colour.
    parser.add_argument("--high-chroma-prob", type=float, default=0.0,
                        help="Bernoulli gate: probability this row is "
                             "generated by the directed high-chroma search "
                             "instead of the undirected random sampler.")
    parser.add_argument("--high-chroma-candidate-count", type=int, default=24,
                        help="Stage-1: number of random candidates scored "
                             "against the saturated Lab target.")
    parser.add_argument("--high-chroma-refine-iters", type=int, default=12,
                        help="Stage-2: projected-gradient iterations on "
                             "the winning candidate's thicknesses.")
    parser.add_argument("--high-chroma-optimizer",
                        choices=("dog", "adam"), default="dog",
                        help="Stage-2 optimizer. 'dog' is parameter-free "
                             "(matches inference/src/refine.py's default).")
    parser.add_argument("--high-chroma-chroma-min", type=float, default=60.0)
    parser.add_argument("--high-chroma-chroma-max", type=float, default=110.0)
    parser.add_argument("--high-chroma-lightness-min", type=float, default=25.0)
    parser.add_argument("--high-chroma-lightness-max", type=float, default=75.0)

    # Pool sampler / per-layer material sampling.
    parser.add_argument("--pool-size-min", type=int, default=4)
    parser.add_argument("--pool-size-max", type=int, default=32)
    parser.add_argument("--p-real", type=float, default=0.15,
                        help="Per material (structure layer OR distractor): "
                             "probability of pulling from held-in real instead "
                             "of generating a fresh synthetic")
    parser.add_argument("--all-real", action="store_true",
                        help="Fully-real dataset: every structure layer and "
                             "every distractor is a JLL real material, zero "
                             "synthetic. Forces p_real=1.0 (overrides --p-real).")

    # Train/test set choice.
    parser.add_argument("--use-held-out-reals", action="store_true",
                        help="Flip to the held-out real materials (Tier-B test set)")

    # IO.
    parser.add_argument("--output-dir", type=str, default="data/train")
    parser.add_argument("--jll-materials-dir", type=str, default=None)
    parser.add_argument("--max-attempts-factor", type=float, default=4.0,
                        help="Safety cap: total sampling attempts = target_rows × this")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Exit immediately if the target shard parquet already exists")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Early exit if --skip-existing and shard already on disk. Useful for
    # parallel data generation: each worker is then idempotent and resumable.
    output_path_check = (
        Path(args.output_dir)
        / f"angle_{args.incidence_angle:02d}_substrate_CSi"
        / f"shard_{args.shard_id:05d}.parquet"
    )
    if args.skip_existing and output_path_check.exists():
        print(f"[skip] shard {args.shard_id}: {output_path_check} already exists")
        return

    jll_dir = _find_jll_materials_dir(
        Path(args.jll_materials_dir) if args.jll_materials_dir else None
    )
    print(f"[INFO] Loading JLL materials from {jll_dir}")
    real_pool = load_jll_directory(jll_dir)
    active_real, inactive_real = split_jll_real(
        real_pool, use_held_out_reals=args.use_held_out_reals
    )
    print(
        f"[INFO] active real: {len(active_real)} materials, "
        f"inactive real: {len(inactive_real)} materials "
        f"(use_held_out_reals={args.use_held_out_reals})"
    )

    structure_seed, pool_seed = derive_seeds(args.shard_id)
    accept_seed = structure_seed ^ _GREY_RNG_MAGIC
    # Derived from structure_seed the same way accept_seed is, so the
    # search-path RNG streams are reproducible per shard AND independent
    # of the simulation / grey-filter RNGs. Only used when
    # --high-chroma-prob > 0, but derived unconditionally to keep the
    # manifest layout stable.
    hc_gate_seed = structure_seed ^ _HIGH_CHROMA_GATE_RNG_MAGIC
    hc_target_seed = structure_seed ^ _HIGH_CHROMA_TARGET_RNG_MAGIC
    hc_search_seed = structure_seed ^ _HIGH_CHROMA_SEARCH_RNG_MAGIC
    print(
        f"[INFO] shard_id={args.shard_id} → "
        f"structure_seed={structure_seed}, pool_seed={pool_seed}, "
        f"accept_seed={accept_seed}, high_chroma_seeds=(gate={hc_gate_seed}, "
        f"target={hc_target_seed}, search={hc_search_seed})"
    )

    # --all-real forces every material (structure layer + distractor) to be
    # a JLL real. np.random.random() is in [0, 1) so `random() < 1.0` is
    # always true → zero synthetic.
    effective_p_real = 1.0 if args.all_real else args.p_real
    if args.all_real:
        print(f"[INFO] --all-real: forcing p_real=1.0 "
              f"(was --p-real {args.p_real}); dataset will contain zero "
              f"synthetic materials")
        if args.pool_size_max > len(active_real):
            print(
                f"[WARN] pool_size_max={args.pool_size_max} exceeds the "
                f"{len(active_real)} available real materials. Distractors "
                f"are drawn with replacement, so pools will contain repeated "
                f"materials. For distinct-only real pools, pass "
                f"--pool-size-max {len(active_real)} (or lower)."
            )

    layer_count = LayerCountConfig(
        lam=args.layer_lambda, min_layers=args.layer_min, max_layers=args.layer_max,
    )
    pool_config = PoolSamplerConfig(
        pool_size_min=args.pool_size_min,
        pool_size_max=args.pool_size_max,
        p_real=effective_p_real,
    )

    sim = RandomLayerSimulation(
        held_in_real=active_real,
        layer_count=layer_count,
        incidence_angle=args.incidence_angle,
        p_real=effective_p_real,
        synthetic_weights=pool_config.synthetic_weights,
        seed=structure_seed,
    )
    pool_rng = np.random.default_rng(pool_seed)
    accept_rng = np.random.default_rng(accept_seed)
    hc_gate_rng = np.random.default_rng(hc_gate_seed)
    hc_target_rng = np.random.default_rng(hc_target_seed)
    hc_search_rng = np.random.default_rng(hc_search_seed)

    # Build the high-chroma configs from CLI so the search path uses the
    # user-selected bounds. Import kept local to avoid pulling in JAX
    # transitively when --high-chroma-prob=0.
    hc_target_cfg = None
    hc_search_cfg = None
    if args.high_chroma_prob > 0.0:
        from create_dataset.src.high_chroma_search import (
            HighChromaSearchConfig, HighChromaTargetConfig,
        )
        hc_target_cfg = HighChromaTargetConfig(
            chroma_min=args.high_chroma_chroma_min,
            chroma_max=args.high_chroma_chroma_max,
            lightness_min=args.high_chroma_lightness_min,
            lightness_max=args.high_chroma_lightness_max,
        )
        hc_search_cfg = HighChromaSearchConfig(
            candidate_count=args.high_chroma_candidate_count,
            refine_iters=args.high_chroma_refine_iters,
            optimizer_name=args.high_chroma_optimizer,
        )
        print(
            f"[INFO] High-chroma-search enabled: prob={args.high_chroma_prob}, "
            f"candidate_count={hc_search_cfg.candidate_count}, "
            f"refine_iters={hc_search_cfg.refine_iters}, "
            f"optimizer={hc_search_cfg.optimizer_name}, "
            f"C*∈[{hc_target_cfg.chroma_min},{hc_target_cfg.chroma_max}], "
            f"L*∈[{hc_target_cfg.lightness_min},{hc_target_cfg.lightness_max}]"
        )

    print(
        f"[INFO] Generating {args.target_rows} rows "
        f"(λ={args.layer_lambda}, range=[{args.layer_min}, {args.layer_max}], "
        f"angle={args.incidence_angle}, "
        f"grey_threshold={args.greyscale_threshold}, "
        f"grey_keep_prob={args.greyscale_keep_prob})"
    )
    rows, rejection_stats = build_rows(
        target_rows=args.target_rows,
        sim=sim,
        held_in_real=active_real,
        pool_config=pool_config,
        pool_rng=pool_rng,
        accept_rng=accept_rng,
        structure_seed=structure_seed,
        incidence_angle=args.incidence_angle,
        greyscale_chroma_threshold=args.greyscale_threshold,
        greyscale_keep_prob=args.greyscale_keep_prob,
        max_attempts_factor=args.max_attempts_factor,
        verbose=args.verbose,
        high_chroma_prob=args.high_chroma_prob,
        high_chroma_gate_rng=hc_gate_rng,
        high_chroma_target_rng=hc_target_rng,
        high_chroma_search_rng=hc_search_rng,
        high_chroma_target_cfg=hc_target_cfg,
        high_chroma_search_cfg=hc_search_cfg,
    )

    output_path = get_output_path(
        Path(args.output_dir), args.incidence_angle, args.shard_id
    )
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Wrote {len(rows)} rows to {output_path} (pid={os.getpid()})")
    print(f"[INFO] Rejection stats: {rejection_stats}")

    manifest = {
        "shard_id": args.shard_id,
        "incidence_angle": args.incidence_angle,
        "target_rows": args.target_rows,
        "rows_written": len(rows),
        "rejection_stats": rejection_stats,
        "layer_lambda": args.layer_lambda,
        "layer_range": [args.layer_min, args.layer_max],
        "pool_size_range": [args.pool_size_min, args.pool_size_max],
        "p_real": effective_p_real,
        "all_real": bool(args.all_real),
        "synthetic_weights": list(pool_config.synthetic_weights),
        "greyscale_threshold": args.greyscale_threshold,
        "greyscale_keep_prob": args.greyscale_keep_prob,
        "high_chroma_prob": args.high_chroma_prob,
        "high_chroma_candidate_count": args.high_chroma_candidate_count,
        "high_chroma_refine_iters": args.high_chroma_refine_iters,
        "high_chroma_optimizer": args.high_chroma_optimizer,
        "high_chroma_chroma_range": [args.high_chroma_chroma_min,
                                     args.high_chroma_chroma_max],
        "high_chroma_lightness_range": [args.high_chroma_lightness_min,
                                        args.high_chroma_lightness_max],
        "use_held_out_reals": args.use_held_out_reals,
        "structure_seed": structure_seed,
        "pool_seed": pool_seed,
        "accept_seed": accept_seed,
        "high_chroma_gate_seed": hc_gate_seed,
        "high_chroma_target_seed": hc_target_seed,
        "high_chroma_search_seed": hc_search_seed,
        "code_version": _get_git_hash(),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
