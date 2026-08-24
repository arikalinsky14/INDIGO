"""Directed search for high-chroma structures during data generation.

Motivation. Undirected random-structure sampling (the existing
`RandomLayerSimulation.sample_structure()` path) produces training rows
with a chroma distribution shaped by the physics of random thin-film
stacks — heavily biased toward mid-chroma. The consequence, observed in
`inference/scripts/gamut_eval.py`'s `chromatic_corners` battery on
production checkpoints: at very high chroma (|a|, |b| ≈ 80) the model
falls short of the target by ΔE 5–15 because the training distribution
has too few examples pushing that region.

This module fixes that by adding a *directed* alternate structure
sampler that: (a) picks a saturated Lab target first, then (b) runs a
two-stage search (coarse random-candidate + fine gradient refine) that
lands on a structure whose achieved Lab is close to that target. About
20% of rows will go through this path — the remainder stays with the
existing undirected sampler — so the training distribution is broadened
without losing the diversity the random path provides.

The public contract matches `RandomLayerSimulation.sample_structure()`:

    layer_materials, layer_thicknesses, lab = search_structure_for_target(...)

so `compile_datasets.py::build_rows` can gate between the two paths on a
single Bernoulli draw with no downstream code aware which one produced
the row.

Snap-to-grid caveat. Stage 2 optimises thicknesses continuously in nm,
but the model's vocabulary (`src/materials_vocab.THICKNESSES`) is a
100-bin 2 nm grid; a training row's `layer_thicknesses` must be
grid-legal ints or `encode_layer()` raises `ValueError` at
tokenisation. Every path that leaves this module snaps to the grid AND
re-simulates the achieved Lab at the snapped values (so the stored
label matches the exact discrete structure being stored, not the
pre-snap continuum optimum). The pre-snap floats are also returned so
callers can persist them alongside for possible future use with a
continuous-thickness model head.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from create_dataset.src.random_layer import RandomLayerSimulation
from inference.src.simulate import (
    delta_e_from_thicknesses, pad_pool_nk, pad_structure,
)
from src.material_features import MaterialNK
from src.materials_vocab import MAX_LAYERS, MAX_THICKNESS_NM, THICKNESSES
from src.thickness_optimizer import _adam_step, _dog_init, _dog_step, _project


# ============================================================================
# Config
# ============================================================================

@dataclass
class HighChromaTargetConfig:
    """Bounds for the a*b*-circle target sampler."""
    chroma_min: float = 60.0
    chroma_max: float = 110.0
    lightness_min: float = 25.0
    lightness_max: float = 75.0


@dataclass
class HighChromaSearchConfig:
    """Two-stage search hyperparameters.

    See the accompanying spec doc's §5 for the cost model:
    per attempted row ≈ (candidate_count × forward_ms) + (refine_iters
    × forward_grad_ms). Defaults are tuned for a 20%-of-dataset
    composition at production scale.
    """
    candidate_count: int = 24            # stage-1 forward sims
    refine_iters: int = 12               # stage-2 grad steps
    optimizer_name: str = "dog"          # 'dog' | 'adam'
    adam_lr_nm: float = 1.0              # only used when optimizer_name == 'adam'
    # DoG default r_eps scale — see src.thickness_optimizer._dog_init.
    dog_r_eps_scale: float = 1e-3


# ============================================================================
# Target sampler
# ============================================================================

def sample_high_chroma_target_lab(
    rng: np.random.Generator,
    cfg: HighChromaTargetConfig,
) -> Tuple[float, float, float]:
    """Uniform hue on the a*b* circle × uniform chroma × uniform L*.

    The chroma range sits deliberately past what real sRGB display gamut
    can show (some targets will be physically unreachable by any real
    thin-film stack — consistent with the gamut_eval `chromatic_corners`
    battery that intentionally probes |a|, |b| = 80). That's fine: the
    search still lands on the MOST saturated structure it can find in
    that hue direction, which is the training signal we want.
    """
    hue = rng.uniform(0.0, 2.0 * np.pi)
    chroma = rng.uniform(cfg.chroma_min, cfg.chroma_max)
    L = rng.uniform(cfg.lightness_min, cfg.lightness_max)
    return float(L), float(chroma * np.cos(hue)), float(chroma * np.sin(hue))


# ============================================================================
# Snap-to-grid
# ============================================================================

_GRID_NM = np.asarray(THICKNESSES, dtype=np.float64)   # [5, 10, …, 200]


def _snap_to_grid(thickness_nm: float) -> int:
    """Round a continuous nm value to the nearest 5 nm token grid point.

    Clips out-of-range values to the grid endpoints; every returned
    value is guaranteed to be one of `THICKNESSES`, so `encode_layer()`
    accepts it without raising.
    """
    idx = int(np.argmin(np.abs(_GRID_NM - float(thickness_nm))))
    return int(_GRID_NM[idx])


def _snap_vector(t_continuous: np.ndarray) -> List[int]:
    """Snap every element of a continuous nm vector to the token grid."""
    return [_snap_to_grid(float(x)) for x in t_continuous]


# ============================================================================
# Gradient loss + refinement loop
# ============================================================================

def _delta_e_of(t_active: np.ndarray, pool_n_jax, pool_k_jax,
                slots_jax, mask_jax, target_jax,
                incidence_angle: float) -> float:
    """Numeric ΔE_00 at the given (active-layer-only) thickness vector.

    Pads to MAX_LAYERS internally because `delta_e_from_thicknesses`
    operates on the full-size padded tensors. Returns a Python float.
    """
    t_padded = np.zeros(MAX_LAYERS, dtype=np.float64)
    t_padded[: len(t_active)] = t_active
    return float(delta_e_from_thicknesses(
        jnp.asarray(t_padded), pool_n_jax, pool_k_jax,
        slots_jax, mask_jax, target_jax, incidence_angle,
    ))


def _grad_delta_e(t_active: np.ndarray, pool_n_jax, pool_k_jax,
                  slots_jax, mask_jax, target_jax,
                  incidence_angle: float) -> np.ndarray:
    """∂ΔE/∂t restricted to the active layers (length = len(t_active))."""
    t_padded = np.zeros(MAX_LAYERS, dtype=np.float64)
    L = len(t_active)
    t_padded[: L] = t_active

    def f(t):
        return delta_e_from_thicknesses(
            t, pool_n_jax, pool_k_jax, slots_jax, mask_jax, target_jax,
            incidence_angle,
        )

    grad_full = np.asarray(jax.grad(f)(jnp.asarray(t_padded)))
    return grad_full[: L]


def _refine_thicknesses(
    materials: List[MaterialNK],
    thicknesses_nm: List[int],
    target_lab: Tuple[float, float, float],
    incidence_angle: float,
    cfg: HighChromaSearchConfig,
    bounds_nm: Tuple[float, float] = (float(THICKNESSES[0]),
                                       float(MAX_THICKNESS_NM)),
) -> np.ndarray:
    """Projected DoG/Adam descent on continuous thickness toward `target_lab`.

    Returns the refined continuous-nm vector (length = len(materials)).
    The caller is responsible for snapping this to the token grid and
    re-simulating the achieved Lab.
    """
    L = len(materials)
    if L == 0:
        return np.zeros(0, dtype=np.float64)

    pool_n_jax, pool_k_jax = pad_pool_nk(materials)
    slot_indices = list(range(L))
    slots_jax, _thicks_jax, mask_jax = pad_structure(slot_indices, thicknesses_nm)
    target_jax = jnp.asarray(target_lab, dtype=jnp.float64)

    lo, hi = bounds_nm
    mins = np.full(L, lo, dtype=np.float64)
    maxs = np.full(L, hi, dtype=np.float64)

    t = _project(np.asarray(thicknesses_nm, dtype=np.float64), mins, maxs)
    m = np.zeros_like(t)
    v = np.zeros_like(t)
    dog_state = _dog_init(t, r_eps_scale=cfg.dog_r_eps_scale)

    prev_de = float("inf")
    for it in range(1, cfg.refine_iters + 1):
        de = _delta_e_of(t, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                         target_jax, incidence_angle)
        if abs(prev_de - de) < 1e-4:
            break
        g = _grad_delta_e(t, pool_n_jax, pool_k_jax, slots_jax, mask_jax,
                          target_jax, incidence_angle)
        if cfg.optimizer_name.lower() == "adam":
            t, m, v = _adam_step(t, g, m, v, step_idx=it, lr=cfg.adam_lr_nm)
        else:
            t, dog_state = _dog_step(t, g, dog_state)
        t = _project(t, mins, maxs)
        prev_de = de

    return t


# ============================================================================
# Public entry point — same contract as sim.sample_structure()
# ============================================================================

def search_structure_for_target(
    sim: RandomLayerSimulation,
    target_lab: Tuple[float, float, float],
    cfg: HighChromaSearchConfig,
    rng: np.random.Generator,
) -> Tuple[List[MaterialNK], List[int], List[float], List[float]]:
    """Two-stage directed search for a structure whose achieved Lab is
    close to `target_lab`.

    Stage 1: `candidate_count` fully-random (materials + thicknesses)
    candidates drawn from `sim.random_materials_and_thicknesses()` and
    scored on squared Lab distance to the target. Reusing `sim`'s own
    per-layer sampler keeps the material/layer-count distribution
    identical to the random path — only thicknesses get further tuned
    in stage 2.

    Stage 2: projected gradient descent on the winning candidate's
    thicknesses toward the target, using the shared DoG/Adam primitives
    from `src.thickness_optimizer` and the JAX-differentiable
    `delta_e_from_thicknesses` from `inference.src.simulate`.

    Returns
    -------
    materials : list of MaterialNK
    thicknesses_snapped : list of int   — on the 5 nm token grid
    achieved_lab : list of 3 floats     — Lab at the SNAPPED thicknesses
                                           (label matches stored structure)
    thicknesses_raw_nm : list of float  — pre-snap continuous nm; useful
                                           if a future continuous-thickness
                                           model head lands (§4c/§4e of the
                                           integration spec).
    """
    # ---- Stage 1: coarse random search --------------------------------
    best_materials: Optional[List[MaterialNK]] = None
    best_thick: Optional[List[int]] = None
    best_d2 = float("inf")
    target = np.asarray(target_lab, dtype=np.float64)
    for _ in range(cfg.candidate_count):
        materials, thicks = sim.random_materials_and_thicknesses()
        lab = np.asarray(sim.compute_lab(materials, thicks), dtype=np.float64)
        d2 = float(np.sum((lab - target) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_materials = materials
            best_thick = thicks

    assert best_materials is not None and best_thick is not None, (
        "stage 1 produced no candidates — candidate_count must be > 0"
    )

    # ---- Stage 2: gradient refinement of the winner's thicknesses ------
    refined_nm = _refine_thicknesses(
        best_materials, best_thick, target_lab,
        sim.incidence_angle, cfg,
    )

    # ---- Snap back to the token grid + re-simulate achieved Lab -------
    thicknesses_snapped = _snap_vector(refined_nm)
    achieved_lab = sim.compute_lab(best_materials, thicknesses_snapped)

    return (
        best_materials,
        thicknesses_snapped,
        [float(c) for c in achieved_lab],
        [float(x) for x in refined_nm.tolist()],
    )


# ============================================================================
# Smoke test — assert every returned thickness is on the token grid.
# Reachable via `python -m create_dataset.src.high_chroma_search`.
# ============================================================================

def _smoke_test() -> None:
    from create_dataset.src.random_layer import LayerCountConfig
    from src.material_features import load_jll_material_directory

    # Load enough real materials to have something to sample from.
    real_dir = None
    for cand in [
        Path("./jaxlayerlumos/materials"),
        _repo_root / "jaxlayerlumos" / "materials",
    ]:
        if cand.exists():
            real_dir = cand
            break
    if real_dir is None:
        try:
            import jaxlayerlumos
            real_dir = Path(jaxlayerlumos.__file__).parent / "materials"
        except ImportError:
            print("[smoke] JLL materials dir not found; skipping.")
            return
    reals = load_jll_material_directory(real_dir)[:8]

    sim = RandomLayerSimulation(
        held_in_real=reals,
        layer_count=LayerCountConfig(lam=4.5, min_layers=2, max_layers=6),
        incidence_angle=0,
        p_real=0.5,
        synthetic_weights=(0.25, 0.25, 0.25, 0.25),
        seed=42,
    )
    rng = np.random.default_rng(0)
    target = sample_high_chroma_target_lab(rng, HighChromaTargetConfig())
    print(f"[smoke] target Lab = {target}")

    cfg = HighChromaSearchConfig(candidate_count=8, refine_iters=6)
    materials, snapped, achieved, raw = search_structure_for_target(
        sim, target, cfg, rng,
    )

    grid = set(THICKNESSES)
    assert all(t in grid for t in snapped), (
        f"snap-to-grid failed! Got {snapped}, grid is {sorted(grid)}"
    )
    print(f"[smoke] snapped thicknesses on grid: {snapped}")
    print(f"[smoke] achieved Lab = {[round(c, 2) for c in achieved]}")
    print(f"[smoke] pre-snap continuous = {[round(x, 2) for x in raw]}")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
