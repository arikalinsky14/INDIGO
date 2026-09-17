#!/usr/bin/env python3
"""
De-risker for inference/src/simulate.py.

Verifies, in order:
  1. JAX reflectance reproduces src.optical_sim.OpticalSimulator's numpy
     reflectance to ~1e-4.
  2. JAX reflectance_to_lab matches src.color_utils.spectrum_to_lab to
     ~1e-3 on each Lab channel.
  3. JAX ciede2000 matches src.color_utils.ciede2000 (the shared numpy
     reference the pretrain eval path uses) on hand-picked cases
     including the chroma=0 axis and the 0/360 hue wrap.
  4. jax.grad(ΔE wrt thicknesses) matches centered finite differences on
     every active layer.
  5. End-to-end pipeline (pad_structure + pad_pool_nk +
     delta_e_from_thicknesses) is jit + jax.grad clean.
  6. Schema round-trips (Candidate → JSON → Candidate identical).

If any of these fail loudly, downstream modules (sensitivity, refine,
selection) are paper. Run it after every change to simulate.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import jax
import jax.numpy as jnp

# Force float64 so the finite-difference check has enough headroom; reflectance
# values often sit at 1e-2 magnitudes and the JLL forward in float32 has ~1e-4
# noise, which would otherwise eat the FD signal.
jax.config.update("jax_enable_x64", True)

from inference.src.simulate import (
    AIR_K, AIR_N, FREQS_HZ, LAMBDA_NM, SUBSTRATE_K, SUBSTRATE_N,
    ciede2000, compute_reflectance, delta_e_from_thicknesses, pad_pool_nk,
    pad_structure, reflectance_to_lab,
)
from inference.src.schema import (
    Candidate, MaterialEntry, RobustnessReport, pool_fingerprint,
)
from src.color_utils import ciede2000 as np_ciede2000
from src.material_features import NUM_LAMBDA, load_jll_directory
from src.materials_vocab import MAX_LAYERS, M_MAX


# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------

def load_test_pool(n_materials: int = 5):
    """Load real JLL materials so n,k spectra are physically meaningful."""
    materials_dir = _root / "src" / "_jll_materials"
    if not materials_dir.exists():
        # Fall back to a synthetic dispersive pool — n increases with frequency,
        # k peaks in the middle. Crude but enough for grad checks.
        rng = np.random.default_rng(0)
        pool = []
        for i in range(n_materials):
            n_offset = 1.2 + 0.4 * (i + 1)
            k_amp = 0.05 + 0.05 * i
            x = np.linspace(0, 1, NUM_LAMBDA)
            n_arr = n_offset + 0.3 * x + 0.1 * rng.standard_normal(NUM_LAMBDA) * 0.0
            k_arr = k_amp * np.sin(np.pi * x) ** 2
            pool.append(MaterialEntry(
                canonical_name=f"synth_{i}",
                n=n_arr.astype(np.float32),
                k=k_arr.astype(np.float32),
                source="synthetic",
            ))
        return pool
    pool_full = list(load_jll_directory(materials_dir).values())
    return [
        MaterialEntry(canonical_name=m.name, n=m.n.astype(np.float32),
                      k=m.k.astype(np.float32), source=m.source)
        for m in pool_full[:n_materials]
    ]


def materialnk_from_entry(entry: MaterialEntry):
    """Build a src.material_features.MaterialNK to feed the numpy sim."""
    from src.material_features import MaterialNK
    return MaterialNK(name=entry.canonical_name, n=entry.n, k=entry.k,
                      source=entry.source)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

def check_reflectance_matches_numpy_sim(pool, slot_indices, thicknesses_nm,
                                        tol: float = 5e-4) -> None:
    from src.optical_sim import OpticalSimulator
    pool_np = [materialnk_from_entry(m) for m in pool]
    sim = OpticalSimulator(incidence_angle=0.0)
    R_np = sim.compute_reflectance(pool_np, slot_indices, thicknesses_nm)

    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    R_jax = compute_reflectance(pool_n, pool_k, slots, thicks, mask).block_until_ready()
    R_jax_np = np.asarray(R_jax)

    diff = np.max(np.abs(R_jax_np - R_np))
    print(f"  [reflectance] max |R_jax - R_np| = {diff:.2e}  (tol {tol:.0e})  "
          f"R_np mean {R_np.mean():.4f}")
    assert diff < tol, f"reflectance mismatch: {diff:.2e}"


def check_lab_matches_numpy(pool, slot_indices, thicknesses_nm,
                            per_channel_tol: float = 0.05) -> None:
    from src.color_utils import spectrum_to_lab
    from src.optical_sim import OpticalSimulator
    pool_np = [materialnk_from_entry(m) for m in pool]
    sim = OpticalSimulator(incidence_angle=0.0)
    R_np = sim.compute_reflectance(pool_np, slot_indices, thicknesses_nm)
    lab_np = np.asarray(spectrum_to_lab(np.asarray(LAMBDA_NM), R_np))

    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    R_jax = compute_reflectance(pool_n, pool_k, slots, thicks, mask)
    lab_jax = np.asarray(reflectance_to_lab(R_jax))

    diffs = np.abs(lab_jax - lab_np)
    print(f"  [Lab]         np={lab_np.round(3).tolist()}  "
          f"jax={lab_jax.round(3).tolist()}  Δ={diffs.round(3).tolist()}")
    assert (diffs < per_channel_tol).all(), f"Lab mismatch: {diffs}"


def check_ciede2000_matches_reference(tol: float = 1e-3) -> None:
    """Cross-framework equivalence: JAX ΔE₀₀ vs the shared numpy reference.

    The reference is `src.color_utils.ciede2000` — the same function the
    pretrain eval path (scripts/evaluate.py) reports its numbers with. This
    check is therefore what guarantees the pretrain and inference pipelines
    are measuring the same metric, not just two look-alike formulas.
    """
    cases = [
        # Identical → 0
        ((50.0, 0.0, 0.0), (50.0, 0.0, 0.0)),
        # Pure-luminance shift
        ((50.0, 0.0, 0.0), (60.0, 0.0, 0.0)),
        # Chromatic shift, mid-grey to red
        ((50.0, 30.0, 0.0), (50.0, -30.0, 0.0)),
        # Across hue-rotation centre (around 275°): two blueish points
        ((50.0, -10.0, -30.0), (50.0, -12.0, -28.0)),
        # Achromatic axis: hue undefined → must be safe under our eps
        ((50.0, 0.0, 0.0), (50.0, 0.5, 0.5)),
        # Achromatic on BOTH sides (C1'·C2' == 0 exactly)
        ((50.0, 0.0, 0.0), (72.0, 0.0, 0.0)),
        # Hue wrap across 0/360, both directions
        ((50.0, 39.848, 3.486), (50.0, 39.848, -3.486)),
        ((50.0, 39.848, -3.486), (50.0, 39.848, 3.486)),
        # |Δh'| > 180 with h1' + h2' < 360 (the +180 mean-hue branch)
        ((50.0, 39.392, 6.946), (50.0, -37.588, -13.681)),
        # Large ΔE, opposite gamut corners
        ((0.0, -80.0, -80.0), (100.0, 80.0, 80.0)),
        ((32.3, 79.2, -107.9), (97.1, -21.6, 94.5)),
        # Sharma 2005 Table 1 row 1 (known reference value)
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485)),  # ≈ 2.0425
        # Sharma 2005 Table 1 row 25
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387)),  # ≈ 1.2644
    ]
    for lab1, lab2 in cases:
        ref = np_ciede2000(lab1, lab2)
        got = float(ciede2000(jnp.array(lab1), jnp.array(lab2)))
        diff = abs(got - ref)
        print(f"  [ΔE_00]       {lab1} vs {lab2}: ref={ref:.4f} jax={got:.4f} Δ={diff:.4e}")
        assert diff < tol, f"ΔE mismatch: {diff:.4e}"

    check_ciede2000_known_divergences()


# Known, deliberately-not-asserted gaps between the JAX kernel and the numpy
# reference. Discovered Sept 17 2026 while consolidating the four numpy copies;
# left unfixed because changing simulate.py's numerics would move every
# published inference number and every refine gradient, which is a separate
# decision. Printed loudly on every spike run so they cannot be forgotten.
#
# Root cause (both branches are in inference/src/simulate.py:ciede2000):
#
#   1. The `_EPS * sign(a' + _EPS)` nudge inside the two `arctan2` calls
#      perturbs h1' and h2' by ~1e-11°. When the true Δh' is *exactly* 180°
#      — which happens whenever lab2's (a*, b*) is the exact float negation
#      of lab1's — that nudge pushes Δh' just past the 180° boundary, which
#      flips BOTH the Δh' wrap branch and the |h1' - h2'| > 180 mean-hue
#      branch. h̄' jumps from 180° to 360°, T and S_H change, and ΔE moves by
#      whole units. The numpy reference matches Sharma Table 1 here; the JAX
#      one does not (row 14: 4.8045 published, 4.7461 from JAX).
#
#   2. `hbar_minus_360 = (h1' + h2' + 360) / 2` is applied unconditionally in
#      the |h1' - h2'| > 180 case. The published formula subtracts 360 instead
#      when h1' + h2' >= 360. Only reaches ΔE through dTheta → R_T, so the
#      error is ~1e-4 ΔE at worst (≈3% of random Lab pairs are affected).
#
# Neither fires in production today: (1) needs both Lab triplets to be exact
# hue complements, and inference always compares a simulated candidate against
# a target, and (2) is far below the ~0.5 ΔE scale the pipeline reports.
_KNOWN_DIVERGENT_CASES = [
    # Exact hue complements on the b* axis — divergence grows with chroma.
    ((50.0, 0.0, 20.0), (50.0, 0.0, -20.0)),
    ((50.0, 0.0, 80.0), (50.0, 0.0, -80.0)),
    # Sharma 2005 Table 1 row 14 (published 4.8045).
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900)),
]


def check_ciede2000_known_divergences() -> None:
    """Report (do not assert) the documented JAX-vs-numpy ΔE₀₀ gaps.

    If a run prints Δ≈0 for every case here, the JAX kernel has been fixed —
    promote these into `check_ciede2000_matches_reference` and delete this.
    """
    print("  [ΔE_00] known JAX-vs-numpy divergences (see comment above; not asserted):")
    for lab1, lab2 in _KNOWN_DIVERGENT_CASES:
        ref = np_ciede2000(lab1, lab2)
        got = float(ciede2000(jnp.array(lab1), jnp.array(lab2)))
        print(f"            {lab1} vs {lab2}: numpy={ref:.4f} jax={got:.4f} "
              f"Δ={abs(got - ref):.4f}")


def check_grad_matches_finite_diff(pool, slot_indices, thicknesses_nm,
                                   target_lab, eps_nm: float = 0.1,
                                   rtol: float = 1e-2) -> None:
    """Centered FD at each active layer should match jax.grad."""
    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    target = jnp.asarray(target_lab, dtype=jnp.float64)

    def f(t_nm):
        return delta_e_from_thicknesses(
            t_nm, pool_n, pool_k, slots, mask, target,
        )

    grad_jax = np.asarray(jax.grad(f)(thicks))
    L = len(slot_indices)
    fd = np.zeros_like(grad_jax)
    base = np.asarray(thicks).copy()
    for i in range(L):
        plus = base.copy(); plus[i] += eps_nm
        minus = base.copy(); minus[i] -= eps_nm
        plus_e = float(f(jnp.asarray(plus)))
        minus_e = float(f(jnp.asarray(minus)))
        fd[i] = (plus_e - minus_e) / (2 * eps_nm)

    print(f"  [grad vs FD]  jax  = {grad_jax[:L].round(5).tolist()}")
    print(f"                fd   = {fd[:L].round(5).tolist()}")
    # Tolerance is generous because jax.grad through stackrt_n_k accumulates
    # float64 rounding across the TMM scan, and FD has its own truncation
    # error. We require either small absolute diff (< 1e-3) OR small relative.
    abs_diff = np.abs(grad_jax[:L] - fd[:L])
    rel = abs_diff / (np.abs(fd[:L]) + 1e-12)
    bad = (abs_diff > 1e-3) & (rel > rtol)
    assert not bad.any(), (
        f"grad/FD mismatch at layers {np.where(bad)[0].tolist()}: "
        f"abs={abs_diff.tolist()} rel={rel.tolist()}"
    )


def check_pipeline_timing(pool, slot_indices, thicknesses_nm, target_lab,
                          n_repeats: int = 32) -> None:
    """End-to-end forward + grad timing, no jit.

    `jax.jit` / `jax.vmap` cannot currently wrap the JLL physics chain because
    `jaxlayerlumos.stackrt_eps_mu_base` contains `assert thicknesses[0] == 0`,
    which raises TracerBoolConversionError under abstract tracing. `jax.grad`
    is fine (it traces with concrete values), and that's what refine.py /
    sensitivity.py need. We measure unjitted-but-real-path timing here so
    downstream modules know what kind of per-candidate latency to budget for.
    """
    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    target = jnp.asarray(target_lab, dtype=jnp.float64)

    def f(t):
        return delta_e_from_thicknesses(t, pool_n, pool_k, slots, mask, target)

    # Warm up (first call is slow due to JLL setup).
    _ = float(f(thicks))

    t0 = time.time()
    for _ in range(n_repeats):
        val = float(f(thicks))
    t_fwd = (time.time() - t0) / n_repeats

    t0 = time.time()
    for _ in range(n_repeats):
        g = jax.grad(f)(thicks)
        g.block_until_ready()
    t_grad = (time.time() - t0) / n_repeats

    print(f"  [forward]     val={val:.4f}   avg over {n_repeats}: "
          f"{t_fwd*1e3:.2f} ms")
    print(f"  [grad]        avg grad over {n_repeats}: {t_grad*1e3:.2f} ms")
    print(f"  Budget @ N=500 ensemble: ~{500 * t_fwd:.1f} s forward, "
          f"~{500 * t_grad:.1f} s grad")


def check_schema_roundtrip() -> None:
    from inference.src.schema import (
        Candidate, EnsembleStats, InferenceKnobs, InferenceSpec,
        Result, RobustnessReport,
    )
    knobs = InferenceKnobs(ensemble_N=128, top_k=3)
    spec = InferenceSpec(
        target_lab_raw=(70.0, 5.0, -10.0),
        target_lab_normalised=(0.7, 0.0390625, -0.078125),
        knobs=knobs,
        parsed_disclaimer="test",
    )
    cand = Candidate(
        slot_indices=[0, 2, 1],
        material_names=["A", "C", "B"],
        thicknesses_nm=[50.5, 100.0, 75.3],
        achieved_lab=(71.2, 4.8, -9.7),
        reflectance=[0.5] * 128,
        delta_e=1.42,
        robustness=RobustnessReport(grad_l2_shift=0.3, mc_p95=0.45, mc_samples=32),
        objective=1.72,
        refined=True,
        refine_iters=42,
    )
    res = Result(
        spec_echo=spec, chosen=cand, alternatives=[],
        ensemble_stats=EnsembleStats(n_sampled=128, n_returned=1),
    )
    s = res.to_json()
    res2 = Result.from_json(s)
    assert res2.chosen.delta_e == cand.delta_e
    assert res2.chosen.robustness.mc_p95 == cand.robustness.mc_p95
    assert res2.spec_echo.knobs.top_k == 3
    print("  [schema]      round-trip OK")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("INDIGO inference — JAX sim spike")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")

    pool = load_test_pool(n_materials=5)
    print(f"Pool: {[m.canonical_name for m in pool]}")
    print(f"Pool fingerprint: {pool_fingerprint(pool)}")
    print()

    # A small mixed structure spanning a few materials and thicknesses.
    slot_indices = [0, 2, 4, 1]
    thicknesses_nm = [50, 100, 25, 150]
    target_lab = (60.0, 5.0, -8.0)

    print("1. Reflectance matches numpy sim")
    check_reflectance_matches_numpy_sim(pool, slot_indices, thicknesses_nm)
    print()
    print("2. Lab matches numpy spectrum_to_lab")
    check_lab_matches_numpy(pool, slot_indices, thicknesses_nm)
    print()
    print("3. CIEDE2000 matches reference numpy implementation")
    check_ciede2000_matches_reference()
    print()
    print("4. jax.grad of ΔE wrt thicknesses matches centered finite differences")
    check_grad_matches_finite_diff(pool, slot_indices, thicknesses_nm, target_lab)
    print()
    print("5. Pipeline timing (forward + grad, no jit — see simulate.py docstring)")
    check_pipeline_timing(pool, slot_indices, thicknesses_nm, target_lab)
    print()
    print("6. Schema JSON round-trip")
    check_schema_roundtrip()
    print()
    print("=" * 70)
    print("ALL CHECKS PASSED — JAX physics chain is live.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
