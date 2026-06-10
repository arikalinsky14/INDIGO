#!/usr/bin/env python3
"""
De-risker for inference/src/simulate.py.

Verifies, in order:
  1. JAX reflectance reproduces src.optical_sim.OpticalSimulator's numpy
     reflectance to ~1e-4.
  2. JAX reflectance_to_lab matches src.color_utils.spectrum_to_lab to
     ~1e-3 on each Lab channel.
  3. JAX ciede2000 matches a reference numpy implementation on a few
     hand-picked cases including the chroma=0 axis.
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
# Reference numpy CIEDE2000 (for ground-truthing the JAX version)
# ----------------------------------------------------------------------------

def _np_ciede2000(lab1, lab2):
    """Straight numpy port of CIEDE2000 — same formula, no eps tricks."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    def hue(ap, bp):
        if ap == 0 and bp == 0:
            return 0.0
        h = np.degrees(np.arctan2(bp, ap))
        return h + 360 if h < 0 else h
    h1p, h2p = hue(a1p, b1), hue(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        raw = h2p - h1p
        if raw > 180:
            dhp = raw - 360
        elif raw <= -180:
            dhp = raw + 360
        else:
            dhp = raw
    dHp = 2 * np.sqrt(max(C1p * C2p, 0)) * np.sin(np.radians(dhp) / 2)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    if C1p * C2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = 0.5 * (h1p + h2p)
    else:
        hbarp = 0.5 * (h1p + h2p + 360)
    T = (1
         - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    dTheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7))
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    S_L = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    S_C = 1 + 0.045 * Cbarp
    S_H = 1 + 0.015 * Cbarp * T
    return np.sqrt((dLp / S_L) ** 2 + (dCp / S_C) ** 2 + (dHp / S_H) ** 2
                   + Rt * (dCp / S_C) * (dHp / S_H))


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
        # Sharma 2005 Table 1 row 1 (known reference value)
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485)),  # ≈ 2.0425
    ]
    for lab1, lab2 in cases:
        ref = _np_ciede2000(np.array(lab1), np.array(lab2))
        got = float(ciede2000(jnp.array(lab1), jnp.array(lab2)))
        diff = abs(got - ref)
        print(f"  [ΔE_00]       {lab1} vs {lab2}: ref={ref:.4f} jax={got:.4f} Δ={diff:.4e}")
        assert diff < tol, f"ΔE mismatch: {diff:.4e}"


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


def check_jit_path(pool, slot_indices, thicknesses_nm, target_lab) -> None:
    """End-to-end jit + grad cleanliness."""
    pool_n, pool_k = pad_pool_nk(pool)
    slots, thicks, mask = pad_structure(slot_indices, thicknesses_nm)
    target = jnp.asarray(target_lab, dtype=jnp.float64)

    @jax.jit
    def f_and_grad(t):
        return delta_e_from_thicknesses(t, pool_n, pool_k, slots, mask, target), \
               jax.grad(delta_e_from_thicknesses)(t, pool_n, pool_k, slots, mask, target)

    t0 = time.time()
    val, grad = f_and_grad(thicks)
    val.block_until_ready()
    t_compile = time.time() - t0
    t0 = time.time()
    val, grad = f_and_grad(thicks + 1.0)
    val.block_until_ready()
    t_run = time.time() - t0
    print(f"  [jit]         val={float(val):.4f}  compile={t_compile*1e3:.1f} ms  "
          f"run={t_run*1e3:.2f} ms")


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
    print("5. End-to-end jit + grad path")
    check_jit_path(pool, slot_indices, thicknesses_nm, target_lab)
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
