#!/usr/bin/env python3
"""NEUTRON benchmark (Wang & Guo, iScience 25, 104339, 2022), on INDIGO.

Their repository: github.com/hammer-wang/neutron (cloned read-only; nothing of
theirs is imported, run or vendored here). What this file takes from it:

  materials  10 dielectrics + 10 metals (D, M below), n,k from optical_data/*.csv
  physics    air / 5 layers / glass n = 1.5, 5 um, incoherent / air; normal
             incidence; R = (R_s + R_p) / 2; 400 to 700 nm in 10 nm steps;
             D65, CIE 1931 2 deg, colour-science sd_to_XYZ then XYZ_to_Lab;
             n,k interpolated quadratically (scipy interp1d) onto that grid
  metric     DeltaE00 <= 2 counts as a success ("acc")
  targets    pictures: photo thumbnailed to 512 px, RGB quantized in steps of
             10, the unique colours, Lab = XYZ_to_Lab(sRGB_to_XYZ(RGB / 255));
             Cr replacement: the colour of 5 x 20 nm Cr in their physics;
             test split: test.pkl from their Mendeley dataset (not reachable
             from every network; read when given)

Their task differs from ours, and every report says so: NEUTRON is given the
material of each of its 5 layers (dielectric / absorber / absorber /
dielectric / 100 nm back metal) and predicts 4 thicknesses within fixed bounds.
INDIGO is given the 20 materials as a pool and chooses materials, order and
thicknesses, here with exactly 5 layers. INDIGO's freedom is larger, so a
per-target comparison is of systems, not of optimizers on one problem.

Subcommands:

  tmm-check   our simulator against their physics on 5 structures, level by
              level (n,k interpolation, TMM, geometry, colour pipeline)
  targets     picture and Cr targets (and test.pkl's, if given) -> JSON
  testset     targets JSON -> a parquet test folder in INDIGO's schema
              (pool = the 20 NEUTRON materials), for test_eval.py
  rescore     re-simulate INDIGO's chosen stacks in their physics
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.material_features import (  # noqa: E402
    CANONICAL_LAMBDA_NM, NUM_LAMBDA, MaterialNK, _interp_to_canonical)

D = ['TiO2', 'SiO2', 'MgF2', 'HfO2', 'Al2O3', 'ZnO', 'ZnS', 'ZnSe', 'Ta2O5', 'Fe2O3']
M = ['Ag', 'Al', 'Ni', 'Ge', 'Ti', 'Zn', 'Cr', 'Cu', 'Au', 'W']
MATERIALS = D + M
THEIR_WL_NM = np.arange(400, 701, 10)          # their np.arange(0.4, 0.701, 0.01) um
GLASS_N, GLASS_NM = 1.5, 5000.0                 # the value their PSO and notebook use
# Their task: dielectric / absorber / absorber / dielectric / back metal, with
# these thickness bounds (paper p. 4-5; GlobalBestPSO bounds in their code).
# INDIGO's thickness grid stops at 200 nm, so its dielectrics search 5 to 200,
# a subset of theirs: that can only make INDIGO's numbers worse.
THEIR_BOUNDS = [(5, 250), (5, 15), (5, 15), (5, 250), (100, 100)]
CR_DESIGN = (['HfO2', 'W', 'Ge', 'SiO2', 'Zn'], [105.0, 15.0, 10.0, 83.0, 100.0])
# What the paper reports (Wang & Guo 2022; results/neutron/neutron_extraction.md).
# Table 1: their five chrome designs, Cr excluded from the materials, all
# below DeltaE00 2 after PSO; design I is CR_DESIGN at 0.8. (The notebook's
# 2.35 compares the target with an 8-bit sRGB triple, not a design.)
REPORTED_CR_DE = {"I": 0.8, "II": 1.2, "III": 1.4, "IV": 1.6, "V": 1.6}
# Table 1's material combinations, top to bottom (back metal 100 nm).
TABLE1_COMBOS = {"I": ["HfO2", "W", "Ge", "SiO2", "Zn"],
                 "II": ["HfO2", "Zn", "Ge", "Al2O3", "Ni"],
                 "III": ["SiO2", "Ag", "Ge", "SiO2", "Au"],
                 "IV": ["HfO2", "Zn", "Ge", "Al2O3", "Au"],
                 "V": ["HfO2", "Zn", "Ge", "MgF2", "Au"]}
CR_TARGET_LAB = [84.96137356, -0.24662783, -3.42720451]    # notebook cell 48
# Table 2: DeltaE00 averaged over each image (pixel-weighted), and the PSO
# stage's time in seconds. One material combination per picture, chosen by
# their classifier. "Street" is not identified among the repo's photos.
PICTURES = {"the_white_orchard": "White Orchard", "tulip_fields": "Tulip Field",
            "great_wall": "Great Wall", "flower_guo_group": "Flower (style-transferred)"}
REPORTED_PICTURES = {
    "White Orchard": {"materials": "Al2O3/Ge/Al/Al2O3/Zn", "mdn_only": 18.55,
                      "pso_random_init": 5.16, "neutron": 3.76, "time_s": 9917},
    "Tulip Field": {"materials": "Ta2O5/Au/Ti/Al2O3/Au", "mdn_only": 19.45,
                    "pso_random_init": 4.34, "neutron": 3.48, "time_s": 9246},
    "Great Wall": {"materials": "HfO2/Au/Ti/SiO2/Au", "mdn_only": 21.46,
                   "pso_random_init": 6.45, "neutron": 5.69, "time_s": 6835},
    "Flower (style-transferred)": {"materials": "Ta2O5/Au/Ti/Al2O3/Au", "mdn_only": 19.37,
                                   "pso_random_init": 5.00, "neutron": 3.93, "time_s": 11249},
    "Street (style-transferred)": {"materials": "ZnO/Ti/Au/MgF2/Au", "mdn_only": 21.60,
                                   "pso_random_init": 4.45, "neutron": 3.28, "time_s": 7349},
}


def _csv(repo: Path, name: str):
    import pandas as pd
    d = pd.read_csv(repo / "optical_data" / f"{name}.csv").dropna()
    return d["wl"].to_numpy(float), d["n"].to_numpy(float), d["k"].to_numpy(float)


# --------------------------------------------------------------------------
# Their physics, written from their get_lab (same libraries: tmm, colour)
# --------------------------------------------------------------------------

def their_nk(repo: Path, name: str) -> np.ndarray:
    """Complex index on THEIR_WL_NM, quadratic interpolation as in load_nk."""
    from scipy.interpolate import interp1d
    wl, n, k = _csv(repo, name)
    return interp1d(wl, n + 1j * k, kind="quadratic")(THEIR_WL_NM / 1000.0)


def their_reflectance(nk_layers: Sequence[np.ndarray], thick_nm: Sequence[float],
                      wl_nm: np.ndarray = THEIR_WL_NM, template: bool = True,
                      glass_n=GLASS_N) -> np.ndarray:
    """R(lambda) in their geometry. template=True reproduces their 5-layer
    inc_list exactly (the 5th layer, the 100 nm back metal, incoherent);
    template=False treats every layer coherently (for stacks of our own)."""
    from tmm import inc_tmm
    L = len(nk_layers)
    if template:
        assert L == 5, "their template has 5 layers"
        inc = ['i', 'c', 'c', 'c', 'c', 'i', 'i', 'i']
    else:
        inc = ['i'] + ['c'] * L + ['i', 'i']
    d = [np.inf] + list(map(float, thick_nm)) + [GLASS_NM, np.inf]
    R = []
    for i, lam in enumerate(wl_nm):
        g = glass_n[i] if np.ndim(glass_n) else glass_n
        n_list = [1] + [nk[i] for nk in nk_layers] + [g, 1]
        rs = inc_tmm('s', n_list, d, inc, 0, float(lam))['R']
        rp = inc_tmm('p', n_list, d, inc, 0, float(lam))['R']
        R.append((rs + rp) / 2)
    return np.array(R)


def their_lab_from_R(R: np.ndarray, wl_nm: np.ndarray = THEIR_WL_NM) -> np.ndarray:
    import colour
    sd = colour.SpectralDistribution(dict(zip(wl_nm.astype(int), R)))
    xyz = colour.sd_to_XYZ(sd, colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                           colour.SDS_ILLUMINANTS['D65'])
    return np.asarray(colour.XYZ_to_Lab(xyz / 100))


def de2000(a, b) -> float:
    import colour
    return float(colour.delta_E(np.asarray(a), np.asarray(b), method="CIE 2000"))


def rgb_to_lab(rgb255) -> np.ndarray:
    """Their target colour: XYZ_to_Lab(sRGB_to_XYZ(RGB / 255))."""
    import colour
    return np.asarray(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.asarray(rgb255) / 255.0)))


# --------------------------------------------------------------------------
# Ours
# --------------------------------------------------------------------------

def our_materials(repo: Path) -> Dict[str, MaterialNK]:
    """Their n,k files on our 128-point grid, by src/material_features.py's
    own resampling (linear, edge values held)."""
    out = {}
    for name in MATERIALS:
        wl, n, k = _csv(repo, name)
        out[name] = MaterialNK(name=name, n=_interp_to_canonical(wl, n, 1.0),
                               k=_interp_to_canonical(wl, k, 0.0), source="neutron")
    return out


def our_lab(mats: Dict[str, MaterialNK], names: Sequence[str], thick_nm) -> np.ndarray:
    from src.optical_sim import OpticalSimulator
    pool = [mats[n] for n in names]
    return np.asarray(OpticalSimulator().compute_lab(pool, list(range(len(pool))),
                                                     list(map(float, thick_nm))))


def our_reflectance(mats, names, thick_nm) -> np.ndarray:
    from src.optical_sim import OpticalSimulator
    pool = [mats[n] for n in names]
    return OpticalSimulator().compute_reflectance(pool, list(range(len(pool))),
                                                  list(map(float, thick_nm)))


def our_lab_from_R(R: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    from src.color_utils import spectrum_to_lab
    return np.asarray(spectrum_to_lab(np.asarray(wl_nm, float), np.asarray(R, float)))


# --------------------------------------------------------------------------
# tmm-check
# --------------------------------------------------------------------------

def check_structures(seed: int = 0):
    """Their Cr-replacement design, their Cr reference, and 3 random
    structures from their template within their bounds."""
    rng = np.random.default_rng(seed)
    out = [("Cr replacement design (theirs)", *CR_DESIGN),
           ("Cr reference, 5 x 20 nm", ['Cr'] * 5, [20.0] * 5)]
    for i in range(3):
        ds, ms = rng.choice(D, 2), rng.choice(M, 3)
        t = [rng.uniform(5, 250), rng.uniform(5, 15), rng.uniform(5, 15),
             rng.uniform(5, 250), 100.0]
        out.append((f"random template {i}", [ds[0], ms[0], ms[1], ds[1], ms[2]],
                    [float(round(x)) for x in t]))
    return out


def tmm_check(repo: Path, out_dir: Path) -> dict:
    import jax.numpy as jnp
    import scipy.constants as scic
    from tmm import coh_tmm
    from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
    from src.optical_sim import _SUBSTRATE_N

    mats = our_materials(repo)
    # Our n,k at their wavelengths, by our own linear resampling (for the
    # interpolation level and to feed our TMM at their wavelengths).
    ours_at_theirs = {}
    for name in MATERIALS:
        wl, n, k = _csv(repo, name)
        o = np.argsort(wl)
        ours_at_theirs[name] = (np.interp(THEIR_WL_NM, wl[o] * 1000, n[o])
                                + 1j * np.interp(THEIR_WL_NM, wl[o] * 1000, k[o]))
    sub_at_theirs = np.interp(THEIR_WL_NM, CANONICAL_LAMBDA_NM[::-1], _SUBSTRATE_N[::-1])

    rows = []
    for label, names, t in check_structures():
        nk_t = [their_nk(repo, n) for n in names]
        nk_o = [ours_at_theirs[n] for n in names]
        # 1. interpolation: quadratic (theirs) vs linear (ours), max |dn + i dk|
        d_interp = max(float(np.max(np.abs(a - b))) for a, b in zip(nk_t, nk_o))
        # 2. TMM alone: identical inputs (their n,k, all coherent, semi-infinite
        #    glass n = 1.5) through tmm.coh_tmm and through our stackrt_n_k.
        R_tmm = np.array([(coh_tmm('s', [1] + [x[i] for x in nk_t] + [GLASS_N],
                                   [np.inf] + t + [np.inf], 0, float(lam))['R']
                           + coh_tmm('p', [1] + [x[i] for x in nk_t] + [GLASS_N],
                                     [np.inf] + t + [np.inf], 0, float(lam))['R']) / 2
                          for i, lam in enumerate(THEIR_WL_NM)])
        n_c = np.stack([np.ones(len(THEIR_WL_NM))] + nk_t
                       + [np.full(len(THEIR_WL_NM), GLASS_N)], axis=1)
        R_te, _, R_tm, _ = stackrt_n_k(jnp.array(n_c), jnp.array([0] + t + [0]) * scic.nano,
                                       jnp.array(scic.c / (THEIR_WL_NM * 1e-9)),
                                       jnp.array([0.0]))
        R_jll = np.array((R_te[0] + R_tm[0]) / 2)
        d_tmm = float(np.max(np.abs(R_tmm - R_jll)))
        # 3. geometry: theirs (5 um glass + back air, incoherent, 5th layer
        #    incoherent) vs ours (semi-infinite fused silica, all coherent),
        #    same n,k (theirs), same TMM library.
        R_their = their_reflectance(nk_t, t)
        R_ourgeo = np.array([(coh_tmm('s', [1] + [x[i] for x in nk_t] + [sub_at_theirs[i]],
                                      [np.inf] + t + [np.inf], 0, float(lam))['R']
                              + coh_tmm('p', [1] + [x[i] for x in nk_t] + [sub_at_theirs[i]],
                                        [np.inf] + t + [np.inf], 0, float(lam))['R']) / 2
                             for i, lam in enumerate(THEIR_WL_NM)])
        d_geo_R = float(np.max(np.abs(R_their - R_ourgeo)))
        lab_their = their_lab_from_R(R_their)
        d_geo_E = de2000(lab_their, their_lab_from_R(R_ourgeo))
        # 4. colour pipeline: one spectrum (theirs, 400-700 nm), two pipelines.
        d_col_E = de2000(lab_their, our_lab_from_R(R_their, THEIR_WL_NM))
        # 5. end to end: their full pipeline vs our full pipeline (our grid,
        #    our resampling, our geometry, our colour).
        lab_ours = our_lab(mats, names, t)
        R_ours_full = our_reflectance(mats, names, t)
        # Their glass: 5 um (pictures, chrome) vs 50 um (data generator).
        import tmm as _tmm_mod  # noqa: F401
        global GLASS_NM
        g0, GLASS_NM = GLASS_NM, 50000.0
        lab_glass50 = their_lab_from_R(their_reflectance(nk_t, t))
        GLASS_NM = g0
        R_ours_at_theirs = np.interp(THEIR_WL_NM, CANONICAL_LAMBDA_NM[::-1], R_ours_full[::-1])
        rows.append({
            "structure": label, "materials": list(names), "thickness_nm": t,
            "max_abs_dnk_interp": d_interp,
            "max_abs_dR_tmm_only": d_tmm,
            "max_abs_dR_geometry": d_geo_R, "dE_geometry": d_geo_E,
            "dE_colour_pipeline": d_col_E,
            "max_abs_dR_end_to_end": float(np.max(np.abs(R_their - R_ours_at_theirs))),
            "lab_theirs": lab_their.tolist(), "lab_ours": lab_ours.tolist(),
            "dE_end_to_end": de2000(lab_their, lab_ours),
            "dE_glass_5um_vs_50um": de2000(lab_their, lab_glass50),
        })
    keys = ["dE_glass_5um_vs_50um", "max_abs_dnk_interp", "max_abs_dR_tmm_only", "max_abs_dR_geometry",
            "dE_geometry", "dE_colour_pipeline", "max_abs_dR_end_to_end", "dE_end_to_end"]
    res = {"structures": rows, "max": {k: max(r[k] for r in rows) for k in keys},
           "notes": {
               "theirs": "air / layers / glass n=1.5 5 um incoherent / air; 5th layer "
                         "incoherent (their inc_list); 400-700 nm, 10 nm; colour-science "
                         "sd_to_XYZ D65 2 deg; n,k quadratic",
               "ours": "air / layers / fused silica (Sellmeier) semi-infinite, coherent; "
                       "128 points 300-900 nm uniform in frequency; Lab via jaxlayerlumos "
                       "360-830 nm; n,k linear (src/material_features.py)"}}
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / "tmm_check.json", "w"), indent=2)
    lines = ["# TMM check: INDIGO's simulator against NEUTRON's physics", "",
             "| structure | n,k interp max abs | TMM only, max abs dR | geometry dR / dE00 | "
             "colour pipeline dE00 | end to end dR / dE00 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['structure']} | {r['max_abs_dnk_interp']:.3g} | "
                     f"{r['max_abs_dR_tmm_only']:.2e} | {r['max_abs_dR_geometry']:.3g} / "
                     f"{r['dE_geometry']:.2f} | {r['dE_colour_pipeline']:.2f} | "
                     f"{r['max_abs_dR_end_to_end']:.3g} / {r['dE_end_to_end']:.2f} |")
    m = res["max"]
    lines += ["", f"Max over structures: TMM alone {m['max_abs_dR_tmm_only']:.2e} in R; "
              f"geometry {m['dE_geometry']:.2f}, colour pipeline {m['dE_colour_pipeline']:.2f}, "
              f"end to end {m['dE_end_to_end']:.2f} DeltaE00. Their glass at 5 um vs 50 um: "
              f"{m['dE_glass_5um_vs_50um']:.3f}.", ""]
    (out_dir / "tmm_check.md").write_text("\n".join(lines))
    return res


# --------------------------------------------------------------------------
# targets / testset / rescore
# --------------------------------------------------------------------------

def picture_rgbs(path: Path, quantize: str = "floor", counts: bool = False):
    """Their reconstruction script's targets: thumbnail to 512, RGB quantized
    in steps of 10 (the script truncates; the paper's text says nearest),
    unique. With counts, also each colour's pixel count, for their
    image-averaged DeltaE00."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512), Image.LANCZOS)     # Image.ANTIALIAS, renamed in Pillow 10
    rgb = np.array(im).reshape(-1, 3)
    q = (rgb / 10).astype(int) * 10 if quantize == "floor" else np.around(rgb / 10).astype(int) * 10
    return np.unique(q, axis=0, return_counts=counts)


def _row_materials(d: dict, i: int, lib: Dict[str, np.ndarray]) -> Optional[List[str]]:
    """The 5 materials of test row i, top to bottom, or None for a blend."""
    if "mats" in d:
        m = list(d["mats"][i])
        alphas = np.asarray(d["alphas"][i]) if "alphas" in d else np.ones(5)
        if len(m) == 10 and m[:5] == m[5:] and np.allclose(alphas, 1):
            return [str(x) for x in m[:5]]
        if len(m) == 5:
            return [str(x) for x in m]
        return None
    out = []
    for nk in list(d["nk_dict"][i].values()):
        nk = np.asarray(nk)
        hits = [n for n, ref in lib.items()
                if ref.shape == nk.shape and np.max(np.abs(ref - nk)) < 1e-4 * (1 + np.max(np.abs(ref)))]
        if len(hits) != 1:
            return None
        out.append(hits[0])
    return out


def build_targets(repo: Path, out: Path, test_pkl: Optional[Path], quantize: str) -> dict:
    targets = []
    lab_cr = their_lab_from_R(their_reflectance([their_nk(repo, 'Cr')] * 5, [20.0] * 5))
    if de2000(lab_cr, CR_TARGET_LAB) > 0.01:
        sys.exit(f"Cr target {lab_cr} does not reproduce their hard-coded Lab {CR_TARGET_LAB}")
    # Cr is excluded from the materials in their chrome experiment.
    targets.append({"set": "cr", "name": "Cr 5x20nm", "lab": lab_cr.tolist(),
                    "exclude": ["Cr"]})
    # Their task on chrome: each of Table 1's five combinations, materials
    # fixed, thicknesses free within their bounds.
    for k, combo in TABLE1_COMBOS.items():
        targets.append({"set": f"cr_table1:{k}", "name": f"Cr, Table 1 design {k}",
                        "lab": lab_cr.tolist(), "materials": combo,
                        "reported_dE": REPORTED_CR_DE[k]})
    counts = {}
    for pic, label in PICTURES.items():
        p = repo / "photo_data" / f"{pic}.jpg"
        if not p.is_file():
            continue
        rgbs, n_px = picture_rgbs(p, quantize, counts=True)
        counts[pic] = {"label": label, "floor": len(picture_rgbs(p, "floor")),
                       "round": len(picture_rgbs(p, "round")), "pixels": int(n_px.sum())}
        # Their task on pictures: the one combination NEUTRON chose for the
        # whole picture (Table 2), fixed for every colour.
        combo = REPORTED_PICTURES[label]["materials"].split("/")
        for rgb, n in zip(rgbs, n_px):
            targets.append({"set": f"picture:{pic}", "rgb": rgb.tolist(),
                            "lab": rgb_to_lab(rgb).tolist(), "pixels": int(n),
                            "materials": combo})
    if test_pkl:
        import pickle
        d = pickle.load(open(test_pkl, "rb"))
        # Each row carries their data-generation PSO's result on the row's own
        # fixed materials (random-init PSO, 32 particles, <= 100 iterations;
        # about 25% reached DeltaE00 <= 2): the per-row baseline for their task.
        # The 5 materials are read from the row ('mats', primitive rows) or
        # identified from its n,k; a row matching no single material per layer
        # (a blend) gets materials None and is not run.
        lib = {n: their_nk(repo, n) for n in MATERIALS}
        for i, (lab, rgb) in enumerate(zip(d["target_Lab"], d["target_RGB"])):
            targets.append({"set": "testset", "index": i, "rgb": np.asarray(rgb).tolist(),
                            "lab": np.asarray(lab, float).tolist(),
                            "materials": _row_materials(d, i, lib),
                            "label_pso_acc": int(d["acc"][i]) if "acc" in d else None,
                            "label_pso_deltaE": (float(d["deltaE"][i]) if "deltaE" in d
                                                 else None)})
    res = {"quantize": quantize, "picture_counts": counts, "materials": MATERIALS,
           "n_targets": len(targets), "targets": targets}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    return res


def write_testset(repo: Path, targets_json: Path, out_dir: Path,
                  sets: Optional[List[str]], limit: int, seed: int,
                  their_task: bool = False, per_picture: int = 0,
                  back_metal: bool = False) -> int:
    """One parquet shard, INDIGO's schema, one row per target. Default pool:
    the 20 NEUTRON materials (minus a target's exclusions), INDIGO choosing.
    their_task: each row's pool is its own 5 materials, and row_constraints.json
    (for test_eval.py --row-constraints) fixes each layer's material and
    thickness bounds as in NEUTRON's task. The structure columns hold a
    1-layer placeholder: these targets have no ground-truth stack (test_eval
    reports it, nothing uses it).
    back_metal: every free row (materials chosen by INDIGO) must end in their
    geometry's back reflector, layer 5 from their metal list at 100 nm, so the
    stack is opaque and their glass-and-air backside does not enter (run with
    LAYER_COUNT=5). Layers 1 to 4 stay free over the whole pool."""
    import pandas as pd
    t = json.load(open(targets_json))["targets"]
    if sets:
        t = [x for x in t if any(x["set"] == s or x["set"].startswith(s + ":") for s in sets)]
    if per_picture:
        # Each picture: per_picture colours drawn WITH replacement, with
        # probability proportional to pixel count. The plain mean of their
        # DeltaE00 then estimates the image-averaged DeltaE00 (Table 2's
        # metric) without bias; rescore bootstraps its interval.
        rng = np.random.default_rng(seed)
        keep = [x for x in t if not x["set"].startswith("picture:")]
        for pset in sorted({x["set"] for x in t if x["set"].startswith("picture:")}):
            g = [x for x in t if x["set"] == pset]
            w = np.array([x["pixels"] for x in g], float)
            for j in rng.choice(len(g), per_picture, replace=True, p=w / w.sum()):
                keep.append({**g[j], "pps": True})
        t = keep
    n_blend = 0
    if their_task:
        # Rows with materials take their task; the others (a blended test row,
        # or the chrome free-choice row) stay free over the pool.
        n_blend = sum(1 for x in t if x["set"] == "testset" and not x.get("materials"))
        t = [x for x in t if not (x["set"] == "testset" and not x.get("materials"))]
    if limit and len(t) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(len(t), limit, replace=False))
        t = [t[i] for i in idx]
    mats = our_materials(repo)
    rows, row_constraints = [], {}
    for i, x in enumerate(t):
        row_task = their_task and bool(x.get("materials"))
        x["their_task"] = row_task
        x["back_metal"] = back_metal and not row_task
        if row_task:
            names = list(dict.fromkeys(x["materials"]))     # unique, in layer order
            pool = [mats[n] for n in names]
            row_constraints[str(i)] = {"lab": x["lab"], "materials": x["materials"], "constraints":
                [{"kind": "layer_identity", "position": k, "material": m}
                 for k, m in enumerate(x["materials"])]
                + [{"kind": "thickness_range", "position": k, "min_nm": lo, "max_nm": hi}
                   for k, (lo, hi) in enumerate(THEIR_BOUNDS)]}
        else:
            pool = [mats[n] for n in MATERIALS if n not in x.get("exclude", [])]
            cons = []
            if x["back_metal"]:
                cons = [{"kind": "layer_from_set", "position": 4,
                         "materials": [m.name for m in pool if m.name in M]},
                        {"kind": "thickness_range", "position": 4,
                         "min_nm": THEIR_BOUNDS[4][0], "max_nm": THEIR_BOUNDS[4][1]}]
            if their_task or back_metal:
                # Explicit even when free: test_eval refuses a row missing from the file.
                row_constraints[str(i)] = {"lab": x["lab"], "materials": None,
                                           "constraints": cons}
        rows.append({"lab": json.dumps([float(v) for v in x["lab"]]), "pool_size": len(pool),
                     "pool_n": [m.n.tolist() for m in pool], "pool_k": [m.k.tolist() for m in pool],
                     "pool_names": [m.name for m in pool], "pool_sources": [m.source for m in pool],
                     "layer_slots": [0], "layer_thicknesses": [100], "num_layers": 1,
                     "structure_source": "neutron_" + x["set"]})
    # The layout src/dataset.py scans: <tier>/angle_<deg>_substrate_<name>/shard_*.parquet
    shard_dir = out_dir / "angle_0_substrate_neutron"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard_dir / "shard_000000.parquet", index=False)
    json.dump([{"row": i, **x} for i, x in enumerate(t)],
              open(out_dir / "targets_index.json", "w"))
    if their_task or back_metal:
        json.dump({"rows": row_constraints, "blends_skipped": n_blend},
                  open(out_dir / "row_constraints.json", "w"))
        if n_blend:
            print(f"[testset] {n_blend} blended row(s) skipped: their materials are not "
                  "single library materials")
    return len(rows)


def snap_int_nm(mats, names, t, target_lab, bounds=None):
    """Whole-nm thicknesses for a stack (their constraint), chosen without
    their physics: of every floor/ceil combination within bounds, the one
    OUR simulator puts closest to the target, the objective INDIGO's search
    already uses. At most 2^L simulations. Returns the thickness list."""
    import itertools
    lab0 = np.asarray(target_lab, float)
    opts = []
    for i, x in enumerate(t):
        c = sorted({math.floor(x), math.ceil(x)})
        if bounds is not None:
            lo, hi = bounds[i]
            c = [v for v in c if lo <= v <= hi] or [int(min(max(round(x), lo), hi))]
        opts.append(c)
    best, best_de = None, math.inf
    for combo in itertools.product(*opts):
        d = de2000(our_lab(mats, names, combo), lab0)
        if d < best_de:
            best, best_de = list(combo), d
    return best


def rescore(repo: Path, run_dir: Path, out: Path, index: Optional[Path] = None) -> dict:
    """Each chosen stack in their physics (all layers coherent, their glass
    and back air), against the target, beside our own simulator's DeltaE;
    also with thicknesses truncated to whole nm, as they constrain theirs.
    With the test folder's targets_index.json, rows are grouped by target
    set and pictures get their metric: DeltaE00 averaged over the image's
    pixels."""
    nk = {n: their_nk(repo, n) for n in MATERIALS}
    ours = our_materials(repo)
    idx = {e["row"]: e for e in json.load(open(index))} if index else {}
    rows_out = []
    # A row rerun after the sampler's token-3201 crash (fixed Nov 18) replaces
    # the crashed copy; any other duplicate row stops the rescore.
    def _bug(x: dict) -> bool:
        return "out of range [0, 3201)" in str(x.get("error") or "")

    by_idx, replaced = {}, []
    for rp in sorted(glob.glob(str(run_dir / "**" / "rows.json"), recursive=True)):
        for r in json.load(open(rp)):
            i, old = r["row_idx"], by_idx.get(r["row_idx"])
            if old is None:
                by_idx[i] = (rp, r)
            elif _bug(old[1]) and not r.get("error"):
                by_idx[i] = (rp, r)
                replaced.append(i)
            elif _bug(r) and not old[1].get("error"):
                replaced.append(i)
            else:
                sys.exit(f"[rescore] row {i} appears twice: {old[0]} and {rp}")
    for i in sorted(by_idx):
        rp, r = by_idx[i]
        ent = {"row_idx": r["row_idx"], "target_lab": r.get("target_lab"),
               "dE_ours": r.get("delta_e"), "wall_time_s": r.get("wall_time_s"),
               "file": rp}
        meta = idx.get(r["row_idx"], {})
        if r.get("layers"):
            names = [l["material"] for l in r["layers"]]
            t = [l["thickness_nm"] for l in r["layers"]]
            if meta.get("their_task"):
                # Their task: their exact template (5th layer and glass
                # incoherent), thicknesses as is and truncated to whole nm.
                ok = names == meta.get("materials")
                lab = their_lab_from_R(their_reflectance([nk[n] for n in names], t))
                lab_i = their_lab_from_R(their_reflectance([nk[n] for n in names],
                                                           [int(x) for x in t]))
                ts = snap_int_nm(ours, names, t, r["target_lab"], THEIR_BOUNDS)
                lab_s = their_lab_from_R(their_reflectance([nk[n] for n in names], ts))
                ent.update(materials=names, thickness_nm=t, lab_theirs=lab.tolist(),
                           dE_theirs=de2000(lab, r["target_lab"]),
                           dE_theirs_int_nm=de2000(lab_i, r["target_lab"]),
                           thickness_snap_nm=ts,
                           dE_theirs_int_snap=de2000(lab_s, r["target_lab"]),
                           matches_row_materials=ok,
                           within_bounds=all(lo <= x <= hi for x, (lo, hi)
                                             in zip(t, THEIR_BOUNDS)))
            elif meta.get("back_metal") and all(n in nk for n in names):
                # Free materials, their geometry: 5 layers, the last an
                # opaque 100 nm metal, scored with their exact inc_list.
                bnd = [(0, 10 ** 6)] * 4 + [THEIR_BOUNDS[4]]
                lab = their_lab_from_R(their_reflectance([nk[n] for n in names], t))
                lab_i = their_lab_from_R(their_reflectance([nk[n] for n in names],
                                                           [int(x) for x in t]))
                ts = snap_int_nm(ours, names, t, r["target_lab"], bnd)
                lab_s = their_lab_from_R(their_reflectance([nk[n] for n in names], ts))
                ent.update(materials=names, thickness_nm=t, lab_theirs=lab.tolist(),
                           dE_theirs=de2000(lab, r["target_lab"]),
                           dE_theirs_int_nm=de2000(lab_i, r["target_lab"]),
                           thickness_snap_nm=ts,
                           dE_theirs_int_snap=de2000(lab_s, r["target_lab"]),
                           back_metal_ok=(len(names) == 5 and names[4] in M
                                          and abs(t[4] - 100) < 1e-6))
            elif all(n in nk for n in names):
                lab = their_lab_from_R(their_reflectance([nk[n] for n in names], t,
                                                         template=False))
                lab_i = their_lab_from_R(their_reflectance([nk[n] for n in names],
                                                           [int(x) for x in t],
                                                           template=False))
                ts = snap_int_nm(ours, names, t, r["target_lab"])
                lab_s = their_lab_from_R(their_reflectance([nk[n] for n in names], ts,
                                                           template=False))
                ent.update(materials=names, thickness_nm=t, lab_theirs=lab.tolist(),
                           dE_theirs=de2000(lab, r["target_lab"]),
                           dE_theirs_int_nm=de2000(lab_i, r["target_lab"]),
                           thickness_snap_nm=ts,
                           dE_theirs_int_snap=de2000(lab_s, r["target_lab"]))
            else:
                ent["untraceable"] = f"materials not in the NEUTRON set: {names}"
        ent["set"], ent["pixels"] = meta.get("set"), meta.get("pixels")
        ent["pps"], ent["reported_dE"] = meta.get("pps", False), meta.get("reported_dE")
        ent["label_pso_acc"] = meta.get("label_pso_acc")
        rows_out.append(ent)
    by_set = {}
    for sname in sorted({x["set"] for x in rows_out if x["set"]}):
        g = [x for x in rows_out if x["set"] == sname]
        e = {}
        for key in ("dE_theirs", "dE_theirs_int_nm", "dE_theirs_int_snap", "dE_ours"):
            v = np.array([x[key] if x.get(key) is not None else math.inf for x in g])
            e[key] = {"acc_le2": float(np.mean(v <= 2)), "median": float(np.median(v)),
                      "mean": float(np.mean(v))}
            if sname.startswith("picture:") and all(x.get("pps") for x in g):
                # Pixel-proportional sample: the plain mean estimates the
                # image-averaged DeltaE00; 95% bootstrap interval.
                bs = np.random.default_rng(0).choice(v, size=(2000, len(v))).mean(axis=1)
                e[key]["image_mean_estimate"] = {"est": float(np.mean(v)),
                                                 "lo": float(np.percentile(bs, 2.5)),
                                                 "hi": float(np.percentile(bs, 97.5))}
            elif sname.startswith("picture:") and all(x.get("pixels") for x in g):
                w = np.array([x["pixels"] for x in g], float)
                e[key]["pixel_weighted_mean"] = float(np.sum(w * v) / np.sum(w))
        lab_acc = [x["label_pso_acc"] for x in g if x.get("label_pso_acc") is not None]
        if lab_acc:
            e["label_pso_acc_le2"] = float(np.mean(lab_acc))
        if len(lab_acc) == len(g) and g:
            # Row for row against their PSO label, on whole-nm thicknesses.
            mine = np.array([(x.get("dE_theirs_int_snap") if x.get("dE_theirs_int_snap") is not None
                              else math.inf) <= 2 for x in g])
            theirs = np.array(lab_acc, bool)
            e["paired_vs_label"] = {"both": int(np.sum(mine & theirs)),
                                    "indigo_only": int(np.sum(mine & ~theirs)),
                                    "label_only": int(np.sum(~mine & theirs)),
                                    "neither": int(np.sum(~mine & ~theirs))}
            e["n_off_task"] = int(sum(1 for x in g if x.get("matches_row_materials") is False
                                      or x.get("within_bounds") is False))
        bm = [x for x in g if "back_metal_ok" in x]
        if bm:
            e["n_back_metal_violations"] = int(sum(1 for x in bm if not x["back_metal_ok"]))
        if sname.startswith("picture:"):
            lab_ = PICTURES.get(sname.split(":", 1)[1])
            e["reported_table2"] = REPORTED_PICTURES.get(lab_)
        if g[0].get("reported_dE") is not None:
            e["reported_table1_dE"] = g[0]["reported_dE"]
        e["n"] = len(g)
        e["wall_time_s_sum"] = float(sum(x["wall_time_s"] or 0 for x in g))
        by_set[sname] = e
    de_t = np.array([x.get("dE_theirs", math.inf) for x in rows_out])
    de_o = np.array([x["dE_ours"] if x.get("dE_ours") is not None else math.inf
                     for x in rows_out])
    wt = [x["wall_time_s"] for x in rows_out if x.get("wall_time_s") is not None]
    res = {"run": str(run_dir), "n": len(rows_out),
           "acc_le2_their_physics": float(np.mean(de_t <= 2)) if len(de_t) else None,
           "acc_le2_our_physics": float(np.mean(de_o <= 2)) if len(de_o) else None,
           "median_dE_their_physics": float(np.median(de_t)) if len(de_t) else None,
           "median_dE_our_physics": float(np.median(de_o)) if len(de_o) else None,
           "wall_time_s_per_target": {"mean": float(np.mean(wt)) if wt else None,
                                      "median": float(np.median(wt)) if wt else None,
                                      "n": len(wt)},
           "failed_rows_count_as_failures": True,
           "rerun_after_sampler_bug": sorted(replaced), "by_set": by_set,
           "reported": {"cr_table1": REPORTED_CR_DE, "pictures_table2": REPORTED_PICTURES},
           "rows": rows_out}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("tmm-check", "targets", "testset", "rescore"))
    p.add_argument("--neutron-repo", default="../hammer-wang/neutron")
    p.add_argument("--out", default=None)
    p.add_argument("--targets", default="results/neutron/targets.json")
    p.add_argument("--test-pkl", default=None, help="their test.pkl (Mendeley dataset)")
    p.add_argument("--quantize", choices=("floor", "round"), default="floor")
    p.add_argument("--sets", nargs="*", default=None, help="e.g. cr picture testset")
    p.add_argument("--limit", type=int, default=0, help="random subset of targets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-picture", type=int, default=0,
                   help="testset: per picture, this many colours drawn with replacement "
                        "with probability proportional to pixel count")
    p.add_argument("--their-task", action="store_true",
                   help="testset: each row's own 5 materials and their thickness bounds")
    p.add_argument("--back-metal", action="store_true",
                   help="testset: free rows end in their back reflector (layer 5 from their "
                        "metals, 100 nm); use with LAYER_COUNT=5")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--index", default=None,
                   help="targets_index.json from the test folder, for per-set results")
    a = p.parse_args()
    repo = Path(a.neutron_repo)
    if not (repo / "optical_data").is_dir():
        sys.exit(f"no NEUTRON checkout at {repo} (git clone https://github.com/hammer-wang/neutron)")
    if a.command == "tmm-check":
        r = tmm_check(repo, Path(a.out or "results/neutron"))
        print(json.dumps(r["max"], indent=2))
    elif a.command == "targets":
        r = build_targets(repo, Path(a.out or a.targets),
                          Path(a.test_pkl) if a.test_pkl else None, a.quantize)
        print(f"{r['n_targets']} targets; pictures {r['picture_counts']}")
    elif a.command == "testset":
        n = write_testset(repo, Path(a.targets), Path(a.out or "data/neutron_test/tier_a"),
                          a.sets, a.limit, a.seed, their_task=a.their_task,
                          per_picture=a.per_picture, back_metal=a.back_metal)
        print(f"wrote {n} rows")
    else:
        r = rescore(repo, Path(a.run_dir), Path(a.out or "results/neutron/rescore.json"),
                    Path(a.index) if a.index else None)
        print({k: v for k, v in r.items() if k != "rows"})


if __name__ == "__main__":
    main()
