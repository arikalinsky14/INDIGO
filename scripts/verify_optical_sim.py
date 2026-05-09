#!/usr/bin/env python3
"""
Cross-verify the new optical simulator against the original CHROMA-Lite
named-material simulator on real JLL materials.

Two paths exist:

1. New `OpticalSimulator` (n,k-direct)  — uses MaterialNK objects sampled
   on the canonical wavelength grid.

2. Original `OpticalSimulator` from CHROMA-Lite (named-material via
   JaxLayerLumos's `get_n_k`).

For real materials these should agree to within a fraction of an sRGB
unit. Anything > 1 unit usually indicates a wavelength-grid or
substrate-stack misalignment that must be fixed before committing the
new dataset format.

If the original simulator isn't importable (fresh repo without the
chroma-lite source on PYTHONPATH), the script falls back to:
- 100nm Ag mirror should be near [250, 250, 245] (white-ish reflective).
- Empty stack should be [0, 0, 0].
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.material_features import MaterialNK, load_jll_directory
from src.optical_sim import OpticalSimulator, is_available


def _try_import_original_sim():
    """Import the named-material OpticalSimulator from a chroma-lite checkout.

    Returns (cls, available, source_path or None).

    Loaded by file path via importlib so the import doesn't clash with our
    own `src.optical_sim` (which is already in sys.modules).
    """
    import importlib.util

    candidates = [
        Path("/tmp/sources/CHROMA-lite/src/optical_sim.py"),
        Path.home() / "CHROMA-lite" / "src" / "optical_sim.py",
        Path.home() / "chroma-lite" / "src" / "optical_sim.py",
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_chroma_lite_optical_sim", c)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "OpticalSimulator"), True, c
        except Exception as exc:
            print(f"[WARN] Could not load original simulator from {c}: {exc}")
            continue
    return None, False, None


def _representative_real(real_pool) -> List[Tuple[List[MaterialNK], List[int]]]:
    """A few small structures using common real materials."""
    out = []
    if "Ag" in real_pool:
        out.append(([real_pool["Ag"]], [100]))                                 # 100nm Ag mirror
    if "SiO2" in real_pool:
        out.append(([real_pool["SiO2"]], [200]))                               # 200nm SiO2 dielectric
    if all(name in real_pool for name in ("SiO2", "Ag", "TiO2")):
        out.append((
            [real_pool["SiO2"], real_pool["Ag"], real_pool["TiO2"]],
            [100, 30, 75],
        ))
    if all(name in real_pool for name in ("Si3N4", "Al")):
        out.append(([real_pool["Si3N4"], real_pool["Al"]], [80, 50]))
    return out


def _max_channel_delta(rgb_a, rgb_b) -> float:
    return max(abs(int(a) - int(b)) for a, b in zip(rgb_a, rgb_b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jll-materials-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="sRGB delta threshold above which we consider misaligned")
    args = parser.parse_args()

    if not is_available():
        print("[ERROR] jaxlayerlumos not available; cannot verify optical sim.")
        sys.exit(1)

    candidates = []
    if args.jll_materials_dir:
        candidates.append(Path(args.jll_materials_dir))
    candidates += [
        Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials"),
        Path("./jaxlayerlumos/materials"),
    ]
    materials_dir = next((c for c in candidates if c.exists()), None)
    if materials_dir is None:
        print(f"[ERROR] No JLL materials directory found in {candidates}")
        sys.exit(1)
    print(f"[INFO] Loading real materials from {materials_dir}")
    real_pool = load_jll_directory(materials_dir)

    new_sim = OpticalSimulator(incidence_angle=0)

    # Empty stack sanity check.
    empty_rgb = new_sim.compute_color(pool=[], slot_indices=[], thicknesses_nm=[])
    print(f"[INFO] Empty stack sRGB: {empty_rgb}  (expected [0, 0, 0])")
    assert empty_rgb == [0, 0, 0], f"Empty stack should be [0,0,0], got {empty_rgb}"

    # 100nm Ag mirror reference.
    if "Ag" in real_pool:
        ag_rgb = new_sim.compute_color(pool=[real_pool["Ag"]], slot_indices=[0], thicknesses_nm=[100])
        print(f"[INFO] 100nm Ag mirror sRGB: {ag_rgb}  (expected near [251, 249, 245])")
        if _max_channel_delta(ag_rgb, [251, 249, 245]) > 5:
            print(f"[WARN] 100nm Ag mirror sRGB differs noticeably from reference — investigate")

    # Cross-check against original CHROMA-Lite simulator if available.
    OriginalSim, available, src_path = _try_import_original_sim()
    if not available:
        print("[INFO] Original CHROMA-Lite simulator not importable; "
              "falling back to standalone reference checks only.")
        return
    print(f"[INFO] Cross-checking against original simulator at {src_path}")

    original = OriginalSim(incidence_angle=0)  # type: ignore[misc]

    structures = _representative_real(real_pool)
    if not structures:
        print("[WARN] No representative materials available for cross-check.")
        return

    max_delta_seen = 0
    for materials, thicknesses in structures:
        new_rgb = new_sim.compute_color(
            pool=materials,
            slot_indices=list(range(len(materials))),
            thicknesses_nm=thicknesses,
        )
        original_rgb = original.compute_color(
            [m.name for m in materials],
            thicknesses,
        )
        delta = _max_channel_delta(new_rgb, original_rgb)
        max_delta_seen = max(max_delta_seen, delta)
        labels = " / ".join(f"{m.name}({t}nm)" for m, t in zip(materials, thicknesses))
        marker = " OK" if delta <= args.threshold else " MISMATCH"
        print(f"  {labels}: new={new_rgb}, original={original_rgb}, "
              f"max_channel_delta={delta}{marker}")

    print(f"\n[INFO] Worst per-channel delta across all checks: {max_delta_seen}")
    if max_delta_seen > args.threshold:
        print(f"[ERROR] Worst delta exceeds threshold {args.threshold}; "
              f"investigate canonical wavelength grid alignment.")
        sys.exit(1)
    print("[INFO] All checks within threshold.")


if __name__ == "__main__":
    main()
