"""
Material Featurization Module
=============================

Handles refractive-index data (n, k vs wavelength) for a flexible material pool.
Replaces the fixed 25-material vocabulary of the original CHROMA-Lite.

Responsibilities
----------------
1. **Loading**: Read JaxLayerLumos-style CSVs (stacked `wl,n` / `wl,k` sections,
   wavelengths in micrometers) into a canonical (wavelength, n, k) representation.
2. **Resampling**: Interpolate any material's n,k onto a single canonical grid
   shared by the model and the optical simulator (so featurization and physics
   are wavelength-aligned).
3. **Featurization**: Produce a fixed-shape tensor per material that the model
   can consume. Two modes are exposed:
     - `raw_spectrum`: stack n and k on the canonical grid → [2, NUM_LAMBDA].
     - `compact`: subsample on a coarse grid → [2, NUM_COMPACT_LAMBDA].
   The shared material encoder in `model.py` consumes whichever the config
   requests.

Conventions
-----------
- Wavelengths are in **nanometers** internally. JaxLayerLumos uses micrometers
  on disk; we convert on load.
- Canonical grid matches the optical simulator: 128 points uniformly spaced
  **in frequency** between (c/900nm, c/300nm). Wavelengths are therefore NOT
  uniformly spaced — they're denser in the blue. This matches the simulator
  exactly, so a single n,k tensor is reusable for both featurization and
  the transfer-matrix calculation.
- For dielectrics, missing k data (no `wl,k` section in the CSV) is treated
  as k=0 across all wavelengths.

Design notes
------------
The point of this module is to decouple "material identity" from the model.
At training time, the model sees a pool of materials by their n,k features
only — never by name. At inference, the user supplies any materials they
want, the n,k get resampled onto the canonical grid here, and the model
treats them identically to anything it saw in training.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


# ============================================================================
# Canonical wavelength grid — must match src/optical_sim.py
# ============================================================================

# Speed of light, m/s (matches scipy.constants.c, hardcoded so this module
# has no SciPy dependency at import time)
_SPEED_OF_LIGHT_M_S: float = 2.99792458e8

# Visible-plus-margins window for the optical simulator. The simulator uses
# the same range; aligning here lets us reuse n,k tensors directly.
WAVELENGTH_MIN_NM: int = 300
WAVELENGTH_MAX_NM: int = 900
NUM_LAMBDA: int = 128

# Compact featurization (used as an ablation against the full spectrum).
# 16 points covering 380-780nm — the photopic-relevant visible window.
NUM_COMPACT_LAMBDA: int = 16
COMPACT_LAMBDA_MIN_NM: int = 380
COMPACT_LAMBDA_MAX_NM: int = 780


def _build_canonical_frequency_grid() -> np.ndarray:
    """Frequency grid in Hz, uniformly spaced between c/lambda_max and c/lambda_min."""
    f_min = _SPEED_OF_LIGHT_M_S / (WAVELENGTH_MAX_NM * 1e-9)
    f_max = _SPEED_OF_LIGHT_M_S / (WAVELENGTH_MIN_NM * 1e-9)
    return np.linspace(f_min, f_max, NUM_LAMBDA)


def _build_canonical_wavelength_grid_nm() -> np.ndarray:
    """The wavelengths corresponding to the canonical frequency grid, in nm."""
    f = _build_canonical_frequency_grid()
    return _SPEED_OF_LIGHT_M_S / f / 1e-9  # m -> nm


CANONICAL_FREQ_HZ: np.ndarray = _build_canonical_frequency_grid()
CANONICAL_LAMBDA_NM: np.ndarray = _build_canonical_wavelength_grid_nm()

# Indices into CANONICAL_LAMBDA_NM that are nearest to the compact grid.
# Precomputed once to keep featurization branch-free.
_COMPACT_LAMBDA_TARGETS = np.linspace(
    COMPACT_LAMBDA_MIN_NM, COMPACT_LAMBDA_MAX_NM, NUM_COMPACT_LAMBDA
)
COMPACT_INDICES: np.ndarray = np.array(
    [int(np.argmin(np.abs(CANONICAL_LAMBDA_NM - lam))) for lam in _COMPACT_LAMBDA_TARGETS],
    dtype=np.int64,
)


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class MaterialNK:
    """A material represented by its complex refractive index on the canonical grid.

    Attributes
    ----------
    name : str
        Human-readable label (used only for logging / debugging — the model
        never sees this).
    n : np.ndarray
        Real part of the refractive index, shape [NUM_LAMBDA].
    k : np.ndarray
        Imaginary part (extinction coefficient), shape [NUM_LAMBDA]. Zero for
        ideal dielectrics.
    source : str
        Tag indicating origin: 'jaxlayerlumos', 'synthetic', 'user'. Useful
        for stratifying evaluation by data source.
    """

    name: str
    n: np.ndarray
    k: np.ndarray
    source: str = "unknown"

    # Validation toggle. ON by default so synthetic-material generators
    # still get the safety net. The training/eval data loader disables it
    # via `materialnk_validation_disabled()` because parquet rows were
    # already validated at generation time and the per-object numpy
    # reductions are ~9% of the data-pipeline CPU cost (see the pipeline
    # profiling in commit history).
    _VALIDATE: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not MaterialNK._VALIDATE:
            return
        # Defensive validation. These are cheap and catch silent bugs in
        # synthetic-material generators.
        assert self.n.shape == (NUM_LAMBDA,), f"n shape {self.n.shape} != ({NUM_LAMBDA},)"
        assert self.k.shape == (NUM_LAMBDA,), f"k shape {self.k.shape} != ({NUM_LAMBDA},)"
        assert np.all(np.isfinite(self.n)), f"non-finite n in {self.name}"
        assert np.all(np.isfinite(self.k)), f"non-finite k in {self.name}"
        # Physical sanity. n must be positive; k must be non-negative.
        # (We allow n down to ~0.1 because some metals have anomalous n < 1.)
        assert np.all(self.n > 0), f"non-positive n in {self.name}"
        assert np.all(self.k >= 0), f"negative k in {self.name}"


@contextmanager
def materialnk_validation_disabled() -> Iterator[None]:
    """Temporarily skip MaterialNK.__post_init__ validation.

    Use only where the n,k arrays are already known-good (e.g. the
    training-data loader reconstructing pre-validated parquet rows).
    Restores the prior setting on exit, so nesting / generation code is
    unaffected.
    """
    prev = MaterialNK._VALIDATE
    MaterialNK._VALIDATE = False
    try:
        yield
    finally:
        MaterialNK._VALIDATE = prev


# ============================================================================
# Loading from JaxLayerLumos-style CSVs
# ============================================================================


def _parse_jll_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a JaxLayerLumos CSV.

    Supports both the stacked two-column JLL layout (`wl,n` followed by an
    optional `wl,k` block) and a single three-column table (`wl,n,k`).

    Returns
    -------
    wl_n_um : np.ndarray
        Wavelengths in μm where n is sampled.
    n_vals : np.ndarray
        n values at those wavelengths.
    wl_k_um : np.ndarray
        Wavelengths in μm where k is sampled (may be empty).
    k_vals : np.ndarray
        k values (may be empty — caller treats as k=0).
    """
    wl_n: List[float] = []
    val_n: List[float] = []
    wl_k: List[float] = []
    val_k: List[float] = []

    section: Optional[str] = None  # 'n', 'k', 'n_k', or None
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            header = [p.lower() for p in parts]
            if header in (["wl", "n"], ["wavelength", "n"], ["wavelength_um", "n"]):
                section = "n"
                continue
            if header in (["wl", "k"], ["wavelength", "k"], ["wavelength_um", "k"]):
                section = "k"
                continue
            if (
                len(header) == 3
                and header[0] in ("wl", "wavelength", "wavelength_um")
                and header[1:] == ["n", "k"]
            ):
                section = "n_k"
                continue
            # Data row.
            if len(parts) == 3 and section == "n_k":
                try:
                    wl_um = float(parts[0])
                    n_value = float(parts[1])
                    k_value = float(parts[2])
                except ValueError as exc:
                    if any(c.isalpha() for c in line):
                        continue
                    raise ValueError(f"Bad row in {path}: {line!r}") from exc
                wl_n.append(wl_um)
                val_n.append(n_value)
                wl_k.append(wl_um)
                val_k.append(k_value)
                continue
            if len(parts) != 2:
                raise ValueError(f"Bad row in {path}: {line!r}")
            try:
                wl_um = float(parts[0])
                value = float(parts[1])
            except ValueError as exc:
                if any(c.isalpha() for c in line):
                    continue
                raise ValueError(f"Bad row in {path}: {line!r}") from exc
            if section == "n":
                wl_n.append(wl_um)
                val_n.append(value)
            elif section == "k":
                wl_k.append(wl_um)
                val_k.append(value)
            else:
                raise ValueError(f"Data before any section header in {path}")

    return (
        np.array(wl_n, dtype=np.float64),
        np.array(val_n, dtype=np.float64),
        np.array(wl_k, dtype=np.float64),
        np.array(val_k, dtype=np.float64),
    )


def _interp_to_canonical(
    wl_um: np.ndarray, values: np.ndarray, default_value: float
) -> np.ndarray:
    """Interpolate a sparse (wavelength, value) sequence onto CANONICAL_LAMBDA_NM.

    Linear interpolation. Wavelengths outside the source range fall back to
    the default value (0 for k, 1 for n). This is mildly unphysical at the
    edges but is consistent with what JaxLayerLumos does internally
    (it extrapolates without bounds-checking from the user's perspective).

    `wl_um` is converted to nm before interpolation.
    """
    if wl_um.size == 0:
        return np.full(NUM_LAMBDA, default_value, dtype=np.float64)

    # Sort by wavelength (some JaxLayerLumos files are descending).
    order = np.argsort(wl_um)
    wl_nm = wl_um[order] * 1000.0
    values = values[order]

    # np.interp clamps at the edges to the boundary values, which is the
    # behaviour we want for edge wavelengths just outside the data range.
    # However if the source data doesn't cover ANY of the canonical range,
    # we want the default. In practice JaxLayerLumos data is wide enough
    # that this case doesn't fire for known materials, but it's a useful
    # guard for sparse synthetic data.
    if wl_nm.max() < WAVELENGTH_MIN_NM or wl_nm.min() > WAVELENGTH_MAX_NM:
        return np.full(NUM_LAMBDA, default_value, dtype=np.float64)

    return np.interp(CANONICAL_LAMBDA_NM, wl_nm, values)


def load_jll_material(csv_path: Path, name: Optional[str] = None) -> MaterialNK:
    """Load a JaxLayerLumos CSV and resample onto the canonical grid.

    Parameters
    ----------
    csv_path : Path
        Path to a JaxLayerLumos materials CSV.
    name : str, optional
        Override the material name (default: derived from filename).

    Returns
    -------
    MaterialNK
    """
    csv_path = Path(csv_path)
    if name is None:
        # Strip e.g. "Ag-Rakic-LD-1998.csv" → "Ag"
        name = csv_path.stem.split("-")[0]

    wl_n_um, n_raw, wl_k_um, k_raw = _parse_jll_csv(csv_path)
    n = _interp_to_canonical(wl_n_um, n_raw, default_value=1.0)
    k = _interp_to_canonical(wl_k_um, k_raw, default_value=0.0)

    return MaterialNK(name=name, n=n, k=k, source="jaxlayerlumos")


def load_jll_directory(dir_path: Path) -> Dict[str, MaterialNK]:
    """Load all JaxLayerLumos CSVs in a directory into a name-keyed dict.

    Names are derived from the first '-'-separated token of each filename.
    If multiple parameterisations of the same material exist (e.g. Si3N4 has
    both Philipp-1973 and Zarei-2024 sources), only the lexicographically
    first is kept under the bare name. The full path is preserved for the
    others under their full filename stem so callers can disambiguate.
    """
    dir_path = Path(dir_path)
    out: Dict[str, MaterialNK] = {}
    for csv in sorted(dir_path.glob("*.csv")):
        material = load_jll_material(csv)
        # Always store the full disambiguated name.
        out[csv.stem] = MaterialNK(
            name=csv.stem, n=material.n, k=material.k, source="jaxlayerlumos"
        )
        # Keep the bare name for the first-seen parameterisation.
        if material.name not in out:
            out[material.name] = material
    return out


# ============================================================================
# Featurization for the model
# ============================================================================


def featurize(material: MaterialNK, mode: str = "raw_spectrum") -> torch.Tensor:
    """Convert a material's n,k into a fixed-shape feature tensor.

    Parameters
    ----------
    material : MaterialNK
    mode : str
        - 'raw_spectrum': [2, NUM_LAMBDA] = [{n, k}, full canonical grid]
        - 'compact': [2, NUM_COMPACT_LAMBDA] = subsampled on visible band

    Returns
    -------
    torch.Tensor of dtype float32, on CPU. The model's material encoder is
    responsible for moving to device.
    """
    if mode == "raw_spectrum":
        feat = np.stack([material.n, material.k], axis=0)  # [2, NUM_LAMBDA]
    elif mode == "compact":
        feat = np.stack(
            [material.n[COMPACT_INDICES], material.k[COMPACT_INDICES]], axis=0
        )  # [2, NUM_COMPACT_LAMBDA]
    else:
        raise ValueError(f"Unknown featurization mode: {mode!r}")
    return torch.from_numpy(feat).float()


def feature_dim(mode: str = "raw_spectrum") -> int:
    """Flat dimensionality of a featurized material under each mode."""
    if mode == "raw_spectrum":
        return 2 * NUM_LAMBDA
    if mode == "compact":
        return 2 * NUM_COMPACT_LAMBDA
    raise ValueError(f"Unknown featurization mode: {mode!r}")


def featurize_pool(
    pool: List[MaterialNK], mode: str = "raw_spectrum"
) -> torch.Tensor:
    """Featurize a list of materials into a [M, 2, L] tensor."""
    return torch.stack([featurize(m, mode=mode) for m in pool], dim=0)


def pad_pool_features(
    pool_feats: torch.Tensor, m_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a pool feature tensor to size m_max along the slot dim.

    Parameters
    ----------
    pool_feats : torch.Tensor, shape [M, 2, L]
    m_max : int

    Returns
    -------
    padded : torch.Tensor, shape [m_max, 2, L]
        Zeros in the trailing slots.
    mask : torch.Tensor, shape [m_max], dtype=bool
        True for valid slots, False for padding.
    """
    M = pool_feats.size(0)
    if M > m_max:
        raise ValueError(f"Pool size {M} exceeds m_max={m_max}")
    pad_shape = (m_max - M,) + tuple(pool_feats.shape[1:])
    padded = torch.cat(
        [pool_feats, torch.zeros(pad_shape, dtype=pool_feats.dtype)], dim=0
    )
    mask = torch.zeros(m_max, dtype=torch.bool)
    mask[:M] = True
    return padded, mask


# ============================================================================
# Smoke test — run this module directly to validate
# ============================================================================

if __name__ == "__main__":
    import sys

    # Try to find a JaxLayerLumos materials directory next to /home/claude
    # (works for the dev environment; in production the user passes a path).
    candidates = [
        Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials"),
        Path("./jaxlayerlumos/materials"),
    ]
    materials_dir = next((c for c in candidates if c.exists()), None)
    if materials_dir is None:
        print("[smoke] No JaxLayerLumos materials directory found; skipping load test.")
        sys.exit(0)

    print(f"[smoke] Loading materials from {materials_dir}")
    pool = load_jll_directory(materials_dir)
    print(f"[smoke] Loaded {len(pool)} entries")

    # Inspect a metal and a dielectric.
    for name in ["Ag", "SiO2", "TiO2", "Au"]:
        if name not in pool:
            print(f"  - {name}: NOT FOUND")
            continue
        m = pool[name]
        print(
            f"  - {name}: n in [{m.n.min():.3f}, {m.n.max():.3f}], "
            f"k in [{m.k.min():.3f}, {m.k.max():.3f}]"
        )

    # Verify featurization shapes.
    feat = featurize(pool["Ag"], mode="raw_spectrum")
    print(f"[smoke] Ag raw feature shape: {tuple(feat.shape)} (expected (2, {NUM_LAMBDA}))")
    feat_c = featurize(pool["Ag"], mode="compact")
    print(f"[smoke] Ag compact feature shape: {tuple(feat_c.shape)}")

    # Verify pool featurization + padding.
    sample = [pool["Ag"], pool["SiO2"], pool["TiO2"]]
    pf = featurize_pool(sample, mode="raw_spectrum")
    padded, mask = pad_pool_features(pf, m_max=8)
    print(f"[smoke] Padded pool shape: {tuple(padded.shape)}, mask sum: {mask.sum().item()}")
    print("[smoke] OK")
