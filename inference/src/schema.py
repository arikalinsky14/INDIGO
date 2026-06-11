"""
Canonical objects for the INDIGO inference branch.

This module defines the *contract* between every other inference module and
the (eventual) GUI. Everything serialises round-trip to JSON via `to_json`
/ `from_json`.

Design rules:
  - Knobs and constraints are explicit fields, not free-form dicts.
  - Every `Result` carries enough provenance (model checkpoint hash, JAX /
    torch versions, JLL commit, pool fingerprint, seeds) to reproduce the
    decode bit-for-bit. Cheap to add now, brutal to retrofit.
  - Failures (no feasible candidate, parse failures) ride the same envelope
    as success: `chosen=None`, populated `errors`, full `spec_echo`. The
    GUI renders both with one code path.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Material entries (canonical name + n,k spectra)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialEntry:
    """A pool material as the inference branch sees it.

    `canonical_name` is required: the model never reads it, but every
    constraint that says "must include Ag" needs to resolve a name to a
    slot. n, k are on the canonical wavelength grid (NUM_LAMBDA = 128).
    """
    canonical_name: str
    n: np.ndarray  # [NUM_LAMBDA]
    k: np.ndarray  # [NUM_LAMBDA]
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.n.shape != self.k.shape:
            raise ValueError(f"n and k shapes differ: {self.n.shape} vs {self.k.shape}")
        if self.n.ndim != 1:
            raise ValueError(f"n must be 1-D, got shape {self.n.shape}")

    def fingerprint(self) -> str:
        """Stable hash of (canonical_name, n, k). Used in pool fingerprints."""
        h = hashlib.sha256()
        h.update(self.canonical_name.encode("utf-8"))
        h.update(self.n.astype(np.float32).tobytes())
        h.update(self.k.astype(np.float32).tobytes())
        return h.hexdigest()[:16]


def pool_fingerprint(pool: List[MaterialEntry]) -> str:
    """SHA-256 over (sorted) per-material fingerprints. Stable across pool order.

    Two pools with the same materials in different order produce the same
    fingerprint, which is what we want for caching parsed specs and for
    cross-validating two `Result`s claim to use "the same" pool.
    """
    fps = sorted(m.fingerprint() for m in pool)
    h = hashlib.sha256()
    for fp in fps:
        h.update(fp.encode("ascii"))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# Constraints (kept abstract here; concrete subclasses live in constraints.py)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """Base class for the 8 supported constraints.

    Subclasses live in `inference/src/constraints.py` and carry their own
    parameter fields. Two methods that subclasses must implement:
      check(structure, pool) -> bool      # post-hoc safety net, always works
      decode_mask(partial, pool) -> Optional[BoolVec]   # None if not decodable
    The orchestrator decides which constraints go in `enforce_during` vs
    `enforce_post` based on whether `decode_mask` returns non-None.
    """
    kind: str           # short tag, e.g. "allowed_subset", "thickness_range"
    params: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Inference spec (parsed prompt + knobs)
# ----------------------------------------------------------------------------

@dataclass
class InferenceKnobs:
    """User-tunable orchestration parameters.

    Defaults reflect the tunings I'd pick conservatively; tweak per run.
    """
    ensemble_N: int = 500            # candidates sampled by generate.py
    temperature: float = 1.0         # sampling diversity
    tolerance_pct: float = 5.0       # relative thickness jitter, e.g. 0.05 ⇒ ±5 %
    weight_lambda: float = 1.0       # color-vs-robustness trade-off in J = ΔE + λ·R
    top_k: int = 5                   # candidates refined and returned
    refine_max_iters: int = 100      # gradient local-search budget per candidate
    refine_step_size: float = 1.0    # nm; initial Adam step
    mc_samples: int = 32             # Monte-Carlo robustness draws on top_k
    seed: int = 42                   # base RNG seed (deterministic decoding)


@dataclass
class InferenceSpec:
    """Parsed user request: target color + constraints + knobs.

    `target_lab_raw` is the Lab value in its native units (L* ∈ [0, 100],
    a*/b* ∈ ~[-128, 128]); `target_lab_normalised` is in the model's input
    convention (L/100, a/128, b/128 — see src.materials_vocab.normalize_lab).
    """
    target_lab_raw: Tuple[float, float, float]
    target_lab_normalised: Tuple[float, float, float]
    constraints: List[Constraint] = field(default_factory=list)
    enforce_during: List[str] = field(default_factory=list)   # constraint kinds
    enforce_post: List[str] = field(default_factory=list)
    knobs: InferenceKnobs = field(default_factory=InferenceKnobs)
    # Free-form natural-language disclaimer of what the LLM thinks it enforced.
    parsed_disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict on frozen Constraint instances gives a plain dict already.
        return d


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------

@dataclass
class Provenance:
    """Reproducibility receipt — every Candidate / Result carries one.

    Inflated once on construction; downstream consumers should treat as read-
    only. If two Results claim the same provenance fingerprint, they are
    guaranteed to be replays of one decode.
    """
    model_tag: str = ""           # config.tag() of the loaded checkpoint
    model_sha256: str = ""        # short hash of model.pt bytes (first 16 hex)
    pool_fingerprint: str = ""
    seed: int = 0
    torch_version: str = ""
    jax_version: str = ""
    jaxlayerlumos_version: str = ""
    python_version: str = ""
    timestamp_unix: float = 0.0

    @classmethod
    def capture(cls, *, model_tag: str, model_sha256: str,
                pool_fp: str, seed: int) -> "Provenance":
        torch_v = ""
        jax_v = ""
        jll_v = ""
        try:
            import torch
            torch_v = torch.__version__
        except ImportError:
            pass
        try:
            import jax
            jax_v = jax.__version__
        except ImportError:
            pass
        try:
            import jaxlayerlumos
            jll_v = getattr(jaxlayerlumos, "__version__", "")
        except ImportError:
            pass
        return cls(
            model_tag=model_tag,
            model_sha256=model_sha256,
            pool_fingerprint=pool_fp,
            seed=seed,
            torch_version=torch_v,
            jax_version=jax_v,
            jaxlayerlumos_version=jll_v,
            python_version=platform.python_version(),
            timestamp_unix=time.time(),
        )


@dataclass
class RobustnessReport:
    """Robustness summary attached to a (refined) candidate.

    Two methods, both populated when a candidate makes it into the top_k:
      - gradient-based predicted shift (cheap, computed on all candidates
        during selection)
      - Monte-Carlo P95 / worst-case (validation, top_k only)
    Both use the same `tolerance_pct` from the InferenceKnobs.
    """
    grad_predicted_shift: float = float("nan")  # max_i |∂ΔE/∂tᵢ|·τᵢ·tᵢ — kept for back-compat
    grad_l2_shift: float = float("nan")         # sqrt(Σ (∂ΔE/∂tᵢ · τᵢ · tᵢ)²) — default reported
    mc_p50: float = float("nan")
    mc_p95: float = float("nan")
    mc_worst: float = float("nan")
    mc_samples: int = 0


@dataclass
class Candidate:
    """One generated (and optionally refined) structure.

    `thicknesses_nm` is continuous after refinement (we let the optimizer
    leave the 5 nm grid because the tolerance objective already accounts
    for manufacturing precision). `provenance` carries seed-level info so
    a single candidate can be re-decoded independently.
    """
    slot_indices: List[int]
    material_names: List[str]            # resolved from slot_indices + pool
    thicknesses_nm: List[float]          # nm; continuous post-refine, grid-aligned pre-refine
    achieved_lab: Tuple[float, float, float]
    reflectance: List[float]             # [NUM_LAMBDA]
    delta_e: float                       # nominal ΔE_00
    robustness: RobustnessReport = field(default_factory=RobustnessReport)
    objective: float = float("nan")      # J = ΔE + λ·R as evaluated for ranking
    refined: bool = False
    refine_iters: int = 0
    sample_index: int = -1               # index in the ensemble before dedup
    multiplicity: int = 1                # post-dedup count


# ----------------------------------------------------------------------------
# Result envelope (success and failure both ride this)
# ----------------------------------------------------------------------------

@dataclass
class ConstraintCheck:
    """Pass / fail per constraint after the post-hoc check on the chosen structure."""
    kind: str
    passed: bool
    detail: str = ""


@dataclass
class EnsembleStats:
    """Stats about the candidate pipeline. Useful for debugging / GUI render."""
    n_sampled: int = 0
    n_unique_after_dedup: int = 0
    n_feasible: int = 0
    n_refined: int = 0
    n_returned: int = 0
    dropped_per_constraint: Dict[str, int] = field(default_factory=dict)


@dataclass
class Result:
    """Top-level inference result. Serialises with `to_json()` for the GUI."""
    spec_echo: InferenceSpec
    chosen: Optional[Candidate]
    alternatives: List[Candidate] = field(default_factory=list)
    constraints_report: List[ConstraintCheck] = field(default_factory=list)
    ensemble_stats: EnsembleStats = field(default_factory=EnsembleStats)
    provenance: Provenance = field(default_factory=Provenance)
    errors: List[str] = field(default_factory=list)

    # ---- JSON I/O ----

    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, default=_json_default)
        if path is not None:
            Path(path).write_text(s)
        return s

    def to_dict(self) -> Dict[str, Any]:
        return _strip_arrays(asdict(self))

    @classmethod
    def from_json(cls, path_or_str: Any) -> "Result":
        text = (Path(path_or_str).read_text()
                if isinstance(path_or_str, (str, Path)) and Path(str(path_or_str)).exists()
                else str(path_or_str))
        d = json.loads(text)
        return _result_from_dict(d)


# ----------------------------------------------------------------------------
# JSON helpers (numpy arrays survive round-trips as lists; dataclasses survive
# via asdict + manual rebuild on the way back).
# ----------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if dataclasses.is_dataclass(o):
        return _strip_arrays(asdict(o))
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _strip_arrays(d: Any) -> Any:
    """Recursively replace numpy arrays with lists for clean JSON output."""
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_strip_arrays(x) for x in d]
    if isinstance(d, tuple):
        return [_strip_arrays(x) for x in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def _result_from_dict(d: Dict[str, Any]) -> Result:
    """Round-trip the dict back into a Result. Minimal validation — assumes
    the JSON came from `to_json()` of an earlier Result instance."""
    knobs = InferenceKnobs(**d["spec_echo"]["knobs"])
    constraints = [Constraint(kind=c["kind"], params=c.get("params", {}))
                   for c in d["spec_echo"].get("constraints", [])]
    spec = InferenceSpec(
        target_lab_raw=tuple(d["spec_echo"]["target_lab_raw"]),
        target_lab_normalised=tuple(d["spec_echo"]["target_lab_normalised"]),
        constraints=constraints,
        enforce_during=list(d["spec_echo"].get("enforce_during", [])),
        enforce_post=list(d["spec_echo"].get("enforce_post", [])),
        knobs=knobs,
        parsed_disclaimer=d["spec_echo"].get("parsed_disclaimer", ""),
    )

    def _cand(cd: Optional[Dict[str, Any]]) -> Optional[Candidate]:
        if cd is None:
            return None
        rob = RobustnessReport(**cd.get("robustness", {}))
        return Candidate(
            slot_indices=list(cd["slot_indices"]),
            material_names=list(cd["material_names"]),
            thicknesses_nm=list(cd["thicknesses_nm"]),
            achieved_lab=tuple(cd["achieved_lab"]),
            reflectance=list(cd["reflectance"]),
            delta_e=float(cd["delta_e"]),
            robustness=rob,
            objective=float(cd.get("objective", float("nan"))),
            refined=bool(cd.get("refined", False)),
            refine_iters=int(cd.get("refine_iters", 0)),
            sample_index=int(cd.get("sample_index", -1)),
            multiplicity=int(cd.get("multiplicity", 1)),
        )

    return Result(
        spec_echo=spec,
        chosen=_cand(d.get("chosen")),
        alternatives=[_cand(c) for c in d.get("alternatives", []) if c],
        constraints_report=[ConstraintCheck(**c) for c in d.get("constraints_report", [])],
        ensemble_stats=EnsembleStats(**d.get("ensemble_stats", {})),
        provenance=Provenance(**d.get("provenance", {})),
        errors=list(d.get("errors", [])),
    )
