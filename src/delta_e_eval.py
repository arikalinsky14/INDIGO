"""
Shared ΔE₀₀ validation
======================

One implementation of "generate autoregressively, simulate the result, and
score CIEDE2000 against the target", used by:

  * `scripts/training.py`   -- in-training val hook, writes val_de to history.jsonl
  * `scripts/lr_tuning.py`  -- per-scale LR selection
  * (optionally) offline eval

Why this module exists
----------------------
ΔE is the metric the compute-optimal scaling study actually fits its IsoFLOP
parabolas on, and CE is NOT a usable proxy for it: the decoupling between
cross-entropy and ΔE₀₀ has been verified on INDIGO (see the Sept 15 notes in
CLAUDE.md, and the fact that a single production run's CE-optimal and
ΔE-optimal checkpoints are different steps). Anything that selects a
checkpoint, an LR, or a point on a scaling curve must therefore score ΔE
directly rather than ranking on val_loss.

Before this module, ΔE existed only inside `scripts/evaluate.py` as part of a
much larger offline script, so the training loop had no way to see it.

Cost
----
Measured on CPU, steady state:

  * optical sim      ~50 ms/example  (irreducible, serial, CPU-bound)
  * generation        29 ms/example at d_model=256, 305 ms at d_model=1024

The optical sim dominates on small models and is roughly matched by
generation on large ones. A 500-example eval therefore costs ~25-30 s of
simulation, which is 3-4% overhead on the production run's ~700-1100 s
checkpoint interval.

One caveat worth knowing: jaxlayerlumos re-traces per distinct stack shape,
and the shape depends on the layer count, so the first example at each of the
9 possible depths pays a ~440 ms compile. That is ~4 s of one-time cost per
PROCESS (the JAX trace cache is process-global, not per-simulator-instance).
Short evals in fresh processes pay it; a long-running training job pays it
once. Padding stacks to a fixed depth would collapse it to a single trace and
is numerically exact -- zero-thickness layers are a true transfer-matrix
no-op, verified to 6e-14 in Lab -- but measurably does not help, because
there are only 9 shapes to begin with.

Availability
------------
The optical simulator needs `jaxlayerlumos`. When it is missing this module
degrades gracefully: `evaluate_delta_e` returns a result with
`available=False` and no metrics, and callers skip ΔE rather than failing.
A validation hook must never be able to kill a training run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.color_utils import ciede2000, lab_chroma
from src.materials_vocab import denormalize_lab
from src.model import generate_structure

OPTICAL_SIM_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None
try:
    from src.optical_sim import OpticalSimulator, is_available as _optical_is_available
    OPTICAL_SIM_AVAILABLE = _optical_is_available()
    if not OPTICAL_SIM_AVAILABLE:
        from src.optical_sim import get_import_error as _get_import_error
        _IMPORT_ERROR = _get_import_error()
except Exception as exc:  # pragma: no cover - import-environment dependent
    _IMPORT_ERROR = str(exc)


# Chroma bucket edges on C* = sqrt(a*^2 + b*^2), matching
# `src.color_utils.lab_chroma`. C* is the CIE-standard chroma and is the same
# quantity the dataset generator's high-chroma search targets
# (HIGH_CHROMA_CHROMA_MIN/MAX in slurms/generate_data.sh), so buckets defined
# this way relate directly to the HC subset.
DEFAULT_CHROMA_EDGES: Tuple[float, float] = (20.0, 50.0)
CHROMA_BUCKET_NAMES: Tuple[str, str, str] = ("low", "mid", "high")


def chroma_bucket(lab: Sequence[float],
                  edges: Tuple[float, float] = DEFAULT_CHROMA_EDGES) -> str:
    """Bucket a DENORMALISED Lab target by chroma magnitude.

    'low' C* < edges[0], 'mid' edges[0] <= C* < edges[1], 'high' C* >= edges[1].
    """
    c = lab_chroma(lab)
    if c < edges[0]:
        return CHROMA_BUCKET_NAMES[0]
    if c < edges[1]:
        return CHROMA_BUCKET_NAMES[1]
    return CHROMA_BUCKET_NAMES[2]


def _summarise(values: List[float]) -> Dict[str, float]:
    """Distribution summary. p95 is included because the production failure
    mode is the high-chroma tail, not the median (see CLAUDE.md)."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def unavailable_result(reason: Optional[str] = None) -> Dict[str, Any]:
    """The shape callers get when ΔE cannot be computed."""
    return {
        "available": False,
        "reason": reason or _IMPORT_ERROR or "optical sim unavailable",
        "n_evaluated": 0,
    }


def evaluate_delta_e(
    model: torch.nn.Module,
    examples: Iterable,
    device: torch.device,
    *,
    limit: Optional[int] = None,
    sample: bool = False,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    simulator: Optional[Any] = None,
    chroma_edges: Tuple[float, float] = DEFAULT_CHROMA_EDGES,
    incidence_angle: float = 0.0,
    progress_every: int = 0,
) -> Dict[str, Any]:
    """Greedy-decode each example, simulate, and score ΔE₀₀ against the target.

    Parameters
    ----------
    examples
        Iterable of `TrainingExample`. Consumed at most `limit` times, so a
        streaming dataset can be passed directly.
    limit
        Cap on examples scored. ΔE is far more expensive than CE, so callers
        normally pass a few hundred rather than the full split.
    sample
        False (default) = greedy argmax decoding, which is what a val metric
        should be: deterministic and seed-independent. True temperature-samples.
    seed
        Only meaningful when `sample=True`; makes the sampled metric
        reproducible across checkpoints.
    simulator
        Reuse an existing `OpticalSimulator` to keep the JAX trace cache warm
        across calls. One is built on demand if omitted.

    Returns
    -------
    dict with `available`, overall distribution stats, `by_chroma` (the
    per-bucket breakdown the scaling study conditions on), and generation
    diagnostics. On failure, the shape from `unavailable_result`.

    Never raises on a per-example simulation failure: such examples are
    counted in `n_sim_failed` and excluded. This runs inside a training loop.
    """
    if not OPTICAL_SIM_AVAILABLE:
        return unavailable_result()

    if simulator is None:
        try:
            simulator = OpticalSimulator(incidence_angle=incidence_angle)
        except Exception as exc:
            return unavailable_result(f"simulator construction failed: {exc}")

    generator = None
    if sample and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()

    all_de: List[float] = []
    by_bucket: Dict[str, List[float]] = {n: [] for n in CHROMA_BUCKET_NAMES}
    bucket_counts: Dict[str, int] = {n: 0 for n in CHROMA_BUCKET_NAMES}
    n_seen = n_valid = n_eos = n_sim_failed = 0

    try:
        with torch.no_grad():
            for example in examples:
                if limit is not None and n_seen >= limit:
                    break
                n_seen += 1

                gt_lab = denormalize_lab(example.lab)
                bucket = chroma_bucket(gt_lab, chroma_edges)
                bucket_counts[bucket] += 1

                pred_slots, pred_thick, stop_reason = generate_structure(
                    model, example.lab, example.pool, device,
                    sample=sample, temperature=temperature, generator=generator,
                )
                if stop_reason == "EOS":
                    n_eos += 1
                if not pred_slots:
                    continue
                n_valid += 1

                try:
                    pred_lab = simulator.compute_lab(
                        pool=example.pool,
                        slot_indices=pred_slots,
                        thicknesses_nm=pred_thick,
                    )
                except Exception:
                    # A single bad stack must not take down training.
                    n_sim_failed += 1
                    continue

                de = float(ciede2000(gt_lab, pred_lab))
                all_de.append(de)
                by_bucket[bucket].append(de)

                if progress_every and n_seen % progress_every == 0:
                    running = float(np.median(all_de)) if all_de else float("nan")
                    print(f"  [dE] {n_seen} examples, running median={running:.3f}",
                          flush=True)
    finally:
        if was_training:
            model.train()

    overall = _summarise(all_de)
    result: Dict[str, Any] = {
        "available": True,
        "n_evaluated": n_seen,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_seen if n_seen else 0.0,
        "eos_rate": n_eos / n_seen if n_seen else 0.0,
        "n_sim_failed": n_sim_failed,
        "n_scored": overall.get("n", 0),
        "greedy": not sample,
        "temperature": temperature if sample else None,
        "chroma_edges": list(chroma_edges),
    }
    # Flatten the overall stats under a val_de_* prefix so history.jsonl rows
    # stay one level deep and are trivially plottable.
    for key, value in overall.items():
        if key == "n":
            continue
        result[f"delta_e_{key}"] = value
    result["by_chroma"] = {
        name: {**_summarise(by_bucket[name]), "n_examples": bucket_counts[name]}
        for name in CHROMA_BUCKET_NAMES
    }
    return result


def primary_metric(result: Dict[str, Any]) -> float:
    """The single scalar to rank checkpoints / LRs / IsoFLOP points on.

    Median rather than mean: the ΔE distribution has a heavy high-chroma tail
    (CLAUDE.md records p95 ~3.4 against a median ~0.7 at production settings),
    and a mean over that tail is dominated by a handful of edge-of-gamut
    targets, making it a noisy objective. Returns +inf when unavailable so
    argmin-style selection skips the point instead of picking it.
    """
    if not result.get("available") or not result.get("n_scored"):
        return float("inf")
    return float(result["delta_e_median"])
