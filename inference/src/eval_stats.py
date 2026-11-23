"""DeltaE summary statistics for test_eval.py chunk summaries (torch-free).

Moved from inference/scripts/test_eval.py on Nov 23, unchanged, so that
scripts/merge_reruns.py can recompute a chunk's summary with the same code.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

DELTA_E_BUCKETS = (1.0, 2.0, 5.0, 10.0)


def aggregate_delta_e(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ΔE distribution stats + bucket counts. Skips rows that errored."""
    delta_es = [float(r["delta_e"]) for r in rows if "delta_e" in r]
    out: Dict[str, Any] = {
        "n_attempted": len(rows),
        "n_succeeded": len(delta_es),
        "n_failed": len(rows) - len(delta_es),
    }
    if not delta_es:
        out["delta_e"] = None
        out["buckets"] = None
        return out
    arr = np.asarray(delta_es, dtype=np.float64)
    out["delta_e"] = {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p95":    float(np.percentile(arr, 95)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }
    buckets: Dict[str, Any] = {}
    for t in DELTA_E_BUCKETS:
        below = int((arr < t).sum())
        buckets[f"<{t}"] = {"count": below, "fraction": below / len(arr)}
    out["buckets"] = buckets
    return out
