"""Search presets: the one place their knobs are defined.

Read by inference/scripts/gamut_eval.py (--preset), the demo server
(GET /api/presets, and SolveIn.preset) and, through the server, the demo UI.
Before Nov 23 the UI kept its own copy of fast/balanced/best, none of which
was the paper's setting; that drift is why this module exists.

Every key is an InferenceKnobs field. refine_top_n = 0 means "refine all of
top_k". refine_step_size and refine_optimizer are left to InferenceKnobs'
defaults (1.0, "dog"), which is what test_eval.py, and so every paper number,
ran with.

No torch import: the analysis stack and the server's request models load it.
"""
from __future__ import annotations

from typing import Any, Dict, List

PRESETS: Dict[str, Dict[str, Any]] = {
    # THE PAPER'S SETTING (test_eval.py ENSEMBLE_N=250 TOP_K=10
    # REFINE_ITERS=100, tolerance 0, no MC): 250 samples, every one of the
    # top 10 polished with 100 DoG steps, top-1 selection on DeltaE alone.
    "strong":   dict(ensemble_N=250, top_k=10, refine_top_n=0,
                     refine_max_iters=100, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    # The scaling study's light pipeline (slurms/eval_pipeline.sh): best of
    # 16 samples, the top one polished with 5 DoG steps. Small on purpose,
    # so model differences are not hidden by the search.
    "light":    dict(ensemble_N=16, top_k=1, refine_top_n=1,
                     refine_max_iters=5, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    # 50 samples, the top one polished with 20 DoG steps (Oct 28).
    "mid":      dict(ensemble_N=50, top_k=1, refine_top_n=1,
                     refine_max_iters=20, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    # 250 samples, the top one polished with 40 DoG steps (Oct 28).
    "high":     dict(ensemble_N=250, top_k=1, refine_top_n=1,
                     refine_max_iters=40, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    "fast":     dict(ensemble_N=100, top_k=1, refine_top_n=1,
                     refine_max_iters=15, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    "balanced": dict(ensemble_N=150, top_k=3, refine_top_n=1,
                     refine_max_iters=25, tolerance_pct=0.0,
                     weight_lambda=1.0, mc_samples=0, temperature=1.0),
    # NOT more of the same search: tolerance_pct > 0 switches the objective
    # to DeltaE + lambda*R (select.py computes R only then), and MC draws rank
    # by robustness. Shown in the UI as "Robust".
    "best":     dict(ensemble_N=500, top_k=5, refine_top_n=3,
                     refine_max_iters=80, tolerance_pct=5.0,
                     weight_lambda=1.0, mc_samples=16, temperature=1.0),
    # Push the model to its limits; also robustness-weighted (tolerance 5).
    "max":      dict(ensemble_N=1000, top_k=5, refine_top_n=3,
                     refine_max_iters=120, tolerance_pct=5.0,
                     weight_lambda=1.0, mc_samples=32, temperature=0.9),
}

# What the demo offers, in order, and how it names each one.
UI_PRESETS: List[Dict[str, str]] = [
    {"name": "strong", "label": "Paper (250 × top-10, 100 steps)",
     "note": "The setting every number in the paper uses: minimises ΔE₀₀ alone."},
    {"name": "fast", "label": "Fast",
     "note": "Smaller search for a quick look; expect a higher ΔE₀₀ than the paper."},
    {"name": "balanced", "label": "Balanced",
     "note": "Between Fast and Paper; ΔE₀₀ alone."},
    {"name": "best", "label": "Robust (trades color accuracy for thickness tolerance)",
     "note": "Different objective: minimises ΔE₀₀ + λ·R at 5% thickness tolerance, "
             "with 16 Monte-Carlo draws. Not comparable to paper numbers."},
]
DEFAULT_UI_PRESET = "strong"


def knobs_for(name: str) -> Dict[str, Any]:
    """A copy of one preset's knobs; KeyError names the valid presets."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; one of {sorted(PRESETS)}")
    return dict(PRESETS[name])


def ui_presets() -> Dict[str, Any]:
    """The demo's preset list with each preset's knobs, for GET /api/presets."""
    return {"default": DEFAULT_UI_PRESET,
            "presets": [{**p, "knobs": knobs_for(p["name"])} for p in UI_PRESETS]}
