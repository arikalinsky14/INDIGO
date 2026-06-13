"""
FastAPI server for INDIGO inference.

Wraps `inference.src.solve.solve()` behind a tiny HTTP API and serves the
single-page UI from `static/`. The pure-function boundary at `solve()` is
the entire ML interface — this module is purely glue.

Routes
------
  GET  /api/status               loaded checkpoint + pool fingerprint + key flags
  GET  /api/pool                 the canonical-name list of the loaded pool
  POST /api/solve                run inference. Body matches the structured spec
                                  or a free-text prompt; see docstring below.
  GET  /                         single-page UI (static/index.html)
  GET  /static/...               CSS / JS / etc.

Running locally
---------------
  python -m inference.frontend.server \\
      --checkpoint data/checkpoints/<tag>/latest \\
      --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000 in a browser.

If you're on the cluster, port-forward:
  ssh -L 8000:<node>:8000 user@cluster
or run on a node with public-ish networking and bind to 0.0.0.0.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from src.materials_vocab import M_MAX
from inference.src.schema import MaterialEntry


def _effective_pool(subset_names, custom):
    """Build the request-time pool from the named JLL subset + uploaded custom
    materials. If both are empty, falls back to the default server-side pool.

    Caps at M_MAX. Custom materials win on canonical-name collision with
    a JLL entry, so a user can override a built-in spectrum without renaming.
    """
    pool = []
    seen: set = set()
    if subset_names:
        for n in subset_names:
            m = _FULL_JLL_BY_NAME.get(n) if _FULL_JLL_BY_NAME else None
            if m is None:
                raise RuntimeError(f"unknown JLL material: {n!r}")
            if m.canonical_name in seen:
                continue
            pool.append(m); seen.add(m.canonical_name)
    if custom:
        for c in custom:
            # Coerce list[float] → np.float32 arrays once at the boundary.
            mat = MaterialEntry(
                canonical_name=c.canonical_name,
                n=np.asarray(c.n, dtype=np.float32),
                k=np.asarray(c.k, dtype=np.float32),
                source=c.source or "custom",
            )
            # Replace any same-named JLL entry already in the pool.
            pool = [p for p in pool if p.canonical_name != mat.canonical_name]
            pool.append(mat)
            seen.add(mat.canonical_name)
    if not pool:
        return _POOL  # default capped JLL
    if len(pool) > M_MAX:
        raise RuntimeError(
            f"pool of {len(pool)} exceeds M_MAX={M_MAX}; deselect some"
        )
    return pool


# ----------------------------------------------------------------------------
# Process-wide handles (populated on startup)
# ----------------------------------------------------------------------------

_MODEL = None
_MODEL_CONFIG = None
_MODEL_TAG = ""
_MODEL_SHA = ""
_POOL = None             # List[MaterialEntry] — default M_MAX-capped subset
_POOL_ORIGIN = ""
_DEVICE = None
_FULL_JLL_POOL = None    # List[MaterialEntry] — uncapped JLL library
_FULL_JLL_BY_NAME = None # dict canonical_name -> MaterialEntry


def _resolve_device(force_cpu: bool) -> "torch.device":
    """Pick CPU vs CUDA defensively.

    `torch.cuda.is_available()` only checks that the NVIDIA driver is loaded,
    not that every shared library torch was built against (cuDNN, cuBLAS) is
    actually findable. On visualization / non-GPU nodes at Pitt CRC the
    driver is present but `libcudnn_graph.so.9.10.x` is not — a real CUDA
    op core-dumps before any of our code runs.

    Probe with a tiny op; fall back to CPU loudly. `--cpu` forces it.
    """
    import torch
    if force_cpu:
        print("[server] --cpu set; using CPU for the model forward.")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda").sum().item()
        return torch.device("cuda")
    except Exception as exc:
        print(f"[server] CUDA visible but unusable "
              f"({type(exc).__name__}: {exc}). Falling back to CPU.",
              file=sys.stderr)
        return torch.device("cpu")


def _startup(checkpoint: Path, pool_dir: Optional[Path],
             force_cpu: bool = False) -> None:
    """Load model + pool once. Called from main() before serving."""
    import torch
    from inference.scripts.run_inference import (
        cap_pool_at_m_max, default_jll_materials_dir, load_pool_from_jll,
    )
    from inference.src.generate import load_inference_model
    from src.materials_vocab import M_MAX

    global _MODEL, _MODEL_CONFIG, _MODEL_TAG, _MODEL_SHA
    global _POOL, _POOL_ORIGIN, _DEVICE
    global _FULL_JLL_POOL, _FULL_JLL_BY_NAME

    _DEVICE = _resolve_device(force_cpu)
    print(f"[server] device: {_DEVICE}")

    print(f"[server] loading model from {checkpoint}")
    _MODEL, _MODEL_CONFIG, _MODEL_SHA = load_inference_model(
        checkpoint, device=_DEVICE,
    )
    _MODEL_TAG = _MODEL_CONFIG.tag()
    print(f"[server] model tag: {_MODEL_TAG}")
    print(f"[server] model sha: {_MODEL_SHA}")

    dir_ = pool_dir or default_jll_materials_dir()
    full = load_pool_from_jll(dir_)
    _FULL_JLL_POOL = full
    _FULL_JLL_BY_NAME = {m.canonical_name: m for m in full}
    _POOL = cap_pool_at_m_max(full, M_MAX)
    _POOL_ORIGIN = str(dir_)
    print(f"[server] JLL library: {len(full)} materials from {_POOL_ORIGIN}")
    print(f"[server] default pool (M_MAX-capped): {len(_POOL)}")


# ----------------------------------------------------------------------------
# Request models — MUST live at module scope, not inside _make_app(). Pydantic
# v2's TypeAdapter resolves forward references through module globals; a class
# defined inside a closure isn't visible there and FastAPI's body validator
# raises `is not fully defined; … call .rebuild() on the instance.`
# ----------------------------------------------------------------------------

from pydantic import BaseModel, Field


class KnobsIn(BaseModel):
    ensemble_N: int = 500
    temperature: float = 1.0
    tolerance_pct: float = 5.0
    weight_lambda: float = 1.0
    top_k: int = 5
    refine_max_iters: int = 100
    refine_step_size: float = 1.0
    mc_samples: int = 32
    seed: int = 42


class CustomMaterial(BaseModel):
    """User-supplied material — already interpolated to the canonical
    NUM_LAMBDA grid by /api/material_from_csv before reaching solve."""
    canonical_name: str
    n: List[float]
    k: List[float]
    source: str = "custom"


class SolveIn(BaseModel):
    # exactly one of prompt OR target_lab is honoured (prompt wins if both)
    prompt: Optional[str] = None
    target_lab: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    constraints: Optional[List[Dict[str, Any]]] = None
    knobs: Optional[KnobsIn] = None

    # Dynamic pool inputs (all optional; empty = use default capped JLL).
    # `pool_subset` is canonical names already known to the server (subset
    # of the full installed JLL library). `custom_materials` carries
    # user-uploaded n,k arrays. Combined pool is capped at M_MAX.
    pool_subset: Optional[List[str]] = None
    custom_materials: Optional[List[CustomMaterial]] = None

    # Per-request OpenAI key — overrides $OPENAI_API_KEY for this call only.
    # Never persisted server-side; treated as a sensitive header.
    openai_api_key: Optional[str] = None


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------

def _make_app():
    from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    from inference.src.schema import (
        InferenceKnobs, InferenceSpec, pool_fingerprint,
    )
    from inference.src.solve import solve
    from src.materials_vocab import normalize_lab

    app = FastAPI(title="INDIGO inference", version="1.0")
    # Permissive CORS so a co-developer can hit the API from a separate dev
    # server on localhost without fighting the browser. Tighten for prod.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # API routes
    # ------------------------------------------------------------------

    @app.get("/api/status")
    def status() -> Dict[str, Any]:
        return {
            "model_tag": _MODEL_TAG,
            "model_sha256": _MODEL_SHA,
            "pool_size": len(_POOL) if _POOL else 0,
            "pool_fingerprint": pool_fingerprint(_POOL) if _POOL else None,
            "pool_origin": _POOL_ORIGIN,
            "device": str(_DEVICE) if _DEVICE else "unknown",
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "parse_backend": os.environ.get("INDIGO_PARSE_BACKEND", "openai"),
            "m_max": M_MAX,
        }

    @app.get("/api/pool")
    def pool_listing() -> Dict[str, Any]:
        """Full uncapped JLL library — the frontend lets the user pick ≤M_MAX."""
        materials = _FULL_JLL_POOL or []
        return {
            "materials": [
                {"canonical_name": m.canonical_name, "source": m.source}
                for m in materials
            ],
            "default_subset": [m.canonical_name for m in (_POOL or [])],
            "m_max": M_MAX,
        }

    @app.post("/api/material_from_csv")
    def material_from_csv(file: "UploadFile" = File(...),
                          name: str = Form(...)) -> Dict[str, Any]:
        """Parse a user-uploaded CSV → interpolated MaterialNK on the canonical
        128-point frequency grid. Reuses src.material_features.load_jll_material
        so the same parser the training data uses handles the user's CSV.

        See inference/frontend/README.md for the CSV format.
        """
        import tempfile
        from src.material_features import load_jll_material

        name = (name or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        try:
            raw = file.file.read()
        except Exception as exc:
            raise HTTPException(400, f"could not read upload: {exc}")
        if not raw:
            raise HTTPException(400, "uploaded CSV is empty")
        # `load_jll_material` derives the canonical_name from the filename
        # stem; honour the user's `name` form field by giving the tmpfile
        # exactly that stem.
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / f"{name}.csv"
            csv_path.write_bytes(raw)
            try:
                mat = load_jll_material(csv_path)
            except Exception as exc:
                raise HTTPException(
                    400, f"CSV parse failed: {type(exc).__name__}: {exc}. "
                         f"See /api/csv_format for the expected layout."
                )
        return {
            "canonical_name": mat.name,
            "n": mat.n.tolist(),
            "k": mat.k.tolist(),
            "source": "custom",
        }

    @app.get("/api/csv_format")
    def csv_format() -> Dict[str, Any]:
        """Machine-readable copy of the CSV format guide rendered in the UI."""
        return {
            "expected_columns": "wavelength_nm, n, k",
            "example_rows": [
                "wavelength_nm,n,k",
                "300,1.479,0.0",
                "550,1.469,0.0",
                "900,1.464,0.0",
            ],
            "interpolation": (
                "Uploaded rows are interpolated onto a canonical 128-point "
                "grid uniform in FREQUENCY (not wavelength) spanning 300-900 "
                "nm. Wavelengths outside the file's coverage are clamped to "
                "the nearest endpoint. The interpolated arrays are exactly "
                "the format the model was trained against."
            ),
            "max_pool_size": M_MAX,
            "notes": [
                "First column may be wavelength_nm or just nm. The "
                "load_jll_material parser is permissive on header names.",
                "k may be omitted (or all 0) for non-absorbing dielectrics.",
                "Negative n or k are clamped to a small positive epsilon.",
            ],
        }

    @app.post("/api/solve")
    def solve_route(body: SolveIn = Body(...)) -> JSONResponse:
        # `Body(...)` is required on FastAPI ≥ 0.115 when every field on the
        # Pydantic model is Optional — without it FastAPI heuristics route
        # the param to query parsing and the client gets a 422
        # "Field required" at loc=['query','body'].
        if _MODEL is None:
            raise HTTPException(503, "model not loaded")

        knobs_in = body.knobs or KnobsIn()
        knobs = InferenceKnobs(
            ensemble_N=int(knobs_in.ensemble_N),
            temperature=float(knobs_in.temperature),
            tolerance_pct=float(knobs_in.tolerance_pct),
            weight_lambda=float(knobs_in.weight_lambda),
            top_k=int(knobs_in.top_k),
            refine_max_iters=int(knobs_in.refine_max_iters),
            refine_step_size=float(knobs_in.refine_step_size),
            mc_samples=int(knobs_in.mc_samples),
            seed=int(knobs_in.seed),
        )

        # Effective pool: pool_subset (named subset of installed JLL) +
        # custom_materials (uploaded n,k). If both empty, fall back to the
        # default pool the server loaded at startup.
        try:
            effective_pool = _effective_pool(
                subset_names=body.pool_subset or [],
                custom=body.custom_materials or [],
            )
        except Exception as exc:
            raise HTTPException(400, f"pool build failed: {exc}")

        # Two input shapes: prompt (LLM) or explicit Lab + optional constraints.
        if body.prompt:
            from inference.src.parse import ParseError, parse_prompt
            try:
                pr = parse_prompt(
                    body.prompt, pool=effective_pool, knobs=knobs,
                    api_key=body.openai_api_key,
                )
            except ParseError as exc:
                raise HTTPException(400, f"[parse:{exc.gate}] {exc.message}")
            spec = pr.spec
        elif body.target_lab is not None:
            from inference.scripts.run_inference import (
                _build_constraints_from_json,
            )
            target = tuple(float(x) for x in body.target_lab)
            norm_t = normalize_lab(list(target))
            constraints = []
            if body.constraints:
                try:
                    constraints = _build_constraints_from_json(body.constraints)
                except Exception as exc:
                    raise HTTPException(400, f"bad constraints: {exc}")
            spec = InferenceSpec(
                target_lab_raw=target,
                target_lab_normalised=tuple(float(x) for x in norm_t.tolist()),
                constraints=constraints,
                enforce_during=[], enforce_post=[],
                knobs=knobs,
                parsed_disclaimer="structured input from frontend",
            )
        else:
            raise HTTPException(400, "send either `prompt` or `target_lab`")

        try:
            result = solve(
                model=_MODEL, pool=effective_pool, spec=spec,
                model_tag=_MODEL_TAG, model_sha256=_MODEL_SHA,
                device=_DEVICE,
            )
        except Exception as exc:
            raise HTTPException(500, f"solve failed: {type(exc).__name__}: {exc}")

        # Return the same envelope the JSON file persists. The UI knows this
        # shape — see static/app.js.
        from inference.src.schema import _strip_arrays  # internal but stable
        from dataclasses import asdict
        return JSONResponse(content=_strip_arrays(asdict(result)))

    # ------------------------------------------------------------------
    # Static UI — mounted LAST so /api/* routes take precedence.
    # ------------------------------------------------------------------
    static_root = Path(__file__).resolve().parent / "static"
    if not static_root.exists():
        raise RuntimeError(f"static dir missing: {static_root}")
    app.mount("/", StaticFiles(directory=str(static_root), html=True),
              name="static")

    return app


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="INDIGO inference web server")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved checkpoint dir (data/checkpoints/<tag>/latest)")
    p.add_argument("--pool-dir", type=str, default=None,
                   help="JLL materials directory (default: installed package)")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Bind host (default 127.0.0.1; use 0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="dev-only: uvicorn reload on file change")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU for the model forward. Use on viz / "
                        "non-GPU nodes where CUDA libs are incomplete.")
    args = p.parse_args()

    _startup(Path(args.checkpoint),
             Path(args.pool_dir) if args.pool_dir else None,
             force_cpu=args.cpu)

    try:
        import uvicorn
    except ImportError:
        print("[server] uvicorn not installed; pip install uvicorn", file=sys.stderr)
        return 1

    app = _make_app()
    print(f"[server] listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info", reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
