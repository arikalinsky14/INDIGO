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


# ----------------------------------------------------------------------------
# Process-wide handles (populated on startup)
# ----------------------------------------------------------------------------

_MODEL = None
_MODEL_CONFIG = None
_MODEL_TAG = ""
_MODEL_SHA = ""
_POOL = None         # List[MaterialEntry]
_POOL_ORIGIN = ""
_DEVICE = None


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
    _POOL = cap_pool_at_m_max(load_pool_from_jll(dir_), M_MAX)
    _POOL_ORIGIN = str(dir_)
    print(f"[server] pool: {len(_POOL)} materials from {_POOL_ORIGIN}")


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------

def _make_app():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    from inference.src.schema import (
        InferenceKnobs, InferenceSpec, pool_fingerprint,
    )
    from inference.src.solve import solve
    from src.materials_vocab import normalize_lab

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

    class SolveIn(BaseModel):
        # exactly one of prompt OR target_lab is honoured (prompt wins if both)
        prompt: Optional[str] = None
        target_lab: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
        constraints: Optional[List[Dict[str, Any]]] = None
        knobs: Optional[KnobsIn] = None

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
        }

    @app.get("/api/pool")
    def pool_listing() -> Dict[str, Any]:
        return {
            "materials": [
                {"canonical_name": m.canonical_name, "source": m.source}
                for m in _POOL
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

        # Two input shapes: prompt (LLM) or explicit Lab + optional constraints.
        if body.prompt:
            from inference.src.parse import ParseError, parse_prompt
            try:
                pr = parse_prompt(body.prompt, pool=_POOL, knobs=knobs)
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
                model=_MODEL, pool=_POOL, spec=spec,
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
