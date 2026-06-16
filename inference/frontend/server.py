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


def _prewarm_pipeline() -> None:
    """Run a tiny throwaway solve so the first user-facing solve doesn't pay
    the JAX / Python-import cold start. Cuts the user's first wall-clock by
    roughly the cold-start cost (5-30 s depending on the CPU)."""
    import time as _t
    from inference.src.schema import InferenceKnobs, InferenceSpec
    from inference.src.solve import solve as _solve
    from src.materials_vocab import normalize_lab
    if not (_MODEL and _POOL):
        return
    print("[server] prewarming JAX + pipeline (one tiny dummy solve)…",
          flush=True)
    t0 = _t.time()
    sm_pool = _POOL[: min(4, len(_POOL))]
    spec = InferenceSpec(
        target_lab_raw=(50.0, 0.0, 0.0),
        target_lab_normalised=tuple(float(x)
                                    for x in normalize_lab([50.0, 0.0, 0.0]).tolist()),
        constraints=[],
        enforce_during=[], enforce_post=[],
        knobs=InferenceKnobs(
            ensemble_N=8, top_k=1, refine_top_n=1,
            refine_max_iters=2, tolerance_pct=0.0, mc_samples=0,
            seed=0,
        ),
        parsed_disclaimer="prewarm",
    )
    try:
        _solve(model=_MODEL, pool=sm_pool, spec=spec,
               model_tag=_MODEL_TAG, model_sha256=_MODEL_SHA, device=_DEVICE)
        print(f"[server] prewarm done in {_t.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"[server] prewarm skipped: {type(exc).__name__}: {exc}",
              flush=True)


def _startup(checkpoint: Path, pool_dir: Optional[Path],
             force_cpu: bool = False, prewarm: bool = True) -> None:
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

    if prewarm:
        _prewarm_pipeline()


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
    refine_top_n: int = 0          # 0 = refine all top_k; >0 caps it
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

    # Opt-in toggle for the LLM-authored custom-constraint codegen path.
    # Defaults to False so the standard 8 kinds + periodic-pattern
    # expansion handle the request (no second OpenAI call, no sandboxed
    # exec). Users who want the escape hatch flip the Advanced-panel
    # checkbox.
    allow_custom_constraints: bool = False


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------

def _make_app():
    from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
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
        # Group by element prefix so the picker can show one row per
        # actual material with a "+ N variants" affordance instead of
        # 5 near-duplicate rows for "Ag-Rakic-LD-1998",
        # "Ag-Johnson-1972", etc. The differences between variants are
        # measurement-source spectra differences; the picker still lets
        # the user expand and pick a specific variant if needed.
        from inference.src.parse import _element_prefix
        groups: Dict[str, List[str]] = {}
        for m in materials:
            groups.setdefault(_element_prefix(m.canonical_name), []).append(
                m.canonical_name
            )
        # Build a deduplicated default: one variant per element prefix,
        # preferring the recommended variant the server's M_MAX cap chose
        # at startup so we don't drift from the trained pool.
        startup_pool = {m.canonical_name for m in (_POOL or [])}
        def _pick_one(variants: List[str]) -> str:
            preferred = [v for v in variants if v in startup_pool]
            return (preferred or variants)[0]
        deduped_default = [_pick_one(v) for v in groups.values()]
        deduped_default = deduped_default[:M_MAX]
        return {
            "materials": [
                {"canonical_name": m.canonical_name, "source": m.source,
                 "display_name": _element_prefix(m.canonical_name)}
                for m in materials
            ],
            "groups": {prefix: vs for prefix, vs in sorted(groups.items())},
            "default_subset": [m.canonical_name for m in (_POOL or [])],
            "deduped_default": deduped_default,
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
            refine_top_n=int(knobs_in.refine_top_n),
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
                    allow_custom_constraints=body.allow_custom_constraints,
                )
            except ParseError as exc:
                raise HTTPException(400, f"[parse:{exc.gate}] {exc.message}")
            except Exception as exc:
                import traceback as _tb
                _tb.print_exc()
                raise HTTPException(
                    500,
                    f"parse failed: {type(exc).__name__}: {exc}\n"
                    f"(see server logs for traceback)",
                )
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
            import traceback as _tb
            _tb.print_exc()
            raise HTTPException(
                500,
                f"solve failed: {type(exc).__name__}: {exc}\n"
                f"(see server logs for traceback)",
            )

        # Return the same envelope the JSON file persists. The UI knows this
        # shape — see static/app.js.
        from inference.src.schema import _strip_arrays  # internal but stable
        from dataclasses import asdict
        return JSONResponse(content=_strip_arrays(asdict(result)))

    @app.post("/api/solve_stream")
    def solve_stream(body: SolveIn = Body(...)) -> StreamingResponse:
        """Server-Sent Events variant of /api/solve.

        Yields a sequence of `event: progress` frames during the run and a
        final `event: result` (or `event: error`) frame at the end. Each
        frame's `data:` payload is a JSON object.

        The pipeline runs in a worker thread; progress callbacks push events
        into a thread-safe queue that this generator drains in order. The
        client (app.js) parses SSE with a ReadableStream reader since
        `EventSource` is GET-only and we want a JSON body.
        """
        if _MODEL is None:
            raise HTTPException(503, "model not loaded")

        # ---- Same body validation as /api/solve. Failures here become a
        # synchronous HTTPException so the client gets a normal 4xx, not an
        # SSE error frame, before any streaming starts.
        knobs_in = body.knobs or KnobsIn()
        knobs = InferenceKnobs(
            ensemble_N=int(knobs_in.ensemble_N),
            temperature=float(knobs_in.temperature),
            tolerance_pct=float(knobs_in.tolerance_pct),
            weight_lambda=float(knobs_in.weight_lambda),
            top_k=int(knobs_in.top_k),
            refine_top_n=int(knobs_in.refine_top_n),
            refine_max_iters=int(knobs_in.refine_max_iters),
            refine_step_size=float(knobs_in.refine_step_size),
            mc_samples=int(knobs_in.mc_samples),
            seed=int(knobs_in.seed),
        )
        try:
            effective_pool = _effective_pool(
                subset_names=body.pool_subset or [],
                custom=body.custom_materials or [],
            )
        except Exception as exc:
            raise HTTPException(400, f"pool build failed: {exc}")

        if body.prompt:
            from inference.src.parse import ParseError, parse_prompt
            try:
                pr = parse_prompt(
                    body.prompt, pool=effective_pool, knobs=knobs,
                    api_key=body.openai_api_key,
                    allow_custom_constraints=body.allow_custom_constraints,
                )
            except ParseError as exc:
                raise HTTPException(400, f"[parse:{exc.gate}] {exc.message}")
            except Exception as exc:
                import traceback as _tb
                _tb.print_exc()
                raise HTTPException(
                    500,
                    f"parse failed: {type(exc).__name__}: {exc}\n"
                    f"(see server logs for traceback)",
                )
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

        # ---- Streaming setup. The solver runs in a thread; the callback
        # publishes events on a queue this generator drains.
        import json
        import queue
        import threading
        import time as _time
        from dataclasses import asdict
        from inference.src.schema import _strip_arrays, _json_default

        q: "queue.Queue[dict]" = queue.Queue()
        _DONE = {"__sentinel__": True}
        _stream_id = f"{int(_time.time())}-{threading.get_ident()}"

        def _log(msg):
            print(f"[sse {_stream_id}] {msg}", flush=True)

        def _cb(stage, current, total, info):
            q.put({"type": "progress", "stage": stage,
                   "current": int(current), "total": int(total),
                   "info": info or {}})

        def _worker():
            import traceback as _tb
            _log("worker: starting solve()")
            t0 = _time.time()
            try:
                result = solve(
                    model=_MODEL, pool=effective_pool, spec=spec,
                    model_tag=_MODEL_TAG, model_sha256=_MODEL_SHA,
                    device=_DEVICE, on_progress=_cb,
                )
                _log(f"worker: solve() returned in {_time.time()-t0:.1f}s; "
                     f"chosen={result.chosen is not None}")
                try:
                    payload = _strip_arrays(asdict(result))
                except Exception as ser_exc:
                    _log(f"worker: _strip_arrays raised: "
                         f"{type(ser_exc).__name__}: {ser_exc}")
                    _tb.print_exc()
                    raise
                _log(f"worker: payload built (keys={list(payload.keys())}); "
                     f"queuing result")
                q.put({"type": "result", "result": payload})
                _log("worker: result queued")
            except Exception as exc:
                _log(f"worker raised: {type(exc).__name__}: {exc}")
                _tb.print_exc()
                q.put({"type": "error",
                       "message": f"{type(exc).__name__}: {exc}"})
            finally:
                _log("worker: queuing DONE")
                q.put(_DONE)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        def _sse_default(o):
            # Catches numpy scalars (np.float32), JAX arrays, and any other
            # exotic leaf the dataclass walker doesn't unwrap. Without this,
            # json.dumps raises inside the generator and the stream closes
            # silently — the browser sees "Connection closed before a result
            # arrived." which is impossible to debug without server logs.
            try:
                return _json_default(o)
            except Exception:
                pass
            for attr in ("tolist", "item"):
                fn = getattr(o, attr, None)
                if callable(fn):
                    try:
                        return fn()
                    except Exception:
                        pass
            return str(o)

        def _sse_format(event: str, payload: dict) -> str:
            # _strip_arrays sanitises NaN/Inf -> None, numpy scalars ->
            # Python, ndarrays -> lists. Applied to EVERY frame so progress
            # events (which can carry NaN `de` values from JAX physics on
            # bad seed structures) survive the encoder.
            # allow_nan=False is the tripwire: if any NaN/Inf still leaks
            # through _strip_arrays, fail loudly here (the per-yield
            # try/except in _gen converts that to an `error` frame).
            clean = _strip_arrays(payload)
            return (f"event: {event}\n"
                    f"data: {json.dumps(clean, default=_sse_default, allow_nan=False)}\n\n")

        def _gen():
            # Yield an immediate hello so the browser flushes headers and the
            # XHR / fetch reader unblocks even before the first real event.
            _log("gen: yielding hello")
            yield _sse_format("hello", {"ok": True})
            n_progress = 0
            heartbeat_every = 5.0
            last_event_at = _time.time()
            while True:
                # Block with a short timeout so we can heartbeat. SSE comments
                # (lines starting with `:`) keep proxies and dev tools awake
                # without polluting the event stream.
                try:
                    msg = q.get(timeout=heartbeat_every)
                except queue.Empty:
                    _log(f"gen: heartbeat (idle "
                         f"{_time.time()-last_event_at:.1f}s; "
                         f"worker_alive={thread.is_alive()})")
                    yield f": heartbeat {int(_time.time())}\n\n"
                    if not thread.is_alive():
                        _log("gen: worker died without DONE; emitting error")
                        yield _sse_format("error",
                            {"message": "worker thread died silently — see "
                                        "server logs for traceback"})
                        break
                    continue
                last_event_at = _time.time()
                if msg is _DONE:
                    _log(f"gen: got DONE after {n_progress} progress events; "
                         f"closing stream")
                    break
                t = msg.get("type")
                try:
                    if t == "progress":
                        n_progress += 1
                        yield _sse_format("progress", msg)
                    elif t == "result":
                        _log("gen: yielding result frame")
                        chunk = _sse_format("result", msg["result"])
                        _log(f"gen: result frame size={len(chunk)} bytes")
                        yield chunk
                        _log("gen: result frame yielded successfully")
                    elif t == "error":
                        _log(f"gen: yielding error frame "
                             f"({msg.get('message','')[:80]})")
                        yield _sse_format("error", {"message": msg["message"]})
                except Exception as ser_exc:
                    import traceback as _tb
                    _log(f"gen: yield failed on {t}: "
                         f"{type(ser_exc).__name__}: {ser_exc}")
                    _tb.print_exc()
                    err_payload = {
                        "message": f"server SSE serialise failed: "
                                   f"{type(ser_exc).__name__}: {ser_exc}",
                        "trace": _tb.format_exc(),
                    }
                    try:
                        yield (f"event: error\n"
                               f"data: {json.dumps(err_payload)}\n\n")
                    except Exception:
                        pass
            _log("gen: returning (stream end)")

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering
                "Connection": "keep-alive",
            },
        )

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

def _find_default_checkpoint(repo_root: Path) -> Optional[Path]:
    """Auto-pick a checkpoint when --checkpoint is omitted.

    Search order, first hit wins:
      1) $INDIGO_CHECKPOINT env var
      2) data/checkpoints/<tag>/latest  — most recently modified
      3) data/checkpoints/<tag>          — most recently modified leaf

    Returns None if nothing valid is found; the caller renders a friendly
    error instead of crashing inside the loader.
    """
    env = os.environ.get("INDIGO_CHECKPOINT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "model.pt").is_file():
            return p

    root = repo_root / "data" / "checkpoints"
    if not root.is_dir():
        return None

    # Prefer */latest symlinks/dirs (the canonical convention from training).
    latests = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        cand = sub / "latest"
        if cand.is_dir() and (cand / "model.pt").is_file():
            latests.append(cand)
    if latests:
        latests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return latests[0]

    # Fall through: any subdir with a model.pt.
    others = [sub for sub in root.iterdir()
              if sub.is_dir() and (sub / "model.pt").is_file()]
    others.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return others[0] if others else None


def main() -> int:
    p = argparse.ArgumentParser(description="INDIGO inference web server")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved checkpoint dir "
                        "(data/checkpoints/<tag>/latest). "
                        "If omitted, the most recent data/checkpoints/*/latest "
                        "in this repo is used; $INDIGO_CHECKPOINT overrides.")
    p.add_argument("--pool-dir", type=str, default=None,
                   help="JLL materials directory (default: installed package)")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Bind host (default 127.0.0.1; use 0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="dev-only: uvicorn reload on file change")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU for the model forward. The server probes "
                        "CUDA at startup and falls back to CPU automatically "
                        "if cuDNN / cuBLAS aren't loadable; --cpu just skips "
                        "the probe.")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Skip the tiny dummy solve at startup. Saves ~5-30 s "
                        "of cold-start time but makes the first user request "
                        "pay that cost instead.")
    args = p.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = _find_default_checkpoint(_root)
        if ckpt is None:
            print("[server] No checkpoint found.\n"
                  "        Pass --checkpoint <path>, or place one at "
                  "data/checkpoints/<tag>/latest/, or set $INDIGO_CHECKPOINT.",
                  file=sys.stderr)
            return 2
        print(f"[server] auto-selected checkpoint: {ckpt}")

    _startup(ckpt,
             Path(args.pool_dir) if args.pool_dir else None,
             force_cpu=args.cpu, prewarm=not args.no_prewarm)

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
