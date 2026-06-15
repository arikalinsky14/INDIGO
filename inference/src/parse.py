"""
LLM-driven prompt parser for INDIGO inference.

  free-text prompt + pool (canonical names)   ─►   InferenceSpec

Trust model
-----------
The LLM only emits CALLS INTO the pre-coded constraint library defined in
`inference/src/constraints.py`. We do not execute LLM-authored code or
arbitrary structures. The JSON schema enforced by OpenAI's
`response_format` rules out everything outside the 8 known constraint
kinds before we even see the response.

Three validation gates (applied in order)
----------------------------------------
1. **Schema**       — handled by the OpenAI `response_format=json_schema`
                      strict mode. Anything outside the schema is a hard
                      parse error from the API.
2. **Semantic**     — every `material_name` in the response must resolve
                      to a slot in the pool; layer positions in
                      `[0, MAX_LAYERS)`; thicknesses on the 5 nm grid in
                      `[5, MAX_THICKNESS_NM]`; pool size ≤ M_MAX.
3. **Physical**     — `layer_count.min_layers ≤ max_layers` and friends;
                      `total_thickness.max_total_nm` admits at least one
                      valid layer; `allowed_subset` non-empty;
                      `adjacent_forbidden` pairs reference real names;
                      `ordering_before.name_a ≠ name_b`.

Failure modes ride the same error envelope as the orchestrator — the
caller (`run_inference.py`) builds a failure `Result` with `errors=[...]`
populated and `chosen=None`.

LLM backend
-----------
Default: OpenAI (`openai` SDK). Configurable model via `OPENAI_MODEL`
env var; defaults to `gpt-4o-mini` for cost. `OPENAI_API_KEY` required
for live calls.

For tests / dev without an API key, set `INDIGO_PARSE_BACKEND=mock`
and the parser returns a hand-rolled `InferenceSpec` from a tiny
keyword router so the rest of the pipeline can be exercised offline.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.materials_vocab import (
    MAX_LAYERS, MAX_THICKNESS_NM, M_MAX, THICKNESSES, normalize_lab,
)

from inference.src.constraints import (
    AdjacentForbidden, AllowedSubset, LayerCount, LayerIdentity,
    OrderingBefore, Symmetry, ThicknessRange, TotalThickness,
)
from inference.src.schema import (
    InferenceKnobs, InferenceSpec, MaterialEntry,
)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------

class ParseError(RuntimeError):
    """A user request the parser refused to honour.

    Carries the gate that failed and an actionable message so the failure
    Result envelope can render the cause clearly.
    """
    def __init__(self, gate: str, message: str):
        super().__init__(f"[{gate}] {message}")
        self.gate = gate
        self.message = message


# ----------------------------------------------------------------------------
# JSON schema we force the LLM to produce
# ----------------------------------------------------------------------------

def _response_json_schema() -> Dict[str, Any]:
    """Strict JSON-Schema describing the LLM response.

    Mirrors the 8 Constraint subclasses in constraints.py. Fields the user
    didn't talk about (e.g. unconstrained position) are nullable so the
    schema fits all 8 kinds in one anyOf without per-kind branching.
    """
    return {
        "name": "inference_spec",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["target_lab", "constraints", "disclaimer",
                         "custom_constraint_request"],
            "properties": {
                "target_lab": {
                    "type": "array",
                    "description": "CIE Lab target: [L*, a*, b*]. "
                                   "L in [0,100], a/b roughly in [-128, 128].",
                    "items": {"type": "number"},
                    "minItems": 3, "maxItems": 3,
                },
                "constraints": {
                    "type": "array",
                    "description": "List of structural constraints. Empty if "
                                   "the request is underspecified — let the "
                                   "model search the full space.",
                    "items": _constraint_schema(),
                },
                "custom_constraint_request": {
                    "type": ["string", "null"],
                    "description": "ESCAPE HATCH — fill in ONLY when the "
                                   "user's request truly cannot be expressed "
                                   "by the 8 supported constraint kinds "
                                   "(even after expanding patterns like "
                                   "'every other X' into multiple "
                                   "layer_identity entries). When non-null, "
                                   "a SECOND LLM call will generate sandboxed "
                                   "Python code for this constraint. Keep "
                                   "this null for >95% of requests.",
                },
                "disclaimer": {
                    "type": "string",
                    "description": "Human-readable summary of how the prompt "
                                   "was interpreted. Echo any defaults or "
                                   "assumptions the parser made.",
                },
            },
        },
    }


def _constraint_schema() -> Dict[str, Any]:
    """One sub-schema covering all 8 constraint kinds.

    OpenAI's `response_format=json_schema` strict mode requires every key in
    `properties` to also appear in `required`; the trick to keep this single
    schema flexible across kinds is to make all non-`kind` fields nullable.
    The LLM emits null for whichever fields don't apply to the chosen kind.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "kind",
            "allowed_names", "position", "material_name",
            "forbidden_pairs", "min_nm", "max_nm",
            "min_layers", "max_layers", "name_a", "name_b",
            "max_total_nm", "markovian_decode", "match_thickness",
        ],
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "allowed_subset", "layer_identity", "adjacent_forbidden",
                    "thickness_range", "layer_count", "ordering_before",
                    "total_thickness", "symmetry",
                ],
            },
            # All possible params; only the ones relevant to `kind` are read.
            # Nullable so the LLM can emit null when a field doesn't apply.
            "allowed_names": {"type": ["array", "null"],
                              "items": {"type": "string"}},
            "position": {"type": ["integer", "null"]},
            "material_name": {"type": ["string", "null"]},
            "forbidden_pairs": {
                "type": ["array", "null"],
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2, "maxItems": 2,
                },
            },
            "min_nm": {"type": ["integer", "null"]},
            "max_nm": {"type": ["integer", "null"]},
            "min_layers": {"type": ["integer", "null"]},
            "max_layers": {"type": ["integer", "null"]},
            "name_a": {"type": ["string", "null"]},
            "name_b": {"type": ["string", "null"]},
            "max_total_nm": {"type": ["integer", "null"]},
            "markovian_decode": {"type": ["boolean", "null"]},
            "match_thickness": {"type": ["boolean", "null"]},
        },
    }


# ----------------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------------

def _system_prompt(pool_names: List[str]) -> str:
    """Concise system prompt. Lists the 8 constraint kinds and the pool, with
    explicit guidance on the underspecified case."""
    names_str = ", ".join(pool_names)
    return f"""You translate natural-language requests for thin-film optical design into
a structured spec. The user describes a color they want and any structural
constraints; you decide:

  1. Target color as CIE Lab [L*, a*, b*]. L*: 0-100, a*/b*: roughly -128..128.
     Map common color words to Lab using standard sRGB→Lab conversion. If the
     user gives Lab directly, use those values.

  2. Constraints from the 8 supported kinds. Output ONLY constraints the user
     actually requested or implied — DO NOT invent or defend defaults. An
     empty constraints list means "search the full space".

  3. A short disclaimer (≤ 2 sentences) summarising how you read the prompt.

Available materials (canonical names — use EXACTLY as written):
  {names_str}

Constraint kinds (use only these `kind` values):
  - allowed_subset      params: allowed_names: [str, ...]
                        Restricts the model to ONLY these materials.
  - layer_identity      params: position: int, material_name: str
                        Pins one layer position to one specific material.
  - adjacent_forbidden  params: forbidden_pairs: [[str, str], ...]
                        Pairs that may not appear at consecutive positions.
                        Pairs may be self-pairs (e.g. ["Ag","Ag"] = "no two
                        adjacent silver layers"), but ONLY use this when the
                        user is restricting *adjacency*, not global usage.
  - thickness_range     params: min_nm: int (PER-LAYER, 5..200),
                                max_nm: int (PER-LAYER, 5..200),
                                position: int|null (null = global)
                        Bounds for ONE individual layer (or every layer if
                        position is null). For a budget on the *sum* of
                        layer thicknesses use total_thickness instead.
  - layer_count         params: min_layers: int (1..10),
                                max_layers: int (1..10)
  - ordering_before     params: name_a: str, name_b: str
  - total_thickness     params: max_total_nm: int (5..2000),
                                markovian_decode: true|null
                        Budget on the SUM of all layer thicknesses. Use for
                        "no thicker than N nm in total" / "≤ N nm overall
                        stack". The field is `max_total_nm` — leave
                        `min_nm`/`max_nm` as null on this kind.
  - symmetry            params: match_thickness: true|null

Picking the right kind matters. Common phrasings:
  - "no X" / "without X" / "don't use X"
        ⇒ allowed_subset listing every pool name EXCEPT X. NOT
          adjacent_forbidden — that would only block X-next-to-X / X-next-to-Y.
  - "only X and Y"
        ⇒ allowed_subset: ["X", "Y"].
  - "X and Y can't be touching" / "no X-Y interface"
        ⇒ adjacent_forbidden: [["X", "Y"]].
  - "the bottom layer must be X" / "layer 0 is X"
        ⇒ layer_identity: position=0, material_name="X".
  - "between N and M layers" / "at most N layers"
        ⇒ layer_count.
  - "thinner than N nm everywhere"
        ⇒ thickness_range with position=null.
  - "X has to come before Y"
        ⇒ ordering_before.
  - "symmetric stack" / "palindromic"
        ⇒ symmetry.
  - "total stack thickness ≤ N nm" / "thinner than N nm overall"
        ⇒ total_thickness.

Periodic / positional patterns — these need MULTIPLE layer_identity entries,
one per fixed position. There is no "period" primitive; you must enumerate
positions explicitly.

  - "every other layer is X" / "alternating X with anything"
        ⇒ EMIT a layer_identity AT EACH EVEN POSITION (0, 2, 4, 6, 8)
          binding to X. Do NOT use allowed_subset:[X] — that forces EVERY
          layer to X, not every other. Also emit a layer_count constraint
          if the user said anything about how many layers.
        Example: "alternating ZnO" ⇒
          [{{kind:"layer_identity",position:0,material_name:"ZnO"}},
           {{kind:"layer_identity",position:2,material_name:"ZnO"}},
           {{kind:"layer_identity",position:4,material_name:"ZnO"}},
           {{kind:"layer_identity",position:6,material_name:"ZnO"}},
           {{kind:"layer_identity",position:8,material_name:"ZnO"}}]
  - "X then Y then X then Y …" (ABAB stack)
        ⇒ layer_identity at positions 0,2,4,… = X AND positions 1,3,5,… = Y.
  - "first and last layer must be X"
        ⇒ layer_identity at position 0 = X AND layer_identity at position
          (last_index) = X. You don't know the last index for sure; pin
          position 0 and ALSO add a symmetry constraint or emit
          layer_identity at the user-specified positions.
  - "layers 3 through 6 must be Cr"
        ⇒ layer_identity at each of 3, 4, 5, 6.
  - "the middle layer must be X"
        ⇒ layer_identity at position floor(N/2) where N comes from the
          layer_count the user requested; default to position 4 if no count
          was given.

When you enumerate positions, keep them inside [0, 10). If the user implied
a different total layer count via layer_count.max_layers=K, only enumerate
positions in [0, K).

Constants you MUST respect:
  - Layer positions are 0-indexed in [0, 10).
  - Layer count max = 10.
  - Thicknesses are integers in nm, multiples of 5, in [5, 200].
  - Material names MUST be from the list above; do NOT abbreviate or alias.

If the user names a material that is NOT in the list, leave the constraint
out and note it in the disclaimer.
If the user describes an impossible request (e.g. "emits light", a Lab value
outside the achievable gamut), still output your best-effort Lab target and
flag the concern in the disclaimer — the downstream pipeline will report
its actual achievable ΔE.

The ESCAPE HATCH: `custom_constraint_request`
---------------------------------------------
Keep this `null` for the OVERWHELMING majority of requests. The 8 existing
kinds + the periodic-pattern expansion above cover essentially everything
real users ask for.

ONLY populate `custom_constraint_request` with a short natural-language
description (1-3 sentences, plain English) when you have CONFIRMED that
no combination of the 8 standard kinds + position enumeration can
express what the user asked for. Examples of genuinely escape-hatch-only
requests:

  - "the sum of the THICKNESSES of all silver layers must not exceed 80 nm"
        (not a per-layer thickness range; not a total thickness; not
         allowed_subset — it's a conditional sum.)
  - "at least one of the layers must have thickness within 5 nm of 100 nm"
        (existence claim across the stack.)
  - "the cumulative thickness of TiO2 layers must equal the cumulative
     thickness of SiO2 layers"
        (parity between two material groups.)

When you DO populate it:
  - You may STILL also include any standard constraints in the `constraints`
    array — the custom one is added on top of them, not instead of them.
    Push as much as you can into the standard kinds.
  - Be MAXIMALLY EXPLICIT in the description. Name the materials by their
    canonical names. Quantify everything. Say what should pass and what
    should fail. The code-generation pass has only your description to
    work from.
  - In your `disclaimer`, ALWAYS mention that a custom constraint was
    requested and briefly say what it enforces, so the user sees that a
    second model call is happening.

If the user's request can be approximated reasonably by an existing
constraint, prefer the approximation and mention the trade-off in the
disclaimer — `custom_constraint_request` is a last resort.

Output JSON conforming to the schema. Do not output prose."""


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

def _call_openai(prompt: str, pool_names: List[str], model: str,
                 api_key: Optional[str] = None,
                 ) -> Dict[str, Any]:
    """Single forced-JSON OpenAI call. Returns the parsed JSON dict.

    Prefers the official `openai` SDK if it's importable; otherwise falls
    back to a `urllib.request` POST so this works on any env without an
    extra install. `api_key` if given overrides the env-var lookup — used
    when the frontend wants to forward a key the user pasted into the GUI
    so the server never persists it.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ParseError("backend",
                         "OPENAI_API_KEY not set. Export it, paste it into "
                         "the frontend's Advanced panel, or use "
                         "INDIGO_PARSE_BACKEND=mock.")
    try:
        from openai import OpenAI
        return _call_openai_sdk(OpenAI, prompt, pool_names, model, api_key)
    except ImportError:
        return _call_openai_urllib(prompt, pool_names, model, api_key)


def _call_openai_sdk(OpenAI, prompt: str, pool_names: List[str], model: str,
                     api_key: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema",
                         "json_schema": _response_json_schema()},
        messages=[
            {"role": "system", "content": _system_prompt(pool_names)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    if text is None:
        raise ParseError("backend", "OpenAI returned empty content")
    return json.loads(text)


def _call_openai_urllib(prompt: str, pool_names: List[str], model: str,
                        api_key: str) -> Dict[str, Any]:
    """Stdlib HTTP call to OpenAI's chat-completions endpoint.

    Lets parse.py run on any Python env without needing the `openai` SDK
    installed. The request body uses the same response_format=json_schema
    strict mode the SDK path uses. `api_key` is mandatory here — the
    caller (`_call_openai`) handles the env-var fallback.
    """
    import urllib.error
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    body = json.dumps({
        "model": model,
        "response_format": {"type": "json_schema",
                            "json_schema": _response_json_schema()},
        "messages": [
            {"role": "system", "content": _system_prompt(pool_names)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # Surface the response body so quota / model-name errors are legible.
        body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise ParseError(
            "backend",
            f"OpenAI HTTP {exc.code}: {body_text[:400]}"
        ) from exc
    except Exception as exc:
        raise ParseError(
            "backend", f"OpenAI call failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ParseError(
            "backend",
            f"OpenAI response shape unexpected: {json.dumps(payload)[:400]}"
        ) from exc
    if not text:
        raise ParseError("backend", "OpenAI returned empty content")
    return json.loads(text)


def _call_mock(prompt: str, pool_names: List[str]) -> Dict[str, Any]:
    """Tiny offline backend so tests and dev iteration don't burn tokens.

    Routes on a couple of keywords; everything else lands on neutral grey.
    The disclaimer makes it obvious to the user that the LLM wasn't called.
    """
    p = prompt.lower()
    if "red" in p:
        lab = [50.0, 60.0, 40.0]
    elif "blue" in p:
        lab = [40.0, 0.0, -60.0]
    elif "green" in p:
        lab = [60.0, -40.0, 30.0]
    elif "yellow" in p:
        lab = [85.0, 0.0, 80.0]
    elif "grey" in p or "gray" in p:
        lab = [60.0, 0.0, 0.0]
    else:
        lab = [60.0, 5.0, -8.0]
    return {
        "target_lab": lab,
        "constraints": [],
        "disclaimer": "MOCK backend (no LLM call). Routed on keywords only.",
    }


# ----------------------------------------------------------------------------
# Validation gates
# ----------------------------------------------------------------------------

# Which sub-fields each constraint kind actually uses. OpenAI's strict JSON
# schema mode forces the LLM to emit every property (with null for
# irrelevant ones), so we restrict our validation to the fields that the
# CHOSEN kind actually reads — otherwise a non-null leak on an irrelevant
# field (e.g. max_nm=250 alongside kind=total_thickness) would falsely fail
# the gate.
_KIND_RELEVANT_FIELDS: Dict[str, Tuple[str, ...]] = {
    "allowed_subset":      ("allowed_names",),
    "layer_identity":      ("position", "material_name"),
    "adjacent_forbidden":  ("forbidden_pairs",),
    "thickness_range":     ("min_nm", "max_nm", "position"),
    "layer_count":         ("min_layers", "max_layers"),
    "ordering_before":     ("name_a", "name_b"),
    "total_thickness":     ("max_total_nm", "markovian_decode"),
    "symmetry":            ("match_thickness",),
}


def _validate_semantic(spec_dict: Dict[str, Any],
                       pool: List[MaterialEntry]) -> None:
    """Gate 2: material names, layer positions, thickness grid, pool size.

    Only validates fields that the chosen constraint kind actually uses —
    avoids false errors from null-but-present fields that the strict JSON
    schema requires the LLM to emit alongside the ones it cares about.
    """
    target = spec_dict.get("target_lab")
    if not isinstance(target, list) or len(target) != 3:
        raise ParseError("semantic", f"target_lab not [L,a,b]: {target!r}")
    for x in target:
        if not isinstance(x, (int, float)):
            raise ParseError("semantic", f"target_lab has non-numeric {x!r}")

    if len(pool) > M_MAX:
        raise ParseError("semantic",
                         f"pool of {len(pool)} > M_MAX={M_MAX}; trim first")

    pool_names = {m.canonical_name for m in pool}
    unknown_names: List[str] = []
    constraints = spec_dict.get("constraints") or []
    for i, c in enumerate(constraints):
        kind = c.get("kind")
        relevant = _KIND_RELEVANT_FIELDS.get(kind, ())
        if not relevant:
            raise ParseError("schema", f"constraint {i}: unknown kind {kind!r}")

        # Material name fields — only check ones the kind actually uses.
        for key in ("material_name", "name_a", "name_b"):
            if key not in relevant:
                continue
            v = c.get(key)
            if isinstance(v, str) and v and v not in pool_names:
                unknown_names.append(v)
        if "allowed_names" in relevant:
            names_list = c.get("allowed_names")
            if isinstance(names_list, list):
                for v in names_list:
                    if v not in pool_names:
                        unknown_names.append(v)
        if "forbidden_pairs" in relevant:
            pairs = c.get("forbidden_pairs")
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        for v in pair:
                            if v not in pool_names:
                                unknown_names.append(v)

        # Position — only validate when this kind reads it.
        if "position" in relevant:
            pos = c.get("position")
            if pos is not None:
                if not isinstance(pos, int) or not (0 <= pos < MAX_LAYERS):
                    raise ParseError("semantic",
                                     f"constraint {i} ({kind}): position {pos} "
                                     f"out of [0, {MAX_LAYERS})")

        # Per-layer thickness fields. We accept values outside the [5, 200]
        # grid — they just become looser bounds (the downstream constraint
        # check is "min_nm <= t <= max_nm" over an integer t, and the
        # decode_mask iterates the actual grid so out-of-grid mins/maxes
        # naturally degenerate to "no extra restriction"). Type check only.
        for key in ("min_nm", "max_nm"):
            if key not in relevant:
                continue
            v = c.get(key)
            if v is None:
                continue
            if not isinstance(v, int):
                raise ParseError(
                    "semantic",
                    f"constraint {i} ({kind}): {key} not int (got {v!r})"
                )

        # Total-thickness budget. No grid; just sanity-bound to positive.
        if "max_total_nm" in relevant:
            v = c.get("max_total_nm")
            if v is not None and (not isinstance(v, int) or v < THICKNESSES[0]):
                raise ParseError(
                    "semantic",
                    f"constraint {i} ({kind}): max_total_nm must be int "
                    f"≥ {THICKNESSES[0]} (got {v!r})"
                )

        # Layer count
        for key in ("min_layers", "max_layers"):
            if key not in relevant:
                continue
            v = c.get(key)
            if v is None:
                continue
            if not isinstance(v, int) or not (1 <= v <= MAX_LAYERS):
                raise ParseError("semantic",
                                 f"constraint {i} ({kind}): {key}={v} not in "
                                 f"[1, {MAX_LAYERS}]")

    if unknown_names:
        unique = sorted(set(unknown_names))
        raise ParseError(
            "semantic",
            f"constraint references material(s) not in pool: {unique}. "
            f"Pool has {len(pool_names)} canonical names; check spelling "
            f"(canonical names include disambiguators like '-Rakic-LD-1998')."
        )


def _validate_physical(spec_dict: Dict[str, Any]) -> None:
    """Gate 3: contradictions, vacuous ranges, impossible combinations."""
    constraints = spec_dict.get("constraints") or []
    for i, c in enumerate(constraints):
        kind = c.get("kind")
        if kind == "allowed_subset":
            if not c.get("allowed_names"):
                raise ParseError("physical",
                                 f"constraint {i}: allowed_subset must "
                                 "list at least one material")
        elif kind == "layer_count":
            lo = c.get("min_layers", 1)
            hi = c.get("max_layers", MAX_LAYERS)
            if lo is None: lo = 1
            if hi is None: hi = MAX_LAYERS
            if lo > hi:
                raise ParseError("physical",
                                 f"constraint {i}: layer_count "
                                 f"min={lo} > max={hi}")
        elif kind == "thickness_range":
            lo = c.get("min_nm")
            hi = c.get("max_nm")
            if lo is not None and hi is not None and lo > hi:
                raise ParseError("physical",
                                 f"constraint {i}: thickness_range "
                                 f"min={lo} > max={hi}")
        elif kind == "ordering_before":
            a = c.get("name_a")
            b = c.get("name_b")
            if a is None or b is None or a == b:
                raise ParseError("physical",
                                 f"constraint {i}: ordering_before needs "
                                 f"distinct name_a/name_b")
        elif kind == "total_thickness":
            v = c.get("max_total_nm")
            if v is None or v < THICKNESSES[0]:
                raise ParseError("physical",
                                 f"constraint {i}: total_thickness max_total_nm "
                                 f"must be ≥ {THICKNESSES[0]}")
        elif kind == "adjacent_forbidden":
            pairs = c.get("forbidden_pairs") or []
            for pair in pairs:
                if len(pair) != 2:
                    raise ParseError(
                        "physical",
                        f"constraint {i}: adjacent_forbidden pair {pair} "
                        f"must be exactly two names"
                    )
                # Self-pairs (e.g. ("Ag", "Ag") meaning "no two adjacent Ag
                # layers") are legitimate — `check` / `decode_mask` handle
                # them correctly. Only the empty / wrong-arity cases are
                # rejected here.


# ----------------------------------------------------------------------------
# Spec assembly
# ----------------------------------------------------------------------------

def _build_constraints(spec_dict: Dict[str, Any]) -> list:
    """Convert validated JSON constraints → Constraint subclass instances.

    Mirrors the kind dispatch in run_inference.py's `_build_constraints_from_json`
    but tolerates the null params our schema allows.
    """
    out = []
    for c in spec_dict.get("constraints") or []:
        kind = c["kind"]
        if kind == "allowed_subset":
            out.append(AllowedSubset(allowed_names=tuple(c["allowed_names"])))
        elif kind == "layer_identity":
            out.append(LayerIdentity(position=int(c["position"]),
                                     material_name=str(c["material_name"])))
        elif kind == "adjacent_forbidden":
            pairs = tuple(tuple(p) for p in c["forbidden_pairs"])
            out.append(AdjacentForbidden(forbidden_pairs=pairs))
        elif kind == "thickness_range":
            out.append(ThicknessRange(
                min_nm=int(c.get("min_nm") or 5),
                max_nm=int(c.get("max_nm") or MAX_THICKNESS_NM),
                position=c.get("position"),
            ))
        elif kind == "layer_count":
            out.append(LayerCount(
                min_layers=int(c.get("min_layers") or 1),
                max_layers=int(c.get("max_layers") or MAX_LAYERS),
            ))
        elif kind == "ordering_before":
            out.append(OrderingBefore(
                name_a=str(c["name_a"]), name_b=str(c["name_b"]),
            ))
        elif kind == "total_thickness":
            out.append(TotalThickness(
                max_total_nm=int(c["max_total_nm"]),
                markovian_decode=bool(c.get("markovian_decode")
                                      if c.get("markovian_decode") is not None
                                      else True),
            ))
        elif kind == "symmetry":
            out.append(Symmetry(match_thickness=bool(
                c.get("match_thickness") if c.get("match_thickness") is not None
                else True
            )))
        else:
            raise ParseError("schema", f"unknown constraint kind {kind!r}")
    return out


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

@dataclass
class ParseResult:
    spec: InferenceSpec
    raw_response: Dict[str, Any]


def parse_prompt(
    prompt: str,
    pool: List[MaterialEntry],
    knobs: Optional[InferenceKnobs] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ParseResult:
    """Free-text prompt + pool → validated InferenceSpec.

    Backend selection (in order):
      - explicit `backend` argument
      - $INDIGO_PARSE_BACKEND env var ("openai" | "mock")
      - default: "openai"

    `api_key` if given overrides the env-var key — used by the frontend
    so a key the user pasted into the GUI never persists on the server.

    Raises ParseError on any of the three validation gates.
    """
    knobs = knobs or InferenceKnobs()
    backend = backend or os.environ.get("INDIGO_PARSE_BACKEND", "openai")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    pool_names = [m.canonical_name for m in pool]

    if backend == "mock":
        spec_dict = _call_mock(prompt, pool_names)
    elif backend == "openai":
        spec_dict = _call_openai(prompt, pool_names, model, api_key=api_key)
    else:
        raise ParseError("backend", f"unknown backend {backend!r}")

    # Gate 1 was the response_format. Gates 2 + 3:
    _validate_semantic(spec_dict, pool)
    _validate_physical(spec_dict)

    target_raw = tuple(float(x) for x in spec_dict["target_lab"])
    norm = tuple(float(x) for x in normalize_lab(list(target_raw)).tolist())

    constraints = _build_constraints(spec_dict)
    disclaimer = str(spec_dict.get("disclaimer") or "")

    # ---- Optional second call: LLM-authored custom constraint --------------
    # Triggered only when the first-call response populated the escape-hatch
    # field. The standard constraints stay (the LLM is encouraged to include
    # both); the custom one is appended on top.
    custom_req = spec_dict.get("custom_constraint_request") or ""
    custom_req = custom_req.strip() if isinstance(custom_req, str) else ""
    if custom_req and backend == "openai":
        from inference.src.custom_constraint import (
            CustomConstraintError, generate_custom_constraint,
        )
        try:
            custom = generate_custom_constraint(
                description=custom_req, pool=pool,
                api_key=api_key, model=model,
            )
            constraints.append(custom)
            # Loud, unambiguous note in the disclaimer. The frontend ALSO
            # surfaces this in a dedicated banner + source-code drawer, but
            # keep the disclaimer self-contained so anyone reading the spec
            # JSON sees what happened without the UI.
            disclaimer = (
                disclaimer
                + "\n\n⚠ CUSTOM CONSTRAINT CODE GENERATED — the LLM wrote "
                  "and sandboxed Python for this request because no built-in "
                  "constraint kind expressed it.\n"
                + f"  request:    {custom_req}\n"
                + f"  class name: {custom.class_name}\n"
                + "  source:     see spec_echo.constraints[*].source_code "
                  "(or the Custom Constraints panel in the UI)."
            ).strip()
        except CustomConstraintError as exc:
            # Expected failure path — fall back to the standard constraints
            # the first call produced, surface the gate + message.
            print(f"[parse] custom constraint failed: {exc}", flush=True)
            tail = ("\n\n⚠ CUSTOM CONSTRAINT REQUESTED BUT SKIPPED — "
                    f"codegen gate `{exc.gate}` rejected the request: "
                    f"{exc.message}")
            if exc.source:
                tail += (f"\n  Generated source (first 800 chars):\n"
                         f"{exc.source[:800]}")
            disclaimer = (disclaimer + tail).strip()
        except Exception as exc:
            # Belt-and-suspenders. If anything escapes the
            # CustomConstraintError wrapping (it shouldn't — codegen now
            # converts everything), surface it loudly here rather than
            # 500ing the whole request.
            import traceback as _tb
            print(f"[parse] custom constraint UNEXPECTED failure: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            _tb.print_exc()
            disclaimer = (
                disclaimer
                + "\n\n⚠ CUSTOM CONSTRAINT REQUESTED BUT SKIPPED — "
                  f"unexpected {type(exc).__name__}: {exc}"
            ).strip()

    spec = InferenceSpec(
        target_lab_raw=target_raw,
        target_lab_normalised=norm,
        constraints=constraints,
        enforce_during=[],
        enforce_post=[],
        knobs=knobs,
        parsed_disclaimer=disclaimer,
    )
    return ParseResult(spec=spec, raw_response=spec_dict)


# ----------------------------------------------------------------------------
# Self-test (mock backend only — no API key needed)
# ----------------------------------------------------------------------------

def _self_test() -> None:
    import numpy as np
    pool = [
        MaterialEntry(canonical_name="Ag-Rakic-LD-1998",
                      n=np.ones(128, dtype=np.float32),
                      k=np.zeros(128, dtype=np.float32)),
        MaterialEntry(canonical_name="SiO2-Zarei-2024",
                      n=np.full(128, 1.45, dtype=np.float32),
                      k=np.zeros(128, dtype=np.float32)),
    ]

    # Mock backend resolves "blue" → known Lab; empty constraints.
    out = parse_prompt("Make me a blue structure",
                       pool=pool, backend="mock")
    assert out.spec.target_lab_raw == (40.0, 0.0, -60.0)
    assert out.spec.constraints == []
    assert "MOCK" in out.spec.parsed_disclaimer
    print(f"[parse] mock 'blue': Lab={out.spec.target_lab_raw}  "
          f"constraints=0  disclaimer='{out.spec.parsed_disclaimer[:40]}...'")

    # Hand-craft a spec_dict to exercise validation gates without an LLM.
    spec_dict = {
        "target_lab": [60.0, 5.0, -8.0],
        "constraints": [
            {"kind": "allowed_subset",
             "allowed_names": ["Ag-Rakic-LD-1998", "SiO2-Zarei-2024"]},
            {"kind": "layer_count", "min_layers": 2, "max_layers": 6},
            {"kind": "thickness_range", "min_nm": 20, "max_nm": 150},
        ],
        "disclaimer": "hand-crafted",
    }
    _validate_semantic(spec_dict, pool)
    _validate_physical(spec_dict)
    cs = _build_constraints(spec_dict)
    assert len(cs) == 3
    print(f"[parse] valid spec built {len(cs)} constraints")

    # Semantic gate: unknown material
    bad = {
        "target_lab": [60.0, 0.0, 0.0],
        "constraints": [{"kind": "allowed_subset",
                         "allowed_names": ["Ag", "Au"]}],   # Au not in pool
        "disclaimer": "",
    }
    try:
        _validate_semantic(bad, pool)
        assert False, "expected semantic gate to fire"
    except ParseError as exc:
        assert exc.gate == "semantic"
        print(f"[parse] semantic gate caught unknown name: {exc.message[:60]}...")

    # Physical gate: min > max layers
    bad = {
        "target_lab": [60.0, 0.0, 0.0],
        "constraints": [{"kind": "layer_count",
                         "min_layers": 8, "max_layers": 3}],
        "disclaimer": "",
    }
    try:
        _validate_physical(bad)
        assert False, "expected physical gate to fire"
    except ParseError as exc:
        assert exc.gate == "physical"
        print(f"[parse] physical gate caught min>max: {exc.message[:60]}...")

    print("[parse] self-test OK")


if __name__ == "__main__":
    _self_test()
