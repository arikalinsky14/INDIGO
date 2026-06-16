"""LLM-authored custom constraints — escape hatch from the 8-kind library.

  natural-language request the 8 pre-coded constraints can't express
                ─►   second LLM call generates Python source
                ─►   sandboxed exec → CustomConstraint subclass
                ─►   attached to the InferenceSpec like any other constraint

Use sparingly. The two-call parser in `parse.py` routes most requests
through the standard vocabulary; only when the first call decides the
request truly needs a one-off does the code-generation second call fire.

Trust model
-----------
The trust model in `parse.py`'s docstring used to say "we do not execute
LLM-authored code". This module is the explicit exception. To minimise
the blast radius:

  1. **Default to existing constraints.** The first-call system prompt
     pushes the LLM hard to use the 8 kinds + the periodic-pattern
     expansion. The escape hatch only triggers when the LLM explicitly
     populates `custom_constraint_request`.
  2. **Restricted exec namespace.** `__builtins__` is replaced with a
     curated dict (no `open`, `import`, `eval`, `exec`, `__import__`,
     `compile`, `globals`, `locals`, `vars`). The only pre-injected
     names are the Constraint ABC, dataclass helpers, `numpy as np`, and
     the vocab/token layout helpers from `constraints.py`.
  3. **Static substring screen.** Before we even `compile()` the source,
     we reject obvious sandbox-escape attempts (`__import__`, `subprocess`,
     dunder attribute access, etc.) with a useful error.
  4. **Timeout on every method call.** `check()` and `decode_mask()` run
     in a daemon thread with a 5 s wall-clock cap. Timeout → returns
     "constraint not satisfied" so the orchestrator simply drops the
     candidate rather than hanging.
  5. **Source code captured.** The generated source is stored on the
     CustomConstraint instance (`.source_code`) and round-trips through
     the Result envelope so the user can audit what ran.

Feature flag
------------
`INDIGO_ALLOW_CUSTOM_CONSTRAINTS=0` disables this entirely; the parser
raises if the LLM tries to invoke the escape hatch. Default is on so
that the hosted demo works.

What it CANNOT defend against
-----------------------------
This is not a real sandbox. A determined adversary with arbitrary prompt
control could escape (e.g. via dataclass metaclass tricks or numpy
weirdness). It IS hardened enough that an OpenAI-grade LLM trying to
satisfy a benign user request won't trigger anything destructive. Treat
this as defence in depth for a single-tenant demo, not as a security
boundary against hostile inputs.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.materials_vocab import (
    EOS_TOKEN, M_MAX, MAX_LAYERS, NUM_THICKNESSES, THICKNESSES, VOCAB_SIZE,
)

from inference.src.constraints import (
    PartialStructure, FinishedStructure,
    _all_allowed_mask, _mask_slot, _slot_token_range,
)
from inference.src.schema import Constraint, MaterialEntry


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------

class CustomConstraintError(RuntimeError):
    """Anything wrong with an LLM-authored constraint.

    `gate` lets the caller surface the failure cleanly:
      - "disabled"  : feature flag off
      - "generate"  : second LLM call failed (no API key / empty response)
      - "exec"      : source rejected by static screen or compile / exec
                      raised
      - "validate"  : no Constraint subclass defined, or smoke check()
                      failed
    """
    def __init__(self, gate: str, message: str, source: str = ""):
        super().__init__(f"[custom:{gate}] {message}")
        self.gate = gate
        self.message = message
        self.source = source


# ----------------------------------------------------------------------------
# Restricted exec environment
# ----------------------------------------------------------------------------

def _resolve_builtin(name: str) -> Any:
    """Pull a builtin by name whether __builtins__ is a dict or a module
    (depends on how this module was imported)."""
    b = __builtins__
    if isinstance(b, dict):
        return b[name]
    return getattr(b, name)


_SAFE_BUILTIN_NAMES = (
    # Class-statement plumbing. Without __build_class__, no `class X(...):`
    # statement works at all — Python compiles `class …` to a call to this
    # builtin internally. Same for __name__ (every class needs a module
    # name) and the meta-class helpers super / type / staticmethod /
    # classmethod / property that the dataclass decorator can touch.
    "__build_class__", "__name__",
    "super", "staticmethod", "classmethod", "property",
    # Core type constructors + introspection
    "abs", "all", "any", "bool", "callable", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip", "hash", "id", "format", "ord", "chr",
    "True", "False", "None",
    # Exception types — the LLM should not raise, but if it does we'd
    # rather see a Python exception than an opaque NameError.
    "Exception", "ValueError", "TypeError", "AttributeError", "KeyError",
    "IndexError",
)

_SAFE_BUILTINS = {n: _resolve_builtin(n) for n in _SAFE_BUILTIN_NAMES}

# Forbidden substrings — cheap static screen. Real defence is the
# namespace, but rejecting obvious cases up front gives the LLM (and the
# audit log) a useful error message instead of a runtime NameError.
_FORBIDDEN_SUBSTRINGS = (
    "__import__", "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__",
    "open(", "exec(", "eval(", "compile(",
    "globals(", "locals(", "vars(", "getattr(", "setattr(", "delattr(",
    "subprocess", "os.system", "os.popen",
    "socket", "urllib", "requests", "httpx",
)


# Stub module for the sandbox. The @dataclass decorator resolves
# annotations through `sys.modules[cls.__module__].__dict__` (for forward
# refs / typing.get_type_hints fall-throughs). If the module name isn't
# registered, sys.modules returns None and we crash with
#   AttributeError: 'NoneType' object has no attribute '__dict__'
# during class creation. Register an empty stub once at import time so
# every sandbox class has a real module to live under.
_SANDBOX_MODULE_NAME = "indigo_custom_constraint_sandbox"
if _SANDBOX_MODULE_NAME not in sys.modules:
    sys.modules[_SANDBOX_MODULE_NAME] = types.ModuleType(_SANDBOX_MODULE_NAME)


def _make_exec_namespace() -> Dict[str, Any]:
    """Pre-populated namespace for the LLM's code."""
    return {
        "__builtins__": dict(_SAFE_BUILTINS),
        # Class-statement metadata. Python's compiler emits a reference to
        # the module's __name__ when defining a class; if it's missing the
        # @dataclass decorator raises during class creation.
        "__name__": _SANDBOX_MODULE_NAME,
        # Required ABC + dataclass helpers (the LLM can't `import` them).
        "Constraint": Constraint,
        "dataclass": dataclass,
        "field": field,
        # Structure snapshots
        "PartialStructure": PartialStructure,
        "FinishedStructure": FinishedStructure,
        "MaterialEntry": MaterialEntry,
        # Vocab + token helpers
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LAYERS": MAX_LAYERS,
        "M_MAX": M_MAX,
        "NUM_THICKNESSES": NUM_THICKNESSES,
        "EOS_TOKEN": EOS_TOKEN,
        "THICKNESSES": THICKNESSES,
        "_slot_token_range": _slot_token_range,
        "_all_allowed_mask": _all_allowed_mask,
        "_mask_slot": _mask_slot,
        # numpy
        "np": np,
        # Common typing names; harmless and the LLM tends to use them.
        "Optional": Optional,
        "List": List,
        "Tuple": Tuple,
        "Dict": Dict,
    }


def _static_safety_check(source: str) -> None:
    for s in _FORBIDDEN_SUBSTRINGS:
        if s in source:
            raise CustomConstraintError(
                "exec", f"forbidden token in source: {s!r}", source=source,
            )


# ----------------------------------------------------------------------------
# Timeout wrapper
# ----------------------------------------------------------------------------

def _run_with_timeout(fn: Callable[[], Any], timeout_s: float, default: Any
                      ) -> Any:
    """Run `fn` on a daemon thread with a wall-clock timeout.

    The thread is NOT killed on timeout (Python threads aren't killable);
    we stop waiting and return `default`. Acceptable for a research demo
    — the worst case is the rogue thread sits CPU-bound for a while.
    """
    result: List[Any] = [default]
    raised: List[Optional[BaseException]] = [None]

    def runner():
        try:
            result[0] = fn()
        except BaseException as exc:    # noqa: BLE001 — we WANT every error
            raised[0] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return default
    if raised[0] is not None:
        return default
    return result[0]


# ----------------------------------------------------------------------------
# CustomConstraint — what gets attached to the InferenceSpec
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CustomConstraint(Constraint):
    """Wraps the LLM-authored Constraint subclass instance.

    Frozen because the parent `Constraint` is frozen (every concrete
    constraint in `constraints.py` is). Dataclass fields round-trip
    through `_strip_arrays(asdict(...))` into the Result envelope; the
    runtime `_instance` and `_check_timeout_s` live as plain attributes
    set via `object.__setattr__()` so they don't leak through asdict.
    """
    kind: str = "custom"
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source_code: str = ""
    class_name: str = ""

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        inst = getattr(self, "_instance", None)
        if inst is None:
            return False
        timeout = getattr(self, "_check_timeout_s", 5.0)
        return bool(_run_with_timeout(
            lambda: bool(inst.check(fs, pool)),
            timeout_s=timeout,
            default=False,
        ))

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        inst = getattr(self, "_instance", None)
        if inst is None or not hasattr(inst, "decode_mask"):
            return None
        timeout = getattr(self, "_check_timeout_s", 5.0)
        out = _run_with_timeout(
            lambda: inst.decode_mask(p, pool),
            timeout_s=timeout,
            default=None,
        )
        if out is None:
            return None
        # Must be a bool array of length VOCAB_SIZE.
        try:
            arr = np.asarray(out, dtype=bool)
        except Exception:
            return None
        if arr.shape != (VOCAB_SIZE,):
            return None
        return arr


# ----------------------------------------------------------------------------
# Compile + validate
# ----------------------------------------------------------------------------

def _find_constraint_subclass(ns: Dict[str, Any]) -> Optional[type]:
    for v in ns.values():
        if (isinstance(v, type) and v is not Constraint
                and issubclass(v, Constraint)):
            return v
    return None


_FROZEN_REPAIR_RE = re.compile(r"@dataclass\b(?!\s*\()")


def _maybe_repair_frozen(source: str) -> str:
    """If the LLM wrote `@dataclass` (no args), upgrade to `@dataclass(frozen=True)`.

    The parent Constraint is `@dataclass(frozen=True)`, and Python refuses
    to let a non-frozen dataclass inherit from a frozen one. The codegen
    prompt now tells the LLM about this explicitly, but the older
    `@dataclass` form is the most common LLM slip — repair it
    transparently rather than surfacing a confusing TypeError.
    """
    return _FROZEN_REPAIR_RE.sub("@dataclass(frozen=True)", source)


def _compile_source(source: str) -> Tuple[type, Dict[str, Any]]:
    _static_safety_check(source)
    repaired = _maybe_repair_frozen(source)
    ns = _make_exec_namespace()
    # Mirror the namespace into the stub module's __dict__ so any tool
    # that walks `sys.modules[cls.__module__].__dict__` (the @dataclass
    # decorator's annotation resolver does this for forward refs) sees the
    # same names the exec sees. Cleared on each call so leftover state
    # from a previous custom constraint can't leak in.
    sandbox_mod = sys.modules[_SANDBOX_MODULE_NAME]
    sandbox_mod.__dict__.clear()
    sandbox_mod.__dict__.update(ns)
    try:
        compiled = compile(repaired, "<custom_constraint>", "exec")
    except SyntaxError as exc:
        raise CustomConstraintError("exec", f"SyntaxError: {exc}",
                                    source=source)
    try:
        # exec into the stub module's __dict__ directly (it already
        # contains the prepared namespace). This makes the class's
        # __module__ resolve to the stub module without any extra hop.
        exec(compiled, sandbox_mod.__dict__)
        ns = sandbox_mod.__dict__
    except Exception as exc:
        raise CustomConstraintError(
            "exec", f"{type(exc).__name__}: {exc}", source=source,
        )
    cls = _find_constraint_subclass(ns)
    if cls is None:
        raise CustomConstraintError(
            "validate",
            "no Constraint subclass found in the source. Define a "
            "@dataclass(frozen=True) subclass of Constraint with a check() method.",
            source=source,
        )
    return cls, ns


def _smoke_test_instance(inst: Any, pool: List[MaterialEntry]) -> None:
    """Call check() on a small synthetic structure to catch obvious errors.

    Doesn't validate semantics — just that the call doesn't blow up and
    returns a bool within the timeout.
    """
    if len(pool) == 0:
        return
    L = min(3, len(pool))
    fake = FinishedStructure(
        slot_indices=list(range(L)),
        thicknesses_nm=[20, 30, 40][:L],
        pool_size=len(pool),
    )
    raised: List[Optional[BaseException]] = [None]

    def call():
        try:
            return inst.check(fake, pool)
        except BaseException as exc:    # noqa: BLE001
            raised[0] = exc
            return None

    out = _run_with_timeout(call, timeout_s=5.0, default="__TIMEOUT__")
    if raised[0] is not None:
        raise CustomConstraintError(
            "validate",
            f"smoke check() raised: "
            f"{type(raised[0]).__name__}: {raised[0]}",
        )
    if out == "__TIMEOUT__":
        raise CustomConstraintError(
            "validate", "smoke check() exceeded the 5 s timeout",
        )
    if not isinstance(out, (bool, np.bool_)):
        raise CustomConstraintError(
            "validate",
            f"check() returned {type(out).__name__!r}, expected bool",
        )


# ----------------------------------------------------------------------------
# Generation — the second LLM call
# ----------------------------------------------------------------------------

def _generate_system_prompt(pool_names: List[str]) -> str:
    return f"""You are writing Python source for a thin-film optical design
constraint that the standard 8-kind library cannot express. Your output is
executed in a restricted sandbox.

Output ONLY Python source code. No prose, no markdown fences, no
explanation. The text you produce is `exec()`d directly.

The sandbox pre-populates these names. You MAY use them; you may NOT
import anything else, and you may NOT use open / eval / exec / __import__
or any dunder-attribute trickery:

  np                      numpy module — array ops only
  Constraint              base class to subclass
  dataclass, field        for the @dataclass decorator
  PartialStructure        attributes:
                            .slots_so_far          : list[int]
                            .thicknesses_so_far    : list[int]
                            .pool_size             : int
                            .step                  : int   (= len(slots_so_far))
                            .total_thickness_so_far: int
  FinishedStructure       attributes:
                            .slot_indices  : list[int]   (length = number of layers)
                            .thicknesses_nm: list[int]   (parallel to slot_indices)
                            .pool_size     : int
  MaterialEntry           attributes:
                            .canonical_name : str
                            .n, .k          : np.ndarray (refractive index spectra)
  VOCAB_SIZE              total vocabulary size
  MAX_LAYERS              maximum number of layers in a stack
  M_MAX                   maximum pool size (= 32)
  NUM_THICKNESSES         number of thickness tokens per slot
  EOS_TOKEN               vocab id for end-of-stack
  THICKNESSES             tuple of allowed thicknesses in nm (5 nm grid)
  _slot_token_range(s)    -> (lo, hi) inclusive lo, exclusive hi of token
                              ids belonging to slot s
  _all_allowed_mask()     -> np.bool array of length VOCAB_SIZE, all True
  _mask_slot(mask, s, allow)  in-place; flip slot s on (allow=True) or off

Available materials in the pool (canonical names — match EXACTLY):
  {", ".join(pool_names)}

Output structure (your code must define exactly ONE @dataclass that
subclasses Constraint):

  @dataclass(frozen=True)
  class MyCustomConstraint(Constraint):
      kind: str = "custom"
      params: Dict[str, object] = field(default_factory=dict)
      # any extra fields with defaults

      def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
          # Return True if `fs` satisfies the constraint, False otherwise.
          # `pool[s].canonical_name` is the material name at slot s.
          # `fs.slot_indices[i]` is the slot used at layer i,
          # `fs.thicknesses_nm[i]` its thickness in nm.
          ...

  # Optional. Default to omitting it (post-hoc check is fine).
      def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]):
          return None

Hard rules
----------
- Must be a @dataclass(frozen=True) and inherit from Constraint. The
  parent is frozen, so the child MUST also be frozen — `@dataclass`
  alone (without `frozen=True`) will raise TypeError at class-creation
  time.
- check() MUST return a Python bool (True or False).
- check() MUST be O(layers). No nested loops over the thickness grid.
- Do NOT raise exceptions for unexpected inputs; return False instead.
- Do NOT mutate `pool`, `fs`, or `p`.
- Do NOT print, log, or perform any I/O.
- Use ONLY the pre-populated names above. The sandbox blocks `import`,
  `open`, `exec`, `eval`, `__import__`, `subprocess`, and dunder
  attribute access. The source is rejected before execution if any of
  those tokens appear.

If the request is unimplementable under these rules, define a class whose
check() always returns False and put a brief reason in a class attribute
called `reason`.

Output ONLY the source code. Do not wrap it in ```python or ``` fences.
"""


def _strip_code_fences(text: str) -> str:
    """Remove ``` / ```python fences the LLM sometimes adds despite the prompt."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+\-]*\s*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _call_codegen_openai(
    description: str, pool_names: List[str], api_key: str, model: str,
) -> str:
    """Single OpenAI call returning the raw source string."""
    sys_prompt = _generate_system_prompt(pool_names)
    user_prompt = f"Constraint to implement:\n\n{description}\n"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except ImportError:
        # urllib fallback so this works in envs without the openai SDK.
        import urllib.request
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.2,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"] or ""


def generate_custom_constraint(
    description: str,
    pool: List[MaterialEntry],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> CustomConstraint:
    """description → executable CustomConstraint.

    Raises CustomConstraintError on any gate failure (disabled / no key /
    bad source / no class found / smoke test failed).
    """
    if os.environ.get("INDIGO_ALLOW_CUSTOM_CONSTRAINTS", "1") == "0":
        raise CustomConstraintError(
            "disabled",
            "custom constraints disabled "
            "(INDIGO_ALLOW_CUSTOM_CONSTRAINTS=0).",
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise CustomConstraintError(
            "generate",
            "OPENAI_API_KEY required for custom-constraint codegen.",
        )
    model = model or os.environ.get("INDIGO_CUSTOM_MODEL", "gpt-4o-mini")

    pool_names = [m.canonical_name for m in pool]
    print(f"[custom] codegen call: model={model}  "
          f"description={description!r}", flush=True)

    # Any unexpected exception from the network / SDK / json parsing path
    # below must become a CustomConstraintError so the parser's outer
    # except can fall back to standard constraints instead of leaking a
    # 500. Wrap the whole network section.
    try:
        raw = _call_codegen_openai(description, pool_names, api_key, model)
    except CustomConstraintError:
        raise
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        raise CustomConstraintError(
            "generate",
            f"codegen call failed: {type(exc).__name__}: {exc}",
        )

    source = _strip_code_fences(raw)
    if not source:
        raise CustomConstraintError("generate", "LLM returned empty code.")

    cls, _ns = _compile_source(source)

    try:
        instance = cls()
    except Exception as exc:
        raise CustomConstraintError(
            "validate",
            f"cannot construct {cls.__name__}() with defaults: "
            f"{type(exc).__name__}: {exc}",
            source=source,
        )
    _smoke_test_instance(instance, pool)

    custom = CustomConstraint(
        kind=str(getattr(instance, "kind", "custom")),
        params={"class_name": cls.__name__},
        description=description,
        source_code=source,
        class_name=cls.__name__,
    )
    # Live instance + timeout: plain attrs, NOT dataclass fields, so they
    # don't leak through asdict() into the Result JSON. object.__setattr__
    # bypasses the frozen-dataclass guard.
    object.__setattr__(custom, "_instance", instance)
    object.__setattr__(custom, "_check_timeout_s", 5.0)
    print(f"[custom] generated {cls.__name__} "
          f"({len(source)} bytes of source)", flush=True)
    return custom


# ----------------------------------------------------------------------------
# Self-test (no LLM call — exec a hand-written constraint through the sandbox)
# ----------------------------------------------------------------------------

def _self_test() -> None:
    pool = [
        MaterialEntry(canonical_name="ZnO",
                      n=np.ones(128, dtype=np.float32),
                      k=np.zeros(128, dtype=np.float32)),
        MaterialEntry(canonical_name="SiO2",
                      n=np.full(128, 1.45, dtype=np.float32),
                      k=np.zeros(128, dtype=np.float32)),
    ]
    # Hand-written source that mimics what the LLM would produce.
    src = """
@dataclass
class ThirdLayerZnO(Constraint):
    kind: str = "custom"
    params: Dict[str, object] = field(default_factory=dict)
    def check(self, fs, pool):
        if len(fs.slot_indices) < 3:
            return False
        return pool[fs.slot_indices[2]].canonical_name == "ZnO"
"""
    cls, _ns = _compile_source(src)
    inst = cls()
    fs_pass = FinishedStructure(slot_indices=[1, 1, 0],
                                thicknesses_nm=[10, 10, 10], pool_size=2)
    fs_fail = FinishedStructure(slot_indices=[1, 1, 1],
                                thicknesses_nm=[10, 10, 10], pool_size=2)
    assert inst.check(fs_pass, pool) is True
    assert inst.check(fs_fail, pool) is False
    print("[custom] sandbox exec OK")

    # Static-screen rejects obvious escape attempts.
    try:
        _compile_source("import os\n")
    except CustomConstraintError as e:
        assert e.gate == "exec"
        print(f"[custom] static screen OK: {e}")

    # CustomConstraint wrapper + timeout works.
    wrapper = CustomConstraint(
        kind="custom", description="3rd layer must be ZnO",
        source_code=src, class_name="ThirdLayerZnO",
    )
    object.__setattr__(wrapper, "_instance", inst)
    object.__setattr__(wrapper, "_check_timeout_s", 5.0)
    assert wrapper.check(fs_pass, pool) is True
    assert wrapper.check(fs_fail, pool) is False
    print("[custom] wrapper round-trip OK")


if __name__ == "__main__":
    _self_test()
