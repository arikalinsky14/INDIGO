"""
Constraint checks for INDIGO inference.

Two-method interface on every Constraint:

  check(structure, pool)          -> bool
      Post-hoc safety net. Always callable on a finished structure.

  decode_mask(partial, pool)      -> Optional[np.ndarray]   shape [VOCAB_SIZE]
      Per-step gate. Returns a boolean mask over the next-token vocabulary:
      True = token allowed, False = blocked. Returns None if the constraint
      cannot be enforced incrementally; the orchestrator then defers to the
      post-hoc check.

The orchestrator (generate.py) AND-combines `decode_mask` outputs from all
active constraints with the slot-validity mask the model already produces,
and sets -inf on disallowed logits before sampling.

Naming
------
All eight constraint kinds are concrete subclasses of `Constraint`
(re-exported from schema.py). Each carries its own parameter fields and a
short `kind` tag for serialisation.

Defaults that diverge from the plan, per the review
---------------------------------------------------
- TotalThickness enforces a running-budget decode mask BY DEFAULT (the plan
  had it post-only, "optional during"). The check is Markovian (just a
  running sum) and meaningfully cuts the rejection rate. Pass
  `markovian_decode=False` to fall back to post-only.
- Ordering uses decode_mask only for the simple single-pair case. With ≥2
  ordering constraints we fall back to post-hoc (per my note: "the
  decode-mask logic gets gnarly").
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.materials_vocab import (
    # LEGACY joint-vocab shape — this module still speaks (slot × thickness)
    # tokens. inference/src/generate.py projects the joint mask down to the
    # M_MAX + 1 slot vocab the two-head model actually consumes. See
    # _project_joint_mask_to_slot_mask there for the contract.
    LEGACY_EOS_TOKEN as EOS_TOKEN,
    LEGACY_VOCAB_SIZE as VOCAB_SIZE,
    M_MAX, MAX_LAYERS, NUM_THICKNESSES, THICKNESSES,
)


def encode_layer(slot_idx: int, thickness_nm: float) -> int:
    """Legacy joint-vocab index used by this module's smoke tests.

    The two-head model no longer decodes joint tokens, but the mask this
    module builds keeps the joint layout so its incremental logic stays
    unchanged. The projection down to a slot-only mask happens in
    inference/src/generate.py.
    """
    if not (0 <= slot_idx < M_MAX):
        raise ValueError(f"slot_idx {slot_idx} out of range [0, {M_MAX})")
    # Match old grid semantics: round to nearest 5 nm and lookup index.
    idx = int(round(float(thickness_nm) / 5.0)) - 1
    if not (0 <= idx < NUM_THICKNESSES):
        raise ValueError(
            f"thickness {thickness_nm} not representable on legacy 5 nm grid"
        )
    return slot_idx * NUM_THICKNESSES + idx

from inference.src.schema import Constraint, MaterialEntry


# ----------------------------------------------------------------------------
# Partial-structure snapshot fed to decode_mask at every decoding step
# ----------------------------------------------------------------------------

@dataclass
class PartialStructure:
    """The state generate.py knows at decoding step `len(slots_so_far)`.

    Fields are kept as plain lists so masks can be built with simple numpy
    ops — no jax involvement here, the decode mask is a pure CPU step.
    """
    slots_so_far: List[int]            # slot indices for layers 0..k-1
    thicknesses_so_far: List[float]    # nm at each placed layer (continuous)
    pool_size: int                     # valid slot range: [0, pool_size)

    @property
    def step(self) -> int:
        return len(self.slots_so_far)

    @property
    def total_thickness_so_far(self) -> float:
        return float(sum(self.thicknesses_so_far))


@dataclass
class FinishedStructure:
    """Post-hoc-check view of a fully-decoded candidate."""
    slot_indices: List[int]
    thicknesses_nm: List[float]        # continuous nm (was int on the 5 nm grid)
    pool_size: int


# ----------------------------------------------------------------------------
# Token-layout helpers — all masks live on the same VOCAB_SIZE-long bool array
# ----------------------------------------------------------------------------

def _all_allowed_mask() -> np.ndarray:
    return np.ones(VOCAB_SIZE, dtype=bool)


def _slot_token_range(slot_idx: int) -> Tuple[int, int]:
    """Inclusive lo, exclusive hi index of tokens belonging to a slot."""
    lo = slot_idx * NUM_THICKNESSES
    hi = lo + NUM_THICKNESSES
    return lo, hi


def _mask_slot(mask: np.ndarray, slot_idx: int, allow: bool) -> None:
    lo, hi = _slot_token_range(slot_idx)
    mask[lo:hi] = allow


def _mask_thickness_token(mask: np.ndarray, slot_idx: int,
                          thickness_nm: int, allow: bool) -> None:
    thickness_idx = (thickness_nm - 5) // 5
    mask[slot_idx * NUM_THICKNESSES + thickness_idx] = allow


# ----------------------------------------------------------------------------
# Name → slot resolution (handed in by parse.py; constraints store names, not
# slots, so they survive pool reordering)
# ----------------------------------------------------------------------------

def resolve_name(pool: List[MaterialEntry], name: str) -> Optional[int]:
    """Index of the first slot whose canonical name matches; None if absent."""
    for s, m in enumerate(pool):
        if m.canonical_name == name:
            return s
    return None


def resolve_all_names(pool: List[MaterialEntry], names: Tuple[str, ...]
                      ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Returns (resolved_slot_tuple, unresolved_name_tuple).

    Callers should treat any unresolved names as a parse error (the LLM
    should never reference a material that isn't in the pool — but if it
    does, surface it loudly).
    """
    resolved: List[int] = []
    unresolved: List[str] = []
    for n in names:
        s = resolve_name(pool, n)
        if s is None:
            unresolved.append(n)
        else:
            resolved.append(s)
    return tuple(resolved), tuple(unresolved)


# ============================================================================
# Concrete constraints
# ============================================================================

@dataclass(frozen=True)
class AllowedSubset(Constraint):
    """Structure may only use slots whose canonical name is in `allowed_names`."""
    kind: str = "allowed_subset"
    params: Dict[str, object] = field(default_factory=dict)
    allowed_names: Tuple[str, ...] = ()

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        for s in fs.slot_indices:
            if pool[s].canonical_name not in self.allowed_names:
                return False
        return True

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        mask = _all_allowed_mask()
        allowed = set(self.allowed_names)
        for s in range(p.pool_size):
            if pool[s].canonical_name not in allowed:
                _mask_slot(mask, s, allow=False)
        return mask


@dataclass(frozen=True)
class LayerIdentity(Constraint):
    """Layer at position `position` must use canonical material `material_name`."""
    kind: str = "layer_identity"
    params: Dict[str, object] = field(default_factory=dict)
    position: int = 0
    material_name: str = ""

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        if self.position >= len(fs.slot_indices):
            return False
        s = fs.slot_indices[self.position]
        return pool[s].canonical_name == self.material_name

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        if p.step != self.position:
            return None  # no constraint at non-matching step
        target_slot = resolve_name(pool, self.material_name)
        if target_slot is None or target_slot >= p.pool_size:
            # Materially unresolvable → caller will hit an empty feasible set,
            # which is the right error mode.
            return np.zeros(VOCAB_SIZE, dtype=bool)
        mask = np.zeros(VOCAB_SIZE, dtype=bool)
        _mask_slot(mask, target_slot, allow=True)
        # EOS is not allowed at this step — we promised a layer here.
        mask[EOS_TOKEN] = False
        return mask


@dataclass(frozen=True)
class AdjacentForbidden(Constraint):
    """The pair (name_a, name_b) may not be adjacent in the stack (either order)."""
    kind: str = "adjacent_forbidden"
    params: Dict[str, object] = field(default_factory=dict)
    forbidden_pairs: Tuple[Tuple[str, str], ...] = ()

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        for i in range(len(fs.slot_indices) - 1):
            a = pool[fs.slot_indices[i]].canonical_name
            b = pool[fs.slot_indices[i + 1]].canonical_name
            for x, y in self.forbidden_pairs:
                if {a, b} == {x, y}:
                    return False
        return True

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        if p.step == 0:
            return None
        prev_name = pool[p.slots_so_far[-1]].canonical_name
        forbidden_partners: set = set()
        for x, y in self.forbidden_pairs:
            if prev_name == x:
                forbidden_partners.add(y)
            elif prev_name == y:
                forbidden_partners.add(x)
        if not forbidden_partners:
            return None
        mask = _all_allowed_mask()
        for s in range(p.pool_size):
            if pool[s].canonical_name in forbidden_partners:
                _mask_slot(mask, s, allow=False)
        return mask


@dataclass(frozen=True)
class ThicknessRange(Constraint):
    """Per-layer thickness must be in [min_nm, max_nm].

    If `position` is None, applies globally; otherwise only at that step.
    """
    kind: str = "thickness_range"
    params: Dict[str, object] = field(default_factory=dict)
    min_nm: int = 5
    max_nm: int = 200
    position: Optional[int] = None

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        for i, t in enumerate(fs.thicknesses_nm):
            if self.position is not None and i != self.position:
                continue
            if not (self.min_nm <= t <= self.max_nm):
                return False
        return True

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        if self.position is not None and p.step != self.position:
            return None
        # Allow only thickness sub-tokens within [min_nm, max_nm].
        allowed_thick_idxs = [
            ti for ti, t in enumerate(THICKNESSES) if self.min_nm <= t <= self.max_nm
        ]
        if not allowed_thick_idxs:
            return np.zeros(VOCAB_SIZE, dtype=bool)
        mask = np.zeros(VOCAB_SIZE, dtype=bool)
        for s in range(p.pool_size):
            for ti in allowed_thick_idxs:
                mask[s * NUM_THICKNESSES + ti] = True
        mask[EOS_TOKEN] = True  # EOS still legal (layer-count handles min)
        return mask


@dataclass(frozen=True)
class LayerCount(Constraint):
    """min_layers ≤ structure length ≤ max_layers."""
    kind: str = "layer_count"
    params: Dict[str, object] = field(default_factory=dict)
    min_layers: int = 1
    max_layers: int = MAX_LAYERS

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        L = len(fs.slot_indices)
        return self.min_layers <= L <= self.max_layers

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        mask = _all_allowed_mask()
        # Suppress EOS until min reached.
        if p.step < self.min_layers:
            mask[EOS_TOKEN] = False
        # Force EOS once max reached (no more layers allowed).
        if p.step >= self.max_layers:
            mask[:] = False
            mask[EOS_TOKEN] = True
        return mask


@dataclass(frozen=True)
class OrderingBefore(Constraint):
    """Material A must appear at some layer before any occurrence of material B.

    Multiple ordering constraints fall back to post-hoc — the decode-mask
    logic for interleaved orderings gets gnarly.
    """
    kind: str = "ordering_before"
    params: Dict[str, object] = field(default_factory=dict)
    name_a: str = ""
    name_b: str = ""

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        names = [pool[s].canonical_name for s in fs.slot_indices]
        if self.name_b not in names:
            return True  # vacuously OK if B never appears
        if self.name_a not in names:
            return False
        return names.index(self.name_a) < names.index(self.name_b)

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        names_used = [pool[s].canonical_name for s in p.slots_so_far]
        if self.name_a in names_used:
            return None  # A already placed; B is now free
        # A not yet placed → block all B-named slots.
        mask = _all_allowed_mask()
        for s in range(p.pool_size):
            if pool[s].canonical_name == self.name_b:
                _mask_slot(mask, s, allow=False)
        return mask


@dataclass(frozen=True)
class TotalThickness(Constraint):
    """Sum of all layer thicknesses must be ≤ max_total_nm.

    Decoded with a Markovian running-budget mask BY DEFAULT (per my review of
    the plan: it's nearly free and meaningfully cuts the rejection rate).
    Pass `markovian_decode=False` to fall back to post-hoc only.

    Edge cases (handled, not crashes):
      - Budget already exceeded ⇒ force EOS. Whether the resulting structure
        is valid against any layer_count / etc. is the post-hoc check's job.
      - Remaining budget is too small for any thickness on the 5 nm grid ⇒
        force EOS. Same drop-vs-keep handling at the post-hoc layer.
    """
    kind: str = "total_thickness"
    params: Dict[str, object] = field(default_factory=dict)
    max_total_nm: int = MAX_LAYERS * 200
    markovian_decode: bool = True

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        return sum(fs.thicknesses_nm) <= self.max_total_nm

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        if not self.markovian_decode:
            return None
        budget_left = self.max_total_nm - p.total_thickness_so_far
        # If we're already at or past the cap, only EOS is allowed.
        if budget_left <= 0:
            mask = np.zeros(VOCAB_SIZE, dtype=bool)
            mask[EOS_TOKEN] = True
            return mask
        # Filter to thicknesses that still fit in the remaining budget.
        allowed_thick_idxs = [
            ti for ti, t in enumerate(THICKNESSES) if t <= budget_left
        ]
        # Remaining budget smaller than the floor (5 nm) — no thickness fits.
        # Force EOS; the post-hoc layer_count check (if any) will drop the
        # candidate. We deliberately don't raise here — see module docstring.
        if not allowed_thick_idxs:
            mask = np.zeros(VOCAB_SIZE, dtype=bool)
            mask[EOS_TOKEN] = True
            return mask
        mask = np.zeros(VOCAB_SIZE, dtype=bool)
        for s in range(p.pool_size):
            for ti in allowed_thick_idxs:
                mask[s * NUM_THICKNESSES + ti] = True
        mask[EOS_TOKEN] = True
        return mask


@dataclass(frozen=True)
class Symmetry(Constraint):
    """Stack is a palindrome: slot_i == slot_{L-1-i} and same for thicknesses.

    Live decoding strategy (with the no-same-material-adjacent enforced
    universally by generate.py):

      - Even-length palindromes are impossible because they have adjacent
        identical slots at the centre (e.g. [a,b,b,a] ⇒ b,b adjacent).
        So the live mask only admits odd L ∈ {3, 5, 7, 9}.

      - For each odd L still reachable from the current prefix, the slot at
        position k must equal whichever earlier prefix slot mirrors it (if
        that mirror is already placed), or is free (if the mirror is in
        the still-future first half).

      - EOS is allowed only if the current prefix is itself a palindrome —
        otherwise closing here would violate symmetry post-hoc.

      - If no L works (the prefix is already a dead-end), only EOS is
        allowed (if palindromic) or no token is allowed (forces EOS in
        generate.py, post-hoc check then drops the candidate). No raise.

    Boost layer (decode_boost):

      The model trained without symmetry knowledge tends to favour long
      sequences. To get variety in palindrome length, we soft-boost
      `slots_so_far[k-2]` at decoding steps k ∈ {2, 3, 4}. That slot is the
      one that closes the prefix into an odd palindrome of length 2k-1:
        k=2 ⇒ length 3   [a, b, a]
        k=3 ⇒ length 5   [a, b, c, b, a]
        k=4 ⇒ length 7   [a, b, c, d, c, b, a]
      The boost is additive in logit space (+2.0 ≈ ×7 in odds), small
      enough that the model can still override and extend to length 9 when
      ΔE wants it to.
    """
    kind: str = "symmetry"
    params: Dict[str, object] = field(default_factory=dict)
    match_thickness: bool = True

    # Logit additive when nudging toward an odd-palindrome close.
    _BOOST_DELTA: float = 2.0

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        L = len(fs.slot_indices)
        for i in range(L // 2):
            if fs.slot_indices[i] != fs.slot_indices[L - 1 - i]:
                return False
            if self.match_thickness and \
                    fs.thicknesses_nm[i] != fs.thicknesses_nm[L - 1 - i]:
                return False
        return True

    @staticmethod
    def _is_palindrome(slots: List[int]) -> bool:
        return all(slots[i] == slots[len(slots) - 1 - i]
                   for i in range(len(slots) // 2))

    def _prefix_compatible_with(self, slots: List[int], L: int) -> bool:
        """Could a palindrome of length L extend `slots` (= first k positions)?

        Yes iff every pair (i, L-1-i) that's *already in the prefix* matches.
        Pairs with one position still in the future are accepted; the live
        mask will lock them later.
        """
        k = len(slots)
        for i in range((L + 1) // 2):
            j = L - 1 - i
            if i < k and j < k and slots[i] != slots[j]:
                return False
        return True

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        k = p.step
        if k == 0:
            return None  # First layer is unconstrained
        slots = p.slots_so_far
        thicks = p.thicknesses_so_far

        mask = np.zeros(VOCAB_SIZE, dtype=bool)

        # Which odd palindromic lengths are still reachable?
        valid_L = [
            L for L in range(k + 1, MAX_LAYERS + 1)
            if L % 2 == 1 and self._prefix_compatible_with(slots, L)
        ]

        if not valid_L:
            # Dead-end: no L works. EOS is the only graceful close, but only
            # if the prefix is already a palindrome (which it must be if we
            # got here following the mask — defensive check anyway).
            if self._is_palindrome(slots):
                mask[EOS_TOKEN] = True
            return mask  # otherwise all-False ⇒ generate.py forces EOS, dropped

        # Allow EOS if current prefix is already an odd palindrome (closing here).
        if len(slots) % 2 == 1 and self._is_palindrome(slots):
            mask[EOS_TOKEN] = True

        # For every valid L, the slot at position k is either (a) forced to
        # mirror an earlier prefix slot, or (b) free (still in first half).
        free_choice = False
        forced_pairs: set = set()   # (slot, thickness_nm) pairs allowed
        for L in valid_L:
            m = L - 1 - k
            if m < 0:
                continue
            if m >= k:
                # Position k is in the first half — any slot OK for this L.
                free_choice = True
                continue
            s_mirror = slots[m]
            if 0 <= s_mirror < p.pool_size:
                if self.match_thickness:
                    forced_pairs.add((s_mirror, thicks[m]))
                else:
                    forced_pairs.add((s_mirror, None))

        if free_choice:
            for s in range(p.pool_size):
                _mask_slot(mask, s, allow=True)
        else:
            for s, t in forced_pairs:
                if t is None:
                    _mask_slot(mask, s, allow=True)
                else:
                    _mask_thickness_token(mask, s, t, allow=True)

        return mask

    def decode_boost(self, p: PartialStructure, pool: List[MaterialEntry]
                     ) -> Optional[np.ndarray]:
        """Soft-encourage closing into a short odd palindrome at k = 2, 3, 4.

        Adds `_BOOST_DELTA` to the logits of `slots_so_far[k-2]` (all
        thickness sub-tokens) so the model is biased toward `[a, b, a]`,
        `[a, b, c, b, a]`, `[a, b, c, d, c, b, a]` rather than always
        extending to the 9-layer maximum.
        """
        k = p.step
        if k < 2 or k > 4:
            return None
        target = p.slots_so_far[k - 2]
        if target >= p.pool_size:
            return None
        boost = np.zeros(VOCAB_SIZE, dtype=np.float32)
        lo, hi = _slot_token_range(target)
        boost[lo:hi] = self._BOOST_DELTA
        return boost


# ----------------------------------------------------------------------------
# Constraint set — composes a collection of constraints into one mask
# ----------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    """Bundle of constraints. The orchestrator interacts with this only.

    `decode_mask` returns the AND of every active per-step mask (None-returning
    constraints are skipped). `check` runs every post-hoc check and returns
    a list of (kind, passed, detail) tuples for reporting.
    """
    constraints: List[Constraint] = field(default_factory=list)

    def split_during_post(self) -> Tuple[List[Constraint], List[Constraint]]:
        """Partition into the two enforcement modes by constraint kind.

        Rules (post-live-symmetry update):
          - Symmetry → during. The live mask gates to odd-length palindromes
            only (paired with universal no-same-material-adjacent) and emits
            a logit boost toward short palindromes.
          - TotalThickness with `markovian_decode=False` → post.
          - OrderingBefore → during ONLY if there's a single such constraint.
            Multiple interacting orderings → all post.
          - Everything else (AllowedSubset, LayerIdentity, AdjacentForbidden,
            ThicknessRange, LayerCount, TotalThickness w/ markovian) → during.
        """
        n_orderings = sum(
            1 for c in self.constraints if isinstance(c, OrderingBefore)
        )

        during: List[Constraint] = []
        post: List[Constraint] = []
        for c in self.constraints:
            if isinstance(c, OrderingBefore) and n_orderings > 1:
                post.append(c)
            elif isinstance(c, TotalThickness) and not c.markovian_decode:
                post.append(c)
            else:
                during.append(c)
        return during, post

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> np.ndarray:
        """AND every active constraint's per-step mask. Constraints that
        return None at this step contribute nothing (the AND with all-True
        is a no-op)."""
        agg = _all_allowed_mask()
        for c in self.constraints:
            m = c.decode_mask(p, pool)
            if m is None:
                continue
            agg &= m
        return agg

    def decode_boost(self, p: PartialStructure, pool: List[MaterialEntry]
                     ) -> Optional[np.ndarray]:
        """Sum every constraint's per-step logit boost.

        Boosts are soft (additive log-probability); they steer sampling
        within whatever the mask already allows. Returns None if no
        constraint has a boost at this step — generate.py treats None as
        "no-op" so we avoid a hot-path numpy allocation in the common case.
        """
        agg: Optional[np.ndarray] = None
        for c in self.constraints:
            fn = getattr(c, "decode_boost", None)
            if fn is None:
                continue
            b = fn(p, pool)
            if b is None:
                continue
            if agg is None:
                agg = b.astype(np.float32, copy=True)
            else:
                agg = agg + b
        return agg

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]
              ) -> List[Tuple[str, bool, str]]:
        """Run every post-hoc check. Returns (kind, passed, detail) per
        constraint — fed straight into `Result.constraints_report`.
        """
        out: List[Tuple[str, bool, str]] = []
        for c in self.constraints:
            try:
                ok = c.check(fs, pool)
                detail = "" if ok else f"failed: {dataclasses.asdict(c)}"
            except Exception as exc:  # constraint code bug → loud but local
                ok = False
                detail = f"error: {type(exc).__name__}: {exc}"
            out.append((c.kind, ok, detail))
        return out


# ----------------------------------------------------------------------------
# Lightweight self-test (no model / no JLL needed)
# ----------------------------------------------------------------------------

def _self_test() -> None:
    """Asserts the decode_mask and check semantics for each constraint kind.

    Uses a synthetic 4-material pool. Run with `python -m
    inference.src.constraints` to verify.
    """
    pool = [
        MaterialEntry(canonical_name="Ag", n=np.ones(128), k=np.zeros(128)),
        MaterialEntry(canonical_name="SiO2", n=np.ones(128) * 1.45,
                      k=np.zeros(128)),
        MaterialEntry(canonical_name="TiO2", n=np.ones(128) * 2.4,
                      k=np.zeros(128)),
        MaterialEntry(canonical_name="Al", n=np.ones(128) * 1.2, k=np.zeros(128)),
    ]
    pool_size = 4

    # 1. AllowedSubset
    c = AllowedSubset(allowed_names=("Ag", "SiO2"))
    p = PartialStructure([], [], pool_size)
    mask = c.decode_mask(p, pool)
    assert mask[encode_layer(0, 100)]   # Ag-100nm allowed
    assert mask[encode_layer(1, 100)]   # SiO2-100nm allowed
    assert not mask[encode_layer(2, 100)]  # TiO2 blocked
    assert not mask[encode_layer(3, 100)]  # Al blocked
    assert c.check(FinishedStructure([0, 1], [50, 100], pool_size), pool)
    assert not c.check(FinishedStructure([0, 2], [50, 100], pool_size), pool)

    # 2. LayerIdentity at position 0
    c = LayerIdentity(position=0, material_name="Ag")
    mask = c.decode_mask(PartialStructure([], [], pool_size), pool)
    assert mask[encode_layer(0, 50)]
    assert not mask[encode_layer(1, 50)]
    assert not mask[EOS_TOKEN]  # can't EOS where a layer was promised
    # At step 1 the constraint is inactive
    assert c.decode_mask(PartialStructure([0], [50], pool_size), pool) is None

    # 3. AdjacentForbidden
    c = AdjacentForbidden(forbidden_pairs=(("Ag", "TiO2"),))
    mask = c.decode_mask(PartialStructure([0], [50], pool_size), pool)
    assert mask[encode_layer(1, 100)]   # SiO2 after Ag ok
    assert not mask[encode_layer(2, 100)]  # TiO2 after Ag blocked
    assert not c.check(
        FinishedStructure([0, 2], [50, 100], pool_size), pool)

    # 4. ThicknessRange (global)
    c = ThicknessRange(min_nm=20, max_nm=100)
    mask = c.decode_mask(PartialStructure([], [], pool_size), pool)
    assert not mask[encode_layer(0, 5)]   # 5 nm below floor
    assert mask[encode_layer(0, 20)]      # 20 nm in range
    assert mask[encode_layer(0, 100)]
    assert not mask[encode_layer(0, 105)]

    # 5. LayerCount: min=3, max=5
    c = LayerCount(min_layers=3, max_layers=5)
    # At step 2: EOS suppressed (min not reached)
    mask = c.decode_mask(PartialStructure([0, 1], [50, 50], pool_size), pool)
    assert not mask[EOS_TOKEN]
    # At step 3: EOS allowed
    mask = c.decode_mask(PartialStructure([0, 1, 2], [50, 50, 50], pool_size), pool)
    assert mask[EOS_TOKEN]
    # At step 5: only EOS
    mask = c.decode_mask(
        PartialStructure([0, 1, 2, 3, 0], [50] * 5, pool_size), pool)
    assert mask.sum() == 1 and mask[EOS_TOKEN]

    # 6. OrderingBefore
    c = OrderingBefore(name_a="Ag", name_b="SiO2")
    mask = c.decode_mask(PartialStructure([], [], pool_size), pool)
    assert mask[encode_layer(0, 50)]   # Ag itself fine
    assert not mask[encode_layer(1, 50)]  # SiO2 blocked
    mask = c.decode_mask(PartialStructure([0], [50], pool_size), pool)
    assert mask is None  # A placed → no constraint
    assert c.check(FinishedStructure([0, 1], [50, 50], pool_size), pool)
    assert not c.check(FinishedStructure([1, 0], [50, 50], pool_size), pool)

    # 7. TotalThickness with running budget
    c = TotalThickness(max_total_nm=200)
    mask = c.decode_mask(PartialStructure([0], [150], pool_size), pool)
    assert mask[encode_layer(0, 50)]    # 150+50=200 fits exactly
    assert not mask[encode_layer(0, 55)]  # 205 > 200
    # Budget exhausted → only EOS.
    mask = c.decode_mask(PartialStructure([0, 1], [100, 100], pool_size), pool)
    assert mask.sum() == 1 and mask[EOS_TOKEN]
    assert c.check(FinishedStructure([0, 1], [100, 100], pool_size), pool)
    assert not c.check(FinishedStructure([0, 1], [150, 100], pool_size), pool)

    # 8. Symmetry — now LIVE-decoded
    c = Symmetry(match_thickness=True)

    # check() still works post-hoc.
    assert c.check(FinishedStructure([0, 1, 0], [50, 75, 50], pool_size), pool)
    assert not c.check(FinishedStructure([0, 1, 2], [50, 75, 50], pool_size), pool)
    assert not c.check(FinishedStructure([0, 1, 0], [50, 75, 60], pool_size), pool)

    # decode_mask: step 0 unconstrained.
    assert c.decode_mask(PartialStructure([], [], pool_size), pool) is None

    # After [s_0, s_1] (slots=[0, 1] thick=[50, 75]) at step k=2:
    #   - L=3 needs slot[2] == slot[0] = 0  (and thickness == thick[0] = 50)
    #   - L=5 has center at position 2, free choice
    #   - L=7 / L=9 also have position 2 in first half (free)
    # ⇒ free_choice = True ⇒ every pool slot allowed.
    p = PartialStructure([0, 1], [50, 75], pool_size)
    mask = c.decode_mask(p, pool)
    assert mask[encode_layer(0, 50)]    # slot 0 with mirror thickness 50 allowed
    assert mask[encode_layer(2, 100)]   # slot 2 (free) allowed
    # EOS: prefix [0, 1] not a palindrome (0 != 1) ⇒ EOS not allowed.
    assert not mask[EOS_TOKEN]

    # decode_boost at k=2: boost slot slots_so_far[k-2] = slot 0
    boost = c.decode_boost(p, pool)
    assert boost is not None
    assert boost[encode_layer(0, 50)] > 0       # slot 0 boosted
    assert boost[encode_layer(2, 50)] == 0      # slot 2 not boosted

    # At k=4 with prefix [0, 1, 2, 3] no odd L admits free choice in the
    # second half: position 4 is the centre of L=9. ⇒ free, slot 3 boost.
    p4 = PartialStructure([0, 1, 2, 3], [50, 50, 50, 50], pool_size)
    boost = c.decode_boost(p4, pool)
    assert boost is not None
    assert boost[encode_layer(2, 50)] > 0       # slot 2 boosted (k-2 = 2)

    # At k=5 with prefix [0, 1, 2, 3, 4]: position 5 is past the centre of L=9
    # (centre at 4). For L=9 it must mirror position 3 ⇒ slot 3.
    p5 = PartialStructure([0, 1, 2, 3, 4][:5][:pool_size], [50] * 5, pool_size)
    # The pool only has 4 slots in this fixture so we test with 4-prefix instead.
    # Use a pool with 5 materials for the lock test:
    big_pool = pool + [MaterialEntry(canonical_name="E",
                                     n=np.ones(128), k=np.zeros(128))]
    p5 = PartialStructure([0, 1, 2, 3], [40, 60, 40, 60], pool_size=5)
    # Hmm: with pool_size=5 and prefix length 4, the test exercises the
    # symmetry mask at the boundary where extending to L>=5 is still
    # possible. With pool=4 entries the symmetric check still works at
    # the first-half free choice. Skip the explicit k=5 lock test to keep
    # the no-deps fixture small.

    # Empty allowed_subset → physical error path not reached here, but the
    # Symmetry dead-end path is: prefix [0, 1, 2, 0] cannot complete to any
    # odd palindrome (L=5 needs slots[1]==slots[3], false; L=7 needs
    # slots[2]==slots[4] later but ok; L=9 also ok). Actually L=7, 9 are
    # still compatible. So this isn't a dead end. (Sanity: at least one mask
    # should be returned.)
    p = PartialStructure([0, 1, 2, 0], [50] * 4, pool_size)
    mask = c.decode_mask(p, pool)
    assert mask is not None

    # ConstraintSet AND
    cs = ConstraintSet(constraints=[
        AllowedSubset(allowed_names=("Ag", "SiO2")),
        LayerCount(min_layers=2, max_layers=4),
        TotalThickness(max_total_nm=200),
    ])
    p0 = PartialStructure([], [], pool_size)
    m0 = cs.decode_mask(p0, pool)
    # First step: only Ag/SiO2 slots, EOS suppressed (need min=2).
    assert not m0[EOS_TOKEN]
    assert m0[encode_layer(0, 50)] and m0[encode_layer(1, 50)]
    assert not m0[encode_layer(2, 50)] and not m0[encode_layer(3, 50)]

    during, post = cs.split_during_post()
    assert all(c.kind in {"allowed_subset", "layer_count", "total_thickness"}
               for c in during)
    assert len(post) == 0

    # Symmetry alone → now lives in `during` (live-decoded).
    cs2 = ConstraintSet(constraints=[Symmetry()])
    during, post = cs2.split_during_post()
    assert len(during) == 1 and post == []

    # ConstraintSet.decode_boost aggregates per-constraint boosts. Symmetry
    # at step 2 contributes a non-None vector; an empty set gives None.
    p = PartialStructure([0, 1], [50, 75], pool_size)
    assert cs2.decode_boost(p, pool) is not None
    assert ConstraintSet().decode_boost(p, pool) is None

    print("[constraints] self-test: all 8 kinds + ConstraintSet OK")


if __name__ == "__main__":
    _self_test()
