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
    EOS_TOKEN, M_MAX, MAX_LAYERS, NUM_THICKNESSES, THICKNESSES, VOCAB_SIZE,
    encode_layer,
)

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
    thicknesses_so_far: List[int]      # nm at each placed layer (5 nm grid)
    pool_size: int                     # valid slot range: [0, pool_size)

    @property
    def step(self) -> int:
        return len(self.slots_so_far)

    @property
    def total_thickness_so_far(self) -> int:
        return int(sum(self.thicknesses_so_far))


@dataclass
class FinishedStructure:
    """Post-hoc-check view of a fully-decoded candidate."""
    slot_indices: List[int]
    thicknesses_nm: List[int]
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
        if budget_left <= 0:
            # Force EOS — no more thickness room.
            mask = np.zeros(VOCAB_SIZE, dtype=bool)
            mask[EOS_TOKEN] = True
            return mask
        # Allow thickness sub-tokens ≤ budget_left, plus EOS.
        allowed_thick_idxs = [
            ti for ti, t in enumerate(THICKNESSES) if t <= budget_left
        ]
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

    Inherently non-decodable (the last layer's identity depends on the first),
    so this only does post-hoc check. Symmetry decoders would require
    look-ahead and break the AR assumption.
    """
    kind: str = "symmetry"
    params: Dict[str, object] = field(default_factory=dict)
    match_thickness: bool = True

    def check(self, fs: FinishedStructure, pool: List[MaterialEntry]) -> bool:
        L = len(fs.slot_indices)
        for i in range(L // 2):
            if fs.slot_indices[i] != fs.slot_indices[L - 1 - i]:
                return False
            if self.match_thickness and \
                    fs.thicknesses_nm[i] != fs.thicknesses_nm[L - 1 - i]:
                return False
        return True

    def decode_mask(self, p: PartialStructure, pool: List[MaterialEntry]
                    ) -> Optional[np.ndarray]:
        return None  # post-hoc only


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

        Rules:
          - Symmetry → always post (non-Markovian, needs look-ahead).
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
            if isinstance(c, Symmetry):
                post.append(c)
            elif isinstance(c, OrderingBefore) and n_orderings > 1:
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

    # 8. Symmetry (post-hoc only)
    c = Symmetry(match_thickness=True)
    assert c.decode_mask(PartialStructure([], [], pool_size), pool) is None
    assert c.check(FinishedStructure([0, 1, 0], [50, 75, 50], pool_size), pool)
    assert not c.check(FinishedStructure([0, 1, 2], [50, 75, 50], pool_size), pool)
    assert not c.check(FinishedStructure([0, 1, 0], [50, 75, 60], pool_size), pool)

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

    # Symmetry alone → post.
    cs2 = ConstraintSet(constraints=[Symmetry()])
    during, post = cs2.split_during_post()
    assert during == [] and len(post) == 1

    print("[constraints] self-test: all 8 kinds + ConstraintSet OK")


if __name__ == "__main__":
    _self_test()
