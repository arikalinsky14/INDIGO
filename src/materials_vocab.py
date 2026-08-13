"""
Slot-Indexed Token Vocabulary (two-head variant)
================================================

The model now has two output heads: a slot classifier (which material
in the pool to deposit next, or EOS) and a per-slot thickness regressor
(how thick that layer should be, in nm). The joint (slot × thickness)
vocab from the previous version is gone.

Vocabulary layout (used by the slot head only)
----------------------------------------------
- Tokens 0 .. M_MAX-1 : layer tokens (slot index k)
- Token M_MAX         : EOS

There is no thickness in this vocabulary. Thicknesses live in a separate
regression target and never enter the classification loss.

Structure-matrix contract
-------------------------
The structure matrix is still [M_MAX, MAX_LAYERS] of normalized-nm
scalars. Thicknesses are now continuous floats (5..200 nm), not snapped
to the old 5-nm grid. `normalize_thickness` / `denormalize_thickness`
are simple scalar rescales — no rounding.

Backward compatibility
----------------------
This is a HARD schema break vs the previous grid-based vocab. Any
parquet shard produced before the transition (with int layer_thicknesses
and a per-slot × per-thickness token vocab) is not readable by the new
loader. Regenerate the dataset before training.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.material_features import MaterialNK


# ============================================================================
# Configuration
# ============================================================================

# Maximum number of materials in any single pool. Padding fills shorter pools.
M_MAX: int = 32

# Physical thickness range (nm). No grid — thicknesses are continuous.
MIN_THICKNESS_NM: float = 5.0
MAX_THICKNESS_NM: float = 200.0

# Maximum number of layers in a structure.
MAX_LAYERS: int = 10

# Vocab sizes for the SLOT HEAD (thickness is regression, not classification).
NUM_LAYER_TOKENS: int = M_MAX     # one token per slot
EOS_TOKEN: int = NUM_LAYER_TOKENS  # slot index M_MAX == "stop"
VOCAB_SIZE: int = NUM_LAYER_TOKENS + 1   # M_MAX + 1

# Kept for callers that still import the old grid size. Nothing in the new
# training path uses these; the search / random-layer generators sample
# continuously in [MIN_THICKNESS_NM, MAX_THICKNESS_NM]. The inference-time
# constraint machinery (inference/src/constraints.py) still speaks joint
# (slot × thickness) vocab; the LEGACY_* constants below let it keep
# building masks in that shape while the new slot-only vocab (VOCAB_SIZE =
# M_MAX + 1) drives model input/output.
THICKNESSES: List[float] = [
    5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
    55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
    105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0,
    155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0,
]
NUM_THICKNESSES: int = len(THICKNESSES)
LEGACY_VOCAB_SIZE: int = M_MAX * NUM_THICKNESSES + 1  # joint-vocab size
LEGACY_EOS_TOKEN: int = M_MAX * NUM_THICKNESSES       # last index of joint vocab


# ============================================================================
# Slot-token encoding / decoding
# ============================================================================


def encode_slot(slot_idx: int) -> int:
    """Encode a slot index → slot-head token id."""
    if not (0 <= slot_idx < M_MAX):
        raise ValueError(f"slot_idx {slot_idx} out of range [0, {M_MAX})")
    return int(slot_idx)


def decode_slot_token(
    token_id: int, pool: Optional[List[MaterialNK]] = None
) -> Tuple[Optional[str], int, str]:
    """Decode a slot-head token back to (material_name, slot_idx, kind).

    Returns
    -------
    material_name : str or None
        Name looked up from `pool`, if provided and slot_idx < len(pool).
    slot_idx : int
        Slot index, or -1 for EOS.
    kind : str
        'LAYER' or 'EOS'.
    """
    if token_id == EOS_TOKEN:
        return None, -1, "EOS"
    if not (0 <= token_id < NUM_LAYER_TOKENS):
        raise ValueError(f"token_id {token_id} out of range [0, {VOCAB_SIZE})")
    name = None
    if pool is not None and token_id < len(pool):
        name = pool[token_id].name
    return name, int(token_id), "LAYER"


def is_eos(token_id: int) -> bool:
    return token_id == EOS_TOKEN


# --- Legacy shims -----------------------------------------------------------
# Older callers imported `encode_layer(slot_idx, thickness_nm) -> int` and
# `decode_token(token_id) -> (name, thickness_nm, slot_idx, thick_idx, kind)`.
# The transition drops the joint token space entirely, so these shims exist
# only to catch stragglers with a clear error rather than a silent mis-encode.


def encode_layer(slot_idx: int, thickness_nm: float) -> int:  # noqa: D401
    """Deprecated joint-token encoder — thickness is now a regression target.

    Returns a slot-only token; thickness is silently discarded. Kept so
    older callers don't crash mid-refactor, but every hot path should be
    updated to call `encode_slot(slot_idx)` directly.
    """
    return encode_slot(slot_idx)


def decode_token(
    token_id: int, pool: Optional[List[MaterialNK]] = None
) -> Tuple[Optional[str], Optional[float], int, int, str]:
    """Legacy 5-tuple decoder. Thickness is now None (no longer in the vocab).

    Prefer `decode_slot_token` in new code.
    """
    name, slot_idx, kind = decode_slot_token(token_id, pool=pool)
    # Legacy tuple layout kept so old call sites still unpack — thickness
    # comes from the regression head, not from decoding.
    return name, None, slot_idx, -1, kind


# ============================================================================
# Thickness normalisation
# ============================================================================
#
# Thicknesses in the structure matrix are stored as a scalar in [0, 1],
# which is nm / MAX_THICKNESS_NM. The model's regression head emits
# sigmoid outputs on the same scale, so training-time normalisation and
# the model's output space match by construction.


def normalize_thickness(thickness_nm: float) -> float:
    """nm → normalized ∈ (0, 1]."""
    return float(thickness_nm) / MAX_THICKNESS_NM


def denormalize_thickness(thickness_norm: float) -> float:
    """normalized → nm, CLAMPED to [MIN_THICKNESS_NM, MAX_THICKNESS_NM]. No grid snap."""
    nm = float(thickness_norm) * MAX_THICKNESS_NM
    return max(MIN_THICKNESS_NM, min(MAX_THICKNESS_NM, nm))


# ============================================================================
# Structure matrix construction (thicknesses are floats now)
# ============================================================================


def build_structure_matrix(
    slot_indices: List[int], thicknesses_nm: List[float]
) -> torch.Tensor:
    """[M_MAX, MAX_LAYERS] structure matrix of normalized-nm scalars."""
    if len(slot_indices) != len(thicknesses_nm):
        raise ValueError("slot_indices and thicknesses_nm must have same length")
    if len(slot_indices) > MAX_LAYERS:
        raise ValueError(
            f"structure has {len(slot_indices)} layers, max is {MAX_LAYERS}"
        )

    matrix = torch.zeros(M_MAX, MAX_LAYERS, dtype=torch.float32)
    for layer_idx, (slot, thick) in enumerate(zip(slot_indices, thicknesses_nm)):
        if not (0 <= slot < M_MAX):
            raise ValueError(f"slot {slot} out of range [0, {M_MAX})")
        matrix[slot, layer_idx] = normalize_thickness(float(thick))
    return matrix


def decode_structure_matrix(
    matrix: torch.Tensor,
) -> Tuple[List[int], List[float]]:
    """Recover (slot_indices, thicknesses_nm) from a structure matrix.

    Thicknesses come back as floats. Stops at the first all-zero column.
    """
    slot_indices: List[int] = []
    thicknesses: List[float] = []
    for layer_idx in range(MAX_LAYERS):
        col = matrix[:, layer_idx]
        if col.sum().item() == 0:
            break
        slot = int(col.argmax().item())
        slot_indices.append(slot)
        thicknesses.append(denormalize_thickness(col[slot].item()))
    return slot_indices, thicknesses


# ============================================================================
# Lab normalization (unchanged)
# ============================================================================

_L_SCALE: float = 100.0
_AB_SCALE: float = 128.0


def normalize_lab(lab: List[float]) -> torch.Tensor:
    L, a, b = lab
    return torch.tensor(
        [L / _L_SCALE, a / _AB_SCALE, b / _AB_SCALE], dtype=torch.float32
    )


def denormalize_lab(lab_norm: torch.Tensor) -> List[float]:
    return [
        float(lab_norm[0].item() * _L_SCALE),
        float(lab_norm[1].item() * _AB_SCALE),
        float(lab_norm[2].item() * _AB_SCALE),
    ]


# ============================================================================
# Slot-head output masking
# ============================================================================


def build_output_mask(pool_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Mask over the slot-head vocab. Tokens with slot ≥ pool_size get -inf.

    EOS (slot index M_MAX) is always valid.

    Returns
    -------
    mask : torch.Tensor, shape [VOCAB_SIZE = M_MAX + 1]
    """
    if not (1 <= pool_size <= M_MAX):
        raise ValueError(f"pool_size {pool_size} out of range [1, {M_MAX}]")

    mask = torch.full((VOCAB_SIZE,), float("-inf"), dtype=torch.float32, device=device)
    mask[:pool_size] = 0.0
    mask[EOS_TOKEN] = 0.0
    return mask


def build_output_mask_batch(
    pool_sizes: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Vectorised `build_output_mask`. Returns [B, VOCAB_SIZE]."""
    B = pool_sizes.size(0)
    device = device or pool_sizes.device

    slot_ids = torch.arange(VOCAB_SIZE, device=device)
    # Slot index of each token: itself for 0..M_MAX-1, sentinel -1 for EOS
    # so the < pool_size comparison keeps EOS valid for every row.
    token_slots = slot_ids.clone()
    token_slots[EOS_TOKEN] = -1

    pool_sizes_b = pool_sizes.to(device).long().unsqueeze(1)  # [B, 1]
    valid = token_slots.unsqueeze(0) < pool_sizes_b            # [B, V]
    valid[:, EOS_TOKEN] = True

    mask = torch.where(
        valid,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), float("-inf"), dtype=torch.float32, device=device),
    )
    return mask


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print(f"M_MAX            = {M_MAX}")
    print(f"NUM_LAYER_TOKENS = {NUM_LAYER_TOKENS}")
    print(f"EOS_TOKEN        = {EOS_TOKEN}")
    print(f"VOCAB_SIZE       = {VOCAB_SIZE}   (was M_MAX * NUM_THICKNESSES + 1 pre-transition)")

    # Slot round-trip.
    for slot in [0, 5, M_MAX - 1]:
        tok = encode_slot(slot)
        _, s_back, kind = decode_slot_token(tok)
        assert s_back == slot and kind == "LAYER", "slot round-trip failed"
        print(f"  slot={slot} -> token {tok} -> slot={s_back} ({kind}) ✓")

    # EOS.
    _, s, kind = decode_slot_token(EOS_TOKEN)
    assert s == -1 and kind == "EOS"
    print(f"  EOS ({EOS_TOKEN}) -> {kind} ✓")

    # Continuous thickness normalization.
    for t in [5.0, 12.7, 100.0, 199.9]:
        n = normalize_thickness(t)
        back = denormalize_thickness(n)
        assert abs(back - t) < 1e-6, f"thickness round-trip failed: {t} -> {n} -> {back}"
    print("  thickness normalize round-trip ✓")

    # Structure matrix round-trip with floats.
    slots = [2, 5, 0, 7]
    thicks = [50.3, 100.0, 75.1, 199.9]
    M = build_structure_matrix(slots, thicks)
    s_back, t_back = decode_structure_matrix(M)
    assert s_back == slots
    for a, b in zip(thicks, t_back):
        assert abs(a - b) < 1e-4, f"thickness round-trip: {a} vs {b}"
    print(f"  structure matrix (float): slots={slots} thicks={thicks} ✓")

    # Slot-head mask.
    mask = build_output_mask(pool_size=3)
    valid_count = (mask == 0.0).sum().item()
    expected_valid = 3 + 1  # 3 slots + EOS
    assert valid_count == expected_valid
    print(f"  pool_size=3 mask: {int(valid_count)} valid tokens ({expected_valid} expected) ✓")

    mask_batch = build_output_mask_batch(torch.tensor([1, 8, M_MAX]))
    counts = (mask_batch == 0.0).sum(dim=1).tolist()
    expected = [1 + 1, 8 + 1, M_MAX + 1]
    assert counts == expected, f"batched mask: {counts} vs {expected}"
    print(f"  batched mask counts: {counts} ✓")

    print("[smoke] OK")
