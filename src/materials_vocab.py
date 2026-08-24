"""
Slot-Indexed Token Vocabulary
=============================

Replaces the original CHROMA-Lite `materials_vocab.py`. The key change:
tokens no longer encode material *identity*. They encode "use the material
in slot k of the input pool, at thickness bin t".

This is the core mechanism that lets the same trained model handle any
user-provided material pool at inference time. The model never sees a
fixed material vocabulary — only featurized n,k spectra in pool slots.

Vocabulary layout
-----------------
- Tokens 0 .. M_MAX*NUM_THICKNESSES-1: layer tokens
    token_id = slot_idx * NUM_THICKNESSES + thickness_idx
    where slot_idx ∈ [0, M_MAX) and thickness_idx ∈ [0, NUM_THICKNESSES)
- Token M_MAX*NUM_THICKNESSES: EOS

Differences from original CHROMA-Lite vocab
-------------------------------------------
- No ERROR token. Constraints are handled post-hoc by filtering generated
  structures, not by the model.
- M_MAX is configurable. Defaults to 32 — comfortably larger than the JLL
  library, with room for user-supplied custom materials.
- The thickness grid is 2..200 nm in 2 nm steps (100 bins). This is
  finer than the original 5 nm grid — the scripts/thickness_sensitivity
  study showed p95 snap-ΔE drops ~2.5× at 2 nm vs 5 nm, and the model's
  head-Linear grows only ~+60 k params (0.09 % of the cross-attn model),
  so the accuracy win comes essentially free.

Decoding requires the pool
--------------------------
Because slot indices have no meaning without a pool, `decode_token` takes
a `pool` argument. This is a slight ergonomic change vs. the original
`decode_token(token_id)` signature — callers must pass the pool the
structure was generated against.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.material_features import MaterialNK


# ============================================================================
# Configuration
# ============================================================================

# Maximum number of materials in any single pool. Padding fills shorter pools.
# 32 is generous: JLL has ~30 named materials, and even with disambiguated
# parameterisations we stay well under this.
M_MAX: int = 32

# Thickness grid (nm) — unchanged from original CHROMA-Lite.
THICKNESSES: List[int] = list(range(2, 201, 2))  # [2, 4, ..., 200] — 2 nm grid
_THICKNESS_STEP_NM: int = 2                      # step of the token grid
NUM_THICKNESSES: int = len(THICKNESSES)
MAX_THICKNESS_NM: int = 200

# Maximum number of layers in a structure.
MAX_LAYERS: int = 10

# Derived sizes.
NUM_LAYER_TOKENS: int = M_MAX * NUM_THICKNESSES
EOS_TOKEN: int = NUM_LAYER_TOKENS
VOCAB_SIZE: int = NUM_LAYER_TOKENS + 1


# ============================================================================
# Token encoding / decoding
# ============================================================================


def encode_layer(slot_idx: int, thickness_nm: int) -> int:
    """Encode (slot_index, thickness_nm) → token_id.

    Parameters
    ----------
    slot_idx : int
        Which slot of the material pool this layer uses, ∈ [0, M_MAX).
    thickness_nm : int
        Layer thickness in nm. Must be one of THICKNESSES (2..200, step 2).

    Returns
    -------
    token_id : int ∈ [0, NUM_LAYER_TOKENS)
    """
    if not (0 <= slot_idx < M_MAX):
        raise ValueError(f"slot_idx {slot_idx} out of range [0, {M_MAX})")
    if thickness_nm not in THICKNESSES:
        raise ValueError(
            f"thickness {thickness_nm} not in valid grid "
            f"{THICKNESSES[0]}..{THICKNESSES[-1]} step {_THICKNESS_STEP_NM}"
        )
    thickness_idx = (thickness_nm - _THICKNESS_STEP_NM) // _THICKNESS_STEP_NM
    return slot_idx * NUM_THICKNESSES + thickness_idx


def decode_token(
    token_id: int, pool: Optional[List[MaterialNK]] = None
) -> Tuple[Optional[str], Optional[int], int, int, str]:
    """Decode a token back to its components.

    Parameters
    ----------
    token_id : int
    pool : list of MaterialNK, optional
        The material pool the token was generated against. If provided, the
        material name is looked up; otherwise returned as None.

    Returns
    -------
    material_name : str or None
        Name of the material in the slot, or None if pool was not provided
        or the slot index exceeds the pool size.
    thickness_nm : int or None
        Layer thickness in nm, or None for EOS.
    slot_idx : int
        Slot index, or -1 for EOS.
    thickness_idx : int
        Thickness bin index, or -1 for EOS.
    token_type : str
        'LAYER' or 'EOS'.
    """
    if token_id == EOS_TOKEN:
        return None, None, -1, -1, "EOS"
    if not (0 <= token_id < NUM_LAYER_TOKENS):
        raise ValueError(f"token_id {token_id} out of range [0, {VOCAB_SIZE})")

    slot_idx = token_id // NUM_THICKNESSES
    thickness_idx = token_id % NUM_THICKNESSES
    thickness_nm = THICKNESSES[thickness_idx]

    material_name: Optional[str] = None
    if pool is not None and slot_idx < len(pool):
        material_name = pool[slot_idx].name

    return material_name, thickness_nm, slot_idx, thickness_idx, "LAYER"


def is_eos(token_id: int) -> bool:
    return token_id == EOS_TOKEN


# ============================================================================
# Thickness normalisation (unchanged from original)
# ============================================================================


def normalize_thickness(thickness_nm: int) -> float:
    """Normalize thickness to [0, 1] for use in the structure matrix."""
    return thickness_nm / MAX_THICKNESS_NM


def denormalize_thickness(thickness_norm: float) -> int:
    """Recover nm thickness from normalised value, snapped to the valid grid."""
    raw_nm = round(thickness_norm * MAX_THICKNESS_NM / _THICKNESS_STEP_NM) * _THICKNESS_STEP_NM
    return max(_THICKNESS_STEP_NM, min(MAX_THICKNESS_NM, int(raw_nm)))


# ============================================================================
# Structure matrix construction
# ============================================================================
#
# The structure matrix is the model's running record of "which slot is in
# use at which layer position". Same shape pattern as original CHROMA-Lite
# (NUM_MATERIALS × MAX_LAYERS), except the first dim is now M_MAX (slots,
# not fixed material ids), and the row at slot index k is meaningful only
# if that pool position is populated for the example.


def build_structure_matrix(
    slot_indices: List[int], thicknesses_nm: List[int]
) -> torch.Tensor:
    """Build a [M_MAX, MAX_LAYERS] structure matrix from a layer sequence.

    Parameters
    ----------
    slot_indices : list of int
        Slot index for each layer (in deposition order, layer 0 first).
    thicknesses_nm : list of int
        Thickness in nm for each layer.

    Returns
    -------
    matrix : torch.Tensor, shape [M_MAX, MAX_LAYERS], dtype float32
        matrix[s, l] = normalized_thickness if layer l uses slot s, else 0.
    """
    if len(slot_indices) != len(thicknesses_nm):
        raise ValueError("slot_indices and thicknesses_nm must have same length")
    if len(slot_indices) > MAX_LAYERS:
        raise ValueError(f"structure has {len(slot_indices)} layers, max is {MAX_LAYERS}")

    matrix = torch.zeros(M_MAX, MAX_LAYERS, dtype=torch.float32)
    for layer_idx, (slot, thick) in enumerate(zip(slot_indices, thicknesses_nm)):
        if not (0 <= slot < M_MAX):
            raise ValueError(f"slot {slot} out of range [0, {M_MAX})")
        matrix[slot, layer_idx] = normalize_thickness(thick)
    return matrix


def decode_structure_matrix(
    matrix: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    """Recover (slot_indices, thicknesses_nm) from a structure matrix.

    Stops at the first all-zero column (= no layer present). This mirrors
    the original CHROMA-Lite decoder.
    """
    slot_indices: List[int] = []
    thicknesses: List[int] = []
    for layer_idx in range(MAX_LAYERS):
        col = matrix[:, layer_idx]
        if col.sum().item() == 0:
            break
        slot = int(col.argmax().item())
        slot_indices.append(slot)
        thicknesses.append(denormalize_thickness(col[slot].item()))
    return slot_indices, thicknesses


# ============================================================================
# Lab normalization (the canonical color target for the model)
# ============================================================================
#
# Lab is wider gamut than sRGB and perceptually uniform (so CIEDE2000
# distances are meaningful). We scale into roughly [-1, 1] / [0, 1] for
# the MLP: L*/100 (in [0, 1]), a*/128 and b*/128 (in roughly [-1, 1] for
# colors near the sRGB gamut, possibly outside for wide-gamut targets).


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
# Output masking — for handling variable pool sizes
# ============================================================================


def build_output_mask(pool_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build a mask over the vocab that suppresses tokens for unused slots.

    Token i is valid iff i == EOS or its slot index < pool_size. Invalid
    tokens get -inf added to their logits before softmax.

    Parameters
    ----------
    pool_size : int
        Number of valid materials in the pool, ∈ [1, M_MAX].
    device : torch.device, optional

    Returns
    -------
    mask : torch.Tensor, shape [VOCAB_SIZE], dtype float32
        0.0 for valid tokens, -inf for invalid.
    """
    if not (1 <= pool_size <= M_MAX):
        raise ValueError(f"pool_size {pool_size} out of range [1, {M_MAX}]")

    mask = torch.full((VOCAB_SIZE,), float("-inf"), dtype=torch.float32, device=device)
    # Valid layer tokens: slot ∈ [0, pool_size).
    valid_layer_count = pool_size * NUM_THICKNESSES
    mask[:valid_layer_count] = 0.0
    # EOS is always valid.
    mask[EOS_TOKEN] = 0.0
    return mask


def build_output_mask_batch(
    pool_sizes: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Vectorised version of `build_output_mask` over a batch of pool sizes.

    Parameters
    ----------
    pool_sizes : torch.Tensor, shape [B], dtype int

    Returns
    -------
    mask : torch.Tensor, shape [B, VOCAB_SIZE]
    """
    B = pool_sizes.size(0)
    device = device or pool_sizes.device

    # Slot index for each token in the vocab. EOS gets a sentinel of -1
    # so the comparison below treats it as always valid.
    token_slots = torch.arange(VOCAB_SIZE, device=device) // NUM_THICKNESSES
    token_slots[EOS_TOKEN] = -1  # EOS is always valid

    # Compare each example's pool size against every token's slot.
    pool_sizes_b = pool_sizes.to(device).long().unsqueeze(1)  # [B, 1]
    valid = token_slots.unsqueeze(0) < pool_sizes_b           # [B, V]
    valid[:, EOS_TOKEN] = True                                # EOS always valid

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
    print(f"NUM_THICKNESSES  = {NUM_THICKNESSES}")
    print(f"NUM_LAYER_TOKENS = {NUM_LAYER_TOKENS}")
    print(f"EOS_TOKEN        = {EOS_TOKEN}")
    print(f"VOCAB_SIZE       = {VOCAB_SIZE}")

    # Round-trip a few tokens.
    print("\nencode/decode round-trip:")
    for slot, thick in [(0, 5), (3, 100), (M_MAX - 1, 200)]:
        tok = encode_layer(slot, thick)
        _, t_back, s_back, _, _ = decode_token(tok)
        assert s_back == slot and t_back == thick, "round-trip failed"
        print(f"  slot={slot}, thickness={thick}nm -> token {tok} -> "
              f"({s_back}, {t_back}nm) ✓")

    # EOS round-trip.
    _, _, _, _, kind = decode_token(EOS_TOKEN)
    assert kind == "EOS"
    print(f"  EOS ({EOS_TOKEN}) -> {kind} ✓")

    # Structure matrix round-trip.
    slots = [2, 5, 0, 7]
    thicks = [50, 100, 75, 200]
    M = build_structure_matrix(slots, thicks)
    s_back, t_back = decode_structure_matrix(M)
    assert s_back == slots and t_back == thicks
    print(f"\nstructure matrix shape: {tuple(M.shape)}")
    print(f"round-trip slots={slots} thicks={thicks}: ✓")

    # Mask correctness.
    print("\nmask test:")
    mask = build_output_mask(pool_size=3)
    valid_count = (mask == 0.0).sum().item()
    expected_valid = 3 * NUM_THICKNESSES + 1  # 3 slots × 40 thicknesses + EOS
    print(f"  pool_size=3: valid tokens = {int(valid_count)} (expected {expected_valid}) "
          f"{'✓' if valid_count == expected_valid else '✗'}")

    # Vectorized mask
    mask_batch = build_output_mask_batch(torch.tensor([1, 8, M_MAX]))
    print(f"  batched mask shape: {tuple(mask_batch.shape)}")
    counts = (mask_batch == 0.0).sum(dim=1).tolist()
    expected = [
        1 * NUM_THICKNESSES + 1,
        8 * NUM_THICKNESSES + 1,
        M_MAX * NUM_THICKNESSES + 1,
    ]
    print(f"  per-example valid counts: {counts} (expected {expected})")
    assert counts == expected, "batched mask mismatch"
    print("[smoke] OK")
