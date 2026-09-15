"""
ΔE Post-Training Finetune — Rollout + Loss
==========================================

The per-batch training loop for the post-training ΔE finetune stage.
See `analyses/de_finetune/README.md` for the full design (motivation,
credit-assignment fix, STE-on-both, PyTorch↔JAX bridge).

Per-batch flow
--------------
    for each (example b, layer position k in that example):
        # 1. Model forward, teacher-forced on GT (structure_matrix carries
        #    all N GT layers; the causal mask ensures position k only
        #    attends to 0..k-1). One forward per batch.
        logits = model(lab_b, pool_features, structure_matrix_gt)
        # 2. STE on both slot and thickness at position k
        slot_choice_k, thickness_nm_k, extras = ste_pick(logits[b, k, :])
        # 3. Assemble the full N-layer stack:
        #      layers 0..k-1  = GT slots + GT thicknesses (detached)
        #      layer k        = slot_choice_k + thickness_nm_k (differentiable)
        #      layers k+1..N-1 = GT slots + GT thicknesses (detached)
        n_stack, k_stack, thicknesses = assemble_full_stack(...)
        # 4. Differentiable sim → predicted Lab
        predicted_lab = differentiable_compute_lab(n_stack, k_stack, thicknesses)
        # 5. ΔE₀₀ against denormalised target Lab
        loss_b_k = ciede2000_torch(target_lab_denorm, predicted_lab)
    total_loss = sum(loss_b_k) / count
    total_loss.backward()

Gradient flows through layer k only. Layers before and after are frozen
ground-truth values that just contribute to the physics of the full-stack
Lab prediction. This mirrors the standard credit-assignment fix from
sequence-level RL (BLEU on full sequence, credit assigned per token) and
chain-of-thought training (grade only the final answer, credit
intermediate steps by their contribution).

Public entry points
-------------------
    finetune_de_loss(model, batch, ...):
        Compute the mean ΔE₀₀ across all (example, k) positions in a
        batch, plus auxiliary metrics (slot entropy, thickness entropy,
        top-1 slot / thickness match to GT). Backward-ready scalar loss.

    freeze_encoder_for_decoder_only(model):
        Set requires_grad=False on MaterialEncoder + slot_encoder;
        leaves decoder + thickness_head + eos_head trainable. Use for
        Experiment A.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.materials_vocab import (
    EOS_TOKEN,
    M_MAX,
    MAX_LAYERS,
    NUM_THICKNESSES,
    THICKNESSES,
    VOCAB_SIZE,
    build_structure_matrix,
    denormalize_lab,
    normalize_lab,
)
from src.optical_sim_diff import (
    compute_lab_no_grad,
    differentiable_compute_lab,
    is_available as sim_is_available,
)


# ============================================================================
# Constants (once, cached)
# ============================================================================

# Bin centers in nm — exactly the training grid so soft-thickness collapses
# to a real token value in the forward pass.
_THICKNESS_BIN_CENTERS_NP = np.asarray(THICKNESSES, dtype=np.float64)

# nm-value → bin-index reverse map for the CE anchor. GT thicknesses come
# out of the dataset as ints on the training grid, so exact lookup should
# always succeed; a stray value falls back to the nearest bin.
_NM_TO_BIN: Dict[int, int] = {int(nm): i for i, nm in enumerate(THICKNESSES)}


def _nm_to_bin(nm: int) -> int:
    v = _NM_TO_BIN.get(int(nm))
    if v is not None:
        return v
    diffs = np.abs(_THICKNESS_BIN_CENTERS_NP - float(nm))
    return int(np.argmin(diffs))


def _thickness_bin_centers(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(_THICKNESS_BIN_CENTERS_NP, device=device, dtype=dtype)


# ============================================================================
# STE (straight-through estimator) — hard forward, soft backward
# ============================================================================


def _ste_onehot(soft_probs: torch.Tensor) -> torch.Tensor:
    """Convert a softmax distribution to a straight-through one-hot vector.

    Forward: one-hot at argmax. Backward: gradient of soft_probs flows
    (the constant terms drop out under differentiation).
    """
    hard_idx = soft_probs.argmax(dim=-1)
    hard_onehot = F.one_hot(hard_idx, num_classes=soft_probs.shape[-1]).to(
        dtype=soft_probs.dtype,
    )
    return hard_onehot + soft_probs - soft_probs.detach()


def ste_pick(
    logits_step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """STE on both slot and thickness at one decoding position.

    Parameters
    ----------
    logits_step : torch.Tensor, shape [VOCAB_SIZE]
        Model output at one (example, position) — the 3201-way vocab
        (M_MAX slots × NUM_THICKNESSES bins + EOS).

    Returns
    -------
    slot_choice   : [M_MAX] — one-hot in forward, softmax gradient in backward
    thickness_nm  : []      — bin center in forward, differentiable in backward
    extras        : dict of diagnostic scalars (slot entropy, thickness entropy,
                    argmax slot idx, argmax thickness idx). Non-differentiable.
    """
    device = logits_step.device
    dtype = logits_step.dtype
    bin_centers = _thickness_bin_centers(device, dtype)

    # Sanitize -inf from the model's output mask. `apply_output_mask=True`
    # writes -inf into padded-slot logits. The slot STE tolerates that
    # (softmax over -inf entries drops them to 0), but the thickness STE
    # multiplies slot_choice (0 at padded rows) by layer_logits (-inf at
    # padded rows), and IEEE-754 gives 0 * -inf = NaN — which then
    # contaminates every bin under sum, softmax, and every downstream op.
    # Replace with a large-negative-but-finite floor: still ~0 probability
    # under softmax, safe under multiplication.
    logits_step = torch.nan_to_num(logits_step, neginf=-1e9, posinf=1e9)

    # Split the vocab: first M_MAX*NUM_THICKNESSES tokens are (slot, thickness)
    # pairs (row-major: token = slot * NUM_THICKNESSES + thickness_bin);
    # the last is EOS.
    layer_logits = logits_step[:M_MAX * NUM_THICKNESSES].view(
        M_MAX, NUM_THICKNESSES,
    )

    # --- Slot: STE over argmax slot ---
    # Slot "score" = max-thickness logit per slot (a simple pooling for
    # slot preference; symmetric with how greedy inference picks tokens).
    slot_scores = layer_logits.max(dim=-1).values                  # [M_MAX]
    slot_probs = F.softmax(slot_scores, dim=-1)                    # [M_MAX]
    slot_choice = _ste_onehot(slot_probs)                          # [M_MAX]

    # --- Thickness: STE over argmax bin (conditioned on chosen slot) ---
    # (slot_choice.unsqueeze(-1) * layer_logits) is hard-slot in the forward
    # (all zeros except the chosen row), and softmax-differentiable in
    # backward.
    conditioned_thick_logits = (
        slot_choice.unsqueeze(-1) * layer_logits
    ).sum(dim=0)                                                   # [NUM_THICKNESSES]
    thick_probs = F.softmax(conditioned_thick_logits, dim=-1)      # [NUM_THICKNESSES]
    thick_choice = _ste_onehot(thick_probs)                        # [NUM_THICKNESSES]
    thickness_nm = (thick_choice * bin_centers).sum()              # scalar

    # Diagnostics (not part of the gradient path).
    with torch.no_grad():
        slot_entropy = -(slot_probs * (slot_probs.clamp_min(1e-12).log())).sum()
        thick_entropy = -(
            thick_probs * (thick_probs.clamp_min(1e-12).log())
        ).sum()
        extras = {
            "slot_entropy": slot_entropy.detach(),
            "thickness_entropy": thick_entropy.detach(),
            "slot_argmax": slot_choice.argmax().detach(),
            "thickness_argmax": thick_choice.argmax().detach(),
        }

    return slot_choice, thickness_nm, extras


# ============================================================================
# ΔE₀₀ in torch (differentiable)
# ============================================================================


def ciede2000_torch(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between two Lab triples. Fully differentiable.

    Mirrors src.evaluate.ciede2000 exactly. Both inputs must be
    torch.Tensors of shape [3]; returns a scalar tensor.
    """
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]
    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - torch.sqrt(C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)

    def _hue(y, x):
        # atan2 in degrees, mod 360
        h = torch.rad2deg(torch.atan2(y, x))
        return torch.remainder(h, 360.0)

    h1p = _hue(b1, a1p)
    h2p = _hue(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    # dh' with the CIEDE branch for near-zero chroma / cross-360 wraps.
    zero_chroma = (C1p * C2p) == 0
    raw_dh = h2p - h1p
    dhp = torch.where(
        torch.abs(raw_dh) > 180.0,
        torch.where(raw_dh > 180.0, raw_dh - 360.0, raw_dh + 360.0),
        raw_dh,
    )
    dhp = torch.where(zero_chroma, torch.zeros_like(dhp), dhp)

    dHp = 2.0 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dhp / 2.0))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)

    # h_bar' with the CIEDE branch (paired with the dh' branch above).
    h_diff = torch.abs(h1p - h2p)
    h_sum = h1p + h2p
    hbp_base = 0.5 * h_sum
    hbp = torch.where(
        (~zero_chroma) & (h_diff > 180.0),
        torch.where(h_sum < 360.0, hbp_base + 180.0, hbp_base - 180.0),
        hbp_base,
    )

    T = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(hbp - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * hbp))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * hbp + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * hbp - 63.0))
    )
    dTheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    R_C = 2.0 * torch.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    S_L = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / torch.sqrt(20.0 + (Lbp - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cbp
    S_H = 1.0 + 0.015 * Cbp * T
    R_T = -torch.sin(torch.deg2rad(2.0 * dTheta)) * R_C

    return torch.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


# ============================================================================
# Freezing helpers (for Experiment A: decoder-only)
# ============================================================================


def freeze_encoder_for_decoder_only(model) -> Dict[str, int]:
    """Freeze the pool encoder (MaterialEncoder + slot_encoder) so only
    the decoder + thickness_head + eos_head train.

    Works on FlexMaterialCrossAttn (production head). Returns a dict
    of {'frozen': N, 'trainable': M} for logging.

    Structure of the cross-attn head (see src/model.py:
    FlexMaterialCrossAttn):
        material_encoder  <- per-slot MaterialEncoder    (FREEZE)
        slot_encoder      <- transformer over M slots    (FREEZE)
        decoder           <- causal + cross-attn         (TRAIN)
        thickness_head    <- per-slot thickness MLP      (TRAIN)
        eos_head          <- EOS MLP                     (TRAIN)
    """
    frozen = 0
    trainable = 0
    freeze_prefixes = ("material_encoder", "slot_encoder")
    for name, p in model.named_parameters():
        if any(name.startswith(pref) or f".{pref}." in f".{name}"
               for pref in freeze_prefixes):
            p.requires_grad_(False)
            frozen += p.numel()
        else:
            p.requires_grad_(True)
            trainable += p.numel()
    return {"frozen": frozen, "trainable": trainable}


def unfreeze_all(model) -> Dict[str, int]:
    """Undo any freezing (for Experiment B / eval)."""
    total = 0
    for p in model.parameters():
        p.requires_grad_(True)
        total += p.numel()
    return {"trainable": total}


# ============================================================================
# CE anchor loss — keeps the finetune close to the pretrained CE optimum
# ============================================================================


def _build_ce_targets(
    target_slots_batch: List[List[int]],
    target_thicknesses_batch: List[List[int]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a [B, seq_len] target-token tensor for cross-entropy against
    the model's per-position logits.

    Token layout matches the vocab in src.materials_vocab: for a layer
    position, target = slot * NUM_THICKNESSES + bin_index(thickness_nm).
    Positions beyond the example's n_layers are marked -100 so torch's
    cross-entropy ignores them (ignore_index=-100 by default there too).

    We do NOT include the EOS position in the CE target for now — we're
    training the layer picks, not re-teaching EOS. That's consistent with
    the ΔE loss which also only runs over layer positions 0..n_layers-1.
    """
    batch_size = len(target_slots_batch)
    targets = torch.full(
        (batch_size, seq_len), -100, dtype=torch.long, device=device,
    )
    for b in range(batch_size):
        gt_slots = target_slots_batch[b]
        gt_thicknesses = target_thicknesses_batch[b]
        n = min(len(gt_slots), min(seq_len, MAX_LAYERS))
        for k in range(n):
            slot = int(gt_slots[k])
            bin_idx = _nm_to_bin(int(gt_thicknesses[k]))
            targets[b, k] = slot * NUM_THICKNESSES + bin_idx
    return targets


def ce_anchor_loss(
    logits: torch.Tensor,                    # [B, seq_len, VOCAB]
    target_slots_batch: List[List[int]],
    target_thicknesses_batch: List[List[int]],
) -> torch.Tensor:
    """Per-position cross-entropy against GT tokens, averaged over all
    valid layer positions in the batch. Same shape/semantics as the
    pretrain CE loss.

    The logits come from the model's forward with GT structure_matrix
    (teacher forcing), so the causal mask makes position k's prediction
    conditioned on GT layers 0..k-1 — exactly the pretrain setup. This
    is the same tensor already computed for the ΔE loss; no extra
    forward is needed.
    """
    device = logits.device
    seq_len = logits.shape[1]
    targets = _build_ce_targets(
        target_slots_batch, target_thicknesses_batch, seq_len, device,
    )
    # F.cross_entropy handles -100 via ignore_index (default), reduction
    # 'mean' averages over all non-ignored positions.
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ============================================================================
# NaN diagnostics — helps first-run debugging without spamming logs.
# Set _NAN_DEBUG_LIMIT to 0 (or via env INDIGO_NAN_DEBUG=0) to silence.
# ============================================================================

import os as _os

_NAN_DEBUG_LIMIT = int(_os.environ.get("INDIGO_NAN_DEBUG", "5"))
_NAN_DEBUG_COUNT = 0


def _log_nan(**fields) -> None:
    """Print a compact one-shot dump for the first N NaN events per process."""
    global _NAN_DEBUG_COUNT
    if _NAN_DEBUG_COUNT >= _NAN_DEBUG_LIMIT:
        return
    _NAN_DEBUG_COUNT += 1
    print(f"[NAN-DEBUG #{_NAN_DEBUG_COUNT}]", flush=True)
    for k, v in fields.items():
        print(f"    {k}: {v}", flush=True)


# ============================================================================
# Per-example rollout — internal helper
# ============================================================================


def _rollout_one(
    logits_all_positions: torch.Tensor,  # [MAX_LAYERS + 1, VOCAB_SIZE]
    pool_n: torch.Tensor,                # [M_MAX, NUM_LAMBDA]
    pool_k_ch: torch.Tensor,             # [M_MAX, NUM_LAMBDA]
    gt_slots: List[int],                 # length N (== num true layers)
    gt_thicknesses: List[int],           # length N
    target_lab_denorm: torch.Tensor,     # [3]
    incidence_angle: float,
) -> Tuple[torch.Tensor, Dict[str, float], int]:
    """Sum ΔE loss across all layer positions of one example.

    Returns:
        total_loss   — scalar sum of ΔE₀₀ across k=0..N-1
        metrics      — averaged diagnostics
        n_positions  — N (used for correct mean aggregation upstream)
    """
    device = pool_n.device
    dtype = pool_n.dtype
    n_layers = len(gt_slots)
    if n_layers == 0:
        return (
            torch.zeros((), device=device, dtype=dtype),
            {}, 0,
        )

    # Precompute GT one-hots and thickness scalars (constants in the sim).
    gt_slot_onehots = F.one_hot(
        torch.tensor(gt_slots, device=device), num_classes=M_MAX,
    ).to(dtype=dtype)                                 # [N, M_MAX]
    gt_thicknesses_nm = torch.tensor(
        gt_thicknesses, device=device, dtype=dtype,
    )                                                 # [N]

    # Assemble the GT layer n/k stack once — [N, NUM_LAMBDA].
    gt_n_stack = gt_slot_onehots @ pool_n              # [N, NUM_LAMBDA]
    gt_k_stack = gt_slot_onehots @ pool_k_ch           # [N, NUM_LAMBDA]

    losses = []
    per_pos_diag: List[Dict[str, float]] = []
    for k in range(n_layers):
        step_logits = logits_all_positions[k]          # [VOCAB_SIZE]
        slot_choice_k, thickness_nm_k, extras = ste_pick(step_logits)

        # Layer-k n/k comes from the STE'd slot pick.
        n_layer_k = slot_choice_k @ pool_n             # [NUM_LAMBDA]
        k_layer_k = slot_choice_k @ pool_k_ch          # [NUM_LAMBDA]

        # Splice layer k into the GT stack. Prefix and suffix are
        # detached — no gradient flows through them.
        n_stack_full = torch.cat([
            gt_n_stack[:k].detach(),
            n_layer_k.unsqueeze(0),
            gt_n_stack[k + 1:].detach(),
        ], dim=0)                                      # [N, NUM_LAMBDA]
        k_stack_full = torch.cat([
            gt_k_stack[:k].detach(),
            k_layer_k.unsqueeze(0),
            gt_k_stack[k + 1:].detach(),
        ], dim=0)
        thicknesses_full = torch.cat([
            gt_thicknesses_nm[:k].detach(),
            thickness_nm_k.unsqueeze(0),
            gt_thicknesses_nm[k + 1:].detach(),
        ], dim=0)                                      # [N]

        # Sim -> Lab (differentiable through layer k).
        predicted_lab = differentiable_compute_lab(
            n_stack_full, k_stack_full, thicknesses_full,
            incidence_angle=incidence_angle,
        )
        loss_k = ciede2000_torch(target_lab_denorm, predicted_lab)
        losses.append(loss_k)

        # Diagnose first few NaN/Inf events so we can find the offending
        # tensor. Silenced after _NAN_DEBUG_LIMIT dumps per process.
        if not torch.isfinite(loss_k):
            with torch.no_grad():
                argmax_slot = int(extras["slot_argmax"].item())
                argmax_thick_bin = int(extras["thickness_argmax"].item())
                _log_nan(
                    k=k,
                    n_layers=n_layers,
                    argmax_slot=argmax_slot,
                    argmax_thick_bin=argmax_thick_bin,
                    thickness_nm_k=float(thickness_nm_k.item()),
                    target_lab=target_lab_denorm.detach().cpu().tolist(),
                    predicted_lab=predicted_lab.detach().cpu().tolist(),
                    predicted_lab_finite=bool(
                        torch.isfinite(predicted_lab).all().item()
                    ),
                    n_layer_k_finite=bool(
                        torch.isfinite(n_layer_k).all().item()
                    ),
                    k_layer_k_finite=bool(
                        torch.isfinite(k_layer_k).all().item()
                    ),
                    n_layer_k_stats=[
                        float(n_layer_k.min().item()),
                        float(n_layer_k.max().item()),
                        float(n_layer_k.mean().item()),
                    ],
                    k_layer_k_stats=[
                        float(k_layer_k.min().item()),
                        float(k_layer_k.max().item()),
                        float(k_layer_k.mean().item()),
                    ],
                    gt_slots=gt_slots,
                    gt_thicknesses=gt_thicknesses,
                    prefix_n_finite=bool(
                        torch.isfinite(gt_n_stack[:k]).all().item()
                    ) if k > 0 else True,
                    suffix_n_finite=bool(
                        torch.isfinite(gt_n_stack[k + 1:]).all().item()
                    ) if k + 1 < n_layers else True,
                    thicknesses_full=thicknesses_full.detach().cpu().tolist(),
                    argmax_slot_pool_n_finite=bool(
                        torch.isfinite(pool_n[argmax_slot]).all().item()
                    ),
                    argmax_slot_pool_n_stats=[
                        float(pool_n[argmax_slot].min().item()),
                        float(pool_n[argmax_slot].max().item()),
                        float(pool_n[argmax_slot].mean().item()),
                    ],
                )

        with torch.no_grad():
            per_pos_diag.append({
                "slot_entropy": extras["slot_entropy"].item(),
                "thickness_entropy": extras["thickness_entropy"].item(),
                "slot_match_gt": float(
                    extras["slot_argmax"].item() == gt_slots[k]
                ),
                "thickness_match_gt": float(
                    THICKNESSES[extras["thickness_argmax"].item()]
                    == gt_thicknesses[k]
                ),
                "loss_de": loss_k.item(),
            })

    total_loss = torch.stack(losses).sum()

    # Aggregate metrics across positions.
    if per_pos_diag:
        keys = per_pos_diag[0].keys()
        metrics = {k: float(np.mean([d[k] for d in per_pos_diag])) for k in keys}
    else:
        metrics = {}
    return total_loss, metrics, n_layers


# ============================================================================
# Sim-feedback residual — helper to compute partial-stack Lab residuals
# ============================================================================
#
# Called once per batch before the model forward. For every example, for
# every position k in the batched sequence layout [0 .. MAX_LAYERS], we
# compute:
#
#     residual[b, k] = normalized(target_lab_b) − normalized(sim(GT[0:k]))
#
# where sim(GT[0:k]) is the real-JLL Lab of the stack formed by the first
# k GT layers of example b.
#
#   • k = 0: prefix is empty. Residual is undefined; we set it to 0. The
#     model at position k=0 only sees the target and has no "residual" to
#     react to.
#   • k = N_b (past this example's real layer count): position doesn't
#     make a prediction that ever contributes to loss. Residual = 0.
#
# Residual is in NORMALISED Lab space (matching the model's `lab` input)
# because the residual projection lives in that same feature space.
#
# Cost: N_b − 1 partial sims per example. For a 4-layer example that's
# 3 partial sims — ~25% overhead on top of the K sims/position for the
# top-K path. Trivial vs the value of "how am I doing" feedback.
# ============================================================================


def _compute_partial_residuals(
    batch: Dict[str, torch.Tensor],
    incidence_angle: float,
    device: torch.device,
    prefix_aug_prob: float = 0.0,
    prefix_aug_thickness_scale: float = 0.15,
) -> torch.Tensor:
    """Compute [B, MAX_LAYERS+1, 3] partial-Lab residuals in normalised
    Lab space. Uses real (non-differentiable) JLL sim on GT prefixes.

    prefix_aug_prob > 0 enables prefix augmentation: for each example,
    each prefix layer's thickness is independently jittered with prob
    p_aug by multiplying by Uniform(1-scale, 1+scale). The perturbed
    thicknesses are used ONLY to compute the residual sim'd here — the
    model's teacher-forced input tokens and the top-K target computation
    both still use true GT. This trains the model to consume a noisy
    residual channel, narrowing the gap between the clean train
    residual (sim(GT_prefix)) and the noisier val/inference residual
    (sim(model_prefix)). Cost is unchanged vs the no-aug path.
    """
    from src.optical_sim_diff import compute_lab_no_grad

    from src.materials_vocab import _L_SCALE, _AB_SCALE  # local: avoid cycle
    scale = torch.tensor(
        [_L_SCALE, _AB_SCALE, _AB_SCALE], device=device, dtype=torch.float64,
    )

    B = batch["lab"].size(0)
    SEQ_LEN = MAX_LAYERS + 1
    residuals = torch.zeros(B, SEQ_LEN, 3, device=device, dtype=torch.float32)

    pool_features = batch["pool_features"].to(device=device, dtype=torch.float64)

    aug_on = float(prefix_aug_prob) > 0.0
    aug_scale = max(0.0, float(prefix_aug_thickness_scale))

    for b in range(B):
        gt_slots = list(batch["target_slots"][b])[:MAX_LAYERS]
        gt_thicknesses = list(batch["target_thicknesses"][b])[:MAX_LAYERS]
        n_layers = len(gt_slots)
        if n_layers == 0:
            continue

        # Denormalised target Lab for this example.
        target_lab_denorm = torch.tensor(
            denormalize_lab(batch["lab"][b]),
            device=device, dtype=torch.float64,
        )

        # Prefix n/k stacks — GT one-hot selects the material.
        pool_n_b = pool_features[b, :, 0, :]                        # [M_MAX, L]
        pool_k_b = pool_features[b, :, 1, :]
        gt_onehots = F.one_hot(
            torch.tensor(gt_slots, device=device), num_classes=M_MAX,
        ).to(dtype=torch.float64)                                   # [N, M_MAX]
        gt_n_stack = gt_onehots @ pool_n_b                          # [N, L]
        gt_k_stack = gt_onehots @ pool_k_b
        gt_thick_nm = torch.tensor(
            gt_thicknesses, device=device, dtype=torch.float64,
        )                                                            # [N]

        # Prefix augmentation: perturb the thickness we'll sim through
        # for the residual signal only. One draw per prefix layer,
        # coherent across k so growing prefixes see a consistent noisy
        # trajectory (matches inference where the model's prefix state
        # persists across positions). Materials are NOT swapped —
        # material errors produce residuals too large / off-distribution
        # to be useful supervision here.
        sim_thick_nm = gt_thick_nm
        if aug_on and aug_scale > 0.0:
            perturb_mask = (
                torch.rand(n_layers, device=device, dtype=torch.float64)
                < prefix_aug_prob
            )
            if perturb_mask.any():
                # Multiplicative jitter in [1-scale, 1+scale].
                jitter = (
                    torch.rand(n_layers, device=device, dtype=torch.float64)
                    * (2.0 * aug_scale) + (1.0 - aug_scale)
                )
                factor = torch.where(
                    perturb_mask, jitter,
                    torch.ones(n_layers, device=device, dtype=torch.float64),
                )
                sim_thick_nm = gt_thick_nm * factor

        # For each k in 1..n_layers-1: sim the first k prefix layers,
        # get Lab, compute residual = (target − partial) / scale. k=0
        # stays 0. For k = n_layers .. MAX_LAYERS: keep 0.
        for k in range(1, n_layers):
            partial_lab = compute_lab_no_grad(
                gt_n_stack[:k], gt_k_stack[:k], sim_thick_nm[:k],
                incidence_angle=incidence_angle,
            )                                                       # [3]
            residual_denorm = target_lab_denorm - partial_lab
            residual_norm = (residual_denorm / scale).to(dtype=torch.float32)
            residuals[b, k, :] = residual_norm
    return residuals


# ============================================================================
# Top-K real-sim loss — replaces the STE linearization with actual ΔE
# ============================================================================
#
# Motivation. The STE loss is a linearization of ΔE at the model's argmax
# pick: it uses a single sim per position and projects the gradient onto
# every candidate via the winning material's (n, k) directions. The Sept 8
# STE-projection diagnostic showed only ~41% top-1 accuracy at model-argmax
# anchors and ~80% sign agreement, and the Sept 9 λ ∈ {0.1, 1, 10, 100}
# sweep confirmed the finetune drifts *away* from the pretrain manifold at
# every ΔE weight — the linearization is unusable as a training signal.
#
# The top-K real-sim loss removes the linearization entirely:
#
#   1. Per position, marginalize joint logits to per-slot scores
#      (max-over-thickness — same pool as ste_pick).
#   2. Take the top-K slot candidates (padded slots masked out).
#   3. For each candidate, use the model's conditional-argmax thickness
#      for that slot; splice into GT prefix/suffix; run a real sim.
#   4. Compute real ΔE₀₀ per candidate (K reals per position, stop-grad).
#   5. Loss = listwise CE(softmax(slot_scores[topK]) ‖ softmax(-β·ΔE)):
#      gradient flows only through the model's slot logits at the top-K
#      indices, pushed toward the actual argmin-ΔE candidate.
#
# Cost. K sims per position (vs 1 for STE) — K=5 is ~5× the sim cost,
# K=15 ~15×. Model fwd/bwd cost is unchanged. Thickness gradient is
# dropped from this loss; pair with CE anchor (ce_loss_weight > 0) to
# keep thickness training if desired.
# ============================================================================


def _topK_sim_loss_for_example(
    logits_all_positions: torch.Tensor,  # [MAX_LAYERS+1, VOCAB_SIZE]  fp64
    pool_n: torch.Tensor,                # [M_MAX, NUM_LAMBDA]  fp64
    pool_k_ch: torch.Tensor,             # [M_MAX, NUM_LAMBDA]  fp64
    pool_mask: torch.Tensor,             # [M_MAX]  fp64
    gt_slots: List[int],                 # length N
    gt_thicknesses: List[int],           # length N
    target_lab_denorm: torch.Tensor,     # [3]  fp64
    incidence_angle: float,
    top_k: int,
    beta: float,
    topk_mode: str = "slot",             # "slot" | "joint" | "hierarchical"
    epsilon: float = 0.0,                # ε-exploration fraction
    thickness_topn: int = 1,             # N thicknesses per slot in "hierarchical" mode
) -> Tuple[torch.Tensor, Dict[str, float], int]:
    """Per-example top-K real-sim loss. Returns (sum_loss, metrics, n_pos).

    Three modes:
      "slot"          — top-K over the marginal per-slot score (max over
                        thickness). Uses each slot's argmax-thickness bin.
                        Gradient hits K joint cells: (slot, argmax_thickness).
      "joint"         — top-K over the flat (slot × thickness) logit grid
                        (M_MAX × NUM_THICKNESSES cells). Sept 10 finding:
                        this concentrates on 1-2 slots' neighboring
                        thickness bins → all candidates similar ΔE →
                        loss_topk stuck at log(K) uniform. Kept for
                        completeness but not recommended.
      "hierarchical"  — top-K slots by marginal (like "slot") AND top-N
                        thicknesses per slot. Total = K · N candidates,
                        all with distinct (slot, thickness). Forces
                        material diversity (K slots) AND thickness
                        exploration (N per slot). Sim cost = K · N per
                        position; gradient hits K · N joint cells.
                        Set `thickness_topn > 1` to enable; N=1 is
                        equivalent to "slot" mode.

    ε-exploration (RL-inspired). If `epsilon > 0`, replace
    floor(K · epsilon) of the K candidates with uniform-random draws
    over the active grid (never duplicating a top-K pick). The random
    picks broaden the search — if a random candidate has low ΔE, the
    target softmax(-β·ΔE) puts weight on it and the model gets a
    strong "raise this logit" gradient, escaping local minima where
    top-K is always similar. Same loss form (softmax over all K) so
    no code path changes downstream. Caller anneals epsilon over
    training (typical schedule: 0.3 → 0.0).

    `metrics["loss_de"]` mirrors the STE path's semantics — the ΔE at
    the model's greedy (argmax slot × argmax thickness) pick — so
    `val_loss_de` stays comparable across training modes.
    """
    device = pool_n.device
    dtype_sim = pool_n.dtype
    n_layers = len(gt_slots)
    if n_layers == 0:
        return (
            torch.zeros((), device=device, dtype=dtype_sim),
            {}, 0,
        )

    # GT prefix/suffix (constants — no grad, detached where used).
    gt_slot_onehots = F.one_hot(
        torch.tensor(gt_slots, device=device), num_classes=M_MAX,
    ).to(dtype=dtype_sim)                              # [N, M_MAX]
    gt_thicknesses_nm = torch.tensor(
        gt_thicknesses, device=device, dtype=dtype_sim,
    )                                                  # [N]
    gt_n_stack = gt_slot_onehots @ pool_n              # [N, NUM_LAMBDA]
    gt_k_stack = gt_slot_onehots @ pool_k_ch           # [N, NUM_LAMBDA]

    active_mask_bool = pool_mask.bool()                # [M_MAX]
    n_active = int(active_mask_bool.sum().item())
    if topk_mode == "joint":
        K_eff = max(1, min(top_k, n_active * NUM_THICKNESSES))
    else:
        K_eff = max(1, min(top_k, n_active))
    # Hierarchical mode: N thicknesses per slot. Total candidates =
    # K_eff * N_thick. N_thick clamped to NUM_THICKNESSES so we can't
    # ask for more than exist.
    N_thick = max(1, min(int(thickness_topn), NUM_THICKNESSES))
    is_hierarchical = (topk_mode == "hierarchical") and N_thick > 1

    # ε-exploration slot budget. floor(K · ε) of the K candidates are
    # uniform-random draws; the rest come from top-K by logit. Clamped
    # so K_top ≥ 1 (never fully-random — the model's own picks are
    # what we're training).
    K_random = int(K_eff * max(0.0, min(1.0, epsilon)))
    K_random = min(K_random, K_eff - 1)  # keep at least 1 top-K pick
    K_top = K_eff - K_random

    losses = []
    per_pos_diag: List[Dict[str, float]] = []

    for k in range(n_layers):
        # Sanitize -inf just like ste_pick does (padded-slot masking).
        step_logits = torch.nan_to_num(
            logits_all_positions[k], neginf=-1e9, posinf=1e9,
        )
        layer_logits = step_logits[:M_MAX * NUM_THICKNESSES].view(
            M_MAX, NUM_THICKNESSES,
        )                                              # [M_MAX, NUM_THICKNESSES]

        if is_hierarchical:
            # Hierarchical M × N: K slots × N_thick thickness bins per
            # slot. Every candidate has a distinct (slot, thick) pair →
            # material diversity AND thickness exploration.
            # ε-exploration applies at the SLOT level (matching slot
            # mode): floor(K_eff · ε) of the K slots are uniform-random
            # draws; each such slot still gets its own top-N thickness
            # bins. This diversifies material choice; the "random slot
            # + argmax thick" candidates might have high ΔE, but the
            # softmax target then puts weight on the top-scored slots
            # regardless, so the exploration is bounded.
            slot_scores = layer_logits.max(dim=-1).values      # [M_MAX]
            slot_scores_masked = slot_scores.masked_fill(
                ~active_mask_bool, -1e9,
            )
            top_by_logit = torch.topk(
                slot_scores_masked, K_top, dim=-1,
            ).indices                                          # [K_top]
            if K_random > 0:
                active_slots = torch.nonzero(
                    active_mask_bool, as_tuple=False,
                ).squeeze(-1)                                  # [n_active]
                top_set = torch.zeros(
                    active_mask_bool.numel(), dtype=torch.bool,
                    device=device,
                )
                top_set[top_by_logit] = True
                pool_for_random = active_slots[~top_set[active_slots]]
                k_r = min(K_random, pool_for_random.numel())
                if k_r > 0:
                    perm = torch.randperm(
                        pool_for_random.numel(), device=device,
                    )[:k_r]
                    random_picks = pool_for_random[perm]
                    topk_slots_only = torch.cat(
                        [top_by_logit, random_picks], dim=0,
                    )
                else:
                    topk_slots_only = top_by_logit
            else:
                topk_slots_only = top_by_logit
            # For each selected slot, top-N thickness bins by conditional
            # logit. Shape [K_slots, N_thick].
            per_slot_thick_logits = layer_logits[topk_slots_only]  # [K_s, NT]
            topk_thick_bins_per_slot = torch.topk(
                per_slot_thick_logits, N_thick, dim=-1,
            ).indices                                              # [K_s, N_thick]
            # Flatten to lists of (slot, thick_bin) pairs of length K_s * N_thick.
            topk_slots = topk_slots_only.repeat_interleave(N_thick)  # [K_s*N_thick]
            topk_thick_bins = topk_thick_bins_per_slot.reshape(-1)   # [K_s*N_thick]
            # Model logits over the K·N joint cells, differentiable.
            # gather along last dim so grad flows into each cell.
            topk_model_logits = per_slot_thick_logits.gather(
                dim=-1, index=topk_thick_bins_per_slot,
            ).reshape(-1)                                           # [K_s*N_thick]
        elif topk_mode == "joint":
            # Top-K over the flattened joint (slot × thickness) grid.
            joint_logits = layer_logits.reshape(-1)   # [M_MAX*NUM_THICKNESSES]
            joint_slot_mask = active_mask_bool.unsqueeze(-1).expand(
                -1, NUM_THICKNESSES,
            ).reshape(-1)                              # [M_MAX*NUM_THICKNESSES]
            joint_masked = joint_logits.masked_fill(
                ~joint_slot_mask, -1e9,
            )
            top_by_logit = torch.topk(
                joint_masked, K_top, dim=-1,
            ).indices                                  # [K_top]
            if K_random > 0:
                # Uniform-random draws over active grid, excluding
                # the already-selected top-K_top picks.
                active_positions = torch.nonzero(
                    joint_slot_mask, as_tuple=False,
                ).squeeze(-1)                          # [n_active * NUM_THICKNESSES]
                # Exclude top_by_logit via a membership mask.
                top_set = torch.zeros(
                    joint_slot_mask.numel(), dtype=torch.bool,
                    device=device,
                )
                top_set[top_by_logit] = True
                pool_for_random = active_positions[~top_set[active_positions]]
                k_r = min(K_random, pool_for_random.numel())
                if k_r > 0:
                    perm = torch.randperm(
                        pool_for_random.numel(), device=device,
                    )[:k_r]
                    random_picks = pool_for_random[perm]
                    topk_joint = torch.cat(
                        [top_by_logit, random_picks], dim=0,
                    )
                else:
                    topk_joint = top_by_logit
            else:
                topk_joint = top_by_logit
            topk_slots = torch.div(
                topk_joint, NUM_THICKNESSES, rounding_mode="floor",
            )                                          # [K_eff]
            topk_thick_bins = topk_joint % NUM_THICKNESSES  # [K_eff]
            # Model logits over the top-K joint cells — differentiable
            # slice; grad flows into each (slot, thickness) joint cell.
            topk_model_logits = joint_logits[topk_joint]  # [K_eff]
        else:
            # Slot mode: per-slot score = max over thickness (matches
            # ste_pick). Uses argmax-thick per top-K slot.
            slot_scores = layer_logits.max(dim=-1).values      # [M_MAX]
            slot_scores_masked = slot_scores.masked_fill(
                ~active_mask_bool, -1e9,
            )
            top_by_logit = torch.topk(
                slot_scores_masked, K_top, dim=-1,
            ).indices                                          # [K_top]
            if K_random > 0:
                active_slots = torch.nonzero(
                    active_mask_bool, as_tuple=False,
                ).squeeze(-1)                                  # [n_active]
                top_set = torch.zeros(
                    active_mask_bool.numel(), dtype=torch.bool,
                    device=device,
                )
                top_set[top_by_logit] = True
                pool_for_random = active_slots[~top_set[active_slots]]
                k_r = min(K_random, pool_for_random.numel())
                if k_r > 0:
                    perm = torch.randperm(
                        pool_for_random.numel(), device=device,
                    )[:k_r]
                    random_picks = pool_for_random[perm]
                    topk_slots = torch.cat(
                        [top_by_logit, random_picks], dim=0,
                    )
                else:
                    topk_slots = top_by_logit
            else:
                topk_slots = top_by_logit
            topk_thick_bins = layer_logits[topk_slots].argmax(dim=-1)
            # Differentiable per-slot logits (via .max above).
            topk_model_logits = slot_scores[topk_slots]        # [K_eff]

        # Fixed prefix/suffix pieces (detached — no grad through these).
        prefix_n = gt_n_stack[:k].detach()
        prefix_k = gt_k_stack[:k].detach()
        suffix_n = gt_n_stack[k + 1:].detach()
        suffix_k = gt_k_stack[k + 1:].detach()
        prefix_t = gt_thicknesses_nm[:k].detach()
        suffix_t = gt_thicknesses_nm[k + 1:].detach()

        # K sims, real ΔE per candidate. All stop-gradient (targets).
        # Use actual topk_slots length — may be < K_eff if the
        # pool_for_random branch had fewer positions available than
        # K_random (very rare for typical pools).
        K_actual = topk_slots.numel()
        delta_e_candidates: List[torch.Tensor] = []
        for c in range(K_actual):
            slot_idx = int(topk_slots[c].item())
            thick_bin = int(topk_thick_bins[c].item())
            thick_nm = float(_THICKNESS_BIN_CENTERS_NP[thick_bin])

            candidate_n = pool_n[slot_idx].unsqueeze(0)     # [1, NUM_LAMBDA]
            candidate_k = pool_k_ch[slot_idx].unsqueeze(0)
            candidate_t = torch.tensor(
                [thick_nm], device=device, dtype=dtype_sim,
            )
            n_stack_full = torch.cat([prefix_n, candidate_n, suffix_n], dim=0)
            k_stack_full = torch.cat([prefix_k, candidate_k, suffix_k], dim=0)
            t_full = torch.cat([prefix_t, candidate_t, suffix_t], dim=0)

            with torch.no_grad():
                lab_c = compute_lab_no_grad(
                    n_stack_full, k_stack_full, t_full,
                    incidence_angle=incidence_angle,
                )
                de_c = ciede2000_torch(target_lab_denorm, lab_c)
            delta_e_candidates.append(de_c.detach())

        delta_e = torch.stack(delta_e_candidates)      # [K_eff]

        # Target distribution: peaked on argmin ΔE, sharpness = β.
        target_probs = F.softmax(-beta * delta_e, dim=-1)  # [K_eff]

        topk_log_probs = F.log_softmax(topk_model_logits, dim=-1)
        loss_k = -(target_probs.detach() * topk_log_probs).sum()
        losses.append(loss_k)

        with torch.no_grad():
            # Model's own greedy pick — over the joint grid so it matches
            # inference-time greedy semantics (argmax joint gives the
            # (slot, thick) pair that would be picked at inference).
            joint_full = layer_logits.reshape(-1)
            joint_full_masked = joint_full.masked_fill(
                ~active_mask_bool.unsqueeze(-1).expand(
                    -1, NUM_THICKNESSES,
                ).reshape(-1),
                -1e9,
            )
            model_argmax_joint = int(joint_full_masked.argmax().item())
            model_argmax_slot = model_argmax_joint // NUM_THICKNESSES
            model_argmax_thick_bin = model_argmax_joint % NUM_THICKNESSES

            # Where does the model's greedy pick sit inside top-K?
            # Build joint IDs for the candidates to compare consistently
            # across modes (slot mode uses argmax-thick per top-K slot,
            # hierarchical enumerates K·N joint cells, joint mode's
            # topk_joint already is joint IDs).
            topk_joint_ids = topk_slots * NUM_THICKNESSES + topk_thick_bins
            same = (topk_joint_ids == model_argmax_joint).nonzero(as_tuple=False)
            model_pick_idx_in_topk = int(same[0, 0].item()) if same.numel() > 0 else 0
            de_model_pick = float(delta_e[model_pick_idx_in_topk].item())
            de_best_topk = float(delta_e.min().item())
            de_mean_topk = float(delta_e.mean().item())

            argmin_idx = int(delta_e.argmin().item())
            topk_argmin_matches_model = float(
                model_pick_idx_in_topk == argmin_idx
            )

            slot_match_gt = float(model_argmax_slot == gt_slots[k])
            thick_match_gt = float(
                THICKNESSES[model_argmax_thick_bin] == gt_thicknesses[k]
            )

            # Entropy diagnostics: marginal slot dist + thickness dist
            # conditioned on the model's argmax slot (matches inference
            # semantics).
            slot_scores_diag = layer_logits.max(dim=-1).values.masked_fill(
                ~active_mask_bool, -1e9,
            )
            slot_probs_all = F.softmax(slot_scores_diag, dim=-1)
            slot_entropy = -(
                slot_probs_all * slot_probs_all.clamp_min(1e-12).log()
            ).sum().item()
            argmax_slot_thick_probs = F.softmax(
                layer_logits[model_argmax_slot], dim=-1,
            )
            thick_entropy = -(
                argmax_slot_thick_probs
                * argmax_slot_thick_probs.clamp_min(1e-12).log()
            ).sum().item()

            # Flat-target diagnostics (Sept 15). Answer "is the target
            # distribution meaningfully peaked, or is it near-uniform?"
            # A near-uniform target gives no rank signal — the pathology
            # observed on hierarchical M=4×N=3 where loss_topk hovered
            # at log(12) throughout training.
            #   target_entropy   — H(softmax(-β·ΔE)); log(K) = uniform
            #   target_max_prob  — max target weight; 1/K = uniform,
            #                      1.0 = fully peaked on one candidate
            #   delta_e_range    — max ΔE − min ΔE across candidates;
            #                      small range → no β can peak it
            target_entropy = -(
                target_probs * target_probs.clamp_min(1e-12).log()
            ).sum().item()
            target_max_prob = float(target_probs.max().item())
            delta_e_range = float(
                (delta_e.max() - delta_e.min()).item()
            )

            per_pos_diag.append({
                "loss_de": de_model_pick,
                "loss_de_best_topk": de_best_topk,
                "loss_de_mean_topk": de_mean_topk,
                "loss_topk": float(loss_k.item()),
                "slot_match_gt": slot_match_gt,
                "thickness_match_gt": thick_match_gt,
                "topk_argmin_matches_model": topk_argmin_matches_model,
                "slot_entropy": slot_entropy,
                "thickness_entropy": thick_entropy,
                # Flat-target diagnostics: is the CE target peaked?
                "target_entropy": target_entropy,
                "target_max_prob": target_max_prob,
                "topk_delta_e_range": delta_e_range,
                # Fraction of the K slot picks filled by ε-random draws
                # this step (K_random / K_eff). Uniform across modes:
                # counted in slots, not cells, so hierarchical N_thick
                # doesn't inflate the number.
                "explore_frac": K_random / max(K_eff, 1),
            })

    total_loss = torch.stack(losses).sum()
    if per_pos_diag:
        keys = per_pos_diag[0].keys()
        metrics = {kk: float(np.mean([d[kk] for d in per_pos_diag])) for kk in keys}
    else:
        metrics = {}
    return total_loss, metrics, n_layers


# ============================================================================
# Batch loss (public API)
# ============================================================================


def finetune_de_loss(
    model,
    batch: Dict[str, torch.Tensor],
    incidence_angle: float = 0.0,
    device: Optional[torch.device] = None,
    ce_loss_weight: float = 0.0,
    real_sim_topk: int = 0,
    sim_target_beta: float = 1.0,
    topk_mode: str = "slot",
    epsilon: float = 0.0,
    thickness_topn: int = 1,
    sim_feedback: bool = False,
    prefix_aug_prob: float = 0.0,
    prefix_aug_thickness_scale: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the finetune loss + diagnostics for one batch.

    Two training modes selected by real_sim_topk:

      real_sim_topk == 0   (STE mode, original)
        primary = mean STE ΔE₀₀ per position.
      real_sim_topk >  0   (top-K real-sim mode)
        primary = mean listwise CE per position, where the target
        distribution over the K candidates is softmax(-β · ΔE_real).
        See _topK_sim_loss_for_example for details.

    Combined loss = primary + ce_loss_weight * CE_anchor.

    Setting ce_loss_weight > 0 adds a per-token cross-entropy anchor
    against the GT (slot, thickness_bin) at every layer position. This
    is the same CE loss the pretrain minimized; adding it here binds
    the finetune to the pretrain manifold. Under STE mode it's a
    correction for the STE linearization drift; under top-K real-sim
    mode it also keeps the thickness head trained (the top-K loss
    only puts gradient on slot logits).

    Expected batch keys (produced by src/de_finetune.collate_fn below):
        lab            : [B, 3]                — normalised target Lab
        pool_features  : [B, M_MAX, 2, NUM_LAMBDA]
        pool_mask      : [B, M_MAX]
        pool_size      : [B]
        target_slots       : list[list[int]]   — length B, variable-length
        target_thicknesses : list[list[int]]   — length B, variable-length

    Returns
    -------
    loss : scalar tensor — combined loss (ΔE + λ·CE). Backward-ready.
    metrics : dict of averaged diagnostics — carries:
        - loss_de   : mean ΔE₀₀ per position (the target metric,
                      unchanged by ce_loss_weight)
        - loss_ce   : mean CE per position (present iff ce_loss_weight > 0)
        - loss_total: the combined loss returned above
        - slot/thickness entropies, slot/thickness match-to-GT,
          n_positions, batch_size
    """
    if not sim_is_available():
        raise RuntimeError(
            "differentiable optical sim not available — install jaxlayerlumos."
        )

    if device is None:
        device = next(model.parameters()).device

    # Build the batched teacher-forced structure matrix: for each example,
    # a stack matrix built from all N ground-truth layers. The model's
    # causal mask keeps position k from seeing layers >= k.
    batch_size = batch["lab"].size(0)
    structure_matrices = []
    for b in range(batch_size):
        gt_slots = batch["target_slots"][b]
        gt_thicknesses = batch["target_thicknesses"][b]
        # Truncate to MAX_LAYERS defensively.
        gt_slots_use = list(gt_slots)[:MAX_LAYERS]
        gt_thicknesses_use = list(gt_thicknesses)[:MAX_LAYERS]
        structure_matrices.append(
            build_structure_matrix(gt_slots_use, gt_thicknesses_use)
        )
    structure_matrix = torch.stack(structure_matrices, dim=0).to(device)

    lab = batch["lab"].to(device)
    pool_features = batch["pool_features"].to(device)   # [B, M_MAX, 2, L]
    pool_mask = batch["pool_mask"].to(device)
    pool_size = batch["pool_size"].to(device)

    # Optional sim-feedback: real (JLL) sim on each GT prefix gives a
    # per-position residual (target − partial-sim), fed to the model so
    # position k's prediction knows how far off the prefix already is.
    # Adds N-1 sims per example (~25% overhead) but closes the "how am
    # I doing" loop the pre-residual model didn't have.
    residual_labs = None
    if sim_feedback:
        residual_labs = _compute_partial_residuals(
            batch, incidence_angle, device,
            prefix_aug_prob=prefix_aug_prob,
            prefix_aug_thickness_scale=prefix_aug_thickness_scale,
        )

    # One model forward for the whole batch.
    # For FlexMaterialCrossAttn.forward, we pass lab / pool_features /
    # pool_mask / pool_size / structure_matrix and get [B, MAX_LAYERS+1,
    # VOCAB_SIZE]. See src/model.py.
    logits = model(
        lab=lab,
        pool_features=pool_features,
        pool_mask=pool_mask,
        pool_size=pool_size,
        structure_matrix=structure_matrix,
        apply_output_mask=True,
        residual_labs=residual_labs,
    )
    if logits.dim() != 3:
        raise RuntimeError(
            f"finetune_de_loss expects a 3-D logits tensor [B, T, V]; "
            f"got shape {tuple(logits.shape)}. This finetune requires the "
            f"cross-attn head (packed decoder). Set HEAD_MODE=cross_attn."
        )

    # One-shot sanity: log if the model's raw output contains NaN or +inf.
    # -inf is expected (apply_output_mask=True writes -inf into padded-slot
    # tokens; ste_pick sanitizes these to -1e9 before any multiplication)
    # so we deliberately do NOT flag it here.
    if _NAN_DEBUG_COUNT < _NAN_DEBUG_LIMIT:
        with torch.no_grad():
            has_nan = bool(torch.isnan(logits).any().item())
            has_posinf = bool(torch.isposinf(logits).any().item())
        if has_nan or has_posinf:
            _log_nan(
                where="model.forward output logits",
                has_nan=has_nan,
                has_posinf=has_posinf,
                logits_min=float(logits[torch.isfinite(logits)].min().item())
                if torch.isfinite(logits).any() else float("nan"),
                logits_max=float(logits[torch.isfinite(logits)].max().item())
                if torch.isfinite(logits).any() else float("nan"),
                batch_size=batch_size,
                lab_finite=bool(torch.isfinite(lab).all().item()),
                pool_features_finite=bool(
                    torch.isfinite(pool_features).all().item()
                ),
            )

    # sim / STE work in float64 for numerical stability; convert on the
    # boundary so the model can keep bf16/fp32 for its heavy tensors.
    sim_dtype = torch.float64

    # Sim needs raw n/k spectra: [M_MAX, NUM_LAMBDA] each. Pool features
    # are stored as [M_MAX, 2, NUM_LAMBDA] with channel 0 = n, 1 = k.
    total_loss = torch.zeros((), device=device, dtype=sim_dtype)
    all_metrics: List[Dict[str, float]] = []
    total_positions = 0

    for b in range(batch_size):
        gt_slots = list(batch["target_slots"][b])[:MAX_LAYERS]
        gt_thicknesses = list(batch["target_thicknesses"][b])[:MAX_LAYERS]
        if not gt_slots:
            continue

        pool_n_b = pool_features[b, :, 0, :].to(dtype=sim_dtype)   # [M_MAX, L]
        pool_k_b = pool_features[b, :, 1, :].to(dtype=sim_dtype)
        # Zero out padded slots so a stray STE gradient can't pick them up
        # (belt-and-suspenders — the model's output mask already blocks
        # those slot tokens).
        active_mask = pool_mask[b].to(dtype=sim_dtype).unsqueeze(-1)  # [M_MAX, 1]
        pool_n_b = pool_n_b * active_mask
        pool_k_b = pool_k_b * active_mask

        target_lab_denorm = torch.tensor(
            denormalize_lab(batch["lab"][b]),
            device=device, dtype=sim_dtype,
        )

        logits_b = logits[b].to(dtype=sim_dtype)     # [MAX_LAYERS+1, V]

        if real_sim_topk > 0:
            pool_mask_b = pool_mask[b].to(dtype=sim_dtype)
            loss_b, metrics_b, n_pos = _topK_sim_loss_for_example(
                logits_b, pool_n_b, pool_k_b, pool_mask_b,
                gt_slots, gt_thicknesses, target_lab_denorm,
                incidence_angle=incidence_angle,
                top_k=real_sim_topk,
                beta=sim_target_beta,
                topk_mode=topk_mode,
                epsilon=epsilon,
                thickness_topn=thickness_topn,
            )
        else:
            loss_b, metrics_b, n_pos = _rollout_one(
                logits_b, pool_n_b, pool_k_b,
                gt_slots, gt_thicknesses, target_lab_denorm,
                incidence_angle=incidence_angle,
            )
        total_loss = total_loss + loss_b
        total_positions += n_pos
        if metrics_b:
            # Weight each example's mean by its layer count so batch-level
            # means reflect the true per-position average.
            metrics_b_weighted = {
                k: v * n_pos for k, v in metrics_b.items()
            }
            metrics_b_weighted["_weight"] = n_pos
            all_metrics.append(metrics_b_weighted)

    if total_positions == 0:
        return total_loss, {"n_positions": 0, "batch_size": batch_size}

    # In STE mode this scalar IS the ΔE mean (comparable across runs).
    # In top-K mode it's the listwise CE — a different unit; the ΔE
    # metric is carried per-position via metrics["loss_de"] instead.
    primary_loss = total_loss / total_positions   # sim_dtype (fp64)

    # ---- Optional CE anchor ----
    # CE uses the model's native dtype for numerical parity with pretrain;
    # convert to sim_dtype before combining so the scalar loss returned to
    # the training loop has one consistent dtype (fp64). This is fine —
    # the sim_dtype cast is cheap and the combined loss is a scalar.
    if ce_loss_weight > 0:
        ce_loss = ce_anchor_loss(
            logits,
            batch["target_slots"],
            batch["target_thicknesses"],
        )
        combined_loss = primary_loss + ce_loss_weight * ce_loss.to(dtype=sim_dtype)
    else:
        ce_loss = None
        combined_loss = primary_loss

    # ---- Aggregate the per-example metrics ----
    metrics_out: Dict[str, float] = {
        "n_positions": total_positions,
        "batch_size": batch_size,
    }
    if all_metrics:
        total_w = sum(m["_weight"] for m in all_metrics)
        keys = [k for k in all_metrics[0].keys() if k != "_weight"]
        for k in keys:
            metrics_out[k] = float(
                sum(m[k] for m in all_metrics) / max(total_w, 1)
            )

    # Report components separately so training logs + history.jsonl can
    # carry them; lets us compare val_loss_de across runs with different
    # ce_loss_weight / real_sim_topk settings. In STE mode, loss_de is
    # overwritten from primary_loss (which IS the ΔE mean); in top-K
    # mode we KEEP the per-position aggregation from metrics_out (which
    # carries the greedy-pick ΔE) — do NOT overwrite it.
    if real_sim_topk > 0:
        metrics_out["loss_topk"] = float(primary_loss.item())
        # metrics_out["loss_de"] already populated above via aggregation.
    else:
        metrics_out["loss_de"] = float(primary_loss.item())
    metrics_out["loss_total"] = float(combined_loss.item())
    if ce_loss is not None:
        metrics_out["loss_ce"] = float(ce_loss.item())

    return combined_loss, metrics_out


# ============================================================================
# Collate helper — variable-length target_slots / target_thicknesses
# ============================================================================


def collate_fn(examples) -> Dict[str, torch.Tensor]:
    """Simpler collate than pretrain's: we pass full (variable-length)
    ground-truth structures through and let the rollout iterate positions.
    """
    from src.material_features import featurize_pool, pad_pool_features

    all_lab, all_pool_feats, all_pool_masks, all_pool_sizes = [], [], [], []
    all_target_slots, all_target_thicknesses = [], []

    for ex in examples:
        pool_feats_unpadded = featurize_pool(ex.pool, mode="raw_spectrum")
        pool_feats, pool_mask = pad_pool_features(
            pool_feats_unpadded, m_max=M_MAX,
        )
        all_lab.append(ex.lab)
        all_pool_feats.append(pool_feats)
        all_pool_masks.append(pool_mask)
        all_pool_sizes.append(len(ex.pool))
        all_target_slots.append(list(ex.target_slots))
        all_target_thicknesses.append(list(ex.target_thicknesses))

    return {
        "lab": torch.stack(all_lab),
        "pool_features": torch.stack(all_pool_feats),
        "pool_mask": torch.stack(all_pool_masks),
        "pool_size": torch.tensor(all_pool_sizes, dtype=torch.long),
        "target_slots": all_target_slots,
        "target_thicknesses": all_target_thicknesses,
    }
