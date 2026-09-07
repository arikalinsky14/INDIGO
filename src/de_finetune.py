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
)
from src.optical_sim_diff import (
    differentiable_compute_lab,
    is_available as sim_is_available,
)


# ============================================================================
# Constants (once, cached)
# ============================================================================

# Bin centers in nm — exactly the training grid so soft-thickness collapses
# to a real token value in the forward pass.
_THICKNESS_BIN_CENTERS_NP = np.asarray(THICKNESSES, dtype=np.float64)


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
# Batch loss (public API)
# ============================================================================


def finetune_de_loss(
    model,
    batch: Dict[str, torch.Tensor],
    incidence_angle: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute mean ΔE₀₀ loss + diagnostics for one finetune batch.

    Expected batch keys (produced by src/de_finetune.collate_fn below):
        lab            : [B, 3]                — normalised target Lab
        pool_features  : [B, M_MAX, 2, NUM_LAMBDA]
        pool_mask      : [B, M_MAX]
        pool_size      : [B]
        target_slots       : list[list[int]]   — length B, variable-length
        target_thicknesses : list[list[int]]   — length B, variable-length

    Returns
    -------
    loss_de : scalar tensor — mean ΔE₀₀ across all (example, k-position)
              pairs in the batch. Backward-ready.
    metrics : dict of averaged diagnostics
              (slot_entropy, thickness_entropy, slot_match_gt,
               thickness_match_gt, n_positions, batch_size).
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
    )
    if logits.dim() != 3:
        raise RuntimeError(
            f"finetune_de_loss expects a 3-D logits tensor [B, T, V]; "
            f"got shape {tuple(logits.shape)}. This finetune requires the "
            f"cross-attn head (packed decoder). Set HEAD_MODE=cross_attn."
        )

    # One-shot sanity: log if the model's raw output is already NaN before
    # STE / sim. Isolates model-forward bugs from sim / loss bugs.
    if _NAN_DEBUG_COUNT < _NAN_DEBUG_LIMIT:
        with torch.no_grad():
            finite_share = float(torch.isfinite(logits).float().mean().item())
        if finite_share < 1.0:
            _log_nan(
                where="model.forward output logits",
                finite_share=finite_share,
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

    mean_loss = total_loss / total_positions
    # Aggregate the per-example metrics.
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
    return mean_loss, metrics_out


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
