"""
Flexible-Material Color → Structure Model (two-head variant)
============================================================

Same overall backbone (MLP or cross-attn) as before, but the head has
been split:

  - slot_head       classifies which pool slot to deposit next, or EOS.
                    Output: [..., M_MAX + 1] logits.
  - thickness_head  regresses the layer thickness in nm, ONE prediction
                    per slot per position (sigmoid × MAX_THICKNESS_NM).
                    Output: [..., M_MAX] scalars in nm ∈ (0, 200].

At training time we know the true slot, so we gather the thickness
prediction at that slot and MSE it against the true normalized
thickness. At inference we pick a slot (greedy or sampled), then gather
the corresponding thickness for that slot.

The joint (slot × thickness) vocab is gone. Existing checkpoints trained
against the old vocab will not load — vocab size and head parameter
shapes differ.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.material_features import (
    NUM_LAMBDA,
    NUM_COMPACT_LAMBDA,
    feature_dim,
    MaterialNK,
    featurize_pool,
    pad_pool_features,
)
from src.materials_vocab import (
    EOS_TOKEN,
    M_MAX,
    MAX_LAYERS,
    MAX_THICKNESS_NM,
    VOCAB_SIZE,
    build_output_mask,
    build_output_mask_batch,
    build_structure_matrix,
    decode_slot_token,
    denormalize_thickness,
    normalize_thickness,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ModelConfig:
    """Hyperparameters for the flexible-material model."""

    # Material encoder
    feature_mode: str = "raw_spectrum"  # 'raw_spectrum' or 'compact'
    encoder_hidden: int = 128
    encoder_out: int = 64
    encoder_dropout: float = 0.1

    # Backbone
    d_model: int = 1024
    n_layers: int = 8
    dropout: float = 0.1

    # Head architecture.
    head_mode: str = "mlp"            # 'mlp' or 'cross_attn'
    n_heads: int = 8
    slot_encoder_layers: int = 0
    decoder_layers: int = 1

    # Loss weight on the thickness head. Slot loss is CE; thickness loss is
    # MSE on normalized thickness. Fixed λ combo (see training.py).
    thickness_loss_weight: float = 1.0

    # Training bookkeeping (in-config so tags match the run's hypers)
    learning_rate: float = 4.42e-5
    batch_size: int = 64
    epochs: int = 1
    limit_examples: Optional[int] = None

    # Vocabulary (recorded for checkpoint sanity, not configurable)
    vocab_size: int = VOCAB_SIZE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def tag(self) -> str:
        base = (
            f"flex2h_{self.feature_mode}_"
            f"enc{self.encoder_hidden}-{self.encoder_out}_"
            f"d{self.d_model}_L{self.n_layers}_do{self.dropout}_"
            f"lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}"
        )
        if self.head_mode != "mlp":
            base += f"_{self.head_mode}H{self.n_heads}"
            se = self.slot_encoder_layers or self.n_layers
            base += f"_se{se}_dec{self.decoder_layers}"
        if self.thickness_loss_weight != 1.0:
            base += f"_tw{self.thickness_loss_weight}"
        if self.limit_examples is not None:
            base += f"_lim{self.limit_examples}"
        return base


# ============================================================================
# Material encoder (unchanged)
# ============================================================================


class MaterialEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        in_dim = feature_dim(config.feature_mode)
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.encoder_hidden),
            nn.ReLU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.encoder_hidden, config.encoder_out),
            nn.ReLU(),
        )

    def forward(self, pool_features: torch.Tensor) -> torch.Tensor:
        B, M, _, L = pool_features.shape
        flat = pool_features.reshape(B * M, 2 * L)
        emb = self.net(flat)
        return emb.reshape(B, M, -1)


# ============================================================================
# Two-head output helper
# ============================================================================


def _thickness_from_scalar(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid → thickness in nm ∈ (0, MAX_THICKNESS_NM)."""
    return torch.sigmoid(x) * MAX_THICKNESS_NM


# ============================================================================
# FlexMaterialMLP (two-head)
# ============================================================================


class FlexMaterialMLP(nn.Module):
    """Flatten-then-MLP backbone with two heads.

    Slot head : [B, M_MAX + 1]  (with EOS as the last logit)
    Thickness : [B, M_MAX]      (nm ∈ (0, 200])
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.material_encoder = MaterialEncoder(config)

        input_dim = (
            3
            + M_MAX * config.encoder_out
            + M_MAX * MAX_LAYERS
            + 1
        )
        self._input_dim = input_dim

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, config.d_model))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.d_model, config.d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
        # Trunk stops one Linear short of the old design — the two heads
        # come off the shared representation.
        self.trunk = nn.Sequential(*layers)

        # Slot classifier (M_MAX + 1) and per-slot thickness regressor (M_MAX).
        self.slot_head = nn.Linear(config.d_model, M_MAX + 1)
        self.thickness_head = nn.Linear(config.d_model, M_MAX)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        lab: torch.Tensor,
        pool_features: torch.Tensor,
        pool_mask: torch.Tensor,
        structure_matrix: torch.Tensor,
        pool_size: torch.Tensor,
        apply_output_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Two-head forward.

        Returns
        -------
        {'slot_logits': [B, M_MAX + 1],
         'thickness_nm': [B, M_MAX]}
        """
        B = lab.size(0)

        emb = self.material_encoder(pool_features)         # [B, M_MAX, E]
        emb = emb * pool_mask.unsqueeze(-1).float()

        emb_flat = emb.reshape(B, -1)
        struct_flat = structure_matrix.reshape(B, -1)
        pool_size_norm = (pool_size.float() / float(M_MAX)).unsqueeze(-1)

        x = torch.cat([lab, emb_flat, struct_flat, pool_size_norm], dim=1)
        assert x.size(1) == self._input_dim

        h = self.trunk(x)                                  # [B, d_model]

        slot_logits = self.slot_head(h)                    # [B, M_MAX + 1]
        thickness_nm = _thickness_from_scalar(self.thickness_head(h))  # [B, M_MAX]

        if apply_output_mask:
            mask = build_output_mask_batch(pool_size, device=slot_logits.device)
            slot_logits = slot_logits + mask

        return {"slot_logits": slot_logits, "thickness_nm": thickness_nm}


# ============================================================================
# FlexMaterialCrossAttn (two-head)
# ============================================================================


class FlexMaterialCrossAttn(nn.Module):
    """Packed pointer-head decoder with two heads.

    Slot logits  : [B, SEQ_LEN, M_MAX + 1]
    Thicknesses  : [B, SEQ_LEN, M_MAX]  (nm)
    """

    SEQ_LEN = MAX_LAYERS + 1

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.material_encoder = MaterialEncoder(config)
        self.slot_proj = nn.Linear(config.encoder_out, config.d_model)

        slot_depth = config.slot_encoder_layers or config.n_layers
        slot_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(slot_layer, num_layers=slot_depth)

        self.past_proj = nn.Linear(config.encoder_out, config.d_model)
        self.layer_pos_emb = nn.Parameter(torch.zeros(self.SEQ_LEN, config.d_model))

        self.lab_proj = nn.Linear(3 + 1, config.d_model)
        self.start_token = nn.Parameter(torch.zeros(1, config.d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=config.decoder_layers
        )

        # Per-position-per-slot heads. Each takes (slot_token, query_state).
        # Slot head produces one logit per slot; a separate EOS logit lives
        # off the pure query state (EOS has no "slot").
        self.slot_pointer_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),   # one slot logit per (position, slot)
        )
        self.thickness_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),   # one thickness scalar per (position, slot)
        )
        self.eos_head = nn.Linear(config.d_model, 1)

        causal = torch.triu(
            torch.ones(self.SEQ_LEN, self.SEQ_LEN, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in (
            self.slot_proj,
            self.past_proj,
            self.lab_proj,
            self.eos_head,
            *self.slot_pointer_head.modules(),
            *self.thickness_head.modules(),
        ):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.layer_pos_emb, std=0.02)
        nn.init.normal_(self.start_token, std=0.02)

    def forward(
        self,
        lab: torch.Tensor,
        pool_features: torch.Tensor,
        pool_mask: torch.Tensor,
        structure_matrix: torch.Tensor,
        pool_size: torch.Tensor,
        apply_output_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B = lab.size(0)
        device = lab.device

        emb = self.material_encoder(pool_features)
        slot_tokens = self.slot_proj(emb)

        slot_key_padding_mask = ~pool_mask
        slot_tokens = self.slot_encoder(
            slot_tokens, src_key_padding_mask=slot_key_padding_mask
        )

        past_emb = torch.einsum("bsl,bse->ble", structure_matrix, emb)
        past_tokens = self.past_proj(past_emb)

        pool_size_norm = (pool_size.float() / float(M_MAX)).unsqueeze(-1)
        goal = self.lab_proj(torch.cat([lab, pool_size_norm], dim=1))
        start = (goal + self.start_token).unsqueeze(1)

        seq = torch.cat([start, past_tokens], dim=1)
        seq = seq + self.layer_pos_emb.unsqueeze(0)

        col_active = (structure_matrix.abs().sum(dim=1) > 0)
        seq_key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device),
             ~col_active],
            dim=1,
        )

        dec_out = self.decoder(
            tgt=seq,
            memory=slot_tokens,
            tgt_mask=self._causal_mask,
            tgt_key_padding_mask=seq_key_padding_mask,
            memory_key_padding_mask=slot_key_padding_mask,
        )                                                              # [B, SEQ_LEN, d_model]

        slot_expand = slot_tokens.unsqueeze(1).expand(-1, self.SEQ_LEN, -1, -1)
        query_expand = dec_out.unsqueeze(2).expand(-1, -1, M_MAX, -1)
        slot_query = torch.cat([slot_expand, query_expand], dim=-1)

        slot_logits_per_slot = self.slot_pointer_head(slot_query).squeeze(-1)      # [B, SEQ_LEN, M_MAX]
        thickness_nm = _thickness_from_scalar(
            self.thickness_head(slot_query).squeeze(-1)
        )                                                                          # [B, SEQ_LEN, M_MAX]
        eos_logits = self.eos_head(dec_out)                                        # [B, SEQ_LEN, 1]
        slot_logits = torch.cat([slot_logits_per_slot, eos_logits], dim=-1)        # [B, SEQ_LEN, M_MAX + 1]

        if apply_output_mask:
            mask = build_output_mask_batch(pool_size, device=device).unsqueeze(1)
            slot_logits = slot_logits + mask

        return {"slot_logits": slot_logits, "thickness_nm": thickness_nm}


# ============================================================================
# Factory
# ============================================================================


def build_model(config: ModelConfig) -> nn.Module:
    if config.head_mode == "mlp":
        return FlexMaterialMLP(config)
    if config.head_mode == "cross_attn":
        return FlexMaterialCrossAttn(config)
    raise ValueError(
        f"Unknown head_mode {config.head_mode!r}; expected 'mlp' or 'cross_attn'"
    )


# ============================================================================
# Loss (two-head)
# ============================================================================
#
# CE on the slot logits + λ · MSE on the gathered normalized thickness.
# Thickness loss is computed on the NORMALIZED thickness (∈ [0, 1]) so
# the loss magnitude is comparable to per-token CE regardless of the
# 200-nm scale factor.
#
# EOS positions contribute to CE only (there's no thickness to regress).


def _thickness_loss(
    thickness_nm_pred: torch.Tensor,          # [N, M_MAX] in nm
    slot_target: torch.Tensor,                # [N] with EOS=M_MAX
    thickness_target_nm: torch.Tensor,        # [N] in nm (arbitrary at EOS rows)
) -> torch.Tensor:
    """Gather the predicted thickness at the true slot and MSE it in
    normalized space. EOS rows (slot == EOS_TOKEN) are excluded.
    """
    non_eos = slot_target != EOS_TOKEN
    if not non_eos.any():
        return thickness_nm_pred.new_zeros(())
    pred = thickness_nm_pred[non_eos]                              # [n, M_MAX]
    tgt_slot = slot_target[non_eos].unsqueeze(-1)                  # [n, 1]
    gathered_nm = pred.gather(1, tgt_slot).squeeze(-1)             # [n]
    pred_norm = gathered_nm / MAX_THICKNESS_NM
    tgt_norm = thickness_target_nm[non_eos] / MAX_THICKNESS_NM
    return F.mse_loss(pred_norm, tgt_norm)


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Fanned-out (per-step) two-head loss.

    Batch fields:
      lab, pool_features, pool_mask, structure_matrix, pool_size
      slot_target       : [B] with EOS=M_MAX
      thickness_target  : [B] in nm (any value on EOS rows; masked out)
    """
    out = model(
        lab=batch["lab"],
        pool_features=batch["pool_features"],
        pool_mask=batch["pool_mask"],
        structure_matrix=batch["structure_matrix"],
        pool_size=batch["pool_size"],
    )
    slot_logits = out["slot_logits"]         # [B, ..., M_MAX + 1]
    thickness_nm = out["thickness_nm"]       # [B, ..., M_MAX]

    slot_target = batch["slot_target"]
    thickness_target = batch["thickness_target"]

    if slot_logits.dim() == 3:
        # cross_attn returns [B, SEQ_LEN, V]. Gather the prediction position
        # matching the number of already-deposited layers.
        n_laid = (batch["structure_matrix"].abs().sum(dim=1) > 0).sum(dim=1)
        gather_idx_slot = n_laid.view(-1, 1, 1).expand(-1, 1, slot_logits.size(-1))
        slot_logits = slot_logits.gather(1, gather_idx_slot).squeeze(1)
        gather_idx_thick = n_laid.view(-1, 1, 1).expand(-1, 1, thickness_nm.size(-1))
        thickness_nm = thickness_nm.gather(1, gather_idx_thick).squeeze(1)

    slot_loss = F.cross_entropy(slot_logits, slot_target)
    thick_loss = _thickness_loss(thickness_nm, slot_target, thickness_target)
    lam = getattr(model, "config", None)
    lam_w = lam.thickness_loss_weight if lam is not None else 1.0
    loss = slot_loss + lam_w * thick_loss

    with torch.no_grad():
        slot_acc = (slot_logits.argmax(dim=-1) == slot_target).float().mean()
        non_eos = slot_target != EOS_TOKEN
        if non_eos.any():
            pred_thick = thickness_nm[non_eos].gather(
                1, slot_target[non_eos].unsqueeze(-1)
            ).squeeze(-1)
            thickness_mae_nm = (pred_thick - thickness_target[non_eos]).abs().mean()
        else:
            thickness_mae_nm = thickness_nm.new_zeros(())

    return {
        "loss": loss,
        "slot_loss": slot_loss.detach(),
        "thickness_loss": thick_loss.detach(),
        "accuracy": slot_acc,               # kept name for callers
        "slot_accuracy": slot_acc,
        "thickness_mae_nm": thickness_mae_nm,
    }


def compute_loss_packed(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Packed teacher-forced two-head loss.

    Batch fields (added over the fanned-out version):
      slot_targets      : [B, SEQ_LEN]  int, with -100 at masked positions
      thickness_targets : [B, SEQ_LEN]  float nm, arbitrary at masked positions
    """
    out = model(
        lab=batch["lab"],
        pool_features=batch["pool_features"],
        pool_mask=batch["pool_mask"],
        structure_matrix=batch["structure_matrix"],
        pool_size=batch["pool_size"],
    )
    slot_logits = out["slot_logits"]         # [B, SEQ_LEN, M_MAX + 1]
    thickness_nm = out["thickness_nm"]       # [B, SEQ_LEN, M_MAX]

    if slot_logits.dim() != 3:
        raise RuntimeError(
            f"compute_loss_packed expects 3-D slot_logits; got {tuple(slot_logits.shape)}"
        )

    slot_targets = batch["slot_targets"]                   # [B, SEQ_LEN]
    thickness_targets = batch["thickness_targets"]         # [B, SEQ_LEN]
    B, S, V = slot_logits.shape

    slot_loss = F.cross_entropy(
        slot_logits.reshape(-1, V), slot_targets.reshape(-1), ignore_index=-100
    )

    valid = slot_targets != -100
    # Flatten to [N] rows so _thickness_loss's EOS filter can reuse it.
    flat_slot = slot_targets.reshape(-1)
    flat_thick_pred = thickness_nm.reshape(-1, M_MAX)
    flat_thick_tgt = thickness_targets.reshape(-1)
    # Rows to keep: valid (not -100) AND not EOS. Replace masked positions
    # with EOS so _thickness_loss drops them; that keeps its gather safe.
    keep = valid.reshape(-1)
    safe_slot = torch.where(keep, flat_slot,
                            torch.full_like(flat_slot, EOS_TOKEN))
    thick_loss = _thickness_loss(flat_thick_pred, safe_slot, flat_thick_tgt)

    lam = getattr(model, "config", None)
    lam_w = lam.thickness_loss_weight if lam is not None else 1.0
    loss = slot_loss + lam_w * thick_loss

    with torch.no_grad():
        pred = slot_logits.argmax(dim=-1)
        if valid.any():
            slot_acc = (pred[valid] == slot_targets[valid]).float().mean()
            non_eos = valid & (slot_targets != EOS_TOKEN)
            if non_eos.any():
                # Gather the predicted thickness at the true slot for
                # every non-EOS supervised position.
                true_slots = slot_targets[non_eos].clamp(min=0)
                pred_flat = thickness_nm[non_eos]                # [n, M_MAX]
                pred_at_slot = pred_flat.gather(
                    1, true_slots.unsqueeze(-1)
                ).squeeze(-1)
                thickness_mae_nm = (
                    pred_at_slot - thickness_targets[non_eos]
                ).abs().mean()
            else:
                thickness_mae_nm = thickness_nm.new_zeros(())
        else:
            slot_acc = slot_logits.new_zeros(())
            thickness_mae_nm = thickness_nm.new_zeros(())

    return {
        "loss": loss,
        "slot_loss": slot_loss.detach(),
        "thickness_loss": thick_loss.detach(),
        "accuracy": slot_acc,
        "slot_accuracy": slot_acc,
        "thickness_mae_nm": thickness_mae_nm,
    }


# ============================================================================
# Autoregressive generation
# ============================================================================


def generate_structure(
    model: nn.Module,
    lab: torch.Tensor,
    pool: List[MaterialNK],
    device: torch.device,
    max_layers: int = MAX_LAYERS,
    sample: bool = False,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[int], List[float], str]:
    """Autoregressively decode a structure.

    Returns
    -------
    slot_indices : list of int
    thicknesses_nm : list of float
    termination : 'EOS' or 'MAX_LEN'
    """
    if len(pool) > M_MAX:
        raise ValueError(f"Pool of {len(pool)} exceeds M_MAX={M_MAX}")

    pool_size = len(pool)
    pool_feats_unpadded = featurize_pool(pool, mode=model.config.feature_mode)
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)

    lab_b = lab.unsqueeze(0).to(device)
    pool_feats_b = pool_feats.unsqueeze(0).to(device)
    pool_mask_b = pool_mask.unsqueeze(0).to(device)
    pool_size_b = torch.tensor([pool_size], dtype=torch.long, device=device)

    slot_indices: List[int] = []
    thicknesses_nm: List[float] = []
    termination = "MAX_LEN"

    structure = torch.zeros(M_MAX, MAX_LAYERS, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for step in range(max_layers):
            structure_b = structure.unsqueeze(0)
            out = model(
                lab=lab_b,
                pool_features=pool_feats_b,
                pool_mask=pool_mask_b,
                structure_matrix=structure_b,
                pool_size=pool_size_b,
            )
            slot_logits = out["slot_logits"]         # [1, V] or [1, SEQ_LEN, V]
            thickness_nm = out["thickness_nm"]       # [1, M_MAX] or [1, SEQ_LEN, M_MAX]

            if slot_logits.dim() == 3:
                slot_logits = slot_logits[:, step, :]
                thickness_nm = thickness_nm[:, step, :]

            if sample:
                scaled = slot_logits / max(temperature, 1e-6)
                probs = F.softmax(scaled, dim=-1)
                slot_id = torch.multinomial(
                    probs, num_samples=1, generator=generator
                ).item()
            else:
                slot_id = int(slot_logits.argmax(dim=-1).item())

            if slot_id == EOS_TOKEN:
                termination = "EOS"
                break

            thick_nm = float(thickness_nm[0, slot_id].item())
            # Physical range guard (sigmoid can't hit exactly 0 but clamp
            # defensively so downstream simulators don't see 0-thickness).
            thick_nm = max(1e-3, thick_nm)

            slot_indices.append(int(slot_id))
            thicknesses_nm.append(thick_nm)

            structure[slot_id, step] = normalize_thickness(thick_nm)

    return slot_indices, thicknesses_nm, termination


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    from src.material_features import load_jll_directory
    from src.materials_vocab import build_structure_matrix, encode_slot

    cfg = ModelConfig(
        feature_mode="raw_spectrum",
        d_model=256,
        n_layers=4,
        dropout=0.0,
        encoder_hidden=64,
        encoder_out=32,
    )
    print(f"Config tag: {cfg.tag()}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model param count: {n_params:,}")

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        print("[smoke] No JLL materials; skipping forward pass test.")
        raise SystemExit(0)

    pool_full = list(load_jll_directory(materials_dir).values())
    pool = pool_full[:5]

    lab = torch.tensor([0.5, 0.2, -0.4])
    pool_feats_unpadded = featurize_pool(pool, mode=cfg.feature_mode)
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)

    # Use continuous float thicknesses now.
    structure = build_structure_matrix([0, 2], [102.5, 47.3])
    pool_size = torch.tensor([5], dtype=torch.long)

    batch = {
        "lab": lab.unsqueeze(0),
        "pool_features": pool_feats.unsqueeze(0),
        "pool_mask": pool_mask.unsqueeze(0),
        "structure_matrix": structure.unsqueeze(0),
        "pool_size": pool_size,
        "slot_target": torch.tensor([1]),
        "thickness_target": torch.tensor([75.7]),
    }

    out = compute_loss(model, batch)
    print(f"\nForward pass (MLP):")
    print(f"  loss = {out['loss'].item():.4f}")
    print(f"  slot_loss = {out['slot_loss'].item():.4f}, "
          f"thick_loss = {out['thickness_loss'].item():.4f}")
    print(f"  slot_acc = {out['slot_accuracy'].item():.3f}, "
          f"thick_mae_nm = {out['thickness_mae_nm'].item():.2f}")

    with torch.no_grad():
        raw = model(
            lab=batch["lab"],
            pool_features=batch["pool_features"],
            pool_mask=batch["pool_mask"],
            structure_matrix=batch["structure_matrix"],
            pool_size=batch["pool_size"],
        )
    slot_logits = raw["slot_logits"]
    thickness_nm = raw["thickness_nm"]
    valid_slots = torch.isfinite(slot_logits[0]).sum().item()
    expected_valid = 5 + 1
    print(f"  slot_logits finite = {int(valid_slots)} (expected {expected_valid}) "
          f"{'✓' if valid_slots == expected_valid else '✗'}")
    print(f"  thickness_nm range: [{float(thickness_nm.min()):.2f}, "
          f"{float(thickness_nm.max()):.2f}] "
          f"(expected ⊂ (0, {MAX_THICKNESS_NM}]) "
          f"{'✓' if (thickness_nm > 0).all() and (thickness_nm <= MAX_THICKNESS_NM + 1e-4).all() else '✗'}")

    print(f"\nAutoregressive generation (MLP):")
    slots, thicks, term = generate_structure(model, lab, pool, torch.device("cpu"))
    print(f"  termination: {term}")
    print(f"  slots: {slots}")
    print(f"  thicknesses (float nm): {[f'{t:.2f}' for t in thicks]}")
    print(f"  resolved materials: {[pool[s].name for s in slots]}")

    # Cross-attn.
    print(f"\nCross-attention variant:")
    cfg_xa = ModelConfig(
        feature_mode="raw_spectrum",
        d_model=128, n_layers=2, n_heads=4, dropout=0.0,
        encoder_hidden=64, encoder_out=32,
        head_mode="cross_attn",
    )
    model_xa = build_model(cfg_xa)
    out_xa = compute_loss(model_xa, batch)
    print(f"  loss = {out_xa['loss'].item():.4f}, "
          f"slot_acc = {out_xa['slot_accuracy'].item():.3f}, "
          f"thick_mae_nm = {out_xa['thickness_mae_nm'].item():.2f}")

    with torch.no_grad():
        raw_xa = model_xa(
            lab=batch["lab"],
            pool_features=batch["pool_features"],
            pool_mask=batch["pool_mask"],
            structure_matrix=batch["structure_matrix"],
            pool_size=batch["pool_size"],
        )
    print(f"  slot_logits shape: {tuple(raw_xa['slot_logits'].shape)}")
    print(f"  thickness_nm shape: {tuple(raw_xa['thickness_nm'].shape)}")

    # Packed-loss smoke.
    seq_len_xa = raw_xa["slot_logits"].size(1)
    slot_seq = torch.full((1, seq_len_xa), -100, dtype=torch.long)
    thick_seq = torch.zeros((1, seq_len_xa), dtype=torch.float32)
    slot_seq[0, 0] = encode_slot(0)
    thick_seq[0, 0] = 50.5
    slot_seq[0, 1] = encode_slot(2)
    thick_seq[0, 1] = 100.25
    slot_seq[0, 2] = EOS_TOKEN
    thick_seq[0, 2] = 0.0
    packed_batch = {
        **batch,
        "slot_targets": slot_seq,
        "thickness_targets": thick_seq,
    }
    out_packed = compute_loss_packed(model_xa, packed_batch)
    print(f"  packed loss = {out_packed['loss'].item():.4f}, "
          f"slot_acc = {out_packed['slot_accuracy'].item():.3f}, "
          f"thick_mae_nm = {out_packed['thickness_mae_nm'].item():.2f}")
    out_packed["loss"].backward()
    n_with_grad = sum(
        1 for p in model_xa.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for _ in model_xa.parameters())
    print(f"  packed backward: {n_with_grad}/{n_total} params received grad")

    print("\n[smoke] OK")
