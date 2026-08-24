"""
Flexible-Material Color → Structure Model
=========================================

Replaces `pretrain_rgb_to_structure/src/model.py` from the original
CHROMA-Lite. Key architectural changes:

1. The model takes a *variable* pool of materials (n,k spectra) as input,
   alongside the target color (CIE Lab) and the partial structure.
   It does not have a fixed 25-material vocabulary.

2. A shared `MaterialEncoder` MLP is applied independently to each pool
   slot's n,k features. Sharing weights enforces permutation equivariance:
   the model treats material identity as opaque — only n,k features matter.
   Padded slots are zeroed via a mask so they contribute nothing.

3. The output head produces logits over a slot-indexed vocabulary
   (M_MAX × NUM_THICKNESSES + EOS = 3201 tokens on the 2 nm grid;
   32 × 100 + 1). At inference the user-supplied pool dictates which slots
   are valid; an output mask suppresses logits for unused slots before
   softmax.

The main backbone remains a feedforward MLP. The MLP-beats-transformer
finding from the original CHROMA-Lite ablations is task-agnostic enough to
carry over.

Forward pass shape walkthrough
------------------------------
Inputs:
    lab              : [B, 3]
    pool_features    : [B, M_MAX, 2, NUM_LAMBDA]   (zero-padded)
    pool_mask        : [B, M_MAX]   bool, True for valid slots
    structure_matrix : [B, M_MAX, MAX_LAYERS]      (also zero in padded slots)

Material encoder applied per slot:
    pool_features    -> [B, M_MAX, emb_dim]
Mask zero-out:
    embeddings *= pool_mask[..., None]            -> [B, M_MAX, emb_dim]
Flatten:
    pool_flat        : [B, M_MAX * emb_dim]
    structure_flat   : [B, M_MAX * MAX_LAYERS]

Concatenate:
    x = [lab, pool_flat, structure_flat, pool_size_norm]
        -> [B, 3 + M_MAX*emb_dim + M_MAX*MAX_LAYERS + 1]

Backbone MLP:
    x -> Linear(d_model) -> ReLU -> Dropout -> ... -> Linear(VOCAB_SIZE)

Output masking (training and inference):
    logits += output_mask                          -> [B, VOCAB_SIZE]
    where output_mask is 0 on valid slots, -inf elsewhere.
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
    NUM_THICKNESSES,
    VOCAB_SIZE,
    build_output_mask,
    build_output_mask_batch,
    build_structure_matrix,
    decode_token,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ModelConfig:
    """Hyperparameters for the flexible-material model.

    The fields below capture everything needed to reconstruct the model from
    a checkpoint. The hyperparameter tag for the checkpoint directory name
    is composed from these.
    """

    # Material encoder
    feature_mode: str = "raw_spectrum"  # 'raw_spectrum' or 'compact'
    encoder_hidden: int = 128
    encoder_out: int = 64
    encoder_dropout: float = 0.1

    # Backbone
    d_model: int = 1024
    n_layers: int = 8
    dropout: float = 0.1

    # Architecture choice. 'mlp' = original flatten-then-MLP backbone.
    # 'cross_attn' = pointer head: per-slot transformer encoder + a
    # (lab, structure)-derived query that cross-attends onto the pool, with
    # a per-slot thickness head. Pointer keeps permutation-equivariance by
    # construction; the MLP relies on pool_sampler's slot-shuffling to learn
    # it as augmentation.
    head_mode: str = "mlp"            # 'mlp' or 'cross_attn'
    n_heads: int = 8                  # only used when head_mode == 'cross_attn'
    # Slot encoder depth (cross_attn only). 0 means "use n_layers" — preserved
    # for backward compat. 4 is the recommended new default for cross_attn:
    # 8-layer self-attention over <=32 set elements is overkill.
    slot_encoder_layers: int = 0
    # Decoder depth (cross_attn only). Each layer does causal self-attn over
    # the [start, past_0..past_{MAX_LAYERS-1}] sequence + cross-attn onto the
    # pool keys + FFN. 1 layer is the recommended default.
    decoder_layers: int = 1

    # Training (carried in config so checkpoint tags include them, matching
    # the convention in original CHROMA-Lite)
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
        # Drop unknown keys to allow forward-compatibility.
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def tag(self) -> str:
        base = (
            f"flex_{self.feature_mode}_"
            f"enc{self.encoder_hidden}-{self.encoder_out}_"
            f"d{self.d_model}_L{self.n_layers}_do{self.dropout}_"
            f"lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}"
        )
        # Only suffix non-default head modes so existing 'mlp' checkpoint
        # directories keep their current names.
        if self.head_mode != "mlp":
            base += f"_{self.head_mode}H{self.n_heads}"
            # Depth knobs disambiguate cross_attn runs at different settings.
            se = self.slot_encoder_layers or self.n_layers
            base += f"_se{se}_dec{self.decoder_layers}"
        if self.limit_examples is not None:
            base += f"_lim{self.limit_examples}"
        return base


# ============================================================================
# Material encoder — shared across pool slots
# ============================================================================


class MaterialEncoder(nn.Module):
    """Shared encoder mapping a single material's n,k spectrum to an embedding.

    Same weights are applied to every slot of every example in the batch.
    Padded slots produce arbitrary embeddings; the caller is responsible for
    zeroing them out using `pool_mask`.
    """

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
        """
        Parameters
        ----------
        pool_features : torch.Tensor, shape [B, M_MAX, 2, L]

        Returns
        -------
        embeddings : torch.Tensor, shape [B, M_MAX, encoder_out]
        """
        B, M, _, L = pool_features.shape
        flat = pool_features.reshape(B * M, 2 * L)
        emb = self.net(flat)
        return emb.reshape(B, M, -1)


# ============================================================================
# Main model
# ============================================================================


class FlexMaterialMLP(nn.Module):
    """RGB + variable material pool -> next-layer token (autoregressive MLP).

    Mirrors `ThinFilmMLP` from the original CHROMA-Lite, with the input
    expanded to include encoded material features and the output expanded
    to a slot-indexed vocabulary.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.material_encoder = MaterialEncoder(config)

        # Backbone input dim:
        #   3                          : Lab target
        #   M_MAX * encoder_out        : encoded material pool (zeroed in padded slots)
        #   M_MAX * MAX_LAYERS         : structure matrix (zeroed in padded slots)
        #   1                          : pool size normalised to (0, 1]
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
        layers.append(nn.Linear(config.d_model, config.vocab_size))
        self.backbone = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for ReLU stack (matches original CHROMA-Lite)."""
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
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        lab : torch.Tensor, [B, 3]
        pool_features : torch.Tensor, [B, M_MAX, 2, L]
        pool_mask : torch.Tensor, [B, M_MAX], dtype=bool
        structure_matrix : torch.Tensor, [B, M_MAX, MAX_LAYERS]
        pool_size : torch.Tensor, [B], int
            Number of valid slots per example.
        apply_output_mask : bool
            If True, add -inf to logits for invalid slots before returning.

        Returns
        -------
        logits : torch.Tensor, [B, VOCAB_SIZE]
        """
        B = lab.size(0)

        # 1. Encode each slot's n,k. Shared weights → permutation-equivariant.
        emb = self.material_encoder(pool_features)         # [B, M_MAX, E]

        # 2. Zero-out padded slots so they don't contribute to the backbone.
        emb = emb * pool_mask.unsqueeze(-1).float()        # [B, M_MAX, E]

        # 3. Flatten and concat all inputs.
        emb_flat = emb.reshape(B, -1)                      # [B, M_MAX*E]
        struct_flat = structure_matrix.reshape(B, -1)      # [B, M_MAX*MAX_LAYERS]
        pool_size_norm = (pool_size.float() / float(M_MAX)).unsqueeze(-1)  # [B, 1]

        x = torch.cat([lab, emb_flat, struct_flat, pool_size_norm], dim=1)
        assert x.size(1) == self._input_dim, (
            f"Input dim mismatch: got {x.size(1)}, expected {self._input_dim}"
        )

        # 4. Backbone.
        logits = self.backbone(x)                          # [B, VOCAB_SIZE]

        # 5. Output masking.
        if apply_output_mask:
            mask = build_output_mask_batch(pool_size, device=logits.device)
            logits = logits + mask

        return logits


# ============================================================================
# Cross-attention pointer head — packed transformer-decoder variant
# ============================================================================
#
# Clean transformer-LM design:
#
#   keys/values  = pure-n,k slot embeddings   (the static pool; encoded once)
#   sequence     = [start, past_0, ..., past_{MAX_LAYERS-1}]
#   each sequence position cross-attends to the pool keys,
#   each sequence position predicts the (slot, thickness) for that layer.
#
# Token construction
# ------------------
#   - start  = lab_proj([lab, pool_size_norm]) + learned next_query
#   - past_k = past_proj( Σ_s structure[b,s,k] * emb[b,s,:] ) + pos_emb[k+1]
#     where the sum-over-slots collapses (each column has exactly one nonzero
#     slot) to thickness_at_k * material_emb_of_slot_used_at_k — so each past
#     token carries both material identity (direction) AND thickness
#     (magnitude). Permutation-equivariant by construction.
#
# Decoder: `decoder_layers` × TransformerDecoderLayer with
#   - causal self-attention over the sequence
#   - cross-attention onto slot_tokens (with padding mask for unused slots)
#   - GELU FFN
#
# Output at every sequence position p:
#   - per-slot thickness logits via (slot_tokens, query_state[p])
#   - EOS logit
#   concatenated to match the existing (slot×thickness, EOS) vocab layout.
#
# Why this matters for compute
# ----------------------------
# Old single-step path: collate_fn fanned each example into L+1 sub-samples,
# re-encoding the same pool for each step. With this design and the packed
# collate (scripts/training.py:collate_fn_packed), the slot encoder runs
# ONCE per example for ALL L+1 token predictions. ~5-6× FLOPs reduction for
# cross_attn.
#
# Single-step inference (autoregressive `generate_structure`) still works:
# pass the partial structure (only layers 0..k-1 filled), read
# logits[:, k, :] for the next-token distribution. The unused past positions
# get masked out by the key-padding mask derived from `structure_matrix.sum`.


class FlexMaterialCrossAttn(nn.Module):
    """Packed pointer-head decoder.

    Forward returns logits of shape [B, MAX_LAYERS+1, VOCAB_SIZE].
    Position p of the output is the prediction for layer p (or EOS at the
    final position). Loss/collate is responsible for masking out positions
    beyond each example's actual structure length.
    """

    # Sequence positions: 0 = start (predicts layer 0), j ∈ [1, MAX_LAYERS]
    # holds past_{j-1} and predicts layer j. Total length L+1.
    SEQ_LEN = MAX_LAYERS + 1

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Per-slot material encoder (shared across slots).
        self.material_encoder = MaterialEncoder(config)

        # Slot keys/values: pure n,k embedding projected to d_model. State-
        # independent — the past stays on the decoder side.
        self.slot_proj = nn.Linear(config.encoder_out, config.d_model)

        # Self-attention over slot keys lets slots reason about each other.
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

        # Past-decisions side: each past layer becomes its own token with a
        # positional embedding (so ORDER matters, not just content).
        self.past_proj = nn.Linear(config.encoder_out, config.d_model)
        # Positional embeddings for all SEQ_LEN positions (start + MAX_LAYERS past).
        self.layer_pos_emb = nn.Parameter(torch.zeros(self.SEQ_LEN, config.d_model))

        # Goal projection: (lab, pool_size_norm) -> d_model. Summed with a
        # learned start token to form sequence position 0.
        self.lab_proj = nn.Linear(3 + 1, config.d_model)
        self.start_token = nn.Parameter(torch.zeros(1, config.d_model))

        # Combined decoder: causal self-attn over the sequence + cross-attn
        # onto slot keys. Standard transformer decoder pattern.
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

        # Per-position-per-slot thickness head + EOS head.
        self.thickness_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, NUM_THICKNESSES),
        )
        self.eos_head = nn.Linear(config.d_model, 1)

        # Causal mask is a function of MAX_LAYERS, so we can precompute it.
        # PyTorch expects True (or float -inf) to mean "mask out".
        causal = torch.triu(
            torch.ones(self.SEQ_LEN, self.SEQ_LEN, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Init projection and head Linears; trust PyTorch defaults for the
        transformer modules (tuned for the pre-LN / GELU stack)."""
        for module in (
            self.slot_proj,
            self.past_proj,
            self.lab_proj,
            self.eos_head,
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
    ) -> torch.Tensor:
        """Same input signature as FlexMaterialMLP.forward; output shape is
        [B, MAX_LAYERS+1, VOCAB_SIZE] (one prediction per sequence position).

        For single-step autoregressive use, pass the partial structure (only
        layers 0..k-1 filled) and read `logits[:, k, :]`. Past positions
        whose structure column is zero get masked out of attention so they
        don't bias earlier predictions.
        """
        B = lab.size(0)
        device = lab.device

        # 1. Encode each slot's n,k (shared weights — permutation-equivariant).
        emb = self.material_encoder(pool_features)            # [B, M_MAX, E]

        # 2. Slot keys: pure n,k embedding.
        slot_tokens = self.slot_proj(emb)                     # [B, M_MAX, d_model]

        # 3. Self-attend over slots (padded slots masked out).
        slot_key_padding_mask = ~pool_mask                    # [B, M_MAX]
        slot_tokens = self.slot_encoder(
            slot_tokens, src_key_padding_mask=slot_key_padding_mask
        )                                                     # [B, M_MAX, d_model]

        # 4. Per-layer past tokens. Each column of structure_matrix has
        # exactly one nonzero slot, so the einsum collapses to
        #     past_emb[b, l, :] = thickness_at_l * emb[b, slot_used_at_l, :]
        # jointly encoding material identity (direction) and thickness
        # (magnitude). Unused layer positions stay zero.
        past_emb = torch.einsum(
            "bsl,bse->ble", structure_matrix, emb
        )                                                     # [B, MAX_LAYERS, E]
        past_tokens = self.past_proj(past_emb)                # [B, MAX_LAYERS, d_model]

        # 5. Start token (sequence position 0).
        pool_size_norm = (pool_size.float() / float(M_MAX)).unsqueeze(-1)
        goal = self.lab_proj(torch.cat([lab, pool_size_norm], dim=1))  # [B, d_model]
        start = (goal + self.start_token).unsqueeze(1)                 # [B, 1, d_model]

        # 6. Assemble the sequence and add positional embeddings.
        seq = torch.cat([start, past_tokens], dim=1)                   # [B, SEQ_LEN, d_model]
        seq = seq + self.layer_pos_emb.unsqueeze(0)

        # 7. Per-sequence-position key-padding mask. Position 0 (start) is
        # always valid; position j ∈ [1, MAX_LAYERS] is valid iff layer j-1
        # has been deposited (= structure column j-1 has any nonzero).
        # We derive this from structure_matrix so the same forward serves
        # both packed teacher-forcing and step-wise autoregressive inference.
        col_active = (structure_matrix.abs().sum(dim=1) > 0)           # [B, MAX_LAYERS]
        seq_key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device),
             ~col_active],
            dim=1,
        )                                                              # [B, SEQ_LEN]

        # 8. Causal self-attn + cross-attn over the sequence.
        dec_out = self.decoder(
            tgt=seq,
            memory=slot_tokens,
            tgt_mask=self._causal_mask,
            tgt_key_padding_mask=seq_key_padding_mask,
            memory_key_padding_mask=slot_key_padding_mask,
        )                                                              # [B, SEQ_LEN, d_model]

        # 9. Per-position-per-slot thickness logits + per-position EOS.
        # Broadcast (slot_tokens, query) to [B, SEQ_LEN, M_MAX, 2*d_model].
        slot_expand = slot_tokens.unsqueeze(1).expand(-1, self.SEQ_LEN, -1, -1)
        query_expand = dec_out.unsqueeze(2).expand(-1, -1, M_MAX, -1)
        slot_query = torch.cat([slot_expand, query_expand], dim=-1)
        thickness_logits = self.thickness_head(slot_query)             # [B, SEQ_LEN, M, NT]
        thickness_logits = thickness_logits.reshape(
            B, self.SEQ_LEN, M_MAX * NUM_THICKNESSES
        )
        eos_logits = self.eos_head(dec_out)                            # [B, SEQ_LEN, 1]
        logits = torch.cat([thickness_logits, eos_logits], dim=-1)     # [B, SEQ_LEN, V]

        # 10. Output masking — broadcast over the sequence dim.
        if apply_output_mask:
            mask = build_output_mask_batch(pool_size, device=device).unsqueeze(1)
            logits = logits + mask

        return logits


# ============================================================================
# Factory
# ============================================================================


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate the model variant selected by `config.head_mode`."""
    if config.head_mode == "mlp":
        return FlexMaterialMLP(config)
    if config.head_mode == "cross_attn":
        return FlexMaterialCrossAttn(config)
    raise ValueError(
        f"Unknown head_mode {config.head_mode!r}; expected 'mlp' or 'cross_attn'"
    )


# ============================================================================
# Loss
# ============================================================================


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Single-token cross-entropy loss (MLP-style, fanned-out collate).

    `batch` carries lab, pool_features, pool_mask, structure_matrix,
    pool_size, target_token (shape [B]).

    For models that return a 3-D [B, SEQ_LEN, V] logits tensor (cross_attn),
    the position to score is derived from each row's structure length —
    `target_token` is the prediction for the layer at index n_layers, so we
    gather logits at that index.
    """
    logits = model(
        lab=batch["lab"],
        pool_features=batch["pool_features"],
        pool_mask=batch["pool_mask"],
        structure_matrix=batch["structure_matrix"],
        pool_size=batch["pool_size"],
    )
    target = batch["target_token"]
    if logits.dim() == 3:
        # cross_attn returns [B, SEQ_LEN, V]. Pick the prediction position
        # corresponding to the fanned-out step (= number of laid-down layers).
        n_laid = (batch["structure_matrix"].abs().sum(dim=1) > 0).sum(dim=1)  # [B]
        gather_idx = n_laid.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        logits = logits.gather(1, gather_idx).squeeze(1)                     # [B, V]
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.argmax(dim=-1) == target).float().mean()
    return {"loss": loss, "accuracy": accuracy}


def compute_loss_packed(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Packed teacher-forced loss: score every layer position in one pass.

    `batch` carries the same fields as `compute_loss` plus a 2-D
    `target_tokens` of shape [B, SEQ_LEN] (with `-100` at positions past
    each example's actual structure length). `structure_matrix` here is the
    FULL deposited structure; the causal mask inside the model ensures
    position p only sees layers 0..p-1.

    Only valid (target != -100) positions contribute to loss and accuracy.
    """
    logits = model(
        lab=batch["lab"],
        pool_features=batch["pool_features"],
        pool_mask=batch["pool_mask"],
        structure_matrix=batch["structure_matrix"],
        pool_size=batch["pool_size"],
    )                                                                       # [B, SEQ_LEN, V]
    if logits.dim() != 3:
        raise RuntimeError(
            f"compute_loss_packed expects 3-D logits [B,SEQ_LEN,V]; got {tuple(logits.shape)}"
        )
    target = batch["target_tokens"]                                          # [B, SEQ_LEN]
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1), ignore_index=-100
    )
    pred = logits.argmax(dim=-1)
    valid = target != -100
    if valid.any():
        accuracy = (pred[valid] == target[valid]).float().mean()
    else:
        accuracy = torch.tensor(0.0, device=logits.device)
    return {"loss": loss, "accuracy": accuracy}


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
) -> Tuple[List[int], List[int], str]:
    """Autoregressively decode a structure for a single (lab, pool) example.

    Parameters
    ----------
    model : FlexMaterialMLP
    lab : torch.Tensor, [3]
        Normalised CIE Lab target.
    pool : list of MaterialNK
        The material pool for this example. Length must be ≤ M_MAX.
    device : torch.device
    max_layers : int
    sample : bool
        If True, sample tokens stochastically; if False, greedy argmax.
    temperature : float
        Sampling temperature (only used when sample=True).
    generator : torch.Generator, optional
        For reproducible sampling.

    Returns
    -------
    slot_indices : list of int
        The slot index used at each layer.
    thicknesses_nm : list of int
        The thickness (in nm) at each layer.
    termination : str
        'EOS' if model emitted EOS within max_layers, 'MAX_LEN' otherwise.

    Note: this returns slot indices rather than material names so the caller
    can decide how to present the result. Use `pool[slot_idx].name` to get
    the human-readable name for slot k.
    """
    if len(pool) > M_MAX:
        raise ValueError(f"Pool of {len(pool)} exceeds M_MAX={M_MAX}")

    pool_size = len(pool)
    pool_feats_unpadded = featurize_pool(pool, mode=model.config.feature_mode)
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)

    # Add batch dim and move to device.
    lab_b = lab.unsqueeze(0).to(device)
    pool_feats_b = pool_feats.unsqueeze(0).to(device)
    pool_mask_b = pool_mask.unsqueeze(0).to(device)
    pool_size_b = torch.tensor([pool_size], dtype=torch.long, device=device)

    slot_indices: List[int] = []
    thicknesses_nm: List[int] = []
    termination = "MAX_LEN"

    structure = torch.zeros(M_MAX, MAX_LAYERS, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for step in range(max_layers):
            structure_b = structure.unsqueeze(0)
            logits = model(
                lab=lab_b,
                pool_features=pool_feats_b,
                pool_mask=pool_mask_b,
                structure_matrix=structure_b,
                pool_size=pool_size_b,
            )  # [1, VOCAB_SIZE] for MLP, [1, SEQ_LEN, VOCAB_SIZE] for cross_attn
            if logits.dim() == 3:
                # cross_attn: read the prediction at this step's position.
                logits = logits[:, step, :]

            if sample:
                scaled = logits / max(temperature, 1e-6)
                probs = F.softmax(scaled, dim=-1)
                token_id = torch.multinomial(
                    probs, num_samples=1, generator=generator
                ).item()
            else:
                token_id = int(logits.argmax(dim=-1).item())

            if token_id == EOS_TOKEN:
                termination = "EOS"
                break

            _, thickness, slot_idx, _, kind = decode_token(token_id, pool=pool)
            assert kind == "LAYER"
            slot_indices.append(slot_idx)
            thicknesses_nm.append(thickness)

            # Update running structure for the next step.
            structure[slot_idx, step] = thickness / 200.0  # normalize_thickness

    return slot_indices, thicknesses_nm, termination


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    from src.material_features import load_jll_directory
    from src.materials_vocab import build_structure_matrix, encode_layer

    cfg = ModelConfig(
        feature_mode="raw_spectrum",
        d_model=256,  # small for smoke test
        n_layers=4,
        dropout=0.0,
        encoder_hidden=64,
        encoder_out=32,
    )
    print(f"Config tag: {cfg.tag()}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model param count: {n_params:,}")
    if hasattr(model, "_input_dim"):
        print(f"Backbone input dim: {model._input_dim}")

    # Build a tiny batch.
    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        print("[smoke] No JLL materials; skipping forward pass test.")
        raise SystemExit(0)

    pool_full = list(load_jll_directory(materials_dir).values())
    pool = pool_full[:5]  # 5-material pool

    lab = torch.tensor([0.5, 0.2, -0.4])  # normalised: L*=50, a*=25.6, b*=-51.2
    pool_feats_unpadded = featurize_pool(pool, mode=cfg.feature_mode)
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)

    # Pretend the structure-so-far is [slot 0 @ 100nm, slot 2 @ 50nm].
    structure = build_structure_matrix(slot_indices=[0, 2], thicknesses_nm=[100, 50])
    pool_size = torch.tensor([5], dtype=torch.long)

    batch = {
        "lab": lab.unsqueeze(0),
        "pool_features": pool_feats.unsqueeze(0),
        "pool_mask": pool_mask.unsqueeze(0),
        "structure_matrix": structure.unsqueeze(0),
        "pool_size": pool_size,
        "target_token": torch.tensor([encode_layer(slot_idx=1, thickness_nm=75)]),
    }

    # Forward pass.
    out = compute_loss(model, batch)
    print(f"\nForward pass:")
    print(f"  loss = {out['loss'].item():.4f}")
    print(f"  accuracy = {out['accuracy'].item():.4f}")

    # Verify masking: tokens for slots ≥ 5 should be -inf in logits.
    with torch.no_grad():
        logits = model(
            lab=batch["lab"],
            pool_features=batch["pool_features"],
            pool_mask=batch["pool_mask"],
            structure_matrix=batch["structure_matrix"],
            pool_size=batch["pool_size"],
        )
    valid_count = torch.isfinite(logits[0]).sum().item()
    expected_valid = 5 * NUM_THICKNESSES + 1  # 5 slots × 40 thicknesses + EOS
    print(f"  finite logits = {int(valid_count)} (expected {expected_valid}) "
          f"{'✓' if valid_count == expected_valid else '✗'}")

    # Autoregressive generation.
    print(f"\nAutoregressive generation:")
    slots, thicks, term = generate_structure(model, lab, pool, torch.device("cpu"))
    print(f"  termination: {term}")
    print(f"  slots: {slots}")
    print(f"  thicknesses: {thicks}")
    print(f"  resolved materials: {[pool[s].name for s in slots]}")

    # Permutation equivariance check: shuffle the pool and re-run; the decoded
    # material names (not slot indices) should match if the model truly
    # depends only on n,k features. Untrained model won't be invariant in
    # general, but the slot-indexed decoding should at least produce valid
    # tokens.

    # Cross-attention variant: same forward signature, same vocab, same
    # masking/loss path — should be drop-in for training/eval.
    print(f"\nCross-attention variant:")
    cfg_xa = ModelConfig(
        feature_mode="raw_spectrum",
        d_model=128, n_layers=2, n_heads=4, dropout=0.0,
        encoder_hidden=64, encoder_out=32,
        head_mode="cross_attn",
    )
    print(f"  Config tag: {cfg_xa.tag()}")
    model_xa = build_model(cfg_xa)
    n_params_xa = sum(p.numel() for p in model_xa.parameters())
    print(f"  Model param count: {n_params_xa:,}")
    out_xa = compute_loss(model_xa, batch)
    print(f"  loss = {out_xa['loss'].item():.4f}")
    print(f"  accuracy = {out_xa['accuracy'].item():.4f}")
    with torch.no_grad():
        logits_xa = model_xa(
            lab=batch["lab"],
            pool_features=batch["pool_features"],
            pool_mask=batch["pool_mask"],
            structure_matrix=batch["structure_matrix"],
            pool_size=batch["pool_size"],
        )
    # cross_attn output is [B, SEQ_LEN, V]; check each position has the
    # expected number of finite logits.
    valid_per_pos = torch.isfinite(logits_xa[0]).sum(dim=-1).tolist()
    seq_len_xa = logits_xa.size(1)
    ok = all(v == expected_valid for v in valid_per_pos)
    print(f"  logits shape: {tuple(logits_xa.shape)}")
    print(f"  finite logits per position: {valid_per_pos} "
          f"(expected {expected_valid} each) {'✓' if ok else '✗'}")
    slots_xa, thicks_xa, term_xa = generate_structure(
        model_xa, lab, pool, torch.device("cpu")
    )
    print(f"  autoregressive: termination={term_xa}, slots={slots_xa}, "
          f"thicknesses={thicks_xa}")

    # Packed-loss smoke. Build a fake packed batch (target sequence per
    # example) and verify compute_loss_packed runs end-to-end.
    target_seq = torch.full((1, seq_len_xa), -100, dtype=torch.long)
    target_seq[0, 0] = encode_layer(slot_idx=0, thickness_nm=50)
    target_seq[0, 1] = encode_layer(slot_idx=2, thickness_nm=100)
    target_seq[0, 2] = EOS_TOKEN  # 3-layer structure
    packed_batch = {**batch, "target_tokens": target_seq}
    out_packed = compute_loss_packed(model_xa, packed_batch)
    print(f"  packed loss = {out_packed['loss'].item():.4f}, "
          f"acc = {out_packed['accuracy'].item():.4f}")
    out_packed["loss"].backward()
    n_with_grad = sum(1 for p in model_xa.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model_xa.parameters())
    print(f"  packed backward: {n_with_grad}/{n_total} params received grad")

    print("\n[smoke] OK")
