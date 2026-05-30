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
   (M_MAX × NUM_THICKNESSES + EOS = 1281 tokens by default). At inference
   the user-supplied pool dictates which slots are valid; an output mask
   suppresses logits for unused slots before softmax.

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
# Cross-attention pointer head — alternative to the flatten-then-MLP backbone
# ============================================================================
#
# Rationale
# ---------
# The MLP backbone flattens [B, M_MAX, encoder_out] -> [B, M_MAX*encoder_out]
# and consumes it positionally. That throws away the set structure the
# encoder preserves: slot k at position k*E becomes a permutation-sensitive
# coordinate. pool_sampler shuffles slots each epoch so the model relearns
# invariance as augmentation, but it's paying for it twice.
#
# This head encodes each slot as a token (material embedding + that slot's
# structure stripe), runs a transformer encoder over the slot tokens, then
# cross-attends from a (lab + structure-summary + pool-size) query onto the
# slot tokens. Per-slot thickness logits come from a small head over
# (updated slot token, query); EOS is a separate scalar from the query.
#
# The output vocabulary layout is unchanged — token id = slot*NUM_THICKNESSES
# + thickness_idx, EOS at the end — so loss, masking, and decoding paths
# (build_output_mask_batch, decode_token, generate_structure) work as-is.


class FlexMaterialCrossAttn(nn.Module):
    """Pointer-head alternative to FlexMaterialMLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.material_encoder = MaterialEncoder(config)

        # Per-slot token = encoded material + slot's structure column (thicknesses
        # used at this slot across the layer positions).
        slot_in_dim = config.encoder_out + MAX_LAYERS
        self.slot_proj = nn.Linear(slot_in_dim, config.d_model)

        # Self-attention over the pool. Lets slots reason about each other
        # (e.g. "the only slot whose n,k matches this color is k=3").
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # Query input dim:
        #   3            : lab
        #   1            : pool_size_norm
        #   MAX_LAYERS   : per-layer thickness sequence so far (sum over slots;
        #                  since each layer position is filled by exactly one
        #                  slot, the sum equals that slot's thickness)
        query_in_dim = 3 + 1 + MAX_LAYERS
        self.query_proj = nn.Linear(query_in_dim, config.d_model)

        # Single cross-attention block: query attends to slot tokens.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Per-slot thickness head: combine each (post-attn) slot token with the
        # (post-attn) query state to produce NUM_THICKNESSES logits per slot.
        self.thickness_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, NUM_THICKNESSES),
        )

        # EOS head: a single scalar from the query state.
        self.eos_head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Init projection and head Linears; trust PyTorch defaults for the
        transformer modules (Xavier-uniform for attn, Kaiming-uniform for FFN)
        — they're tuned for the pre-LN / GELU stack."""
        for module in (
            self.slot_proj,
            self.query_proj,
            self.eos_head,
            *self.thickness_head.modules(),
        ):
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
        """Same signature as FlexMaterialMLP.forward."""
        B = lab.size(0)

        # 1. Encode each slot's n,k (shared weights — permutation-equivariant).
        emb = self.material_encoder(pool_features)            # [B, M_MAX, E]

        # 2. Build per-slot tokens (material + structure stripe), project.
        slot_in = torch.cat([emb, structure_matrix], dim=-1)  # [B, M_MAX, E+MAX_LAYERS]
        slot_tokens = self.slot_proj(slot_in)                 # [B, M_MAX, d_model]

        # 3. Self-attend over slots. PyTorch convention: key_padding_mask=True
        # for positions to IGNORE.
        key_padding_mask = ~pool_mask                         # [B, M_MAX]
        slot_tokens = self.slot_encoder(
            slot_tokens, src_key_padding_mask=key_padding_mask
        )                                                     # [B, M_MAX, d_model]

        # 4. Build query: lab + pool size + structure summary (per-layer thickness).
        pool_size_norm = (pool_size.float() / float(M_MAX)).unsqueeze(-1)  # [B, 1]
        struct_summary = structure_matrix.sum(dim=1)          # [B, MAX_LAYERS]
        query_in = torch.cat([lab, pool_size_norm, struct_summary], dim=1)
        query = self.query_proj(query_in).unsqueeze(1)        # [B, 1, d_model]

        # 5. Cross-attention: query attends to (post-self-attn) slot tokens.
        query_out = self.query_decoder(
            tgt=query,
            memory=slot_tokens,
            memory_key_padding_mask=key_padding_mask,
        )                                                     # [B, 1, d_model]
        query_state = query_out.squeeze(1)                    # [B, d_model]

        # 6. Per-slot thickness logits.
        query_broadcast = query_state.unsqueeze(1).expand(-1, M_MAX, -1)
        slot_query = torch.cat([slot_tokens, query_broadcast], dim=-1)
        thickness_logits = self.thickness_head(slot_query)    # [B, M_MAX, NUM_THICKNESSES]
        thickness_logits = thickness_logits.reshape(B, M_MAX * NUM_THICKNESSES)

        # 7. EOS logit, then concat to match the (slot×thickness, EOS) vocab layout.
        eos_logit = self.eos_head(query_state)                # [B, 1]
        logits = torch.cat([thickness_logits, eos_logit], dim=1)  # [B, VOCAB_SIZE]

        # 8. Output masking — same path as the MLP head.
        if apply_output_mask:
            mask = build_output_mask_batch(pool_size, device=logits.device)
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
    """Standard cross-entropy loss + accuracy.

    `batch` is expected to have:
        lab, pool_features, pool_mask, structure_matrix, pool_size,
        target_token

    The output mask is applied automatically inside the forward pass, so
    invalid tokens are -inf and contribute zero gradient through softmax.
    """
    logits = model(
        lab=batch["lab"],
        pool_features=batch["pool_features"],
        pool_mask=batch["pool_mask"],
        structure_matrix=batch["structure_matrix"],
        pool_size=batch["pool_size"],
    )
    target = batch["target_token"]
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.argmax(dim=-1) == target).float().mean()
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
            )  # [1, VOCAB_SIZE]

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
    valid_xa = torch.isfinite(logits_xa[0]).sum().item()
    print(f"  finite logits = {int(valid_xa)} (expected {expected_valid}) "
          f"{'✓' if valid_xa == expected_valid else '✗'}")
    slots_xa, thicks_xa, term_xa = generate_structure(
        model_xa, lab, pool, torch.device("cpu")
    )
    print(f"  autoregressive: termination={term_xa}, slots={slots_xa}, "
          f"thicknesses={thicks_xa}")

    print("\n[smoke] OK")
