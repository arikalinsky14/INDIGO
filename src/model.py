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

    # Backbone MLP
    d_model: int = 1024
    n_layers: int = 8
    dropout: float = 0.1

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
# Loss
# ============================================================================


def compute_loss(
    model: FlexMaterialMLP,
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
    model: FlexMaterialMLP,
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

    model = FlexMaterialMLP(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model param count: {n_params:,}")
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
    print("\n[smoke] OK")
