"""
Full Model Architecture — Constraint-Informed Thin-Film Structure Prediction

Combines pretrained frozen pipelines with new trainable constraint extraction:

    Text ──┬──► [Frozen TextToRGB → Frozen RGB→Structure MLP] ──► Base logits [1002]
           │                    (cached offline)                        │
           │                                                           ▼
           └──► [TinyLlama + LoRA] ──► [ConstraintMLP] ──► [MixingMLP + residual] ──► Final logits
                   (trainable)           (trainable)            (trainable)

The MixingMLP uses a residual connection: output = base_logits + delta(base_logits, constraint, structure).
It receives the flattened structure matrix at each step, giving it visibility into what has
been generated so far (enabling constraint-aware decisions like layer count and material tracking).
At init, delta ≈ 0 (last-layer zero-init) so the model starts with pretrained behavior.

Vocabulary: 1002 tokens (1000 material-thickness + EOS + ERROR).
ERROR token (1001) is predicted for impossible/contradictory requests.
"""

import sys
import json
import math
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# System prompt for constraint extraction
# ============================================================================

CONSTRAINT_SYSTEM_PROMPT = (
    "You are an assistant materials scientist designing a multilayer thin-film "
    "structure for structural coloration. Your goal is to produce a stack that "
    "achieves the specific color requested by the user while meeting all provided "
    "design constraints.\n"
    "Follow these rules:\n"
    "Constraints: Respect all user-specified limits on layer thickness, layer count, "
    "material identity, and adjacency.\n"
    "Validation: Verify that all constraints are consistent and that the request "
    "uses only materials within the training set.\n"
    "Observation: Assume a 0 degree viewing angle (normal incidence) unless the "
    "user specifies otherwise.\n"
    "Training set materials:\n"
    "Ag (Silver), Al (Aluminum), Al2O3 (Aluminum Oxide), Au (Gold), "
    "AZO (Aluminum-Doped Zinc Oxide), Cr (Chromium), GaAs (Gallium Arsenide), "
    "GaInP (Gallium Indium Phosphide), GaP (Gallium Phosphide), Ge (Germanium), "
    "InP (Indium Phosphide), ITO (Indium Tin Oxide), Mn (Manganese), Ni (Nickel), "
    "Pd (Palladium), Pt (Platinum), Si3N4 (Silicon Nitride), SiO2 (Silicon Dioxide), "
    "Ti (Titanium), TiN (Titanium Nitride), TiO2 (Titanium Dioxide), "
    "a-Si (Amorphous Silicon), c-Si (Crystalline Silicon), W (Tungsten), ZnO (Zinc Oxide)"
)


# ============================================================================
# Configuration
# ============================================================================

class FullModelConfig:
    """
    Configuration for the full text → structure model.

    Architecture:
        d_model:           Width of ConstraintMLP and MixingMLP hidden layers
        constraint_layers: Depth of ConstraintMLP
        mixing_layers:     Depth of MixingMLP
        dropout:           Dropout rate

    LLM / LoRA:
        encoder_name:      HuggingFace model name for TinyLlama
        llm_hidden_dim:    Hidden dimension of the LLM (2048 for TinyLlama)
        max_text_len:      Max tokenized prompt length
        lora_rank:         LoRA rank for constraint extraction
        lora_alpha:        LoRA alpha scaling
        lora_targets:      Comma-separated target module names

    Training:
        learning_rate, batch_size, epochs, etc.

    Pretrained checkpoints (used by cache_pretrained.py only):
        text_to_rgb_checkpoint:     Path to frozen TextToRGB checkpoint
        rgb_to_structure_checkpoint: Path to frozen RGB→Structure checkpoint
    """

    def __init__(
        self,
        # Architecture
        d_model: int = 1024,
        constraint_layers: int = 4,
        mixing_layers: int = 4,
        dropout: float = 0.1,
        # LLM
        encoder_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        llm_hidden_dim: int = 2048,
        max_text_len: int = 756,
        # LoRA
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_targets: str = "q_proj,v_proj",
        # Vocab
        vocab_size: int = 1002,  # with ERROR token
        # Training
        learning_rate: float = 7e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        grad_accum_steps: int = 1,
        epochs: int = 10,
        limit_examples: Optional[int] = None,
        grad_clip: float = 1.0,
        warmup_fraction: float = 0.02,
        # Pretrained checkpoints (for caching step only)
        text_to_rgb_checkpoint: Optional[str] = None,
        rgb_to_structure_checkpoint: Optional[str] = None,
    ):
        self.d_model = d_model
        self.constraint_layers = constraint_layers
        self.mixing_layers = mixing_layers
        self.dropout = dropout
        self.encoder_name = encoder_name
        self.llm_hidden_dim = llm_hidden_dim
        self.max_text_len = max_text_len
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = lora_targets
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.limit_examples = limit_examples
        self.grad_clip = grad_clip
        self.warmup_fraction = warmup_fraction
        self.text_to_rgb_checkpoint = text_to_rgb_checkpoint
        self.rgb_to_structure_checkpoint = rgb_to_structure_checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FullModelConfig':
        valid_keys = set(cls.__init__.__code__.co_varnames[1:])
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def encoder_slug(self) -> str:
        return self.encoder_name.replace("/", "-").replace(":", "-")

    def tag(self) -> str:
        slug = self.encoder_slug()
        targets_short = self.lora_targets.replace(',', '-')
        base = (f"full_{slug}"
                f"_d{self.d_model}_cL{self.constraint_layers}_mL{self.mixing_layers}"
                f"_do{self.dropout}"
                f"_r{self.lora_rank}_a{self.lora_alpha}_{targets_short}"
                f"_lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}"
                f"_wd{self.weight_decay}_gc{self.grad_clip}_wu{self.warmup_fraction}")
        if self.limit_examples is not None:
            base += f"_lim{self.limit_examples}"
        return base


# ============================================================================
# ConstraintMLP
# ============================================================================

class ConstraintMLP(nn.Module):
    """
    Maps LLM last hidden state → constraint embedding.

    Architecture:
        Linear(llm_hidden_dim, d_model) → ReLU → Dropout
        [Linear(d_model, d_model) → ReLU → Dropout] × (n_layers - 1)

    Output: constraint embedding [B, d_model], computed ONCE per example
    and reused across all autoregressive structure-prediction steps.
    """

    def __init__(self, llm_hidden_dim: int = 2048, d_model: int = 1024,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []

        # First layer: project from LLM hidden dim
        layers.append(nn.Linear(llm_hidden_dim, d_model))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, llm_hidden_dim] last hidden state from LLM
        Returns:
            constraint_embed: [B, d_model]
        """
        return self.network(hidden_state)


# ============================================================================
# MixingMLP
# ============================================================================

class MixingMLP(nn.Module):
    """
    Fuses base structure logits with constraint embedding via residual delta.

    Architecture:
        Concat(base_logits, constraint_embed, structure_flat)
            → [input_dim = vocab_size + d_model + structure_dim]
        Linear(input_dim, d_model) → ReLU → Dropout
        [Linear(d_model, d_model) → ReLU → Dropout] × (n_layers - 2)
        Linear(d_model, vocab_size)  ← ZERO-INITIALIZED for identity at start

    Output: final_logits = base_logits + delta

    At init: delta ≡ 0, so output ≡ base_logits (pretrained behavior preserved).
    During training: model gradually learns constraint-informed corrections.

    The structure_flat input (flattened [NUM_MATERIALS × MAX_LAYERS] = 200 dims)
    gives the MixingMLP visibility into the structure generated so far, enabling
    constraint-aware decisions like layer count satisfaction and material tracking.
    """

    def __init__(self, vocab_size: int = 1002, d_model: int = 1024,
                 n_layers: int = 4, dropout: float = 0.1,
                 structure_dim: int = 200):
        super().__init__()
        self.structure_dim = structure_dim
        input_dim = vocab_size + d_model + structure_dim

        layers = []

        # First layer: project concatenated input
        layers.append(nn.Linear(input_dim, d_model))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer → delta logits (zero-init for residual identity)
        layers.append(nn.Linear(d_model, vocab_size))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for hidden layers, zero-init for output (residual identity)."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                if module is self.network[-1]:
                    # Last layer: zero init → delta starts at 0
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, base_logits: torch.Tensor,
                constraint_embed: torch.Tensor,
                structure_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_logits:      [N, vocab_size] from frozen pretrained pipeline
            constraint_embed: [N, d_model] from ConstraintMLP (expanded)
            structure_flat:   [N, structure_dim] flattened structure matrix at each step
        Returns:
            final_logits: [N, vocab_size] = base_logits + learned_delta
        """
        x = torch.cat([base_logits, constraint_embed, structure_flat], dim=-1)
        delta = self.network(x)
        return base_logits + delta


# ============================================================================
# FullModel
# ============================================================================

class FullModel(nn.Module):
    """
    End-to-end text → structure model.

    Wraps TinyLlama (with constraint LoRA) + ConstraintMLP + MixingMLP.
    Base logits from the pretrained pipeline are provided externally (cached).

    Training forward pass:
        1. Tokenize text → run through LLM + LoRA → last hidden state [B, 2048]
        2. ConstraintMLP(hidden) → constraint embedding [B, d_model]
        3. Expand constraint embedding to match total autoregressive steps
        4. MixingMLP(cached_base_logits, constraint_expanded) → final logits
        5. Cross-entropy loss vs target tokens
    """

    def __init__(self, config: FullModelConfig):
        super().__init__()
        self.config = config

        # Trainable components (created immediately)
        self.constraint_mlp = ConstraintMLP(
            llm_hidden_dim=config.llm_hidden_dim,
            d_model=config.d_model,
            n_layers=config.constraint_layers,
            dropout=config.dropout,
        )
        self.mixing_mlp = MixingMLP(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.mixing_layers,
            dropout=config.dropout,
        )

        # LLM (loaded lazily via load_llm)
        self.llm = None
        self.tokenizer = None

    def load_llm(self, device: torch.device,
                 gradient_checkpointing: bool = False):
        """Load LLM, freeze base weights, apply constraint LoRA.

        Args:
            device: Target device.
            gradient_checkpointing: Enable gradient checkpointing to reduce
                activation memory (trades ~30% compute for ~60-70% less memory).
                Essential for large models (8B+).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[INFO] Loading LLM: {self.config.encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.encoder_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.encoder_name,
            dtype=torch.bfloat16,
        ).to(device)

        # Auto-detect hidden dimension from loaded model
        detected_dim = self.llm.config.hidden_size
        if detected_dim != self.config.llm_hidden_dim:
            print(f"[INFO] Auto-updating llm_hidden_dim: "
                  f"{self.config.llm_hidden_dim} → {detected_dim}")
            self.config.llm_hidden_dim = detected_dim
            # Rebuild ConstraintMLP with correct input dimension
            self.constraint_mlp = ConstraintMLP(
                llm_hidden_dim=detected_dim,
                d_model=self.config.d_model,
                n_layers=self.config.constraint_layers,
                dropout=self.config.dropout,
            )

        # Freeze all base LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing before LoRA (reduces activation memory)
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[INFO] Gradient checkpointing enabled")

        # Apply LoRA for constraint extraction
        self._apply_lora()

        self._gradient_checkpointing = gradient_checkpointing
        print(f"[INFO] LLM loaded and LoRA applied (rank={self.config.lora_rank})")

    def _apply_lora(self):
        """Apply LoRA adapters to the LLM for constraint extraction."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("[ERROR] peft library required. Install with: pip install peft")
            sys.exit(1)

        targets = [t.strip() for t in self.config.lora_targets.split(',')]
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

    def get_hidden_states(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Run text through LLM backbone, return last hidden states.

        Args:
            input_ids:      [B, seq_len]
            attention_mask:  [B, seq_len]
        Returns:
            hidden_states: [B, seq_len, llm_hidden_dim]
        """
        # Navigate through peft wrapper to get the base transformer
        if hasattr(self.llm, 'peft_config'):
            base = self.llm.base_model.model.model
        else:
            base = self.llm.model

        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cached_base_logits: torch.Tensor,
        n_steps: torch.Tensor,
        structure_matrices: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            input_ids:          [B, seq_len] tokenized text prompts
            attention_mask:      [B, seq_len]
            cached_base_logits: [total_steps, vocab_size] flattened across batch
            n_steps:            [B] number of autoregressive steps per example
            structure_matrices: [total_steps, NUM_MATERIALS * MAX_LAYERS] flattened
                                structure state at each autoregressive step
            target_tokens:      [total_steps] ground truth token IDs (optional)

        Returns:
            dict with 'loss' (if targets given), 'logits' [total_steps, vocab_size]
        """
        # 1. Get LLM hidden states → last token embedding
        hidden = self.get_hidden_states(input_ids, attention_mask)  # [B, seq, dim]

        # Use last non-padding token for each example
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B], index of last real token
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lengths]  # [B, llm_hidden_dim]

        # 2. Constraint embedding (once per example)
        # autocast handles dtype; manual cast only when running without autocast
        constraint_embed = self.constraint_mlp(last_hidden)  # [B, d_model]

        # 3. Expand constraint embeddings for all autoregressive steps
        # repeat_interleave: example i's embedding is repeated n_steps[i] times
        constraint_expanded = torch.repeat_interleave(
            constraint_embed, n_steps, dim=0)  # [total_steps, d_model]

        # 4. Mix with cached base logits + structure context
        final_logits = self.mixing_mlp(
            cached_base_logits, constraint_expanded,
            structure_matrices)  # [total_steps, vocab_size]

        result = {'logits': final_logits}

        # 5. Loss
        if target_tokens is not None:
            loss = F.cross_entropy(final_logits, target_tokens)
            accuracy = (final_logits.argmax(dim=-1) == target_tokens).float().mean()
            result['loss'] = loss
            result['accuracy'] = accuracy

        return result


# ============================================================================
# Chat Formatting (for constraint extraction)
# ============================================================================

def format_constraint_input(
    user_text: str,
    tokenizer,
    max_text_len: int = 756,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Format a text prompt for constraint extraction via TinyLlama chat template.
    Returns input_ids and attention_mask (no generation, encoding only).
    """
    messages = [
        {"role": "system", "content": CONSTRAINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # Use chat template if available; fall back to simple concatenation
    # for base models (e.g. Meta-Llama-3-8B) that lack a chat_template.
    if getattr(tokenizer, 'chat_template', None) is not None:
        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        full_str = CONSTRAINT_SYSTEM_PROMPT + "\n\n" + user_text

    encoded = tokenizer(
        full_str,
        max_length=max_text_len,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
    return input_ids, attention_mask


def collate_constraint_inputs(
    texts: List[str],
    tokenizer,
    max_text_len: int = 756,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize + left-pad a batch of text prompts for constraint extraction.
    Left-pads for causal LM convention (last token = most recent).
    """
    all_ids, all_masks = [], []
    for text in texts:
        ids, mask = format_constraint_input(text, tokenizer, max_text_len)
        all_ids.append(ids)
        all_masks.append(mask)

    # Left-pad to longest in batch
    max_len = max(ids.size(0) for ids in all_ids)
    pad_id = tokenizer.pad_token_id

    padded_ids, padded_masks = [], []
    for ids, mask in zip(all_ids, all_masks):
        pad_len = max_len - ids.size(0)
        padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
        padded_masks.append(F.pad(mask, (pad_len, 0), value=0))

    return {
        'input_ids': torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_masks),
    }