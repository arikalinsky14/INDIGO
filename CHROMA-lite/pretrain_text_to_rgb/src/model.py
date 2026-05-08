"""
Model Architecture - Constrained LLM Fine-tuning for Text-to-RGB

Fine-tunes TinyLlama with a surgically reduced lm_head that can ONLY
predict valid [R,G,B] format tokens (45 tokens vs 32,000 original).

Pipeline:
    System prompt: "Output the color ... in the format [R,G,B]"
    + User text prompt
    -> TinyLlama (all layers, full input embeddings)
    -> Compact lm_head (hidden_dim -> 45)
    -> Constrained generation: [DDD,DDD,DDD]</s>

The input embedding layer is kept at full vocabulary (32,000) so the model
can process arbitrary text prompts. Only the output projection is reduced.

Training loss: Cross-entropy on assistant response tokens only (system +
user prompt positions are masked with label=-100).

Fine-tuning modes:
    - LoRA: Freeze base, train LoRA adapters + compact lm_head
    - Last-N: Freeze early layers, train last N transformer layers + lm_head
    - Full: Train all parameters (most expensive)
"""

import sys
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# System prompt used for all examples
# ============================================================================

SYSTEM_PROMPT = (
    "Output the color associated with this request in the format [R,G,B], "
    "e.g. [113,234,34]. Output ONLY the color value, nothing else."
)


# ============================================================================
# Compact Vocabulary
# ============================================================================

# Explicitly blocked token IDs
BLOCKED_TOKEN_IDS = {11167}  # ',\r' — comma with carriage return


def build_compact_vocab(tokenizer) -> Dict[str, torch.Tensor]:
    """
    Build the compact vocabulary mapping from a tokenizer.

    Scans the full vocabulary and identifies the 45 valid tokens for
    [R,G,B] output format (single digits, brackets, comma, whitespace, EOS).

    Returns dict with:
        valid_token_ids:   [n_valid] original token IDs
        orig_to_compact:   [vocab_size] mapping (-1 for invalid)
        compact_to_orig:   [n_valid] reverse mapping
        n_valid:           number of valid tokens
        vocab_size:        original vocabulary size
        eos_compact_id:    compact ID for EOS token
        eos_orig_id:       original ID for EOS token
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    valid_ids = set()

    for token_id in range(vocab_size):
        if token_id in BLOCKED_TOKEN_IDS:
            continue

        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        stripped = decoded.strip()

        # EOS
        if token_id == tokenizer.eos_token_id:
            valid_ids.add(token_id)
            continue

        # Skip BOS, UNK
        if token_id in [tokenizer.bos_token_id, 0]:
            continue

        # Pure whitespace
        if decoded and all(c == ' ' for c in decoded):
            valid_ids.add(token_id)
            continue

        if not stripped:
            continue

        # Single digit, bracket, or comma
        if len(stripped) == 1 and stripped in "0123456789[],":
            valid_ids.add(token_id)
            continue

    sorted_valid = sorted(valid_ids)
    n_valid = len(sorted_valid)

    orig_to_compact = torch.full((vocab_size,), -1, dtype=torch.long)
    for compact_id, orig_id in enumerate(sorted_valid):
        orig_to_compact[orig_id] = compact_id

    compact_to_orig = torch.tensor(sorted_valid, dtype=torch.long)

    eos_compact = orig_to_compact[tokenizer.eos_token_id].item()
    assert eos_compact >= 0, "EOS token not in valid set!"

    return {
        'valid_token_ids': torch.tensor(sorted_valid, dtype=torch.long),
        'orig_to_compact': orig_to_compact,
        'compact_to_orig': compact_to_orig,
        'n_valid': n_valid,
        'vocab_size': vocab_size,
        'eos_compact_id': eos_compact,
        'eos_orig_id': tokenizer.eos_token_id,
    }


# ============================================================================
# Configuration
# ============================================================================

class ConstrainedTextToRGBConfig:
    """
    Configuration for constrained LLM fine-tuning.

    Hyperparameters:
        encoder_name:     HuggingFace model name
        max_text_len:     Max tokenized length for user text prompt
        finetune_mode:    'lora', 'last_n', or 'full'
        lora_rank:        LoRA rank (only used if finetune_mode='lora')
        lora_alpha:       LoRA alpha scaling
        lora_targets:     Which modules to apply LoRA to
        unfreeze_layers:  Number of final layers to unfreeze (finetune_mode='last_n')
        train_lm_head:    Whether to train the compact lm_head (usually True)
        learning_rate:    Base learning rate
        batch_size:       Training batch size
        epochs:           Number of training epochs
        limit_examples:   Limit dataset size (for testing)
    """
    def __init__(
        self,
        encoder_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_text_len: int = 756,
        finetune_mode: str = "lora",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_targets: str = "q_proj,v_proj",
        unfreeze_layers: int = 4,
        train_lm_head: bool = True,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        epochs: int = 3,
        limit_examples: Optional[int] = None,
        grad_clip: float = 1.0,
        warmup_fraction: float = 0.03,
    ):
        self.encoder_name = encoder_name
        self.max_text_len = max_text_len
        self.finetune_mode = finetune_mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = lora_targets
        self.unfreeze_layers = unfreeze_layers
        self.train_lm_head = train_lm_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.limit_examples = limit_examples
        self.grad_clip = grad_clip
        self.warmup_fraction = warmup_fraction

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ConstrainedTextToRGBConfig':
        valid_keys = set(cls.__init__.__code__.co_varnames[1:])
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def encoder_slug(self) -> str:
        return self.encoder_name.replace("/", "-").replace(":", "-")

    def tag(self) -> str:
        slug = self.encoder_slug()
        base = (f"constrained_{slug}_{self.finetune_mode}"
                f"_lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}"
                f"_wd{self.weight_decay}_gc{self.grad_clip}_wu{self.warmup_fraction}")
        if self.finetune_mode == 'lora':
            targets_short = self.lora_targets.replace(',', '-')
            base += f"_r{self.lora_rank}_a{self.lora_alpha}_{targets_short}"
        elif self.finetune_mode == 'last_n':
            base += f"_un{self.unfreeze_layers}"
        if self.limit_examples is not None:
            base += f"_lim{self.limit_examples}"
        return base


# ============================================================================
# Model
# ============================================================================

class ConstrainedTextToRGBModel(nn.Module):
    """
    TinyLlama with surgically reduced lm_head for constrained [R,G,B] output.

    The input embedding stays at full vocabulary (32,000) so arbitrary text
    can be processed. The output lm_head is replaced with a compact projection
    to only the 45 valid tokens.
    """

    def __init__(self, config: ConstrainedTextToRGBConfig):
        super().__init__()
        self.config = config
        self.model = None       # HF CausalLM (loaded lazily)
        self.tokenizer = None
        self.compact_vocab = None
        self.compact_lm_head = None
        self._original_lm_head_removed = False

    def load_model(self, device: Optional[torch.device] = None,
                   inference_only: bool = False):
        """
        Load the base model, tokenizer, build compact vocab, perform surgery.

        Args:
            device:          Target device (None to stay on CPU).
            inference_only:  If True, skip fine-tuning configuration (LoRA, freezing, etc.).
                             Use this for zero-shot evaluation or inference from a
                             checkpoint where adapters are loaded separately.
        """
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[INFO] Loading model: {self.config.encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.encoder_name,
            dtype=torch.bfloat16,
        )

        n_base_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Base model loaded: {n_base_params:,} params")

        # Build compact vocabulary
        self.compact_vocab = build_compact_vocab(self.tokenizer)
        n_valid = self.compact_vocab['n_valid']
        print(f"[INFO] Compact vocab: {n_valid} valid tokens")

        # Perform lm_head surgery
        self._perform_surgery()

        # Configure fine-tuning mode (skip for inference-only / zero-shot)
        if not inference_only:
            self._configure_finetuning()
        else:
            print("[INFO] Inference-only mode: skipping fine-tuning configuration")

        # Move to device if specified
        if device is not None:
            self.to(device)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({n_trainable/n_total*100:.2f}%)")

    def _perform_surgery(self):
        """Replace lm_head with compact version (45 outputs instead of 32,000)."""
        row_indices = self.compact_vocab['valid_token_ids']
        hidden_dim = self.model.config.hidden_size
        n_valid = self.compact_vocab['n_valid']

        # Extract rows from original lm_head
        with torch.no_grad():
            original_weight = self.model.lm_head.weight.data  # [32000, hidden_dim]
            compact_weight = original_weight[row_indices].clone()  # [n_valid, hidden_dim]

        # Replace lm_head
        self.compact_lm_head = nn.Linear(hidden_dim, n_valid, bias=False)
        self.compact_lm_head.weight.data = compact_weight

        # Remove original lm_head to free memory
        del self.model.lm_head
        self._original_lm_head_removed = True

        old_params = 32000 * hidden_dim
        new_params = n_valid * hidden_dim
        print(f"[INFO] lm_head surgery: Linear({hidden_dim}, 32000) -> "
              f"Linear({hidden_dim}, {n_valid})")
        print(f"[INFO] lm_head params: {old_params:,} -> {new_params:,} "
              f"({new_params/old_params*100:.2f}%)")

    def _configure_finetuning(self):
        """Freeze/unfreeze parameters based on finetune_mode."""
        mode = self.config.finetune_mode

        if mode == 'full':
            # Everything is trainable
            for param in self.model.parameters():
                param.requires_grad = True
            self.compact_lm_head.weight.requires_grad = True
            print("[INFO] Fine-tune mode: full (all parameters trainable)")

        elif mode == 'last_n':
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze last N transformer layers
            n = self.config.unfreeze_layers
            total_layers = len(self.model.model.layers)
            start_layer = max(0, total_layers - n)

            for i in range(start_layer, total_layers):
                for param in self.model.model.layers[i].parameters():
                    param.requires_grad = True

            # Unfreeze final layer norm
            if hasattr(self.model.model, 'norm'):
                for param in self.model.model.norm.parameters():
                    param.requires_grad = True

            # Compact lm_head always trainable
            self.compact_lm_head.weight.requires_grad = True

            print(f"[INFO] Fine-tune mode: last_{n} "
                  f"(layers {start_layer}-{total_layers-1} + norm + lm_head)")

        elif mode == 'lora':
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False

            # Apply LoRA
            self._apply_lora()

            # Compact lm_head trainable
            self.compact_lm_head.weight.requires_grad = self.config.train_lm_head

            print(f"[INFO] Fine-tune mode: LoRA (rank={self.config.lora_rank}, "
                  f"alpha={self.config.lora_alpha})")

        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")

    def _apply_lora(self):
        """Apply LoRA adapters to target modules."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("[ERROR] peft library required for LoRA mode.")
            print("        Install with: pip install peft")
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

        # peft wraps the model in place
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def get_hidden_states(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Run input through the transformer backbone, return hidden states.

        Args:
            input_ids:      [B, seq_len] original vocab token IDs
            attention_mask:  [B, seq_len] attention mask

        Returns:
            hidden_states: [B, seq_len, hidden_dim]
        """
        # Access the inner transformer model (without lm_head).
        # LlamaForCausalLM.model -> LlamaModel
        # PeftModel wraps the CausalLM, so we need to go deeper.
        if hasattr(self.model, 'peft_config'):
            # peft-wrapped: PeftModel -> LlamaForCausalLM -> LlamaModel
            base = self.model.base_model.model.model
        else:
            # standard HF CausalLM: LlamaForCausalLM -> LlamaModel
            base = self.model.model

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
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with compact lm_head.

        Args:
            input_ids:      [B, seq_len] original vocab token IDs
            attention_mask:  [B, seq_len]
            labels:         [B, seq_len] compact vocab IDs, -100 for masked positions

        Returns:
            dict with 'loss', 'logits' (compact), and optionally 'predictions'
        """
        hidden = self.get_hidden_states(input_ids, attention_mask)
        compact_logits = self.compact_lm_head(hidden)  # [B, seq_len, n_valid]

        result = {'logits': compact_logits}

        if labels is not None:
            # Standard causal LM shift
            shift_logits = compact_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.compact_vocab['n_valid']),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

        return result

    def generate_rgb(
        self,
        texts: List[str],
        device: torch.device,
        max_new_tokens: int = 14,
        temperature: float = 0.0,
    ) -> List[Optional[Tuple[int, int, int]]]:
        """
        Generate [R,G,B] outputs for a batch of text prompts.

        Args:
            texts:          List of user text prompts
            device:         Target device
            max_new_tokens: Max tokens to generate (14 covers [255,255,255])
            temperature:    Sampling temperature (0 = greedy)

        Returns:
            List of (R, G, B) tuples, or None for failed parses
        """
        self.eval()
        results = []

        for text in texts:
            input_ids, _ = format_chat_input(text, self.tokenizer,
                                             self.config.max_text_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)

            generated_compact_ids = []

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    hidden = self.get_hidden_states(input_ids, attention_mask)
                    last_hidden = hidden[:, -1, :]  # [1, hidden_dim]
                    logits = self.compact_lm_head(last_hidden)  # [1, n_valid]

                    if temperature > 0:
                        probs = F.softmax(logits / temperature, dim=-1)
                        compact_id = torch.multinomial(probs, 1).item()
                    else:
                        compact_id = logits.argmax(dim=-1).item()

                    # Check for EOS
                    if compact_id == self.compact_vocab['eos_compact_id']:
                        break

                    generated_compact_ids.append(compact_id)

                    # Append original token ID for next input
                    orig_id = self.compact_vocab['compact_to_orig'][compact_id].item()
                    next_token = torch.tensor([[orig_id]], device=device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.ones_like(input_ids)

            # Decode compact IDs back to text and parse RGB
            orig_ids = [self.compact_vocab['compact_to_orig'][c].item()
                        for c in generated_compact_ids]
            decoded = self.tokenizer.decode(orig_ids, skip_special_tokens=True)
            rgb = parse_rgb_string(decoded)
            results.append(rgb)

        return results


# ============================================================================
# Chat Formatting
# ============================================================================

def format_chat_input(
    user_text: str,
    tokenizer,
    max_text_len: int = 756,
) -> Tuple[torch.Tensor, int]:
    """
    Format a user text prompt into the chat template for TinyLlama.

    Truncation strategy: if the full prompt exceeds max_text_len, we truncate
    the USER TEXT tokens while preserving the system prompt prefix and the
    assistant generation suffix (``<|assistant|>\\n``). This ensures the model
    always sees the generation cue.

    Returns:
        input_ids:       [seq_len] tokenized full prompt (system + user + assistant prefix)
        assistant_start: index where assistant response begins

    The TinyLlama chat format is:
        <|system|>\\n{system}\\n</s>\\n<|user|>\\n{user}\\n</s>\\n<|assistant|>\\n
    """
    # Build the system-only prefix (everything before user content)
    prefix_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},  # empty placeholder
    ]
    prefix_str = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True,
    )
    # The prefix_str contains the template with empty user content.
    # Split on the empty user content to get the before/after parts.
    # For TinyLlama: "...<|user|>\n\n</s>\n<|assistant|>\n"
    #                      prefix ^  ^ suffix

    # Simpler approach: tokenize the full prompt and the prefix separately
    # to find exact boundaries.
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    full_str = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    if len(full_ids) <= max_text_len:
        # Fits within budget — no truncation needed
        assistant_start = len(full_ids)
        return torch.tensor(full_ids, dtype=torch.long), assistant_start

    # Need to truncate. Figure out how many tokens the template overhead uses
    # by tokenizing with an empty user message.
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    overhead_tokens = len(prefix_ids)  # system + user wrapper + assistant prefix

    # Budget for user text tokens
    user_budget = max_text_len - overhead_tokens
    if user_budget < 1:
        # Extremely tight budget — just use prefix with no user content
        assistant_start = len(prefix_ids)
        return torch.tensor(prefix_ids, dtype=torch.long), assistant_start

    # Tokenize just the user text, truncate, then rebuild
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    user_ids_truncated = user_ids[:user_budget]
    user_text_truncated = tokenizer.decode(user_ids_truncated, skip_special_tokens=True)

    # Rebuild with truncated user text
    truncated_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text_truncated},
    ]
    truncated_str = tokenizer.apply_chat_template(
        truncated_messages, tokenize=False, add_generation_prompt=True,
    )
    truncated_ids = tokenizer.encode(truncated_str, add_special_tokens=False)

    # Final safety check — the decode/re-encode may shift length slightly
    if len(truncated_ids) > max_text_len:
        truncated_ids = truncated_ids[:max_text_len]

    assistant_start = len(truncated_ids)
    return torch.tensor(truncated_ids, dtype=torch.long), assistant_start


def format_training_example(
    user_text: str,
    rgb: Tuple[int, int, int],
    tokenizer,
    compact_vocab: Dict[str, torch.Tensor],
    max_text_len: int = 756,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Format a single training example: user text + target RGB.

    Returns dict with:
        input_ids:  [seq_len] original vocab token IDs (full sequence)
        labels:     [seq_len] compact vocab IDs for assistant tokens, -100 elsewhere

    Returns None if the example can't be formatted (e.g., RGB out of range).
    """
    r, g, b = rgb
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        return None

    # Format the target as [R,G,B]
    target_str = f"[{r},{g},{b}]"

    # Get prompt (system + user + assistant prefix)
    prompt_ids, assistant_start = format_chat_input(
        user_text, tokenizer, max_text_len
    )

    # Tokenize the target response + EOS
    target_token_ids = tokenizer.encode(target_str, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    # Full input sequence: prompt + target + EOS
    full_ids = torch.cat([
        prompt_ids,
        torch.tensor(target_token_ids, dtype=torch.long),
        torch.tensor([eos_id], dtype=torch.long),
    ])

    # Labels: -100 for prompt positions, compact IDs for target positions
    orig_to_compact = compact_vocab['orig_to_compact']
    labels = torch.full_like(full_ids, -100)

    for i in range(assistant_start, len(full_ids)):
        orig_id = full_ids[i].item()
        compact_id = orig_to_compact[orig_id].item()
        if compact_id < 0:
            # Token not in compact vocab — this shouldn't happen for valid RGB
            print(f"[WARN] Token {orig_id} ({repr(tokenizer.decode([orig_id]))}) "
                  f"not in compact vocab! Skipping example.")
            return None
        labels[i] = compact_id

    return {
        'input_ids': full_ids,
        'labels': labels,
    }


# ============================================================================
# Collation
# ============================================================================

def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length examples into a padded batch.

    Pads input_ids with pad_token_id, labels with -100, attention_mask with 0.
    Left-pads for causal LM convention (so the last token aligns).
    """
    max_len = max(ex['input_ids'].size(0) for ex in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for ex in batch:
        seq_len = ex['input_ids'].size(0)
        pad_len = max_len - seq_len

        # Left-pad
        input_ids_list.append(F.pad(ex['input_ids'], (pad_len, 0), value=pad_token_id))
        labels_list.append(F.pad(ex['labels'], (pad_len, 0), value=-100))
        mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                          torch.ones(seq_len, dtype=torch.long)])
        attention_mask_list.append(mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.stack(labels_list),
    }


# ============================================================================
# RGB Parsing
# ============================================================================

def parse_rgb_string(s: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse a generated string into an (R, G, B) tuple.

    Handles formats like '[128,64,192]', '128,64,192', with optional spaces.
    Returns None if parsing fails or values are out of range.
    """
    s = s.strip().replace(' ', '')

    # Try [R,G,B] format
    match = re.match(r'^\[?(\d{1,3}),(\d{1,3}),(\d{1,3})\]?$', s)
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            return (r, g, b)

    return None


def rgb_to_normalized(rgb: Tuple[int, int, int]) -> torch.Tensor:
    """Convert (R, G, B) int tuple to normalized [0,1] tensor."""
    return torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0],
                        dtype=torch.float32)


def normalized_to_rgb(t: torch.Tensor) -> Tuple[int, int, int]:
    """Convert normalized [0,1] tensor to (R, G, B) int tuple."""
    return (
        max(0, min(255, round(t[0].item() * 255))),
        max(0, min(255, round(t[1].item() * 255))),
        max(0, min(255, round(t[2].item() * 255))),
    )