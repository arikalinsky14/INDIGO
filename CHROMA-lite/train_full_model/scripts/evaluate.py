#!/usr/bin/env python3
"""
Evaluation Script for CHROMA-Lite — Full Text → Structure Model

Evaluates the constraint-informed model that combines:
    [TinyLlama + LoRA] → ConstraintMLP → MixingMLP(cached_base_logits, constraint)

Modes:
- --low-compute: Fast evaluation with teacher forcing only (CE loss + accuracy)
                 Requires --cache-dir pointing to cached eval data.
- Full mode (default): Teacher forcing + autoregressive generation + optical sim + CIEDE2000
                       Requires --cache-dir (teacher forcing) AND
                       --rgb-to-structure-checkpoint (autoregressive base logits).
- --constraint-test: Evaluate on the 1200-example constraint test set.
                     Skips teacher forcing (no --cache-dir needed).
                     Reports per-type constraint pass/fail rates alongside
                     standard autoregressive metrics + CIEDE2000.
                     Requires --rgb-to-structure-checkpoint.

Usage:
    # Fast evaluation (teacher forcing only, needs eval cache):
    python train_full_model/scripts/evaluate.py \\
        --checkpoint train_full_model/data/checkpoints/<tag>/latest \\
        --cache-dir train_full_model/data/cache_test \\
        --low-compute

    # Full evaluation (+ autoregressive + optical sim):
    python train_full_model/scripts/evaluate.py \\
        --checkpoint train_full_model/data/checkpoints/<tag>/latest \\
        --cache-dir train_full_model/data/cache_test \\
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \\
        --limit-examples 500

    # Full evaluation with sampling:
    python train_full_model/scripts/evaluate.py \\
        --checkpoint train_full_model/data/checkpoints/<tag>/latest \\
        --cache-dir train_full_model/data/cache_test \\
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \\
        --sample-predictions --temperature 0.5

    # Constraint test set evaluation:
    python train_full_model/scripts/evaluate.py \\
        --checkpoint train_full_model/data/checkpoints/<tag>/latest \\
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \\
        --constraint-test
"""

import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, asdict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import (
    denormalize_rgb, NUM_MATERIALS, MAX_LAYERS, EOS_TOKEN, VOCAB_SIZE,
    build_structure_matrix, encode_layer, decode_token,
    MATERIAL_TO_IDX, normalize_thickness,
)
from src.dataset import TextThinFilmDataset, ConstraintTestDataset, find_repo_root
from src.constraints import StructuredConstraints, check_constraints

from train_full_model.src.model import (
    FullModelConfig, FullModel,
    format_constraint_input, collate_constraint_inputs,
)
from train_full_model.scripts.training import build_structure_matrices_from_tokens

# Pretrained RGB→Structure MLP (for autoregressive base logits)
from pretrain_rgb_to_structure.src.model import (
    ThinFilmMLP, ModelConfig as PretrainedMLPConfig,
)

# ERROR token lives one past EOS in the full-model vocabulary
ERROR_TOKEN = VOCAB_SIZE  # 1001
VOCAB_SIZE_WITH_ERROR = VOCAB_SIZE + 1  # 1002
ERROR_PAD_VALUE = -100.0  # sentinel logit for ERROR position on base logits

# Try to import optical simulation
OPTICAL_SIM_AVAILABLE = False
try:
    from src.optical_sim import OpticalSimulator, is_available as optical_is_available
    OPTICAL_SIM_AVAILABLE = optical_is_available()
    if not OPTICAL_SIM_AVAILABLE:
        from src.optical_sim import get_import_error
        print(f"[WARN] Optical simulation not available: {get_import_error()}")
except ImportError as e:
    print(f"[WARN] Could not import optical_sim module: {e}")

# Try to import matplotlib for color swatch
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[WARN] matplotlib not available — color swatch will be skipped")


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# COLOR UTILITIES  (identical to pretrain eval — kept self-contained)
# ============================================================================

def denormalize_rgb_float(rgb_norm: torch.Tensor) -> List[float]:
    """Convert normalized RGB [0-1] back to [0-255] as floats (full precision)."""
    return [c.item() * 255.0 for c in rgb_norm]


def sRGB_to_Lab(sRGB: Union[List[int], List[float]]) -> Tuple[float, float, float]:
    """Convert sRGB [0-255] to CIELAB under D65 illuminant."""
    rgb = np.array(sRGB, dtype=np.float64) / 255.0
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = M @ linear_rgb
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / white

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

    fxyz = f(xyz_n)
    L = 116.0 * fxyz[1] - 16.0
    a = 500.0 * (fxyz[0] - fxyz[1])
    b = 200.0 * (fxyz[1] - fxyz[2])
    return float(L), float(a), float(b)


def ciede2000(lab1: Tuple[float, float, float],
              lab2: Tuple[float, float, float]) -> float:
    """Compute CIEDE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - math.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = math.sqrt(a1p ** 2 + b1 ** 2)
    C2p = math.sqrt(a2p ** 2 + b2 ** 2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(dhp) > 180:
        dhp += -360.0 if dhp > 180 else 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))
    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hbp = (h1p + h2p) / 2.0
    if C1p * C2p != 0 and abs(h1p - h2p) > 180:
        hbp += 180.0 if h1p + h2p < 360 else -180.0
    T = (1.0
         - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dTheta = 30.0 * math.exp(-((hbp - 275) / 25.0) ** 2)
    RC = 2.0 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    SL = 1.0 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20.0 + (Lbp - 50) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2.0 * dTheta)) * RC
    return math.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def compute_color_difference(rgb1: Union[List[int], List[float]],
                             rgb2: Union[List[int], List[float]]) -> float:
    """CIEDE2000 between two sRGB colours (0-255 range)."""
    return ciede2000(sRGB_to_Lab(rgb1), sRGB_to_Lab(rgb2))


# ============================================================================
# RESULT CONTAINER
# ============================================================================

@dataclass
class EvalResult:
    idx: int
    gt_materials: List[str]
    gt_thicknesses: List[int]
    gt_sRGB: List[int]
    pred_materials: List[str]
    pred_thicknesses: List[int]
    pred_sRGB: Optional[List[float]]
    stop_reason: str          # 'EOS', 'ERROR', 'MAX_LEN'
    n_layers_gt: int
    n_layers_pred: int
    ciede2000: Optional[float]
    is_valid: bool
    constraint_checks: Optional[List[Dict]] = None  # per-constraint pass/fail
    constraints_passed: Optional[int] = None
    constraints_total: Optional[int] = None
    materials_set_detail: Optional[Dict] = None  # set breakdown for strict/helpful/extra


# ============================================================================
# CACHED DATASET (reused from training — for teacher forcing)
# ============================================================================

class CachedFullModelDataset(torch.utils.data.Dataset):
    """Random-access over cached pretrained pipeline outputs (sharded .pt)."""

    def __init__(self, cache_dir: Path, limit_examples: Optional[int] = None,
                 seed: int = 42):
        cache_dir = Path(cache_dir)
        with open(cache_dir / 'index.json') as f:
            self.index = json.load(f)
        self.n_shards = self.index['n_shards']
        self.shard_paths = sorted(cache_dir.glob('shard_*.pt'))
        assert len(self.shard_paths) == self.n_shards, (
            f"Expected {self.n_shards} shards, found {len(self.shard_paths)}")

        self._shard_sizes = []
        self._cumulative = [0]
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location='cpu', weights_only=False)
            n = len(shard['texts'])
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)
        self._full_total = self._cumulative[-1]

        self._index_map = None
        if limit_examples is not None and limit_examples < self._full_total:
            g = torch.Generator()
            g.manual_seed(seed)
            self._index_map = torch.randperm(self._full_total, generator=g)

        self.total = min(self._full_total, limit_examples or self._full_total)
        log(f"[CachedDataset] {self.total:,} examples across {self.n_shards} shards"
            f" (of {self._full_total:,} total)")

    @lru_cache(maxsize=4)
    def _load_shard(self, shard_idx: int):
        return torch.load(self.shard_paths[shard_idx],
                          map_location='cpu', weights_only=False)

    def _global_to_shard(self, global_idx: int) -> Tuple[int, int]:
        for s in range(self.n_shards):
            if global_idx < self._cumulative[s + 1]:
                return s, global_idx - self._cumulative[s]
        raise IndexError(f"Index {global_idx} out of range")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        real_idx = (self._index_map[idx].item()
                    if self._index_map is not None else idx)
        shard_idx, local_idx = self._global_to_shard(real_idx)
        shard = self._load_shard(shard_idx)
        n_steps = shard['n_steps'][local_idx].item()
        target_tokens = shard['target_tokens'][local_idx]  # [MAX_STEPS]

        # Reconstruct per-step structure matrices from target tokens
        structure_matrices = build_structure_matrices_from_tokens(
            target_tokens, n_steps)  # [n_steps, NUM_MATERIALS * MAX_LAYERS]

        return {
            'text': shard['texts'][local_idx],
            'base_logits': shard['base_logits'][local_idx, :n_steps],
            'target_tokens': target_tokens[:n_steps],
            'structure_matrices': structure_matrices,
            'n_steps': n_steps,
            'incorrect': shard['incorrect'][local_idx].item(),
        }


def collate_full_model(batch: List[Dict]) -> Dict:
    """Flatten variable-length cached examples into a single training-style batch."""
    texts = [item['text'] for item in batch]
    n_steps = torch.tensor([item['n_steps'] for item in batch], dtype=torch.long)
    base_logits = torch.cat([item['base_logits'] for item in batch], dim=0)
    target_tokens = torch.cat([item['target_tokens'] for item in batch], dim=0)
    structure_matrices = torch.cat(
        [item['structure_matrices'] for item in batch], dim=0)
    incorrect = torch.tensor([item['incorrect'] for item in batch], dtype=torch.bool)
    return {
        'texts': texts,
        'base_logits': base_logits,
        'target_tokens': target_tokens,
        'structure_matrices': structure_matrices,
        'n_steps': n_steps,
        'incorrect': incorrect,
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_full_model(checkpoint_dir: Path,
                    device: torch.device) -> Tuple[FullModel, FullModelConfig]:
    """
    Load a trained full-model checkpoint (LoRA + ConstraintMLP + MixingMLP).
    """
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / 'config.json') as f:
        config = FullModelConfig.from_dict(json.load(f))

    model = FullModel(config)

    # --- LLM + trained LoRA ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"[INFO] Loading LLM: {config.encoder_name}")
    model.tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_name, padding_side='left')
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        config.encoder_name, dtype=torch.bfloat16,
    ).to(device)
    for param in base_llm.parameters():
        param.requires_grad = False

    lora_dir = checkpoint_dir / 'constraint_lora'
    if lora_dir.exists():
        from peft import PeftModel
        log(f"[INFO] Loading LoRA adapters from {lora_dir}")
        model.llm = PeftModel.from_pretrained(base_llm, str(lora_dir)).to(device)
    else:
        log("[WARN] No constraint_lora directory found — using base LLM only")
        model.llm = base_llm

    # --- ConstraintMLP ---
    cmlp_path = checkpoint_dir / 'constraint_mlp.pt'
    model.constraint_mlp.load_state_dict(
        torch.load(cmlp_path, map_location=device, weights_only=True))
    model.constraint_mlp.to(device)

    # --- MixingMLP ---
    mmlp_path = checkpoint_dir / 'mixing_mlp.pt'
    model.mixing_mlp.load_state_dict(
        torch.load(mmlp_path, map_location=device, weights_only=True))
    model.mixing_mlp.to(device)

    model.eval()
    return model, config


def load_pretrained_mlp(checkpoint_dir: Path,
                        device: torch.device) -> ThinFilmMLP:
    """Load the frozen pretrained RGB→Structure MLP (for autoregressive base logits)."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'config.json') as f:
        mlp_config = PretrainedMLPConfig.from_dict(json.load(f))
    mlp = ThinFilmMLP(mlp_config)
    mlp.load_state_dict(
        torch.load(checkpoint_dir / 'model.pt', map_location=device, weights_only=True))
    mlp.to(device).eval()
    log(f"[INFO] Pretrained MLP loaded: {sum(p.numel() for p in mlp.parameters()):,} params")
    return mlp


# ============================================================================
# TEACHER FORCING EVALUATION
# ============================================================================

def evaluate_teacher_forcing(
    model: FullModel,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    limit_examples: Optional[int] = None,
) -> Dict:
    """
    Teacher forcing evaluation on cached data.

    Mirrors the training forward pass: tokenise texts → LLM + LoRA →
    ConstraintMLP → MixingMLP(cached_base_logits, constraint) → CE loss.

    Also computes per-example ERROR detection confusion matrix by checking
    whether the model's step-0 prediction is ERROR vs the ground-truth
    incorrect flag.
    """
    dataset = CachedFullModelDataset(cache_dir, limit_examples=limit_examples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_full_model,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_steps = 0

    # ERROR detection confusion matrix (per-example, using step-0 prediction)
    error_tp = 0  # incorrect example, model predicted ERROR at step 0
    error_fp = 0  # correct example,   model predicted ERROR at step 0
    error_tn = 0  # correct example,   model did NOT predict ERROR at step 0
    error_fn = 0  # incorrect example, model did NOT predict ERROR at step 0

    log("[INFO] Running teacher forcing evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            tokenized = collate_constraint_inputs(
                batch['texts'], model.tokenizer, model.config.max_text_len)

            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            base_logits = batch['base_logits'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            structure_matrices = batch['structure_matrices'].to(device)
            n_steps = batch['n_steps'].to(device)
            incorrect = batch['incorrect']  # [B] bool, stays on CPU

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cached_base_logits=base_logits,
                n_steps=n_steps,
                structure_matrices=structure_matrices,
                target_tokens=target_tokens,
            )

            count = base_logits.size(0)  # total steps in this batch
            total_loss += outputs['loss'].item() * count
            total_correct += int(outputs['accuracy'].item() * count)
            total_steps += count

            # --- Per-example ERROR detection via step-0 predictions ---
            # offsets[i] = index into flattened logits for example i's step 0
            offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                n_steps.cumsum(0)[:-1],
            ])
            step0_logits = outputs['logits'][offsets]       # [B, vocab_size]
            step0_preds = step0_logits.argmax(dim=-1).cpu() # [B]

            pred_error = (step0_preds == ERROR_TOKEN)
            gt_incorrect = incorrect

            error_tp += int((pred_error & gt_incorrect).sum())
            error_fp += int((pred_error & ~gt_incorrect).sum())
            error_tn += int((~pred_error & ~gt_incorrect).sum())
            error_fn += int((~pred_error & gt_incorrect).sum())

            if (batch_idx + 1) % 50 == 0:
                rl = total_loss / total_steps
                ra = total_correct / total_steps
                log(f"  Batch {batch_idx + 1}: loss={rl:.4f}, acc={ra:.3f}")

    avg_loss = total_loss / max(total_steps, 1)
    avg_acc = total_correct / max(total_steps, 1)

    # Derived metrics
    n_gt_incorrect = error_tp + error_fn
    n_gt_correct = error_tn + error_fp
    error_precision = error_tp / max(error_tp + error_fp, 1)
    error_recall = error_tp / max(error_tp + error_fn, 1)
    error_f1 = (2 * error_precision * error_recall
                / max(error_precision + error_recall, 1e-12))

    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'n_steps': total_steps,
        'n_examples': len(dataset),
        'error_detection': {
            'tp': error_tp,
            'fp': error_fp,
            'tn': error_tn,
            'fn': error_fn,
            'n_gt_incorrect': n_gt_incorrect,
            'n_gt_correct': n_gt_correct,
            'precision': error_precision,
            'recall': error_recall,
            'f1': error_f1,
        },
    }


# ============================================================================
# AUTOREGRESSIVE GENERATION (FULL MODEL)
# ============================================================================

def _get_constraint_embedding(
    model: FullModel,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Run text through LLM + LoRA → ConstraintMLP → constraint embedding [1, d_model].
    """
    input_ids, attention_mask = format_constraint_input(
        text, model.tokenizer, model.config.max_text_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    hidden = model.get_hidden_states(input_ids, attention_mask)  # [1, seq, dim]
    seq_len = attention_mask.sum(dim=1) - 1  # last non-pad position
    last_hidden = hidden[0, seq_len[0]].float().unsqueeze(0)  # [1, llm_dim]
    return model.constraint_mlp(last_hidden)  # [1, d_model]


def generate_structure_full(
    full_model: FullModel,
    pretrained_mlp: ThinFilmMLP,
    text: str,
    rgb: torch.Tensor,
    device: torch.device,
    max_layers: int = MAX_LAYERS,
    sample: bool = False,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[str], List[int], str]:
    """
    Autoregressively generate a thin-film structure using the full model.

    At each step:
        1. Pretrained MLP(rgb, structure) → base_logits [1001]
        2. Pad to [1002] (ERROR slot = -100)
        3. MixingMLP(base_logits_1002, constraint_embed, structure_flat) → final_logits [1002]
        4. Argmax (or sample) → next token

    Returns:
        (materials, thicknesses, stop_reason)
        stop_reason ∈ {'EOS', 'ERROR', 'MAX_LEN'}
    """
    full_model.eval()
    pretrained_mlp.eval()

    rgb = rgb.to(device)
    structure = torch.zeros(NUM_MATERIALS, MAX_LAYERS, device=device)
    materials: List[str] = []
    thicknesses: List[int] = []

    # Compute constraint embedding once for this example
    constraint_embed = _get_constraint_embedding(full_model, text, device)

    with torch.no_grad():
        for step in range(max_layers):
            # Base logits from pretrained MLP [1, VOCAB_SIZE=1001]
            base_logits_1001 = pretrained_mlp(
                rgb.unsqueeze(0), structure.unsqueeze(0))

            # Pad ERROR slot → [1, 1002]
            error_pad = torch.full(
                (1, 1), ERROR_PAD_VALUE, device=device, dtype=base_logits_1001.dtype)
            base_logits_1002 = torch.cat([base_logits_1001, error_pad], dim=-1)

            # Apply constraint via MixingMLP (with structure context)
            structure_flat = structure.view(1, -1)  # [1, NUM_MATERIALS * MAX_LAYERS]
            final_logits = full_model.mixing_mlp(
                base_logits_1002.float(), constraint_embed,
                structure_flat)  # [1, 1002]

            # Select token
            if sample and temperature > 0:
                probs = F.softmax(final_logits[0] / temperature, dim=-1)
                token_id = torch.multinomial(
                    probs, 1, generator=generator).item()
            else:
                token_id = final_logits.argmax(dim=-1).item()

            # ERROR
            if token_id == ERROR_TOKEN:
                return materials, thicknesses, 'ERROR'

            # EOS
            if token_id == EOS_TOKEN:
                return materials, thicknesses, 'EOS'

            # LAYER — decode and update structure
            material, thickness, _ = decode_token(token_id)
            materials.append(material)
            thicknesses.append(thickness)
            structure[MATERIAL_TO_IDX[material], step] = normalize_thickness(thickness)

    return materials, thicknesses, 'MAX_LEN'


# ============================================================================
# COLOR SWATCH
# ============================================================================

def create_color_swatch(results: List[EvalResult], output_path: str, n: int = 10):
    """Create a colour swatch comparing GT vs predicted colours."""
    if not MATPLOTLIB_AVAILABLE:
        log("[WARN] matplotlib not available — skipping colour swatch")
        return

    valid = [r for r in results if r.pred_sRGB is not None][:n]
    if not valid:
        log("[WARN] No valid results for colour swatch")
        return

    fig, axes = plt.subplots(len(valid), 3, figsize=(8, 2 * len(valid)))
    if len(valid) == 1:
        axes = [axes]

    for i, r in enumerate(valid):
        gt_c = [c / 255.0 for c in r.gt_sRGB]
        axes[i][0].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=gt_c))
        axes[i][0].set_xlim(0, 1); axes[i][0].set_ylim(0, 1)
        axes[i][0].axis('off')
        axes[i][0].set_title(f'GT: {r.gt_sRGB}')

        pred_c = [min(1.0, max(0.0, c / 255.0)) for c in r.pred_sRGB]
        axes[i][1].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=pred_c))
        axes[i][1].set_xlim(0, 1); axes[i][1].set_ylim(0, 1)
        axes[i][1].axis('off')
        axes[i][1].set_title(f'Pred: {[int(round(c)) for c in r.pred_sRGB]}')

        axes[i][2].axis('off')
        info = f'ΔE₀₀: {r.ciede2000:.2f}\nLayers: {r.n_layers_gt} → {r.n_layers_pred}'
        if r.stop_reason == 'ERROR':
            info += '\n(ERROR predicted)'
        axes[i][2].text(0.5, 0.5, info, ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"[INFO] Colour swatch saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate CHROMA-Lite full text → structure model')

    # Checkpoint
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to full-model checkpoint directory')

    # Data
    p.add_argument('--cache-dir', type=str, default=None,
                   help='Path to cached eval data (sharded .pt from cache_pretrained.py). '
                        'Required for --low-compute and full mode, not needed for --constraint-test.')
    p.add_argument('--data-dir', type=str, default=None,
                   help='Raw parquet dir (default: create_dataset/data_prompts)')
    p.add_argument('--split', type=str, default='validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit-examples', type=int, default=None)

    # Pretrained MLP (for autoregressive base logits)
    p.add_argument('--rgb-to-structure-checkpoint', type=str, default=None,
                   help='Pretrained RGB→Structure MLP checkpoint '
                        '(required for full eval, not needed for --low-compute)')

    # Architecture defaults (must match training; used for logging only)
    p.add_argument('--d-model', type=int, default=1024)
    p.add_argument('--constraint-layers', type=int, default=4)
    p.add_argument('--mixing-layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)

    # LLM / LoRA
    p.add_argument('--encoder', type=str,
                   default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    p.add_argument('--max-text-len', type=int, default=756)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-targets', type=str, default='q_proj,v_proj')

    # Eval settings
    p.add_argument('--batch-size', type=int, default=16,
                   help='Batch size for teacher forcing (matches training default)')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--swatch-examples', type=int, default=10)

    # Mode flags
    p.add_argument('--low-compute', action='store_true',
                   help='Teacher forcing only (skip autoregressive + optical sim)')
    p.add_argument('--no-optical-sim', action='store_true')
    p.add_argument('--no-swatch', action='store_true')

    # Sampling
    p.add_argument('--sample-predictions', action='store_true',
                   help='Use stochastic sampling instead of greedy argmax')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Sampling temperature (default: 1.0)')

    # Constraint test set evaluation
    p.add_argument('--constraint-test', action='store_true',
                   help='Evaluate on constraint test set (1200 examples). '
                        'Skips teacher forcing. Requires --rgb-to-structure-checkpoint.')
    p.add_argument('--constraint-test-csv', type=str, default=None,
                   help='Path to constraint test set CSV '
                        '(default: create_dataset/data_prompts/test_set.csv)')

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[INFO] Device: {device}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    # ================================================================
    # Load full model
    # ================================================================
    checkpoint_dir = Path(args.checkpoint)
    ckpt_step = checkpoint_dir.name  # e.g. "step_4000" or "latest"
    log(f"[INFO] Loading full model from {checkpoint_dir}")
    model, config = load_full_model(checkpoint_dir, device)

    n_constraint = sum(p.numel() for p in model.constraint_mlp.parameters())
    n_mixing = sum(p.numel() for p in model.mixing_mlp.parameters())
    n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    log(f"[INFO] Config tag: {config.tag()}")
    log(f"[INFO] Trainable params: LoRA={n_lora:,}  ConstraintMLP={n_constraint:,}  "
        f"MixingMLP={n_mixing:,}  total={n_lora + n_constraint + n_mixing:,}")

    # ================================================================
    # LOW-COMPUTE MODE
    # ================================================================
    if args.low_compute:
        if args.cache_dir is None:
            log("[ERROR] --cache-dir is required for --low-compute mode.")
            sys.exit(1)
        log("\n" + "=" * 60)
        log("LOW-COMPUTE MODE: Teacher Forcing Evaluation")
        log("=" * 60)

        tf = evaluate_teacher_forcing(
            model, Path(args.cache_dir), device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit_examples=args.limit_examples,
        )

        metrics = {
            'mode': 'low_compute',
            'teacher_forcing_loss': tf['loss'],
            'teacher_forcing_accuracy': tf['accuracy'],
            'n_steps': tf['n_steps'],
            'n_examples': tf['n_examples'],
            'config_tag': config.tag(),
            'error_detection': tf['error_detection'],
        }

        ed = tf['error_detection']
        log(f"\n--- Teacher Forcing Results ---")
        log(f"  Loss:     {tf['loss']:.4f}")
        log(f"  Accuracy: {tf['accuracy']:.3f} ({100 * tf['accuracy']:.1f}%)")
        log(f"  Steps:    {tf['n_steps']:,}")

        log(f"\n--- ERROR Detection (step-0 predictions) ---")
        log(f"  GT incorrect: {ed['n_gt_incorrect']:,}  |  GT correct: {ed['n_gt_correct']:,}")
        log(f"  TP: {ed['tp']:,}  FP: {ed['fp']:,}  TN: {ed['tn']:,}  FN: {ed['fn']:,}")
        log(f"  Precision: {ed['precision']:.3f}  Recall: {ed['recall']:.3f}  F1: {ed['f1']:.3f}")
        log("=" * 60)

        output_path = (Path(args.output) if args.output else
                       repo_root / 'train_full_model' / 'outputs'
                       / f'eval_{config.tag()}_{ckpt_step}_lowcompute.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'metrics': metrics}, f, indent=2)
        log(f"\n[INFO] Results saved to {output_path}")
        return

    # ================================================================
    # CONSTRAINT TEST MODE
    # ================================================================
    if args.constraint_test:
        if args.rgb_to_structure_checkpoint is None:
            log("[ERROR] --rgb-to-structure-checkpoint is required for --constraint-test.")
            sys.exit(1)

        log("\n" + "=" * 60)
        log("CONSTRAINT TEST MODE: Autoregressive + Constraint Adherence")
        log("=" * 60)

        # Load pretrained MLP
        log(f"\n[INFO] Loading pretrained RGB→Structure MLP from "
            f"{args.rgb_to_structure_checkpoint}")
        pretrained_mlp = load_pretrained_mlp(
            Path(args.rgb_to_structure_checkpoint), device)

        # Optical sim
        run_optical_sim = OPTICAL_SIM_AVAILABLE and not args.no_optical_sim
        simulator = None
        if run_optical_sim:
            log("[INFO] Optical simulation enabled — will compute predicted colours")
            simulator = OpticalSimulator(incidence_angle=0)
        elif args.no_optical_sim:
            log("[INFO] Optical simulation disabled by --no-optical-sim flag")
        else:
            log("[WARN] Optical simulation not available — CIEDE2000 will not be computed")

        # Sampling generator
        sampling_gen = None
        if args.sample_predictions:
            sampling_gen = torch.Generator(device=device)
            sampling_gen.manual_seed(args.seed)
            log(f"[INFO] Sampling with temperature={args.temperature}")

        # Load constraint test dataset
        csv_path = Path(args.constraint_test_csv) if args.constraint_test_csv else None
        constraint_dataset = ConstraintTestDataset(csv_path=csv_path, verbose=True)

        if args.limit_examples:
            n_eval = min(args.limit_examples, len(constraint_dataset))
        else:
            n_eval = len(constraint_dataset)

        log(f"[INFO] Evaluating {n_eval} constraint test examples")

        results: List[EvalResult] = []
        from collections import defaultdict
        constraint_type_counts = defaultdict(lambda: {'total': 0, 'passed': 0})

        for idx in range(n_eval):
            example = constraint_dataset[idx]

            pred_mats, pred_thicks, stop_reason = generate_structure_full(
                model, pretrained_mlp,
                text=example.text,
                rgb=example.rgb,
                device=device,
                sample=args.sample_predictions,
                temperature=args.temperature,
                generator=sampling_gen,
            )

            gt_sRGB_float = denormalize_rgb_float(example.rgb)
            gt_sRGB_int = denormalize_rgb(example.rgb)

            is_valid = len(pred_mats) > 0 and stop_reason != 'ERROR'
            pred_sRGB = None
            ciede_val = None

            if run_optical_sim and is_valid:
                try:
                    pred_sRGB = simulator.compute_color(pred_mats, pred_thicks)
                    ciede_val = compute_color_difference(gt_sRGB_float, pred_sRGB)
                except Exception as e:
                    log(f"[WARN] Optical sim failed for example {idx}: {e}")
                    pred_sRGB = None
                    ciede_val = None
                    is_valid = False

            # Constraint checking
            constraints = StructuredConstraints.from_dict(example.constraints)
            cr = check_constraints(pred_mats, pred_thicks, constraints)

            constraint_checks_list = [
                {'type': c.constraint_type, 'passed': c.passed, 'detail': c.detail}
                for c in cr.checks
            ]

            # Accumulate per-type stats
            for c in cr.checks:
                constraint_type_counts[c.constraint_type]['total'] += 1
                if c.passed:
                    constraint_type_counts[c.constraint_type]['passed'] += 1

            results.append(EvalResult(
                idx=idx,
                gt_materials=example.target_materials,
                gt_thicknesses=example.target_thicknesses,
                gt_sRGB=gt_sRGB_int,
                pred_materials=pred_mats,
                pred_thicknesses=pred_thicks,
                pred_sRGB=pred_sRGB,
                stop_reason=stop_reason,
                n_layers_gt=len(example.target_materials),
                n_layers_pred=len(pred_mats),
                ciede2000=ciede_val,
                is_valid=is_valid,
                constraint_checks=constraint_checks_list,
                constraints_passed=cr.n_passed,
                constraints_total=cr.n_total,
                materials_set_detail=cr.materials_set_detail,
            ))

            if (idx + 1) % 100 == 0:
                n_valid = sum(1 for r in results if r.is_valid)
                total_c = sum(r.constraints_total for r in results if r.constraints_total)
                passed_c = sum(r.constraints_passed for r in results if r.constraints_passed is not None)
                log(f"[INFO] {idx + 1}/{n_eval} examples: valid={n_valid}/{len(results)}, "
                    f"constraints={passed_c}/{total_c} "
                    f"({100 * passed_c / max(total_c, 1):.1f}%)")

        # ---- Compute metrics ----
        n_total = len(results)
        n_eos = sum(1 for r in results if r.stop_reason == 'EOS')
        n_error = sum(1 for r in results if r.stop_reason == 'ERROR')
        n_maxlen = sum(1 for r in results if r.stop_reason == 'MAX_LEN')
        n_valid = sum(1 for r in results if r.is_valid)
        n_exact = sum(1 for r in results
                      if r.pred_materials == r.gt_materials
                      and r.pred_thicknesses == r.gt_thicknesses)
        n_layers_match = sum(1 for r in results if r.n_layers_pred == r.n_layers_gt)
        layer_diffs = [abs(r.n_layers_pred - r.n_layers_gt) for r in results]
        avg_layer_diff = sum(layer_diffs) / max(n_total, 1)

        # Constraint aggregate metrics
        total_constraints = sum(r.constraints_total for r in results if r.constraints_total)
        total_passed = sum(r.constraints_passed for r in results if r.constraints_passed is not None)
        total_failed = total_constraints - total_passed
        n_all_pass = sum(1 for r in results
                         if r.constraints_total and r.constraints_total > 0
                         and r.constraints_passed == r.constraints_total)

        by_type = {}
        for ctype, counts in sorted(constraint_type_counts.items()):
            t, p = counts['total'], counts['passed']
            by_type[ctype] = {
                'count': t,
                'passed': p,
                'failed': t - p,
                'pass_rate': p / max(t, 1),
            }

        # Materials set breakdown (strict, helpful, extra)
        materials_breakdown = {}
        for mat_type in ('strict', 'helpful', 'extra'):
            details = [r.materials_set_detail for r in results
                       if r.materials_set_detail and r.materials_set_detail['type'] == mat_type]
            if details:
                n = len(details)
                jaccards = [d['jaccard'] for d in details]
                ja = np.array(jaccards)
                materials_breakdown[mat_type] = {
                    'count': n,
                    'subset_rate': sum(1 for d in details if d['is_subset']) / n,
                    'exact_rate': sum(1 for d in details if d['is_exact']) / n,
                    'superset_rate': sum(1 for d in details if d['is_superset']) / n,
                    'jaccard_mean': float(np.mean(ja)),
                    'jaccard_median': float(np.median(ja)),
                }

        constraint_metrics = {
            'total_constraints': total_constraints,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'overall_pass_rate': total_passed / max(total_constraints, 1),
            'all_constraints_satisfied_rate': n_all_pass / max(n_total, 1),
            'n_all_pass': n_all_pass,
            'by_type': by_type,
            'materials_breakdown': materials_breakdown,
        }

        metrics = {
            'mode': 'constraint_test',
            'sample_predictions': args.sample_predictions,
            'temperature': args.temperature if args.sample_predictions else None,
            'seed': args.seed,
            'n_examples': n_total,
            'eos_rate': n_eos / n_total if n_total > 0 else 0,
            'error_rate': n_error / n_total if n_total > 0 else 0,
            'maxlen_rate': n_maxlen / n_total if n_total > 0 else 0,
            'valid_rate': n_valid / n_total if n_total > 0 else 0,
            'exact_match': n_exact / n_total if n_total > 0 else 0,
            'layer_count_match': n_layers_match / n_total if n_total > 0 else 0,
            'avg_layer_diff': avg_layer_diff,
            'config_tag': config.tag(),
            'constraint_metrics': constraint_metrics,
        }

        ciede_values = [r.ciede2000 for r in results if r.ciede2000 is not None]
        if ciede_values:
            ca = np.array(ciede_values)
            metrics['ciede2000_mean'] = float(np.mean(ca))
            metrics['ciede2000_median'] = float(np.median(ca))
            metrics['ciede2000_q1'] = float(np.percentile(ca, 25))
            metrics['ciede2000_q3'] = float(np.percentile(ca, 75))
            metrics['ciede2000_min'] = float(np.min(ca))
            metrics['ciede2000_max'] = float(np.max(ca))
            metrics['ciede2000_n_computed'] = len(ciede_values)

        # ---- Print summary ----
        log("\n" + "=" * 60)
        log("CONSTRAINT TEST RESULTS")
        if args.sample_predictions:
            log(f"(Sampling: temperature={args.temperature}, seed={args.seed})")
        log("=" * 60)

        log(f"\n--- Autoregressive Generation ---")
        log(f"  Total examples:     {n_total}")
        log(f"  Valid predictions:  {n_valid} ({100 * n_valid / max(n_total, 1):.1f}%)")
        log(f"  EOS rate:           {100 * metrics['eos_rate']:.1f}%")
        log(f"  ERROR rate:         {100 * metrics['error_rate']:.1f}%")
        log(f"  MAX_LEN rate:       {100 * metrics['maxlen_rate']:.1f}%")
        log(f"  Exact match:        {100 * metrics['exact_match']:.1f}%")
        log(f"  Layer count match:  {100 * metrics['layer_count_match']:.1f}%")
        log(f"  Avg layer diff:     {avg_layer_diff:.2f}")

        if ciede_values:
            log(f"\n--- CIEDE2000 Colour Difference (ΔE₀₀) ---")
            log(f"  Mean:   {metrics['ciede2000_mean']:.2f}")
            log(f"  Median: {metrics['ciede2000_median']:.2f}")
            log(f"  Q1:     {metrics['ciede2000_q1']:.2f}")
            log(f"  Q3:     {metrics['ciede2000_q3']:.2f}")
            log(f"  Min:    {metrics['ciede2000_min']:.2f}")
            log(f"  Max:    {metrics['ciede2000_max']:.2f}")

        log(f"\n--- Constraint Adherence ---")
        log(f"  Total constraints:    {total_constraints}")
        log(f"  Passed:               {total_passed} ({100 * total_passed / max(total_constraints, 1):.1f}%)")
        log(f"  Failed:               {total_failed} ({100 * total_failed / max(total_constraints, 1):.1f}%)")
        log(f"  All-pass rate:        {100 * n_all_pass / max(n_total, 1):.1f}% "
            f"({n_all_pass}/{n_total} examples where every constraint satisfied)")
        log(f"\n  By type:")
        for ctype, info in sorted(by_type.items()):
            log(f"    {ctype:<28s} {info['passed']:>4d}/{info['count']:<4d} "
                f"({100 * info['pass_rate']:.1f}%)")

        if materials_breakdown:
            log(f"\n--- Materials Set Analysis ---")
            for mat_type in ('strict', 'helpful', 'extra'):
                if mat_type not in materials_breakdown:
                    continue
                mb = materials_breakdown[mat_type]
                label = "CONSTRAINT" if mat_type == "strict" else "SUGGESTION"
                log(f"  {mat_type.capitalize()} ({mb['count']} examples) — {label}")
                log(f"    Subset  (missing mats):   {100 * mb['subset_rate']:.1f}%")
                log(f"    Exact match:              {100 * mb['exact_rate']:.1f}%")
                log(f"    Superset (all + extras):  {100 * mb['superset_rate']:.1f}%")
                log(f"    Jaccard overlap:          mean={100 * mb['jaccard_mean']:.1f}%, "
                    f"median={100 * mb['jaccard_median']:.1f}%")

        log("=" * 60)

        # ---- Save ----
        if args.output:
            output_path = Path(args.output)
        else:
            suffix = ''
            if args.sample_predictions:
                suffix = f"_sampling_T{args.temperature}".replace('.', 'p')
            output_path = (repo_root / 'train_full_model' / 'outputs'
                           / f'eval_{config.tag()}_{ckpt_step}_constraint_test{suffix}.json')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': [asdict(r) for r in results],
            }, f, indent=2)
        log(f"\n[INFO] Results saved to {output_path}")

        # Colour swatch
        if not args.no_swatch and MATPLOTLIB_AVAILABLE and run_optical_sim:
            suffix = ''
            if args.sample_predictions:
                suffix = f"_sampling_T{args.temperature}".replace('.', 'p')
            swatch_path = output_path.parent / f'swatch_{config.tag()}_{ckpt_step}_constraint_test{suffix}.png'
            create_color_swatch(results, str(swatch_path), n=args.swatch_examples)

        return

    # ================================================================
    # FULL MODE
    # ================================================================
    if args.rgb_to_structure_checkpoint is None:
        log("[ERROR] --rgb-to-structure-checkpoint is required for full eval mode.")
        log("        Use --low-compute for teacher-forcing-only evaluation.")
        sys.exit(1)
    if args.cache_dir is None:
        log("[ERROR] --cache-dir is required for full eval mode.")
        log("        Use --constraint-test for constraint-only evaluation.")
        sys.exit(1)

    log("\n" + "=" * 60)
    if args.sample_predictions:
        log(f"FULL MODE: Teacher Forcing + Autoregressive (SAMPLING, T={args.temperature})")
    else:
        log("FULL MODE: Teacher Forcing + Autoregressive Generation")
    log("=" * 60)

    # --- Phase 1: Teacher Forcing ---
    log("\n--- Phase 1: Teacher Forcing ---")
    tf = evaluate_teacher_forcing(
        model, Path(args.cache_dir), device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_examples=args.limit_examples,
    )

    # --- Load pretrained MLP for autoregressive ---
    log(f"\n[INFO] Loading pretrained RGB→Structure MLP from "
        f"{args.rgb_to_structure_checkpoint}")
    pretrained_mlp = load_pretrained_mlp(
        Path(args.rgb_to_structure_checkpoint), device)

    # --- Phase 2: Autoregressive Generation ---
    log("\n--- Phase 2: Autoregressive Generation ---")
    if args.sample_predictions:
        log(f"[INFO] Sampling with temperature={args.temperature}")

    run_optical_sim = OPTICAL_SIM_AVAILABLE and not args.no_optical_sim
    simulator = None
    if run_optical_sim:
        log("[INFO] Optical simulation enabled — will compute predicted colours")
        simulator = OpticalSimulator(incidence_angle=0)
    elif args.no_optical_sim:
        log("[INFO] Optical simulation disabled by --no-optical-sim flag")
    else:
        log("[WARN] Optical simulation not available — CIEDE2000 will not be computed")

    # Sampling generator for reproducibility
    sampling_gen = None
    if args.sample_predictions:
        sampling_gen = torch.Generator(device=device)
        sampling_gen.manual_seed(args.seed)

    # Load raw dataset (text + RGB + GT structure)
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / 'create_dataset' / 'data_prompts'
    log(f"[INFO] Loading raw {args.split} data from {data_dir}")
    raw_dataset = TextThinFilmDataset(
        data_dir, seed=args.seed, split=args.split,
        verbose=True, limit_examples=args.limit_examples,
    )

    results: List[EvalResult] = []
    skipped = 0

    for idx, example in enumerate(raw_dataset):
        # Guard: skip invalid text
        if example.text is None or not isinstance(example.text, str) or not example.text.strip():
            skipped += 1
            continue

        pred_mats, pred_thicks, stop_reason = generate_structure_full(
            model, pretrained_mlp,
            text=example.text,
            rgb=example.rgb,
            device=device,
            sample=args.sample_predictions,
            temperature=args.temperature,
            generator=sampling_gen,
        )

        gt_sRGB_float = denormalize_rgb_float(example.rgb)
        gt_sRGB_int = denormalize_rgb(example.rgb)

        is_valid = len(pred_mats) > 0 and stop_reason != 'ERROR'
        pred_sRGB = None
        ciede_val = None

        if run_optical_sim and is_valid:
            try:
                pred_sRGB = simulator.compute_color(pred_mats, pred_thicks)
                ciede_val = compute_color_difference(gt_sRGB_float, pred_sRGB)
            except Exception as e:
                log(f"[WARN] Optical sim failed for example {idx}: {e}")
                pred_sRGB = None
                ciede_val = None
                is_valid = False

        results.append(EvalResult(
            idx=idx,
            gt_materials=example.materials,
            gt_thicknesses=example.thicknesses,
            gt_sRGB=gt_sRGB_int,
            pred_materials=pred_mats,
            pred_thicknesses=pred_thicks,
            pred_sRGB=pred_sRGB,
            stop_reason=stop_reason,
            n_layers_gt=len(example.materials),
            n_layers_pred=len(pred_mats),
            ciede2000=ciede_val,
            is_valid=is_valid,
        ))

        if (idx + 1) % 100 == 0:
            n_valid = sum(1 for r in results if r.is_valid)
            n_error = sum(1 for r in results if r.stop_reason == 'ERROR')
            if run_optical_sim:
                cv = [r.ciede2000 for r in results if r.ciede2000 is not None]
                mean_de = sum(cv) / len(cv) if cv else float('nan')
                log(f"[INFO] {idx + 1} examples: valid={n_valid}/{len(results)}, "
                    f"ERROR={n_error}, mean_ΔE={mean_de:.2f}")
            else:
                log(f"[INFO] {idx + 1} examples: valid={n_valid}/{len(results)}, "
                    f"ERROR={n_error}")

    if skipped:
        log(f"[INFO] Skipped {skipped} examples with invalid text")

    # ================================================================
    # Compute metrics
    # ================================================================
    n_total = len(results)
    n_eos = sum(1 for r in results if r.stop_reason == 'EOS')
    n_error = sum(1 for r in results if r.stop_reason == 'ERROR')
    n_maxlen = sum(1 for r in results if r.stop_reason == 'MAX_LEN')
    n_valid = sum(1 for r in results if r.is_valid)
    n_exact = sum(1 for r in results
                  if r.pred_materials == r.gt_materials
                  and r.pred_thicknesses == r.gt_thicknesses)
    n_layers_match = sum(1 for r in results if r.n_layers_pred == r.n_layers_gt)
    layer_diffs = [abs(r.n_layers_pred - r.n_layers_gt) for r in results]
    avg_layer_diff = sum(layer_diffs) / max(n_total, 1)

    metrics = {
        'mode': 'full_sampling' if args.sample_predictions else 'full',
        'sample_predictions': args.sample_predictions,
        'temperature': args.temperature if args.sample_predictions else None,
        'seed': args.seed,
        'n_examples': n_total,
        'teacher_forcing_loss': tf['loss'],
        'teacher_forcing_accuracy': tf['accuracy'],
        'teacher_forcing_error_detection': tf['error_detection'],
        'eos_rate': n_eos / n_total if n_total > 0 else 0,
        'error_rate': n_error / n_total if n_total > 0 else 0,
        'maxlen_rate': n_maxlen / n_total if n_total > 0 else 0,
        'valid_rate': n_valid / n_total if n_total > 0 else 0,
        'exact_match': n_exact / n_total if n_total > 0 else 0,
        'layer_count_match': n_layers_match / n_total if n_total > 0 else 0,
        'avg_layer_diff': avg_layer_diff,
        'config_tag': config.tag(),
        # Autoregressive ERROR detection — TextThinFilmDataset only has correct
        # examples, so any ERROR prediction is a false positive.
        'autoregressive_error_detection': {
            'note': 'all examples are correct (no incorrect prompts in raw dataset)',
            'fp': n_error,
            'tn': n_total - n_error,
            'false_positive_rate': n_error / max(n_total, 1),
        },
    }

    ciede_values = [r.ciede2000 for r in results if r.ciede2000 is not None]
    if ciede_values:
        ca = np.array(ciede_values)
        metrics['ciede2000_mean'] = float(np.mean(ca))
        metrics['ciede2000_median'] = float(np.median(ca))
        metrics['ciede2000_q1'] = float(np.percentile(ca, 25))
        metrics['ciede2000_q3'] = float(np.percentile(ca, 75))
        metrics['ciede2000_min'] = float(np.min(ca))
        metrics['ciede2000_max'] = float(np.max(ca))
        metrics['ciede2000_n_computed'] = len(ciede_values)

    # ================================================================
    # Print summary
    # ================================================================
    log("\n" + "=" * 60)
    log("EVALUATION RESULTS")
    if args.sample_predictions:
        log(f"(Sampling: temperature={args.temperature}, seed={args.seed})")
    log("=" * 60)

    log(f"\n--- Teacher Forcing ---")
    log(f"  Loss:     {tf['loss']:.4f}")
    log(f"  Accuracy: {tf['accuracy']:.3f} ({100 * tf['accuracy']:.1f}%)")

    ed = tf['error_detection']
    log(f"\n--- ERROR Detection (teacher forcing, step-0) ---")
    log(f"  GT incorrect: {ed['n_gt_incorrect']:,}  |  GT correct: {ed['n_gt_correct']:,}")
    log(f"  TP: {ed['tp']:,}  FP: {ed['fp']:,}  TN: {ed['tn']:,}  FN: {ed['fn']:,}")
    log(f"  Precision: {ed['precision']:.3f}  Recall: {ed['recall']:.3f}  F1: {ed['f1']:.3f}")

    log(f"\n--- Autoregressive Generation ---")
    log(f"  Total examples:     {n_total}")
    log(f"  Valid predictions:  {n_valid} ({100 * n_valid / max(n_total, 1):.1f}%)")
    log(f"  EOS rate:           {100 * metrics['eos_rate']:.1f}%")
    log(f"  ERROR rate:         {100 * metrics['error_rate']:.1f}%  (all are FP — dataset has no incorrect examples)")
    log(f"  MAX_LEN rate:       {100 * metrics['maxlen_rate']:.1f}%")
    log(f"  Exact match:        {100 * metrics['exact_match']:.1f}%")
    log(f"  Layer count match:  {100 * metrics['layer_count_match']:.1f}%")
    log(f"  Avg layer diff:     {avg_layer_diff:.2f}")

    if ciede_values:
        log(f"\n--- CIEDE2000 Colour Difference (ΔE₀₀) ---")
        log(f"  Mean:   {metrics['ciede2000_mean']:.2f}")
        log(f"  Median: {metrics['ciede2000_median']:.2f}")
        log(f"  Q1:     {metrics['ciede2000_q1']:.2f}")
        log(f"  Q3:     {metrics['ciede2000_q3']:.2f}")
        log(f"  Min:    {metrics['ciede2000_min']:.2f}")
        log(f"  Max:    {metrics['ciede2000_max']:.2f}")

    log("=" * 60)

    # ================================================================
    # Save
    # ================================================================
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = ''
        if args.sample_predictions:
            suffix = f"_sampling_T{args.temperature}".replace('.', 'p')
        output_path = (repo_root / 'train_full_model' / 'outputs'
                       / f'eval_{config.tag()}_{ckpt_step}{suffix}.json')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': [asdict(r) for r in results],
        }, f, indent=2)
    log(f"\n[INFO] Results saved to {output_path}")

    # Colour swatch
    if not args.no_swatch and MATPLOTLIB_AVAILABLE and run_optical_sim:
        suffix = ''
        if args.sample_predictions:
            suffix = f"_sampling_T{args.temperature}".replace('.', 'p')
        swatch_path = output_path.parent / f'swatch_{config.tag()}_{ckpt_step}{suffix}.png'
        create_color_swatch(results, str(swatch_path), n=args.swatch_examples)


if __name__ == '__main__':
    main()