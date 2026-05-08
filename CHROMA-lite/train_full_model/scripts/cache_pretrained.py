#!/usr/bin/env python3
"""
Cache Pretrained Pipeline Outputs for Full Model Training — TWO-PHASE

Phase 1 — RGB Prediction (GPU-bound, LLM):
    Iterate all texts → batched LLM generation with KV-CACHE → save RGB map
    KV-cache avoids reprocessing the full prompt at each autoregressive step.
    Only the new token is processed after the initial prompt encoding.
    ~14x speedup over recomputing the full sequence each step.

Phase 2 — Structure Logits (near-instant, tiny MLP):
    Load RGB map → for each example build teacher-forced (rgb, structure) pairs
    → batch ALL pairs into one MLP call → save sharded .pt files

Can be run as a single command (both phases) or split for flexibility:
    --phase 1    Run Phase 1 only (RGB prediction)
    --phase 2    Run Phase 2 only (structure logits, requires Phase 1 output)
    (default)    Run both phases sequentially

Usage:
    # Full pipeline:
    python train_full_model/scripts/cache_pretrained.py \\
        --text-to-rgb-checkpoint pretrain_text_to_rgb/data/checkpoints/<tag>/latest \\
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest

    # Phase 1 only (RGB prediction):
    python train_full_model/scripts/cache_pretrained.py --phase 1 \\
        --text-to-rgb-checkpoint pretrain_text_to_rgb/data/checkpoints/<tag>/latest

    # Phase 2 only (structure logits from saved RGBs):
    python train_full_model/scripts/cache_pretrained.py --phase 2 \\
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest
"""

import sys
import json
import time
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import (
    NUM_MATERIALS, MAX_LAYERS, VOCAB_SIZE, VOCAB_SIZE_WITH_ERROR,
    EOS_TOKEN, ERROR_TOKEN, encode_layer,
    MATERIAL_TO_IDX, normalize_thickness,
)
from src.dataset import FullModelDataset, FullTrainingExample, find_repo_root
from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# TextToRGB Model Loading
# ============================================================================

def load_text_to_rgb_model(checkpoint_dir: Path, device: torch.device):
    """Load pretrained TextToRGB model from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / 'compact_lm_head.pt').exists():
        log("[INFO] Detected constrained LLM text-to-RGB model")
        return _load_constrained_model(checkpoint_dir, device)
    raise FileNotFoundError(
        f"Cannot detect TextToRGB model type in {checkpoint_dir}.")


def _load_constrained_model(checkpoint_dir: Path, device: torch.device):
    sys.path.insert(0, str(_repo_root / 'pretrain_text_to_rgb'))
    from pretrain_text_to_rgb.src.model import (
        ConstrainedTextToRGBConfig, ConstrainedTextToRGBModel
    )

    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = ConstrainedTextToRGBConfig.from_dict(json.load(f))
    else:
        config = ConstrainedTextToRGBConfig()

    model = ConstrainedTextToRGBModel(config)
    model.load_model(device=device, inference_only=True)

    compact_path = checkpoint_dir / 'compact_lm_head.pt'
    if compact_path.exists():
        model.compact_lm_head.load_state_dict(
            torch.load(compact_path, map_location=device, weights_only=True))

    lora_path = checkpoint_dir / 'lora_adapters'
    if lora_path.exists():
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, str(lora_path))

    model.eval()
    return model


# ============================================================================
# Phase 1: Batched RGB Generation with KV-Cache
# ============================================================================

def _get_llm_base(model):
    """Navigate peft/HF wrapper to get the inner LlamaModel."""
    if hasattr(model.model, 'peft_config'):
        return model.model.base_model.model.model  # PeftModel → CausalLM → LlamaModel
    return model.model.model  # CausalLM → LlamaModel


def generate_rgb_batch_kvcache(
    model,
    texts: List[str],
    device: torch.device,
    max_new_tokens: int = 14,
) -> Tuple[List[Optional[Tuple[int, int, int]]], int]:
    """
    Batched RGB generation with KV-CACHE for ~14x LLM speedup.

    Step 0: Encode full prompts → get hidden states + cache all KV pairs
    Steps 1-13: Process ONLY the new token (1 token per example) using cached KVs

    Without KV-cache: 14 passes × ~760 tokens = 10,640 token-steps per example
    With KV-cache:    1 pass × ~760 tokens + 13 passes × 1 token = 773 token-steps
    Speedup: ~14x on the LLM forward pass component.

    Returns: (list of RGB tuples or None, n_parse_failures)
    """
    from pretrain_text_to_rgb.src.model import format_chat_input, parse_rgb_string

    model.eval()
    B = len(texts)
    base_model = _get_llm_base(model)

    # --- Tokenize and left-pad ---
    all_ids, all_lengths = [], []
    for text in texts:
        ids, _ = format_chat_input(text, model.tokenizer, model.config.max_text_len)
        all_ids.append(ids)
        all_lengths.append(ids.size(0))

    max_len = max(all_lengths)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (ids, length) in enumerate(zip(all_ids, all_lengths)):
        input_ids[i, max_len - length:] = ids.to(device)
        attention_mask[i, max_len - length:] = 1

    generated_compact_ids = [[] for _ in range(B)]
    done = [False] * B
    eos_compact_id = model.compact_vocab['eos_compact_id']
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(done):
                break

            # LLM forward with KV-cache
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Get last-token hidden states → compact logits
            last_hidden = outputs.last_hidden_state[:, -1, :]  # [B, dim]
            # Match compact_lm_head dtype (may be bfloat16 from checkpoint)
            head_dtype = next(model.compact_lm_head.parameters()).dtype
            logits = model.compact_lm_head(last_hidden.to(head_dtype))  # [B, n_valid]
            compact_ids = logits.argmax(dim=-1)  # [B]

            # Build next input (single token per example)
            next_ids = []
            for i in range(B):
                if done[i]:
                    next_ids.append(pad_id)
                    continue
                cid = compact_ids[i].item()
                if cid == eos_compact_id:
                    done[i] = True
                    next_ids.append(pad_id)
                else:
                    generated_compact_ids[i].append(cid)
                    next_ids.append(
                        model.compact_vocab['compact_to_orig'][cid].item())

            # Next step: only the new token (KV-cache has everything before it)
            input_ids = torch.tensor(next_ids, dtype=torch.long,
                                     device=device).unsqueeze(1)  # [B, 1]

            # Extend attention mask by 1 (1 for active, 0 for done)
            next_mask = torch.tensor(
                [0 if done[i] else 1 for i in range(B)],
                dtype=torch.long, device=device).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    # Decode and parse
    results = []
    n_fail = 0
    for i in range(B):
        if not generated_compact_ids[i]:
            results.append(None)
            n_fail += 1
            continue
        orig_ids = [model.compact_vocab['compact_to_orig'][c].item()
                    for c in generated_compact_ids[i]]
        decoded = model.tokenizer.decode(orig_ids, skip_special_tokens=True)
        rgb = parse_rgb_string(decoded)
        if rgb is None:
            n_fail += 1
        results.append(rgb)

    return results, n_fail


def run_phase1(
    text_to_rgb_model,
    dataset: FullModelDataset,
    device: torch.device,
    cache_dir: Path,
    text_batch_size: int = 64,
) -> Path:
    """
    Phase 1: Generate all RGB predictions and save to disk.

    Saves:
        cache_dir/rgb_predictions.pt — dict mapping global_idx → (R, G, B) or None
    """
    rgb_path = cache_dir / 'rgb_predictions.pt'
    total = len(dataset)
    log(f"\n{'='*60}")
    log(f"[Phase 1] Generating RGB predictions for {total:,} examples")
    log(f"          Batch size: {text_batch_size}")
    log(f"{'='*60}")

    all_texts = []
    all_indices = []  # sequential index for mapping back
    idx = 0
    for example in dataset:
        if not example.text or not isinstance(example.text, str) or not example.text.strip():
            idx += 1
            continue
        all_texts.append(example.text)
        all_indices.append(idx)
        idx += 1

    log(f"[Phase 1] {len(all_texts):,} valid texts out of {total:,} examples")

    # Generate in batches
    rgb_map = {}  # index → (R,G,B) tuple or None
    n_processed = 0
    n_failed = 0
    t0 = time.time()

    for start in range(0, len(all_texts), text_batch_size):
        end = min(start + text_batch_size, len(all_texts))
        batch_texts = all_texts[start:end]
        batch_indices = all_indices[start:end]

        rgb_results, n_fail = generate_rgb_batch_kvcache(
            text_to_rgb_model, batch_texts, device)

        for bi, rgb in zip(batch_indices, rgb_results):
            rgb_map[bi] = rgb  # (R,G,B) tuple or None

        n_processed += len(batch_texts)
        n_failed += n_fail

        if n_processed % 5000 < text_batch_size or end == len(all_texts):
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta_h = (len(all_texts) - n_processed) / rate / 3600 if rate > 0 else 0
            log(f"  [Phase 1] {n_processed:,}/{len(all_texts):,} "
                f"({n_failed:,} parse fails) — {rate:.0f} ex/s, ETA {eta_h:.1f}h")

    # Save
    torch.save({
        'rgb_map': rgb_map,
        'n_total': len(all_texts),
        'n_failed': n_failed,
    }, rgb_path)

    elapsed = time.time() - t0
    log(f"\n[Phase 1 DONE] {n_processed:,} RGBs in {elapsed:.0f}s "
        f"({n_processed/max(elapsed,1):.0f} ex/s)")
    log(f"               {n_failed:,} parse failures → mid-gray fallback")
    log(f"               Saved to {rgb_path}")

    # Free LLM GPU memory before Phase 2
    del text_to_rgb_model
    torch.cuda.empty_cache()

    return rgb_path


# ============================================================================
# Phase 2: Batched Structure Logits (tiny MLP, very fast)
# ============================================================================

MAX_STEPS = 9  # max 8 layers + 1 EOS step


def run_phase2(
    rgb_to_structure: ThinFilmMLP,
    dataset: FullModelDataset,
    device: torch.device,
    cache_dir: Path,
    rgb_path: Path,
    shard_size: int = 10000,
    mlp_batch_size: int = 16384,
):
    """
    Phase 2: Compute all structure logits using cached RGB predictions.

    The RGB→Structure MLP is tiny (~4M params) so we can process thousands
    of (rgb, structure) pairs per second in large batches.
    """
    log(f"\n{'='*60}")
    log(f"[Phase 2] Computing structure logits from cached RGBs")
    log(f"          MLP batch size: {mlp_batch_size}")
    log(f"{'='*60}")

    # Load RGB predictions
    rgb_data = torch.load(rgb_path, map_location='cpu', weights_only=False)
    rgb_map = rgb_data['rgb_map']
    log(f"[Phase 2] Loaded {len(rgb_map):,} RGB predictions")

    MID_GRAY = (128, 128, 128)

    # Process dataset and build shards
    shard_idx = 0
    shard_data = _new_shard_data()
    n_processed = 0
    n_incorrect = 0
    n_rgb_failed = 0
    t0 = time.time()

    # Accumulate examples into large MLP batches
    example_buffer = []
    rgb_buffer = []
    global_idx = 0

    for example in dataset:
        if not example.text or not isinstance(example.text, str) or not example.text.strip():
            global_idx += 1
            continue

        # Look up cached RGB
        rgb_tuple = rgb_map.get(global_idx)
        if rgb_tuple is None:
            rgb_tuple = MID_GRAY
            n_rgb_failed += 1

        r, g, b = rgb_tuple
        rgb_norm = torch.tensor([r / 255.0, g / 255.0, b / 255.0],
                                dtype=torch.float32)

        example_buffer.append(example)
        rgb_buffer.append(rgb_norm)
        global_idx += 1

        # Process MLP batch
        if len(example_buffer) >= mlp_batch_size // MAX_STEPS:
            _process_mlp_batch(
                example_buffer, rgb_buffer,
                rgb_to_structure, device, shard_data)

            n_processed += len(example_buffer)
            n_incorrect += sum(1 for ex in example_buffer if ex.incorrect)
            example_buffer.clear()
            rgb_buffer.clear()

            # Save shard if full
            while len(shard_data['texts']) >= shard_size:
                _save_shard_slice(shard_data, shard_size, cache_dir, shard_idx)
                shard_idx += 1

            # Progress
            if n_processed % 50000 < (mlp_batch_size // MAX_STEPS):
                elapsed = time.time() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0
                log(f"  [Phase 2] {n_processed:,}/{len(dataset):,} "
                    f"— {rate:.0f} ex/s")

    # Process remaining
    if example_buffer:
        _process_mlp_batch(
            example_buffer, rgb_buffer,
            rgb_to_structure, device, shard_data)
        n_processed += len(example_buffer)
        n_incorrect += sum(1 for ex in example_buffer if ex.incorrect)

    # Save remaining shards
    while len(shard_data['texts']) >= shard_size:
        _save_shard_slice(shard_data, shard_size, cache_dir, shard_idx)
        shard_idx += 1

    # Save final partial shard
    if shard_data['texts']:
        shard_path = cache_dir / f'shard_{shard_idx:06d}.pt'
        _save_shard_to_file(shard_data, shard_path)
        log(f"  Shard {shard_idx}: {len(shard_data['texts'])} examples → {shard_path.name}")
        shard_idx += 1

    # Save index
    elapsed = time.time() - t0
    index = {
        'total_examples': n_processed,
        'n_incorrect': n_incorrect,
        'n_rgb_parse_failed': n_rgb_failed,
        'n_shards': shard_idx,
        'shard_size': shard_size,
        'max_steps': MAX_STEPS,
        'vocab_size': VOCAB_SIZE_WITH_ERROR,
    }
    with open(cache_dir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    log(f"\n[Phase 2 DONE] {n_processed:,} examples in {elapsed:.0f}s "
        f"({n_processed/max(elapsed,1):.0f} ex/s)")
    log(f"               {shard_idx} shards saved to {cache_dir}")
    log(f"               {n_incorrect:,} incorrect, {n_rgb_failed:,} RGB fallbacks")


def _process_mlp_batch(
    examples: List[FullTrainingExample],
    rgbs: List[torch.Tensor],
    mlp: ThinFilmMLP,
    device: torch.device,
    shard_data: Dict,
):
    """
    Build all (rgb, structure) pairs across examples × steps,
    run through MLP in one batched forward pass.
    """
    all_rgbs = []
    all_structures = []
    step_counts = []

    for example, rgb in zip(examples, rgbs):
        target_tokens = example.get_target_tokens()
        n_steps = len(target_tokens)
        step_counts.append(n_steps)

        for step in range(n_steps):
            structure = torch.zeros(NUM_MATERIALS, MAX_LAYERS, dtype=torch.float32)
            for layer_idx in range(min(step, len(example.target_materials))):
                mat = example.target_materials[layer_idx]
                thick = example.target_thicknesses[layer_idx]
                structure[MATERIAL_TO_IDX[mat], layer_idx] = normalize_thickness(thick)
            all_rgbs.append(rgb)
            all_structures.append(structure)

    if not all_rgbs:
        return

    # Single batched MLP forward
    rgb_batch = torch.stack(all_rgbs).to(device)
    struct_batch = torch.stack(all_structures).to(device)

    CHUNK = 16384
    logits_chunks = []
    with torch.no_grad():
        for s in range(0, rgb_batch.size(0), CHUNK):
            e = min(s + CHUNK, rgb_batch.size(0))
            out = mlp(rgb_batch[s:e], struct_batch[s:e])  # [chunk, 1001]
            logits_chunks.append(F.pad(out, (0, 1), value=-100.0).half().cpu())

    all_logits = torch.cat(logits_chunks, dim=0)  # [total_steps, 1002]

    # Regroup by example
    flat_idx = 0
    for ex_i, (example, n_steps) in enumerate(zip(examples, step_counts)):
        target_tokens = example.get_target_tokens()
        ex_logits = all_logits[flat_idx:flat_idx + n_steps]
        ex_targets = torch.tensor(target_tokens, dtype=torch.long)
        flat_idx += n_steps

        shard_data['texts'].append(example.text)
        shard_data['base_logits'].append(ex_logits)
        shard_data['target_tokens'].append(ex_targets)
        shard_data['n_steps'].append(n_steps)
        shard_data['incorrect'].append(example.incorrect)


# ============================================================================
# Shard I/O
# ============================================================================

def _new_shard_data() -> Dict:
    return {'texts': [], 'base_logits': [], 'target_tokens': [],
            'n_steps': [], 'incorrect': []}


def _save_shard_to_file(shard_data: Dict, shard_path: Path):
    """Save shard, padding all examples to MAX_STEPS."""
    n = len(shard_data['texts'])
    padded_logits = torch.zeros(n, MAX_STEPS, VOCAB_SIZE_WITH_ERROR, dtype=torch.float16)
    padded_targets = torch.full((n, MAX_STEPS), -1, dtype=torch.long)

    for i in range(n):
        ns = shard_data['n_steps'][i]
        padded_logits[i, :ns] = shard_data['base_logits'][i]
        padded_targets[i, :ns] = shard_data['target_tokens'][i]

    torch.save({
        'texts': shard_data['texts'],
        'base_logits': padded_logits,
        'target_tokens': padded_targets,
        'n_steps': torch.tensor(shard_data['n_steps'], dtype=torch.int8),
        'incorrect': torch.tensor(shard_data['incorrect'], dtype=torch.bool),
    }, shard_path)


def _save_shard_slice(shard_data: Dict, shard_size: int,
                      cache_dir: Path, shard_idx: int):
    """Extract shard_size examples from the front of shard_data, save to file."""
    slice_data = {
        'texts': shard_data['texts'][:shard_size],
        'base_logits': shard_data['base_logits'][:shard_size],
        'target_tokens': shard_data['target_tokens'][:shard_size],
        'n_steps': shard_data['n_steps'][:shard_size],
        'incorrect': shard_data['incorrect'][:shard_size],
    }
    shard_path = cache_dir / f'shard_{shard_idx:06d}.pt'
    _save_shard_to_file(slice_data, shard_path)
    log(f"  Shard {shard_idx}: {shard_size} examples → {shard_path.name}")

    # Remove saved examples from accumulator
    for key in shard_data:
        shard_data[key] = shard_data[key][shard_size:]


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache pretrained pipeline outputs (two-phase)')

    parser.add_argument('--text-to-rgb-checkpoint', type=str, default=None,
                        help='TextToRGB checkpoint (required for Phase 1)')
    parser.add_argument('--rgb-to-structure-checkpoint', type=str, default=None,
                        help='RGB→Structure checkpoint (required for Phase 2)')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None)
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--text-batch-size', type=int, default=64,
                        help='Batch size for TextToRGB LLM inference (Phase 1)')
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2],
                        help='0=both phases, 1=RGB only, 2=structure only')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[INFO] Device: {device}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    data_dir = Path(args.data_dir) if args.data_dir else (
        repo_root / 'create_dataset' / 'data_prompts')
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        repo_root / 'train_full_model' / 'data' / f'cache_{args.split}')
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Data dir:  {data_dir}")
    log(f"[INFO] Cache dir: {cache_dir}")
    log(f"[INFO] Phase:     {'both' if args.phase == 0 else args.phase}")

    # Load dataset (needed for both phases)
    log(f"\n[INFO] Loading {args.split} dataset...")
    dataset = FullModelDataset(
        data_dir, seed=args.seed, split=args.split,
        verbose=True, limit_examples=args.limit_examples)

    rgb_path = cache_dir / 'rgb_predictions.pt'

    # ================================================================
    # Phase 1: RGB Prediction
    # ================================================================
    if args.phase in (0, 1):
        if args.text_to_rgb_checkpoint is None:
            log("[ERROR] --text-to-rgb-checkpoint required for Phase 1")
            sys.exit(1)

        log(f"\n[INFO] Loading TextToRGB: {args.text_to_rgb_checkpoint}")
        text_to_rgb_model = load_text_to_rgb_model(
            Path(args.text_to_rgb_checkpoint), device)

        rgb_path = run_phase1(
            text_to_rgb_model, dataset, device, cache_dir,
            text_batch_size=args.text_batch_size)

        # text_to_rgb_model is deleted inside run_phase1

    # ================================================================
    # Phase 2: Structure Logits
    # ================================================================
    if args.phase in (0, 2):
        if args.rgb_to_structure_checkpoint is None:
            log("[ERROR] --rgb-to-structure-checkpoint required for Phase 2")
            sys.exit(1)

        if not rgb_path.exists():
            log(f"[ERROR] RGB predictions not found at {rgb_path}")
            log("        Run Phase 1 first, or use --phase 0 for both.")
            sys.exit(1)

        log(f"\n[INFO] Loading RGB→Structure: {args.rgb_to_structure_checkpoint}")
        ckpt_dir = Path(args.rgb_to_structure_checkpoint)
        with open(ckpt_dir / 'config.json') as f:
            mlp_config = ModelConfig.from_dict(json.load(f))
        rgb_to_structure = ThinFilmMLP(mlp_config).to(device)
        rgb_to_structure.load_state_dict(
            torch.load(ckpt_dir / 'model.pt', map_location=device,
                       weights_only=True))
        rgb_to_structure.eval()
        log(f"[INFO] RGB→Structure MLP loaded (d={mlp_config.d_model}, "
            f"L={mlp_config.n_layers})")

        run_phase2(
            rgb_to_structure, dataset, device, cache_dir, rgb_path,
            shard_size=args.shard_size)

    log(f"\n[INFO] All done! Cache directory: {cache_dir}")


if __name__ == '__main__':
    main()