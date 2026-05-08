#!/usr/bin/env python3
"""
Evaluation Script for CHROMA-Lite - Constrained LLM Text-to-RGB

Evaluates the constrained LLM by generating [R,G,B] outputs for test prompts
and comparing against ground truth. Supports two modes:

1. ZERO-SHOT (no checkpoint): Base TinyLlama with lm_head surgery only.
   Tests how well the LLM already "knows" colors without any fine-tuning.

2. FINE-TUNED (--checkpoint): Loads LoRA adapters or trained weights
   from a fine-tuning checkpoint.

Metrics: MAE (0-255 scale), CIEDE2000, parse success rate.
Outputs: JSON metrics file + color swatch PNG.

Usage:
    # Zero-shot baseline (no fine-tuning, just constrained decoding):
    python pretrain_text_to_rgb/scripts/evaluate.py

    # Evaluate a fine-tuned checkpoint:
    python pretrain_text_to_rgb/scripts/evaluate.py \\
        --checkpoint pretrain_text_to_rgb/data/checkpoints/<tag>/best

    # Quick test with limited examples:
    python pretrain_text_to_rgb/scripts/evaluate.py \\
        --limit-examples 50 --no-swatch

    # Evaluate with sampling:
    python pretrain_text_to_rgb/scripts/evaluate.py \\
        --checkpoint pretrain_text_to_rgb/data/checkpoints/<tag>/best \\
        --temperature 0.3
"""

import sys
import json
import argparse
import math
import time
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / 'pretrain_text_to_rgb'))

from src.dataset import TextThinFilmDataset, find_repo_root
from pretrain_text_to_rgb.src.model import (
    ConstrainedTextToRGBConfig,
    ConstrainedTextToRGBModel,
    format_chat_input,
    parse_rgb_string,
    normalized_to_rgb,
)

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ============================================================================
# Color Science (same as MLP head evaluate.py)
# ============================================================================

def sRGB_to_Lab(sRGB):
    """Convert sRGB [0-255] to CIELAB under D65 illuminant."""
    rgb = np.array(sRGB, dtype=np.float64) / 255.0
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92,
                          ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = M @ linear_rgb
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / white
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), t / (3 * delta**2) + 4/29)
    fxyz = f(xyz_n)
    return (float(116 * fxyz[1] - 16),
            float(500 * (fxyz[0] - fxyz[1])),
            float(200 * (fxyz[1] - fxyz[2])))


def ciede2000(lab1, lab2) -> float:
    """Compute CIEDE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    if C1p * C2p == 0: dhp = 0
    elif abs(dhp) > 180: dhp -= 360 if dhp > 180 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hbp = (h1p + h2p) / 2
    if C1p * C2p != 0 and abs(h1p - h2p) > 180:
        hbp += 180 if h1p + h2p < 360 else -180
    T = (1 - 0.17*math.cos(math.radians(hbp-30))
         + 0.24*math.cos(math.radians(2*hbp))
         + 0.32*math.cos(math.radians(3*hbp+6))
         - 0.20*math.cos(math.radians(4*hbp-63)))
    dTh = 30 * math.exp(-((hbp - 275) / 25)**2)
    RC = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    SL = 1 + 0.015*(Lbp-50)**2 / math.sqrt(20+(Lbp-50)**2)
    SC = 1 + 0.045*Cbp
    SH = 1 + 0.015*Cbp*T
    RT = -math.sin(math.radians(2*dTh)) * RC
    return math.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2
                     + RT*(dCp/SC)*(dHp/SH))


# ============================================================================
# Model Loading
# ============================================================================

def load_model(
    checkpoint_dir: Optional[Path],
    config: ConstrainedTextToRGBConfig,
    device: torch.device,
) -> ConstrainedTextToRGBModel:
    """
    Load model in one of two modes:

    1. checkpoint_dir is None  -> zero-shot base model with lm_head surgery only
    2. checkpoint_dir provided -> load fine-tuned weights from checkpoint
    """
    model = ConstrainedTextToRGBModel(config)

    if checkpoint_dir is None:
        # Zero-shot: just load base model + perform surgery
        log("[INFO] Mode: ZERO-SHOT (base model + lm_head surgery, no fine-tuning)")
        model.load_model(device=device, inference_only=True)
        return model

    # Fine-tuned: load checkpoint
    log(f"[INFO] Mode: FINE-TUNED (loading from {checkpoint_dir})")

    # Load config from checkpoint
    ckpt_config_path = checkpoint_dir / 'config.json'
    if ckpt_config_path.exists():
        with open(ckpt_config_path) as f:
            ckpt_config = ConstrainedTextToRGBConfig.from_dict(json.load(f))
        # Update model config from checkpoint
        model.config = ckpt_config
        log(f"[INFO] Loaded config from checkpoint: {ckpt_config.tag()}")

    # Load base model + surgery (skip finetune config, we load adapters separately)
    model.load_model(device=device, inference_only=True)

    # Then load fine-tuned weights
    compact_lm_head_path = checkpoint_dir / 'compact_lm_head.pt'
    if compact_lm_head_path.exists():
        model.compact_lm_head.load_state_dict(
            torch.load(compact_lm_head_path, map_location=device, weights_only=True)
        )
        log("[INFO] Loaded fine-tuned compact_lm_head")

    # Load LoRA adapters or trainable params
    lora_path = checkpoint_dir / 'lora_adapters'
    trainable_path = checkpoint_dir / 'trainable_params.pt'

    if lora_path.exists():
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(
            model.model, str(lora_path)
        )
        log("[INFO] Loaded LoRA adapters")
    elif trainable_path.exists():
        trainable_state = torch.load(trainable_path, map_location=device,
                                     weights_only=True)
        # Load only the saved trainable parameters
        model_state = model.model.state_dict()
        model_state.update(trainable_state)
        model.model.load_state_dict(model_state)
        log(f"[INFO] Loaded {len(trainable_state)} trainable parameter tensors")

    return model


# ============================================================================
# Generation
# ============================================================================

def generate_batch(
    model: ConstrainedTextToRGBModel,
    texts: List[str],
    device: torch.device,
    max_new_tokens: int = 14,
    temperature: float = 0.0,
) -> List[dict]:
    """
    Generate [R,G,B] for a list of texts. Returns per-example results:
        {
            'pred_rgb': (R, G, B) or None,
            'raw_output': str,
            'n_tokens': int,
            'parse_ok': bool,
        }
    """
    model.eval()
    results = []

    for text in texts:
        input_ids, _ = format_chat_input(
            text, model.tokenizer, model.config.max_text_len
        )
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        generated_compact_ids = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                hidden = model.get_hidden_states(input_ids, attention_mask)
                last_hidden = hidden[:, -1, :]
                logits = model.compact_lm_head(last_hidden)

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    compact_id = torch.multinomial(probs, 1).item()
                else:
                    compact_id = logits.argmax(dim=-1).item()

                if compact_id == model.compact_vocab['eos_compact_id']:
                    break

                generated_compact_ids.append(compact_id)

                orig_id = model.compact_vocab['compact_to_orig'][compact_id].item()
                next_token = torch.tensor([[orig_id]], device=device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.ones_like(input_ids)

        # Decode
        orig_ids = [model.compact_vocab['compact_to_orig'][c].item()
                    for c in generated_compact_ids]
        raw_output = model.tokenizer.decode(orig_ids, skip_special_tokens=True)
        pred_rgb = parse_rgb_string(raw_output)

        results.append({
            'pred_rgb': pred_rgb,
            'raw_output': raw_output,
            'n_tokens': len(generated_compact_ids),
            'parse_ok': pred_rgb is not None,
        })

    return results


# ============================================================================
# Color Swatch
# ============================================================================

def _truncate_text(text: str, max_chars: int = 2500) -> str:
    if not text:
        return "(no text)"
    text = text.strip().replace('\n', ' ')
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    return text


def create_color_swatch(
    gt_rgbs: List[Tuple[int, int, int]],
    pred_rgbs: List[Tuple[int, int, int]],
    ciede_values: List[float],
    texts: List[str],
    raw_outputs: List[str],
    output_path: str,
    n: int = 10,
):
    """
    Color swatch: GT vs predicted RGB with text prompt, raw model output, and dE.
    Picks examples evenly across the dE range (best to worst).
    """
    if not MATPLOTLIB_AVAILABLE:
        log("[WARN] matplotlib not available - skipping color swatch")
        return

    n_total = len(ciede_values)
    if n_total == 0:
        log("[WARN] No valid examples for color swatch")
        return

    sorted_indices = np.argsort(ciede_values)
    n = min(n, n_total)
    if n_total <= n:
        pick_indices = sorted_indices
    else:
        positions = np.linspace(0, n_total - 1, n, dtype=int)
        pick_indices = sorted_indices[positions]

    row_height = 2.0
    fig, axes = plt.subplots(len(pick_indices), 5,
                             figsize=(20, row_height * len(pick_indices)),
                             gridspec_kw={'width_ratios': [1, 1, 0.6, 1.2, 5]})

    if len(pick_indices) == 1:
        axes = [axes]

    fig.text(0.06, 0.995, 'Ground Truth', ha='center', va='top',
             fontsize=10, fontweight='bold')
    fig.text(0.16, 0.995, 'Predicted', ha='center', va='top',
             fontsize=10, fontweight='bold')
    fig.text(0.24, 0.995, '\u0394E\u2080\u2080', ha='center', va='top',
             fontsize=10, fontweight='bold')
    fig.text(0.32, 0.995, 'Raw Output', ha='center', va='top',
             fontsize=10, fontweight='bold')
    fig.text(0.65, 0.995, 'Text Prompt', ha='center', va='top',
             fontsize=10, fontweight='bold')

    for i, idx in enumerate(pick_indices):
        gt = gt_rgbs[idx]
        pred = pred_rgbs[idx]
        de = ciede_values[idx]
        text = texts[idx] if idx < len(texts) else ""
        raw = raw_outputs[idx] if idx < len(raw_outputs) else ""

        # GT swatch
        axes[i][0].add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=[c/255 for c in gt]))
        axes[i][0].set_xlim(0, 1); axes[i][0].set_ylim(0, 1)
        axes[i][0].axis('off')
        axes[i][0].set_title(f'{list(gt)}', fontsize=7, pad=1)

        # Predicted swatch
        axes[i][1].add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=[c/255 for c in pred]))
        axes[i][1].set_xlim(0, 1); axes[i][1].set_ylim(0, 1)
        axes[i][1].axis('off')
        axes[i][1].set_title(f'{list(pred)}', fontsize=7, pad=1)

        # dE color-coded
        axes[i][2].axis('off')
        if de < 5:
            de_color = '#2a9d8f'
        elif de < 15:
            de_color = '#e9c46a'
        else:
            de_color = '#e63946'
        axes[i][2].text(0.5, 0.5, f'{de:.1f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=de_color)

        # Raw model output
        axes[i][3].axis('off')
        axes[i][3].text(0.5, 0.5, raw.strip(), ha='center', va='center',
                        fontsize=9, family='monospace')

        # Text prompt (truncated)
        axes[i][4].axis('off')
        display_text = _truncate_text(text, max_chars=2500)
        wrapped = textwrap.fill(display_text, width=90)
        axes[i][4].text(0.0, 0.5, wrapped, ha='left', va='center',
                        fontsize=6.5, family='monospace',
                        transform=axes[i][4].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"[INFO] Color swatch saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate constrained LLM for text-to-RGB')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned checkpoint directory. '
                             'If omitted, evaluates base model zero-shot.')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)

    # Model
    parser.add_argument('--encoder', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-text-len', type=int, default=756)

    # Generation
    parser.add_argument('--max-new-tokens', type=int, default=14,
                        help='Max tokens to generate (14 covers [255,255,255])')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0 = greedy)')

    # Swatch
    parser.add_argument('--swatch-examples', type=int, default=10)
    parser.add_argument('--no-swatch', action='store_true')

    # Verbosity
    parser.add_argument('--show-examples', type=int, default=20,
                        help='Print this many example predictions to stdout')

    return parser.parse_args()


# ============================================================================
# Main
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
    # Load model
    # ================================================================

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None

    config = ConstrainedTextToRGBConfig(
        encoder_name=args.encoder,
        max_text_len=args.max_text_len,
    )

    model = load_model(checkpoint_dir, config, device)
    model.eval()

    mode_label = "zero-shot" if checkpoint_dir is None else f"fine-tuned ({checkpoint_dir.name})"

    # ================================================================
    # Load test data
    # ================================================================

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'

    log(f"[INFO] Loading {args.split} data from {data_dir}")
    dataset = TextThinFilmDataset(
        data_dir, seed=args.seed, split=args.split,
        verbose=True, limit_examples=args.limit_examples,
    )

    texts, rgbs_norm = [], []
    skipped = 0
    for example in dataset:
        # Same guards as MLP head evaluate
        if example.text is None or not isinstance(example.text, str):
            skipped += 1
            continue
        if not example.text.strip():
            skipped += 1
            continue
        texts.append(example.text)
        rgbs_norm.append(example.rgb)
    log(f"[INFO] Collected {len(texts):,} examples ({skipped} skipped)")

    if len(texts) == 0:
        log("[ERROR] No valid text examples found!")
        sys.exit(1)

    # Convert ground truth to integer RGB
    gt_rgbs = [normalized_to_rgb(rgb) for rgb in rgbs_norm]

    # ================================================================
    # Generate predictions
    # ================================================================

    log(f"[INFO] Generating predictions for {len(texts):,} examples "
        f"(temp={args.temperature}, max_tokens={args.max_new_tokens})...")

    all_results = []
    t0 = time.time()

    for i in range(len(texts)):
        result = generate_batch(
            model, [texts[i]], device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )[0]
        all_results.append(result)

        done = i + 1
        if done % 50 == 0 or done == len(texts):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - done) / rate if rate > 0 else 0
            n_ok = sum(1 for r in all_results if r['parse_ok'])
            log(f"  Generated {done:,}/{len(texts):,} "
                f"({rate:.1f} ex/s, ETA {eta:.0f}s) "
                f"parse_ok={n_ok}/{done} ({n_ok/done*100:.0f}%)")

    gen_time = time.time() - t0
    log(f"[INFO] Generation complete: {len(texts):,} examples in {gen_time:.1f}s "
        f"({len(texts)/gen_time:.1f} ex/s)")

    # ================================================================
    # Compute metrics
    # ================================================================

    n_parse_ok = sum(1 for r in all_results if r['parse_ok'])
    n_parse_fail = len(all_results) - n_parse_ok
    parse_rate = n_parse_ok / len(all_results) * 100

    log(f"\n[INFO] Parse results: {n_parse_ok}/{len(all_results)} "
        f"({parse_rate:.1f}%) successfully parsed as [R,G,B]")

    if n_parse_fail > 0 and args.show_examples > 0:
        log(f"[INFO] Sample parse failures:")
        fail_count = 0
        for i, r in enumerate(all_results):
            if not r['parse_ok']:
                log(f"  [{i}] raw={repr(r['raw_output'])} "
                    f"(text: {texts[i][:80]}...)")
                fail_count += 1
                if fail_count >= min(10, args.show_examples):
                    break

    # Compute CIEDE2000 and MAE for successfully parsed examples
    ciede_values = []
    mae_values = []
    valid_indices = []  # indices where both parse succeeded and metrics computed

    for i, r in enumerate(all_results):
        if not r['parse_ok']:
            continue

        pred = r['pred_rgb']
        gt = gt_rgbs[i]

        # MAE per channel (0-255)
        mae = (abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]) + abs(pred[2] - gt[2])) / 3.0
        mae_values.append(mae)

        # CIEDE2000
        try:
            de = ciede2000(sRGB_to_Lab(gt), sRGB_to_Lab(pred))
            ciede_values.append(de)
            valid_indices.append(i)
        except Exception:
            ciede_values.append(float('nan'))

    ciede_arr = np.array([v for v in ciede_values if not math.isnan(v)])
    mae_arr = np.array(mae_values)

    # Build metrics dict
    config_tag = model.config.tag() if checkpoint_dir else "zero_shot"
    metrics = {
        'mode': 'zero-shot' if checkpoint_dir is None else 'fine-tuned',
        'checkpoint': str(checkpoint_dir) if checkpoint_dir else None,
        'n_examples': len(texts),
        'n_parse_ok': n_parse_ok,
        'n_parse_fail': n_parse_fail,
        'parse_rate': parse_rate,
        'temperature': args.temperature,
        'generation_time_s': gen_time,
    }

    if len(ciede_arr) > 0:
        metrics.update({
            'mae_255_mean': float(np.mean(mae_arr)),
            'mae_255_median': float(np.median(mae_arr)),
            'ciede2000_mean': float(np.mean(ciede_arr)),
            'ciede2000_median': float(np.median(ciede_arr)),
            'ciede2000_q1': float(np.percentile(ciede_arr, 25)),
            'ciede2000_q3': float(np.percentile(ciede_arr, 75)),
            'ciede2000_p90': float(np.percentile(ciede_arr, 90)),
            'ciede2000_max': float(np.max(ciede_arr)),
        })
    else:
        log("[WARN] No valid CIEDE2000 values computed!")

    # ================================================================
    # Print results
    # ================================================================

    log(f"\n{'='*60}")
    log(f"CONSTRAINED LLM TEXT-TO-RGB EVALUATION ({mode_label})")
    log(f"{'='*60}")
    log(f"  Examples:       {len(texts):,}")
    log(f"  Parse rate:     {n_parse_ok}/{len(texts)} ({parse_rate:.1f}%)")
    log(f"  Temperature:    {args.temperature}")
    log(f"  Gen time:       {gen_time:.1f}s ({len(texts)/gen_time:.1f} ex/s)")

    if len(ciede_arr) > 0:
        log(f"")
        log(f"  MAE (0-255):    {metrics['mae_255_mean']:.2f} mean, "
            f"{metrics['mae_255_median']:.2f} median")
        log(f"  CIEDE2000 mean: {metrics['ciede2000_mean']:.2f}")
        log(f"  CIEDE2000 med:  {metrics['ciede2000_median']:.2f}")
        log(f"  CIEDE2000 Q1:   {metrics['ciede2000_q1']:.2f}")
        log(f"  CIEDE2000 Q3:   {metrics['ciede2000_q3']:.2f}")
        log(f"  CIEDE2000 P90:  {metrics['ciede2000_p90']:.2f}")
        log(f"  CIEDE2000 max:  {metrics['ciede2000_max']:.2f}")
    log(f"{'='*60}")

    # Print sample predictions
    if args.show_examples > 0:
        log(f"\n[INFO] Sample predictions ({min(args.show_examples, len(texts))} examples):")
        for i in range(min(args.show_examples, len(texts))):
            r = all_results[i]
            gt = gt_rgbs[i]
            pred_str = str(list(r['pred_rgb'])) if r['parse_ok'] else "PARSE_FAIL"
            raw = r['raw_output'].strip()
            text_preview = texts[i][:60].replace('\n', ' ')
            de_str = ""
            if r['parse_ok']:
                try:
                    de = ciede2000(sRGB_to_Lab(gt), sRGB_to_Lab(r['pred_rgb']))
                    de_str = f" dE={de:.1f}"
                except Exception:
                    pass
            log(f"  [{i:3d}] gt={list(gt)} pred={pred_str} "
                f"raw={repr(raw)}{de_str}")
            log(f"         text: {text_preview}...")

    # ================================================================
    # Save metrics
    # ================================================================

    if args.output:
        output_path = Path(args.output)
    else:
        tag = config_tag
        output_path = (repo_root / 'pretrain_text_to_rgb' / 'outputs' /
                       f'eval_constrained_{tag}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({'metrics': metrics}, f, indent=2)
    log(f"[INFO] Metrics saved to {output_path}")

    # ================================================================
    # Color swatch
    # ================================================================

    if not args.no_swatch and MATPLOTLIB_AVAILABLE and len(valid_indices) > 0:
        swatch_gt = [gt_rgbs[i] for i in valid_indices]
        swatch_pred = [all_results[i]['pred_rgb'] for i in valid_indices]
        swatch_de = [ciede_values[j] for j in range(len(valid_indices))]
        swatch_texts = [texts[i] for i in valid_indices]
        swatch_raw = [all_results[i]['raw_output'] for i in valid_indices]

        swatch_path = output_path.parent / f'swatch_constrained_{config_tag}.png'
        create_color_swatch(
            swatch_gt, swatch_pred, swatch_de, swatch_texts, swatch_raw,
            str(swatch_path), n=args.swatch_examples,
        )
    elif not args.no_swatch and not MATPLOTLIB_AVAILABLE:
        log("[WARN] matplotlib not available - skipping color swatch")


if __name__ == "__main__":
    main()