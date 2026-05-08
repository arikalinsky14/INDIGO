#!/usr/bin/env python3
"""
Evaluation Script for CHROMA-Lite - MLP Version

Modes:
- --low-compute: Fast evaluation with teacher forcing only (CE loss + accuracy)
- Full mode (default): Teacher forcing + autoregressive generation + optical simulation + CIEDE2000
- --sample-predictions: Use stochastic sampling instead of greedy argmax for generation

Usage:
    # Fast evaluation (teacher forcing only)
    python scripts/evaluate.py --lr 2e-3 --low-compute
    
    # Full evaluation (includes autoregressive + optical sim)
    python scripts/evaluate.py --checkpoint data/checkpoints/<tag>/latest --limit-examples 100
    
    # Full evaluation with sampling (stochastic predictions)
    python scripts/evaluate.py --checkpoint data/checkpoints/<tag>/latest --sample-predictions --temperature 1.0
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import (
    denormalize_rgb, NUM_MATERIALS, MAX_LAYERS, EOS_TOKEN, 
    build_structure_matrix, encode_layer
)
from src.dataset import ThinFilmDataset, TrainingExample, find_repo_root
from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig, generate_structure, compute_loss

# Alias for backward compatibility
ThinFilmTransformer = ThinFilmMLP

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

# Try to import matplotlib for color swatch generation
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[WARN] matplotlib not available - color swatch will be skipped")


def _find_repo_root_eval() -> Path:
    """Find chroma-lite repo root for evaluation."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "src").exists() and (current / "scripts").exists():
            return current
        current = current.parent
    return Path.cwd()


# ============================================================================
# TEACHER FORCING EVALUATION
# ============================================================================

def collate_fn(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """
    Create training samples for autoregressive prediction.
    
    For N < 8 layers: N+1 samples (N layer predictions + EOS)
    For N = 8 layers: 8 samples (no EOS needed at max length)
    """
    all_rgb = []
    all_matrices = []
    all_targets = []
    
    for ex in examples:
        n_layers = len(ex.target_materials)
        
        # For N < 8: create N+1 steps; for N = 8: create 8 steps (no EOS)
        max_step = n_layers if n_layers < MAX_LAYERS else MAX_LAYERS - 1
        
        for step in range(max_step + 1):
            all_rgb.append(ex.rgb)
            
            if step == 0:
                all_matrices.append(torch.zeros(NUM_MATERIALS, MAX_LAYERS))
            else:
                all_matrices.append(build_structure_matrix(
                    ex.target_materials[:step], 
                    ex.target_thicknesses[:step]
                ))
            
            if step < n_layers:
                all_targets.append(encode_layer(
                    ex.target_materials[step], 
                    ex.target_thicknesses[step]
                ))
            else:
                all_targets.append(EOS_TOKEN)
    
    return {
        'rgb': torch.stack(all_rgb),
        'structure_matrix': torch.stack(all_matrices),
        'target_token': torch.tensor(all_targets, dtype=torch.long),
    }


def evaluate_teacher_forcing(model, dataset, device, batch_size=32, num_workers=4):
    """
    Fast evaluation using teacher forcing.
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.eval()
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    print("[INFO] Running teacher forcing evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            rgb = batch['rgb'].to(device)
            structure_matrix = batch['structure_matrix'].to(device)
            target_token = batch['target_token'].to(device)
            
            # Simple forward pass with MLP
            losses = compute_loss(model, rgb, structure_matrix, target_token)
            
            count = rgb.size(0)
            total_loss += losses['loss'].item() * count
            total_correct += int(losses['accuracy'].item() * count)
            total_samples += count
            
            if (batch_idx + 1) % 100 == 0:
                running_loss = total_loss / total_samples
                running_acc = total_correct / total_samples
                print(f"  Batch {batch_idx + 1}: loss={running_loss:.4f}, acc={running_acc:.3f}")
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'n_samples': total_samples,
    }


# ============================================================================
# COLOR UTILITIES
# ============================================================================

def denormalize_rgb_float(rgb_norm: torch.Tensor) -> List[float]:
    """
    Convert normalized RGB [0-1] back to [0-255] as floats (no rounding).
    
    This preserves full precision for accurate CIEDE2000 calculation.
    """
    return [c.item() * 255.0 for c in rgb_norm]


def sRGB_to_Lab(sRGB: Union[List[int], List[float]]) -> Tuple[float, float, float]:
    """
    Convert sRGB [0-255] to CIELAB under D65 illuminant.
    
    Accepts both int and float RGB values for flexibility.
    """
    rgb = np.array(sRGB, dtype=np.float64) / 255.0
    
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = M @ linear_rgb
    
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / white
    
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), t / (3 * delta**2) + 4/29)
    
    f_xyz = f(xyz_normalized)
    
    L = 116 * f_xyz[1] - 16
    a = 500 * (f_xyz[0] - f_xyz[1])
    b = 200 * (f_xyz[1] - f_xyz[2])
    
    return float(L), float(a), float(b)


def ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Compute CIEDE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360
    
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    dh_prime = h2_prime - h1_prime
    if C1_prime * C2_prime == 0:
        dh_prime = 0
    elif abs(dh_prime) > 180:
        if dh_prime > 180:
            dh_prime -= 360
        else:
            dh_prime += 360
    
    dH_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2))
    
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    h_bar_prime = (h1_prime + h2_prime) / 2
    if C1_prime * C2_prime != 0 and abs(h1_prime - h2_prime) > 180:
        if h1_prime + h2_prime < 360:
            h_bar_prime += 180
        else:
            h_bar_prime -= 180
    
    T = (1 - 0.17 * math.cos(math.radians(h_bar_prime - 30))
         + 0.24 * math.cos(math.radians(2 * h_bar_prime))
         + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6))
         - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63)))
    
    dTheta = 30 * math.exp(-((h_bar_prime - 275) / 25) ** 2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / math.sqrt(20 + (L_bar_prime - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -math.sin(math.radians(2 * dTheta)) * R_C
    
    dE = math.sqrt(
        (dL_prime / S_L)**2 +
        (dC_prime / S_C)**2 +
        (dH_prime / S_H)**2 +
        R_T * (dC_prime / S_C) * (dH_prime / S_H)
    )
    
    return dE


def compute_color_difference(rgb1: Union[List[int], List[float]], 
                             rgb2: Union[List[int], List[float]]) -> float:
    """
    Compute CIEDE2000 color difference between two sRGB colors.
    
    Accepts both int and float RGB values in [0-255] range.
    """
    lab1 = sRGB_to_Lab(rgb1)
    lab2 = sRGB_to_Lab(rgb2)
    return ciede2000(lab1, lab2)


@dataclass
class EvalResult:
    """Result for a single evaluation example."""
    idx: int
    gt_materials: List[str]
    gt_thicknesses: List[int]
    gt_sRGB: List[int]  # Stored as ints for readability
    pred_materials: List[str]
    pred_thicknesses: List[int]
    pred_sRGB: Optional[List[float]]  # Can be floats from optical sim
    stop_reason: str
    n_layers_gt: int
    n_layers_pred: int
    ciede2000: Optional[float]
    is_valid: bool


def create_color_swatch(results: List[EvalResult], output_path: str, n: int = 10):
    """Create a color swatch visualization comparing GT vs design colors."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available - skipping color swatch")
        return
    
    # Filter to results with valid predictions
    valid_results = [r for r in results if r.pred_sRGB is not None][:n]
    
    if not valid_results:
        print("[WARN] No valid results for color swatch")
        return
    
    fig, axes = plt.subplots(len(valid_results), 3, figsize=(8, 2 * len(valid_results)))
    
    if len(valid_results) == 1:
        axes = [axes]
    
    for i, result in enumerate(valid_results):
        # Ground truth color (normalize to 0-1 for matplotlib)
        gt_color = [c / 255 for c in result.gt_sRGB]
        axes[i][0].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=gt_color))
        axes[i][0].set_xlim(0, 1)
        axes[i][0].set_ylim(0, 1)
        axes[i][0].axis('off')
        axes[i][0].set_title(f'GT: {result.gt_sRGB}')
        
        # Design color (normalize to 0-1, handling floats)
        pred_color = [min(1.0, max(0.0, c / 255)) for c in result.pred_sRGB]
        axes[i][1].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=pred_color))
        axes[i][1].set_xlim(0, 1)
        axes[i][1].set_ylim(0, 1)
        axes[i][1].axis('off')
        # Display rounded values for readability
        pred_display = [int(round(c)) for c in result.pred_sRGB]
        axes[i][1].set_title(f'Pred: {pred_display}')
        
        # Info panel
        axes[i][2].axis('off')
        info_text = f'ΔE₀₀: {result.ciede2000:.2f}\nLayers: {result.n_layers_gt} → {result.n_layers_pred}'
        axes[i][2].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Color swatch saved to {output_path}")


def load_model(checkpoint_dir: Path, device: torch.device) -> Tuple[ThinFilmMLP, ModelConfig]:
    """Load model from checkpoint."""
    config_path = checkpoint_dir / 'config.json'
    model_path = checkpoint_dir / 'model.pt'
    
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
    
    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    
    model = ThinFilmMLP(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model, config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CHROMA-Lite MLP model')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-examples', type=int, default=None)
    
    # Model hyperparameters (simplified for MLP)
    parser.add_argument('--d-model', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    
    # Evaluation settings
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--swatch-examples', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Mode flags
    parser.add_argument('--low-compute', action='store_true')
    parser.add_argument('--no-optical-sim', action='store_true')
    parser.add_argument('--no-swatch', action='store_true')
    
    # Sampling options
    parser.add_argument('--sample-predictions', action='store_true',
                        help='Use stochastic sampling instead of greedy argmax')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0, higher = more random)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # Find repo root
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = Path(__file__).parent.parent
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    else:
        config = ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            limit_examples=args.limit_examples,
            epochs=args.epochs
        )
        checkpoint_dir = repo_root / 'pretrain_rgb_to_structure' / 'data' / 'checkpoints' / config.tag() / 'latest'
    
    print(f"[INFO] Loading model from {checkpoint_dir}")
    model, config = load_model(checkpoint_dir, device)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Config tag: {config.tag()}")
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / 'create_dataset' / 'data_prompts'
    
    limit = args.limit_examples
    dataset = ThinFilmDataset(data_dir, seed=args.seed, split=args.split, limit_examples=limit)
    print(f"[INFO] Evaluating {len(dataset)} examples from split '{args.split}'")
    
    # ========================================================================
    # LOW-COMPUTE MODE
    # ========================================================================
    if args.low_compute:
        print("\n" + "="*60)
        print("LOW-COMPUTE MODE: Teacher Forcing Evaluation")
        print("="*60)
        
        tf_results = evaluate_teacher_forcing(
            model, dataset, device, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        metrics = {
            'mode': 'low_compute',
            'teacher_forcing_loss': tf_results['loss'],
            'teacher_forcing_accuracy': tf_results['accuracy'],
            'n_samples': tf_results['n_samples'],
            'n_examples': len(dataset),
            'config_tag': config.tag(),
        }
        
        print(f"\n--- Teacher Forcing Results ---")
        print(f"  Loss:     {tf_results['loss']:.4f}")
        print(f"  Accuracy: {tf_results['accuracy']:.3f} ({100*tf_results['accuracy']:.1f}%)")
        print(f"  Samples:  {tf_results['n_samples']:,}")
        print("="*60)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = repo_root / 'pretrain_rgb_to_structure' / 'outputs' / f'eval_{config.tag()}_lowcompute.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'metrics': metrics}, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")
        
        return
    
    # ========================================================================
    # FULL MODE
    # ========================================================================
    print("\n" + "="*60)
    if args.sample_predictions:
        print(f"FULL MODE: Teacher Forcing + Autoregressive Generation (SAMPLING, T={args.temperature})")
    else:
        print("FULL MODE: Teacher Forcing + Autoregressive Generation")
    print("="*60)
    
    # Teacher forcing metrics
    print("\n--- Phase 1: Teacher Forcing ---")
    tf_results = evaluate_teacher_forcing(
        model, dataset, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Autoregressive generation
    print("\n--- Phase 2: Autoregressive Generation ---")
    if args.sample_predictions:
        print(f"[INFO] Using stochastic sampling with temperature={args.temperature}")
    
    run_optical_sim = OPTICAL_SIM_AVAILABLE and not args.no_optical_sim
    simulator = None
    if run_optical_sim:
        print("[INFO] Optical simulation enabled - will compute predicted colors")
        from src.optical_sim import OpticalSimulator
        simulator = OpticalSimulator(incidence_angle=0)
    else:
        if args.no_optical_sim:
            print("[INFO] Optical simulation disabled by --no-optical-sim flag")
        else:
            print("[WARN] Optical simulation not available - CIEDE2000 will not be computed")
    
    # Create a generator for reproducible sampling
    if args.sample_predictions:
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(args.seed)
    else:
        sampling_generator = None
    
    results = []
    for idx, example in enumerate(dataset):
        # Generate structure with optional sampling
        pred_materials, pred_thicknesses, stop_reason = generate_structure(
            model, 
            example.rgb, 
            device,
            sample=args.sample_predictions,
            temperature=args.temperature,
            generator=sampling_generator,
        )
        
        # Get GT RGB as both float (for CIEDE2000) and int (for storage)
        gt_sRGB_float = denormalize_rgb_float(example.rgb)  # Full precision for ΔE
        gt_sRGB_int = denormalize_rgb(example.rgb)  # Rounded ints for display/storage
        
        is_valid = len(pred_materials) > 0
        pred_sRGB = None
        ciede_value = None
        
        if run_optical_sim and is_valid:
            try:
                # pred_sRGB is returned as floats from optical_sim
                pred_sRGB = simulator.compute_color(pred_materials, pred_thicknesses)
                # Both gt_sRGB_float and pred_sRGB are now floats for accurate ΔE
                ciede_value = compute_color_difference(gt_sRGB_float, pred_sRGB)
            except Exception as e:
                print(f"[WARN] Optical simulation failed for example {idx}: {e}")
                pred_sRGB = None
                ciede_value = None
                is_valid = False
        
        results.append(EvalResult(
            idx=idx,
            gt_materials=example.target_materials,
            gt_thicknesses=example.target_thicknesses,
            gt_sRGB=gt_sRGB_int,  # Store ints for human readability
            pred_materials=pred_materials,
            pred_thicknesses=pred_thicknesses,
            pred_sRGB=pred_sRGB,  # Store floats (or None)
            stop_reason=stop_reason,
            n_layers_gt=len(example.target_materials),
            n_layers_pred=len(pred_materials),
            ciede2000=ciede_value,
            is_valid=is_valid,
        ))
        
        if (idx + 1) % 100 == 0:
            n_valid = sum(1 for r in results if r.is_valid)
            if run_optical_sim:
                ciede_vals = [r.ciede2000 for r in results if r.ciede2000 is not None]
                mean_ciede = sum(ciede_vals) / len(ciede_vals) if ciede_vals else float('nan')
                print(f"[INFO] Evaluated {idx + 1} examples, valid={n_valid}/{len(results)}, mean_ΔE={mean_ciede:.2f}")
            else:
                print(f"[INFO] Evaluated {idx + 1} examples, valid={n_valid}/{len(results)}")
    
    # Compute metrics
    n_total = len(results)
    n_eos = sum(1 for r in results if r.stop_reason == 'EOS')
    n_valid = sum(1 for r in results if r.is_valid)
    n_exact = sum(1 for r in results if r.pred_materials == r.gt_materials 
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
        'teacher_forcing_loss': tf_results['loss'],
        'teacher_forcing_accuracy': tf_results['accuracy'],
        'eos_rate': n_eos / n_total if n_total > 0 else 0,
        'valid_rate': n_valid / n_total if n_total > 0 else 0,
        'exact_match': n_exact / n_total if n_total > 0 else 0,
        'layer_count_match': n_layers_match / n_total if n_total > 0 else 0,
        'avg_layer_diff': avg_layer_diff,
        'config_tag': config.tag(),
    }
    
    # CIEDE2000 statistics
    ciede_values = [r.ciede2000 for r in results if r.ciede2000 is not None]
    if ciede_values:
        ciede_arr = np.array(ciede_values)
        metrics['ciede2000_mean'] = float(np.mean(ciede_arr))
        metrics['ciede2000_median'] = float(np.median(ciede_arr))
        metrics['ciede2000_q1'] = float(np.percentile(ciede_arr, 25))
        metrics['ciede2000_q3'] = float(np.percentile(ciede_arr, 75))
        metrics['ciede2000_min'] = float(np.min(ciede_arr))
        metrics['ciede2000_max'] = float(np.max(ciede_arr))
        metrics['ciede2000_n_computed'] = len(ciede_values)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    if args.sample_predictions:
        print(f"(Sampling mode: temperature={args.temperature}, seed={args.seed})")
    print("="*60)
    
    print(f"\n--- Teacher Forcing ---")
    print(f"  Loss:     {tf_results['loss']:.4f}")
    print(f"  Accuracy: {tf_results['accuracy']:.3f} ({100*tf_results['accuracy']:.1f}%)")
    
    print(f"\n--- Autoregressive Generation ---")
    print(f"  Total examples:     {n_total}")
    print(f"  Valid predictions:  {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  EOS rate:           {100*metrics['eos_rate']:.1f}%")
    print(f"  Exact match:        {100*metrics['exact_match']:.1f}%")
    print(f"  Layer count match:  {100*metrics['layer_count_match']:.1f}%")
    print(f"  Avg layer diff:     {avg_layer_diff:.2f}")
    
    if ciede_values:
        print(f"\n--- CIEDE2000 Color Difference (ΔE₀₀) ---")
        print(f"  Mean:   {metrics['ciede2000_mean']:.2f}")
        print(f"  Median: {metrics['ciede2000_median']:.2f}")
        print(f"  Q1:     {metrics['ciede2000_q1']:.2f}")
        print(f"  Q3:     {metrics['ciede2000_q3']:.2f}")
        print(f"  Min:    {metrics['ciede2000_min']:.2f}")
        print(f"  Max:    {metrics['ciede2000_max']:.2f}")
    
    print("="*60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        if args.sample_predictions:
            temp_str = f"_T{args.temperature}".replace('.', 'p')
            output_path = repo_root / 'pretrain_rgb_to_structure' / 'outputs' / f'eval_{config.tag()}_sampling{temp_str}.json'
        else:
            output_path = repo_root / 'pretrain_rgb_to_structure' / 'outputs' / f'eval_{config.tag()}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics, 
            'results': [asdict(r) for r in results]
        }, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")
    
    # Generate color swatch
    if not args.no_swatch and MATPLOTLIB_AVAILABLE and run_optical_sim:
        if args.sample_predictions:
            temp_str = f"_T{args.temperature}".replace('.', 'p')
            swatch_path = output_path.parent / f'swatch_{config.tag()}_sampling{temp_str}.png'
        else:
            swatch_path = output_path.parent / f'swatch_{config.tag()}.png'
        create_color_swatch(results, str(swatch_path), n=args.swatch_examples)


if __name__ == "__main__":
    main()