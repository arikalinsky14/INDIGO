#!/usr/bin/env python3
"""
Evaluation Script for INDIGO (FlexMaterialMLP / FlexMaterialCrossAttn).

Modes:
- --low-compute: teacher-forcing only (CE loss + token accuracy).
- Full mode (default): teacher forcing + autoregressive generation
  + optical simulation + CIEDE2000 metrics + color swatch.
- --sample-predictions: stochastic sampling instead of greedy argmax.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.color_utils import lab_to_srgb_int
from src.dataset import FlexThinFilmDataset, TrainingExample, find_repo_root
from src.material_features import featurize_pool, pad_pool_features
from src.materials_vocab import (
    EOS_TOKEN,
    M_MAX,
    MAX_LAYERS,
    build_structure_matrix,
    denormalize_lab,
    encode_slot,
)
from src.model import ModelConfig, build_model, compute_loss, generate_structure


# Optional: optical simulator for autoregressive eval.
OPTICAL_SIM_AVAILABLE = False
try:
    from src.optical_sim import OpticalSimulator, is_available as optical_is_available
    OPTICAL_SIM_AVAILABLE = optical_is_available()
    if not OPTICAL_SIM_AVAILABLE:
        from src.optical_sim import get_import_error
        print(f"[WARN] Optical simulation not available: {get_import_error()}")
except ImportError as e:
    print(f"[WARN] Could not import optical_sim module: {e}")

# Optional: matplotlib for color swatches.
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[WARN] matplotlib not available - color swatch will be skipped")


# ============================================================================
# Collate (mirror training.py)
# ============================================================================


def collate_fn(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """Same expansion as training.py:collate_fn (two-head target layout)."""
    all_lab, all_pool_feats, all_pool_masks, all_pool_sizes = [], [], [], []
    all_structures, all_slot_targets, all_thick_targets = [], [], []

    for ex in examples:
        pool_feats_unpadded = featurize_pool(ex.pool, mode="raw_spectrum")
        pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
        pool_size = len(ex.pool)
        n_layers = len(ex.target_slots)
        max_step = n_layers if n_layers < MAX_LAYERS else MAX_LAYERS - 1

        for step in range(max_step + 1):
            all_lab.append(ex.lab)
            all_pool_feats.append(pool_feats)
            all_pool_masks.append(pool_mask)
            all_pool_sizes.append(pool_size)
            if step == 0:
                all_structures.append(torch.zeros(M_MAX, MAX_LAYERS))
            else:
                all_structures.append(build_structure_matrix(
                    ex.target_slots[:step],
                    ex.target_thicknesses[:step],
                ))
            if step < n_layers:
                all_slot_targets.append(encode_slot(ex.target_slots[step]))
                all_thick_targets.append(float(ex.target_thicknesses[step]))
            else:
                all_slot_targets.append(EOS_TOKEN)
                all_thick_targets.append(0.0)

    return {
        "lab": torch.stack(all_lab),
        "pool_features": torch.stack(all_pool_feats),
        "pool_mask": torch.stack(all_pool_masks),
        "pool_size": torch.tensor(all_pool_sizes, dtype=torch.long),
        "structure_matrix": torch.stack(all_structures),
        "slot_target": torch.tensor(all_slot_targets, dtype=torch.long),
        "thickness_target": torch.tensor(all_thick_targets, dtype=torch.float32),
    }


def evaluate_teacher_forcing(model, dataset, device, batch_size=32,
                              num_workers=4, prefetch_factor=1):
    model.eval()
    loader_kw = {}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kw,
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print("[INFO] Running teacher forcing evaluation...")

    total_thick_mae_nm = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            losses = compute_loss(model, batch_on_device)
            count = batch_on_device["lab"].size(0)
            total_loss += losses["loss"].item() * count
            total_correct += int(losses["accuracy"].item() * count)
            total_thick_mae_nm += float(
                losses.get("thickness_mae_nm", torch.tensor(0.0)).item()
            ) * count
            total_samples += count
            if (batch_idx + 1) % 100 == 0:
                running_loss = total_loss / total_samples
                running_slot_acc = total_correct / total_samples
                running_mae = total_thick_mae_nm / total_samples
                print(
                    f"  Batch {batch_idx + 1}: loss={running_loss:.4f}, "
                    f"slot_acc={running_slot_acc:.3f}, "
                    f"thick_mae_nm={running_mae:.2f}"
                )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "slot_accuracy": total_correct / max(total_samples, 1),
        "thickness_mae_nm": total_thick_mae_nm / max(total_samples, 1),
        "n_samples": total_samples,
    }


# ============================================================================
# Color utilities (CIEDE2000)
# ============================================================================


def ciede2000(lab1, lab2) -> float:
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
        dh_prime -= 360 if dh_prime > 180 else -360
    dH_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2))
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    h_bar_prime = (h1_prime + h2_prime) / 2
    if C1_prime * C2_prime != 0 and abs(h1_prime - h2_prime) > 180:
        h_bar_prime += 180 if h1_prime + h2_prime < 360 else -180
    T = (1 - 0.17 * math.cos(math.radians(h_bar_prime - 30))
         + 0.24 * math.cos(math.radians(2 * h_bar_prime))
         + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6))
         - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63)))
    dTheta = 30 * math.exp(-((h_bar_prime - 275) / 25) ** 2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / math.sqrt(20 + (L_bar_prime - 50) ** 2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -math.sin(math.radians(2 * dTheta)) * R_C
    return math.sqrt(
        (dL_prime / S_L) ** 2 + (dC_prime / S_C) ** 2 + (dH_prime / S_H) ** 2
        + R_T * (dC_prime / S_C) * (dH_prime / S_H)
    )


def lab_diff_ciede2000(lab1, lab2) -> float:
    """ΔE_00 between two Lab colors. Targets and predictions are already
    Lab in the new pipeline, so no sRGB conversion is needed."""
    return ciede2000(lab1, lab2)


# ============================================================================
# Result records
# ============================================================================


@dataclass
class EvalResult:
    idx: int
    gt_pool_names: List[str]
    gt_slots: List[int]
    gt_materials: List[str]
    gt_thicknesses: List[float]
    gt_lab: List[float]
    pred_slots: List[int]
    pred_materials: List[str]
    pred_thicknesses: List[float]
    pred_lab: Optional[List[float]]
    stop_reason: str
    n_layers_gt: int
    n_layers_pred: int
    ciede2000: Optional[float]
    is_valid: bool


def create_color_swatch(results: List[EvalResult], output_path: str, n: int = 10) -> None:
    """Save side-by-side swatches of GT vs predicted colors.

    Both sides are Lab in the training pipeline; we convert through
    `lab_to_srgb_int` only for display. Out-of-sRGB-gamut Lab values
    get clipped to the nearest in-gamut sRGB.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available - skipping color swatch")
        return
    valid_results = [r for r in results if r.pred_lab is not None][:n]
    if not valid_results:
        print("[WARN] No valid results for color swatch")
        return
    fig, axes = plt.subplots(len(valid_results), 3, figsize=(9, 2 * len(valid_results)))
    if len(valid_results) == 1:
        axes = [axes]
    for i, result in enumerate(valid_results):
        gt_srgb = lab_to_srgb_int(result.gt_lab)
        pred_srgb = lab_to_srgb_int(result.pred_lab)

        gt_color = [c / 255 for c in gt_srgb]
        axes[i][0].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=gt_color))
        axes[i][0].set_xlim(0, 1); axes[i][0].set_ylim(0, 1); axes[i][0].axis("off")
        gt_lab_str = f"L*={result.gt_lab[0]:.0f} a*={result.gt_lab[1]:.0f} b*={result.gt_lab[2]:.0f}"
        axes[i][0].set_title(f"GT (sRGB display)\n{gt_lab_str}", fontsize=9)

        pred_color = [c / 255 for c in pred_srgb]
        axes[i][1].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=pred_color))
        axes[i][1].set_xlim(0, 1); axes[i][1].set_ylim(0, 1); axes[i][1].axis("off")
        pred_lab_str = f"L*={result.pred_lab[0]:.0f} a*={result.pred_lab[1]:.0f} b*={result.pred_lab[2]:.0f}"
        axes[i][1].set_title(f"Pred (sRGB display)\n{pred_lab_str}", fontsize=9)

        axes[i][2].axis("off")
        info_text = f"ΔE₀₀: {result.ciede2000:.2f}\nLayers: {result.n_layers_gt} → {result.n_layers_pred}"
        axes[i][2].text(0.5, 0.5, info_text, ha="center", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Color swatch saved to {output_path}")


# ============================================================================
# Checkpoint loading
# ============================================================================


def load_model(checkpoint_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, ModelConfig]:
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
    with open(config_path) as f:
        config = ModelConfig.from_dict(json.load(f))
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate INDIGO flex-material model")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream dataset shard-by-shard (recommended at "
                             "production scale; the legacy mode OOMs).")

    # Model hyperparameters (used for checkpoint lookup if --checkpoint not given)
    parser.add_argument("--feature-mode", type=str, default="raw_spectrum",
                        choices=["raw_spectrum", "compact"])
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-mode", type=str, default="mlp",
                        choices=["mlp", "cross_attn"],
                        help="Architecture variant (must match the checkpoint).")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--slot-encoder-layers", type=int, default=0,
                        help="Slot encoder depth (cross_attn). 0 = use n-layers.")
    parser.add_argument("--decoder-layers", type=int, default=1,
                        help="Decoder depth (cross_attn).")
    parser.add_argument("--lr", type=float, default=4.42e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)

    # Evaluation settings
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--swatch-examples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="DataLoader prefetch_factor (default: 1; matches "
                             "training.py — pipeline is producer-bound)")

    # Mode flags
    parser.add_argument("--low-compute", action="store_true")
    parser.add_argument("--no-optical-sim", action="store_true")
    parser.add_argument("--no-swatch", action="store_true")

    # Sampling options
    parser.add_argument("--sample-predictions", action="store_true",
                        help="Use stochastic sampling instead of greedy argmax")
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent

    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    else:
        cfg_for_lookup = ModelConfig(
            feature_mode=args.feature_mode,
            encoder_hidden=args.encoder_hidden,
            encoder_out=args.encoder_out,
            encoder_dropout=args.encoder_dropout,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            head_mode=args.head_mode,
            n_heads=args.n_heads,
            slot_encoder_layers=args.slot_encoder_layers,
            decoder_layers=args.decoder_layers,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            limit_examples=args.limit_examples,
        )
        checkpoint_dir = repo_root / "data" / "checkpoints" / cfg_for_lookup.tag() / "latest"

    print(f"[INFO] Loading model from {checkpoint_dir}")
    model, config = load_model(checkpoint_dir, device)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Config tag: {config.tag()}")

    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "train"
    dataset = FlexThinFilmDataset(
        data_dir, seed=args.seed, split=args.split,
        limit_examples=args.limit_examples, streaming=args.streaming
    )
    print(f"[INFO] Evaluating {len(dataset)} examples from split '{args.split}'")

    # ----- Low-compute -----
    if args.low_compute:
        print("\n" + "=" * 60)
        print("LOW-COMPUTE MODE: Teacher Forcing Evaluation")
        print("=" * 60)
        tf_results = evaluate_teacher_forcing(
            model, dataset, device, batch_size=args.batch_size,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        metrics = {
            "mode": "low_compute",
            "teacher_forcing_loss": tf_results["loss"],
            "teacher_forcing_accuracy": tf_results["accuracy"],
            "n_samples": tf_results["n_samples"],
            "n_examples": len(dataset),
            "config_tag": config.tag(),
        }
        print(f"\n--- Teacher Forcing Results ---")
        print(f"  Loss:     {tf_results['loss']:.4f}")
        print(f"  Accuracy: {tf_results['accuracy']:.3f} ({100 * tf_results['accuracy']:.1f}%)")
        print(f"  Samples:  {tf_results['n_samples']:,}")
        print("=" * 60)

        output_path = (Path(args.output) if args.output
                       else repo_root / "outputs" / f"eval_{config.tag()}_lowcompute.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics}, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")
        return

    # ----- Full mode -----
    print("\n" + "=" * 60)
    if args.sample_predictions:
        print(f"FULL MODE: Teacher Forcing + Autoregressive (SAMPLING, T={args.temperature})")
    else:
        print("FULL MODE: Teacher Forcing + Autoregressive Generation")
    print("=" * 60)

    print("\n--- Phase 1: Teacher Forcing ---")
    tf_results = evaluate_teacher_forcing(
        model, dataset, device, batch_size=args.batch_size,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
    )

    print("\n--- Phase 2: Autoregressive Generation ---")
    if args.sample_predictions:
        print(f"[INFO] Using stochastic sampling with temperature={args.temperature}")

    run_optical_sim = OPTICAL_SIM_AVAILABLE and not args.no_optical_sim
    simulator = None
    if run_optical_sim:
        print("[INFO] Optical simulation enabled - will compute predicted colors")
        simulator = OpticalSimulator(incidence_angle=0)
    else:
        if args.no_optical_sim:
            print("[INFO] Optical simulation disabled by --no-optical-sim flag")
        else:
            print("[WARN] Optical simulation not available - CIEDE2000 will not be computed")

    sampling_generator = None
    if args.sample_predictions:
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(args.seed)

    results: List[EvalResult] = []
    for idx, example in enumerate(dataset):
        pred_slots, pred_thicknesses, stop_reason = generate_structure(
            model,
            example.lab,
            example.pool,
            device,
            sample=args.sample_predictions,
            temperature=args.temperature,
            generator=sampling_generator,
        )
        pred_materials = [example.pool[s].name for s in pred_slots]
        gt_materials = [example.pool[s].name for s in example.target_slots]

        gt_lab = denormalize_lab(example.lab)

        is_valid = len(pred_slots) > 0
        pred_lab = None
        ciede_value = None
        if run_optical_sim and is_valid:
            try:
                pred_lab = simulator.compute_lab(
                    pool=example.pool,
                    slot_indices=pred_slots,
                    thicknesses_nm=pred_thicknesses,
                )
                ciede_value = lab_diff_ciede2000(gt_lab, pred_lab)
            except Exception as e:
                print(f"[WARN] Optical simulation failed for example {idx}: {e}")
                pred_lab = None
                ciede_value = None
                is_valid = False

        results.append(EvalResult(
            idx=idx,
            gt_pool_names=[m.name for m in example.pool],
            gt_slots=list(example.target_slots),
            gt_materials=gt_materials,
            gt_thicknesses=list(example.target_thicknesses),
            gt_lab=gt_lab,
            pred_slots=list(pred_slots),
            pred_materials=pred_materials,
            pred_thicknesses=list(pred_thicknesses),
            pred_lab=pred_lab,
            stop_reason=stop_reason,
            n_layers_gt=len(example.target_slots),
            n_layers_pred=len(pred_slots),
            ciede2000=ciede_value,
            is_valid=is_valid,
        ))

        if (idx + 1) % 100 == 0:
            n_valid = sum(1 for r in results if r.is_valid)
            if run_optical_sim:
                ciede_vals = [r.ciede2000 for r in results if r.ciede2000 is not None]
                mean_ciede = sum(ciede_vals) / len(ciede_vals) if ciede_vals else float("nan")
                print(f"[INFO] Evaluated {idx + 1} examples, valid={n_valid}/{len(results)}, mean_ΔE={mean_ciede:.2f}")
            else:
                print(f"[INFO] Evaluated {idx + 1} examples, valid={n_valid}/{len(results)}")

    n_total = len(results)
    n_eos = sum(1 for r in results if r.stop_reason == "EOS")
    n_valid = sum(1 for r in results if r.is_valid)
    n_exact = sum(
        1 for r in results
        if r.pred_materials == r.gt_materials and r.pred_thicknesses == r.gt_thicknesses
    )
    n_layers_match = sum(1 for r in results if r.n_layers_pred == r.n_layers_gt)
    layer_diffs = [abs(r.n_layers_pred - r.n_layers_gt) for r in results]
    avg_layer_diff = sum(layer_diffs) / max(n_total, 1)

    metrics = {
        "mode": "full_sampling" if args.sample_predictions else "full",
        "sample_predictions": args.sample_predictions,
        "temperature": args.temperature if args.sample_predictions else None,
        "seed": args.seed,
        "n_examples": n_total,
        "teacher_forcing_loss": tf_results["loss"],
        "teacher_forcing_accuracy": tf_results["accuracy"],
        "eos_rate": n_eos / n_total if n_total else 0,
        "valid_rate": n_valid / n_total if n_total else 0,
        "exact_match": n_exact / n_total if n_total else 0,
        "layer_count_match": n_layers_match / n_total if n_total else 0,
        "avg_layer_diff": avg_layer_diff,
        "config_tag": config.tag(),
    }

    ciede_values = [r.ciede2000 for r in results if r.ciede2000 is not None]
    if ciede_values:
        ciede_arr = np.array(ciede_values)
        metrics["ciede2000_mean"] = float(np.mean(ciede_arr))
        metrics["ciede2000_median"] = float(np.median(ciede_arr))
        metrics["ciede2000_q1"] = float(np.percentile(ciede_arr, 25))
        metrics["ciede2000_q3"] = float(np.percentile(ciede_arr, 75))
        metrics["ciede2000_min"] = float(np.min(ciede_arr))
        metrics["ciede2000_max"] = float(np.max(ciede_arr))
        metrics["ciede2000_n_computed"] = len(ciede_values)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    if args.sample_predictions:
        print(f"(Sampling mode: temperature={args.temperature}, seed={args.seed})")
    print("=" * 60)
    print(f"\n--- Teacher Forcing ---")
    print(f"  Loss:     {tf_results['loss']:.4f}")
    print(f"  Accuracy: {tf_results['accuracy']:.3f} ({100 * tf_results['accuracy']:.1f}%)")
    print(f"\n--- Autoregressive Generation ---")
    print(f"  Total examples:     {n_total}")
    print(f"  Valid predictions:  {n_valid} ({100 * n_valid / max(n_total, 1):.1f}%)")
    print(f"  EOS rate:           {100 * metrics['eos_rate']:.1f}%")
    print(f"  Exact match:        {100 * metrics['exact_match']:.1f}%")
    print(f"  Layer count match:  {100 * metrics['layer_count_match']:.1f}%")
    print(f"  Avg layer diff:     {avg_layer_diff:.2f}")
    if ciede_values:
        print(f"\n--- CIEDE2000 Color Difference (ΔE₀₀) ---")
        print(f"  Mean:   {metrics['ciede2000_mean']:.2f}")
        print(f"  Median: {metrics['ciede2000_median']:.2f}")
        print(f"  Q1:     {metrics['ciede2000_q1']:.2f}")
        print(f"  Q3:     {metrics['ciede2000_q3']:.2f}")
        print(f"  Min:    {metrics['ciede2000_min']:.2f}")
        print(f"  Max:    {metrics['ciede2000_max']:.2f}")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
    else:
        if args.sample_predictions:
            temp_str = f"_T{args.temperature}".replace(".", "p")
            output_path = repo_root / "outputs" / f"eval_{config.tag()}_sampling{temp_str}.json"
        else:
            output_path = repo_root / "outputs" / f"eval_{config.tag()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    if not args.no_swatch and MATPLOTLIB_AVAILABLE and run_optical_sim:
        if args.sample_predictions:
            temp_str = f"_T{args.temperature}".replace(".", "p")
            swatch_path = output_path.parent / f"swatch_{config.tag()}_sampling{temp_str}.png"
        else:
            swatch_path = output_path.parent / f"swatch_{config.tag()}.png"
        create_color_swatch(results, str(swatch_path), n=args.swatch_examples)


if __name__ == "__main__":
    main()
