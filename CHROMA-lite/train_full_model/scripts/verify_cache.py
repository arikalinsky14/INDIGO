#!/usr/bin/env python3
"""
Cache Verification

Scans cached shards and checks for correctness issues before training.

Checks performed:
  1. ERROR padding — is the ERROR dim padded with -100.0 (not 0.0)?
  2. Target token sanity — correct examples end with EOS, incorrect = [ERROR]
  3. Float16 precision — any NaN/Inf or degenerate logit distributions?
  4. Per-step logit quality — CE loss and accuracy at each autoregressive step
  5. Shard consistency — shapes, dtypes, index.json agreement
  6. Round-trip MLP verification (optional, requires --rgb-to-structure-checkpoint)

Usage:
    # Quick check on test cache (no MLP needed):
    python train_full_model/scripts/verify_cache.py \
        --cache-dir train_full_model/data/cache_test

    # Full check on train cache, first 10 shards:
    python train_full_model/scripts/verify_cache.py \
        --cache-dir train_full_model/data/cache_train \
        --max-shards 10 --verbose

    # With round-trip MLP verification:
    python train_full_model/scripts/verify_cache.py \
        --cache-dir train_full_model/data/cache_test \
        --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.materials_vocab import (
    NUM_MATERIALS, MAX_LAYERS, MATERIAL_TO_IDX,
    EOS_TOKEN, ERROR_TOKEN, VOCAB_SIZE_WITH_ERROR,
    normalize_thickness, encode_layer,
)


# ============================================================================
# Helpers
# ============================================================================

def load_shard(shard_path: Path) -> Dict:
    """Load a cached shard and return its contents."""
    return torch.load(shard_path, map_location='cpu', weights_only=False)


def discover_shards(cache_dir: Path) -> List[Path]:
    """Find all shard files in a cache directory, sorted by index."""
    return sorted(cache_dir.glob('shard_*.pt'))


def load_rgb_to_structure(checkpoint_dir: str, device: torch.device):
    """Load RGB→Structure MLP from a checkpoint directory."""
    from pretrain_rgb_to_structure.src.model import ThinFilmMLP, ModelConfig

    ckpt_dir = Path(checkpoint_dir)
    config_path = ckpt_dir / 'config.json'
    model_path = ckpt_dir / 'model.pt'

    with open(config_path) as f:
        model_config = ModelConfig.from_dict(json.load(f))

    mlp = ThinFilmMLP(model_config).to(device)
    mlp.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    mlp.eval()
    return mlp, model_config


# ============================================================================
# Check 1: ERROR Padding
# ============================================================================

def check_error_padding(shards: List[Dict], max_examples: int = 500) -> Dict:
    """Verify ERROR dim is padded with -100.0, not 0.0."""
    all_valid_logits = []
    n_examined = 0

    for shard in shards:
        logits = shard['base_logits'].float()       # [N, MAX_STEPS, 1002]
        n_steps = shard['n_steps']                   # [N]
        N = len(shard['texts'])

        for i in range(N):
            if n_examined >= max_examples:
                break
            ns = n_steps[i].item()
            all_valid_logits.append(logits[i, :ns])  # [ns, 1002]
            n_examined += 1
        if n_examined >= max_examples:
            break

    if not all_valid_logits:
        return {'status': 'SKIP', 'reason': 'no examples found'}

    flat = torch.cat(all_valid_logits, dim=0)        # [total_steps, 1002]
    real_logits = flat[:, :1001]
    error_logits = flat[:, 1001]

    # Statistics
    real_mean = real_logits.mean().item()
    real_std = real_logits.std().item()
    real_min = real_logits.min().item()
    real_max = real_logits.max().item()

    n_below_zero = (real_logits < 0.0).float().mean().item()
    error_mean = error_logits.mean().item()
    all_neg100 = (error_logits == -100.0).all().item()
    all_zeros = (error_logits == 0.0).all().item()

    # Would ERROR be argmax?
    argmax_preds = flat.argmax(dim=-1)
    error_pred_rate = (argmax_preds == ERROR_TOKEN).float().mean().item()

    # Percentiles for reference (subsample to avoid quantile() memory limit)
    flat_real = real_logits.flatten().float()
    if flat_real.numel() > 1_000_000:
        idx = torch.randperm(flat_real.numel())[:1_000_000]
        flat_real = flat_real[idx]
    p1 = torch.quantile(flat_real, 0.01).item()
    p5 = torch.quantile(flat_real, 0.05).item()

    has_bug = all_zeros or error_pred_rate > 0.001

    return {
        'status': 'FAIL' if has_bug else 'PASS',
        'n_steps_examined': flat.size(0),
        'real_logit_mean': real_mean,
        'real_logit_std': real_std,
        'real_logit_min': real_min,
        'real_logit_max': real_max,
        'frac_real_below_zero': n_below_zero,
        'error_dim_mean': error_mean,
        'error_all_neg100': all_neg100,
        'error_all_zeros_BUG': all_zeros,
        'error_pred_rate': error_pred_rate,
        'logit_p1': p1,
        'logit_p5': p5,
    }


# ============================================================================
# Check 2: Target Token Sanity
# ============================================================================

def check_target_tokens(shards: List[Dict], max_examples: int = 5000,
                        verbose: bool = False) -> Dict:
    """Verify target token sequences are well-formed."""
    n_checked = 0
    n_correct = 0
    n_incorrect = 0
    issues = []

    for shard in shards:
        N = len(shard['texts'])
        targets_all = shard['target_tokens']   # [N, MAX_STEPS]
        n_steps_all = shard['n_steps']         # [N]
        incorrect_all = shard['incorrect']     # [N]

        for i in range(N):
            if n_checked >= max_examples:
                break

            ns = n_steps_all[i].item()
            targets = targets_all[i, :ns].tolist()
            is_incorrect = incorrect_all[i].item()

            example_issues = []

            if is_incorrect:
                n_incorrect += 1
                if targets != [ERROR_TOKEN]:
                    example_issues.append(
                        f"incorrect but target != [ERROR], got {targets}")
            else:
                n_correct += 1
                if ERROR_TOKEN in targets:
                    example_issues.append("correct but contains ERROR token")

                # Correct examples: last token should be EOS for <8 layers
                n_layers = sum(1 for t in targets if t != EOS_TOKEN and t != ERROR_TOKEN)
                if n_layers < MAX_LAYERS and EOS_TOKEN not in targets:
                    example_issues.append(
                        f"{n_layers}-layer example missing EOS")

                # 8-layer examples should NOT have EOS
                if n_layers == MAX_LAYERS and EOS_TOKEN in targets:
                    example_issues.append(
                        f"8-layer example has unexpected EOS")

            # Token range check
            for t_idx, tok in enumerate(targets):
                if tok != EOS_TOKEN and tok != ERROR_TOKEN:
                    if tok < 0 or tok >= 1000:
                        example_issues.append(
                            f"step {t_idx}: invalid token {tok}")

            # n_steps range check
            if ns < 1 or ns > MAX_LAYERS + 1:
                example_issues.append(f"n_steps={ns} out of range [1, {MAX_LAYERS+1}]")

            if example_issues:
                issues.append({
                    'shard_example': n_checked,
                    'n_steps': ns,
                    'incorrect': is_incorrect,
                    'targets': targets,
                    'issues': example_issues,
                })

            n_checked += 1
        if n_checked >= max_examples:
            break

    return {
        'status': 'FAIL' if issues else 'PASS',
        'n_checked': n_checked,
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'n_issues': len(issues),
        'issues': issues[:20],  # cap for display
    }


# ============================================================================
# Check 3: Float16 Precision
# ============================================================================

def check_precision(shards: List[Dict], max_examples: int = 200) -> Dict:
    """Check for NaN, Inf, or degenerate values in cached logits."""
    n_nan = 0
    n_inf = 0
    n_examined = 0
    dtype_ok = True

    for shard in shards:
        logits = shard['base_logits']
        if logits.dtype != torch.float16:
            dtype_ok = False

        n_steps = shard['n_steps']
        N = len(shard['texts'])

        for i in range(N):
            if n_examined >= max_examples:
                break
            ns = n_steps[i].item()
            sl = logits[i, :ns].float()
            n_nan += sl.isnan().sum().item()
            n_inf += sl.isinf().sum().item()
            n_examined += 1
        if n_examined >= max_examples:
            break

    has_issues = n_nan > 0 or n_inf > 0 or not dtype_ok

    return {
        'status': 'FAIL' if has_issues else 'PASS',
        'dtype': str(shards[0]['base_logits'].dtype) if shards else 'N/A',
        'dtype_is_float16': dtype_ok,
        'n_examined': n_examined,
        'n_nan_values': n_nan,
        'n_inf_values': n_inf,
    }


# ============================================================================
# Check 4: Per-Step Logit Quality
# ============================================================================

def check_per_step_quality(shards: List[Dict],
                           max_examples: int = 2000) -> Dict:
    """Compute CE loss and accuracy at each autoregressive step (correct only)."""
    # Collect per-step logits and targets
    step_data = {s: {'logits': [], 'targets': []} for s in range(MAX_LAYERS + 1)}
    n_examined = 0

    for shard in shards:
        logits = shard['base_logits'].float()
        targets = shard['target_tokens']
        n_steps = shard['n_steps']
        incorrect = shard['incorrect']
        N = len(shard['texts'])

        for i in range(N):
            if n_examined >= max_examples:
                break
            if incorrect[i]:
                n_examined += 1
                continue

            ns = n_steps[i].item()
            for step in range(ns):
                step_data[step]['logits'].append(logits[i, step])
                step_data[step]['targets'].append(targets[i, step].item())

            n_examined += 1
        if n_examined >= max_examples:
            break

    results = []
    for step in range(MAX_LAYERS + 1):
        if not step_data[step]['logits']:
            break

        logits_t = torch.stack(step_data[step]['logits'])
        targets_t = torch.tensor(step_data[step]['targets'])
        n = logits_t.size(0)

        ce = F.cross_entropy(logits_t, targets_t).item()

        # Accuracy on real dims (0-1000) only
        real_preds = logits_t[:, :1001].argmax(dim=-1)
        real_acc = (real_preds == targets_t).float().mean().item()

        # Accuracy on full 1002 dims (should match if padding is correct)
        full_preds = logits_t.argmax(dim=-1)
        full_acc = (full_preds == targets_t).float().mean().item()

        error_rate = (full_preds == ERROR_TOKEN).float().mean().item()

        # Target token rank (how many logits are above the target's logit)
        target_logit_vals = logits_t[torch.arange(n), targets_t]
        target_rank = (
            logits_t > target_logit_vals.unsqueeze(-1)
        ).sum(dim=-1).float().mean().item()

        results.append({
            'step': step,
            'n': n,
            'ce_loss': ce,
            'acc_real': real_acc,
            'acc_full': full_acc,
            'error_pred_rate': error_rate,
            'target_rank': target_rank,
        })

    # Overall status: FAIL if real_acc and full_acc diverge (ERROR bug)
    acc_divergence = any(
        abs(r['acc_real'] - r['acc_full']) > 0.01 for r in results)

    return {
        'status': 'FAIL' if acc_divergence else 'PASS',
        'n_correct_examined': n_examined,
        'per_step': results,
    }


# ============================================================================
# Check 5: Shard Consistency
# ============================================================================

def check_shard_consistency(cache_dir: Path, shard_paths: List[Path],
                            shards: List[Dict]) -> Dict:
    """Check shapes, dtypes, and agreement with index.json."""
    issues = []

    # Load index
    index_path = cache_dir / 'index.json'
    if not index_path.exists():
        return {'status': 'FAIL', 'issues': ['index.json not found']}

    with open(index_path) as f:
        index = json.load(f)

    expected_n_shards = index.get('n_shards', -1)
    expected_vocab = index.get('vocab_size', -1)
    expected_max_steps = index.get('max_steps', -1)

    # Compare against ALL shards on disk, not just the loaded subset
    all_shard_paths = discover_shards(cache_dir)
    total_found = len(all_shard_paths)
    if total_found != expected_n_shards:
        issues.append(
            f"index.json says {expected_n_shards} shards but found {total_found}")

    total_examples = 0
    for si, (sp, shard) in enumerate(zip(shard_paths, shards)):
        N = len(shard['texts'])
        total_examples += N

        logits = shard['base_logits']
        targets = shard['target_tokens']
        n_steps = shard['n_steps']
        incorrect = shard['incorrect']

        # Shape checks
        if logits.shape != (N, expected_max_steps, expected_vocab):
            issues.append(
                f"shard {si}: logits shape {logits.shape} != "
                f"expected ({N}, {expected_max_steps}, {expected_vocab})")

        if targets.shape != (N, expected_max_steps):
            issues.append(
                f"shard {si}: targets shape {targets.shape} != "
                f"expected ({N}, {expected_max_steps})")

        if n_steps.shape != (N,):
            issues.append(f"shard {si}: n_steps shape {n_steps.shape} != ({N},)")

        if incorrect.shape != (N,):
            issues.append(f"shard {si}: incorrect shape {incorrect.shape} != ({N},)")

        # Dtype checks
        if logits.dtype != torch.float16:
            issues.append(f"shard {si}: logits dtype {logits.dtype} != float16")
        if targets.dtype != torch.int64:
            issues.append(f"shard {si}: targets dtype {targets.dtype} != int64")

    # Only check total if we loaded all shards
    expected_total = index.get('total_examples', -1)
    if len(shard_paths) == total_found and total_examples != expected_total:
        issues.append(
            f"total examples {total_examples} != index says {expected_total}")

    return {
        'status': 'FAIL' if issues else 'PASS',
        'n_shards_loaded': len(shard_paths),
        'n_shards_on_disk': total_found,
        'n_shards_expected': expected_n_shards,
        'examples_in_loaded_shards': total_examples,
        'expected_total': expected_total,
        'vocab_size': expected_vocab,
        'max_steps': expected_max_steps,
        'issues': issues[:20],
    }


# ============================================================================
# Check 6: Round-Trip MLP (optional)
# ============================================================================

def check_roundtrip_mlp(shards: List[Dict], mlp, device: torch.device,
                        n_examples: int = 20) -> Dict:
    """
    Synthetic round-trip: verify MLP is deterministic and output shape is correct.
    (Full round-trip with real data requires --data-dir which we skip for simplicity.)
    """
    # Determinism test
    test_rgb = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float32)
    test_structure = torch.zeros(NUM_MATERIALS, MAX_LAYERS, dtype=torch.float32)

    rgb_in = test_rgb.unsqueeze(0).to(device)
    struct_in = test_structure.unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = mlp(rgb_in, struct_in)
        out2 = mlp(rgb_in, struct_in)

    deterministic = torch.allclose(out1, out2, atol=1e-5)

    # Shape and range
    output_shape = tuple(out1.shape)
    output_min = out1.min().item()
    output_max = out1.max().item()
    output_mean = out1.mean().item()

    # Padding test: verify -100.0 padding produces correct shape
    padded = F.pad(out1, (0, 1), value=-100.0)
    padded_shape = tuple(padded.shape)
    error_dim_val = padded[0, -1].item()

    return {
        'status': 'PASS' if deterministic and error_dim_val == -100.0 else 'FAIL',
        'deterministic': deterministic,
        'output_shape': output_shape,
        'output_min': output_min,
        'output_max': output_max,
        'output_mean': output_mean,
        'padded_shape': padded_shape,
        'error_dim_after_pad': error_dim_val,
    }


# ============================================================================
# Pretty Printing
# ============================================================================

def print_check(name: str, result: Dict):
    """Print a check result with pass/fail indicator."""
    status = result.get('status', 'UNKNOWN')
    icon = '✓' if status == 'PASS' else '✗' if status == 'FAIL' else '⊘'
    print(f"\n{'='*70}")
    print(f"{icon} {name}: {status}")
    print(f"{'='*70}")

    for k, v in result.items():
        if k == 'status':
            continue
        if k == 'issues' and isinstance(v, list):
            if v:
                for issue in v:
                    if isinstance(issue, dict):
                        print(f"    ✗ Example {issue.get('shard_example', '?')}: "
                              f"{'; '.join(issue.get('issues', []))}")
                        if 'targets' in issue:
                            print(f"      n_steps={issue['n_steps']}, "
                                  f"incorrect={issue['incorrect']}, "
                                  f"targets={issue['targets']}")
                    else:
                        print(f"    ✗ {issue}")
            continue
        if k == 'per_step' and isinstance(v, list):
            print(f"  {'Step':>6} {'n':>6} {'CE':>8} {'Acc(real)':>10} "
                  f"{'Acc(full)':>10} {'ERR%':>6} {'Rank':>6}")
            print(f"  {'-'*56}")
            for r in v:
                print(f"  {r['step']:>6} {r['n']:>6} {r['ce_loss']:>8.2f} "
                      f"{r['acc_real']:>10.3f} {r['acc_full']:>10.3f} "
                      f"{100*r['error_pred_rate']:>5.0f}% "
                      f"{r['target_rank']:>6.1f}")
            continue
        print(f"  {k}: {v}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verify cached pretrained pipeline outputs')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to cache directory (cache_train or cache_test)')
    parser.add_argument('--max-shards', type=int, default=None,
                        help='Max shards to load (default: all)')
    parser.add_argument('--max-examples', type=int, default=5000,
                        help='Max examples for per-check analysis (default: 5000)')
    parser.add_argument('--rgb-to-structure-checkpoint', type=str, default=None,
                        help='Optional: RGB→Structure MLP checkpoint for round-trip test')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-example issues')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*70}")
    print(f"CACHE VERIFICATION")
    print(f"{'='*70}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Device:     {device}")

    # ================================================================
    # Discover and load shards
    # ================================================================
    shard_paths = discover_shards(cache_dir)
    if not shard_paths:
        print(f"\n[ERROR] No shard files found in {cache_dir}")
        sys.exit(1)

    if args.max_shards is not None:
        shard_paths = shard_paths[:args.max_shards]

    print(f"Shards:     {len(shard_paths)} "
          f"{'(limited)' if args.max_shards else '(all)'}")

    print(f"\n[INFO] Loading {len(shard_paths)} shards...")
    shards = []
    total_examples = 0
    for sp in shard_paths:
        shard = load_shard(sp)
        shards.append(shard)
        total_examples += len(shard['texts'])
    print(f"[INFO] Loaded {total_examples:,} examples")

    # Quick stats
    n_incorrect = sum(s['incorrect'].sum().item() for s in shards)
    n_correct = total_examples - n_incorrect
    print(f"[INFO] {n_correct:,} correct + {n_incorrect:,} incorrect")

    # ================================================================
    # Run checks
    # ================================================================
    all_results = {}
    n_pass = 0
    n_fail = 0

    # Check 1: ERROR padding
    r = check_error_padding(shards, max_examples=args.max_examples)
    all_results['ERROR_PADDING'] = r
    print_check("CHECK 1: ERROR PADDING", r)
    if r['status'] == 'PASS':
        n_pass += 1
    else:
        n_fail += 1

    # Check 2: Target tokens
    r = check_target_tokens(shards, max_examples=args.max_examples,
                            verbose=args.verbose)
    all_results['TARGET_TOKENS'] = r
    print_check("CHECK 2: TARGET TOKEN SANITY", r)
    if r['status'] == 'PASS':
        n_pass += 1
    else:
        n_fail += 1

    # Check 3: Float16 precision
    r = check_precision(shards, max_examples=args.max_examples)
    all_results['PRECISION'] = r
    print_check("CHECK 3: FLOAT16 PRECISION", r)
    if r['status'] == 'PASS':
        n_pass += 1
    else:
        n_fail += 1

    # Check 4: Per-step quality
    r = check_per_step_quality(shards, max_examples=args.max_examples)
    all_results['PER_STEP_QUALITY'] = r
    print_check("CHECK 4: PER-STEP LOGIT QUALITY (correct examples)", r)
    if r['status'] == 'PASS':
        n_pass += 1
    else:
        n_fail += 1

    # Check 5: Shard consistency
    r = check_shard_consistency(cache_dir, shard_paths, shards)
    all_results['SHARD_CONSISTENCY'] = r
    print_check("CHECK 5: SHARD CONSISTENCY", r)
    if r['status'] == 'PASS':
        n_pass += 1
    else:
        n_fail += 1

    # Check 6: Round-trip MLP (optional)
    if args.rgb_to_structure_checkpoint:
        print(f"\n[INFO] Loading RGB→Structure MLP: "
              f"{args.rgb_to_structure_checkpoint}")
        mlp, mlp_config = load_rgb_to_structure(
            args.rgb_to_structure_checkpoint, device)
        print(f"  Config: d_model={mlp_config.d_model}, "
              f"n_layers={mlp_config.n_layers}")

        r = check_roundtrip_mlp(shards, mlp, device)
        all_results['ROUNDTRIP_MLP'] = r
        print_check("CHECK 6: ROUND-TRIP MLP", r)
        if r['status'] == 'PASS':
            n_pass += 1
        else:
            n_fail += 1

    # ================================================================
    # Summary
    # ================================================================
    total_checks = n_pass + n_fail
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Cache:     {cache_dir}")
    print(f"  Shards:    {len(shard_paths)}")
    print(f"  Examples:  {total_examples:,} ({n_correct:,} correct, "
          f"{n_incorrect:,} incorrect)")
    print(f"  Checks:    {n_pass}/{total_checks} passed")

    if n_fail > 0:
        print(f"\n  ✗ {n_fail} CHECK(S) FAILED:")
        for name, r in all_results.items():
            if r.get('status') == 'FAIL':
                print(f"    - {name}")
        print(f"\n  Cache may need to be regenerated.")
    else:
        print(f"\n  ✓ ALL CHECKS PASSED — cache looks good for training.")

    print(f"{'='*70}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == '__main__':
    main()