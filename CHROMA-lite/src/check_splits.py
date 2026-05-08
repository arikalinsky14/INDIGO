#!/usr/bin/env python3
"""
Split Consistency Checker

Verifies that the train/validation split for correct examples is identical across
all three dataset classes:
    1. ThinFilmDataset        (pretrain_rgb_to_structure)
    2. TextThinFilmDataset    (pretrain_text_to_rgb)
    3. FullModelDataset       (train_full_model)

Also checks:
    - No overlap between train and validation within any dataset
    - Train ∪ validation covers all examples
    - Incorrect examples in FullModelDataset are split independently

Usage:
    python src/check_splits.py
    python src/check_splits.py --data-dir /path/to/data_prompts --seed 42
"""

import sys
import argparse
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import (
    ThinFilmDataset, TextThinFilmDataset, FullModelDataset,
    scan_files, scan_files_with_incorrect, make_permutation,
)


def main():
    parser = argparse.ArgumentParser(description='Verify split consistency')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from src.dataset import find_repo_root
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = _repo_root

    data_dir = Path(args.data_dir) if args.data_dir else (
        repo_root / 'create_dataset' / 'data_prompts')

    print(f"Data dir: {data_dir}")
    print(f"Seed:     {args.seed}")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "✓ PASS" if condition else "✗ FAIL"
        msg = f"  {status}: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if condition:
            passed += 1
        else:
            failed += 1

    # ================================================================
    # Load all datasets
    # ================================================================

    print("[1] Loading datasets...")

    tf_train = ThinFilmDataset(data_dir, seed=args.seed, split='train', verbose=True)
    tf_val = ThinFilmDataset(data_dir, seed=args.seed, split='validation', verbose=True)

    ttf_train = TextThinFilmDataset(data_dir, seed=args.seed, split='train', verbose=True)
    ttf_val = TextThinFilmDataset(data_dir, seed=args.seed, split='validation', verbose=True)

    fm_train = FullModelDataset(data_dir, seed=args.seed, split='train', verbose=True)
    fm_val = FullModelDataset(data_dir, seed=args.seed, split='validation', verbose=True)

    # ================================================================
    # Check 1: ThinFilmDataset == TextThinFilmDataset (same N, same seed)
    # ================================================================

    print(f"\n[2] ThinFilmDataset vs TextThinFilmDataset:")

    check("Same train size",
          len(tf_train) == len(ttf_train),
          f"{len(tf_train):,} vs {len(ttf_train):,}")

    check("Same validation size",
          len(tf_val) == len(ttf_val),
          f"{len(tf_val):,} vs {len(ttf_val):,}")

    check("Train indices identical",
          torch.equal(tf_train.order, ttf_train.order))

    check("Validation indices identical",
          torch.equal(tf_val.order, ttf_val.order))

    # ================================================================
    # Check 2: No train/validation overlap within each dataset
    # ================================================================

    print(f"\n[3] No train/validation overlap:")

    tf_train_set = set(tf_train.order.tolist())
    tf_val_set = set(tf_val.order.tolist())
    check("ThinFilmDataset",
          len(tf_train_set & tf_val_set) == 0,
          f"overlap: {len(tf_train_set & tf_val_set)}")

    ttf_train_set = set(ttf_train.order.tolist())
    ttf_val_set = set(ttf_val.order.tolist())
    check("TextThinFilmDataset",
          len(ttf_train_set & ttf_val_set) == 0,
          f"overlap: {len(ttf_train_set & ttf_val_set)}")

    fm_train_set = set(fm_train.order.tolist())
    fm_val_set = set(fm_val.order.tolist())
    check("FullModelDataset",
          len(fm_train_set & fm_val_set) == 0,
          f"overlap: {len(fm_train_set & fm_val_set)}")

    # ================================================================
    # Check 3: Train ∪ validation covers all examples
    # ================================================================

    print(f"\n[4] Complete coverage (train ∪ validation = all):")

    correct_files = scan_files(data_dir)
    N_correct = sum(f.nrows for f in correct_files)

    check("ThinFilmDataset covers all correct",
          len(tf_train_set | tf_val_set) == N_correct,
          f"{len(tf_train_set | tf_val_set):,} / {N_correct:,}")

    all_files = scan_files_with_incorrect(data_dir)
    N_total = sum(f.nrows for f in all_files)
    N_incorrect = N_total - N_correct

    check("FullModelDataset covers all (correct + incorrect)",
          len(fm_train_set | fm_val_set) == N_total,
          f"{len(fm_train_set | fm_val_set):,} / {N_total:,}")

    # ================================================================
    # Check 4: FullModelDataset correct examples match pretrained splits
    # ================================================================

    print(f"\n[5] FullModelDataset correct examples match pretrained splits:")

    # Extract correct-only indices from FullModelDataset
    fm_train_correct = set(
        idx for idx in fm_train.order.tolist()
        if not fm_train.incorrect_flags[idx].item())
    fm_val_correct = set(
        idx for idx in fm_val.order.tolist()
        if not fm_val.incorrect_flags[idx].item())

    # The correct indices in FullModelDataset should map to the same
    # global positions as ThinFilmDataset (since correct files come first
    # in scan_files_with_incorrect, same order as scan_files)
    check("Correct train indices match ThinFilmDataset train",
          fm_train_correct == tf_train_set,
          f"FullModel correct train: {len(fm_train_correct):,}, "
          f"ThinFilm train: {len(tf_train_set):,}, "
          f"intersection: {len(fm_train_correct & tf_train_set):,}")

    check("Correct validation indices match ThinFilmDataset validation",
          fm_val_correct == tf_val_set,
          f"FullModel correct validation: {len(fm_val_correct):,}, "
          f"ThinFilm validation: {len(tf_val_set):,}, "
          f"intersection: {len(fm_val_correct & tf_val_set):,}")

    # ================================================================
    # Check 5: No correct validation examples leaked into FullModel train
    # ================================================================

    print(f"\n[6] No data leakage (correct validation examples not in FullModel train):")

    leaked = tf_val_set & fm_train_correct
    check("Zero pretrained-validation examples in FullModel train",
          len(leaked) == 0,
          f"leaked: {len(leaked)}")

    leaked_reverse = tf_train_set & fm_val_correct
    check("Zero pretrained-train examples in FullModel validation",
          len(leaked_reverse) == 0,
          f"leaked: {len(leaked_reverse)}")

    # ================================================================
    # Check 6: Incorrect examples are independent
    # ================================================================

    print(f"\n[7] Incorrect examples split independently:")

    fm_train_incorrect = set(
        idx for idx in fm_train.order.tolist()
        if fm_train.incorrect_flags[idx].item())
    fm_val_incorrect = set(
        idx for idx in fm_val.order.tolist()
        if fm_val.incorrect_flags[idx].item())

    check("No incorrect train/validation overlap",
          len(fm_train_incorrect & fm_val_incorrect) == 0)

    check("All incorrect examples covered",
          len(fm_train_incorrect | fm_val_incorrect) == N_incorrect,
          f"{len(fm_train_incorrect | fm_val_incorrect):,} / {N_incorrect:,}")

    expected_incorrect_train = int(0.995 * N_incorrect)
    actual_incorrect_train = len(fm_train_incorrect)
    check("Incorrect train ratio ~99.5%",
          abs(actual_incorrect_train - expected_incorrect_train) <= 1,
          f"{actual_incorrect_train:,} (expected ~{expected_incorrect_train:,})")

    # ================================================================
    # Check 7: Permutation determinism
    # ================================================================

    print(f"\n[8] Permutation determinism (same seed → same result):")

    perm1 = make_permutation(N_correct, args.seed)
    perm2 = make_permutation(N_correct, args.seed)
    check("make_permutation is deterministic",
          torch.equal(perm1, perm2))

    perm3 = make_permutation(N_correct, args.seed + 1)
    check("Different seed → different permutation",
          not torch.equal(perm1, perm3))

    # ================================================================
    # Summary
    # ================================================================

    print(f"\n{'='*60}")
    print(f"  Dataset sizes:")
    print(f"    N_correct:   {N_correct:,}")
    print(f"    N_incorrect: {N_incorrect:,}")
    print(f"    N_total:     {N_total:,}")
    print(f"    Train/validation:  99.5% / 0.5%")
    print(f"\n  Results: {passed} passed, {failed} failed")

    if failed == 0:
        print(f"  ✓ ALL CHECKS PASSED — splits are consistent across all modules")
    else:
        print(f"  ✗ {failed} CHECK(S) FAILED — investigate above")

    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
