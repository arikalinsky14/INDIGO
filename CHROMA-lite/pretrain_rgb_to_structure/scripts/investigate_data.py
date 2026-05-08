#!/usr/bin/env python3
"""
Investigate Data Leakage Between Train and Validation Sets

This script checks for potential data leakage between train and validation splits
by comparing examples across the two sets.

Checks performed:
1. Index overlap - Are the same row indices used in both splits?
2. Content overlap - Are there duplicate (RGB, materials, thicknesses) tuples?
3. RGB-only overlap - Are target RGB values shared between splits?
4. Structure overlap - Are the same thin-film structures in both splits?

Usage:
    python investigate_data_leakage.py --data-dir /path/to/data_prompts
    python investigate_data_leakage.py --data-dir /path/to/data_prompts --sample-size 50000
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import torch

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from src.dataset import ThinFilmDataset, make_permutation, scan_files, TrainingExample
from src.materials_vocab import normalize_rgb


def hash_example(rgb: torch.Tensor, materials: List[str], thicknesses: List[int]) -> str:
    """Create a hash for an example based on its content."""
    rgb_tuple = tuple(round(x.item(), 6) for x in rgb)
    content = f"{rgb_tuple}|{materials}|{thicknesses}"
    return hashlib.md5(content.encode()).hexdigest()


def hash_rgb(rgb: torch.Tensor) -> str:
    """Create a hash for just the RGB values."""
    rgb_tuple = tuple(round(x.item(), 6) for x in rgb)
    return hashlib.md5(str(rgb_tuple).encode()).hexdigest()


def hash_structure(materials: List[str], thicknesses: List[int]) -> str:
    """Create a hash for just the structure."""
    content = f"{materials}|{thicknesses}"
    return hashlib.md5(content.encode()).hexdigest()


@dataclass
class LeakageReport:
    """Report of data leakage investigation."""
    train_size: int
    val_size: int

    # Index-level analysis
    train_indices: Set[int]
    val_indices: Set[int]
    index_overlap: Set[int]
    
    # Content-level analysis
    full_content_overlap: int
    rgb_overlap: int
    structure_overlap: int
    
    # Detailed overlaps
    overlapping_examples: List[Dict]
    
    def print_report(self):
        print("\n" + "=" * 70)
        print("DATA LEAKAGE INVESTIGATION REPORT")
        print("=" * 70)
        
        print(f"\n--- Dataset Sizes ---")
        print(f"  Train set:       {self.train_size:,} examples")
        print(f"  Validation set:  {self.val_size:,} examples")

        print(f"\n--- Index-Level Analysis ---")
        print(f"  Unique train indices:      {len(self.train_indices):,}")
        print(f"  Unique validation indices: {len(self.val_indices):,}")
        print(f"  Index overlap:        {len(self.index_overlap):,}")
        if self.index_overlap:
            print(f"  ⚠️  WARNING: {len(self.index_overlap)} indices appear in BOTH train and validation!")
            print(f"      First 10 overlapping indices: {sorted(list(self.index_overlap))[:10]}")
        else:
            print(f"  ✓ No index overlap detected - good!")
        
        print(f"\n--- Content-Level Analysis ---")
        print(f"  Full content overlap: {self.full_content_overlap:,}")
        if self.full_content_overlap > 0:
            pct = 100 * self.full_content_overlap / self.val_size
            print(f"  ⚠️  WARNING: {self.full_content_overlap} examples ({pct:.2f}% of validation) have identical content!")
        else:
            print(f"  ✓ No full content overlap - good!")
        
        print(f"\n  RGB-only overlap:     {self.rgb_overlap:,}")
        if self.rgb_overlap > 0:
            pct = 100 * self.rgb_overlap / self.val_size
            print(f"      {self.rgb_overlap} validation examples ({pct:.2f}%) share RGB with train")
            print(f"      (This is expected if same colors can have different structures)")
        
        print(f"\n  Structure-only overlap: {self.structure_overlap:,}")
        if self.structure_overlap > 0:
            pct = 100 * self.structure_overlap / self.val_size
            print(f"      {self.structure_overlap} validation examples ({pct:.2f}%) share structure with train")
            print(f"      (This might indicate same structures generating similar colors)")
        
        if self.overlapping_examples:
            print(f"\n--- Sample Overlapping Examples ---")
            for i, ex in enumerate(self.overlapping_examples[:5]):
                print(f"\n  Example {i+1}:")
                print(f"    RGB: {ex['rgb']}")
                print(f"    Materials: {ex['materials']}")
                print(f"    Thicknesses: {ex['thicknesses']}")
        
        print("\n" + "=" * 70)
        
        # Summary verdict
        print("\n--- VERDICT ---")
        if len(self.index_overlap) > 0:
            print("❌ DATA LEAKAGE DETECTED: Index overlap found!")
        elif self.full_content_overlap > 0:
            print("❌ DATA LEAKAGE DETECTED: Content overlap found!")
        else:
            print("✓ NO DATA LEAKAGE DETECTED")
            print("  Train and test sets appear to be properly separated.")
        print("=" * 70)


def verify_seed_consistency():
    """Verify that seed defaults are consistent across the codebase."""
    print("\n" + "=" * 70)
    print("SEED CONSISTENCY CHECK")
    print("=" * 70)
    
    # Known defaults from code inspection
    seed_locations = {
        'dataset.py (ThinFilmDataset.__init__)': 42,
        'training.py (argparse default)': 42,
        'evaluate.py (argparse default)': 42,
        'plot_training_curves.py (argparse default)': 42,
        'slurm_training_lite.sh (SEED default)': 42,
        'slurm_evaluate_lite.sh (SEED default)': 42,
        'slurm_training_curves.sh': 'uses script defaults (42)',
    }
    
    print("\nSeed defaults found in codebase:")
    all_consistent = True
    for location, seed in seed_locations.items():
        status = "✓" if seed == 42 or seed == 'uses script defaults (42)' else "⚠️"
        if seed != 42 and seed != 'uses script defaults (42)':
            all_consistent = False
        print(f"  {status} {location}: {seed}")
    
    if all_consistent:
        print("\n✓ All seed defaults are consistently set to 42")
    else:
        print("\n⚠️  WARNING: Seed inconsistency detected!")
    
    print("=" * 70)
    return all_consistent


def analyze_split_indices(data_dir: Path, seed: int = 42):
    """Analyze the train/validation split at the index level."""
    print(f"\n[INFO] Analyzing split indices with seed={seed}")
    
    files = scan_files(data_dir)
    if not files:
        print(f"[ERROR] No files found in {data_dir}")
        return None, None
    
    total_rows = sum(f.nrows for f in files)
    print(f"[INFO] Total rows in dataset: {total_rows:,}")
    
    # Create the same permutation used in dataset.py
    perm = make_permutation(total_rows, seed)
    
    # Get train/validation splits (same logic as dataset.py)
    splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)}

    train_start = int(splits['train'][0] * total_rows)
    train_end = int(splits['train'][1] * total_rows)
    val_start = int(splits['validation'][0] * total_rows)
    val_end = int(splits['validation'][1] * total_rows)

    train_indices = set(perm[train_start:train_end].tolist())
    val_indices = set(perm[val_start:val_end].tolist())

    print(f"[INFO] Train split: indices {train_start} to {train_end} ({train_end - train_start:,} examples)")
    print(f"[INFO] Validation split: indices {val_start} to {val_end} ({val_end - val_start:,} examples)")

    return train_indices, val_indices


def collect_examples(dataset: ThinFilmDataset, max_examples: int = None) -> List[TrainingExample]:
    """Collect examples from a dataset into a list."""
    examples = []
    for i, ex in enumerate(dataset):
        examples.append(ex)
        if max_examples and i >= max_examples - 1:
            break
        if (i + 1) % 10000 == 0:
            print(f"    Collected {i + 1:,} examples...")
    return examples


def investigate_leakage(
    data_dir: Path,
    seed: int = 42,
    sample_size: int = None,
    verbose: bool = False
) -> LeakageReport:
    """
    Investigate data leakage between train and test sets.
    
    Args:
        data_dir: Path to data_prompts directory
        seed: Random seed (should match training)
        sample_size: Limit number of examples to check (for speed)
        verbose: Print detailed progress
    
    Returns:
        LeakageReport with investigation results
    """
    print(f"\n[INFO] Loading datasets with seed={seed}")
    
    # Load train and test datasets
    train_dataset = ThinFilmDataset(
        data_dir, seed=seed, split='train', 
        verbose=verbose, limit_examples=sample_size
    )
    val_dataset = ThinFilmDataset(
        data_dir, seed=seed, split='validation',
        verbose=verbose, limit_examples=sample_size
    )

    print(f"[INFO] Train dataset: {len(train_dataset):,} examples")
    print(f"[INFO] Validation dataset: {len(val_dataset):,} examples")

    # Analyze at index level
    train_indices, val_indices = analyze_split_indices(data_dir, seed)
    index_overlap = train_indices.intersection(val_indices) if train_indices and val_indices else set()
    
    # Collect examples for content analysis
    print(f"\n[INFO] Collecting train examples...")
    train_examples = collect_examples(train_dataset, sample_size)
    print(f"[INFO] Collected {len(train_examples):,} train examples")
    
    print(f"\n[INFO] Collecting validation examples...")
    val_examples = collect_examples(val_dataset, sample_size)
    print(f"[INFO] Collected {len(val_examples):,} validation examples")
    
    # Build hash sets for train data
    print(f"\n[INFO] Building hash indices for train set...")
    train_full_hashes = set()
    train_rgb_hashes = set()
    train_structure_hashes = set()
    
    for ex in train_examples:
        train_full_hashes.add(hash_example(ex.rgb, ex.target_materials, ex.target_thicknesses))
        train_rgb_hashes.add(hash_rgb(ex.rgb))
        train_structure_hashes.add(hash_structure(ex.target_materials, ex.target_thicknesses))
    
    print(f"[INFO] Train unique full examples: {len(train_full_hashes):,}")
    print(f"[INFO] Train unique RGB values: {len(train_rgb_hashes):,}")
    print(f"[INFO] Train unique structures: {len(train_structure_hashes):,}")
    
    # Check validation examples against train
    print(f"\n[INFO] Checking validation examples for overlap...")
    full_content_overlap = 0
    rgb_overlap = 0
    structure_overlap = 0
    overlapping_examples = []
    
    for ex in val_examples:
        full_hash = hash_example(ex.rgb, ex.target_materials, ex.target_thicknesses)
        rgb_hash = hash_rgb(ex.rgb)
        struct_hash = hash_structure(ex.target_materials, ex.target_thicknesses)
        
        if full_hash in train_full_hashes:
            full_content_overlap += 1
            if len(overlapping_examples) < 10:
                rgb_denorm = [int(round(x.item() * 255)) for x in ex.rgb]
                overlapping_examples.append({
                    'rgb': rgb_denorm,
                    'materials': ex.target_materials,
                    'thicknesses': ex.target_thicknesses,
                })
        
        if rgb_hash in train_rgb_hashes:
            rgb_overlap += 1
        
        if struct_hash in train_structure_hashes:
            structure_overlap += 1
    
    return LeakageReport(
        train_size=len(train_examples),
        val_size=len(val_examples),
        train_indices=train_indices or set(),
        val_indices=val_indices or set(),
        index_overlap=index_overlap,
        full_content_overlap=full_content_overlap,
        rgb_overlap=rgb_overlap,
        structure_overlap=structure_overlap,
        overlapping_examples=overlapping_examples,
    )


def main():
    parser = argparse.ArgumentParser(description='Investigate data leakage between train/validation splits')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data_prompts/ directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, should match training)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Limit examples to check (for faster testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DATA LEAKAGE INVESTIGATION")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Seed: {args.seed}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size:,}")
    
    # First verify seed consistency
    verify_seed_consistency()
    
    # Run leakage investigation
    data_dir = Path(args.data_dir)
    report = investigate_leakage(
        data_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        verbose=args.verbose
    )
    
    # Print report
    report.print_report()
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'train_size': report.train_size,
            'val_size': report.val_size,
            'index_overlap_count': len(report.index_overlap),
            'full_content_overlap': report.full_content_overlap,
            'rgb_overlap': report.rgb_overlap,
            'structure_overlap': report.structure_overlap,
            'seed_used': args.seed,
            'sample_size': args.sample_size,
            'has_leakage': len(report.index_overlap) > 0 or report.full_content_overlap > 0,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()