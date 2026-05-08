"""
Dataset Module

Provides streaming datasets for all pipeline stages:
- ThinFilmDataset:       RGB → structure (pretrain_rgb_to_structure)
- TextThinFilmDataset:   text → RGB (pretrain_text_to_rgb)
- FullModelDataset:      text → structure with incorrect prompt handling (train_full_model)
- ConstraintTestDataset: text → structure with structured constraint annotations (test set)

Note: ThinFilmDataset and TextThinFilmDataset skip incorrect_prompts/ by default
(unchanged from original behavior). FullModelDataset includes them.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Tuple
from dataclasses import dataclass, field

import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq

from .materials_vocab import (
    normalize_rgb, build_structure_matrix, encode_layer,
    NUM_MATERIALS, MAX_LAYERS, EOS_TOKEN, ERROR_TOKEN  # <-- ERROR_TOKEN added
)

_LAYER_DIR_RX = re.compile(r"layers_(\d+)_angle_(\d+)_substrate_.*", re.IGNORECASE)
_SEED_RX = re.compile(r"seed_(\d+)\.parquet$", re.IGNORECASE)


@dataclass(frozen=True)
class FileMeta:
    file_id: int
    path: str
    num_layers: int
    incidence_angle: int
    structure_seed: int
    nrows: int


@dataclass
class TrainingExample:
    rgb: torch.Tensor
    target_materials: List[str]
    target_thicknesses: List[int]
    
    def get_target_tokens(self) -> List[int]:
        tokens = [encode_layer(m, t) for m, t in zip(self.target_materials, self.target_thicknesses)]
        tokens.append(EOS_TOKEN)
        return tokens


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = start or Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "create_dataset").exists():
            return parent
    raise FileNotFoundError("Could not find repository root")


def scan_files(data_prompts_dir: Path) -> List[FileMeta]:
    """Scan for parquet files, skipping incorrect_prompts directory."""
    files: List[FileMeta] = []
    file_id = 0
    
    if not data_prompts_dir.exists():
        return files
    
    for layer_dir in sorted(data_prompts_dir.iterdir()):
        if not layer_dir.is_dir() or layer_dir.name == "incorrect_prompts":
            continue
        match = _LAYER_DIR_RX.match(layer_dir.name)
        if not match:
            continue
        
        num_layers = int(match.group(1))
        incidence_angle = int(match.group(2))
        
        for pq_file in sorted(layer_dir.glob("*.parquet")):
            seed_match = _SEED_RX.search(pq_file.name)
            if not seed_match:
                continue
            seed_match = _SEED_RX.search(pq_file.name)
            if not seed_match:
                continue
            try:
                nrows = pq.ParquetFile(pq_file).metadata.num_rows
            except Exception:
                continue
            
            files.append(FileMeta(
                file_id=file_id, path=str(pq_file),
                num_layers=num_layers, incidence_angle=incidence_angle,
                structure_seed=int(seed_match.group(1)), nrows=nrows
            ))
            file_id += 1
    return files

######Delete split from function, return train and test set indices at same time !!!!
def make_permutation(N: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.argsort(torch.rand(N, generator=g), stable=True)


class ThinFilmDataset(IterableDataset):
    def __init__(self, data_prompts_dir: Path, seed: int = 42, split: str = "train", 
                 verbose: bool = False, limit_examples: Optional[int] = None):
        self.seed, self.split, self.verbose = seed, split, verbose
        self.limit_examples = limit_examples
        data_prompts_dir = Path(data_prompts_dir)
        self.files = scan_files(data_prompts_dir)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_prompts_dir}")
        
        self._file_by_id = {f.file_id: f for f in self.files}
        total_rows = sum(f.nrows for f in self.files)
        
        file_ids, row_idxs = [], []
        for f in self.files:
            file_ids.extend([f.file_id] * f.nrows)
            row_idxs.extend(range(f.nrows))
        
        self.file_ids = torch.tensor(file_ids, dtype=torch.int32)
        self.row_idxs = torch.tensor(row_idxs, dtype=torch.int32)
        
        perm = make_permutation(total_rows, seed)
        splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)} #train size 2,985,000
        start_frac, end_frac = splits[split]
        self.order = perm[int(start_frac * total_rows):int(end_frac * total_rows)]
        
        # Apply limit_examples if specified
        if limit_examples is not None and limit_examples < len(self.order):
            self.order = self.order[:limit_examples]
            if verbose:
                print(f"[Dataset] Limited to first {limit_examples} examples")
        
        if verbose:
            print(f"[Dataset] {split}: {len(self.order):,} examples from {len(self.files)} files")
    
    def __len__(self) -> int:
        return len(self.order)
    
    def __iter__(self) -> Iterator[TrainingExample]:
        worker_info = get_worker_info()
        indices = self.order if worker_info is None else self.order[worker_info.id::worker_info.num_workers]
        
        file_to_positions: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices.tolist()):
            fid = self.file_ids[global_idx].item()
            row_idx = self.row_idxs[global_idx].item()
            file_to_positions.setdefault(fid, []).append((local_idx, row_idx))
        
        results: Dict[int, TrainingExample] = {}
        for fid, positions in file_to_positions.items():
            file_meta = self._file_by_id[fid]
            try:
                table = pq.read_table(file_meta.path, columns=['materials', 'thicknesses', 'sRGB_R'])
            except Exception:
                continue
            
            for local_idx, row_idx in positions:
                try:
                    row = {col: table[col][row_idx].as_py() for col in table.column_names}
                    materials = json.loads(row['materials']) if isinstance(row['materials'], str) else row['materials']
                    thicknesses = json.loads(row['thicknesses']) if isinstance(row['thicknesses'], str) else row['thicknesses']
                    rgb = json.loads(row['sRGB_R']) if isinstance(row['sRGB_R'], str) else row['sRGB_R']
                    results[local_idx] = TrainingExample(
                        rgb=normalize_rgb(rgb), target_materials=materials, target_thicknesses=thicknesses
                    )
                except Exception:
                    continue
        
        for i in range(len(indices)):
            if i in results:
                yield results[i]


# ============================================================================
# Text-to-RGB Dataset (used by pretrain_text_to_rgb)
# ============================================================================

@dataclass
class TextRGBExample:
    """A text-to-RGB training example."""
    text: str
    rgb: torch.Tensor           # normalized [0-1], shape [3]
    materials: List[str]        # for reference / downstream
    thicknesses: List[int]      # for reference / downstream


class TextThinFilmDataset(IterableDataset):
    """
    Streaming dataset for text-to-RGB prediction.

    Reads parquet files that contain a 'text' column (natural language prompt)
    and 'sRGB_R' column (ground truth RGB color).

    Uses the same scan_files, make_permutation, and split logic as
    ThinFilmDataset so ordering and splits are identical.

    Yields TextRGBExample with:
    - text: natural language prompt string
    - rgb: normalized [0-1] tensor of shape [3]
    - materials: list of material names (for reference)
    - thicknesses: list of thicknesses in nm (for reference)
    """
    def __init__(self, data_prompts_dir: Path, seed: int = 42, split: str = "train",
                 verbose: bool = False, limit_examples: Optional[int] = None):
        self.seed, self.split, self.verbose = seed, split, verbose
        self.limit_examples = limit_examples
        data_prompts_dir = Path(data_prompts_dir)
        self.files = scan_files(data_prompts_dir)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_prompts_dir}")

        self._file_by_id = {f.file_id: f for f in self.files}
        total_rows = sum(f.nrows for f in self.files)

        file_ids, row_idxs = [], []
        for f in self.files:
            file_ids.extend([f.file_id] * f.nrows)
            row_idxs.extend(range(f.nrows))

        self.file_ids = torch.tensor(file_ids, dtype=torch.int32)
        self.row_idxs = torch.tensor(row_idxs, dtype=torch.int32)

        perm = make_permutation(total_rows, seed)
        splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)}
        start_frac, end_frac = splits[split]
        self.order = perm[int(start_frac * total_rows):int(end_frac * total_rows)]

        if limit_examples is not None and limit_examples < len(self.order):
            self.order = self.order[:limit_examples]
            if verbose:
                print(f"[TextRGBDataset] Limited to first {limit_examples} examples")

        if verbose:
            print(f"[TextRGBDataset] {split}: {len(self.order):,} examples from {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[TextRGBExample]:
        worker_info = get_worker_info()
        indices = self.order if worker_info is None else self.order[worker_info.id::worker_info.num_workers]

        file_to_positions: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices.tolist()):
            fid = self.file_ids[global_idx].item()
            row_idx = self.row_idxs[global_idx].item()
            file_to_positions.setdefault(fid, []).append((local_idx, row_idx))

        results: Dict[int, TextRGBExample] = {}
        for fid, positions in file_to_positions.items():
            file_meta = self._file_by_id[fid]
            try:
                table = pq.read_table(file_meta.path,
                                      columns=['text', 'materials', 'thicknesses', 'sRGB_R'])
            except Exception:
                continue

            for local_idx, row_idx in positions:
                try:
                    row = {col: table[col][row_idx].as_py() for col in table.column_names}
                    text = row['text']
                    materials = json.loads(row['materials']) if isinstance(row['materials'], str) else row['materials']
                    thicknesses = json.loads(row['thicknesses']) if isinstance(row['thicknesses'], str) else row['thicknesses']
                    rgb = json.loads(row['sRGB_R']) if isinstance(row['sRGB_R'], str) else row['sRGB_R']
                    results[local_idx] = TextRGBExample(
                        text=text,
                        rgb=normalize_rgb(rgb),
                        materials=materials,
                        thicknesses=thicknesses,
                    )
                except Exception:
                    continue

        for i in range(len(indices)):
            if i in results:
                yield results[i]


# ============================================================================
# NEW — Full Model Dataset (used by train_full_model)
# ============================================================================

@dataclass(frozen=True)
class FileMetaFull:
    """FileMeta with incorrect flag for full model training."""
    file_id: int
    path: str
    num_layers: int
    incidence_angle: int
    structure_seed: int
    nrows: int
    incorrect: bool = False


@dataclass
class FullTrainingExample:
    """A full-model training example (text → structure, with ERROR handling).

    For correct examples:
        incorrect = False
        text, rgb, target_materials, target_thicknesses all populated
        target_tokens = [layer_tok_0, ..., layer_tok_N-1, EOS_TOKEN]

    For incorrect (impossible) examples:
        incorrect = True
        text populated, rgb/materials/thicknesses present from parquet but unused
        target_tokens = [ERROR_TOKEN]  (single step: predict ERROR at step 0)
    """
    text: str
    rgb: torch.Tensor           # normalized [0-1], shape [3]
    target_materials: List[str]
    target_thicknesses: List[int]
    incorrect: bool

    def get_target_tokens(self) -> List[int]:
        if self.incorrect:
            return [ERROR_TOKEN]
        tokens = [encode_layer(m, t) for m, t in
                  zip(self.target_materials, self.target_thicknesses)]
        n_layers = len(self.target_materials)
        if n_layers < MAX_LAYERS:
            tokens.append(EOS_TOKEN)
        return tokens


def scan_files_with_incorrect(data_prompts_dir: Path) -> List[FileMetaFull]:
    """Scan for BOTH correct and incorrect parquet files.

    Used by train_full_model to include impossible-prompt examples.
    Correct files come from the main directory, incorrect files from
    the incorrect_prompts/ subdirectory. All get unique file_ids.

    Returns:
        List[FileMetaFull] with .incorrect field indicating source.
    """
    data_prompts_dir = Path(data_prompts_dir)
    files: List[FileMetaFull] = []
    file_id = 0

    # --- Correct examples (main directory, same logic as scan_files) ---
    if data_prompts_dir.exists():
        for layer_dir in sorted(data_prompts_dir.iterdir()):
            if not layer_dir.is_dir() or layer_dir.name == "incorrect_prompts":
                continue
            match = _LAYER_DIR_RX.match(layer_dir.name)
            if not match:
                continue

            num_layers = int(match.group(1))
            incidence_angle = int(match.group(2))

            for pq_file in sorted(layer_dir.glob("*.parquet")):
                seed_match = _SEED_RX.search(pq_file.name)
                if not seed_match:
                    continue
                try:
                    nrows = pq.ParquetFile(pq_file).metadata.num_rows
                except Exception:
                    continue

                files.append(FileMetaFull(
                    file_id=file_id, path=str(pq_file),
                    num_layers=num_layers, incidence_angle=incidence_angle,
                    structure_seed=int(seed_match.group(1)), nrows=nrows,
                    incorrect=False,
                ))
                file_id += 1

    # --- Incorrect examples (incorrect_prompts/ subdirectory) ---
    incorrect_base = data_prompts_dir / "incorrect_prompts"
    if incorrect_base.exists():
        for layer_dir in sorted(incorrect_base.iterdir()):
            if not layer_dir.is_dir():
                continue
            match = _LAYER_DIR_RX.match(layer_dir.name)
            if not match:
                continue

            num_layers = int(match.group(1))
            incidence_angle = int(match.group(2))

            for pq_file in sorted(layer_dir.glob("*.parquet")):
                seed_match = _SEED_RX.search(pq_file.name)
                if not seed_match:
                    continue
                try:
                    nrows = pq.ParquetFile(pq_file).metadata.num_rows
                except Exception:
                    continue

                files.append(FileMetaFull(
                    file_id=file_id, path=str(pq_file),
                    num_layers=num_layers, incidence_angle=incidence_angle,
                    structure_seed=int(seed_match.group(1)), nrows=nrows,
                    incorrect=True,
                ))
                file_id += 1

    return files


class FullModelDataset(IterableDataset):
    """
    Streaming dataset for full text → structure training with ERROR handling.

    Scans BOTH correct and incorrect_prompts/ parquet files using
    scan_files_with_incorrect().

    SPLIT CONSISTENCY: To avoid data leakage between pretrained modules and the
    full model, correct and incorrect examples are split INDEPENDENTLY:

        - Correct examples use make_permutation(N_correct, seed) — the SAME
          permutation and split as ThinFilmDataset and TextThinFilmDataset.
          This is possible because scan_files_with_incorrect() enumerates
          correct files first (same order as scan_files), so correct examples
          occupy global indices 0..N_correct-1 in both functions.

        - Incorrect examples use make_permutation(N_incorrect, seed) with their
          own independent permutation and 99.5/0.5 split.

    The final train/test order is the concatenation of both pools, ensuring
    that a correct example in the full model's test set was also in the
    pretrained modules' test set (and never in their training set).

    Correct examples yield target tokens [layer_0, ..., layer_N-1, EOS].
    Incorrect examples yield target tokens [ERROR].
    """

    def __init__(self, data_prompts_dir: Path, seed: int = 42, split: str = "train",
                 verbose: bool = False, limit_examples: Optional[int] = None):
        self.seed, self.split, self.verbose = seed, split, verbose
        self.limit_examples = limit_examples
        data_prompts_dir = Path(data_prompts_dir)

        self.files = scan_files_with_incorrect(data_prompts_dir)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_prompts_dir}")

        self._file_by_id = {f.file_id: f for f in self.files}

        file_ids, row_idxs, incorrect_flags = [], [], []
        for f in self.files:
            file_ids.extend([f.file_id] * f.nrows)
            row_idxs.extend(range(f.nrows))
            incorrect_flags.extend([f.incorrect] * f.nrows)

        self.file_ids = torch.tensor(file_ids, dtype=torch.int32)
        self.row_idxs = torch.tensor(row_idxs, dtype=torch.int32)
        self.incorrect_flags = torch.tensor(incorrect_flags, dtype=torch.bool)

        # --- Split correct and incorrect pools independently ---
        # Correct examples: global indices 0..N_correct-1 (same as scan_files)
        correct_indices = torch.where(~self.incorrect_flags)[0]
        incorrect_indices = torch.where(self.incorrect_flags)[0]

        N_correct = correct_indices.size(0)
        N_incorrect = incorrect_indices.size(0)

        splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)}
        start_frac, end_frac = splits[split]

        # Correct: same permutation as ThinFilmDataset / TextThinFilmDataset
        correct_perm = make_permutation(N_correct, seed)
        c_start = int(start_frac * N_correct)
        c_end = int(end_frac * N_correct)
        correct_split = correct_indices[correct_perm[c_start:c_end]]

        # Incorrect: independent permutation, same split ratio
        incorrect_perm = make_permutation(N_incorrect, seed)
        i_start = int(start_frac * N_incorrect)
        i_end = int(end_frac * N_incorrect)
        incorrect_split = incorrect_indices[incorrect_perm[i_start:i_end]]

        # Combine both pools
        self.order = torch.cat([correct_split, incorrect_split])

        if limit_examples is not None and limit_examples < len(self.order):
            self.order = self.order[:limit_examples]
            if verbose:
                print(f"[FullModelDataset] Limited to first {limit_examples} examples")

        n_correct_split = correct_split.size(0)
        n_incorrect_split = incorrect_split.size(0)

        if verbose:
            print(f"[FullModelDataset] {split}: {len(self.order):,} examples "
                  f"({n_correct_split:,} correct + {n_incorrect_split:,} incorrect) "
                  f"from {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[FullTrainingExample]:
        worker_info = get_worker_info()
        indices = (self.order if worker_info is None
                   else self.order[worker_info.id::worker_info.num_workers])

        file_to_positions: Dict[int, List[Tuple[int, int, bool]]] = {}
        for local_idx, global_idx in enumerate(indices.tolist()):
            fid = self.file_ids[global_idx].item()
            row_idx = self.row_idxs[global_idx].item()
            is_incorrect = self.incorrect_flags[global_idx].item()
            file_to_positions.setdefault(fid, []).append(
                (local_idx, row_idx, is_incorrect))

        results: Dict[int, FullTrainingExample] = {}
        for fid, positions in file_to_positions.items():
            file_meta = self._file_by_id[fid]
            try:
                table = pq.read_table(
                    file_meta.path,
                    columns=['text', 'materials', 'thicknesses', 'sRGB_R'])
            except Exception:
                continue

            for local_idx, row_idx, is_incorrect in positions:
                try:
                    row = {col: table[col][row_idx].as_py()
                           for col in table.column_names}
                    text = row['text']
                    materials = json.loads(row['materials']) if isinstance(
                        row['materials'], str) else row['materials']
                    thicknesses = json.loads(row['thicknesses']) if isinstance(
                        row['thicknesses'], str) else row['thicknesses']
                    rgb_raw = json.loads(row['sRGB_R']) if isinstance(
                        row['sRGB_R'], str) else row['sRGB_R']
                    rgb = normalize_rgb(rgb_raw)

                    results[local_idx] = FullTrainingExample(
                        text=text, rgb=rgb,
                        target_materials=materials,
                        target_thicknesses=thicknesses,
                        incorrect=is_incorrect,
                    )
                except Exception:
                    continue

        for i in range(len(indices)):
            if i in results:
                yield results[i]


# ============================================================================
# Constraint Test Dataset (used for constraint-adherence evaluation)
# ============================================================================

@dataclass
class ConstraintTestExample:
    """A test example with structured constraint annotations.

    Used for evaluating whether model predictions respect the constraints
    embedded in the natural language prompt.
    """
    text: str
    rgb: torch.Tensor           # normalized [0-1], shape [3]
    target_materials: List[str]
    target_thicknesses: List[int]
    constraints: Dict  # StructuredConstraints as dict (see src/constraints.py)


class ConstraintTestDataset:
    """
    Dataset for constraint-adherence testing.

    Loads the test set CSV from create_dataset/data_prompts/test_set.csv.
    Each example includes structured constraint annotations that can be
    checked against model predictions using src/constraints.check_constraints().

    Unlike the training/validation datasets, this is NOT an IterableDataset —
    it loads all 1200 examples into memory (small enough to fit easily).
    """

    def __init__(self, csv_path: Optional[Path] = None, verbose: bool = False):
        if csv_path is None:
            csv_path = find_repo_root() / 'create_dataset' / 'data_prompts' / 'test_set.csv'
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Test set CSV not found at {csv_path}. "
                f"Generate it with: python create_dataset/src/generate_test_set.py")

        import pandas as pd
        df = pd.read_csv(csv_path)

        self.examples: List[ConstraintTestExample] = []
        for _, row in df.iterrows():
            materials = json.loads(row['materials'])
            thicknesses = json.loads(row['thicknesses'])
            rgb = json.loads(row['sRGB_R'])
            constraints = json.loads(row['constraints'])

            self.examples.append(ConstraintTestExample(
                text=row['text'],
                rgb=normalize_rgb(rgb),
                target_materials=materials,
                target_thicknesses=thicknesses,
                constraints=constraints,
            ))

        if verbose:
            print(f"[ConstraintTestDataset] Loaded {len(self.examples)} examples from {csv_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ConstraintTestExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[ConstraintTestExample]:
        return iter(self.examples)