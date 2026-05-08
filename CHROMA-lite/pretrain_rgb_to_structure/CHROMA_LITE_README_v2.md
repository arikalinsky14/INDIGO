# CHROMA-Lite: Simplified Thin-Film Optical Design ML

> **Purpose:** This document provides complete specifications to recreate the entire codebase from scratch in a single pass. The architecture has been streamlined to remove the LLM encoder, using direct RGB color input and a material-thickness matrix representation instead.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Format Specification](#3-data-format-specification)
4. [File Specifications](#4-file-specifications)
   - 4.1 [materials_vocab.py](#41-materials_vocabpy)
   - 4.2 [dataset.py](#42-datasetpy)
   - 4.3 [model.py](#43-modelpy)
   - 4.4 [training.py](#44-trainingpy)
   - 4.5 [evaluate.py](#45-evaluatepy)
5. [Model Architecture](#5-model-architecture)
6. [Key Algorithms](#6-key-algorithms)
7. [Type Reference](#7-type-reference)
8. [Dependencies](#8-dependencies)
9. [Usage Examples](#9-usage-examples)

---

## 1. Project Overview

This codebase implements a **transformer-based model for optical thin-film design** with simplified input/output representations:

| Component | Description |
|-----------|-------------|
| **Input** | RGB color (3 floats, normalized 0-1) + structure matrix [25 × 8] |
| **Structure Matrix** | 25 rows (materials) × 8 columns (layer positions), values = normalized thickness (0-1) |
| **Transformer** | Causal self-attention enabling single-pass training over all positions |
| **Output** | 8 parallel classification heads over 1002 tokens (one per layer position) |
| **Vocabulary** | 1002 tokens: 25 materials × 40 thicknesses = 1000 pairs + ERROR + EOS |

### Key Design Decisions

- **No LLM:** Removed frozen language model encoder entirely. Color is specified directly as RGB values.
- **Matrix representation:** Structure state is a 25×8 matrix where each row is a material and each column is a layer position. Cell values are normalized thicknesses.
- **Single-pass causal training:** Using causal masking, all 8 layer positions are predicted in a single forward pass. Each position can only attend to RGB and previous layers, preventing data leakage.
- **Parallel prediction heads:** After attention, each position independently predicts through a shared classification head - the positions "diverge" after the transformer to make their own predictions.
- **Unified vocabulary:** Model predicts from 1002 discrete tokens (material-thickness pairs + ERROR + EOS). Single cross-entropy loss.

### Training Efficiency

The causal masking approach provides **~8x speedup** over naive autoregressive training:

| Approach | Forward Passes per Example | Notes |
|----------|---------------------------|-------|
| Naive (multi-pass) | N+1 (up to 9) | Recompute attention for each layer |
| Causal (single-pass) | 1 | All positions predicted simultaneously |

### Design Simplifications from Original CHROMA

| Original | Simplified |
|----------|------------|
| Natural language prompts | RGB values (3 floats) |
| Frozen TinyLlama encoder | Removed entirely |
| Cross-attention to encoder memory | Causal self-attention only |
| 1004-token vocabulary (with PAD, BOS) | 1002 tokens (no PAD, no BOS) |
| Sharp attention with learnable temperature | Standard attention |
| Multi-pass autoregressive training | Single-pass causal training |

---

## 2. Repository Structure

```
chroma-lite/
├── src/
│   ├── materials_vocab.py      # Material constants and token encoding/decoding
│   ├── dataset.py              # Parquet dataset loading
│   ├── model.py                # Model architecture
│   └── utils.py                # Shared utilities (optional)
├── scripts/
│   ├── training.py             # Main training script
│   └── evaluate.py             # Evaluation with color metrics
├── data/
│   └── checkpoints/            # Model checkpoints (auto-created)
│       └── <model_tag>/
│           ├── config.json
│           ├── model.pt
│           └── meta.json
├── create_dataset/
│   └── data_prompts/           # Training data directory
│       └── layers_<N>_angle_<A>_substrate_<S>/
│           └── seed_<X>.parquet
└── outputs/
    └── eval_results.json
```

---

## 3. Data Format Specification

### 3.1 Parquet File Structure

**Location pattern:** `create_dataset/data_prompts/layers_<N>_angle_<A>_substrate_<S>/seed_<X>.parquet`

#### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `materials` | `str` (JSON) | Layer materials, e.g., `'["SiO2", "Ag"]'` |
| `thicknesses` | `str` (JSON) | Layer thicknesses in nm, e.g., `'[100, 50]'` |
| `num_layers` | `int` | Number of layers (1-8) |
| `sRGB_R` | `str` (JSON) or `List[int]` | Target color `'[R, G, B]'` (0-255) |

#### Incorrect Prompts

Located in `incorrect_prompts/` subdirectory. These represent impossible color/constraint combinations where no valid thin-film stack exists. For these examples, the model should learn to output `ERROR` at position 0.

### 3.2 Materials and Vocabulary

```python
MATERIALS: List[str] = [
    'Ag', 'Al', 'Al2O3', 'Au', 'AZO', 'Cr', 'GaAs', 'GaInP', 'GaP', 'Ge',
    'InP', 'ITO', 'Mn', 'Ni', 'Pd', 'Pt', 'Si3N4', 'SiO2', 'Ti', 'TiN',
    'TiO2', 'aSi', 'cSi', 'W', 'ZnO'
]  # 25 materials

THICKNESSES: List[int] = list(range(5, 201, 5))  # [5, 10, 15, ..., 200] — 40 values

NUM_MATERIALS: int = 25
NUM_THICKNESSES: int = 40
MAX_LAYERS: int = 8
MAX_THICKNESS: int = 200  # nm, for normalization

# === Vocabulary (1002 tokens) ===
# Tokens 0-999: (material, thickness) pairs
#   Token ID = material_idx * 40 + thickness_idx
#   where thickness_idx = (thickness_nm - 5) / 5
# Token 1000: ERROR (impossible design)
# Token 1001: EOS (design complete)

ERROR_TOKEN: int = 1000
EOS_TOKEN: int = 1001
VOCAB_SIZE: int = 1002
```

### 3.3 Token ID Encoding/Decoding

```python
def encode_layer(material: str, thickness_nm: int) -> int:
    """Convert (material, thickness) pair to token ID."""
    mat_idx = MATERIAL_TO_IDX[material]
    thick_idx = (thickness_nm - 5) // 5  # 0-39
    return mat_idx * NUM_THICKNESSES + thick_idx

def decode_token(token_id: int) -> Tuple[Optional[str], Optional[int], str]:
    """
    Convert token ID back to (material, thickness) or special token.
    
    Returns:
        (material_name, thickness_nm, token_type)
        token_type is one of: 'LAYER', 'ERROR', 'EOS'
        For ERROR/EOS, material and thickness are None
    """
    if token_id == ERROR_TOKEN:
        return None, None, 'ERROR'
    if token_id == EOS_TOKEN:
        return None, None, 'EOS'
    
    mat_idx = token_id // NUM_THICKNESSES
    thick_idx = token_id % NUM_THICKNESSES
    
    material = IDX_TO_MATERIAL[mat_idx]
    thickness = THICKNESSES[thick_idx]  # or: 5 + thick_idx * 5
    
    return material, thickness, 'LAYER'
```

### 3.4 Input Representation

#### RGB Input
```python
# Original: [R, G, B] in 0-255
# Normalized: [r, g, b] in 0.0-1.0
rgb_normalized = torch.tensor([R/255.0, G/255.0, B/255.0])  # Shape: [3]
```

#### Structure Matrix
```python
# Shape: [NUM_MATERIALS, MAX_LAYERS] = [25, 8]
# Each cell: thickness / MAX_THICKNESS (0.0 to 1.0)
# Zero means material not present at that layer position

# Example: Structure with [SiO2@100nm, Ag@50nm]
structure_matrix = torch.zeros(25, 8)
structure_matrix[MATERIAL_TO_IDX['SiO2'], 0] = 100 / 200  # 0.5
structure_matrix[MATERIAL_TO_IDX['Ag'], 1] = 50 / 200     # 0.25
# Remaining columns are 0 (no more layers)
```

### 3.5 Output Representation

The model outputs logits for all 8 positions simultaneously:

```python
# Output: [B, 8, 1002] logits
# Position i predicts what token should be placed at layer i

# During training: parallel prediction at all positions
# During inference: autoregressive (use position 0, then 1, etc.)
```

### 3.6 Causal Masking and Data Flow

The key insight is that causal masking allows single-pass training:

```
Input Sequence: [RGB, L0, L1, L2, L3, L4, L5, L6, L7]
Position:         0    1   2   3   4   5   6   7   8

With causal mask, each position can only attend to itself and earlier positions:
- Position 1 (L0) sees: RGB              → predicts token for layer 0
- Position 2 (L1) sees: RGB, L0          → predicts token for layer 1
- Position 3 (L2) sees: RGB, L0, L1      → predicts token for layer 2
- Position 4 (L3) sees: RGB, L0, L1, L2  → predicts token for layer 3
- ... and so on

Output positions 1-8 each produce [1002] logits independently.
The RGB token (position 0) does not produce predictions.
```

**Important:** After the causal self-attention layers, each position has a contextualized representation that includes information from RGB and all previous layers. These representations then pass through a **shared** classification head independently - they "diverge" at this point, each making its own prediction without further interaction.

### 3.7 Training Targets and Loss Masking

For a structure with N layers (where N ≤ 8):

```python
# Example: 3-layer structure ['SiO2'@100, 'Ag'@50, 'TiO2'@75]
# Target tokens: [tok0, tok1, tok2, EOS, PAD, PAD, PAD, PAD]
#                  ↓     ↓     ↓    ↓    ↓    ↓    ↓    ↓
# Positions:       1     2     3    4    5    6    7    8

targets = [
    encode_layer('SiO2', 100),  # Position 1 predicts layer 0
    encode_layer('Ag', 50),      # Position 2 predicts layer 1
    encode_layer('TiO2', 75),    # Position 3 predicts layer 2
    EOS_TOKEN,                   # Position 4 predicts EOS
    IGNORE_INDEX,                # Positions 5-8: masked from loss
    IGNORE_INDEX,
    IGNORE_INDEX,
    IGNORE_INDEX,
]

# For incorrect examples:
targets = [ERROR_TOKEN, IGNORE_INDEX, IGNORE_INDEX, ...]  # Only position 1 active
```

Use `F.cross_entropy(..., ignore_index=IGNORE_INDEX)` to mask out padding positions.

### 3.8 Train/Validation Split

```python
splits: Dict[str, Tuple[float, float]] = {
    "train": (0.0, 0.995),   # 99.5% of data
    "validation": (0.995, 1.0)     # 0.5% of data
}
```

---

## 4. File Specifications

### 4.1 `materials_vocab.py`

**Location:** `src/materials_vocab.py`
**Purpose:** Material/thickness vocabulary and token encoding/decoding.

```python
from typing import List, Dict, Tuple, Optional
import torch

# === Constants ===

MATERIALS: List[str] = [
    'Ag', 'Al', 'Al2O3', 'Au', 'AZO', 'Cr', 'GaAs', 'GaInP', 'GaP', 'Ge',
    'InP', 'ITO', 'Mn', 'Ni', 'Pd', 'Pt', 'Si3N4', 'SiO2', 'Ti', 'TiN',
    'TiO2', 'aSi', 'cSi', 'W', 'ZnO'
]

THICKNESSES: List[int] = list(range(5, 201, 5))  # [5, 10, ..., 200]

NUM_MATERIALS: int = 25
NUM_THICKNESSES: int = 40
MAX_LAYERS: int = 8
MAX_THICKNESS: int = 200  # nm

# Special tokens
ERROR_TOKEN: int = 1000
EOS_TOKEN: int = 1001
VOCAB_SIZE: int = 1002

# For loss masking (positions after EOS)
IGNORE_INDEX: int = -100  # PyTorch's default ignore index for cross_entropy

# Lookup dicts
MATERIAL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(MATERIALS)}
IDX_TO_MATERIAL: Dict[int, str] = {i: m for i, m in enumerate(MATERIALS)}


# === Token Encoding/Decoding ===

def encode_layer(material: str, thickness_nm: int) -> int:
    """
    Convert (material, thickness) pair to token ID.
    
    Args:
        material: Material name (e.g., 'SiO2')
        thickness_nm: Thickness in nm (5, 10, 15, ..., 200)
    
    Returns:
        Token ID in range [0, 999]
    """
    mat_idx = MATERIAL_TO_IDX[material]
    thick_idx = (thickness_nm - 5) // 5
    return mat_idx * NUM_THICKNESSES + thick_idx


def decode_token(token_id: int) -> Tuple[Optional[str], Optional[int], str]:
    """
    Convert token ID to (material, thickness) or special token type.
    
    Args:
        token_id: Token ID in range [0, 1001]
    
    Returns:
        Tuple of (material, thickness_nm, token_type)
        token_type is one of: 'LAYER', 'ERROR', 'EOS'
        For ERROR/EOS, material and thickness are None
    """
    if token_id == ERROR_TOKEN:
        return None, None, 'ERROR'
    if token_id == EOS_TOKEN:
        return None, None, 'EOS'
    
    mat_idx = token_id // NUM_THICKNESSES
    thick_idx = token_id % NUM_THICKNESSES
    
    material = IDX_TO_MATERIAL[mat_idx]
    thickness = THICKNESSES[thick_idx]
    
    return material, thickness, 'LAYER'


def is_terminal_token(token_id: int) -> bool:
    """Check if token is ERROR or EOS."""
    return token_id >= ERROR_TOKEN


# === RGB Utilities ===

def normalize_rgb(rgb: List[int]) -> torch.Tensor:
    """
    Convert RGB [0-255] to normalized tensor [0-1].
    
    Args:
        rgb: List of 3 integers [R, G, B] in 0-255 range
    
    Returns:
        torch.Tensor of shape [3] with values in 0-1
    """
    return torch.tensor([c / 255.0 for c in rgb], dtype=torch.float32)


def denormalize_rgb(rgb_norm: torch.Tensor) -> List[int]:
    """
    Convert normalized RGB [0-1] back to [0-255].
    
    Args:
        rgb_norm: Tensor of shape [3] with values in 0-1
    
    Returns:
        List of 3 integers [R, G, B] in 0-255 range
    """
    return [int(round(c.item() * 255)) for c in rgb_norm]


# === Structure Matrix Utilities ===

def normalize_thickness(thickness_nm: int) -> float:
    """Normalize thickness from nm to 0-1 range."""
    return thickness_nm / MAX_THICKNESS


def build_structure_matrix(
    materials: List[str],
    thicknesses: List[int]
) -> torch.Tensor:
    """
    Build structure matrix from material-thickness lists.
    
    Args:
        materials: List of material names, e.g., ['SiO2', 'Ag']
        thicknesses: List of thicknesses in nm, e.g., [100, 50]
    
    Returns:
        torch.Tensor of shape [NUM_MATERIALS, MAX_LAYERS] = [25, 8]
        Each cell contains normalized thickness (0-1) or 0 if not present.
    """
    matrix = torch.zeros(NUM_MATERIALS, MAX_LAYERS, dtype=torch.float32)
    
    for layer_idx, (mat, thick) in enumerate(zip(materials, thicknesses)):
        if layer_idx >= MAX_LAYERS:
            break
        mat_idx = MATERIAL_TO_IDX[mat]
        matrix[mat_idx, layer_idx] = normalize_thickness(thick)
    
    return matrix


def decode_structure_matrix(
    matrix: torch.Tensor
) -> Tuple[List[str], List[int]]:
    """
    Decode structure matrix back to material-thickness lists.
    
    Args:
        matrix: Tensor of shape [NUM_MATERIALS, MAX_LAYERS]
    
    Returns:
        Tuple of (materials_list, thicknesses_list)
    """
    materials = []
    thicknesses = []
    
    for layer_idx in range(MAX_LAYERS):
        col = matrix[:, layer_idx]
        if col.sum() == 0:  # No material at this layer
            break
        
        mat_idx = col.argmax().item()
        thickness_norm = col[mat_idx].item()
        
        if thickness_norm > 0:
            materials.append(IDX_TO_MATERIAL[mat_idx])
            # Snap to nearest valid thickness
            thickness_nm = round(thickness_norm * MAX_THICKNESS / 5) * 5
            thickness_nm = max(5, min(200, thickness_nm))
            thicknesses.append(thickness_nm)
    
    return materials, thicknesses


def get_num_layers(matrix: torch.Tensor) -> int:
    """
    Count number of layers in structure matrix.
    
    Args:
        matrix: Tensor of shape [NUM_MATERIALS, MAX_LAYERS]
    
    Returns:
        Number of occupied layer columns
    """
    for layer_idx in range(MAX_LAYERS):
        if matrix[:, layer_idx].sum() == 0:
            return layer_idx
    return MAX_LAYERS


def build_target_sequence(
    materials: List[str],
    thicknesses: List[int],
    is_incorrect: bool
) -> torch.Tensor:
    """
    Build target token sequence for training.
    
    Args:
        materials: List of material names
        thicknesses: List of thicknesses in nm
        is_incorrect: Whether this is an impossible design
    
    Returns:
        torch.Tensor of shape [MAX_LAYERS] with token IDs
        Positions after EOS are filled with IGNORE_INDEX
    """
    targets = torch.full((MAX_LAYERS,), IGNORE_INDEX, dtype=torch.long)
    
    if is_incorrect:
        # Only first position predicts ERROR
        targets[0] = ERROR_TOKEN
    else:
        n_layers = len(materials)
        
        # Fill in layer tokens
        for i, (mat, thick) in enumerate(zip(materials, thicknesses)):
            targets[i] = encode_layer(mat, thick)
        
        # Position after last layer predicts EOS
        if n_layers < MAX_LAYERS:
            targets[n_layers] = EOS_TOKEN
        # If exactly 8 layers, last position predicts EOS
        else:
            targets[MAX_LAYERS - 1] = EOS_TOKEN
    
    return targets
```

---

### 4.2 `dataset.py`

**Location:** `src/dataset.py`
**Purpose:** Parquet dataset loading with structure matrix and target sequence conversion.

```python
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq

from .materials_vocab import (
    normalize_rgb, build_structure_matrix, build_target_sequence,
    NUM_MATERIALS, MAX_LAYERS, IGNORE_INDEX
)

# === Constants ===

_LAYER_DIR_RX = re.compile(r"layers_(\d+)_angle_(\d+)_substrate_", re.IGNORECASE)
_SEED_RX = re.compile(r"seed_(\d+)\.parquet$", re.IGNORECASE)


# === Data Classes ===

@dataclass(frozen=True)
class FileMeta:
    """Metadata for a single parquet file."""
    file_id: int
    path: str
    num_layers: int
    incidence_angle: int
    structure_seed: int
    incorrect: bool
    nrows: int


@dataclass
class TrainingExample:
    """
    Single training example with all required tensors.
    
    For single-pass causal training, each example produces:
    - rgb: [3] normalized RGB target color
    - structure_matrix: [25, 8] full ground truth structure
    - targets: [8] token IDs for each position (with IGNORE_INDEX for padding)
    """
    rgb: torch.Tensor              # [3] normalized RGB
    structure_matrix: torch.Tensor # [25, 8] full structure (for input)
    targets: torch.Tensor          # [8] target token IDs
    target_materials: List[str]    # Ground truth materials (for evaluation)
    target_thicknesses: List[int]  # Ground truth thicknesses (for evaluation)
    is_incorrect: bool             # True if impossible design
    num_layers: int                # Number of actual layers (0 for incorrect)


# === Functions ===

def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up directory tree to find repository root."""
    current = start or Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "create_dataset").exists():
            return parent
    raise FileNotFoundError("Could not find repository root with create_dataset/")


def scan_files(data_prompts_dir: Path) -> List[FileMeta]:
    """
    Scan filesystem for all parquet files.
    
    Scans both main directory and incorrect_prompts/ subdirectory.
    
    Returns:
        List of FileMeta, sorted by file_id
    """
    files = []
    file_id = 0
    
    for is_incorrect, base_dir in [(False, data_prompts_dir), 
                                     (True, data_prompts_dir / "incorrect_prompts")]:
        if not base_dir.exists():
            continue
            
        for layer_dir in sorted(base_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            
            match = _LAYER_DIR_RX.match(layer_dir.name)
            if not match:
                continue
            
            num_layers = int(match.group(1))
            incidence_angle = int(match.group(2))
            
            for pq_file in sorted(layer_dir.glob("seed_*.parquet")):
                seed_match = _SEED_RX.search(pq_file.name)
                if not seed_match:
                    continue
                
                structure_seed = int(seed_match.group(1))
                
                # Get row count from metadata (no data read)
                pf = pq.ParquetFile(pq_file)
                nrows = pf.metadata.num_rows
                
                files.append(FileMeta(
                    file_id=file_id,
                    path=str(pq_file),
                    num_layers=num_layers,
                    incidence_angle=incidence_angle,
                    structure_seed=structure_seed,
                    incorrect=is_incorrect,
                    nrows=nrows
                ))
                file_id += 1
    
    return files


def make_permutation(N: int, seed: int) -> torch.Tensor:
    """Generate deterministic shuffle permutation."""
    g = torch.Generator()
    g.manual_seed(seed)
    key = torch.rand(N, generator=g)
    return torch.argsort(key, stable=True)


# === Main Dataset Class ===

class ThinFilmDataset(IterableDataset):
    """
    Streaming dataset over Parquet files with deterministic shuffling.
    
    Features:
    - Single shuffled pass over all rows
    - Multi-worker DataLoader compatible
    - Outputs RGB + full structure matrix + target sequence for causal training
    - One example per structure (not per layer)
    """
    
    def __init__(
        self,
        data_prompts_dir: Path,
        seed: int = 42,
        split: str = "train",
        verbose: bool = False
    ) -> None:
        """
        Args:
            data_prompts_dir: Path to data_prompts/ directory
            seed: Random seed for shuffling
            split: "train" (0-99.5%) or "validation" (99.5-100%)
            verbose: Print loading info
        """
        self.seed = seed
        self.split = split
        self.verbose = verbose
        
        # Scan files and build manifest
        self.files = scan_files(data_prompts_dir)
        self._file_by_id = {f.file_id: f for f in self.files}
        
        # Build global index
        total_rows = sum(f.nrows for f in self.files)
        
        # Create arrays for each row's file_id, row_idx, incorrect flag
        file_ids = []
        row_idxs = []
        incorrects = []
        
        for f in self.files:
            file_ids.extend([f.file_id] * f.nrows)
            row_idxs.extend(range(f.nrows))
            incorrects.extend([f.incorrect] * f.nrows)
        
        self.file_ids = torch.tensor(file_ids, dtype=torch.int32)
        self.row_idxs = torch.tensor(row_idxs, dtype=torch.int32)
        self.incorrects = torch.tensor(incorrects, dtype=torch.bool)
        
        # Shuffle and split
        perm = make_permutation(total_rows, seed)
        
        splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)}
        start_frac, end_frac = splits[split]
        start_idx = int(start_frac * total_rows)
        end_idx = int(end_frac * total_rows)
        
        self.order = perm[start_idx:end_idx]
        
        if verbose:
            print(f"[Dataset] {split}: {len(self.order)} examples from {len(self.files)} files")
    
    def __len__(self) -> int:
        return len(self.order)
    
    def __iter__(self) -> Iterator[TrainingExample]:
        """Yield TrainingExample objects in shuffled order."""
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process
            indices = self.order
        else:
            # Multi-worker: stride sharding
            indices = self.order[worker_info.id::worker_info.num_workers]
        
        # Group by file for efficient batch reads
        file_to_positions: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices.tolist()):
            fid = self.file_ids[global_idx].item()
            row_idx = self.row_idxs[global_idx].item()
            if fid not in file_to_positions:
                file_to_positions[fid] = []
            file_to_positions[fid].append((local_idx, row_idx))
        
        # Read and yield in order
        results: Dict[int, TrainingExample] = {}
        
        for fid, positions in file_to_positions.items():
            file_meta = self._file_by_id[fid]
            row_indices = [p[1] for p in positions]
            
            # Read batch from parquet
            table = pq.read_table(
                file_meta.path,
                columns=['materials', 'thicknesses', 'sRGB_R']
            )
            
            for local_idx, row_idx in positions:
                row = {col: table[col][row_idx].as_py() for col in table.column_names}
                
                # Parse JSON fields
                materials = json.loads(row['materials']) if isinstance(row['materials'], str) else row['materials']
                thicknesses = json.loads(row['thicknesses']) if isinstance(row['thicknesses'], str) else row['thicknesses']
                rgb = json.loads(row['sRGB_R']) if isinstance(row['sRGB_R'], str) else row['sRGB_R']
                
                # Build tensors
                rgb_tensor = normalize_rgb(rgb)
                structure_matrix = build_structure_matrix(materials, thicknesses)
                targets = build_target_sequence(materials, thicknesses, file_meta.incorrect)
                
                example = TrainingExample(
                    rgb=rgb_tensor,
                    structure_matrix=structure_matrix,
                    targets=targets,
                    target_materials=materials,
                    target_thicknesses=thicknesses,
                    is_incorrect=file_meta.incorrect,
                    num_layers=0 if file_meta.incorrect else len(materials)
                )
                results[local_idx] = example
        
        # Yield in original shuffled order
        for i in range(len(indices)):
            yield results[i]


def collate_fn(examples: List[TrainingExample]) -> Dict[str, torch.Tensor]:
    """
    Collate TrainingExamples into batched tensors.
    
    Returns:
        Dict with:
            'rgb': [B, 3]
            'structure_matrix': [B, 25, 8]
            'targets': [B, 8]
    """
    return {
        'rgb': torch.stack([ex.rgb for ex in examples]),
        'structure_matrix': torch.stack([ex.structure_matrix for ex in examples]),
        'targets': torch.stack([ex.targets for ex in examples]),
    }
```

---

### 4.3 `model.py`

**Location:** `src/model.py`
**Purpose:** Model architecture with causal self-attention for single-pass training.

```python
from typing import Dict, Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .materials_vocab import (
    NUM_MATERIALS, MAX_LAYERS, VOCAB_SIZE,
    ERROR_TOKEN, EOS_TOKEN, IGNORE_INDEX,
    decode_token, is_terminal_token, normalize_thickness, MATERIAL_TO_IDX
)


# === Configuration ===

class ModelConfig:
    """Model configuration parameters."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: Optional[int] = None,  # Default: 4 * d_model
        dropout: float = 0.1,
        max_layers: int = MAX_LAYERS,
    ) -> None:
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim or (4 * d_model)
        self.dropout = dropout
        self.max_layers = max_layers
        
        # Input dimensions
        self.rgb_dim = 3
        self.matrix_dim = NUM_MATERIALS * MAX_LAYERS  # 25 * 8 = 200
        
        # Sequence length: 1 (RGB) + 8 (layers) = 9
        self.seq_len = 1 + max_layers
        
        # Output: unified vocabulary
        self.vocab_size = VOCAB_SIZE  # 1002
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'max_layers': self.max_layers,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**d)


# === Positional Encoding ===

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# === Causal Transformer ===

class CausalTransformerEncoder(nn.Module):
    """
    Transformer encoder with causal (autoregressive) masking.
    
    Uses nn.TransformerEncoder but applies a causal mask so each position
    can only attend to itself and previous positions.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
        
        Returns:
            [B, T, D] output with causal masking applied
        """
        T = x.size(1)
        
        # Create causal mask: True means "mask out" (cannot attend)
        # Upper triangular = future positions masked
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        
        return self.transformer(x, mask=causal_mask, is_causal=True)


# === Main Model ===

class ThinFilmTransformer(nn.Module):
    """
    Transformer model for thin-film structure prediction.
    
    Input: RGB color [B, 3] + structure matrix [B, 25, 8]
    Output: Logits for all 8 layer positions [B, 8, 1002]
    
    Architecture:
    1. Project RGB to d_model (position 0)
    2. Project each structure column to d_model (positions 1-8)
    3. Apply causal self-attention (each position sees only previous)
    4. Shared classification head applied to positions 1-8
    
    Training: Single forward pass predicts all positions simultaneously
    Inference: Autoregressive generation using positions sequentially
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # === Input Projections ===
        # RGB token projection (position 0)
        self.rgb_proj = nn.Linear(config.rgb_dim, config.d_model)
        
        # Structure matrix: project each column (layer) as a token
        # Each column has NUM_MATERIALS values
        self.layer_proj = nn.Linear(NUM_MATERIALS, config.d_model)
        
        # Positional encoding for the sequence
        # Sequence: [RGB, L0, L1, ..., L7] = 9 tokens
        self.pos_encoder = PositionalEncoding(
            config.d_model, 
            max_len=config.seq_len,  # 9
            dropout=config.dropout
        )
        
        # === Causal Transformer ===
        self.transformer = CausalTransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout
        )
        
        # === Output Head ===
        # Shared classification head applied to each of positions 1-8
        # Each position independently predicts its token (they "diverge" here)
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.vocab_size)  # 1002
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        rgb: torch.Tensor,              # [B, 3]
        structure_matrix: torch.Tensor, # [B, 25, 8]
    ) -> torch.Tensor:
        """
        Forward pass for training (single-pass, all positions).
        
        Args:
            rgb: Normalized RGB values [B, 3]
            structure_matrix: Current structure state [B, NUM_MATERIALS, MAX_LAYERS]
        
        Returns:
            logits: [B, 8, 1002] logits for positions 1-8 (layer predictions)
        """
        B = rgb.size(0)
        
        # === Build token sequence ===
        # RGB token: [B, 3] -> [B, 1, d_model]
        rgb_token = self.rgb_proj(rgb).unsqueeze(1)
        
        # Layer tokens: [B, 25, 8] -> [B, 8, 25] -> [B, 8, d_model]
        layer_tokens = self.layer_proj(structure_matrix.transpose(1, 2))
        
        # Concatenate: [B, 9, d_model]
        # Sequence: [RGB, L0, L1, L2, L3, L4, L5, L6, L7]
        tokens = torch.cat([rgb_token, layer_tokens], dim=1)
        
        # Add positional encoding
        tokens = self.pos_encoder(tokens)
        
        # === Causal Transformer ===
        # Each position can only attend to itself and previous positions
        encoded = self.transformer(tokens)  # [B, 9, d_model]
        
        # === Output ===
        # Take positions 1-8 (layer positions, not RGB)
        layer_outputs = encoded[:, 1:, :]  # [B, 8, d_model]
        
        # Apply shared classification head to each position independently
        # This is where positions "diverge" - each makes its own prediction
        logits = self.output_head(layer_outputs)  # [B, 8, 1002]
        
        return logits
    
    def generate(
        self,
        rgb: torch.Tensor,  # [3] single example
        device: torch.device
    ) -> Tuple[torch.Tensor, List[int], str]:
        """
        Autoregressive generation for inference.
        
        Args:
            rgb: Normalized RGB values [3]
            device: Device to run on
        
        Returns:
            structure_matrix: [25, 8] generated structure
            tokens: List of generated token IDs
            stop_reason: 'EOS', 'ERROR', or 'MAX_LEN'
        """
        self.eval()
        
        structure = torch.zeros(NUM_MATERIALS, MAX_LAYERS, device=device)
        rgb = rgb.to(device)
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(MAX_LAYERS):
                # Forward pass
                logits = self.forward(
                    rgb.unsqueeze(0),
                    structure.unsqueeze(0)
                )  # [1, 8, 1002]
                
                # Get prediction for current step (position step)
                step_logits = logits[0, step, :]  # [1002]
                token_id = step_logits.argmax().item()
                
                generated_tokens.append(token_id)
                
                # Check for termination
                if token_id == ERROR_TOKEN:
                    return structure, generated_tokens, 'ERROR'
                
                if token_id == EOS_TOKEN:
                    return structure, generated_tokens, 'EOS'
                
                # Decode and update structure matrix
                material, thickness, _ = decode_token(token_id)
                mat_idx = MATERIAL_TO_IDX[material]
                structure[mat_idx, step] = normalize_thickness(thickness)
        
        return structure, generated_tokens, 'MAX_LEN'


# === Loss Function ===

def compute_loss(
    model: ThinFilmTransformer,
    rgb: torch.Tensor,               # [B, 3]
    structure_matrix: torch.Tensor,  # [B, 25, 8]
    targets: torch.Tensor,           # [B, 8] token IDs with IGNORE_INDEX for padding
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss (single-pass over all positions).
    
    Args:
        model: The transformer model
        rgb: Input RGB values
        structure_matrix: Full ground truth structure
        targets: Target token IDs for each position [B, 8]
                 Positions after EOS are IGNORE_INDEX
    
    Returns:
        Dict with 'loss' and 'accuracy'
    """
    logits = model(rgb, structure_matrix)  # [B, 8, 1002]
    
    # Reshape for cross_entropy: [B*8, 1002] and [B*8]
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    
    # Cross-entropy with ignore_index handles padding automatically
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=IGNORE_INDEX)
    
    # Accuracy (only on non-ignored positions)
    pred_tokens = logits.argmax(dim=-1)  # [B, 8]
    mask = targets != IGNORE_INDEX
    if mask.any():
        correct = (pred_tokens == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
    else:
        accuracy = torch.tensor(0.0, device=rgb.device)
    
    return {
        'loss': loss,
        'accuracy': accuracy
    }
```

---

### 4.4 `training.py`

**Location:** `scripts/training.py`
**Purpose:** Main training script with single-pass causal training.

```python
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials_vocab import VOCAB_SIZE, MAX_LAYERS
from src.dataset import ThinFilmDataset, collate_fn, find_repo_root
from src.model import ThinFilmTransformer, ModelConfig, compute_loss


def save_checkpoint(
    model: nn.Module,
    config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    save_dir: Path
) -> None:
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / 'model.pt')
    torch.save(optimizer.state_dict(), save_dir / 'optimizer.pt')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    with open(save_dir / 'meta.json', 'w') as f:
        json.dump({'step': step, 'loss': loss}, f, indent=2)
    
    print(f"[Checkpoint] Saved to {save_dir} at step {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train thin-film transformer')
    
    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data_prompts/ directory')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/default')
    parser.add_argument('--save-every', type=int, default=1000)
    
    # Misc
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        repo_root = find_repo_root()
        data_dir = repo_root / 'create_dataset' / 'data_prompts'
    
    print(f"[INFO] Loading data from {data_dir}")
    
    # Dataset
    dataset = ThinFilmDataset(
        data_prompts_dir=data_dir,
        seed=args.seed,
        split=args.split,
        verbose=args.verbose
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    )
    
    model = ThinFilmTransformer(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")
    print(f"[INFO] Vocabulary size: {config.vocab_size}")
    print(f"[INFO] Sequence length: {config.seq_len} (1 RGB + {MAX_LAYERS} layers)")
    print(f"[INFO] Single-pass causal training enabled")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    save_dir = Path(args.save_dir)
    global_step = 0
    
    print("[INFO] Starting training...")
    sys.stdout.flush()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch in loader:
            # Move to device
            rgb = batch['rgb'].to(device)
            structure_matrix = batch['structure_matrix'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward + loss (single pass for all positions)
            losses = compute_loss(model, rgb, structure_matrix, targets)
            
            # Backward
            optimizer.zero_grad()
            losses['loss'].backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Logging
            epoch_loss += losses['loss'].item()
            epoch_acc += losses['accuracy'].item()
            n_batches += 1
            global_step += 1
            
            if global_step % 100 == 0:
                print(f"  Step {global_step}: loss={losses['loss'].item():.4f}, "
                      f"acc={losses['accuracy'].item():.3f}")
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(
                    model, config, optimizer,
                    global_step, losses['loss'].item(),
                    save_dir / f'step_{global_step}'
                )
        
        # Epoch summary
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}, acc={avg_acc:.3f}")
        
        # Save end of epoch
        save_checkpoint(
            model, config, optimizer,
            global_step, avg_loss,
            save_dir / 'latest'
        )
    
    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
```

---

### 4.5 `evaluate.py`

**Location:** `scripts/evaluate.py`
**Purpose:** Evaluation with autoregressive generation and color metrics.

```python
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials_vocab import (
    NUM_MATERIALS, MAX_LAYERS, VOCAB_SIZE,
    ERROR_TOKEN, EOS_TOKEN, MATERIAL_TO_IDX,
    decode_token, normalize_rgb, denormalize_rgb,
    decode_structure_matrix
)
from src.dataset import ThinFilmDataset, find_repo_root
from src.model import ThinFilmTransformer, ModelConfig


# === Color Utilities ===

def sRGB_to_Lab(sRGB: List[int]) -> Tuple[float, float, float]:
    """Convert sRGB [0-255] to CIELAB under D65 illuminant."""
    # Normalize to 0-1
    rgb = np.array(sRGB) / 255.0
    
    # sRGB to linear RGB
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    # Linear RGB to XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = M @ rgb
    
    # Normalize by D65 white point
    xyz_n = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_n
    
    # XYZ to Lab
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1/3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16/116)
    
    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])
    
    return (L, a, b)


def ciede2000(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float]
) -> float:
    """
    Compute CIEDE2000 color difference.
    
    Returns:
        ΔE value (lower = more similar)
        < 1: imperceptible
        1-2: perceptible through close observation  
        2-10: perceptible at a glance
        > 10: colors are different
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    dL = L2 - L1
    dC = C2_prime - C1_prime
    
    dh = h2_prime - h1_prime
    if abs(dh) > 180:
        if dh > 0:
            dh -= 360
        else:
            dh += 360
    
    dH = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh / 2))
    
    L_bar = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    h_bar = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > 180:
        h_bar += 180
    
    T = (1 - 0.17 * np.cos(np.radians(h_bar - 30))
         + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6))
         - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    
    dTheta = 30 * np.exp(-((h_bar - 275) / 25)**2)
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    
    S_L = 1 + (0.015 * (L_bar - 50)**2) / np.sqrt(20 + (L_bar - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    
    R_T = -np.sin(np.radians(2 * dTheta)) * R_C
    
    dE = np.sqrt(
        (dL / S_L)**2 +
        (dC / S_C)**2 +
        (dH / S_H)**2 +
        R_T * (dC / S_C) * (dH / S_H)
    )
    
    return dE


# === Evaluation Result ===

@dataclass
class EvalResult:
    """Single example evaluation result."""
    idx: int
    gt_materials: List[str]
    gt_thicknesses: List[int]
    gt_sRGB: List[int]
    pred_materials: List[str]
    pred_thicknesses: List[int]
    pred_sRGB: Optional[List[int]]
    ciede2000: Optional[float]
    stop_reason: str  # 'EOS', 'ERROR', 'MAX_LEN'
    is_incorrect: bool
    n_layers_gt: int
    n_layers_pred: int


def evaluate_example(
    model: ThinFilmTransformer,
    example: Any,  # TrainingExample
    device: torch.device,
    idx: int,
    compute_color: bool = False,
    simulator: Optional[Any] = None
) -> EvalResult:
    """Evaluate a single example using autoregressive generation."""
    
    # Generate prediction
    structure, tokens, stop_reason = model.generate(example.rgb, device)
    
    # Decode predicted structure
    pred_materials, pred_thicknesses = decode_structure_matrix(structure)
    
    # Ground truth
    gt_materials = example.target_materials
    gt_thicknesses = example.target_thicknesses
    gt_sRGB = denormalize_rgb(example.rgb)
    
    # Compute color if requested and possible
    pred_sRGB = None
    ciede = None
    
    if compute_color and simulator and len(pred_materials) > 0:
        try:
            pred_sRGB = simulator.compute_color(pred_materials, pred_thicknesses)
            ciede = ciede2000(sRGB_to_Lab(gt_sRGB), sRGB_to_Lab(pred_sRGB))
        except Exception as e:
            print(f"[WARN] Color computation failed for example {idx}: {e}")
    
    return EvalResult(
        idx=idx,
        gt_materials=gt_materials,
        gt_thicknesses=gt_thicknesses,
        gt_sRGB=gt_sRGB,
        pred_materials=pred_materials,
        pred_thicknesses=pred_thicknesses,
        pred_sRGB=pred_sRGB,
        ciede2000=ciede,
        stop_reason=stop_reason,
        is_incorrect=example.is_incorrect,
        n_layers_gt=example.num_layers,
        n_layers_pred=len(pred_materials)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate thin-film transformer')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--output', type=str, default='outputs/eval_results.json')
    parser.add_argument('--compute-color', action='store_true',
                        help='Compute CIEDE2000 color difference (requires jaxlayerlumos)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load model
    checkpoint_dir = Path(args.checkpoint)
    
    with open(checkpoint_dir / 'config.json') as f:
        config_dict = json.load(f)
    
    config = ModelConfig.from_dict(config_dict)
    model = ThinFilmTransformer(config).to(device)
    
    state_dict = torch.load(checkpoint_dir / 'model.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"[INFO] Loaded model from {checkpoint_dir}")
    print(f"[INFO] Vocabulary size: {config.vocab_size}")
    
    # Load data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        repo_root = find_repo_root()
        data_dir = repo_root / 'create_dataset' / 'data_prompts'
    
    dataset = ThinFilmDataset(
        data_prompts_dir=data_dir,
        seed=args.seed,
        split=args.split
    )
    
    # Optional: optical simulator for color computation
    simulator = None
    if args.compute_color:
        try:
            from jaxlayerlumos import OpticalSimulator
            simulator = OpticalSimulator()
            print("[INFO] Optical simulator loaded for color computation")
        except ImportError:
            print("[WARN] jaxlayerlumos not available, skipping color computation")
    
    # Evaluate
    results = []
    
    for idx, example in enumerate(dataset):
        if args.max_examples and idx >= args.max_examples:
            break
        
        result = evaluate_example(
            model, example, device, idx,
            compute_color=args.compute_color,
            simulator=simulator
        )
        results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Evaluated {idx + 1} examples")
    
    # Compute metrics
    n_total = len(results)
    n_correct_termination = sum(1 for r in results if 
        (r.is_incorrect and r.stop_reason == 'ERROR') or
        (not r.is_incorrect and r.stop_reason in ['EOS', 'MAX_LEN']))
    
    n_exact_match = sum(1 for r in results if 
        r.pred_materials == r.gt_materials and
        r.pred_thicknesses == r.gt_thicknesses)
    
    n_correct_layers = sum(1 for r in results if r.n_layers_pred == r.n_layers_gt)
    
    ciede_values = [r.ciede2000 for r in results if r.ciede2000 is not None]
    
    metrics = {
        'n_examples': n_total,
        'termination_accuracy': n_correct_termination / n_total if n_total > 0 else 0,
        'exact_match': n_exact_match / n_total if n_total > 0 else 0,
        'layer_count_accuracy': n_correct_layers / n_total if n_total > 0 else 0,
        'mean_ciede2000': float(np.mean(ciede_values)) if ciede_values else None,
        'median_ciede2000': float(np.median(ciede_values)) if ciede_values else None,
    }
    
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': [asdict(r) for r in results]
        }, f, indent=2)
    
    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
```

---

## 5. Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ThinFilmTransformer                              │
│                    (Single-Pass Causal Training)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUTS:                                                                 │
│  ┌──────────────────┐    ┌─────────────────────────────────────┐        │
│  │ RGB [B, 3]       │    │ Structure Matrix [B, 25, 8]         │        │
│  │ (normalized 0-1) │    │ (materials × layers, normalized)    │        │
│  └────────┬─────────┘    └─────────────────┬───────────────────┘        │
│           │                                 │                            │
│           ▼                                 ▼                            │
│  ┌────────────────────┐    ┌─────────────────────────────────────┐      │
│  │ rgb_proj           │    │ layer_proj (per column)             │      │
│  │ Linear(3 → d)      │    │ Linear(25 → d) × 8 columns          │      │
│  └────────┬───────────┘    └─────────────────┬───────────────────┘      │
│           │                                   │                          │
│           │ [B, 1, d]                        │ [B, 8, d]                 │
│           │                                   │                          │
│           └──────────────┬───────────────────┘                          │
│                          │ concat                                        │
│                          ▼                                               │
│              ┌───────────────────────────┐                              │
│              │ Token Sequence [B, 9, d]  │                              │
│              │ Pos: [0,   1,  2,  3,  4,  5,  6,  7,  8]                │
│              │      [RGB, L0, L1, L2, L3, L4, L5, L6, L7]               │
│              └───────────────┬───────────┘                              │
│                              │                                           │
│                              ▼                                           │
│              ┌───────────────────────────┐                              │
│              │ Positional Encoding       │                              │
│              │ (sinusoidal, fixed)       │                              │
│              └───────────────┬───────────┘                              │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Causal Transformer Encoder (N layers)                 │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │ Causal Mask (upper triangular):                             │  │  │
│  │  │                                                              │  │  │
│  │  │   Position can attend to:                                   │  │  │
│  │  │   Pos 0 (RGB): only itself                                  │  │  │
│  │  │   Pos 1 (L0):  RGB                     → predicts layer 0   │  │  │
│  │  │   Pos 2 (L1):  RGB, L0                 → predicts layer 1   │  │  │
│  │  │   Pos 3 (L2):  RGB, L0, L1             → predicts layer 2   │  │  │
│  │  │   Pos 4 (L3):  RGB, L0, L1, L2         → predicts layer 3   │  │  │
│  │  │   ...                                                        │  │  │
│  │  │   Pos 8 (L7):  RGB, L0, L1, ..., L6    → predicts layer 7   │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  Each layer: Self-Attention → LayerNorm → FFN → LayerNorm         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              │ [B, 9, d]                                │
│                              │                                           │
│                              ▼                                           │
│              ┌───────────────────────────┐                              │
│              │ Extract positions 1-8     │                              │
│              │ (discard RGB position 0)  │                              │
│              │ [B, 8, d]                 │                              │
│              └───────────────┬───────────┘                              │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Shared Output Head                              │  │
│  │         (Applied independently to each position)                   │  │
│  │                                                                    │  │
│  │    ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐     │  │
│  │    │ Pos 1   │   │ Pos 2   │   │ Pos 3   │  ...  │ Pos 8   │     │  │
│  │    │ [B, d]  │   │ [B, d]  │   │ [B, d]  │       │ [B, d]  │     │  │
│  │    └────┬────┘   └────┬────┘   └────┬────┘       └────┬────┘     │  │
│  │         │             │             │                  │          │  │
│  │         ▼             ▼             ▼                  ▼          │  │
│  │    ┌─────────────────────────────────────────────────────────┐   │  │
│  │    │     Linear(d→d) → GELU → Dropout → Linear(d→1002)       │   │  │
│  │    │                    (shared weights)                      │   │  │
│  │    └─────────────────────────────────────────────────────────┘   │  │
│  │         │             │             │                  │          │  │
│  │         ▼             ▼             ▼                  ▼          │  │
│  │    ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐     │  │
│  │    │ [B,1002]│   │ [B,1002]│   │ [B,1002]│       │ [B,1002]│     │  │
│  │    │ logits  │   │ logits  │   │ logits  │       │ logits  │     │  │
│  │    └─────────┘   └─────────┘   └─────────┘       └─────────┘     │  │
│  │                                                                    │  │
│  │    Positions "diverge" here - each makes independent prediction   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│              ┌───────────────────────────┐                              │
│              │ Output: [B, 8, 1002]      │                              │
│              │                           │                              │
│              │ logits[b, i, :] = probs   │                              │
│              │ for position i+1 to       │                              │
│              │ predict layer i token     │                              │
│              └───────────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tensor Shapes Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| `rgb` | `[B, 3]` | Normalized RGB input |
| `structure_matrix` | `[B, 25, 8]` | Full ground truth structure |
| `tokens` | `[B, 9, d_model]` | Token sequence after projection |
| `causal_mask` | `[9, 9]` | Upper triangular mask (True = masked) |
| `encoded` | `[B, 9, d_model]` | Transformer output |
| `layer_outputs` | `[B, 8, d_model]` | Positions 1-8 only |
| `logits` | `[B, 8, 1002]` | Output over unified vocabulary |
| `targets` | `[B, 8]` | Target tokens (with IGNORE_INDEX padding) |

Where: B=batch, d_model=256 (default)

---

## 6. Key Algorithms

### 6.1 Token Encoding/Decoding

```python
# Vocabulary layout (1002 tokens total):
# Tokens 0-999: (material, thickness) pairs
# Token 1000: ERROR
# Token 1001: EOS

# Encoding: (material, thickness) -> token_id
def encode_layer(material: str, thickness_nm: int) -> int:
    mat_idx = MATERIAL_TO_IDX[material]  # 0-24
    thick_idx = (thickness_nm - 5) // 5   # 0-39
    return mat_idx * 40 + thick_idx       # 0-999

# Decoding: token_id -> (material, thickness)
def decode_token(token_id: int):
    if token_id == 1000: return None, None, 'ERROR'
    if token_id == 1001: return None, None, 'EOS'
    
    mat_idx = token_id // 40
    thick_idx = token_id % 40
    return MATERIALS[mat_idx], 5 + thick_idx * 5, 'LAYER'

# Examples:
# ('Ag', 5)   -> 0 * 40 + 0  = 0
# ('Ag', 200) -> 0 * 40 + 39 = 39
# ('Al', 5)   -> 1 * 40 + 0  = 40
# ('ZnO', 200)-> 24 * 40 + 39 = 999
```

### 6.2 Causal Masking

```python
# Create causal mask for sequence length T
# True = position is masked (cannot attend)
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

# Result for T=9:
#     0  1  2  3  4  5  6  7  8
# 0 [ F  T  T  T  T  T  T  T  T ]  RGB sees only itself
# 1 [ F  F  T  T  T  T  T  T  T ]  L0 sees RGB
# 2 [ F  F  F  T  T  T  T  T  T ]  L1 sees RGB, L0
# 3 [ F  F  F  F  T  T  T  T  T ]  L2 sees RGB, L0, L1
# ...
# 8 [ F  F  F  F  F  F  F  F  F ]  L7 sees all previous
```

### 6.3 Single-Pass Training

```python
# Training: one forward pass predicts all positions
logits = model(rgb, structure_matrix)  # [B, 8, 1002]

# Targets include IGNORE_INDEX for positions after EOS
# Example 3-layer structure: targets = [tok0, tok1, tok2, EOS, -100, -100, -100, -100]

loss = F.cross_entropy(
    logits.view(-1, 1002), 
    targets.view(-1), 
    ignore_index=IGNORE_INDEX
)
```

### 6.4 Autoregressive Inference

```python
# Inference: still autoregressive, but efficient
structure = torch.zeros(25, 8)

for step in range(8):
    logits = model(rgb, structure)  # [1, 8, 1002]
    token_id = logits[0, step, :].argmax()  # Prediction for current step
    
    if token_id == ERROR_TOKEN or token_id == EOS_TOKEN:
        break
    
    # Decode and update structure for next iteration
    material, thickness, _ = decode_token(token_id)
    mat_idx = MATERIAL_TO_IDX[material]
    structure[mat_idx, step] = thickness / MAX_THICKNESS
```

### 6.5 Target Sequence Construction

```python
def build_target_sequence(materials, thicknesses, is_incorrect):
    targets = torch.full((8,), IGNORE_INDEX, dtype=torch.long)
    
    if is_incorrect:
        targets[0] = ERROR_TOKEN  # Position 1 predicts ERROR
    else:
        for i, (mat, thick) in enumerate(zip(materials, thicknesses)):
            targets[i] = encode_layer(mat, thick)
        
        # Position after last layer predicts EOS
        if len(materials) < 8:
            targets[len(materials)] = EOS_TOKEN
        else:
            # 8-layer structure: last position predicts layer 7 (no EOS)
            # Or alternatively: targets[7] = EOS_TOKEN and skip last layer
            pass  # Current design: 8-layer structures don't explicitly predict EOS
    
    return targets
```

---

## 7. Type Reference

### Common Type Aliases

```python
from typing import List, Dict, Tuple, Optional

# Token types
TokenID = int          # 0-1001
MaterialThicknessToken = int  # 0-999 (material-thickness pairs)
ControlToken = int     # 1000 (ERROR) or 1001 (EOS)

# Indices
MaterialIndex = int    # 0-24
ThicknessIndex = int   # 0-39 (maps to 5-200nm in 5nm steps)
LayerIndex = int       # 0-7
PositionIndex = int    # 0-8 (0=RGB, 1-8=layers)

# Normalized values
NormalizedRGB = List[float]      # [r, g, b], 0-1
NormalizedThickness = float      # 0-1

# Raw values  
RawRGB = List[int]               # [R, G, B], 0-255
RawThickness = int               # 5, 10, 15, ..., 200 nm
```

### Tensor Dtypes

| Usage | Dtype |
|-------|-------|
| RGB, structure matrix | `torch.float32` |
| Token IDs (targets) | `torch.long` |
| Causal mask | `torch.bool` |
| Model logits | `torch.float32` |

---

## 8. Dependencies

### Core Requirements

```
torch>=2.0
pyarrow>=12.0
numpy>=1.24
```

### Optical Simulation (optional, for evaluation)

```
jax>=0.4
jaxlib>=0.4
jaxlayerlumos
scipy>=1.10
```

### Environment Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch pyarrow numpy

# Optional: for color evaluation
pip install jax jaxlib scipy
pip install jaxlayerlumos
```

---

## 9. Usage Examples

### Training

```bash
# Basic training
python scripts/training.py \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 8 \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 10 \
  --save-dir data/checkpoints/run1

# Quick test with smaller model
python scripts/training.py \
  --d-model 128 \
  --n-layers 2 \
  --epochs 1 \
  --verbose
```

### Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py \
  --checkpoint data/checkpoints/run1/latest \
  --split validation

# With color metrics
python scripts/evaluate.py \
  --checkpoint data/checkpoints/run1/latest \
  --compute-color \
  --output outputs/eval_with_color.json
```

### Python API

```python
from src.model import ThinFilmTransformer, ModelConfig
from src.materials_vocab import normalize_rgb, decode_structure_matrix

# Create model
config = ModelConfig(d_model=256, n_layers=4, n_heads=8)
model = ThinFilmTransformer(config)

# Generate structure for a target color
rgb = normalize_rgb([255, 100, 50])  # Orange-ish
structure, tokens, stop_reason = model.generate(rgb, device)

# Decode result
materials, thicknesses = decode_structure_matrix(structure)
print(f"Generated: {list(zip(materials, thicknesses))}")
print(f"Stop reason: {stop_reason}")
```

---

## Appendix: Migration from Original CHROMA

| Original Component | Simplified Replacement |
|--------------------|----------------------|
| TinyLlama encoder | Removed |
| Natural language prompts | RGB values [B, 3] |
| Cross-attention layers | Removed (causal self-attention only) |
| 1004-token vocabulary (with PAD, BOS) | 1002 tokens (no PAD, no BOS) |
| Sharp attention (learnable temp) | Standard attention |
| Multi-pass autoregressive training | **Single-pass causal training** |
| Complex checkpointing | Simple checkpoint save/load |
| SLURM scripts | Removed (add as needed) |
| LoRA adapters | Removed |

### Key Design Changes

1. **Single-pass training:** ~8x speedup by predicting all positions in one forward pass
2. **Causal masking:** Prevents data leakage - each position only sees RGB + previous layers
3. **Divergent prediction:** After attention, positions independently predict through shared head
4. **Same vocabulary:** 25 materials × 40 thicknesses = 1000 pairs + ERROR + EOS = 1002 tokens
5. **Simple cross-entropy:** Loss computed over all valid positions with ignore_index for padding
6. **Minimal dependencies:** Only PyTorch and PyArrow required

### Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Forward passes | 1 per example | Up to 8 per example |
| Output used | All 8 positions | One position at a time |
| Structure input | Full ground truth | Incrementally built |
| Parallelism | Fully parallel | Sequential |
