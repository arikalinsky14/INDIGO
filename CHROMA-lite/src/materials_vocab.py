"""
Materials Vocabulary Module

Vocabulary Layout (1001 base tokens, 1002 with ERROR):
- Tokens 0-999: (material, thickness) pairs
  - Token ID = material_idx * 40 + thickness_idx
  - where thickness_idx = (thickness_nm - 5) / 5
- Token 1000: EOS (design complete)
- Token 1001: ERROR (invalid/impossible request) — used by train_full_model only

Backward compatibility:
  VOCAB_SIZE = 1001          (unchanged — used by pretrain_rgb_to_structure)
  VOCAB_SIZE_WITH_ERROR = 1002  (new — used by train_full_model)
"""

from typing import List, Dict, Tuple, Optional
import torch

# === Material Constants ===
MATERIALS: List[str] = [
    'Ag', 'Al', 'Al2O3', 'Au', 'AZO', 'Cr', 'GaAs', 'GaInP', 'GaP', 'Ge',
    'InP', 'ITO', 'Mn', 'Ni', 'Pd', 'Pt', 'Si3N4', 'SiO2', 'Ti', 'TiN',
    'TiO2', 'aSi', 'cSi', 'W', 'ZnO'
]
THICKNESSES: List[int] = list(range(5, 201, 5))  # [5, 10, ..., 200]

NUM_MATERIALS: int = 25
NUM_THICKNESSES: int = 40
MAX_LAYERS: int = 8
MAX_THICKNESS: int = 200

# Special tokens
EOS_TOKEN: int = 1000
ERROR_TOKEN: int = 1001                   # NEW — invalid/impossible request

# Vocabulary sizes
VOCAB_SIZE: int = 1001                    # UNCHANGED — 1000 layer tokens + EOS
VOCAB_SIZE_WITH_ERROR: int = 1002         # NEW — + ERROR token for full model

# Lookup dictionaries
MATERIAL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(MATERIALS)}
IDX_TO_MATERIAL: Dict[int, str] = {i: m for i, m in enumerate(MATERIALS)}


def encode_layer(material: str, thickness_nm: int) -> int:
    """Convert (material, thickness) pair to token ID (0-999)."""
    mat_idx = MATERIAL_TO_IDX[material]
    thick_idx = (thickness_nm - 5) // 5
    return mat_idx * NUM_THICKNESSES + thick_idx


def decode_token(token_id: int) -> Tuple[Optional[str], Optional[int], str]:
    """Convert token ID to (material, thickness, token_type)."""
    if token_id == EOS_TOKEN:
        return None, None, 'EOS'
    if token_id == ERROR_TOKEN:
        return None, None, 'ERROR'
    mat_idx = token_id // NUM_THICKNESSES
    thick_idx = token_id % NUM_THICKNESSES
    return IDX_TO_MATERIAL[mat_idx], THICKNESSES[thick_idx], 'LAYER'


def is_eos_token(token_id: int) -> bool:
    """Check if token is EOS."""
    return token_id == EOS_TOKEN


def is_error_token(token_id: int) -> bool:
    """Check if token is ERROR (invalid/impossible request)."""
    return token_id == ERROR_TOKEN


def normalize_rgb(rgb: List[int]) -> torch.Tensor:
    """Convert RGB [0-255] to normalized tensor [0-1]."""
    return torch.tensor([c / 255.0 for c in rgb], dtype=torch.float32)


def denormalize_rgb(rgb_norm: torch.Tensor) -> List[int]:
    """Convert normalized RGB [0-1] back to [0-255]."""
    return [int(round(c.item() * 255)) for c in rgb_norm]


def normalize_thickness(thickness_nm: int) -> float:
    """Normalize thickness from nm to 0-1 range."""
    return thickness_nm / MAX_THICKNESS


def denormalize_thickness(thickness_norm: float) -> int:
    """Convert normalized thickness back to nm, snapped to valid values."""
    thickness_nm = round(thickness_norm * MAX_THICKNESS / 5) * 5
    return max(5, min(200, int(thickness_nm)))


def build_structure_matrix(materials: List[str], thicknesses: List[int]) -> torch.Tensor:
    """Build structure matrix [25, 8] from material-thickness lists."""
    matrix = torch.zeros(NUM_MATERIALS, MAX_LAYERS, dtype=torch.float32)
    for layer_idx, (mat, thick) in enumerate(zip(materials, thicknesses)):
        if layer_idx >= MAX_LAYERS:
            break
        matrix[MATERIAL_TO_IDX[mat], layer_idx] = normalize_thickness(thick)
    return matrix


def decode_structure_matrix(matrix: torch.Tensor) -> Tuple[List[str], List[int]]:
    """Decode structure matrix back to material-thickness lists."""
    materials, thicknesses = [], []
    for layer_idx in range(MAX_LAYERS):
        col = matrix[:, layer_idx]
        if col.sum() == 0:
            break
        mat_idx = col.argmax().item()
        if col[mat_idx].item() > 0:
            materials.append(IDX_TO_MATERIAL[mat_idx])
            thicknesses.append(denormalize_thickness(col[mat_idx].item()))
    return materials, thicknesses