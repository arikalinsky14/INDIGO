"""
Flexible-Material Dataset
=========================

Streaming RGB → (slot_indices, thicknesses) dataset where each example
carries its own material pool. The pool is stored with each row so the
model never sees a fixed material vocabulary.

Mirrors the structure of the original CHROMA-Lite `ThinFilmDataset`:
deterministic file scanning, deterministic 99.5/0.5 train/validation
split, worker-safe sharding for `num_workers > 0`. The schema is
different — see §6.1 of the build spec — but the iteration logic is the
same.

On-disk parquet schema (per row)
--------------------------------
| column              | type                 | meaning                                            |
|---------------------|----------------------|----------------------------------------------------|
| rgb_R               | str (JSON)           | '[R, G, B]' integers in 0-255, the target colour   |
| pool_size           | int                  | Valid materials in the pool, ∈ [n_layers, M_MAX]   |
| pool_n              | list<list<float>>    | shape [pool_size, NUM_LAMBDA=128]                  |
| pool_k              | list<list<float>>    | shape [pool_size, NUM_LAMBDA=128]                  |
| pool_names          | list<string>         | length pool_size, debugging only                   |
| pool_sources        | list<string>         | e.g. 'jaxlayerlumos' / 'synthetic_lorentz'         |
| layer_slots         | list<int>            | length num_layers, slot index per layer            |
| layer_thicknesses   | list<int>            | length num_layers, thickness in nm                 |
| num_layers          | int                  | 1 to MAX_LAYERS=8                                  |

Pools are stored unpadded (pool_size, NUM_LAMBDA). Padding to M_MAX
happens at collate time inside the training script.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.material_features import NUM_LAMBDA, MaterialNK
from src.materials_vocab import normalize_rgb


_ANGLE_DIR_RX = re.compile(r"angle_(\d+)_substrate_.*", re.IGNORECASE)
_SHARD_RX = re.compile(r"shard_(\d+)\.parquet$", re.IGNORECASE)


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class FileMeta:
    """Metadata for a single parquet shard.

    Layer count is per-row in the new schema, so it doesn't appear here.
    """

    file_id: int
    path: str
    incidence_angle: int
    shard_id: int
    nrows: int


@dataclass
class TrainingExample:
    """One (RGB, pool, structure) example yielded by FlexThinFilmDataset.

    Attributes
    ----------
    rgb : torch.Tensor, [3]
        Normalised in [0, 1].
    pool : list of MaterialNK
        Length equals pool_size; unpadded. The collate function pads to
        M_MAX before batching.
    target_slots : list of int
        Slot indices into `pool` for each layer.
    target_thicknesses : list of int
        Thickness in nm for each layer (must align with target_slots).
    """

    rgb: torch.Tensor
    pool: List[MaterialNK]
    target_slots: List[int]
    target_thicknesses: List[int]


# ============================================================================
# Repo / file discovery utilities
# ============================================================================


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward looking for the create_dataset/ marker."""
    current = start or Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "create_dataset").exists():
            return parent
    raise FileNotFoundError("Could not find repository root")


def scan_files(data_prompts_dir: Path) -> List[FileMeta]:
    """Enumerate `layers_N_angle_A_substrate_S/seed_X.parquet` files.

    Identical convention to the original CHROMA-Lite scan_files. The
    incorrect_prompts/ subdirectory branch from the original is dropped
    because INDIGO has no incorrect-prompt pipeline.
    """
    files: List[FileMeta] = []
    file_id = 0

    if not data_prompts_dir.exists():
        return files

    for angle_dir in sorted(data_prompts_dir.iterdir()):
        if not angle_dir.is_dir():
            continue
        match = _ANGLE_DIR_RX.match(angle_dir.name)
        if not match:
            continue

        incidence_angle = int(match.group(1))

        for pq_file in sorted(angle_dir.glob("shard_*.parquet")):
            shard_match = _SHARD_RX.search(pq_file.name)
            if not shard_match:
                continue
            try:
                nrows = pq.ParquetFile(pq_file).metadata.num_rows
            except Exception as exc:
                print(f"[WARN] Could not read parquet metadata for {pq_file}: {exc}")
                continue

            files.append(FileMeta(
                file_id=file_id,
                path=str(pq_file),
                incidence_angle=incidence_angle,
                shard_id=int(shard_match.group(1)),
                nrows=nrows,
            ))
            file_id += 1
    return files


def make_permutation(N: int, seed: int) -> torch.Tensor:
    """Deterministic permutation of [0, N) under the given seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.argsort(torch.rand(N, generator=g), stable=True)


# ============================================================================
# Row → TrainingExample reconstruction
# ============================================================================


_REQUIRED_COLUMNS = (
    "rgb_R", "pool_size", "pool_n", "pool_k", "pool_names", "pool_sources",
    "layer_slots", "layer_thicknesses", "num_layers",
)


def _maybe_json(value):
    """JSON-decode a cell if it's stringly typed; otherwise pass through."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_example(row: Dict[str, object]) -> TrainingExample:
    """Reconstruct a TrainingExample from a parquet row dict."""
    rgb_list = _maybe_json(row["rgb_R"])
    pool_size = int(row["pool_size"])
    pool_n = _maybe_json(row["pool_n"])
    pool_k = _maybe_json(row["pool_k"])
    pool_names = _maybe_json(row["pool_names"])
    pool_sources = _maybe_json(row["pool_sources"])
    layer_slots = _maybe_json(row["layer_slots"])
    layer_thicknesses = _maybe_json(row["layer_thicknesses"])

    pool: List[MaterialNK] = []
    for slot in range(pool_size):
        n_arr = np.asarray(pool_n[slot], dtype=np.float64)
        k_arr = np.asarray(pool_k[slot], dtype=np.float64)
        if n_arr.shape != (NUM_LAMBDA,) or k_arr.shape != (NUM_LAMBDA,):
            raise ValueError(
                f"pool slot {slot} has shape n={n_arr.shape} k={k_arr.shape}, "
                f"expected ({NUM_LAMBDA},)"
            )
        pool.append(MaterialNK(
            name=str(pool_names[slot]),
            n=n_arr,
            k=k_arr,
            source=str(pool_sources[slot]),
        ))

    return TrainingExample(
        rgb=normalize_rgb(list(rgb_list)),
        pool=pool,
        target_slots=[int(s) for s in layer_slots],
        target_thicknesses=[int(t) for t in layer_thicknesses],
    )


# ============================================================================
# Streaming dataset
# ============================================================================


class FlexThinFilmDataset(IterableDataset):
    """Streaming RGB → structure dataset with per-example material pools.

    Wraps the same scan_files / make_permutation / 99.5%-train / 0.5%-val
    logic as the original ThinFilmDataset. Each yielded TrainingExample
    carries its own pool (unpadded MaterialNK list). The collate_fn in
    `scripts/training.py` is responsible for padding to M_MAX before
    feeding into the model.
    """

    def __init__(
        self,
        data_prompts_dir: Path,
        seed: int = 42,
        split: str = "train",
        verbose: bool = False,
        limit_examples: Optional[int] = None,
    ):
        self.seed = seed
        self.split = split
        self.verbose = verbose
        self.limit_examples = limit_examples

        data_prompts_dir = Path(data_prompts_dir)
        self.files = scan_files(data_prompts_dir)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_prompts_dir}")

        self._file_by_id = {f.file_id: f for f in self.files}
        total_rows = sum(f.nrows for f in self.files)

        file_ids: List[int] = []
        row_idxs: List[int] = []
        for f in self.files:
            file_ids.extend([f.file_id] * f.nrows)
            row_idxs.extend(range(f.nrows))

        self.file_ids = torch.tensor(file_ids, dtype=torch.int32)
        self.row_idxs = torch.tensor(row_idxs, dtype=torch.int32)

        perm = make_permutation(total_rows, seed)
        splits = {"train": (0.0, 0.995), "validation": (0.995, 1.0)}
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; expected one of {list(splits)}")
        start_frac, end_frac = splits[split]
        self.order = perm[int(start_frac * total_rows):int(end_frac * total_rows)]

        if limit_examples is not None and limit_examples < len(self.order):
            self.order = self.order[:limit_examples]
            if verbose:
                print(f"[Dataset] Limited to first {limit_examples} examples")

        if verbose:
            print(
                f"[Dataset] {split}: {len(self.order):,} examples "
                f"from {len(self.files)} files"
            )

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[TrainingExample]:
        worker_info = get_worker_info()
        indices = (
            self.order
            if worker_info is None
            else self.order[worker_info.id::worker_info.num_workers]
        )

        file_to_positions: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices.tolist()):
            fid = self.file_ids[global_idx].item()
            row_idx = self.row_idxs[global_idx].item()
            file_to_positions.setdefault(fid, []).append((local_idx, row_idx))

        results: Dict[int, TrainingExample] = {}
        for fid, positions in file_to_positions.items():
            file_meta = self._file_by_id[fid]
            try:
                table = pq.read_table(file_meta.path, columns=list(_REQUIRED_COLUMNS))
            except Exception as exc:
                print(f"[WARN] Could not read {file_meta.path}: {exc}")
                continue

            for local_idx, row_idx in positions:
                try:
                    row = {col: table[col][row_idx].as_py() for col in table.column_names}
                    results[local_idx] = _row_to_example(row)
                except Exception as exc:
                    print(
                        f"[WARN] Skipping row {row_idx} of "
                        f"{file_meta.path}: {exc}"
                    )
                    continue

        for i in range(len(indices)):
            if i in results:
                yield results[i]


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    """In-memory smoke test: build a fake row in the new schema, run it through
    _row_to_example, and confirm a forward pass through the model works."""
    import torch
    from src.material_features import (
        load_jll_directory,
        featurize_pool,
        pad_pool_features,
    )
    from src.materials_vocab import (
        M_MAX,
        build_structure_matrix,
        encode_layer,
    )
    from src.model import FlexMaterialMLP, ModelConfig, compute_loss

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        print("[smoke] No JLL materials directory; skipping forward-pass test.")
        raise SystemExit(0)

    real = load_jll_directory(materials_dir)
    pool_materials = [real["Ag"], real["SiO2"], real["TiO2"], real["Al2O3"]]

    fake_row = {
        "rgb_R": json.dumps([120, 80, 200]),
        "pool_size": len(pool_materials),
        "pool_n": [m.n.tolist() for m in pool_materials],
        "pool_k": [m.k.tolist() for m in pool_materials],
        "pool_names": [m.name for m in pool_materials],
        "pool_sources": [m.source for m in pool_materials],
        "layer_slots": [0, 2, 1],
        "layer_thicknesses": [50, 100, 75],
        "num_layers": 3,
    }
    example = _row_to_example(fake_row)
    print(
        f"[smoke] Reconstructed example: rgb={example.rgb.tolist()}, "
        f"pool_size={len(example.pool)}, "
        f"slots={example.target_slots}, thicknesses={example.target_thicknesses}"
    )

    cfg = ModelConfig(d_model=128, n_layers=2, encoder_hidden=32, encoder_out=16, dropout=0.0)
    model = FlexMaterialMLP(cfg)

    pool_feats_unpadded = featurize_pool(example.pool, mode=cfg.feature_mode)
    pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
    structure = build_structure_matrix(example.target_slots[:2], example.target_thicknesses[:2])
    target_token = encode_layer(example.target_slots[2], example.target_thicknesses[2])

    batch = {
        "rgb": example.rgb.unsqueeze(0),
        "pool_features": pool_feats.unsqueeze(0),
        "pool_mask": pool_mask.unsqueeze(0),
        "structure_matrix": structure.unsqueeze(0),
        "pool_size": torch.tensor([len(example.pool)], dtype=torch.long),
        "target_token": torch.tensor([target_token], dtype=torch.long),
    }
    out = compute_loss(model, batch)
    print(f"[smoke] Forward pass: loss={out['loss'].item():.4f}, acc={out['accuracy'].item():.4f}")
    print("[smoke] OK")
