"""
Flexible-Material Dataset
=========================

Streaming RGB → (slot_indices, thicknesses) dataset where each example
carries its own material pool. The pool is stored with each row so the
model never sees a fixed material vocabulary.

Mirrors the structure of the original CHROMA-Lite `ThinFilmDataset`:
deterministic file scanning, deterministic 99.95/0.05 train/validation
split (5k rows out of 10M — enough for a low-variance loss estimate,
small enough that the training loss is essentially unaffected), and
worker-safe sharding for `num_workers > 0`. The schema is
different — see §6.1 of the build spec — but the iteration logic is the
same.

On-disk parquet schema (per row)
--------------------------------
| column              | type                 | meaning                                            |
|---------------------|----------------------|----------------------------------------------------|
| lab                 | str (JSON)           | '[L*, a*, b*]' floats in CIE Lab, the target colour|
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
from collections.abc import Sequence
from typing import Dict, Iterator, List, Optional, Tuple

from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.material_features import (
    NUM_LAMBDA,
    MaterialNK,
    materialnk_validation_disabled,
)
from src.materials_vocab import normalize_lab


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
    """One (Lab target, pool, structure) example yielded by FlexThinFilmDataset.

    Attributes
    ----------
    lab : torch.Tensor, [3]
        CIE Lab target normalised by `normalize_lab` from `materials_vocab`.
        Replaces the legacy RGB target — Lab is wider gamut and perceptually
        uniform (so CIEDE2000 distances are meaningful).
    pool : list of MaterialNK
        Length equals pool_size; unpadded. The collate function pads to
        M_MAX before batching.
    target_slots : list of int
        Slot indices into `pool` for each layer.
    target_thicknesses : list of int
        Thickness in nm for each layer (must align with target_slots).
    """

    lab: torch.Tensor
    pool: List[MaterialNK]
    target_slots: List[int]
    target_thicknesses: List[int]
    # Optional: 'high_chroma_search' or 'random' (or None on legacy
    # parquets that predate the column). Populated for eval-time
    # splits by source; ignored during training.
    structure_source: Optional[str] = None
    # The pool already featurized, [pool_size, 2, NUM_LAMBDA] float32: exactly
    # featurize_pool(pool, "raw_spectrum"), computed for a whole shard at once
    # by the streaming reader. The collate uses it when present and falls back
    # to featurizing the pool when not; the values are the same either way.
    pool_features: Optional[torch.Tensor] = None


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


# scan_files opens every shard to read its row count, which is a few thousand
# metadata reads on a network filesystem. A single training run builds three
# datasets (train, val, DeltaE), so without this the same scan runs three
# times and dominates startup. Keyed by resolved path; a run never changes its
# shards underneath itself. Call scan_files.cache_clear() if that ever stops
# being true.
@lru_cache(maxsize=16)
def _scan_files_cached(resolved: str) -> Tuple[FileMeta, ...]:
    return tuple(_scan_files_uncached(Path(resolved)))


def scan_files(data_prompts_dir: Path) -> List[FileMeta]:
    """Cached wrapper. See `_scan_files_uncached` for the real work."""
    return list(_scan_files_cached(str(Path(data_prompts_dir).resolve())))


def _scan_files_uncached(data_prompts_dir: Path) -> List[FileMeta]:
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
    "lab", "pool_size", "pool_n", "pool_k", "pool_names", "pool_sources",
    "layer_slots", "layer_thicknesses", "num_layers",
)
# Columns the loader will pick up if the parquet has them, but that
# older parquets pre-date. Missing values become None on the
# TrainingExample.
_OPTIONAL_COLUMNS = ("structure_source",)


def _columns_for_shard(path: str) -> List[str]:
    """Return the intersection of _OPTIONAL_COLUMNS + _REQUIRED_COLUMNS
    with what's actually in the parquet — so `pq.read_table` never asks
    for a missing column."""
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = list(_REQUIRED_COLUMNS)
    cols.extend(c for c in _OPTIONAL_COLUMNS if c in schema_names)
    return cols


def _maybe_json(value):
    """JSON-decode a cell if it's stringly typed; otherwise pass through."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_example(row: Dict[str, object]) -> TrainingExample:
    """Reconstruct a TrainingExample from a parquet row dict."""
    lab_list = _maybe_json(row["lab"])
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

    structure_source = row.get("structure_source")
    if structure_source is not None:
        structure_source = str(structure_source)

    return TrainingExample(
        lab=normalize_lab(list(lab_list)),
        pool=pool,
        target_slots=[int(s) for s in layer_slots],
        target_thicknesses=[int(t) for t in layer_thicknesses],
        structure_source=structure_source,
    )


class _LazyPool(Sequence):
    """A pool of MaterialNK built on first access.

    Training only ever asks a pool its length (the collate reads the
    precomputed `pool_features`), so building ~18 MaterialNK objects per
    example for it was a quarter of the remaining decode time. Anything that
    indexes or iterates the pool (the DeltaE eval, the simulator) gets the
    same MaterialNK list the per-row decode built: same names, sources and
    float64 n, k values. Pickles as a plain list.
    """

    __slots__ = ("_n", "_k", "_names", "_sources", "_built")

    def __init__(self, n: np.ndarray, k: np.ndarray, names, sources):
        # names, sources: this row's pyarrow list scalars, converted to Python
        # strings only if the pool is ever materialized.
        self._n, self._k = n, k
        self._names, self._sources = names, sources
        self._built: Optional[List[MaterialNK]] = None

    def _materialize(self) -> List[MaterialNK]:
        if self._built is None:
            names = self._names.as_py()
            sources = self._sources.as_py()
            with materialnk_validation_disabled():
                self._built = [
                    MaterialNK(name=str(names[j]), n=self._n[j],
                               k=self._k[j], source=str(sources[j]))
                    for j in range(len(self._n))]
            self._names = self._sources = None
        return self._built

    def __len__(self) -> int:
        return len(self._n)

    def __getitem__(self, i):
        return self._materialize()[i]

    def __iter__(self):
        return iter(self._materialize())

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def __reduce__(self):
        return (list, (self._materialize(),))

    def __repr__(self) -> str:
        return f"_LazyPool({len(self)} materials)"


def _shard_examples_vectorized(table, row_idxs: np.ndarray
                               ) -> Optional[List[TrainingExample]]:
    """Every selected row of one shard, decoded column-wise.

    Returns what `_row_to_example` returns row by row, in the same order and
    with the same values, plus `pool_features`; or None when the shard's
    layout is not the one this path verifies (string-encoded columns, nulls,
    a spectrum not NUM_LAMBDA long, a pool whose length disagrees with
    pool_size), in which case the caller decodes it the old way.

    Why: the per-row path converted every spectrum to Python floats
    (`.as_py()`), back to numpy one material at a time, and featurized each
    material again in the collate. Profiled on the training path, that object
    churn was ~85% of each DataLoader worker's time; the parquet read itself
    was under 10%. Here the spectra stay in numpy, cast to float32 once for
    the features (the same round-to-nearest cast featurize() applies).

    Memory: a sparse selection (the validation split, a few rows per shard)
    is copied out with take(), so what the examples hold is sized to the rows
    kept, not to the shard. A dense one (training) skips that copy; its
    examples live only until their batch is collated.
    """
    row_idxs = np.asarray(row_idxs, dtype=np.int64)
    if len(row_idxs) == 0:
        return []
    dense = 2 * len(row_idxs) >= table.num_rows
    sub = table if dense else table.take(pa.array(row_idxs))
    sel = row_idxs if dense else np.arange(len(row_idxs))
    for name in ("pool_n", "pool_k", "pool_names", "pool_sources",
                 "layer_slots", "layer_thicknesses", "pool_size", "lab"):
        if sub[name].null_count:
            return None

    def spectra(name: str):
        col = sub[name].combine_chunks()
        if not pa.types.is_list(col.type) or not pa.types.is_list(col.type.value_type):
            return None
        outer = col.offsets.to_numpy()
        inner_col = col.flatten()
        if inner_col.null_count:
            return None
        inner = inner_col.offsets.to_numpy()
        if not np.all(np.diff(inner) == NUM_LAMBDA):
            return None
        vals = inner_col.flatten()
        if vals.null_count or not pa.types.is_floating(vals.type):
            return None
        vals = vals.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        return outer - outer[0], vals.reshape(-1, NUM_LAMBDA)

    got_n, got_k = spectra("pool_n"), spectra("pool_k")
    if got_n is None or got_k is None:
        return None
    (off_n, n64), (off_k, k64) = got_n, got_k
    pool_size = sub["pool_size"].to_numpy(zero_copy_only=False).astype(np.int64)
    if not (np.array_equal(np.diff(off_n), pool_size)
            and np.array_equal(off_n, off_k)):
        return None

    names_col = sub["pool_names"].combine_chunks()
    sources_col = sub["pool_sources"].combine_chunks()
    for col in (names_col, sources_col):
        if not pa.types.is_list(col.type):
            return None
        # The per-row decode indexes names[slot] for slot < pool_size.
        if np.any(np.diff(col.offsets.to_numpy()) < pool_size):
            return None
    slots = sub["layer_slots"].to_pylist()
    thicks = sub["layer_thicknesses"].to_pylist()
    labs = sub["lab"].to_pylist()
    srcs = (sub["structure_source"].to_pylist()
            if "structure_source" in sub.column_names else None)
    if not all(isinstance(slots[i], list) and isinstance(thicks[i], list)
               for i in sel):
        return None

    # float32 features in one cast. Dense: every material of the shard (the
    # few unselected rows cost less than gathering around them). Sparse: the
    # table was already cut to the selected rows by take().
    feats = np.empty((n64.shape[0], 2, NUM_LAMBDA), dtype=np.float32)
    feats[:, 0] = n64
    feats[:, 1] = k64
    feats_t = torch.from_numpy(feats)

    out: List[TrainingExample] = []
    for i in sel:
        a, b = int(off_n[i]), int(off_n[i + 1])
        src = None if srcs is None else srcs[i]
        out.append(TrainingExample(
            lab=normalize_lab(list(_maybe_json(labs[i]))),
            pool=_LazyPool(n64[a:b], k64[a:b], names_col[int(i)],
                           sources_col[int(i)]),
            target_slots=[int(x) for x in slots[i]],
            target_thicknesses=[int(x) for x in thicks[i]],
            structure_source=None if src is None else str(src),
            pool_features=feats_t[a:b],
        ))
    return out


#: INDIGO_LEGACY_DECODE=1 forces the old row-by-row decode everywhere: for A/B
#: timing, and as the reference the vectorized path is tested against.
def _legacy_decode() -> bool:
    import os
    return os.environ.get("INDIGO_LEGACY_DECODE", "0") == "1"


def _shard_examples(table, row_idxs, path: str) -> Iterator[TrainingExample]:
    """One shard's selected rows, in `row_idxs` order: vectorized when the
    layout allows, else the original per-row decode (including its per-row
    skip-and-warn on a bad row)."""
    fast = None if _legacy_decode() else _shard_examples_vectorized(table, row_idxs)
    if fast is not None:
        yield from fast
        return
    for row_idx in row_idxs:
        try:
            row = {col: table[col][int(row_idx)].as_py()
                   for col in table.column_names}
            yield _row_to_example(row)
        except Exception as exc:
            print(f"[WARN] Skipping row {int(row_idx)} of {path}: {exc}")
            continue


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
        limit_shard_aligned: bool = False,
        streaming: bool = False,
    ):
        self.seed = seed
        self.split = split
        self.verbose = verbose
        self.limit_examples = limit_examples
        self.limit_shard_aligned = limit_shard_aligned
        self.streaming = streaming

        data_prompts_dir = Path(data_prompts_dir)
        self.files = scan_files(data_prompts_dir)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_prompts_dir}")

        self._file_by_id = {f.file_id: f for f in self.files}
        total_rows = sum(f.nrows for f in self.files)

        # Same int32 values the original Python-list build produced, without
        # materialising 2 x 40M Python ints at startup.
        nrows = np.array([f.nrows for f in self.files], dtype=np.int64)
        self.file_ids = torch.from_numpy(np.repeat(
            np.array([f.file_id for f in self.files], dtype=np.int32), nrows))
        self.row_idxs = torch.from_numpy(
            (np.arange(int(nrows.sum()), dtype=np.int64)
             - np.repeat(np.cumsum(nrows) - nrows, nrows)).astype(np.int32))

        perm = make_permutation(total_rows, seed)
        # "all" reads every row — use for held-out tier_a/tier_b eval
        # sets where there's no train/val split to honour.
        splits = {
            "train":      (0.0,    0.9995),
            "validation": (0.9995, 1.0),
            "all":        (0.0,    1.0),
        }
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; expected one of {list(splits)}")
        start_frac, end_frac = splits[split]
        self.order = perm[int(start_frac * total_rows):int(end_frac * total_rows)]

        if limit_examples is not None and limit_examples < len(self.order):
            if limit_shard_aligned:
                # Take WHOLE SHARDS until we have enough rows, then truncate
                # to exactly limit_examples.
                #
                # Why: `self.order` is a global shuffle, so its first N rows
                # are scattered across every shard. `_iter_streaming` reads a
                # full ~140 MB parquet table per shard it touches, so a small
                # limit spread over 2000 shards reads the ENTIRE corpus to
                # yield a fraction of it -- e.g. 76,800 of 10M rows still
                # reads all 2000 shards, ~280 GB, and does it again every
                # epoch. Restricting to the shards we actually need cuts that
                # by 16x at limit=614,400 and 125x at limit=76,800.
                #
                # This stays a valid random sample: shards are generated from
                # independent seeds (the same argument `_iter_streaming`
                # already relies on to justify emitting rows shard-by-shard
                # rather than in globally-shuffled order).
                order_np = self.order.numpy()
                fids = self.file_ids.numpy()[order_np]
                counts = np.bincount(fids, minlength=len(self.files))
                shard_ids = np.unique(fids)
                np.random.default_rng(seed).shuffle(shard_ids)
                chosen, running = [], 0
                for fid in shard_ids:
                    chosen.append(int(fid))
                    running += int(counts[fid])
                    if running >= limit_examples:
                        break
                keep = np.isin(fids, np.asarray(chosen))
                self.order = self.order[torch.from_numpy(keep)][:limit_examples]
                if verbose:
                    print(f"[Dataset] Limited to {len(self.order):,} examples "
                          f"from {len(chosen)} shard(s) of {len(self.files)} "
                          f"(shard-aligned)")
            else:
                self.order = self.order[:limit_examples]
                if verbose:
                    print(f"[Dataset] Limited to first {limit_examples} examples")

        if verbose:
            print(
                f"[Dataset] {split}: {len(self.order):,} examples "
                f"from {len(self.files)} files"
                + (" (streaming)" if streaming else "")
            )

        if streaming:
            self._build_streaming_index()

    def _build_streaming_index(self) -> None:
        """For each shard, the sorted list of row indices that belong to our
        split. Lets `_iter_streaming` read one shard at a time and emit rows
        in row-position order without an in-memory accumulator.
        """
        order_np = self.order.numpy()
        sel_fids = self.file_ids.numpy()[order_np]
        sel_rows = self.row_idxs.numpy()[order_np]
        # Group by file_id via argsort, then sort within each group by row
        # index so each shard is read sequentially.
        sort_idx = np.argsort(sel_fids, kind="stable")
        sorted_fids = sel_fids[sort_idx]
        sorted_rows = sel_rows[sort_idx]
        boundaries = np.searchsorted(sorted_fids, np.arange(len(self.files) + 1))
        self._split_rows_by_file: Dict[int, np.ndarray] = {}
        for fid in range(len(self.files)):
            s, e = int(boundaries[fid]), int(boundaries[fid + 1])
            if s < e:
                self._split_rows_by_file[fid] = np.sort(sorted_rows[s:e])

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[TrainingExample]:
        if self.streaming:
            yield from self._iter_streaming()
        else:
            yield from self._iter_global_order()

    def _iter_streaming(self) -> Iterator[TrainingExample]:
        """Read shards one at a time in shard_id order, yielding rows that
        belong to our split. No per-epoch results accumulator — memory is
        bounded by ~one parquet table (~140 MB) per worker, regardless of
        dataset size. The trade-off vs `_iter_global_order` is that rows
        are no longer in `self.order`'s globally-shuffled order; they come
        out shard-by-shard. Since shards are generated from independent
        seeds, this is still a random sample of the split distribution.
        """
        worker_info = get_worker_info()
        files_sorted = sorted(self.files, key=lambda f: f.shard_id)
        if worker_info is not None:
            my_files = files_sorted[worker_info.id::worker_info.num_workers]
        else:
            my_files = files_sorted

        with materialnk_validation_disabled():
            for f in my_files:
                row_idxs = self._split_rows_by_file.get(f.file_id)
                if row_idxs is None or len(row_idxs) == 0:
                    continue
                try:
                    table = pq.read_table(f.path, columns=_columns_for_shard(f.path))
                except Exception as exc:
                    print(f"[WARN] Could not read {f.path}: {exc}")
                    continue
                yield from _shard_examples(table, row_idxs, f.path)
                del table  # release ~140 MB before opening the next shard

    def _iter_global_order(self) -> Iterator[TrainingExample]:
        """Original behavior: yields examples in the globally-shuffled
        `self.order` sequence. Builds a per-worker in-memory dict of all
        epoch examples before yielding (so it can re-order across shards),
        which OOMs at production dataset sizes (~40 KB/example × millions).
        Kept for small-dataset compatibility / unit testing — prefer
        `streaming=True` for any production run.
        """
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
        # Parquet rows were validated when generated; skip the per-MaterialNK
        # __post_init__ numpy reductions on this hot path (~9% of pipeline
        # CPU). _row_to_example still enforces the n,k shape guard.
        with materialnk_validation_disabled():
            for fid, positions in file_to_positions.items():
                file_meta = self._file_by_id[fid]
                try:
                    table = pq.read_table(
                        file_meta.path, columns=_columns_for_shard(file_meta.path),
                    )
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
# On-disk cache of a split's rows
# ============================================================================

#: Bump when the cached content or its decoding changes meaning.
SPLIT_CACHE_VERSION = 1


def split_cache_key(ds: FlexThinFilmDataset) -> str:
    """Names exactly which rows a streaming dataset yields, and in what order:
    the split, seed, and every (shard path, row count, selected row indices).
    Two datasets with the same key read the same rows."""
    import hashlib
    if not ds.streaming:
        raise ValueError("the split cache needs a streaming dataset")
    h = hashlib.sha256(f"v{SPLIT_CACHE_VERSION}|{ds.split}|{ds.seed}|".encode())
    for f in sorted(ds.files, key=lambda f: f.shard_id):
        rows = ds._split_rows_by_file.get(f.file_id)
        if rows is None or len(rows) == 0:
            continue
        h.update(f"{f.path}|{f.nrows}|".encode())
        h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    return h.hexdigest()[:20]


def load_split_cached(ds: FlexThinFilmDataset, cache_dir: Optional[Path],
                      verbose: bool = True) -> List[TrainingExample]:
    """`list(ds)`, with the rows it reads kept on disk for the next run.

    The validation split is a few rows from each of ~3,600 shards, so reading
    it means opening nearly half the corpus: the bulk of the ~28-minute
    startup every tuning task paid. Every task of a stage reads the same rows
    (same seed and limit), so the first one writes exactly those rows, in
    iteration order, to one parquet file under cache_dir, and later ones read
    that file instead.

    Identity: the cache holds the raw parquet rows, not decoded objects, and
    is decoded by the same `_shard_examples` the shards themselves go through,
    so a cached run sees the same examples as an uncached one (tested in
    tests/test_fast_data_path.py). The file is named by split_cache_key, so a
    different seed, limit, alignment or corpus can never pick it up. Any
    failure to build or read it falls back to `list(ds)`.
    """
    import os
    if cache_dir is None:
        return list(ds)
    try:
        path = Path(cache_dir) / f"{ds.split}_{split_cache_key(ds)}.parquet"
    except Exception as exc:
        print(f"[WARN] split cache unavailable ({exc}); reading the shards")
        return list(ds)

    table = None
    if path.exists():
        try:
            table = pq.read_table(path)
            if verbose:
                print(f"[INFO] {ds.split}: {table.num_rows:,} rows from the "
                      f"split cache {path}")
        except Exception as exc:
            print(f"[WARN] unreadable split cache {path} ({exc}); rebuilding")
            table = None
    if table is None:
        try:
            parts = []
            for f in sorted(ds.files, key=lambda f: f.shard_id):
                rows = ds._split_rows_by_file.get(f.file_id)
                if rows is None or len(rows) == 0:
                    continue
                t = pq.read_table(f.path, columns=_columns_for_shard(f.path))
                parts.append(t.take(pa.array(np.asarray(rows, dtype=np.int64))))
            table = pa.concat_tables(parts, promote_options="default")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            if verbose:
                print(f"[INFO] {ds.split}: wrote {table.num_rows:,} rows to the "
                      f"split cache {path}")
        except Exception as exc:
            print(f"[WARN] could not build the split cache ({exc}); reading "
                  f"the shards")
            return list(ds)

    with materialnk_validation_disabled():
        return list(_shard_examples(table, np.arange(table.num_rows), str(path)))


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
        "lab": json.dumps([55.0, 22.5, -38.7]),  # arbitrary purple-ish target
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
        f"[smoke] Reconstructed example: lab={example.lab.tolist()}, "
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
        "lab": example.lab.unsqueeze(0),
        "pool_features": pool_feats.unsqueeze(0),
        "pool_mask": pool_mask.unsqueeze(0),
        "structure_matrix": structure.unsqueeze(0),
        "pool_size": torch.tensor([len(example.pool)], dtype=torch.long),
        "target_token": torch.tensor([target_token], dtype=torch.long),
    }
    out = compute_loss(model, batch)
    print(f"[smoke] Forward pass: loss={out['loss'].item():.4f}, acc={out['accuracy'].item():.4f}")
    print("[smoke] OK")
