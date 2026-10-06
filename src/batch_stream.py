"""Training batches built straight from parquet columns, in the same order.

Why this exists
---------------
After the Oct 6 decode rewrite the input pipeline still set the pace for the
small end of the ladder: STAGE=speed (job 4230033) found the 81k-parameter
model waiting on its DataLoader 64% of the time. What was left was per-example
Python: a TrainingExample per row, then a collate that walked every example
and every layer. For a scaling study priced in CRC credits that matters,
because a run's cost is its wall time, and wall time spent on Python
bookkeeping is a property of the code rather than of the model.

Here a DataLoader worker never builds an example. It turns each shard's
selected rows into column arrays once (`ShardColumns`), cuts them into batches
of `batch_size` consecutive rows, and collates each batch with a handful of
numpy scatter operations (`collate_columns`).

Why the batches are the same
----------------------------
With `DataLoader(dataset, batch_size=B, num_workers=W)` over the streaming
FlexThinFilmDataset, worker w reads shards `files_sorted[w::W]` in shard_id
order, groups its own stream into consecutive batches of B rows (the last one
partial), and the main process takes batches from the workers in turn. This
module keeps all of that by construction: `PackedBatchStream` is handed to the
DataLoader with `batch_size=None`, so torch's own worker assignment and
round-robin are untouched; each worker reads the same shards, in the same
order, the same rows within each shard, and cuts the same B-row batches,
carrying a partial batch across a shard boundary exactly as the auto-batching
fetcher does. `tests/test_fast_data_path.py` checks the batch sequence against
the old DataLoader, old decode and old collate, bit for bit.

Values: every array is filled with the same operation the per-example path
applied (float64 spectra cast once to float32, normalize_lab's float64
division then float32, thickness / MAX_THICKNESS_NM then float32,
encode_layer's integer arithmetic), and the old path's ValueErrors are raised
for the same bad inputs.

`INDIGO_BATCH_LOADER=0` switches training back to the per-example DataLoader;
`INDIGO_LEGACY_DECODE=1` still forces the row-by-row parquet decode, whose
examples are then converted to columns.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset import (FlexThinFilmDataset, TrainingExample,
                         _columns_for_shard, _legacy_decode, _row_to_example)
from src.material_features import (NUM_LAMBDA, featurize_pool,
                                   materialnk_validation_disabled)
from src.materials_vocab import (_AB_SCALE, _L_SCALE, _THICKNESS_STEP_NM,
                                 EOS_TOKEN, M_MAX, MAX_LAYERS,
                                 MAX_THICKNESS_NM, NUM_THICKNESSES,
                                 THICKNESSES)

#: Start token + MAX_LAYERS positions, as scripts/training.py packs them.
PACKED_SEQ_LEN = MAX_LAYERS + 1

_LAB_SCALE = np.array([_L_SCALE, _AB_SCALE, _AB_SCALE], dtype=np.float64)


def batch_loader_enabled() -> bool:
    return os.environ.get("INDIGO_BATCH_LOADER", "1") != "0"


@dataclass
class ShardColumns:
    """Selected rows of one shard (or of several, concatenated), as columns.

    Row i's materials are feats[feat_off[i]:feat_off[i+1]] and its layers
    slots/thick[layer_off[i]:layer_off[i+1]].
    """

    lab: np.ndarray          # [n, 3] float32, normalized
    pool_size: np.ndarray    # [n] int64
    feat_off: np.ndarray     # [n+1] int64
    feats: np.ndarray        # [m, 2, NUM_LAMBDA] float32
    layer_off: np.ndarray    # [n+1] int64
    slots: np.ndarray        # [k] int64
    thick: np.ndarray        # [k] int64

    @property
    def n(self) -> int:
        return len(self.pool_size)

    def slice(self, lo: int, hi: int) -> "ShardColumns":
        fa, fb = int(self.feat_off[lo]), int(self.feat_off[hi])
        la, lb = int(self.layer_off[lo]), int(self.layer_off[hi])
        return ShardColumns(
            lab=self.lab[lo:hi], pool_size=self.pool_size[lo:hi],
            feat_off=self.feat_off[lo:hi + 1] - fa, feats=self.feats[fa:fb],
            layer_off=self.layer_off[lo:hi + 1] - la,
            slots=self.slots[la:lb], thick=self.thick[la:lb])

    @staticmethod
    def concat(parts: List["ShardColumns"]) -> "ShardColumns":
        if len(parts) == 1:
            return parts[0]

        def offsets(name):
            out, base = [np.zeros(1, np.int64)], 0
            for p in parts:
                o = getattr(p, name)
                out.append(o[1:] + base)
                base += int(o[-1])
            return np.concatenate(out)

        return ShardColumns(
            lab=np.concatenate([p.lab for p in parts]),
            pool_size=np.concatenate([p.pool_size for p in parts]),
            feat_off=offsets("feat_off"),
            feats=np.concatenate([p.feats for p in parts]),
            layer_off=offsets("layer_off"),
            slots=np.concatenate([p.slots for p in parts]),
            thick=np.concatenate([p.thick for p in parts]))


def _nested_spectra(col) -> Optional[tuple]:
    """(row offsets into materials, [materials, NUM_LAMBDA] float64) or None."""
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


def _int_lists(col) -> Optional[tuple]:
    """(row offsets, int64 values) for a list<integer> column, else None."""
    if not pa.types.is_list(col.type) or not pa.types.is_integer(col.type.value_type):
        return None
    vals = col.flatten()
    if vals.null_count:
        return None
    off = col.offsets.to_numpy()
    return off - off[0], vals.to_numpy(zero_copy_only=False).astype(np.int64)


def shard_columns(table, row_idxs: np.ndarray) -> Optional[ShardColumns]:
    """One shard's selected rows, in row_idxs order, as columns; or None when
    the layout is not the verified one (the caller then decodes row by row)."""
    row_idxs = np.asarray(row_idxs, dtype=np.int64)
    sub = table.take(pa.array(row_idxs))
    for name in ("pool_n", "pool_k", "layer_slots", "layer_thicknesses",
                 "pool_size", "lab", "pool_names", "pool_sources"):
        if sub[name].null_count:
            return None
    got_n = _nested_spectra(sub["pool_n"].combine_chunks())
    got_k = _nested_spectra(sub["pool_k"].combine_chunks())
    if got_n is None or got_k is None:
        return None
    (off_n, n64), (off_k, k64) = got_n, got_k
    pool_size = sub["pool_size"].to_numpy(zero_copy_only=False).astype(np.int64)
    if not (np.array_equal(np.diff(off_n), pool_size)
            and np.array_equal(off_n, off_k)):
        return None
    # The per-row decode reads names[slot], sources[slot] for slot < pool_size
    # and drops the row if one is missing; here the shard falls back instead.
    for name in ("pool_names", "pool_sources"):
        col = sub[name].combine_chunks()
        if not pa.types.is_list(col.type) or np.any(
                np.diff(col.offsets.to_numpy()) < pool_size):
            return None
    sl = _int_lists(sub["layer_slots"].combine_chunks())
    th = _int_lists(sub["layer_thicknesses"].combine_chunks())
    if sl is None or th is None or not np.array_equal(sl[0], th[0]):
        return None

    lab_col = sub["lab"].combine_chunks()
    if pa.types.is_string(lab_col.type) or pa.types.is_large_string(lab_col.type):
        lab64 = np.array([json.loads(s) for s in lab_col.to_pylist()],
                         dtype=np.float64)
    elif pa.types.is_list(lab_col.type):
        lab64 = np.array(lab_col.to_pylist(), dtype=np.float64)
    else:
        return None
    if lab64.shape != (len(row_idxs), 3):
        return None

    feats = np.empty((n64.shape[0], 2, NUM_LAMBDA), dtype=np.float32)
    feats[:, 0] = n64
    feats[:, 1] = k64
    return ShardColumns(
        lab=(lab64 / _LAB_SCALE).astype(np.float32), pool_size=pool_size,
        feat_off=off_n.astype(np.int64), feats=feats,
        layer_off=sl[0].astype(np.int64), slots=sl[1], thick=th[1])


def columns_from_examples(examples: List[TrainingExample]) -> ShardColumns:
    """The same columns, from examples the per-row decode produced."""
    n = len(examples)
    feats = [ex.pool_features if ex.pool_features is not None
             else featurize_pool(ex.pool, mode="raw_spectrum")
             for ex in examples]
    sizes = np.array([len(ex.pool) for ex in examples], dtype=np.int64)
    nl = np.array([len(ex.target_slots) for ex in examples], dtype=np.int64)
    for ex in examples:
        if len(ex.target_slots) != len(ex.target_thicknesses):
            raise ValueError("slot_indices and thicknesses_nm must have same length")
    return ShardColumns(
        lab=(torch.stack([ex.lab for ex in examples]).numpy()
             if n else np.zeros((0, 3), np.float32)),
        pool_size=sizes,
        feat_off=np.concatenate([[0], np.cumsum([f.shape[0] for f in feats])]).astype(np.int64),
        feats=(torch.cat(feats).numpy() if n
               else np.zeros((0, 2, NUM_LAMBDA), np.float32)),
        layer_off=np.concatenate([[0], np.cumsum(nl)]).astype(np.int64),
        slots=np.array([s for ex in examples for s in ex.target_slots], dtype=np.int64),
        thick=np.array([t for ex in examples for t in ex.target_thicknesses],
                       dtype=np.int64))


_VALID_THICK = np.zeros(MAX_THICKNESS_NM + 1, dtype=bool)
_VALID_THICK[np.array(THICKNESSES)] = True


def collate_columns(c: ShardColumns) -> Dict[str, torch.Tensor]:
    """scripts/training.py collate_fn_packed, for a batch held as columns.

    Same keys, dtypes and values; same ValueErrors for the same bad inputs.
    """
    n = c.n
    P = c.pool_size
    if np.any(P > M_MAX):
        raise ValueError(f"Pool size {int(P.max())} exceeds m_max={M_MAX}")
    n_layers = np.diff(c.layer_off)
    if np.any(n_layers > MAX_LAYERS):
        raise ValueError(f"structure has {int(n_layers.max())} layers, "
                         f"max is {MAX_LAYERS}")
    if np.any((c.slots < 0) | (c.slots >= M_MAX)):
        bad = int(c.slots[(c.slots < 0) | (c.slots >= M_MAX)][0])
        raise ValueError(f"slot {bad} out of range [0, {M_MAX})")
    t = c.thick
    ok = (t >= 0) & (t <= MAX_THICKNESS_NM)
    ok[ok] = _VALID_THICK[t[ok]]
    if not np.all(ok):
        raise ValueError(
            f"thickness {int(t[~ok][0])} not in valid grid "
            f"{THICKNESSES[0]}..{THICKNESSES[-1]} step {_THICKNESS_STEP_NM}")

    pool = np.zeros((n, M_MAX, 2, NUM_LAMBDA), dtype=np.float32)
    mat_row = np.repeat(np.arange(n), P)
    mat_slot = np.arange(len(mat_row)) - np.repeat(c.feat_off[:-1], P)
    pool[mat_row, mat_slot] = c.feats
    mask = np.arange(M_MAX)[None, :] < P[:, None]

    lay_row = np.repeat(np.arange(n), n_layers)
    lay_k = np.arange(len(lay_row)) - np.repeat(c.layer_off[:-1], n_layers)
    structure = np.zeros((n, M_MAX, MAX_LAYERS), dtype=np.float32)
    structure[lay_row, c.slots, lay_k] = c.thick / MAX_THICKNESS_NM
    targets = np.full((n, PACKED_SEQ_LEN), -100, dtype=np.int64)
    targets[lay_row, lay_k] = (c.slots * NUM_THICKNESSES
                               + (c.thick - _THICKNESS_STEP_NM) // _THICKNESS_STEP_NM)
    eos = n_layers < MAX_LAYERS
    targets[np.nonzero(eos)[0], n_layers[eos]] = EOS_TOKEN

    return {
        "lab": torch.from_numpy(np.ascontiguousarray(c.lab)),
        "pool_features": torch.from_numpy(pool),
        "pool_mask": torch.from_numpy(mask),
        "pool_size": torch.from_numpy(P.astype(np.int64)),
        "structure_matrix": torch.from_numpy(structure),
        "target_tokens": torch.from_numpy(targets),
    }


class PackedBatchStream(IterableDataset):
    """Packed training batches of a streaming FlexThinFilmDataset.

    Hand it to `DataLoader(..., batch_size=None)`; see the module docstring
    for why the batch sequence equals `DataLoader(ds, batch_size=B,
    collate_fn=collate_fn_packed)`.
    """

    def __init__(self, dataset: FlexThinFilmDataset, batch_size: int):
        if not dataset.streaming:
            raise ValueError("PackedBatchStream needs a streaming dataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        # Upper bound, as len(DataLoader) is for an iterable dataset; the
        # training loop sizes itself from len(dataset), not from this.
        return -(-len(self.dataset) // self.batch_size)

    def _shards(self) -> Iterator[ShardColumns]:
        ds = self.dataset
        worker = get_worker_info()
        files = sorted(ds.files, key=lambda f: f.shard_id)
        if worker is not None:
            files = files[worker.id::worker.num_workers]
        legacy = _legacy_decode()
        for f in files:
            rows = ds._split_rows_by_file.get(f.file_id)
            if rows is None or len(rows) == 0:
                continue
            try:
                table = pq.read_table(f.path, columns=_columns_for_shard(f.path))
            except Exception as exc:
                print(f"[WARN] Could not read {f.path}: {exc}")
                continue
            cols = None if legacy else shard_columns(table, rows)
            if cols is None:
                examples = []
                for row_idx in rows:
                    try:
                        row = {col: table[col][int(row_idx)].as_py()
                               for col in table.column_names}
                        examples.append(_row_to_example(row))
                    except Exception as exc:
                        print(f"[WARN] Skipping row {int(row_idx)} of "
                              f"{f.path}: {exc}")
                cols = columns_from_examples(examples)
            del table
            if cols.n:
                yield cols

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        B = self.batch_size
        pending: List[ShardColumns] = []
        held = 0
        with materialnk_validation_disabled():
            for cols in self._shards():
                start = 0
                while start < cols.n:
                    take = min(B - held, cols.n - start)
                    pending.append(cols.slice(start, start + take))
                    held += take
                    start += take
                    if held == B:
                        yield collate_columns(ShardColumns.concat(pending))
                        pending, held = [], 0
            if held:
                yield collate_columns(ShardColumns.concat(pending))
