"""The fast data path must feed the model exactly what the old one did.

Throughput work is only acceptable if it leaves every finished result valid:
the beta2 study, the first IsoFLOP sweep, and anything compared against them.
So these tests do not check "close": they check bit-for-bit equality of every
example, every collated batch, the batch ORDER through a multi-worker
DataLoader, and the trained weights, between the vectorized decode and the
original row-by-row decode (INDIGO_LEGACY_DECODE=1) feeding a verbatim copy of
the original collate.

Needs torch and pyarrow (the CRC environment has both); skipped otherwise.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.dataset import FlexThinFilmDataset  # noqa: E402
from src.material_features import featurize_pool, pad_pool_features  # noqa: E402
from src.materials_vocab import (EOS_TOKEN, M_MAX, MAX_LAYERS,  # noqa: E402
                                 build_structure_matrix, encode_layer)
import training as T  # noqa: E402

_PACKED_SEQ_LEN = MAX_LAYERS + 1


def old_collate_fn_packed(examples):
    """scripts/training.py collate_fn_packed as it was before the fast path,
    copied verbatim: the reference every new batch must equal."""
    all_lab, all_pool_feats, all_pool_masks = [], [], []
    all_pool_sizes, all_structures, all_targets = [], [], []
    for ex in examples:
        pool_feats_unpadded = featurize_pool(ex.pool, mode="raw_spectrum")
        pool_feats, pool_mask = pad_pool_features(pool_feats_unpadded, m_max=M_MAX)
        pool_size = len(ex.pool)
        n_layers = len(ex.target_slots)
        emits_eos = n_layers < MAX_LAYERS
        full_structure = build_structure_matrix(ex.target_slots,
                                                ex.target_thicknesses)
        targets = torch.full((_PACKED_SEQ_LEN,), -100, dtype=torch.long)
        for k in range(n_layers):
            targets[k] = encode_layer(ex.target_slots[k], ex.target_thicknesses[k])
        if emits_eos:
            targets[n_layers] = EOS_TOKEN
        all_lab.append(ex.lab)
        all_pool_feats.append(pool_feats)
        all_pool_masks.append(pool_mask)
        all_pool_sizes.append(pool_size)
        all_structures.append(full_structure)
        all_targets.append(targets)
    return {
        "lab": torch.stack(all_lab),
        "pool_features": torch.stack(all_pool_feats),
        "pool_mask": torch.stack(all_pool_masks),
        "pool_size": torch.tensor(all_pool_sizes, dtype=torch.long),
        "structure_matrix": torch.stack(all_structures),
        "target_tokens": torch.stack(all_targets),
    }


def _write_shards(root: Path, n_shards: int = 6, rows: int = 120, seed: int = 0):
    """Shards in the generator's schema (pandas to_parquet of Python lists:
    list<list<double>> spectra, JSON-string lab), with float64 spectra whose
    float32 cast is not exact, so a wrong cast would show."""
    rng = np.random.default_rng(seed)
    d = root / "angle_0_substrate_glass"
    d.mkdir(parents=True)
    for s in range(n_shards):
        R = []
        for _ in range(rows):
            P = int(rng.integers(4, M_MAX + 1))
            L = int(rng.integers(1, MAX_LAYERS + 1))
            R.append({
                "lab": json.dumps([float(x) for x in rng.uniform(-60, 95, 3)]),
                "pool_size": P,
                "pool_n": [list(rng.uniform(0.2, 4, 128)) for _ in range(P)],
                "pool_k": [list(rng.uniform(0, 5, 128) * (rng.random() < .5))
                           for _ in range(P)],
                "pool_names": [f"m{j}" for j in range(P)],
                "pool_sources": ["jaxlayerlumos"] * P,
                "layer_slots": [int(x) for x in rng.integers(0, P, L)],
                "layer_thicknesses": [int(x) for x in rng.integers(1, 101, L) * 2],
                "num_layers": L,
                "structure_source": "random",
            })
        # Several row groups, so take() sees a multi-chunk table.
        pq.write_table(pa.Table.from_pylist(R), d / f"shard_{s}.parquet",
                       row_group_size=37)
    return root


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    return _write_shards(tmp_path_factory.mktemp("shards"))


def _examples(root, legacy, **kw):
    os.environ["INDIGO_LEGACY_DECODE"] = "1" if legacy else "0"
    try:
        ds = FlexThinFilmDataset(root, split="all", streaming=True, **kw)
        return list(ds)
    finally:
        os.environ.pop("INDIGO_LEGACY_DECODE", None)


def _same_example(a, b):
    assert torch.equal(a.lab, b.lab)
    assert a.target_slots == b.target_slots
    assert a.target_thicknesses == b.target_thicknesses
    assert a.structure_source == b.structure_source
    assert len(a.pool) == len(b.pool)
    for ma, mb in zip(a.pool, b.pool):
        assert ma.name == mb.name and ma.source == mb.source
        assert ma.n.dtype == mb.n.dtype == np.float64
        assert np.array_equal(ma.n, mb.n) and np.array_equal(ma.k, mb.k)


def test_examples_identical(shards):
    old = _examples(shards, legacy=True)
    new = _examples(shards, legacy=False)
    assert len(old) == len(new) == 720
    for a, b in zip(old, new):
        _same_example(a, b)
        assert a.pool_features is None
        assert torch.equal(b.pool_features, featurize_pool(b.pool))


def test_subset_identical(shards):
    """A shard-aligned training subset: only some rows of each shard."""
    kw = dict(limit_examples=300, limit_shard_aligned=True)
    os.environ["INDIGO_LEGACY_DECODE"] = "1"
    old = list(FlexThinFilmDataset(shards, split="train", streaming=True, **kw))
    os.environ["INDIGO_LEGACY_DECODE"] = "0"
    new = list(FlexThinFilmDataset(shards, split="train", streaming=True, **kw))
    os.environ.pop("INDIGO_LEGACY_DECODE")
    assert len(old) == len(new) == 300
    for a, b in zip(old, new):
        _same_example(a, b)


def test_batches_identical(shards):
    old = _examples(shards, legacy=True)
    new = _examples(shards, legacy=False)
    for i in range(0, len(old), 64):
        ref = old_collate_fn_packed(old[i:i + 64])
        for got in (T.collate_fn_packed(new[i:i + 64]),
                    T.collate_fn_packed(old[i:i + 64])):
            assert ref.keys() == got.keys()
            for k in ref:
                assert ref[k].dtype == got[k].dtype, k
                assert torch.equal(ref[k], got[k]), k


def _loader_batches(root, legacy, collate, workers=3, prefetch=2):
    from torch.utils.data import DataLoader
    os.environ["INDIGO_LEGACY_DECODE"] = "1" if legacy else "0"
    try:
        ds = FlexThinFilmDataset(root, split="train", streaming=True,
                                 limit_examples=500, limit_shard_aligned=True)
        return list(DataLoader(ds, batch_size=32, collate_fn=collate,
                               num_workers=workers,
                               prefetch_factor=prefetch if workers else None))
    finally:
        os.environ.pop("INDIGO_LEGACY_DECODE", None)


def test_dataloader_order_identical(shards):
    """Batch order through worker processes is what the training trajectory
    depends on; it must not move."""
    # The old settings (prefetch 1) against the new (prefetch 4), six workers.
    ref = _loader_batches(shards, True, old_collate_fn_packed, 6, 1)
    got = _loader_batches(shards, False, T.collate_fn_packed, 6, 4)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        for k in a:
            assert torch.equal(a[k], b[k]), k


def old_run_one_epoch(model, optimizer, loader, device, *, total_steps,
                      base_lr, warmup_fraction, lr_schedule, grad_clip,
                      loss_fn):
    """The optimisation core of run_one_epoch as it was before the sync-free
    loop: per-step .item() reads, blocking host-to-device copies."""
    model.train()
    epoch_loss = epoch_acc = 0.0
    n_batches = global_step = 0
    last_loss = float("nan")
    for batch in loader:
        lr = T.get_lr_schedule(global_step, total_steps, base_lr,
                               warmup_fraction, lr_schedule)
        T.set_lr(optimizer, lr)
        optimizer.zero_grad()
        losses = loss_fn(model, {k: v.to(device) for k, v in batch.items()})
        losses["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        last_loss = losses["loss"].item()
        epoch_loss += last_loss
        epoch_acc += losses["accuracy"].item()
        n_batches += 1
        global_step += 1
    return {"avg_loss": epoch_loss / n_batches, "avg_acc": epoch_acc / n_batches,
            "last_loss": last_loss, "global_step": global_step}


def test_training_identical(shards):
    """Optimizer steps on CPU: the old data path through the old loop, against
    the new data path through the new run_one_epoch. The weights must be
    bit-identical, and so must the epoch-mean and last training loss."""
    from src.model import ModelConfig, build_model, compute_loss_packed
    from torch.optim import AdamW

    kw = dict(total_steps=None, base_lr=1e-3, warmup_fraction=0.1,
              lr_schedule="cosine", grad_clip=1.0, loss_fn=compute_loss_packed)

    def train(batches, new):
        T.set_seed(0)
        cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, dropout=0.1,
                          encoder_hidden=32, encoder_out=16,
                          head_mode="cross_attn", slot_encoder_layers=1,
                          decoder_layers=1)
        model = build_model(cfg)
        opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                    weight_decay=0.01)
        args = dict(kw, total_steps=len(batches))
        if new:
            out = T.run_one_epoch(model, opt, batches, torch.device("cpu"),
                                  log_every=3, verbose=True, **args)
        else:
            out = old_run_one_epoch(model, opt, batches, torch.device("cpu"),
                                    **args)
        return model.state_dict(), out

    ref_w, ref = train(_loader_batches(shards, True, old_collate_fn_packed, 0), False)
    got_w, got = train(_loader_batches(shards, False, T.collate_fn_packed, 0), True)
    for k in ref_w:
        assert torch.equal(ref_w[k], got_w[k]), k
    assert got["global_step"] == ref["global_step"]
    assert got["avg_loss"] == ref["avg_loss"]
    assert got["last_loss"] == ref["last_loss"]
    # Accuracy is a logged diagnostic; it is now computed without a host sync
    # and may differ from the old value in the last bit.
    assert abs(got["avg_acc"] - ref["avg_acc"]) < 1e-6


def test_split_cache_identical(shards, tmp_path, capsys):
    """The validation split read through the on-disk cache, both when the
    cache is built and when it is read back, equals reading the shards."""
    from src.dataset import load_split_cached, split_cache_key
    # A few rows scattered over every shard, as the real validation split is.
    kw = dict(split="train", streaming=True, seed=7, limit_examples=60)
    ref = list(FlexThinFilmDataset(shards, **kw))
    assert len(ref) == 60
    assert len(FlexThinFilmDataset(shards, **kw)._split_rows_by_file) == 6
    ds = FlexThinFilmDataset(shards, **kw)
    built = load_split_cached(ds, tmp_path)
    assert "wrote" in capsys.readouterr().out
    again = load_split_cached(FlexThinFilmDataset(shards, **kw), tmp_path)
    assert "from the split cache" in capsys.readouterr().out
    for got in (built, again):
        assert len(got) == len(ref)
        for a, b in zip(ref, got):
            _same_example(a, b)
            assert torch.equal(a.pool_features, b.pool_features)
    # Different rows, different file.
    other = FlexThinFilmDataset(shards, **dict(kw, seed=8))
    assert split_cache_key(other) != split_cache_key(ds)


# ---------------------------------------------------------------------------
# Batch-native loader (src/batch_stream.py)
# ---------------------------------------------------------------------------

def _stream_batches(root, workers, prefetch, batch=32, **kw):
    from torch.utils.data import DataLoader
    from src.batch_stream import PackedBatchStream
    kw = {"limit_examples": 500, "limit_shard_aligned": True, **kw}
    ds = FlexThinFilmDataset(root, split="train", streaming=True, **kw)
    return list(DataLoader(PackedBatchStream(ds, batch), batch_size=None,
                           num_workers=workers,
                           prefetch_factor=prefetch if workers else None))


def _same_batches(ref, got):
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype, k
            assert a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("workers", [0, 1, 3, 6, 9])
def test_batch_stream_identical(shards, workers):
    """Old decode, old collate, old DataLoader settings against the batch
    stream: same batches, same order, including batches that span a shard
    boundary, each worker's partial last batch, and more workers than
    shards."""
    ref = _loader_batches(shards, True, old_collate_fn_packed, workers, 1)
    got = _stream_batches(shards, workers, 4)
    assert any(b["lab"].shape[0] < 32 for b in ref) or workers == 0
    _same_batches(ref, got)


def test_batch_stream_full_split(shards):
    """Every row of the corpus, no limit: whole shards, dense selection."""
    from torch.utils.data import DataLoader
    os.environ["INDIGO_LEGACY_DECODE"] = "1"
    try:
        ds = FlexThinFilmDataset(shards, split="train", streaming=True)
        ref = list(DataLoader(ds, batch_size=64, collate_fn=old_collate_fn_packed,
                              num_workers=4, prefetch_factor=1))
    finally:
        os.environ.pop("INDIGO_LEGACY_DECODE")
    from src.batch_stream import PackedBatchStream
    got = list(DataLoader(PackedBatchStream(
        FlexThinFilmDataset(shards, split="train", streaming=True), 64),
        batch_size=None, num_workers=4, prefetch_factor=4))
    _same_batches(ref, got)


def test_batch_stream_legacy_paths(shards):
    """The fallbacks: INDIGO_LEGACY_DECODE=1, and examples converted to
    columns, give the same batches."""
    ref = _loader_batches(shards, True, old_collate_fn_packed, 3, 1)
    os.environ["INDIGO_LEGACY_DECODE"] = "1"
    try:
        got = _stream_batches(shards, 3, 2)
    finally:
        os.environ.pop("INDIGO_LEGACY_DECODE")
    _same_batches(ref, got)


@pytest.fixture(scope="module")
def mixed_shards(tmp_path_factory):
    """Two ordinary shards and one in an older layout (string-encoded
    spectra and slots), which the column path must refuse and the per-row
    path must decode."""
    root = _write_shards(tmp_path_factory.mktemp("mixed"), n_shards=2, seed=5)
    rng = np.random.default_rng(9)
    R = []
    for _ in range(90):
        P = int(rng.integers(4, 12)); L = int(rng.integers(1, MAX_LAYERS + 1))
        R.append({
            "lab": json.dumps([float(x) for x in rng.uniform(-60, 95, 3)]),
            "pool_size": P,
            "pool_n": json.dumps([list(rng.uniform(0.2, 4, 128)) for _ in range(P)]),
            "pool_k": json.dumps([list(rng.uniform(0, 5, 128)) for _ in range(P)]),
            "pool_names": [f"m{j}" for j in range(P)],
            "pool_sources": ["synthetic"] * P,
            "layer_slots": json.dumps([int(x) for x in rng.integers(0, P, L)]),
            "layer_thicknesses": json.dumps([int(x) for x in rng.integers(1, 101, L) * 2]),
            "num_layers": L,
        })
    pq.write_table(pa.Table.from_pylist(R),
                   root / "angle_0_substrate_glass" / "shard_7.parquet")
    return root


def test_batch_stream_old_layout_shard(mixed_shards):
    ref = _loader_batches(mixed_shards, True, old_collate_fn_packed, 2, 1)
    got = _stream_batches(mixed_shards, 2, 4)
    _same_batches(ref, got)


def test_collate_columns_rejects_what_old_collate_rejects(shards):
    """Bad inputs raise in both, rather than silently training on them."""
    from src.batch_stream import collate_columns, columns_from_examples
    exs = _examples(shards, legacy=True)[:8]
    for mutate in (lambda e: e.target_thicknesses.__setitem__(0, 3),
                   lambda e: e.target_slots.__setitem__(0, M_MAX),
                   lambda e: e.target_slots.extend([0] * MAX_LAYERS)
                   or e.target_thicknesses.extend([2] * MAX_LAYERS)):
        bad = _examples(shards, legacy=True)[:8]
        mutate(bad[3])
        with pytest.raises(ValueError):
            old_collate_fn_packed(bad)
        with pytest.raises(ValueError):
            collate_columns(columns_from_examples(bad))
    _same_batches([old_collate_fn_packed(exs)],
                  [collate_columns(columns_from_examples(exs))])


def test_existing_results_pins_the_cell(tmp_path):
    """A finished cell with the same architecture but a different example
    count (the same model on another IsoFLOP rung) must not count as done."""
    import argparse
    import lr_tuning as L
    args = argparse.Namespace(epochs=1, d_model=96, slot_encoder_layers=2,
                              batch_size=256, beta2=0.999, lr_schedule="cosine",
                              seed=42, output_suffix="", limit_examples=2395648)
    other = tmp_path / "lr_search_ep1_lim793088_d96_se2_bs256_b20.999.json"
    json.dump({"limit_examples": 793088, "epochs": 1, "seed": 42}, open(other, "w"))
    assert L.existing_results(tmp_path, args) == []
    mine = tmp_path / "lr_search_ep1_lim2395648_d96_se2_bs256_b20.999.json"
    json.dump({"limit_examples": 2395648, "epochs": 1, "seed": 42}, open(mine, "w"))
    assert L.existing_results(tmp_path, args) == [mine]
