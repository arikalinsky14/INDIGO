"""_sample_token never returns an index past the vocabulary. With thousands of
allowed tokens, the float32 cumulative probabilities can end below 1
(0.9999996 here), and a uniform draw above that used to map to token 3201
(one failed NEUTRON row in 601, Nov 18)."""
import numpy as np
import torch

from inference.src import generate as g
from src.materials_vocab import VOCAB_SIZE


def _row_with_short_total():
    rng = np.random.default_rng(0)
    for _ in range(100):
        k = int(rng.integers(3, VOCAB_SIZE))
        vals = rng.normal(0, 2, k).astype(np.float32)
        row = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
        idx = np.sort(rng.choice(VOCAB_SIZE, k, replace=False))
        row[idx] = vals
        exp = np.exp(row - np.max(row[np.isfinite(row)]))
        exp[~np.isfinite(row)] = 0.0
        total = np.cumsum(exp / exp.sum())[-1]
        if total < np.float32(0.99999994):
            return row, idx
    raise AssertionError("no row with a short float32 total found")


def test_draw_above_rounded_total_maps_to_last_allowed(monkeypatch):
    row, idx = _row_with_short_total()
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(np.float32(0.99999994)))
    tok = g._sample_token(row, 1.0, None, "cpu")
    assert tok < VOCAB_SIZE
    assert tok == idx[-1]


def test_ordinary_draws_unchanged(monkeypatch):
    row = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
    row[[10, 500]] = [0.0, 0.0]
    for u, want in ((0.1, 10), (0.49, 10), (0.51, 500), (0.9, 500)):
        monkeypatch.setattr(torch, "rand", lambda *a, _u=u, **k: torch.tensor(_u))
        assert g._sample_token(row, 1.0, None, "cpu") == want
