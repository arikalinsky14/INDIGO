"""Splitting a cell's learning-rate grid across array tasks must not change it.

Three things have to hold for a split cell to be the same experiment as an
unsplit one: the tasks cover every grid point exactly once, the sub-grid the
SLURM wrapper hands each task is exactly those points of the full grid, and
the merged file selects the optimum the way lr_tuning.py would have.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import lr_grid_cells as G      # noqa: E402
import merge_lr_parts as M     # noqa: E402


def _stage2_tasks(max_hours):
    out = subprocess.run(
        # The pre-Oct-6 rate, at which stage 2's long cells must split; at
        # the current rate none would, and the test would check nothing.
        [sys.executable, "scripts/lr_grid_cells.py", "--stage", "2",
         "--format", "lines", "--rate", "2033",
         "--max-task-hours", str(max_hours)],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [l.split() for l in out.strip().splitlines()]


def test_tasks_cover_every_grid_point_once():
    unsplit = _stage2_tasks(0)
    split = _stage2_tasks(6)
    assert len(split) > len(unsplit)
    cover = {}
    for f in split:
        key = tuple(f[:7])
        if f[7] == "-":
            cover.setdefault(key, []).extend(range(7))
        else:
            span, n = f[7].split("/")
            k0, k1 = map(int, span.split("-"))
            assert int(n) == 7
            cover.setdefault(key, []).extend(range(k0, k1 + 1))
    assert set(cover) == {tuple(f[:7]) for f in unsplit}
    for key, ks in cover.items():
        assert sorted(ks) == list(range(7)), key


def test_tasks_fit_the_target():
    cells = [{"D": 18_522_880, "lr": None}, {"D": 400_000, "lr": None}]
    tasks = G.split_tasks(cells, 7, 6.0, 2033.0, 10_000, 2048)
    hours = [G.cell_hours(t["D"], t["task_lrs"], 2033.0, 10_000, 2048)
             for t in tasks]
    assert max(hours) <= 6.0
    assert sum(t["part"] is None for t in tasks) == 1       # the small cell
    assert G.split_tasks(cells, 7, None, 2033.0, 10_000, 2048)[0]["part"] is None


def test_sub_grid_is_the_full_grid_restricted():
    """The arithmetic slurms/lr_grid.sh does for a part, against np.logspace
    as lr_tuning.py builds the full grid."""
    lo, hi, n = 2.12e-4 / 8, 2.12e-4 * 5, 7
    full = np.logspace(np.log10(lo), np.log10(hi), n)
    at = lambda k: 10 ** (math.log10(lo) + k * (math.log10(hi) - math.log10(lo)) / (n - 1))
    for k0, k1 in [(0, 1), (2, 3), (4, 5), (6, 6), (0, 3), (4, 6)]:
        a, b = float(f"{at(k0):.10e}"), float(f"{at(k1):.10e}")
        sub = np.logspace(np.log10(a), np.log10(b), k1 - k0 + 1)
        assert np.allclose(sub, full[k0:k1 + 1], rtol=1e-9)


def _trial(lr, de, diverged=False):
    return {"lr": lr, "best_val_loss": 80.0 if diverged else 6.0 + de / 100,
            "final_val_de": de, "final_val_de_p95": de + 5,
            "final_val_de_by_chroma": None, "de_result": None}


def _part(tmp, stem, k0, k1, trials, **over):
    d = {k: 1 for k in M.MUST_MATCH}
    d.update({"selection_metric": "delta_e", "seed": 42, "results": trials,
              "optimal_lr": trials[0]["lr"]})
    d.update(over)
    path = tmp / f"{stem}{M.part_suffix(k0, k1, 7)}.json"
    json.dump(d, open(path, "w"))
    return path


def test_merge_reselects_over_all_rates(tmp_path):
    parts, out = tmp_path / "parts", tmp_path
    parts.mkdir()
    stem = "lr_search_ep1_lim100_d96_se3_bs256_b20.999"
    lrs = np.logspace(-4, -2, 7)
    des = [14, 12, 11, 10.5, 9.0, 30, 30]      # best at index 4, top diverges
    trials = [_trial(l, d, diverged=i == 6) for i, (l, d) in
              enumerate(zip(lrs, des))]
    _part(parts, stem, 0, 1, trials[0:2])
    _part(parts, stem, 2, 3, trials[2:4])
    st = M.merge_dir(parts, out, verbose=False)
    assert "waiting" in st[stem] and not (out / f"{stem}.json").exists()
    _part(parts, stem, 4, 5, trials[4:6])
    _part(parts, stem, 6, 6, trials[6:7])
    st = M.merge_dir(parts, out, verbose=False)
    merged = json.load(open(out / f"{stem}.json"))
    assert merged["n_lrs"] == 7 and merged["merged_from_parts"] == 4
    assert [r["lr"] for r in merged["results"]] == sorted(lrs.tolist())
    assert merged["optimal_lr"] == lrs[4] and merged["optimal_val_de"] == 9.0
    # A finished cell is never rewritten.
    assert M.merge_dir(parts, out, verbose=False)[stem] == "already merged"


def test_diverged_rate_never_wins(tmp_path):
    t = [_trial(1e-3, 5.0, diverged=True), _trial(1e-4, 12.0)]
    assert M.select(t, "delta_e")["lr"] == 1e-4


def test_parts_from_different_runs_refuse_to_merge(tmp_path):
    stem = "lr_search_x"
    a = _part(tmp_path, stem, 0, 3, [_trial(1e-4, 10)] * 4)
    b = _part(tmp_path, stem, 4, 6, [_trial(1e-3, 10)] * 3, seed=43)
    with pytest.raises(ValueError, match="seed"):
        M.merge([json.load(open(a)), json.load(open(b))])
