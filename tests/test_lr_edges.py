"""STAGE=2x and STAGE=2edge: finding stage 2's open edges and closing them.

Cells are planted in a temporary stage-2 directory in lr_tuning.py's results
format: one bracketed, one whose best rate is the lowest, one whose best is
the top of the grid, one whose best stable rate sits under a diverged one.
The planner must extend exactly the open ones, the right way, and an
extension file must merge back into its cell.
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

import lr_edges as E  # noqa: E402
from lr_grid_cells import cells_for  # noqa: E402

FIT = "analyses/scaling/results/porian_fit.json"
GRID = [0.125 * 40 ** (k / 6) for k in range(7)]


def trial(lr, de, div=False):
    return {"lr": lr, "best_val_loss": 80.0 if div else 6.0,
            "final_val_de": de, "final_val_de_by_chroma": None,
            "final_val_loss": 6.0}


def curve(opt_mult, diverge_from=None):
    """DeltaE over the 7-rate grid, minimum at opt_mult x prior."""
    out = []
    for g in GRID:
        div = diverge_from is not None and g >= diverge_from
        out.append(trial(g * 1e-3, 30.0 if div else
                         10 + 3 * math.log(g / opt_mult) ** 2, div))
    return out


def test_lr_extension_kinds():
    assert E.lr_extension(curve(0.8)) is None                # bracketed
    kind, rates = E.lr_extension(curve(0.05))                 # best at bottom
    assert kind == "down" and max(rates) < GRID[0] * 1e-3
    kind, rates = E.lr_extension(curve(20))                   # best at top
    assert kind == "up" and min(rates) > GRID[-1] * 1e-3
    kind, rates = E.lr_extension(curve(20, diverge_from=5.0))  # under a cliff
    assert kind == "cliff" and len(rates) == 1
    assert GRID[-2] * 1e-3 < rates[0] < GRID[-1] * 1e-3


def test_rung_extension_sizes():
    ladder = [(53_000, 16, 1), (81_000, 32, 1), (100_000, 40, 1),
              (124_000, 48, 1), (151_000, 56, 1), (182_000, 64, 1),
              (900_000, 128, 3), (2_000_000, 192, 3)]
    side, got = E.rung_extension_sizes([182_000, 232_000, 456_000], 182_000, ladder)
    assert side == "left" and [s[0] for s in got] == [124_000, 81_000]
    side, got = E.rung_extension_sizes([182_000, 232_000, 456_000], 250_000, ladder)
    assert side == "" and got == []
    side, got = E.rung_extension_sizes([81_000, 182_000], 81_000, ladder[1:])
    assert side == "left" and got == []                       # ladder ends


def _write(sdir: Path, c: dict, trials, suffix=""):
    name = (f"lr_search_ep{c['epochs']}_lim{c['limit']}_d{c['d_model']}"
            f"_se{c['se']}_bs256_b20.999{suffix}.json")
    best = min((t for t in trials if not E.diverged(t)),
               key=lambda t: t["final_val_de"])
    json.dump({"epochs": c["epochs"], "seed": 42, "limit_shard_aligned": True,
               "beta2": 0.999, "n_params": c["n_params"],
               "train_examples": c["limit"], "limit_examples": c["limit"],
               "d_model": c["d_model"], "slot_encoder_layers": c["se"],
               "batch_size": 256, "selection_metric": "delta_e",
               "optimal_lr": best["lr"], "results": trials},
              open(sdir / name, "w"))
    return sdir / name


def _lines(stage, sdir, *extra):
    r = subprocess.run([sys.executable, "scripts/lr_grid_cells.py", "--stage",
                        stage, "--format", "lines", "--stage2-dir", str(sdir),
                        "--rungs", "5", *extra],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [l.split() for l in r.stdout.strip().splitlines() if l.strip()]


def test_stage_2x_extends_only_open_cells(tmp_path):
    cells = cells_for(2, str(REPO / FIT), 0.999, 5)
    shapes = [curve(0.8), curve(0.05), curve(20), curve(20, diverge_from=5.0)]
    for i, c in enumerate(cells):
        _write(tmp_path, c, shapes[i % 4] if i < 4 else curve(0.8))
    got = _lines("2x", tmp_path)
    assert len(got) == 3
    for f in got:
        assert f[8] == "_ext1"
    # Close the "down" cell with an extension whose best is interior.
    c = cells[1]
    lo = GRID[0] * 1e-3
    _write(tmp_path, c, [trial(lo / 4, 14.0), trial(lo / 2, 12.0)], "_ext1")
    got = _lines("2x", tmp_path)
    assert len(got) == 2
    main = tmp_path / (f"lr_search_ep{c['epochs']}_lim{c['limit']}_d{c['d_model']}"
                       f"_se{c['se']}_bs256_b20.999.json")
    _, trials = E.cell_trials(main)
    assert len(trials) == 9 and trials[0]["lr"] == lo / 4
    assert E.next_ext_suffix(main) == "_ext2"


def test_stage_2edge_reads_the_fit(tmp_path):
    cells = cells_for(2, str(REPO / FIT), 0.999, 5)
    budgets = sorted({c["budget"] for c in cells})
    rungs = []
    for b in budgets:
        ns = sorted(c["n_params"] for c in cells if c["budget"] == b)
        on_edge = b == budgets[1]
        rungs.append({"budget": b, "on_edge": on_edge,
                      "n_star_akima": ns[0] if on_edge else ns[2]})
    fit = tmp_path / "fit.json"
    json.dump({"by_metric": {"pooled": {"rungs": rungs}}}, open(fit, "w"))
    got = _lines("2edge", tmp_path, "--extension-from", str(fit))
    assert len(got) == 2
    smallest = min(c["n_params"] for c in cells if c["budget"] == budgets[1])
    from src.scaling.flops import n_params, train_flops_per_example
    for f in got:
        d, se, lim, ep = int(f[0]), int(f[1]), int(f[2]), int(f[4])
        arch = E.arch_for(d, se)
        assert n_params(arch) < smallest
        flops = train_flops_per_example(arch) * lim * ep
        assert abs(flops - budgets[1]) / budgets[1] < 0.02


def test_soft_edge_sides():
    """An end point within k sigma of the minimum is a soft edge; a rung
    whose ends are both well above its minimum has none."""
    vals = {100: 10.25, 200: 9.99, 300: 10.8, 400: 19.8}
    assert E.soft_edge_sides(vals, sigma=0.52) == ["left"]
    assert E.soft_edge_sides(vals, sigma=0.52, k_sigma=0.4) == []
    assert E.soft_edge_sides({1: 12.0, 2: 9.0, 3: 12.5}, sigma=0.52) == []
    assert E.soft_edge_sides({1: 9.2, 2: 9.0, 3: 9.3}, sigma=0.52) == ["left", "right"]
