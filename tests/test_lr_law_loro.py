"""fit_lr_law.py's leave-one-rung-out check, on a planted law.

Rows are placed at the first sweep's own (N, D) points on its five usable
rungs, with lr* = a N^b D^c times a little noise. Holding out any rung must
predict it to within the noise, and every row must be assigned to the rung
it came from.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fit_lr_law as F  # noqa: E402

SWEEP = REPO / "analyses/scaling/results/isoflop_fit.json"


def _rows(noise, seed=0, b=-0.45, c=-0.12):
    rng = np.random.default_rng(seed)
    runs = json.load(open(SWEEP))["runs"]
    budgets = sorted(json.load(open(SWEEP))["budgets"])[:5]
    rows, truth = [], []
    for r in runs:
        if not r["name"].endswith("_s42"):
            continue
        bud = min(budgets, key=lambda x: abs(x - r["flops"]))
        if abs(bud - r["flops"]) / bud > 0.25:
            continue
        lr = 0.03 * r["n_params"] ** b * r["passes"] ** c * np.exp(rng.normal(0, noise))
        rows.append({"n_params": r["n_params"], "train_examples": r["passes"],
                     "lr_star": lr, "usable": True})
        truth.append(bud)
    return rows, truth


def test_rows_land_on_their_rung():
    rows, truth = _rows(0.0)
    got = F.rung_of(rows, str(SWEEP))
    assert all(abs(g - t) / t < 1e-9 for g, t in zip(got, truth))


def test_planted_law_predicts_held_out_rungs():
    rows, _ = _rows(0.05)
    out = F.leave_one_rung_out(rows, str(SWEEP))
    assert out["status"] == "ok" and len(out["rungs"]) == 5
    assert out["rungs"][-1]["held_out"].startswith("top")
    for r in out["rungs"]:
        assert 0.85 < r["median_ratio"] < 1.15, r
        assert 0.75 < r["worst_ratio"] < 1.33, r


def test_a_law_that_bends_fails_at_the_top():
    """Curvature the power law cannot follow shows up in the held-out top
    rung, which is what the check exists to catch."""
    rows, _ = _rows(0.0)
    for r in rows:
        r["lr_star"] *= np.exp(0.25 * (np.log(r["n_params"]) - 12) ** 2)
    top = F.leave_one_rung_out(rows, str(SWEEP))["rungs"][-1]
    assert abs(np.log(top["worst_ratio"])) > np.log(1.4)
