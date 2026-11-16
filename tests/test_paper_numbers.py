"""scripts/paper_numbers.py: failed and missing rows count as failures,
chunks merge, settings are checked, intervals and paired differences behave."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "paper_numbers.py"


def write_run(root: Path, de_by_tier, settings, chunk=100, drop=(), fail=()):
    """test_eval.py's layout, split into row chunks like a chunked run."""
    for tier, des in de_by_tier.items():
        for lo in range(0, len(des), chunk):
            d = root / f"chunk_{lo:03d}" / "ckpt" / f"tier_{tier}"
            d.mkdir(parents=True)
            rows = []
            for i in range(lo, min(lo + chunk, len(des))):
                if (tier, i) in drop:
                    continue
                lab = [50.0, 4.0 * (i % 10), 0.0]        # chroma 0..36: all three buckets
                if (tier, i) in fail:
                    rows.append({"row_idx": i, "target_lab": lab, "error": "boom"})
                else:
                    rows.append({"row_idx": i, "target_lab": lab, "delta_e": float(des[i])})
            json.dump(rows, open(d / "rows.json", "w"))
            json.dump({"model_sha256": "abc", "args": dict(settings, offset=lo)},
                      open(d / "summary.json", "w"))


SET = dict(ensemble_n=250, top_k=10, refine_iters=100, proposer="model", seed=42,
           temperature=1.0, tolerance=0.0, weight_lambda=1.0, mc_samples=0)


def run(tmp, *args):
    out = tmp / "out" / "headline.json"
    r = subprocess.run([sys.executable, str(SCRIPT), *args, "--rows", "200",
                        "--draws", "500", "--out", str(out)],
                       capture_output=True, text=True)
    return r, out


def test_counts_failures_and_missing_and_pairs(tmp_path):
    rng = np.random.default_rng(1)
    good = {t: rng.uniform(0, 4, 200) for t in "ab"}       # P(<2) = 0.5
    bad = {t: good[t] + 1.0 for t in "ab"}                  # P(<2) = 0.25
    write_run(tmp_path / "m", good, SET, fail={("a", 3)}, drop={("b", 150)})
    write_run(tmp_path / "r", bad, dict(SET, proposer="random"))
    r, out = run(tmp_path, "--run", f"INDIGO={tmp_path/'m'}", "--run", f"random={tmp_path/'r'}",
                 "--compare", "INDIGO:random")
    assert r.returncode == 0, r.stderr
    h = json.load(open(out))
    a = h["runs"]["INDIGO"]["tiers"]["A"]
    assert a["n_failed"] == 1 and a["n_missing"] == 0
    exp = np.mean(np.where(np.arange(200) == 3, np.inf, good["a"]) < 2)
    assert a["all"]["pass_lt_2"]["est"] == pytest.approx(exp)
    assert a["all"]["pass_lt_2"]["lo"] <= exp <= a["all"]["pass_lt_2"]["hi"]
    b = h["runs"]["INDIGO"]["tiers"]["B"]
    assert b["n_missing"] == 1 and b["all"]["n"] == 200     # missing row kept, as a failure
    assert set(a["by_chroma"]) == {"C<8", "8<=C<30", "C>=30"}
    assert sum(c["n"] for c in a["by_chroma"].values()) == 200   # failed row bucketed too
    assert h["runs"]["random"]["settings"]["proposer"] == "random"
    d = h["comparisons"]["INDIGO - random"]["A+B"]["pass_lt_2"]
    assert 0.15 < d["est"] < 0.35 and d["lo"] > 0                 # planted +0.25
    assert Path(str(out).replace(".json", ".md")).read_text().count("| INDIGO |") >= 3


def test_refuses_chunks_with_different_settings(tmp_path):
    rng = np.random.default_rng(2)
    des = {t: rng.uniform(0, 4, 200) for t in "ab"}
    write_run(tmp_path / "m", des, SET)
    s = json.load(open(tmp_path / "m/chunk_100/ckpt/tier_a/summary.json"))
    s["args"]["top_k"] = 5
    json.dump(s, open(tmp_path / "m/chunk_100/ckpt/tier_a/summary.json", "w"))
    r, _ = run(tmp_path, "--run", f"x={tmp_path/'m'}")
    assert r.returncode != 0 and "disagree" in (r.stderr + r.stdout)


def test_imports_without_torch():
    src = SCRIPT.read_text()
    assert "import torch" not in src and "src.model" not in src
