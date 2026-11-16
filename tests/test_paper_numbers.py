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


def _rows_with_pools(n, held_frac_rows, rng):
    """Tier B rows whose pools and stacks are recorded, as test_eval.py now
    writes them. Rows i % 3 == 0: pool and stack hold Au; i % 3 == 1: pool
    holds Au, stack does not; i % 3 == 2: no held-out real in the pool."""
    rows = []
    for i in range(n):
        pool = ["perturb_small(Au)", "SiO2", "lorentz_dielectric_n1"]
        if i % 3 != 2:
            pool.append("Au")
        stack = ["Au", "SiO2"] if i % 3 == 0 else ["SiO2", "perturb_small(Au)"]
        rows.append({"row_idx": i, "target_lab": [50.0, 5.0 * (i % 8), 0.0],
                     "pool_names": pool, "pool_sources": ["x"] * len(pool),
                     "delta_e": float(rng.uniform(0, 4)),
                     "layers": [{"material": m, "thickness_nm": 50.0} for m in stack]})
    return rows


def test_tier_b_held_out_split_and_gaps(tmp_path):
    rng = np.random.default_rng(3)
    for name, prop in (("m", "model"), ("r", "random")):
        d = tmp_path / name / "ckpt" / "tier_b"
        d.mkdir(parents=True)
        json.dump(_rows_with_pools(200, None, rng), open(d / "rows.json", "w"))
        json.dump({"model_sha256": "abc", "args": dict(SET, proposer=prop)},
                  open(d / "summary.json", "w"))
    r, out = run(tmp_path, "--run", f"INDIGO={tmp_path/'m'}", "--run", f"random={tmp_path/'r'}",
                 "--compare", "INDIGO:random")
    assert r.returncode == 0, r.stderr
    h = json.load(open(out))
    run_ = h["runs"]["INDIGO"]
    assert run_["tiers"]["A"] == {"not_run": True} and run_["tiers"]["A+B"] == {"not_run": True}
    g = run_["tier_b_held_out"]["groups"]
    assert g["a_pool_has_held_out"]["n"] == 134        # i % 3 in (0, 1)
    assert g["b_stack_uses_held_out"]["n"] == 67       # i % 3 == 0
    assert g["c_pool_has_none"]["n"] == 66
    assert run_["tier_b_held_out"]["untraceable_pool"] == 0
    assert run_["folder"].endswith("/m") and run_["rows_files"]
    gap = h["comparisons"]["INDIGO - random"]["B"]["by_chroma"]
    assert set(gap) == {"C<8", "8<=C<30", "C>=30"}
    assert all(c["pass_lt_2"]["lo"] <= c["pass_lt_2"]["est"] <= c["pass_lt_2"]["hi"]
               for c in gap.values())


def _write_shards(root, tier, pools_by_lab, manifest_flag=True):
    import pandas as pd
    d = root / f"tier_{tier}" / "angle_0_substrate_x"
    d.mkdir(parents=True)
    rows = [{"lab": json.dumps(lab), "pool_names": names, "pool_sources": ["s"] * len(names)}
            for lab, names in pools_by_lab]
    pd.DataFrame(rows).to_parquet(d / "shard_000000.parquet", index=False)
    json.dump({"use_held_out_reals": manifest_flag}, open(d / "shard_000000.manifest.json", "w"))


def test_old_rows_traced_through_shards_and_list_checked(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(4)
    labs = [[50.0, 3.0 * i, 1.0] for i in range(200)]
    pools = [["Au", "lorentz_metal_1"] if i % 2 else ["perturb_large(Ge)", "interp(Au,Ge,w=0.5)"] for i in range(200)]
    _write_shards(tmp_path / "test", "b", list(zip(labs, pools)))
    _write_shards(tmp_path / "test", "a", [([1.0, 2.0, 3.0], ["Ag", "SiO2"])], False)
    d = tmp_path / "old" / "ckpt" / "tier_b"
    d.mkdir(parents=True)
    # Old rows: no pool recorded; one row's target matches nothing.
    rows = [{"row_idx": i, "target_lab": labs[i] if i != 7 else [99.0, 99.0, 99.0],
             "delta_e": float(rng.uniform(0, 4)),
             "layers": [{"material": "Au" if i % 2 else "interp(Au,Ge,w=0.5)", "thickness_nm": 9.0}]}
            for i in range(200)]
    json.dump(rows, open(d / "rows.json", "w"))
    json.dump({"model_sha256": "abc", "args": SET}, open(d / "summary.json", "w"))
    r, out = run(tmp_path, "--run", f"old={tmp_path/'old'}", "--test-dir", str(tmp_path / "test"))
    assert r.returncode == 0, r.stderr
    h = json.load(open(out))
    hb = h["runs"]["old"]["tier_b_held_out"]
    assert hb["untraceable_pool"] == 1 and hb["pool_from"]["shards"] == 199
    assert hb["groups"]["a_pool_has_held_out"]["n"] == 99      # odd rows but 7
    assert hb["groups"]["b_stack_uses_held_out"]["n"] == 99
    assert h["held_out_reals"]["check"]["verified"] is True
    # A held-out real in a tier A pool: the list no longer matches the data.
    _write_shards(tmp_path / "bad", "b", list(zip(labs, pools)))
    _write_shards(tmp_path / "bad", "a", [([1.0, 2.0, 3.0], ["Au", "SiO2"])], False)
    r, _ = run(tmp_path, "--run", f"old={tmp_path/'old'}", "--test-dir", str(tmp_path / "bad"))
    assert r.returncode != 0 and "held-out list does not match" in (r.stderr + r.stdout)
