"""collect_run_times.py reads N, D and per-rate train time from lr_grid.sh
logs, marks the first rate cold, and skips logs from the old pipeline."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import collect_run_times as C  # noqa: E402

LOG = """==================
 STAGE 2, cell 18 of 31
   d_model / se        96 / 3  (n_heads 3)
   N (parameters)      568389
   D (examples)        1800000 x 1 epoch(s)
   seed                42
[1/2] Training with LR = 1.07e-04
[loader] worker 0: 300 rows; read 1 s
    [time] train 30.0 min, val 0.0 min, dE 3.3 min
[2/2] Training with LR = 1.97e-04
    [time] train 10.0 min, val 0.0 min, dE 3.2 min
"""


def test_parse_and_rates(tmp_path):
    p = tmp_path / "indigo-lr-grid.1_18.out"
    p.write_text(LOG)
    old = tmp_path / "indigo-lr-grid.1_19.out"
    old.write_text(LOG.replace("[loader]", "[other]"))
    ts = C.parse_log(p)
    assert C.parse_log(old) == []
    assert [t["cold"] for t in ts] == [True, False]
    assert ts[0]["ex_per_sec"] == 1_800_000 / 1800
    assert ts[1]["gpu_hours"] == 10 / 60
    assert ts[0]["n_params"] == 568389 and ts[0]["flops"] > 0
    table = C.rate_table(ts)
    assert table["568389"]["cold"] == 1000 and table["568389"]["warm"] == 3000
    table["1000000"] = {"cold": 2000.0, "warm": None}
    m = C.RateModel(table, "cold")
    assert abs(m(568389) - 1000) < 1e-6 and abs(m(5e6) - 2000) < 1e-6
    assert abs(m.gpu_hours(1e6, 7.2e6) - 1.0) < 1e-9
