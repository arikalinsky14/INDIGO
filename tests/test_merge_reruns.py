"""scripts/merge_reruns.py: a sampler-crash rerun replaces its crashed row in
place, the run then holds each row once, and nothing else can be replaced."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "merge_reruns.py"
CRASH = "generate failed: ValueError: token_id 3201 out of range [0, 3201)"


def _chunk(run: Path, name: str, rows, sha="abc"):
    d = run / name / "tag" / "tier_b"
    d.mkdir(parents=True)
    (d / "rows.json").write_text(json.dumps(rows))
    (d / "summary.json").write_text(json.dumps({"model_sha256": sha, "n_failed": 9}))
    (d.parent / "summary.json").write_text(json.dumps({"tiers": {"tier_b": {"n_failed": 9}}}))


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def test_merge_then_check(tmp_path):
    run = tmp_path / "runs" / "strong"
    _chunk(run, "rows000", [{"row_idx": 0, "delta_e": 0.5},
                            {"row_idx": 1, "target_lab": [50, 0, 0], "error": CRASH}])
    _chunk(run, "rerun_1", [{"row_idx": 1, "target_lab": [50, 0, 0], "delta_e": 1.5}])
    assert _run("--check", tmp_path / "runs").returncode == 1
    r = _run(run)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = json.loads((run / "rows000/tag/tier_b/rows.json").read_text())
    assert [x["row_idx"] for x in rows] == [0, 1]
    assert rows[1]["delta_e"] == 1.5 and rows[1]["rerun_of_sampler_crash"]["replaced_error"] == CRASH
    summ = json.loads((run / "rows000/tag/tier_b/summary.json").read_text())
    assert summ["n_failed"] == 0 and summ["n_succeeded"] == 2 and summ["merged_reruns"] == [1]
    assert not (run / "rerun_1").exists()
    assert (tmp_path / "runs" / "_merged_reruns" / "strong" / "rerun_1").is_dir()
    assert _run("--check", tmp_path / "runs").returncode == 0
    assert "no rerun folders" in _run(run).stdout          # idempotent


def test_refuses_to_replace_a_real_failure(tmp_path):
    run = tmp_path / "runs" / "strong"
    _chunk(run, "rows000", [{"row_idx": 1, "error": "no feasible candidate"}])
    _chunk(run, "rerun_1", [{"row_idx": 1, "delta_e": 1.5}])
    r = _run(run)
    assert r.returncode != 0 and "not a sampler crash" in (r.stdout + r.stderr)
    assert (run / "rerun_1").is_dir()


def test_refuses_another_checkpoint(tmp_path):
    run = tmp_path / "runs" / "strong"
    _chunk(run, "rows000", [{"row_idx": 1, "error": CRASH}], sha="abc")
    _chunk(run, "rerun_1", [{"row_idx": 1, "delta_e": 1.5}], sha="zzz")
    r = _run(run)
    assert r.returncode != 0 and "checkpoint" in (r.stdout + r.stderr)
