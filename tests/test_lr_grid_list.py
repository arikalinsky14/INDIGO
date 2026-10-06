"""slurms/lr_grid.sh --list must work for every stage.

A stage whose --list exits early is a stage nobody can size or submit; one
bug of exactly that kind (a failing arithmetic test inside a command
substitution, fatal under `set -e`) broke every stage but 2 and check.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["probe", "speed", "1", "beta2", "2", "check", "3"])
def test_list_runs(stage):
    r = subprocess.run(["bash", "slurms/lr_grid.sh", "--list"], cwd=REPO,
                       env={**os.environ, "STAGE": stage},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"STAGE {stage}:" in r.stdout
    assert "submit with --array=0-" in r.stdout
