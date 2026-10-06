"""The compute-axis analysis must import without torch.

`scripts/fit_wall_model.py` is meant to run on a login node with no
environment loaded. It did not: `src/scaling/flops.py` documented itself as
torch-free but imported `src/materials_vocab.py` and `src/material_features.py`,
which both imported torch at module scope, so the script died with
ModuleNotFoundError before doing anything.

Those two now import torch lazily. This locks that down, by importing the
analysis chain in a subprocess where `import torch` raises.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROBE = """
import sys


class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked by this test")
        return None


sys.meta_path.insert(0, _Blocked())
import src.scaling.flops            # noqa: F401
import src.scaling.configs          # noqa: F401
import src.scaling.porian           # noqa: F401

# Importing is not enough. achievable_sizes() builds a model shape on every
# call, and when that was a ModelConfig it needed torch at CALL time even
# though the import had been made lazy, so `lr_grid_cells.py --list` still died
# on a login node. Exercise the functions the analysis actually calls.
from src.scaling.configs import achievable_sizes, build_grid, lr_for
sizes = achievable_sizes()
assert len(sizes) > 100 and sizes[0][0] == 80965, sizes[:2]
grid = build_grid([1e14])
assert grid and grid[0].n_params > 0 and grid[0].lr > 0

import subprocess
for stage in ("probe", "1", "2", "check"):
    r = subprocess.run([sys.executable, "scripts/lr_grid_cells.py",
                        "--stage", stage], capture_output=True, text=True)
    assert r.returncode == 0, f"stage {stage}: {r.stderr}"
    assert r.stdout.strip(), f"stage {stage} emitted no cells"
# Stage 3 cannot emit runnable cells until stage 2 is fitted, by design, but it
# must still price itself before then: that is what --list is for.
for fmt in ("table", "count"):
    r = subprocess.run([sys.executable, "scripts/lr_grid_cells.py",
                        "--stage", "3", "--format", fmt],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stage 3 {fmt}: {r.stderr}"
    assert r.stdout.strip(), f"stage 3 {fmt} printed nothing"
import scripts.collect_isoflop      # noqa: F401
import scripts.fit_beta2            # noqa: F401
import scripts.merge_lr_parts       # noqa: F401

# And the wall-model analysis end to end.
import importlib.util
spec = importlib.util.spec_from_file_location("fwm", "scripts/fit_wall_model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.forward_cost(128, 2) > 0

assert "torch" not in sys.modules, "something imported torch at module scope"
print("ok")
"""


def test_scaling_analysis_imports_without_torch():
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


if __name__ == "__main__":
    test_scaling_analysis_imports_without_torch()
    print("PASS")
