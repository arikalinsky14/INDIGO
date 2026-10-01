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

# And the thing this exists for: the wall-model analysis end to end.
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
