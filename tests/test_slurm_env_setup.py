"""Every SLURM script must set up the environment the way this cluster needs.

Two scripts were written with an invented conda activation that does not exist
here, and swallowed the failure with `|| true`. Both reached a compute node,
ran against the system python and died on `import torch` after the scheduler
had already allocated a GPU.

This checks that every slurm either loads the cluster's module and venv, or
explicitly says it needs no python. It also forbids silencing the activation.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLURMS = sorted((REPO / "slurms").glob("*.sh"))

MODULE = "module load python/pytorch_251_311_cu124"
VENV = 'envs/llm-env/bin/activate'

# Patterns that mean "the activation failure is being hidden".
SILENCED = re.compile(r"(module load|bin/activate)[^\n]*(\|\|\s*true|2>\s*/dev/null)")


def test_every_slurm_sets_up_the_cluster_environment():
    missing, silenced = [], []
    for path in SLURMS:
        text = path.read_text()
        if "python" not in text:
            continue                      # nothing to set up
        if MODULE not in text or VENV not in text:
            missing.append(path.name)
        if SILENCED.search(text):
            silenced.append(path.name)
    assert not missing, (
        f"these slurms run python but do not load the cluster environment "
        f"({MODULE!r} + {VENV!r}): {missing}")
    assert not silenced, (
        f"these slurms hide an environment failure behind `|| true` or "
        f"/dev/null, which turns a 2-second error into a wasted allocation: "
        f"{silenced}")


def test_no_slurm_imports_modelconfig_inline():
    """Inline python in a slurm should use flops.ArchSpec, not ModelConfig.

    ModelConfig pulls in torch, which makes a cell-planning heredoc fail on a
    login node and inside a job whose environment did not come up.
    """
    offenders = [p.name for p in SLURMS if "from src.model import" in p.read_text()]
    assert not offenders, (
        f"these slurms import src.model inline, which needs torch; use "
        f"src.scaling.flops.ArchSpec instead: {offenders}")


if __name__ == "__main__":
    test_every_slurm_sets_up_the_cluster_environment()
    test_no_slurm_imports_modelconfig_inline()
    print("PASS")
