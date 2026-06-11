#!/usr/bin/env bash
#SBATCH --job-name=indigo-inf-spike
#SBATCH --output=job-outputs/indigo-inf-spike.%j.out
#SBATCH --error=job-outputs/indigo-inf-spike.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# INDIGO inference physics-chain de-risker
# ============================================================================
#
# Runs `inference/scripts/sim_spike.py` — exercises the JAX-native physics
# chain end-to-end:
#   (1) reflectance matches src/optical_sim.py
#   (2) Lab matches src/color_utils.spectrum_to_lab
#   (3) ΔE_00 matches the reference numpy port
#   (4) jax.grad of ΔE wrt thicknesses matches centered finite differences
#   (5) forward + grad timing (no jit — JLL assert blocks it; see
#       inference/src/simulate.py docstring)
#   (6) schema round-trip
#
# CPU-only is fine; the chain doesn't touch the model. Use this whenever
# you bump JAX / jaxlayerlumos versions or change inference/src/simulate.py.
# ============================================================================

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124   # gives us jax + jaxlayerlumos + torch

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"   # spike is CPU-bound; opt in to GPU
                                               # via JAX_PLATFORMS=cuda if desired

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "INDIGO INFERENCE SPIKE - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo "Devices:  $(python -c 'import jax; print(jax.devices())' 2>/dev/null || echo unavailable)"
echo

CMD=(python inference/scripts/sim_spike.py)
echo "COMMAND: ${CMD[*]}"
echo
"${CMD[@]}"
EXIT_CODE=$?

echo
echo "============================================================================"
echo "INFERENCE SPIKE COMPLETE  exit=${EXIT_CODE}  ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE
# ============================================================================
#   sbatch slurms/inference_spike.sh
#
# Override the JAX backend (default: CPU):
#   JAX_PLATFORMS=cuda sbatch slurms/inference_spike.sh    # use GPU if available
#
# Override the JLL materials location used by the spike (it loads a tiny
# synthetic pool if no real materials are available, so this only matters
# if you want to exercise the spike against a specific JLL build):
#   JLL_MATERIALS_DIR=/path/to/materials sbatch slurms/inference_spike.sh
# ============================================================================
