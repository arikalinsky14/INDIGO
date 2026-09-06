#!/usr/bin/env bash
#SBATCH --job-name=indigo-diff-sim-smoke
#SBATCH --output=job-outputs/indigo-diff-sim-smoke.%j.out
#SBATCH --error=job-outputs/indigo-diff-sim-smoke.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Smoke test for the differentiable optical simulator
# ============================================================================
#
# Runs src/optical_sim_diff.py's built-in smoke test, which:
#   1. Verifies the differentiable forward matches OpticalSimulator's numpy
#      forward bit-for-bit (max Lab diff < 1e-6).
#   2. Finite-difference-checks the analytic thickness gradient from
#      jax.vjp against a central-difference numeric estimate (max err
#      < 1e-3).
#
# Should be re-run after any change to src/optical_sim_diff.py before
# starting a real finetune. ~1-2 min wall including JAX warmup.
#
# Usage:
#     sbatch slurms/smoke_optical_sim_diff.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "INDIGO DIFF-SIM SMOKE TEST - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python -m src.optical_sim_diff

EXIT_CODE=$?

echo
echo "============================================================================"
echo "SMOKE TEST COMPLETE   exit=${EXIT_CODE}   ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}
