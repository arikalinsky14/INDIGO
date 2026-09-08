#!/usr/bin/env bash
#SBATCH --job-name=indigo-ste-proj
#SBATCH --output=job-outputs/indigo-ste-proj.%j.out
#SBATCH --error=job-outputs/indigo-ste-proj.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=03:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# STE Projection Quality Test — wrapper
# ============================================================================
#
# Runs analyses/de_finetune/ste_projection_quality.py to quantify how
# well the ΔE finetune's linear projection of the sim gradient onto every
# pool material approximates the true "swap and re-sim" ΔE.
#
# Verdict (rule of thumb, exact thresholds in the summary):
#   HOLDS   — keep current STE
#   MIXED   — worth trying winning-material-only as ablation
#   NOISE   — switch to winning-material-only
#
# CPU-only (no GPU) via JAX; sim runs at a few tens of ms per call.
# Cost: N_EXAMPLES × avg_layers × avg_pool_size sims + JAX warmup.
# Ballpark: 500 examples × 4 layers × 25 materials = 50k sims at ~30 ms
# each = ~25 min on 4 CPU cores after warmup. Time budget generous.
#
# Environment variables (all optional; sensible defaults):
#   DATA_DIR              default: data/finetune
#   SPLIT                 default: all
#   N_EXAMPLES            default: 500
#   LAYERS_PER_EXAMPLE    default: all   ('all', 'random', or an int)
#   INCIDENCE_ANGLE       default: 0
#   SEED                  default: 42
#   OUTPUT_DIR            default: analyses/de_finetune/results/ste_projection_<JOBID>
#
# Usage:
#   sbatch slurms/ste_projection_quality.sh
#
# Quick smoke (50 examples, 1 layer each):
#   N_EXAMPLES=50 LAYERS_PER_EXAMPLE=random \
#       sbatch --time=00:30:00 slurms/ste_projection_quality.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# JAX CPU-only + silence noisy CUDA probes we're not using.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# ---- Defaults ----
: "${DATA_DIR:=data/finetune}"
: "${SPLIT:=all}"
: "${N_EXAMPLES:=500}"
: "${LAYERS_PER_EXAMPLE:=all}"
: "${INCIDENCE_ANGLE:=0}"
: "${SEED:=42}"
: "${OUTPUT_DIR:=analyses/de_finetune/results/ste_projection_${SLURM_JOB_ID:-local}}"

echo "============================================================================"
echo "INDIGO STE PROJECTION QUALITY - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:              $(pwd)"
echo "Node:             $(hostname)"
echo "Python:           $(which python)"
echo "Started:          $(date)"
echo
python --version
python -c "import jax, jaxlayerlumos; print(f'JAX: {jax.__version__}  jaxlayerlumos: {jaxlayerlumos.__version__}')" 2>/dev/null || echo "[WARN] JAX / jaxlayerlumos import check failed"
echo
echo "----- Configuration -----"
echo "DATA_DIR            : ${DATA_DIR}"
echo "SPLIT               : ${SPLIT}"
echo "N_EXAMPLES          : ${N_EXAMPLES}"
echo "LAYERS_PER_EXAMPLE  : ${LAYERS_PER_EXAMPLE}"
echo "INCIDENCE_ANGLE     : ${INCIDENCE_ANGLE}"
echo "SEED                : ${SEED}"
echo "OUTPUT_DIR          : ${OUTPUT_DIR}"
echo "============================================================================"
echo

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python analyses/de_finetune/ste_projection_quality.py
    --data-dir            "${DATA_DIR}"
    --split               "${SPLIT}"
    --n-examples          "${N_EXAMPLES}"
    --layers-per-example  "${LAYERS_PER_EXAMPLE}"
    --incidence-angle     "${INCIDENCE_ANGLE}"
    --seed                "${SEED}"
    --output-dir          "${OUTPUT_DIR}"
)

echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "STE PROJECTION QUALITY COMPLETE   exit=${EXIT_CODE}   ended=$(date)"
echo "OUTPUTS:  ${OUTPUT_DIR}/"
echo "  - results.json"
echo "  - summary.txt"
echo "  - projected_vs_true.png"
echo "  - spearman_hist.png"
echo "  - sign_agreement_hist.png"
echo "============================================================================"
exit ${EXIT_CODE}
