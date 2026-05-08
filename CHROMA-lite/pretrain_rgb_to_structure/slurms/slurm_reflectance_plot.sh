#!/usr/bin/env bash
#SBATCH --job-name=chroma-reflectance
#SBATCH --output=job-outputs/slurm-reflectance.%j.out
#SBATCH --error=job-outputs/slurm-reflectance.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "REFLECTANCE COMPARISON PLOT - Job ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================================================"

# Configuration
CHECKPOINT="${CHECKPOINT:-pretrain_rgb_to_structure/data/checkpoints/mlp_d1024_L8_do0.1_lr0.000686_bs512_ep200/latest}"
SPLIT="${SPLIT:-validation}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-1.0}"

echo "Checkpoint:   ${CHECKPOINT}"
echo "Split:        ${SPLIT}"
echo "Temperature:  ${TEMPERATURE}"
echo "Seed:         ${SEED}"
echo

python pretrain_rgb_to_structure/scripts/plot_reflectance_comparison.py \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    "$@"

echo "============================================================================"
echo "COMPLETE - $(date)"
echo "============================================================================"

# ============================================================================
# USAGE
# ============================================================================
#
# Default (2 spectra + 16 swatch pairs, temp=1.0, validation set):
#   sbatch pretrain_rgb_to_structure/slurms/slurm_reflectance_plot.sh
#
# Custom checkpoint:
#   CHECKPOINT=pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
#     sbatch pretrain_rgb_to_structure/slurms/slurm_reflectance_plot.sh
#
# Lower temperature:
#   TEMPERATURE=0.7 sbatch pretrain_rgb_to_structure/slurms/slurm_reflectance_plot.sh
#
# ============================================================================