#!/usr/bin/env bash
#SBATCH --job-name=chroma-curves
#SBATCH --output=job-outputs/slurm-curves.%j.out
#SBATCH --error=job-outputs/slurm-curves.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=12:00:00
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
echo "TRAINING CURVES EVALUATION - Job ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================================================"

# Default values
CHECKPOINT_DIR="${CHECKPOINT_DIR:-pretrain_rgb_to_structure/data/checkpoints/mlp_d1024_L8_do0.1_lr0.001_bs256_ep200}"
START_STEP="${START_STEP:-1000}"
END_STEP="${END_STEP:-233000}"
STEP_INTERVAL="${STEP_INTERVAL:-1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-2000}"

python pretrain_rgb_to_structure/scripts/plot_training_curves.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --start-step "${START_STEP}" \
    --end-step "${END_STEP}" \
    --step-interval "${STEP_INTERVAL}" \
    --eval-examples "${EVAL_EXAMPLES}" \
    --plot \
    "$@"

echo "============================================================================"
echo "COMPLETE - $(date)"
echo "============================================================================"