#!/usr/bin/env bash
#SBATCH --job-name=chroma-lr
#SBATCH --output=job-outputs/slurm-full-lr.%j.out
#SBATCH --error=job-outputs/slurm-full-lr.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# LR Tuning for Full Text → Structure Model
# ============================================================================
#
# USAGE:
#   # Default (8 LRs, 3 epochs):
#   sbatch train_full_model/slurms/slurm_lr_tuning.sh
#
#   # With limited examples for quick scaling study:
#   LIMIT_EXAMPLES=5000 EPOCHS=3 sbatch train_full_model/slurms/slurm_lr_tuning.sh
#   LIMIT_EXAMPLES=20000 EPOCHS=3 sbatch train_full_model/slurms/slurm_lr_tuning.sh
#   LIMIT_EXAMPLES=100000 EPOCHS=3 sbatch train_full_model/slurms/slurm_lr_tuning.sh
#
#   # Custom LR range:
#   LR_MIN=1e-5 LR_MAX=1e-2 N_LRS=10 sbatch train_full_model/slurms/slurm_lr_tuning.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CHROMA-LITE FULL MODEL LR TUNING - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR_TRAIN="${CACHE_DIR_TRAIN:-train_full_model/data/cache_train}"
CACHE_DIR_VAL="${CACHE_DIR_VAL:-train_full_model/data/cache_test}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"
LIMIT_VAL="${LIMIT_VAL:-}"

# LR search
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
N_LRS="${N_LRS:-8}"
EPOCHS="${EPOCHS:-3}"

# Architecture
D_MODEL="${D_MODEL:-1024}"
CONSTRAINT_LAYERS="${CONSTRAINT_LAYERS:-4}"
MIXING_LAYERS="${MIXING_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"

# LLM / LoRA
ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"

# Training
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Performance (all ON by default for throughput)
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
COMPILE="${COMPILE:-1}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ============================================================================
# BUILD AND RUN COMMAND
# ============================================================================

ARGS=()
ARGS+=(python train_full_model/scripts/lr_tuning.py)
ARGS+=(--cache-dir-train "${CACHE_DIR_TRAIN}")
ARGS+=(--cache-dir-val "${CACHE_DIR_VAL}")
ARGS+=(--lr-min "${LR_MIN}")
ARGS+=(--lr-max "${LR_MAX}")
ARGS+=(--n-lrs "${N_LRS}")
ARGS+=(--epochs "${EPOCHS}")
ARGS+=(--d-model "${D_MODEL}")
ARGS+=(--constraint-layers "${CONSTRAINT_LAYERS}")
ARGS+=(--mixing-layers "${MIXING_LAYERS}")
ARGS+=(--dropout "${DROPOUT}")
ARGS+=(--encoder "${ENCODER}")
ARGS+=(--max-text-len "${MAX_TEXT_LEN}")
ARGS+=(--lora-rank "${LORA_RANK}")
ARGS+=(--lora-alpha "${LORA_ALPHA}")
ARGS+=(--lora-targets "${LORA_TARGETS}")
ARGS+=(--batch-size "${BATCH_SIZE}")
ARGS+=(--grad-accum-steps "${GRAD_ACCUM}")
ARGS+=(--weight-decay "${WEIGHT_DECAY}")
ARGS+=(--grad-clip "${GRAD_CLIP}")
ARGS+=(--warmup-fraction "${WARMUP_FRACTION}")
ARGS+=(--num-workers "${NUM_WORKERS}")
ARGS+=(--mixed-precision "${MIXED_PRECISION}")
ARGS+=(--plot --verbose)

[ -n "${LIMIT_EXAMPLES}" ] && ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
[ -n "${LIMIT_VAL}" ] && ARGS+=(--limit-val "${LIMIT_VAL}")
[ -n "${OUTPUT_DIR}" ] && ARGS+=(--output-dir "${OUTPUT_DIR}")
[ -n "${GRADIENT_CHECKPOINTING}" ] && ARGS+=(--gradient-checkpointing)
[ -n "${COMPILE}" ] && ARGS+=(--compile)

ARGS+=("$@")

echo "Command: ${ARGS[*]}"
echo "============================================================================"
echo

"${ARGS[@]}"
EXIT_CODE=$?

echo
echo "============================================================================"
echo "COMPLETE - $(date) — exit code: ${EXIT_CODE}"
echo "============================================================================"

exit ${EXIT_CODE}
