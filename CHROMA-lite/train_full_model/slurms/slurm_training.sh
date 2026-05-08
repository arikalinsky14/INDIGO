#!/usr/bin/env bash
#SBATCH --job-name=chroma-full
#SBATCH --output=job-outputs/slurm-full-train.%j.out
#SBATCH --error=job-outputs/slurm-full-train.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=nvlink_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=72:00:00
#SBATCH --qos=long
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Train Full Text → Structure Model
# ============================================================================
#
# Trains constraint LoRA + ConstraintMLP + MixingMLP on cached base logits.
#
# USAGE:
#   # Default (uses cache_train, 3 epochs, full dataset):
#   sbatch train_full_model/slurms/slurm_training.sh
#
#   # Custom cache directory:
#   CACHE_DIR=train_full_model/data/cache_train \
#     sbatch train_full_model/slurms/slurm_training.sh
#
#   # Quick test:
#   LIMIT_EXAMPLES=1000 EPOCHS=2 BATCH_SIZE=4 \
#     sbatch train_full_model/slurms/slurm_training.sh
#
#   # Override via command line:
#   sbatch train_full_model/slurms/slurm_training.sh --lr 3e-4 --epochs 5
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CHROMA-LITE FULL MODEL TRAINING - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "[WARN] transformers not available"
python -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null || echo "[WARN] peft not available"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

# -------------------- Data --------------------
CACHE_DIR="${CACHE_DIR:-train_full_model/data/cache_train}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"

# -------------------- Architecture --------------------
D_MODEL="${D_MODEL:-1024}"
CONSTRAINT_LAYERS="${CONSTRAINT_LAYERS:-4}"
MIXING_LAYERS="${MIXING_LAYERS:-10}"
DROPOUT="${DROPOUT:-0.1}"

# -------------------- LLM / LoRA --------------------
ENCODER="${ENCODER:-meta-llama/Meta-Llama-3.1-8B}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"

# -------------------- Training --------------------
LR="${LR:-4.42e-05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -------------------- Performance (all ON by default for throughput) ----------
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
COMPILE="${COMPILE:-1}"

# -------------------- Checkpointing --------------------
SAVE_DIR="${SAVE_DIR:-}"
SAVE_EVERY="${SAVE_EVERY:-4000}"

# ============================================================================
# BUILD AND RUN COMMAND
# ============================================================================

ARGS=()
ARGS+=(python train_full_model/scripts/training.py)
ARGS+=(--cache-dir "${CACHE_DIR}")
ARGS+=(--d-model "${D_MODEL}")
ARGS+=(--constraint-layers "${CONSTRAINT_LAYERS}")
ARGS+=(--mixing-layers "${MIXING_LAYERS}")
ARGS+=(--dropout "${DROPOUT}")
ARGS+=(--encoder "${ENCODER}")
ARGS+=(--max-text-len "${MAX_TEXT_LEN}")
ARGS+=(--lora-rank "${LORA_RANK}")
ARGS+=(--lora-alpha "${LORA_ALPHA}")
ARGS+=(--lora-targets "${LORA_TARGETS}")
ARGS+=(--lr "${LR}")
ARGS+=(--weight-decay "${WEIGHT_DECAY}")
ARGS+=(--batch-size "${BATCH_SIZE}")
ARGS+=(--grad-accum-steps "${GRAD_ACCUM}")
ARGS+=(--epochs "${EPOCHS}")
ARGS+=(--grad-clip "${GRAD_CLIP}")
ARGS+=(--warmup-fraction "${WARMUP_FRACTION}")
ARGS+=(--num-workers "${NUM_WORKERS}")
ARGS+=(--save-every "${SAVE_EVERY}")
ARGS+=(--mixed-precision "${MIXED_PRECISION}")

[ -n "${LIMIT_EXAMPLES}" ] && ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
[ -n "${SAVE_DIR}" ] && ARGS+=(--save-dir "${SAVE_DIR}")
[ -n "${GRADIENT_CHECKPOINTING}" ] && ARGS+=(--gradient-checkpointing)
[ -n "${COMPILE}" ] && ARGS+=(--compile)

# Append any extra args from sbatch command line
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