#!/usr/bin/env bash
#SBATCH --job-name=chroma-lr-tune
#SBATCH --output=job-outputs/slurm-lr-tune.%j.out
#SBATCH --error=job-outputs/slurm-lr-tune.%j.err

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
# SLURM LR Tuning Script for Constrained LLM Text-to-RGB
# ============================================================================
#
# USAGE:
#   # Default (8 LRs, 1 epoch, LoRA, 5000 examples):
#   LIMIT_EXAMPLES=5000 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_lr_tuning.sh
#
#   # Sweep data sizes (one job per size):
#   for n in 1000 5000 10000 50000; do
#     LIMIT_EXAMPLES=$n sbatch pretrain_text_to_rgb/slurms/slurm_lr_tuning.sh
#   done
#
#   # Custom LR range:
#   LIMIT_EXAMPLES=5000 LR_MIN=1e-5 LR_MAX=1e-2 N_LRS=12 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_lr_tuning.sh
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

pip install --quiet peft 2>/dev/null

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CONSTRAINED LLM LR TUNING - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null
python -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data
DATA_DIR="${DATA_DIR:-}"
SEED="${SEED:-42}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"

# Model
ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"

# Fine-tuning mode
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-4}"

# LR search
LR_MIN="${LR_MIN:-5e-5}"
LR_MAX="${LR_MAX:-5e-3}"
N_LRS="${N_LRS:-8}"

# Training per trial
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.03}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --seed "${SEED}"
  --val-fraction "${VAL_FRACTION}"
  --encoder "${ENCODER}"
  --max-text-len "${MAX_TEXT_LEN}"
  --finetune-mode "${FINETUNE_MODE}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-targets "${LORA_TARGETS}"
  --unfreeze-layers "${UNFREEZE_LAYERS}"
  --lr-min "${LR_MIN}"
  --lr-max "${LR_MAX}"
  --n-lrs "${N_LRS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --warmup-fraction "${WARMUP_FRACTION}"
)

if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

# Pass through any extra CLI args
ARGS+=("$@")

CMD=(python pretrain_text_to_rgb/scripts/lr_tuning.py "${ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo "============================================================================"
echo "LR TUNING CONFIGURATION"
echo "============================================================================"
echo
echo "Data:"
echo "  Seed:            ${SEED}"
echo "  Val fraction:    ${VAL_FRACTION}"
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  echo "  Limited to:      ${LIMIT_EXAMPLES} examples"
fi
echo
echo "Model:             ${ENCODER}"
echo
echo "Fine-tuning:       ${FINETUNE_MODE}"
if [[ "${FINETUNE_MODE}" == "lora" ]]; then
  echo "  LoRA rank:       ${LORA_RANK}"
  echo "  LoRA alpha:      ${LORA_ALPHA}"
  echo "  LoRA targets:    ${LORA_TARGETS}"
elif [[ "${FINETUNE_MODE}" == "last_n" ]]; then
  echo "  Unfreeze layers: ${UNFREEZE_LAYERS}"
fi
echo
echo "LR Search:"
echo "  Range:           [${LR_MIN}, ${LR_MAX}]"
echo "  N candidates:    ${N_LRS}"
echo "  Epochs/trial:    ${EPOCHS}"
echo "  Batch size:      ${BATCH_SIZE} micro x ${GRAD_ACCUM_STEPS} accum = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) effective"
echo
echo "============================================================================"
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

# ============================================================================
# RUN
# ============================================================================

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "LR TUNING COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"

exit ${EXIT_CODE}