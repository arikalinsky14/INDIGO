#!/usr/bin/env bash
#SBATCH --job-name=chroma-constrained-train
#SBATCH --output=job-outputs/slurm-constrained-train.%j.out
#SBATCH --error=job-outputs/slurm-constrained-train.%j.err

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
# SLURM Training Script for Constrained LLM Text-to-RGB
# ============================================================================
#
# Fine-tunes TinyLlama with surgically reduced lm_head (45 valid tokens)
# to predict [R,G,B] from text prompts.
#
# USAGE:
#   # LoRA fine-tuning (recommended):
#   sbatch pretrain_text_to_rgb/slurms/slurm_training.sh
#
#   # Last-4 layers:
#   FINETUNE_MODE=last_n UNFREEZE_LAYERS=4 LR=5e-5 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_training.sh
#
#   # Quick test:
#   LIMIT_EXAMPLES=500 EPOCHS=2 BATCH_SIZE=4 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_training.sh
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

# Ensure dependencies are installed
pip install --quiet peft 2>/dev/null

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CONSTRAINED LLM TEXT-TO-RGB TRAINING - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || {
    echo "[WARN] transformers not available"
}
python -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null || {
    echo "[WARN] peft not available (required for LoRA mode)"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# -------------------- Data --------------------
DATA_DIR="${DATA_DIR:-}"
SEED="${SEED:-42}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"

# -------------------- Encoder --------------------
ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"

# -------------------- Fine-tuning Mode --------------------
FINETUNE_MODE="${FINETUNE_MODE:-lora}"     # lora | last_n | full
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-4}"

# -------------------- Optimization --------------------
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.03}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -------------------- Checkpointing --------------------
SAVE_DIR="${SAVE_DIR:-}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

# ============================================================================
# BUILD COMMAND
# ============================================================================

VERBOSE=0
USER_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--verbose" ]]; then
    VERBOSE=1
  else
    USER_ARGS+=("$arg")
  fi
done

ARGS=(
  --seed "${SEED}"
  --encoder "${ENCODER}"
  --max-text-len "${MAX_TEXT_LEN}"
  --finetune-mode "${FINETUNE_MODE}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-targets "${LORA_TARGETS}"
  --unfreeze-layers "${UNFREEZE_LAYERS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --warmup-fraction "${WARMUP_FRACTION}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --save-every "${SAVE_EVERY}"
)

if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

if [[ -n "${SAVE_DIR}" ]]; then
  ARGS+=(--save-dir "${SAVE_DIR}")
fi

if [[ $VERBOSE -eq 1 ]]; then
  ARGS+=(--verbose)
fi

CMD=(python pretrain_text_to_rgb/scripts/training.py "${ARGS[@]}" "${USER_ARGS[@]+"${USER_ARGS[@]}"}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "TRAINING CONFIGURATION"
echo "============================================================================"
echo
echo "Data:"
echo "  Seed:            ${SEED}"
echo "  Batch size:      ${BATCH_SIZE} micro x ${GRAD_ACCUM_STEPS} accum = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) effective"
echo "  Num workers:     ${NUM_WORKERS}"
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  echo "  Limited to:      ${LIMIT_EXAMPLES} examples"
fi
if [[ -n "${DATA_DIR}" ]]; then
  echo "  Data dir:        ${DATA_DIR}"
fi
echo
echo "Model:             ${ENCODER}"
echo "  Max text len:    ${MAX_TEXT_LEN}"
echo
echo "Fine-tuning:"
echo "  Mode:            ${FINETUNE_MODE}"
if [[ "${FINETUNE_MODE}" == "lora" ]]; then
  echo "  LoRA rank:       ${LORA_RANK}"
  echo "  LoRA alpha:      ${LORA_ALPHA}"
  echo "  LoRA targets:    ${LORA_TARGETS}"
elif [[ "${FINETUNE_MODE}" == "last_n" ]]; then
  echo "  Unfreeze layers: ${UNFREEZE_LAYERS}"
fi
echo
echo "Optimization:"
echo "  Learning rate:   ${LR}"
echo "  Weight decay:    ${WEIGHT_DECAY}"
echo "  Grad clip:       ${GRAD_CLIP}"
echo "  Warmup:          ${WARMUP_FRACTION}"
echo "  Epochs:          ${EPOCHS}"
echo
echo "Checkpointing:"
echo "  Save every:      ${SAVE_EVERY} steps"
if [[ -n "${SAVE_DIR}" ]]; then
  echo "  Save dir:        ${SAVE_DIR}"
else
  echo "  Save dir:        (auto-generated from hyperparameters)"
fi
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
echo "TRAINING COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"

exit ${EXIT_CODE}