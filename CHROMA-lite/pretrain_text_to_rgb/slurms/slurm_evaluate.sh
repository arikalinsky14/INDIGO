#!/usr/bin/env bash
#SBATCH --job-name=chroma-constrained-eval
#SBATCH --output=job-outputs/slurm-constrained-eval.%j.out
#SBATCH --error=job-outputs/slurm-constrained-eval.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# SLURM Evaluation Script for Constrained LLM Text-to-RGB
# ============================================================================
#
# Two modes:
#   1. Zero-shot (no checkpoint): Tests base TinyLlama + lm_head surgery
#   2. Fine-tuned (--checkpoint): Tests a fine-tuned checkpoint
#
# USAGE:
#   # Zero-shot baseline:
#   sbatch pretrain_text_to_rgb/slurms/slurm_evaluate.sh
#
#   # Fine-tuned checkpoint:
#   CHECKPOINT=pretrain_text_to_rgb/data/checkpoints/<tag>/best \
#     sbatch pretrain_text_to_rgb/slurms/slurm_evaluate.sh
#
#   # Quick test:
#   LIMIT_EXAMPLES=50 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_evaluate.sh
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
echo "CONSTRAINED LLM TEXT-TO-RGB EVALUATION - Job ${SLURM_JOB_ID}"
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
    echo "[WARN] peft not available (needed for LoRA checkpoints)"
}
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')" 2>/dev/null || {
    echo "[WARN] matplotlib not available - color swatch will be skipped"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT="${CHECKPOINT:-}"               # Empty = zero-shot
DATA_DIR="${DATA_DIR:-}"
SPLIT="${SPLIT:-validation}"
SEED="${SEED:-42}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"

ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-14}"
TEMPERATURE="${TEMPERATURE:-0.0}"

SWATCH_EXAMPLES="${SWATCH_EXAMPLES:-10}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-20}"
OUTPUT="${OUTPUT:-}"

# ============================================================================
# PARSE COMMAND LINE
# ============================================================================

NO_SWATCH=0
USER_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--no-swatch" ]]; then
    NO_SWATCH=1
  else
    USER_ARGS+=("$arg")
  fi
done

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --split "${SPLIT}"
  --seed "${SEED}"
  --encoder "${ENCODER}"
  --max-text-len "${MAX_TEXT_LEN}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --swatch-examples "${SWATCH_EXAMPLES}"
  --show-examples "${SHOW_EXAMPLES}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

if [[ $NO_SWATCH -eq 1 ]]; then
  ARGS+=(--no-swatch)
fi

CMD=(python pretrain_text_to_rgb/scripts/evaluate.py "${ARGS[@]}" "${USER_ARGS[@]+"${USER_ARGS[@]}"}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "EVALUATION CONFIGURATION"
echo "============================================================================"
echo
if [[ -n "${CHECKPOINT}" ]]; then
  echo "Mode:              FINE-TUNED"
  echo "Checkpoint:        ${CHECKPOINT}"
else
  echo "Mode:              ZERO-SHOT (base model + lm_head surgery)"
fi
echo
echo "Data:"
echo "  Split:           ${SPLIT}"
echo "  Seed:            ${SEED}"
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  echo "  Limited to:      ${LIMIT_EXAMPLES} examples"
fi
echo
echo "Generation:"
echo "  Max new tokens:  ${MAX_NEW_TOKENS}"
echo "  Temperature:     ${TEMPERATURE}"
echo
echo "Swatch:"
if [[ $NO_SWATCH -eq 1 ]]; then
  echo "  DISABLED"
else
  echo "  ${SWATCH_EXAMPLES} examples"
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
echo "EVALUATION COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"

exit ${EXIT_CODE}