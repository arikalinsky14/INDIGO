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

# ============================================================================
# SLURM Training Curves - Constrained LLM Text-to-RGB
# ============================================================================
#
# USAGE:
#   CHECKPOINT_DIR="pretrain_text_to_rgb/data/checkpoints/<tag>" \
#     sbatch pretrain_text_to_rgb/slurms/slurm_training_curves.sh
#
#   # Custom step range:
#   CHECKPOINT_DIR="..." START_STEP=100 END_STEP=5000 STEP_INTERVAL=100 \
#     sbatch pretrain_text_to_rgb/slurms/slurm_training_curves.sh
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
echo "CONSTRAINED LLM TRAINING CURVES - Job ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

START_STEP="${START_STEP:-50}"
END_STEP="${END_STEP:-5000}"
STEP_INTERVAL="${STEP_INTERVAL:-50}"

DATA_DIR="${DATA_DIR:-}"
ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"

TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-5000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-42}"

SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-5}"
OUTPUT="${OUTPUT:-}"

# ============================================================================
# PARSE CLI ARGS (override env vars)
# ============================================================================

PLOT=1
USER_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)   CHECKPOINT_DIR="$2"; shift 2 ;;
        --start-step)       START_STEP="$2"; shift 2 ;;
        --end-step)         END_STEP="$2"; shift 2 ;;
        --step-interval)    STEP_INTERVAL="$2"; shift 2 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --encoder)          ENCODER="$2"; shift 2 ;;
        --train-examples)   TRAIN_EXAMPLES="$2"; shift 2 ;;
        --eval-examples)    EVAL_EXAMPLES="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --smoothing-window) SMOOTHING_WINDOW="$2"; shift 2 ;;
        --output)           OUTPUT="$2"; shift 2 ;;
        --no-plot)          PLOT=0; shift ;;
        *)                  USER_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    echo "[ERROR] CHECKPOINT_DIR is required"
    echo "  Usage: CHECKPOINT_DIR=<path> sbatch ..."
    exit 1
fi

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --start-step "${START_STEP}"
    --end-step "${END_STEP}"
    --step-interval "${STEP_INTERVAL}"
    --encoder "${ENCODER}"
    --max-text-len "${MAX_TEXT_LEN}"
    --train-examples "${TRAIN_EXAMPLES}"
    --eval-examples "${EVAL_EXAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --smoothing-window "${SMOOTHING_WINDOW}"
)

if [[ -n "${DATA_DIR}" ]]; then
    ARGS+=(--data-dir "${DATA_DIR}")
fi

if [[ -n "${OUTPUT}" ]]; then
    ARGS+=(--output "${OUTPUT}")
fi

if [[ $PLOT -eq 1 ]]; then
    ARGS+=(--plot)
fi

CMD=(python pretrain_text_to_rgb/scripts/plot_training_curves.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo
echo "Checkpoint:        ${CHECKPOINT_DIR}"
echo "Step range:        ${START_STEP} to ${END_STEP} (every ${STEP_INTERVAL})"
echo "Train examples:    ${TRAIN_EXAMPLES}"
echo "Eval examples:     ${EVAL_EXAMPLES}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Plot:              $([ $PLOT -eq 1 ] && echo 'yes' || echo 'no')"
echo
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "TRAINING CURVES COMPLETE — Exit: ${EXIT_CODE} — $(date)"
echo "============================================================================"

exit ${EXIT_CODE}