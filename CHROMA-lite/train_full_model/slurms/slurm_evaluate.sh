#!/usr/bin/env bash
#SBATCH --job-name=chroma-eval
#SBATCH --output=job-outputs/slurm-full-eval.%j.out
#SBATCH --error=job-outputs/slurm-full-eval.%j.err

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
# Evaluate Full Text → Structure Model
# ============================================================================
#
# Evaluates the trained constraint LoRA + ConstraintMLP + MixingMLP model.
#
# MODES:
#   Low-compute (teacher forcing only, fast):
#     LOW_COMPUTE=1 sbatch train_full_model/slurms/slurm_evaluate.sh
#
#   Full (teacher forcing + autoregressive + optical sim + CIEDE2000):
#     sbatch train_full_model/slurms/slurm_evaluate.sh
#
# USAGE:
#   # Default full eval (validation split, all examples):
#   CHECKPOINT=train_full_model/data/checkpoints/<tag>/latest \
#   CACHE_DIR=train_full_model/data/cache_test \
#   MLP_CHECKPOINT=pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
#     sbatch train_full_model/slurms/slurm_evaluate.sh
#
#   # Quick test (100 examples, low-compute):
#   LOW_COMPUTE=1 LIMIT_EXAMPLES=100 \
#   CHECKPOINT=train_full_model/data/checkpoints/<tag>/latest \
#   CACHE_DIR=train_full_model/data/cache_test \
#     sbatch train_full_model/slurms/slurm_evaluate.sh
#
#   # Full eval with sampling:
#   SAMPLE=1 TEMPERATURE=0.5 \
#   CHECKPOINT=train_full_model/data/checkpoints/<tag>/latest \
#   CACHE_DIR=train_full_model/data/cache_test \
#   MLP_CHECKPOINT=pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
#     sbatch train_full_model/slurms/slurm_evaluate.sh
#
#   # Constraint test set evaluation (1200 examples, no teacher forcing):
#   CONSTRAINT_TEST=1 \
#   CHECKPOINT=train_full_model/data/checkpoints/<tag>/latest \
#   MLP_CHECKPOINT=pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
#     sbatch train_full_model/slurms/slurm_evaluate.sh
#
#   # Override via command line:
#   sbatch train_full_model/slurms/slurm_evaluate.sh --limit-examples 500
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
echo "CHROMA-LITE FULL MODEL EVALUATION - Job ${SLURM_JOB_ID}"
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

# -------------------- Required Paths --------------------
CHECKPOINT="${CHECKPOINT:?'Set CHECKPOINT to the full model checkpoint dir'}"
# CACHE_DIR is required for low-compute and full mode, not for --constraint-test
CACHE_DIR="${CACHE_DIR:-}"

# MLP_CHECKPOINT required for full eval and constraint-test (not low-compute)
MLP_CHECKPOINT="${MLP_CHECKPOINT:-}"

# -------------------- Data --------------------
DATA_DIR="${DATA_DIR:-}"
SPLIT="${SPLIT:-validation}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"

# -------------------- Architecture (match training defaults) --------------------
D_MODEL="${D_MODEL:-1024}"
CONSTRAINT_LAYERS="${CONSTRAINT_LAYERS:-4}"
MIXING_LAYERS="${MIXING_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"

# -------------------- LLM / LoRA (match training defaults) --------------------
ENCODER="${ENCODER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-756}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"

# -------------------- Eval Settings --------------------
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
SWATCH_EXAMPLES="${SWATCH_EXAMPLES:-10}"
OUTPUT="${OUTPUT:-}"

# -------------------- Mode Flags --------------------
LOW_COMPUTE="${LOW_COMPUTE:-0}"
NO_OPTICAL_SIM="${NO_OPTICAL_SIM:-0}"
NO_SWATCH="${NO_SWATCH:-0}"
CONSTRAINT_TEST="${CONSTRAINT_TEST:-0}"
CONSTRAINT_TEST_CSV="${CONSTRAINT_TEST_CSV:-}"

# -------------------- Sampling --------------------
SAMPLE="${SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# ============================================================================
# BUILD AND RUN COMMAND
# ============================================================================

ARGS=()
ARGS+=(python train_full_model/scripts/evaluate.py)
ARGS+=(--checkpoint "${CHECKPOINT}")
[ -n "${CACHE_DIR}" ] && ARGS+=(--cache-dir "${CACHE_DIR}")
ARGS+=(--split "${SPLIT}")
ARGS+=(--seed "${SEED}")
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
ARGS+=(--num-workers "${NUM_WORKERS}")
ARGS+=(--swatch-examples "${SWATCH_EXAMPLES}")

[ -n "${LIMIT_EXAMPLES}" ] && ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
[ -n "${DATA_DIR}" ] && ARGS+=(--data-dir "${DATA_DIR}")
[ -n "${OUTPUT}" ] && ARGS+=(--output "${OUTPUT}")
[ -n "${MLP_CHECKPOINT}" ] && ARGS+=(--rgb-to-structure-checkpoint "${MLP_CHECKPOINT}")

[ "${LOW_COMPUTE}" = "1" ] && ARGS+=(--low-compute)
[ "${NO_OPTICAL_SIM}" = "1" ] && ARGS+=(--no-optical-sim)
[ "${NO_SWATCH}" = "1" ] && ARGS+=(--no-swatch)
[ "${CONSTRAINT_TEST}" = "1" ] && ARGS+=(--constraint-test)
[ -n "${CONSTRAINT_TEST_CSV}" ] && ARGS+=(--constraint-test-csv "${CONSTRAINT_TEST_CSV}")
[ "${SAMPLE}" = "1" ] && ARGS+=(--sample-predictions --temperature "${TEMPERATURE}")

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