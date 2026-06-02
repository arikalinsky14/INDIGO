#!/usr/bin/env bash
#SBATCH --job-name=indigo-eval
#SBATCH --output=job-outputs/indigo-eval.%j.out
#SBATCH --error=job-outputs/indigo-eval.%j.err

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
# SLURM Evaluation Script for INDIGO FlexMaterialMLP (evaluate.py)
# ============================================================================
#
# This script evaluates a trained INDIGO FlexMaterialMLP model with two modes:
#
# LOW-COMPUTE MODE (--low-compute):
#   - Fast evaluation with teacher forcing only
#   - Computes cross-entropy loss and token accuracy
#   - No autoregressive generation or optical simulation
#
# FULL MODE (default):
#   - Teacher forcing metrics (loss + accuracy)
#   - Autoregressive structure generation
#   - Optical simulation to compute predicted colors
#   - CIEDE2000 color difference metrics
#   - Color swatch visualization
#
# SAMPLING MODE (--sample-predictions):
#   - Use stochastic sampling instead of greedy argmax
#   - Reproducible given the same seed
#   - Control randomness with --temperature (default: 1.0)
#   - Output files include "sampling" in filename
#
# IMPORTANT: Pass the SAME hyperparameters used during training to identify
# the correct model checkpoint to load.
#
# USAGE EXAMPLES:
#
# 1. Full evaluation with default settings:
#    sbatch slurms/evaluate.sh
#
# 2. Fast evaluation (teacher forcing only):
#    sbatch slurms/evaluate.sh --low-compute
#
# 3. Evaluate with sampling:
#    sbatch slurms/evaluate.sh --sample-predictions
#
# 4. Evaluate with sampling at different temperatures:
#    sbatch slurms/evaluate.sh --sample-predictions --temperature 0.5
#    sbatch slurms/evaluate.sh --sample-predictions --temperature 1.5
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
echo "INDIGO EVALUATION RUN - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check for jaxlayerlumos (optional, for optical simulation)
python -c "from src.optical_sim import is_available; print(f'Optical simulation: {\"available\" if is_available() else \"not available\"}')" 2>/dev/null || {
    echo "[WARN] Optical simulation check failed - may not be available"
}

# Check for matplotlib (for color swatch generation)
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')" 2>/dev/null || {
    echo "[WARN] matplotlib not available - color swatch will be skipped"
}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# HYPERPARAMETERS - Must match training to identify model
# ============================================================================

# -------------------- Data & Evaluation Control --------------------
DATA_DIR="${DATA_DIR:-}"                   # Path to a parquet-shards directory;
                                           # empty -> evaluate.py defaults to
                                           # <repo>/data/train (override e.g.
                                           # DATA_DIR=data/test/tier_a)
SPLIT="${SPLIT:-validation}"               # Dataset split (train/validation)
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"       # Limit to N examples (for testing)
SEED="${SEED:-42}"                         # Random seed

# -------------------- Model Architecture (must match training) --------------------
FEATURE_MODE="${FEATURE_MODE:-raw_spectrum}"
ENCODER_HIDDEN="${ENCODER_HIDDEN:-128}"
ENCODER_OUT="${ENCODER_OUT:-64}"
ENCODER_DROPOUT="${ENCODER_DROPOUT:-0.1}"
D_MODEL="${D_MODEL:-1024}"
N_LAYERS="${N_LAYERS:-8}"
DROPOUT="${DROPOUT:-0.1}"
HEAD_MODE="${HEAD_MODE:-mlp}"              # 'mlp' or 'cross_attn' — MUST
                                           # match the trained checkpoint
                                           # (used for both architecture
                                           # build and tag-based lookup).
N_HEADS="${N_HEADS:-8}"                    # Attention heads (cross_attn only)
if [[ "${HEAD_MODE}" == "cross_attn" ]]; then
  SLOT_ENCODER_LAYERS="${SLOT_ENCODER_LAYERS:-4}"
else
  SLOT_ENCODER_LAYERS="${SLOT_ENCODER_LAYERS:-0}"
fi
DECODER_LAYERS="${DECODER_LAYERS:-1}"

# -------------------- Optimization (used to identify model) --------------------
LR="${LR:-4.42e-5}"                        # Learning rate (for checkpoint lookup)
BATCH_SIZE="${BATCH_SIZE:-64}"             # Batch size (for checkpoint lookup)
EPOCHS="${EPOCHS:-1}"                      # Epochs (for checkpoint lookup)

# -------------------- Evaluation Settings --------------------
CHECKPOINT="${CHECKPOINT:-}"               # Explicit checkpoint path (overrides tag lookup)
OUTPUT="${OUTPUT:-}"                       # Output file path (default: auto-generated)
SWATCH_EXAMPLES="${SWATCH_EXAMPLES:-10}"   # Number of examples in color swatch
NUM_WORKERS="${NUM_WORKERS:-4}"            # DataLoader workers

# -------------------- Sampling Settings --------------------
TEMPERATURE="${TEMPERATURE:-1.0}"          # Temperature for sampling (default: 1.0)

# ============================================================================
# COMMAND LINE OVERRIDE HANDLING
# ============================================================================

# Extract special flags and collect other user arguments
LOW_COMPUTE=0
NO_OPTICAL_SIM=0
NO_SWATCH=0
SAMPLE_PREDICTIONS=0
USER_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--low-compute" ]]; then
    LOW_COMPUTE=1
  elif [[ "$arg" == "--no-optical-sim" ]]; then
    NO_OPTICAL_SIM=1
  elif [[ "$arg" == "--no-swatch" ]]; then
    NO_SWATCH=1
  elif [[ "$arg" == "--sample-predictions" ]]; then
    SAMPLE_PREDICTIONS=1
  else
    USER_ARGS+=("$arg")
  fi
done

# Check for --temperature in USER_ARGS and extract it
NEW_USER_ARGS=()
i=0
while [[ $i -lt ${#USER_ARGS[@]} ]]; do
  if [[ "${USER_ARGS[$i]}" == "--temperature" ]]; then
    TEMPERATURE="${USER_ARGS[$((i+1))]}"
    i=$((i+2))
  else
    NEW_USER_ARGS+=("${USER_ARGS[$i]}")
    i=$((i+1))
  fi
done
USER_ARGS=("${NEW_USER_ARGS[@]}")

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  # Data & evaluation control
  --split "${SPLIT}"
  --seed "${SEED}"

  # Model architecture (for checkpoint lookup)
  --feature-mode "${FEATURE_MODE}"
  --encoder-hidden "${ENCODER_HIDDEN}"
  --encoder-out "${ENCODER_OUT}"
  --encoder-dropout "${ENCODER_DROPOUT}"
  --d-model "${D_MODEL}"
  --n-layers "${N_LAYERS}"
  --dropout "${DROPOUT}"
  --head-mode "${HEAD_MODE}"
  --n-heads "${N_HEADS}"
  --slot-encoder-layers "${SLOT_ENCODER_LAYERS}"
  --decoder-layers "${DECODER_LAYERS}"

  # Optimization (for checkpoint lookup)
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"

  # Evaluation settings
  --swatch-examples "${SWATCH_EXAMPLES}"
  --num-workers "${NUM_WORKERS}"
)

# Add optional data-dir if specified
if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

# Add optional limit-examples if specified
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

# Add optional checkpoint path if specified
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi

# Add optional output path if specified
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

# Add flags
if [[ $LOW_COMPUTE -eq 1 ]]; then
  ARGS+=(--low-compute)
fi

if [[ $NO_OPTICAL_SIM -eq 1 ]]; then
  ARGS+=(--no-optical-sim)
fi

if [[ $NO_SWATCH -eq 1 ]]; then
  ARGS+=(--no-swatch)
fi

# Add sampling flags
if [[ $SAMPLE_PREDICTIONS -eq 1 ]]; then
  ARGS+=(--sample-predictions)
  ARGS+=(--temperature "${TEMPERATURE}")
fi

# User arguments override defaults
CMD=(python scripts/evaluate.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "EVALUATION CONFIGURATION"
echo "============================================================================"
echo
echo "Mode:"
if [[ $LOW_COMPUTE -eq 1 ]]; then
  echo "  LOW-COMPUTE: Teacher forcing only (fast)"
else
  echo "  FULL: Teacher forcing + autoregressive + optical simulation"
fi
echo
if [[ $SAMPLE_PREDICTIONS -eq 1 ]]; then
  echo "Sampling:"
  echo "  ENABLED - Using stochastic sampling"
  echo "  Temperature:     ${TEMPERATURE}"
  echo "  Seed:            ${SEED}"
  echo "  (Output files will include 'sampling' in filename)"
else
  echo "Sampling:"
  echo "  DISABLED - Using greedy argmax"
fi
echo
echo "Data:"
echo "  Split:           ${SPLIT}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  Num workers:     ${NUM_WORKERS}"
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  echo "  Limited to:      ${LIMIT_EXAMPLES} examples"
fi
if [[ -n "${DATA_DIR}" ]]; then
  echo "  Data dir:        ${DATA_DIR}"
fi
echo
echo "Model Identification:"
echo "  head_mode:       ${HEAD_MODE}"
if [[ "${HEAD_MODE}" == "cross_attn" ]]; then
  echo "  n_heads:         ${N_HEADS}"
fi
echo "  d_model:         ${D_MODEL}"
echo "  n_layers:        ${N_LAYERS}"
echo "  dropout:         ${DROPOUT}"
echo "  Learning rate:   ${LR}"
echo "  Epochs:          ${EPOCHS}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "  Checkpoint:      ${CHECKPOINT}"
else
  echo "  Checkpoint:      (auto-detected from hyperparameters)"
fi
echo
echo "Output Settings:"
echo "  Swatch examples: ${SWATCH_EXAMPLES}"
if [[ $NO_OPTICAL_SIM -eq 1 ]]; then
  echo "  Optical sim:     DISABLED"
else
  echo "  Optical sim:     enabled (if available)"
fi
if [[ $NO_SWATCH -eq 1 ]]; then
  echo "  Color swatch:    DISABLED"
else
  echo "  Color swatch:    enabled (if matplotlib available)"
fi
if [[ -n "${OUTPUT}" ]]; then
  echo "  Output file:     ${OUTPUT}"
else
  echo "  Output file:     (auto-generated)"
fi
echo
echo "============================================================================"
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

# ============================================================================
# RUN EVALUATION
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

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. FULL EVALUATION (default model):
#    sbatch slurms/evaluate.sh
#
# 2. FAST EVALUATION (teacher forcing only):
#    sbatch slurms/evaluate.sh --low-compute
#
# 3. EVALUATE WITH SAMPLING (stochastic predictions):
#    sbatch slurms/evaluate.sh --sample-predictions
#
# 4. EVALUATE WITH SAMPLING AT LOWER TEMPERATURE (more deterministic):
#    sbatch slurms/evaluate.sh --sample-predictions --temperature 0.5
#
# 5. EVALUATE WITH SAMPLING AT HIGHER TEMPERATURE (more random):
#    sbatch slurms/evaluate.sh --sample-predictions --temperature 1.5
#
# 6. EVALUATE SPECIFIC MODEL (by hyperparameters):
#    sbatch slurms/evaluate.sh \
#      --lr 1e-3 --d-model 512 --n-layers 6 --epochs 15
#
# 7. EVALUATE WITH EXPLICIT CHECKPOINT:
#    sbatch slurms/evaluate.sh \
#      --checkpoint data/checkpoints/mlp_d256_L4_do0.1_lr0.002_bs256_ep15/latest
#
# 8. EVALUATE WITH LIMITED EXAMPLES:
#    sbatch slurms/evaluate.sh \
#      --limit-examples 500
#
# 9. SKIP OPTICAL SIMULATION (faster but no CIEDE2000):
#    sbatch slurms/evaluate.sh --no-optical-sim
#
# 10. SKIP COLOR SWATCH GENERATION:
#    sbatch slurms/evaluate.sh --no-swatch
#
# 11. COMBINE SAMPLING WITH OTHER OPTIONS:
#    sbatch slurms/evaluate.sh \
#      --sample-predictions \
#      --temperature 0.8 \
#      --limit-examples 1000 \
#      --lr 1e-3 \
#      --d-model 512
#
# 12. ENVIRONMENT VARIABLE OVERRIDE:
#    LR=1e-3 D_MODEL=512 EPOCHS=15 SPLIT=val \
#      sbatch slurms/evaluate.sh
#
# 13. TEMPERATURE SWEEP (compare different sampling temperatures):
#     for temp in 0.5 0.8 1.0 1.2 1.5; do
#       sbatch slurms/evaluate.sh \
#         --sample-predictions --temperature $temp
#     done
#
# ============================================================================