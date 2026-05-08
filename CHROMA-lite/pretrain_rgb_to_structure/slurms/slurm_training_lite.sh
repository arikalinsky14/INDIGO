#!/usr/bin/env bash
#SBATCH --job-name=chroma-lite-train
#SBATCH --output=job-outputs/slurm-lite-train.%j.out
#SBATCH --error=job-outputs/slurm-lite-train.%j.err

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
# SLURM Training Script for CHROMA-Lite MLP (training.py)
# ============================================================================
#
# This script trains the CHROMA-Lite MLP model with:
# - RGB-conditioned thin-film structure generation
# - Simple feedforward architecture (no transformer)
# - Learning rate schedule: linear warmup + cosine decay
# - Checkpoints saved to: pretrain_rgb_to_structure/data/checkpoints/<hparams_tag>/
#
# ARCHITECTURE:
#   Input (203) = RGB (3) + flattened structure (25*8=200)
#   -> Dense(d_model) -> ReLU -> Dropout
#   -> [Dense(d_model) -> ReLU -> Dropout] x (n_layers - 1)
#   -> Dense(vocab_size=1001)
#
# USAGE EXAMPLES:
#
# 1. Full production training (default settings):
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh
#
# 2. Quick validation run (limit to 1000 examples):
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh \
#      --limit-examples 1000
#
# 3. Custom hyperparameters:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh \
#      --lr 1e-3 --d-model 512 --n-layers 6 --epochs 20
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
echo "CHROMA-LITE MLP TRAINING RUN - Job ${SLURM_JOB_ID}"
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
# HYPERPARAMETERS - Organized by Category
# ============================================================================

# -------------------- Data & Training Control --------------------
DATA_DIR="${DATA_DIR:-}"                   # Path to data_prompts/ (default: auto-detect)
SPLIT="${SPLIT:-train}"                    # Dataset split (train/validation)
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"       # Limit to N examples (for testing)
SEED="${SEED:-42}"                         # Random seed

# -------------------- Model Architecture (MLP) --------------------
D_MODEL="${D_MODEL:-256}"                  # Hidden layer dimension
N_LAYERS="${N_LAYERS:-4}"                  # Number of hidden layers
DROPOUT="${DROPOUT:-0.1}"                  # Dropout rate

# -------------------- Optimization --------------------
LR="${LR:-6.86e-4}"                           # Base learning rate (higher for MLP)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"       # AdamW weight decay
GRAD_CLIP="${GRAD_CLIP:-1.0}"              # Gradient clipping norm
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}" # Warmup as % of training (2% = default)
EPOCHS="${EPOCHS:-1}"                      # Number of training epochs

# -------------------- Data Loading --------------------
BATCH_SIZE="${BATCH_SIZE:-256}"            # Batch size (larger for MLP)
NUM_WORKERS="${NUM_WORKERS:-4}"            # DataLoader workers

# -------------------- Checkpointing --------------------
SAVE_DIR="${SAVE_DIR:-}"                   # Override checkpoint dir (default: auto-generated)
SAVE_EVERY="${SAVE_EVERY:-1000}"           # Save checkpoint every N steps

# ============================================================================
# COMMAND LINE OVERRIDE HANDLING
# ============================================================================

# Extract --verbose flag and collect other user arguments
VERBOSE=0
USER_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--verbose" ]]; then
    VERBOSE=1
  else
    USER_ARGS+=("$arg")
  fi
done

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  # Data & training control
  --split "${SPLIT}"
  --seed "${SEED}"
  
  # Model architecture (MLP)
  --d-model "${D_MODEL}"
  --n-layers "${N_LAYERS}"
  --dropout "${DROPOUT}"
  
  # Optimization
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --warmup-fraction "${WARMUP_FRACTION}"
  --epochs "${EPOCHS}"
  
  # Data loading
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  
  # Checkpointing
  --save-every "${SAVE_EVERY}"
)

# Add optional data-dir if specified
if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

# Add optional limit-examples if specified
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

# Add optional save-dir if specified
if [[ -n "${SAVE_DIR}" ]]; then
  ARGS+=(--save-dir "${SAVE_DIR}")
fi

# Add verbose flag if set
if [[ $VERBOSE -eq 1 ]]; then
  ARGS+=(--verbose)
fi

# User arguments override defaults
CMD=(python pretrain_rgb_to_structure/scripts/training.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "TRAINING CONFIGURATION"
echo "============================================================================"
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
echo "Model Architecture (MLP):"
echo "  d_model:         ${D_MODEL}"
echo "  n_layers:        ${N_LAYERS}"
echo "  dropout:         ${DROPOUT}"
echo "  Input dim:       203 (RGB=3 + structure=200)"
echo "  Output dim:      1001 (vocab size)"
echo
echo "Optimization:"
echo "  Learning rate:   ${LR}"
echo "  Weight decay:    ${WEIGHT_DECAY}"
echo "  Grad clip:       ${GRAD_CLIP}"
echo "  Warmup:          ${WARMUP_FRACTION} (fraction of training)"
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
# RUN TRAINING
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

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. FULL PRODUCTION RUN (recommended settings):
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --epochs 15
#
# 2. QUICK TEST (small subset):
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh \
#      --limit-examples 10000 --epochs 5
#
# 3. HYPERPARAMETER SWEEP - Learning Rate:
#    for lr in 1e-3 2e-3 5e-3; do
#      sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --lr $lr --epochs 10
#    done
#
# 4. HYPERPARAMETER SWEEP - Model Size:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --d-model 128 --n-layers 3
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --d-model 256 --n-layers 4
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --d-model 512 --n-layers 6
#
# 5. LARGER MODEL WITH MORE EPOCHS:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh \
#      --d-model 512 --n-layers 6 --epochs 20 --lr 1e-3
#
# 6. HIGHER DROPOUT (regularization):
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --dropout 0.2 --epochs 15
#
# 7. VERBOSE OUTPUT:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --verbose --epochs 10
#
# 8. ENVIRONMENT VARIABLE OVERRIDE:
#    LR=1e-3 EPOCHS=20 D_MODEL=512 \
#      sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh
#
# 9. CUSTOM DATA DIRECTORY:
#    DATA_DIR=/path/to/data_prompts \
#      sbatch pretrain_rgb_to_structure/slurms/slurm_training_lite.sh --epochs 15
#
# ============================================================================
