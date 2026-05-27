#!/usr/bin/env bash
#SBATCH --job-name=indigo-train
#SBATCH --output=job-outputs/indigo-train.%j.out
#SBATCH --error=job-outputs/indigo-train.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# SLURM Training Script for INDIGO (scripts/training.py)
# ============================================================================
#
# This script trains the INDIGO FlexMaterialMLP with:
# - RGB + variable material pool conditioning
# - Shared per-slot material encoder + feedforward backbone
# - Learning rate schedule: linear warmup + cosine decay
# - Checkpoints saved to: data/checkpoints/<hparams_tag>/
#
# ARCHITECTURE:
#   MaterialEncoder (shared per slot): [2, NUM_LAMBDA=128] -> [encoder_out]
#   Backbone input  = RGB (3) + M_MAX*encoder_out + M_MAX*MAX_LAYERS + 1
#                  -> Dense(d_model) -> ReLU -> Dropout
#                  -> [Dense(d_model) -> ReLU -> Dropout] x (n_layers - 1)
#                  -> Dense(vocab_size=1281 = M_MAX*NUM_THICKNESSES + EOS)
#
# USAGE EXAMPLES:
#
# 1. Full production training (default settings):
#    sbatch slurms/training.sh
#
# 2. Quick validation run (limit to 1000 examples):
#    sbatch slurms/training.sh \
#      --limit-examples 1000
#
# 3. Custom hyperparameters:
#    sbatch slurms/training.sh \
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
echo "INDIGO TRAINING RUN - Job ${SLURM_JOB_ID}"
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
DATA_DIR="${DATA_DIR:-}"                   # Path to a parquet-shards directory;
                                           # empty -> training.py defaults to
                                           # <repo>/data/train
SPLIT="${SPLIT:-train}"                    # Dataset split (train/validation)
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"       # Limit to N examples (for testing)
SEED="${SEED:-42}"                         # Random seed

# -------------------- Model Architecture (FlexMaterialMLP) --------------------
FEATURE_MODE="${FEATURE_MODE:-raw_spectrum}"  # 'raw_spectrum' or 'compact'
ENCODER_HIDDEN="${ENCODER_HIDDEN:-128}"       # Material encoder hidden dim
ENCODER_OUT="${ENCODER_OUT:-64}"              # Material encoder output dim
ENCODER_DROPOUT="${ENCODER_DROPOUT:-0.1}"     # Material encoder dropout
D_MODEL="${D_MODEL:-1024}"                    # Backbone hidden dim
N_LAYERS="${N_LAYERS:-8}"                     # Number of backbone hidden layers
DROPOUT="${DROPOUT:-0.1}"                     # Backbone dropout

# -------------------- Optimization --------------------
LR="${LR:-1.44e-3}"                           # Base learning rate
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"          # AdamW weight decay
GRAD_CLIP="${GRAD_CLIP:-1.0}"                 # Gradient clipping norm
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"    # Warmup fraction
EPOCHS="${EPOCHS:-1}"                         # Number of training epochs

# -------------------- Data Loading --------------------
BATCH_SIZE="${BATCH_SIZE:-256}"                # Batch size
NUM_WORKERS="${NUM_WORKERS:-4}"               # DataLoader workers.
                                              # lr_tuning.sh runs at 6 on the
                                              # same allocation, but only ever
                                              # against LIMIT_EXAMPLES subsets.
                                              # training iterates the FULL
                                              # 10M-row dataset, so each worker
                                              # eventually cycles through every
                                              # shard (~140 MB/shard while a
                                              # parquet table is in scope) —
                                              # 6 OOMs at --mem=64G here.
                                              # 4 is the proven-safe value
                                              # against the full dataset; bump
                                              # only if you also bump --mem.

# -------------------- Checkpointing --------------------
SAVE_DIR="${SAVE_DIR:-}"                   # Override checkpoint dir (default: auto-generated)
SAVE_EVERY="${SAVE_EVERY:-1000}"           # Save checkpoint every N steps

# -------------------- Logging --------------------
LOG_EVERY="${LOG_EVERY:-100}"              # Print per-step loss every N steps
                                           # (python defaults to --verbose; pass
                                           # --no-verbose to silence)

# ============================================================================
# COMMAND LINE OVERRIDE HANDLING
# ============================================================================

# All sbatch positional args flow through to training.py.
USER_ARGS=("$@")

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  # Data & training control
  --split "${SPLIT}"
  --seed "${SEED}"

  # Model architecture (FlexMaterialMLP)
  --feature-mode "${FEATURE_MODE}"
  --encoder-hidden "${ENCODER_HIDDEN}"
  --encoder-out "${ENCODER_OUT}"
  --encoder-dropout "${ENCODER_DROPOUT}"
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

  # Logging
  --log-every "${LOG_EVERY}"
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

# User arguments append after ARGS, so they win for any duplicated flag
# (including --no-verbose to silence the default per-step logging).
CMD=(python scripts/training.py "${ARGS[@]}" "${USER_ARGS[@]}")

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
echo "Model Architecture (FlexMaterialMLP):"
echo "  feature mode:    ${FEATURE_MODE}"
echo "  encoder hidden:  ${ENCODER_HIDDEN}"
echo "  encoder out:     ${ENCODER_OUT}"
echo "  d_model:         ${D_MODEL}"
echo "  n_layers:        ${N_LAYERS}"
echo "  dropout:         ${DROPOUT}"
echo "  Output dim:      1281 (M_MAX=32 * NUM_THICKNESSES=40 + EOS)"
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
#    sbatch slurms/training.sh --epochs 15
#
# 2. QUICK TEST (small subset):
#    sbatch slurms/training.sh \
#      --limit-examples 10000 --epochs 5
#
# 3. HYPERPARAMETER SWEEP - Learning Rate:
#    for lr in 1e-3 2e-3 5e-3; do
#      sbatch slurms/training.sh --lr $lr --epochs 10
#    done
#
# 4. HYPERPARAMETER SWEEP - Model Size:
#    sbatch slurms/training.sh --d-model 128 --n-layers 3
#    sbatch slurms/training.sh --d-model 256 --n-layers 4
#    sbatch slurms/training.sh --d-model 512 --n-layers 6
#
# 5. LARGER MODEL WITH MORE EPOCHS:
#    sbatch slurms/training.sh \
#      --d-model 512 --n-layers 6 --epochs 20 --lr 1e-3
#
# 6. HIGHER DROPOUT (regularization):
#    sbatch slurms/training.sh --dropout 0.2 --epochs 15
#
# 7. LOG CADENCE (verbose is on by default, every 100 steps):
#    LOG_EVERY=25 sbatch slurms/training.sh --epochs 10
#    Or silence: sbatch slurms/training.sh --no-verbose --epochs 10
#
# 8. ENVIRONMENT VARIABLE OVERRIDE:
#    LR=1e-3 EPOCHS=20 D_MODEL=512 \
#      sbatch slurms/training.sh
#
# 9. CUSTOM DATA DIRECTORY:
#    DATA_DIR=/path/to/data_prompts \
#      sbatch slurms/training.sh --epochs 15
#
# ============================================================================
