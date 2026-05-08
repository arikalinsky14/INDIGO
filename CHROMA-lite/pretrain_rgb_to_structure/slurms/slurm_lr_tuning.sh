#!/usr/bin/env bash
#SBATCH --job-name=chroma-lr-search
#SBATCH --output=job-outputs/slurm-lr-search.%j.out
#SBATCH --error=job-outputs/slurm-lr-search.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# SLURM Learning Rate Finder for CHROMA-Lite
# ============================================================================
#
# This script searches for the optimal learning rate for a given number of
# epochs by training with multiple LR values on the FULL dataset.
#
# Submit separate jobs for each epoch count to enable parallelization:
#
#   for ep in 4 6 8 12; do
#     sbatch slurm_lr_tuning.sh --epochs $ep
#   done
#
# Then fit the scaling law:
#   python fit_lr_scaling.py --results-dir outputs/lr_search/ --target-epochs 200 --plot
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
echo "CHROMA-LITE LR FINDER - Job ${SLURM_JOB_ID}"
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

# Data settings
DATA_DIR="${DATA_DIR:-create_dataset/data_prompts}"
SEED="${SEED:-42}"

# LR search parameters
EPOCHS="${EPOCHS:-4}"                         # Epochs for each LR trial
LR_MIN="${LR_MIN:-1e-5}"                      # Minimum LR to try
LR_MAX="${LR_MAX:-1e-2}"                      # Maximum LR to try
N_LRS="${N_LRS:-10}"                          # Number of LR values to try

# Model architecture (should match main training)
D_MODEL="${D_MODEL:-2048}"
N_LAYERS="${N_LAYERS:-16}"
DROPOUT="${DROPOUT:-0.0}"

# Training parameters
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-pretrain_rgb_to_structure/outputs/lr_search}"

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

VERBOSE=0
PLOT=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr-min)
            LR_MIN="$2"
            shift 2
            ;;
        --lr-max)
            LR_MAX="$2"
            shift 2
            ;;
        --n-lrs)
            N_LRS="$2"
            shift 2
            ;;
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --n-layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --no-plot)
            PLOT=0
            shift
            ;;
        *)
            echo "[WARN] Unknown argument: $1"
            shift
            ;;
    esac
done

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
    --data-dir "${DATA_DIR}"
    --epochs "${EPOCHS}"
    --seed "${SEED}"
    --lr-min "${LR_MIN}"
    --lr-max "${LR_MAX}"
    --n-lrs "${N_LRS}"
    --d-model "${D_MODEL}"
    --n-layers "${N_LAYERS}"
    --dropout "${DROPOUT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ $VERBOSE -eq 1 ]]; then
    ARGS+=(--verbose)
fi

if [[ $PLOT -eq 1 ]]; then
    ARGS+=(--plot)
fi

CMD=(python pretrain_rgb_to_structure/scripts/lr_tuning.py "${ARGS[@]}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "LR FINDER CONFIGURATION"
echo "============================================================================"
echo
echo "Search Parameters:"
echo "  Epochs per trial:  ${EPOCHS}"
echo "  LR range:          ${LR_MIN} to ${LR_MAX}"
echo "  Number of LRs:     ${N_LRS}"
echo
echo "Data:"
echo "  Data dir:          ${DATA_DIR}"
echo "  Batch size:        ${BATCH_SIZE}"
echo "  Seed:              ${SEED}"
echo "  (Training on FULL dataset)"
echo
echo "Model:"
echo "  d_model:           ${D_MODEL}"
echo "  n_layers:          ${N_LAYERS}"
echo "  dropout:           ${DROPOUT}"
echo
echo "Output:"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Plot:              $([ $PLOT -eq 1 ] && echo 'yes' || echo 'no')"
echo
echo "============================================================================"
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

# ============================================================================
# RUN LR FINDER
# ============================================================================

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "LR FINDER COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"

exit ${EXIT_CODE}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. Run LR search for a single epoch count:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_lr_tuning.sh --epochs 4
#
# 2. Submit parallel jobs for scaling study:
#    for ep in 4 6 8 12; do
#      sbatch pretrain_rgb_to_structure/slurms/slurm_lr_tuning.sh --epochs $ep
#    done
#
# 3. Custom LR range:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_lr_tuning.sh --epochs 8 --lr-min 5e-5 --lr-max 5e-3
#
# 4. More LR values for finer search:
#    sbatch pretrain_rgb_to_structure/slurms/slurm_lr_tuning.sh --epochs 6 --n-lrs 15
#
# 5. After all searches complete, fit the scaling law:
#    python pretrain_rgb_to_structure/scripts/fit_lr_scaling.py \
#      --results-dir pretrain_rgb_to_structure/outputs/lr_search/ \
#      --target-epochs 200 \
#      --plot
#
# ============================================================================