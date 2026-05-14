#!/usr/bin/env bash
#SBATCH --job-name=indigo-lr-search
#SBATCH --output=job-outputs/indigo-lr-search.%j.out
#SBATCH --error=job-outputs/indigo-lr-search.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# SLURM Learning Rate Finder for INDIGO
# ============================================================================
#
# Multi-N scaling-law workflow
# ----------------------------
# For a single-pass production run on N rows, lr_opt scales as a power law
# in N. Fit it by submitting this job at several N values with epochs=1, then
# extrapolate to your target N with scripts/fit_lr_scaling.py.
#
# Submit the three sweeps in parallel (each gets its own GPU, no scheduling
# contention):
#
#   for N in 500000 1000000 2000000; do
#       LIMIT_EXAMPLES=$N sbatch slurms/lr_tuning.sh
#   done
#
# After all three complete:
#
#   python scripts/fit_lr_scaling.py \
#       --results-dir outputs/lr_search \
#       --target-examples 10000000 --plot
#
# Each run writes outputs/lr_search/lr_search_ep1_lim<N>.json, so the three
# jobs don't clobber each other.
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "INDIGO LR FINDER - Job ${SLURM_JOB_ID:-local}"
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

# Data.
DATA_DIR="${DATA_DIR:-data/train}"
SEED="${SEED:-42}"

# Subsetting (the multi-N knobs).
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"             # empty → use full train pool
LIMIT_VAL_EXAMPLES="${LIMIT_VAL_EXAMPLES:-10000}" # cap val per LR trial for speed

# LR search.
EPOCHS="${EPOCHS:-1}"                            # single-pass for the scaling fit
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-5e-3}"
N_LRS="${N_LRS:-6}"

# Model architecture (must match your production training).
FEATURE_MODE="${FEATURE_MODE:-raw_spectrum}"
ENCODER_HIDDEN="${ENCODER_HIDDEN:-128}"
ENCODER_OUT="${ENCODER_OUT:-64}"
ENCODER_DROPOUT="${ENCODER_DROPOUT:-0.1}"
D_MODEL="${D_MODEL:-1024}"
N_LAYERS="${N_LAYERS:-8}"
DROPOUT="${DROPOUT:-0.1}"

# Training. Batch size MUST match your planned production batch size —
# optimal LR depends on it.
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

# Output.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search}"

# Flags.
VERBOSE=0
PLOT=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)  VERBOSE=1; shift ;;
        --no-plot)  PLOT=0; shift ;;
        *)          echo "[WARN] Unknown argument: $1"; shift ;;
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
    --feature-mode "${FEATURE_MODE}"
    --encoder-hidden "${ENCODER_HIDDEN}"
    --encoder-out "${ENCODER_OUT}"
    --encoder-dropout "${ENCODER_DROPOUT}"
    --d-model "${D_MODEL}"
    --n-layers "${N_LAYERS}"
    --dropout "${DROPOUT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${LIMIT_EXAMPLES}" ]]; then
    ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi
if [[ -n "${LIMIT_VAL_EXAMPLES}" ]]; then
    ARGS+=(--limit-val-examples "${LIMIT_VAL_EXAMPLES}")
fi
if [[ $VERBOSE -eq 1 ]]; then
    ARGS+=(--verbose)
fi
if [[ $PLOT -eq 1 ]]; then
    ARGS+=(--plot)
fi

CMD=(python scripts/lr_tuning.py "${ARGS[@]}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "LR FINDER CONFIGURATION"
echo "============================================================================"
echo
echo "Search:"
echo "  Epochs per trial:   ${EPOCHS}"
echo "  LR range:           ${LR_MIN} to ${LR_MAX}"
echo "  N LRs:              ${N_LRS}"
echo
echo "Data:"
echo "  Data dir:           ${DATA_DIR}"
echo "  Train limit:        ${LIMIT_EXAMPLES:-(full)}"
echo "  Val limit:          ${LIMIT_VAL_EXAMPLES:-(full)}"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  DataLoader workers: ${NUM_WORKERS}"
echo "  Seed:               ${SEED}"
echo
echo "Model:"
echo "  feature_mode:       ${FEATURE_MODE}"
echo "  encoder hidden/out: ${ENCODER_HIDDEN}/${ENCODER_OUT}"
echo "  d_model:            ${D_MODEL}"
echo "  n_layers:           ${N_LAYERS}"
echo "  dropout:            ${DROPOUT}"
echo
echo "Output:"
echo "  Output dir:         ${OUTPUT_DIR}"
echo "  Plot:               $([ $PLOT -eq 1 ] && echo 'yes' || echo 'no')"
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
echo "LR FINDER COMPLETE"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. Full multi-N scaling-law sweep (recommended): three jobs in parallel.
#    for N in 500000 1000000 2000000; do
#        LIMIT_EXAMPLES=$N sbatch slurms/lr_tuning.sh
#    done
#
# 2. Single LR search on the full training set (no subsetting):
#    sbatch slurms/lr_tuning.sh
#
# 3. Single LR search on a specific subset:
#    LIMIT_EXAMPLES=1000000 sbatch slurms/lr_tuning.sh
#
# 4. Narrower LR range once you know the rough scale:
#    LIMIT_EXAMPLES=2000000 LR_MIN=5e-5 LR_MAX=5e-4 N_LRS=8 \
#        sbatch slurms/lr_tuning.sh
#
# 5. Override the production batch size you're targeting:
#    LIMIT_EXAMPLES=1000000 BATCH_SIZE=512 sbatch slurms/lr_tuning.sh
#
# 6. After all sweeps complete, fit and extrapolate:
#    python scripts/fit_lr_scaling.py \
#        --results-dir outputs/lr_search \
#        --target-examples 10000000 --plot
#
# ============================================================================
