#!/usr/bin/env bash
#SBATCH --job-name=indigo-lr-grid
#SBATCH --output=job-outputs/indigo-lr-grid.%A_%a.out
#SBATCH --error=job-outputs/indigo-lr-grid.%A_%a.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=06:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Per-configuration learning-rate tuning, Porian et al. style
# ============================================================================
#
# Porian et al. 2024 tune the optimizer EXHAUSTIVELY at every configuration up
# to some scale and fit the law to those optima. INDIGO's deployed law instead
# rests on three measurements, two of which returned the same point of a
# three-point grid and one of which landed on its grid's lower bound. This job
# replaces that with a real grid.
#
# Two axes, because the deployed law has no D term while the sweep's runs span
# 9x in D at fixed N:
#
#   N   model size, taken from the ladder in src/scaling/configs.py so the
#       tuned configs ARE the configs the sweep trains, not nearby ones.
#   D   dataset size, via --limit-examples at EPOCHS=1, so D and the step
#       count move together exactly as they do in the sweep.
#
# Each array task is one (N, D) cell and runs a full --n-lrs grid inside it.
#
# THE GRID MUST BRACKET. scripts/fit_lr_law.py discards any cell whose
# interpolated optimum falls outside the second and second-to-last grid points,
# because such a cell did not measure an optimum, it hit a wall. Defaults below
# are deliberately wide (1e-5 to 5e-3, 6 points, which is the span the
# lr_tuning.sh header has always suggested). If a cell still comes back
# unbracketed, widen it for that cell and re-run rather than using the value.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   # See the cell table and the task count without submitting anything.
#   bash slurms/lr_grid.sh --list
#
#   # Minimal run: the D axis only, at one mid-ladder model size.
#   # 3 cells, measures the D exponent the deployed law omits.
#   D_LADDER="614400 4000000 24000000" N_LADDER="128:2" \
#       sbatch --array=0-2 slurms/lr_grid.sh
#
#   # Full 3x3: both exponents jointly.
#   sbatch --array=0-8 slurms/lr_grid.sh
#
#   # Then fit. The window should stop below the largest size so the
#   # extrapolation has something to be checked against.
#   python scripts/fit_lr_law.py \
#       --results-dir outputs/lr_search/cross_attn \
#       --min-params 2e5 --max-params 3e6 --target-params 17.8e6
#
# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
# A cell costs N_LRS single-epoch runs over D examples. At the ladder's
# calibrated 1301 ex/s that is roughly:
#
#   D = 0.6M    6 LRs     ~0.8 GPU-h
#   D = 4M      6 LRs     ~5.1 GPU-h
#   D = 24M     6 LRs    ~30.7 GPU-h
#
# So the D-only ladder is ~37 GPU-h and the full 3x3 is ~110 GPU-h. Submit the
# D-only ladder first: if the D exponent is inside noise of zero, the deployed
# one-dimensional law is vindicated and the other six cells are unnecessary.
# ============================================================================

# Model sizes as "d_model:slot_encoder_layers", ordered small to large.
N_LADDER="${N_LADDER:-128:2 256:2 512:4}"
# Dataset sizes in examples, at EPOCHS=1.
D_LADDER="${D_LADDER:-614400 4000000 24000000}"

LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-5e-3}"
N_LRS="${N_LRS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SELECTION_METRIC="${SELECTION_METRIC:-delta_e}"
HEAD_MODE="${HEAD_MODE:-cross_attn}"
DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/${HEAD_MODE}}"

read -r -a N_CELLS <<< "${N_LADDER}"
read -r -a D_CELLS <<< "${D_LADDER}"
N_TASKS=$(( ${#N_CELLS[@]} * ${#D_CELLS[@]} ))

if [[ "${1:-}" == "--list" ]]; then
    echo "cells: ${N_TASKS}   (submit with --array=0-$((N_TASKS - 1)))"
    printf '%5s  %-12s  %12s\n' idx model D
    i=0
    for n in "${N_CELLS[@]}"; do
        for d in "${D_CELLS[@]}"; do
            printf '%5d  %-12s  %12s\n' "${i}" "d${n%%:*}/se${n##*:}" "${d}"
            i=$((i + 1))
        done
    done
    exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:?submit as an array job; run with --list to see the cells}"
if (( TASK >= N_TASKS )); then
    echo "task ${TASK} is past the ${N_TASKS} cells in this grid" >&2
    exit 1
fi

N_SPEC="${N_CELLS[$(( TASK / ${#D_CELLS[@]} ))]}"
LIMIT_EXAMPLES="${D_CELLS[$(( TASK % ${#D_CELLS[@]} ))]}"
D_MODEL="${N_SPEC%%:*}"
SLOT_ENCODER_LAYERS="${N_SPEC##*:}"
# Head dim 32, matching the sweep ladder (src/scaling/configs.py:HEAD_DIM).
N_HEADS=$(( D_MODEL / 32 ))

echo "=================================================================="
echo " cell ${TASK} of ${N_TASKS}"
echo "   d_model             ${D_MODEL}"
echo "   slot_encoder_layers ${SLOT_ENCODER_LAYERS}"
echo "   n_heads             ${N_HEADS}"
echo "   D (examples)        ${LIMIT_EXAMPLES}"
echo "   LR grid             ${N_LRS} points, ${LR_MIN} to ${LR_MAX}"
echo "   selection metric    ${SELECTION_METRIC}"
echo "=================================================================="

module purge 2>/dev/null || true
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV:-indigo}" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs "${OUTPUT_DIR}"

srun python scripts/lr_tuning.py \
    --data-dir "${DATA_DIR}" \
    --epochs 1 \
    --limit-examples "${LIMIT_EXAMPLES}" \
    --limit-val-examples "${LIMIT_VAL_EXAMPLES:-10000}" \
    --limit-de-examples "${LIMIT_DE_EXAMPLES:-2048}" \
    --selection-metric "${SELECTION_METRIC}" \
    --lr-min "${LR_MIN}" --lr-max "${LR_MAX}" --n-lrs "${N_LRS}" \
    --feature-mode raw_spectrum \
    --encoder-hidden 128 --encoder-out 64 --encoder-dropout 0.1 \
    --d-model "${D_MODEL}" --n-layers "${N_LAYERS:-8}" --dropout 0.1 \
    --head-mode "${HEAD_MODE}" --n-heads "${N_HEADS}" \
    --slot-encoder-layers "${SLOT_ENCODER_LAYERS}" --decoder-layers 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS:-6}" --prefetch-factor 1 \
    --weight-decay 0.01 --grad-clip 1.0 --warmup-fraction 0.02 \
    --output-dir "${OUTPUT_DIR}" \
    --log-every 100 --streaming --bf16 --plot

echo "[INFO] cell ${TASK} done. Fit once every cell has landed:"
echo "  python scripts/fit_lr_law.py --results-dir ${OUTPUT_DIR} \\"
echo "      --min-params 2e5 --max-params 3e6 --target-params 17.8e6"
