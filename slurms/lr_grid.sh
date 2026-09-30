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
# Per-configuration optimizer tuning, Porian et al. style
# ============================================================================
#
# They tune EXHAUSTIVELY at every configuration up to some scale and fit the
# laws to those optima: 566 sweep runs at 7 model sizes, 6 to 7 learning rates
# x 5 to 7 batch sizes x 3 AdamW beta2 values each. INDIGO's deployed law rests
# on three measurements, none of which bracketed its own optimum, over one
# batch size and one beta2. This replaces that.
#
# Four axes, and why each is here:
#
#   N      model size, taken from the ladder in src/scaling/configs.py so the
#          tuned configs ARE the ones the sweep trains.
#   D      dataset size via --limit-examples at EPOCHS=1. The deployed law has
#          no D term while the sweep spans 9x in D at fixed N.
#   bs     batch size. Never tuned on INDIGO. Their optimum moves with scale,
#          so a fixed 256 is a scale-dependent handicap of unknown sign.
#   beta2  AdamW. Never tuned on INDIGO, which has always run torch's 0.999,
#          the top of their {0.95, 0.99, 0.999}. They report tuning it is
#          essential at LOW batch size, which is the regime INDIGO trains in.
#
# D is held fixed in EXAMPLES while batch size varies, so steps adjust and the
# amount of data each cell sees is identical. This matches their design: within
# a model size their token budget is constant to within 1% across every
# (lr, bs, beta2) cell. Varying batch size at fixed STEPS instead would change
# D and confound the two.
#
# ---------------------------------------------------------------------------
# Staged, because the full grid is not worth buying up front
# ---------------------------------------------------------------------------
#
#   STAGE=1  beta2 and batch size at one model size.    15 cells,  ~12 GPU-h
#            Answers: does beta2 move at all, and where is bs*?
#            If beta2 is flat, stages 2 and 3 drop to one beta2 and get 3x
#            cheaper. If it is not, that alone is a finding.
#
#   STAGE=2  the model-size ladder at the winning beta2. 25 cells,  ~20 GPU-h
#            Gives lr(N) and bs(N) over five sizes, fitted on a window with
#            the largest held out as an extrapolation check.
#
#   STAGE=3  the dataset-size axis at one model size.     3 cells,  ~37 GPU-h
#            Gives the D exponent the deployed law omits. Dominated by the
#            24M cell; run it last and only if stage 2 looks sane.
#
# Total ~69 GPU-h with a decision point after each stage. Compare with the 566
# runs behind their laws.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   STAGE=1 bash slurms/lr_grid.sh --list       # cell table, submits nothing
#   STAGE=1 sbatch --array=0-14%6 slurms/lr_grid.sh
#
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn
#
#   # then, with BETA2 set to what stage 1 picked:
#   STAGE=2 BETA2_LADDER=0.99 sbatch --array=0-24%6 slurms/lr_grid.sh
#   STAGE=3 BETA2_LADDER=0.99 sbatch --array=0-2 --time=12:00:00 --qos=long \
#       slurms/lr_grid.sh
#
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
#       --min-params 5e5 --max-params 7e6 --target-params 17.8e6
#
# Every cell whose optimum lands on a grid endpoint is DISCARDED by the fit,
# not averaged in. If several come back unbracketed, widen that axis for those
# cells and re-run them rather than using the endpoint value.
# ============================================================================

STAGE="${STAGE:-1}"
case "${STAGE}" in
  1) N_LADDER_D="${N_LADDER:-160:3}"
     D_LADDER_D="${D_LADDER:-614400}"
     BS_LADDER_D="${BS_LADDER:-32 64 128 256 512}"
     B2_LADDER_D="${BETA2_LADDER:-0.95 0.99 0.999}" ;;
  2) N_LADDER_D="${N_LADDER:-128:2 160:3 256:2 384:4 512:4}"
     D_LADDER_D="${D_LADDER:-614400}"
     BS_LADDER_D="${BS_LADDER:-32 64 128 256 512}"
     B2_LADDER_D="${BETA2_LADDER:-0.99}" ;;
  3) N_LADDER_D="${N_LADDER:-160:3}"
     D_LADDER_D="${D_LADDER:-614400 4000000 24000000}"
     BS_LADDER_D="${BS_LADDER:-128}"
     B2_LADDER_D="${BETA2_LADDER:-0.99}" ;;
  custom) N_LADDER_D="${N_LADDER:?set N_LADDER for STAGE=custom}"
     D_LADDER_D="${D_LADDER:?set D_LADDER}"
     BS_LADDER_D="${BS_LADDER:?set BS_LADDER}"
     B2_LADDER_D="${BETA2_LADDER:?set BETA2_LADDER}" ;;
  *) echo "STAGE must be 1, 2, 3 or custom (got '${STAGE}')" >&2; exit 1 ;;
esac

LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-5e-3}"
N_LRS="${N_LRS:-6}"
SELECTION_METRIC="${SELECTION_METRIC:-delta_e}"
HEAD_MODE="${HEAD_MODE:-cross_attn}"
DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/${HEAD_MODE}}"
EXAMPLES_PER_SEC="${EXAMPLES_PER_SEC:-1301}"

read -r -a N_CELLS  <<< "${N_LADDER_D}"
read -r -a D_CELLS  <<< "${D_LADDER_D}"
read -r -a BS_CELLS <<< "${BS_LADDER_D}"
read -r -a B2_CELLS <<< "${B2_LADDER_D}"
N_D=${#D_CELLS[@]}; N_BS=${#BS_CELLS[@]}; N_B2=${#B2_CELLS[@]}
N_TASKS=$(( ${#N_CELLS[@]} * N_D * N_BS * N_B2 ))

cell_of() {   # $1 = task index -> sets N_SPEC, LIMIT_EXAMPLES, BATCH_SIZE, BETA2
    local t="$1"
    BETA2="${B2_CELLS[$(( t % N_B2 ))]}";              t=$(( t / N_B2 ))
    BATCH_SIZE="${BS_CELLS[$(( t % N_BS ))]}";         t=$(( t / N_BS ))
    LIMIT_EXAMPLES="${D_CELLS[$(( t % N_D ))]}";       t=$(( t / N_D ))
    N_SPEC="${N_CELLS[${t}]}"
}

if [[ "${1:-}" == "--list" ]]; then
    echo "STAGE ${STAGE}: ${N_TASKS} cells   (submit with --array=0-$((N_TASKS - 1)))"
    printf '%5s  %-12s  %12s  %6s  %7s  %9s\n' idx model D bs beta2 GPU-h
    total=0
    for ((i = 0; i < N_TASKS; i++)); do
        cell_of "${i}"
        h=$(awk -v d="${LIMIT_EXAMPLES}" -v n="${N_LRS}" -v r="${EXAMPLES_PER_SEC}" \
                'BEGIN{printf "%.2f", n*d/r/3600}')
        total=$(awk -v a="${total}" -v b="${h}" 'BEGIN{print a+b}')
        printf '%5d  %-12s  %12s  %6s  %7s  %9s\n' "${i}" \
            "d${N_SPEC%%:*}/se${N_SPEC##*:}" "${LIMIT_EXAMPLES}" \
            "${BATCH_SIZE}" "${BETA2}" "${h}"
    done
    printf '\n%s\n' "estimated total: ${total} GPU-h at ${EXAMPLES_PER_SEC} ex/s"
    exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:?submit as an array job; run with --list to see the cells}"
if (( TASK >= N_TASKS )); then
    echo "task ${TASK} is past the ${N_TASKS} cells in this grid" >&2
    exit 1
fi
cell_of "${TASK}"
D_MODEL="${N_SPEC%%:*}"
SLOT_ENCODER_LAYERS="${N_SPEC##*:}"
# Head dim 32, matching the sweep ladder (src/scaling/configs.py:HEAD_DIM).
N_HEADS=$(( D_MODEL / 32 ))

echo "=================================================================="
echo " STAGE ${STAGE}, cell ${TASK} of ${N_TASKS}"
echo "   d_model / se        ${D_MODEL} / ${SLOT_ENCODER_LAYERS}  (n_heads ${N_HEADS})"
echo "   D (examples)        ${LIMIT_EXAMPLES}"
echo "   batch size          ${BATCH_SIZE}"
echo "   AdamW beta2         ${BETA2}"
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
    --beta1 "${BETA1:-0.9}" --beta2 "${BETA2}" \
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

echo "[INFO] STAGE ${STAGE} cell ${TASK} done. Fit once the stage has landed:"
echo "  python scripts/fit_lr_law.py --results-dir ${OUTPUT_DIR}"
