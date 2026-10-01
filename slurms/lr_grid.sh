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
# laws to those optima. INDIGO's deployed law rests on three measurements, none
# of which bracketed its own optimum. This replaces that.
#
# THREE axes, not four. Batch size is NOT swept: on INDIGO it is a VRAM
# decision, the largest batch the card holds, because these models are
# input-bound rather than GPU-bound. src/scaling/configs.py:DEFAULT_BATCH_SIZE
# is the single source of truth and this script reads it, so the tuning batch
# size and the sweep batch size cannot drift apart. They must match: the
# optimal learning rate depends on batch size, so a law tuned at one and
# applied at another is measuring the wrong thing.
#
#   N      model size, taken from the sweep's own ladder so the tuned configs
#          ARE the configs it trains.
#   D      dataset size via --limit-examples at EPOCHS=1. The deployed law has
#          no D term while the sweep spans 9x in D at fixed N.
#   beta2  AdamW. Never tuned on INDIGO, which has always run torch's 0.999,
#          the top of the {0.95, 0.99, 0.999} they sweep. They report tuning it
#          is essential at LOW batch size, which is the regime we train in.
#
# D is held fixed in EXAMPLES, so every cell sees the same data regardless of
# anything else. That matches their design: within a model size their token
# budget is constant to within 1% across every cell.
#
# ---------------------------------------------------------------------------
# NOTHING IS EVER OVERWRITTEN
# ---------------------------------------------------------------------------
# Each cell writes one file named for (epochs, D, d_model, slot layers, batch
# size, beta2). lr_tuning.py checks for it BEFORE loading data or touching the
# GPU and refuses to replace it. This script passes --skip-existing, so
# re-submitting an array is free and finished cells are left alone. Pass
# FORCE=1 only when you mean to redo a cell.
#
# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
#
#   STAGE=1  beta2, at a small and a large model size.   6 cells,  ~4.7 GPU-h
#            Answers two things: does beta2 move the result at all, and does
#            its optimum move with scale. If it is flat, later stages pin one
#            value and nothing is lost.
#
#   STAGE=2  the model-size ladder at the winning beta2. 6 cells,  ~4.7 GPU-h
#            The sizes SPAN the sweep's full ladder, 0.08M to 17.8M, so the
#            fitted law interpolates for every rung instead of extrapolating
#            down to the small ones. This is the difference from Porian et
#            al., who fit a window and extrapolate above it; our smallest
#            rungs are the ones most sensitive to learning rate, so they get
#            measured rather than predicted.
#
#   STAGE=3  the dataset-size axis at one mid size.      1 cell,   ~31 GPU-h
#            Gives the D exponent the deployed law omits. ONE run at the
#            largest D, scored at fractions of its length, rather than one run
#            per D. That is how Porian et al. get 90 token multipliers out of
#            each sweep run: they read the logged loss curve at 90 fractions.
#            It only works under a CONSTANT learning rate, where a prefix of a
#            run is a complete shorter run; under cosine the prefix has not
#            decayed and is not comparable. Hence LR_SCHEDULE=constant below,
#            and lr_tuning.py refuses the combination otherwise.
#
#            NOTE this makes stage 3 measure the D dependence under a constant
#            LR while the sweep trains with cosine. Their tuned arm uses
#            constant for both. Decide which the sweep should use before
#            trusting the D exponent; EVAL_FRACTIONS= (empty) falls back to
#            separate runs per D under cosine, at ~37 GPU-h.
#
# About 46 GPU-h in total, against the 566 runs behind their laws.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   STAGE=1 bash slurms/lr_grid.sh --list        # cells and cost, submits nothing
#   STAGE=1 sbatch --array=0-5 slurms/lr_grid.sh
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn
#
#   STAGE=2 BETA2_LADDER=0.99 sbatch --array=0-5 slurms/lr_grid.sh
#   STAGE=3 BETA2_LADDER=0.99 sbatch --array=0-2 --qos=long --time=12:00:00 \
#       slurms/lr_grid.sh
#
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
#       --coverage-from analyses/scaling/results/isoflop_fit.json
#
# Any cell whose optimum lands on a grid endpoint is DISCARDED by the fit, not
# averaged in. Widen LR_MIN/LR_MAX for those cells and re-run them.
# ============================================================================

# Batch size comes from the sweep planner, never from a default here.
BATCH_SIZE="${BATCH_SIZE:-$(python3 -c 'import sys; sys.path.insert(0,".");
from src.scaling.configs import DEFAULT_BATCH_SIZE; print(DEFAULT_BATCH_SIZE)' \
    2>/dev/null || echo 256)}"

STAGE="${STAGE:-1}"
case "${STAGE}" in
  # Smallest and largest rungs of the sweep ladder.
  1) N_LADDER_D="${N_LADDER:-32:1 416:7}"
     D_LADDER_D="${D_LADDER:-614400}"
     B2_LADDER_D="${BETA2_LADDER:-0.95 0.99 0.999}" ;;
  # Spans the sweep's ladder end to end: 0.08M, 0.23M, 0.77M, 2.5M, 6.6M, 17.8M.
  2) N_LADDER_D="${N_LADDER:-32:1 64:2 128:2 192:4 288:5 416:7}"
     D_LADDER_D="${D_LADDER:-614400}"
     B2_LADDER_D="${BETA2_LADDER:-0.99}" ;;
  3) N_LADDER_D="${N_LADDER:-160:3}"
     D_LADDER_D="${D_LADDER:-24000000}"
     B2_LADDER_D="${BETA2_LADDER:-0.99}"
     LR_SCHEDULE="${LR_SCHEDULE:-constant}"
     # 0.6M, 2M, 4M, 10M of the 24M run, plus the end.
     EVAL_FRACTIONS="${EVAL_FRACTIONS-0.0256 0.0833 0.1667 0.4167}" ;;
  custom) N_LADDER_D="${N_LADDER:?set N_LADDER for STAGE=custom}"
     D_LADDER_D="${D_LADDER:?set D_LADDER}"
     B2_LADDER_D="${BETA2_LADDER:?set BETA2_LADDER}" ;;
  *) echo "STAGE must be 1, 2, 3 or custom (got '${STAGE}')" >&2; exit 1 ;;
esac

# The LR grid is CENTRED PER CELL on the current law's prediction, rather than
# being one fixed window for every model size. A fixed window has to be wide
# enough for the smallest model, which wants a much higher LR than the largest,
# and with only a handful of points a wide window resolves nothing. Centring
# keeps the span per cell narrow enough to resolve while still bracketing.
#
# The prior is the existing lr_for(N). It is poorly determined, which is why we
# are re-tuning, but it is right to within a factor of a few, and the span
# below covers a factor of LR_SPAN either side of it. Porian et al. use a fixed
# window of 7.5e-4 to 4.8e-2 across all their sizes; ours moves with N instead,
# and if a cell still lands on an endpoint the fit discards it and says so.
#
# Set LR_MIN and LR_MAX explicitly to override the centring for a re-run.
LR_SPAN="${LR_SPAN:-30}"
N_LRS="${N_LRS:-7}"
SELECTION_METRIC="${SELECTION_METRIC:-delta_e}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
EVAL_FRACTIONS="${EVAL_FRACTIONS-}"
HEAD_MODE="${HEAD_MODE:-cross_attn}"
DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/${HEAD_MODE}}"
EXAMPLES_PER_SEC="${EXAMPLES_PER_SEC:-1301}"

read -r -a N_CELLS  <<< "${N_LADDER_D}"
read -r -a D_CELLS  <<< "${D_LADDER_D}"
read -r -a B2_CELLS <<< "${B2_LADDER_D}"
N_D=${#D_CELLS[@]}; N_B2=${#B2_CELLS[@]}
N_TASKS=$(( ${#N_CELLS[@]} * N_D * N_B2 ))

cell_of() {   # $1 = task index -> sets N_SPEC, LIMIT_EXAMPLES, BETA2
    local t="$1"
    BETA2="${B2_CELLS[$(( t % N_B2 ))]}";        t=$(( t / N_B2 ))
    LIMIT_EXAMPLES="${D_CELLS[$(( t % N_D ))]}"; t=$(( t / N_D ))
    N_SPEC="${N_CELLS[${t}]}"
}

if [[ "${1:-}" == "--list" ]]; then
    echo "STAGE ${STAGE}: ${N_TASKS} cells, batch size ${BATCH_SIZE}"
    echo "  submit with --array=0-$((N_TASKS - 1))"
    printf '%5s  %-12s  %12s  %7s  %12s  %11s  %9s\n' \
        idx model D beta2 N "LR window" GPU-h
    total=0
    for ((i = 0; i < N_TASKS; i++)); do
        cell_of "${i}"
        h=$(awk -v d="${LIMIT_EXAMPLES}" -v n="${N_LRS}" -v r="${EXAMPLES_PER_SEC}" \
                'BEGIN{printf "%.2f", n*d/r/3600}')
        total=$(awk -v a="${total}" -v b="${h}" 'BEGIN{print a+b}')
        dm="${N_SPEC%%:*}"; se="${N_SPEC##*:}"
        nh=$(( dm / 32 )); (( nh < 1 )) && nh=1
        read -r np pr lo hi <<< "$(python3 - "${dm}" "${se}" "${nh}" "${LR_SPAN}" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.model import ModelConfig
from src.scaling.flops import n_params
from src.scaling.configs import lr_for
d_model, se, n_heads, span = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
cfg = ModelConfig(feature_mode="raw_spectrum", encoder_hidden=128, encoder_out=64,
                  encoder_dropout=0.1, d_model=d_model, n_layers=8, dropout=0.1,
                  head_mode="cross_attn", n_heads=n_heads,
                  slot_encoder_layers=se, decoder_layers=1)
n = n_params(cfg); prior = lr_for(n)
print(f"{n} {prior:.2e} {prior / span:.1e} {prior * span:.1e}")
PYEOF
)"
        printf '%5d  %-12s  %12s  %7s  %12s  %11s  %9s\n' "${i}" \
            "d${dm}/se${se}" "${LIMIT_EXAMPLES}" "${BETA2}" "${np}" \
            "${lo}-${hi}" "${h}"
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
(( N_HEADS < 1 )) && N_HEADS=1

# Centre the LR window on the prior for THIS model size.
read -r N_PARAMS LR_PRIOR LR_LO LR_HI <<< "$(python3 - "${D_MODEL}" \
    "${SLOT_ENCODER_LAYERS}" "${N_HEADS}" "${LR_SPAN}" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.model import ModelConfig
from src.scaling.flops import n_params
from src.scaling.configs import lr_for

d_model, se, n_heads, span = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
cfg = ModelConfig(feature_mode="raw_spectrum", encoder_hidden=128, encoder_out=64,
                  encoder_dropout=0.1, d_model=d_model, n_layers=8, dropout=0.1,
                  head_mode="cross_attn", n_heads=n_heads,
                  slot_encoder_layers=se, decoder_layers=1)
n = n_params(cfg)
prior = lr_for(n)
print(f"{n} {prior:.6e} {prior / span:.6e} {prior * span:.6e}")
PYEOF
)"
LR_MIN="${LR_MIN:-${LR_LO}}"
LR_MAX="${LR_MAX:-${LR_HI}}"

echo "=================================================================="
echo " STAGE ${STAGE}, cell ${TASK} of ${N_TASKS}"
echo "   d_model / se        ${D_MODEL} / ${SLOT_ENCODER_LAYERS}  (n_heads ${N_HEADS})"
echo "   N (parameters)      ${N_PARAMS}"
echo "   prior lr            ${LR_PRIOR}  (centre of the grid)"
echo "   D (examples)        ${LIMIT_EXAMPLES}"
echo "   batch size          ${BATCH_SIZE}  (fixed; from configs.DEFAULT_BATCH_SIZE)"
echo "   AdamW beta2         ${BETA2}"
echo "   LR grid             ${N_LRS} points, ${LR_MIN} to ${LR_MAX}"
echo "   selection metric    ${SELECTION_METRIC}"
echo "   LR schedule         ${LR_SCHEDULE}"
[[ -n "${EVAL_FRACTIONS}" ]] && echo "   mid-run evals at    ${EVAL_FRACTIONS} of the run"
echo "=================================================================="

module purge 2>/dev/null || true
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV:-indigo}" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs "${OUTPUT_DIR}"

# --skip-existing unless FORCE=1: a resubmitted array must never clobber a
# finished cell.
OVERWRITE_FLAG="--skip-existing"
[[ "${FORCE:-0}" == "1" ]] && OVERWRITE_FLAG="--force"

srun python scripts/lr_tuning.py \
    --data-dir "${DATA_DIR}" \
    --epochs 1 \
    --limit-examples "${LIMIT_EXAMPLES}" \
    --limit-val-examples "${LIMIT_VAL_EXAMPLES:-10000}" \
    --limit-de-examples "${LIMIT_DE_EXAMPLES:-2048}" \
    --selection-metric "${SELECTION_METRIC}" \
    --lr-min "${LR_MIN}" --lr-max "${LR_MAX}" --n-lrs "${N_LRS}" \
    --beta1 "${BETA1:-0.9}" --beta2 "${BETA2}" \
    --lr-schedule "${LR_SCHEDULE}" \
    ${EVAL_FRACTIONS:+--eval-fractions ${EVAL_FRACTIONS}} \
    --feature-mode raw_spectrum \
    --encoder-hidden 128 --encoder-out 64 --encoder-dropout 0.1 \
    --d-model "${D_MODEL}" --n-layers "${N_LAYERS:-8}" --dropout 0.1 \
    --head-mode "${HEAD_MODE}" --n-heads "${N_HEADS}" \
    --slot-encoder-layers "${SLOT_ENCODER_LAYERS}" --decoder-layers 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS:-6}" --prefetch-factor 1 \
    --weight-decay 0.01 --grad-clip 1.0 --warmup-fraction 0.02 \
    --output-dir "${OUTPUT_DIR}" \
    "${OVERWRITE_FLAG}" \
    --log-every 100 --streaming --bf16 --plot

echo "[INFO] STAGE ${STAGE} cell ${TASK} done. Fit once the stage has landed:"
echo "  python scripts/fit_lr_law.py --results-dir ${OUTPUT_DIR} \\"
echo "      --coverage-from analyses/scaling/results/isoflop_fit.json"
