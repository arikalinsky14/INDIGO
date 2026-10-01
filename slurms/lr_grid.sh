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
# Stages, and why stage 2 is not a fixed-D ladder
# ---------------------------------------------------------------------------
#
# Porian et al. tune at a CONSTANT token multiplier: across a 42x range in
# parameters their sweep holds M = tokens/params between 20.0 and 21.1. N and M
# are decoupled by construction, so a law in N alone is the right object for
# them, and their compute-optimal M is itself ~constant, so one M covers the
# whole study.
#
# INDIGO cannot borrow that. Our compute-optimal multiplier is NOT constant:
# D*/N* runs 29.3 down to 0.86 across our budgets, a 34x range, and the 48
# sweep runs occupy M from 0.28 to 52. Tuning at a fixed D would be worse than
# useless, because M = D/N would then vary as 1/N across the ladder, 220x: the
# fitted "lr(N)" would really be lr along a trajectory in M. That is the
# aspect-ratio confound again, in a different variable.
#
#   STAGE=1  beta2 at both ends of the sweep's ladder.  6 cells,  ~6 GPU-h
#            Does beta2 matter, and does its optimum move with scale? Their
#            data says it tracks BATCH SIZE, which we hold fixed, so one value
#            should serve; this checks that on our model.
#
#   STAGE=2  the sweep's own compute-optimal points.    5 cells,  ~32 GPU-h
#            One cell per rung at its (N*, D*), the points that actually
#            determine the IsoFLOP minima. Spans N by 66x and M from 34 to
#            0.93, which is the real trajectory rather than an artefact of
#            holding something fixed. D* barely moves, so this is affordable.
#
#   STAGE=3  the multiplier axis at one fixed N.        1 cell,   ~15 GPU-h
#            Stage 2 is a one-dimensional path through (N, M), so it cannot
#            separate the two exponents on its own. This varies M by 29x at
#            fixed N, inside ONE run, by scoring at fractions of it. Together
#            the two stages support the 2-D fit lr(N, M).
#
# About 53 GPU-h in total, against the 566 runs behind their laws.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   STAGE=1 bash slurms/lr_grid.sh --list        # cells and cost, submits nothing
#   STAGE=1 sbatch --array=0-5 slurms/lr_grid.sh
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn
#
#   # then, with BETA2_WINNER set to what stage 1 picked:
#   STAGE=2 BETA2_WINNER=0.99 sbatch --array=0-4 slurms/lr_grid.sh
#   STAGE=3 BETA2_WINNER=0.99 sbatch --array=0-0 --qos=long --time=20:00:00 \
#       slurms/lr_grid.sh
#
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
#       --coverage-from analyses/scaling/results/isoflop_fit.json
#
# Any cell whose optimum lands on a grid endpoint is DISCARDED by the fit, not
# averaged in. Widen LR_SPAN for those cells and re-run them.
# ============================================================================

# Environment FIRST. Everything below shells out to python, and the cell list
# in particular is computed before any task runs.
#
# Under sbatch this is the submit directory; run by hand it is wherever you
# already are, so --list works from a checkout without guessing a path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
elif [[ ! -f scripts/lr_grid_cells.py && -d "$HOME/Github/INDIGO" ]]; then
    cd "$HOME/Github/INDIGO"
fi
if [[ ! -f scripts/lr_grid_cells.py ]]; then
    echo "run this from the INDIGO checkout (no scripts/lr_grid_cells.py here)" >&2
    exit 1
fi

# -------------------- Environment Setup --------------------
# Same two lines every other slurm in this repo uses. An earlier version of
# this script invented a conda activation that does not exist on this cluster
# and swallowed the failure with `|| true`, so the job ran against the system
# python and died on `import torch` after the scheduler had already given it a
# GPU. Failures here are fatal and loud.
if command -v module >/dev/null 2>&1; then
    module purge
    module load python/pytorch_251_311_cu124
fi
if [[ -f "$HOME/envs/llm-env/bin/activate" ]]; then
    source "$HOME/envs/llm-env/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false

mkdir -p job-outputs

# Batch size comes from the sweep planner, never from a default here.
if [[ -z "${BATCH_SIZE:-}" ]]; then
    BATCH_SIZE="$(python3 -c 'import sys; sys.path.insert(0,".");
from src.scaling.configs import DEFAULT_BATCH_SIZE; print(DEFAULT_BATCH_SIZE)')" \
        || { echo "could not read DEFAULT_BATCH_SIZE. Is the environment "\
                  "active? Set CONDA_ENV, or pass BATCH_SIZE explicitly." >&2
             exit 1; }
fi

STAGE="${STAGE:-1}"
# The probe runs ONE learning rate; every other stage runs the full grid.
[[ "${STAGE}" == "probe" ]] && N_LRS=1
FIT="${FIT:-analyses/scaling/results/porian_fit.json}"
BETA2_WINNER="${BETA2_WINNER:-0.99}"

LR_SPAN="${LR_SPAN:-30}"
N_LRS="${N_LRS:-7}"
SELECTION_METRIC="${SELECTION_METRIC:-delta_e}"
HEAD_MODE="${HEAD_MODE:-cross_attn}"
DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/${HEAD_MODE}}"
# MEASURED on the first stage-1 submission (median of 227 step samples), not
# the 1301 the sweep planner assumes. Sizing against 1301 is what put every
# cell into its wall. Raise it once a probe shows the contention is gone.
EXAMPLES_PER_SEC="${EXAMPLES_PER_SEC:-204}"

# Stage 3 varies the multiplier inside ONE run by scoring at fractions of it,
# which is only valid with a constant LR; lr_tuning.py refuses it otherwise.
if [[ "${STAGE}" == "3" ]]; then
    LR_SCHEDULE="${LR_SCHEDULE:-constant}"
    EVAL_FRACTIONS="${EVAL_FRACTIONS-0.03125 0.1 0.3333}"
else
    LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
    EVAL_FRACTIONS="${EVAL_FRACTIONS-}"
fi

# Cells come from scripts/lr_grid_cells.py, not from a ladder written here,
# because stage 2's cells are DERIVED from the sweep's own compute-optimal
# points. See that script's docstring for why they have to be.
mapfile -t CELL_LINES < <(python3 scripts/lr_grid_cells.py \
    --stage "${STAGE}" --fit "${FIT}" --beta2 "${BETA2_WINNER}")
N_TASKS=${#CELL_LINES[@]}
if (( ! N_TASKS )); then
    echo "no cells for STAGE=${STAGE}." >&2
    echo "  scripts/lr_grid_cells.py produced nothing. Run it directly to see" >&2
    echo "  why; the usual cause is the environment not being active, or" >&2
    echo "  FIT=${FIT} missing." >&2
    exit 1
fi

cell_of() {   # $1 = task index -> sets D_MODEL, SLOT_ENCODER_LAYERS, LIMIT_EXAMPLES, BETA2
    read -r D_MODEL SLOT_ENCODER_LAYERS LIMIT_EXAMPLES BETA2 \
        <<< "${CELL_LINES[$1]}"
}

if [[ "${1:-}" == "--list" ]]; then
    python3 scripts/lr_grid_cells.py --stage "${STAGE}" --fit "${FIT}" \
        --beta2 "${BETA2_WINNER}" --format table --n-lrs "${N_LRS}" \
        --rate "${EXAMPLES_PER_SEC}"
    echo
    echo "batch size ${BATCH_SIZE};  submit with --array=0-$((N_TASKS - 1))"
    exit 0
fi

# Everything past here trains, so torch has to be importable. Fail now rather
# than after the scheduler has handed out a GPU and lr_tuning.py has loaded a
# dataset.
python3 -c 'import torch' 2>/dev/null || {
    echo "[ERROR] torch is not importable after loading the environment." >&2
    echo "        module: python/pytorch_251_311_cu124" >&2
    echo "        venv:   $HOME/envs/llm-env" >&2
    echo "        Failing before this job spends any more of the allocation." >&2
    exit 1
}

TASK="${SLURM_ARRAY_TASK_ID:?submit as an array job; run with --list to see the cells}"
if (( TASK >= N_TASKS )); then
    echo "task ${TASK} is past the ${N_TASKS} cells in this grid" >&2
    exit 1
fi
cell_of "${TASK}"
# Head dim 32, matching the sweep ladder (src/scaling/configs.py:HEAD_DIM).
N_HEADS=$(( D_MODEL / 32 ))
(( N_HEADS < 1 )) && N_HEADS=1

# Centre the LR window on the prior for THIS model size.
read -r N_PARAMS LR_PRIOR LR_LO LR_HI <<< "$(python3 - "${D_MODEL}" \
    "${SLOT_ENCODER_LAYERS}" "${N_HEADS}" "${LR_SPAN}" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.scaling.flops import ArchSpec, n_params
from src.scaling.configs import lr_for

d_model, se, n_heads, span = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
cfg = ArchSpec(d_model=d_model, n_heads=n_heads, head_mode="cross_attn",
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

mkdir -p "${OUTPUT_DIR}"

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
