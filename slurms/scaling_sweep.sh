#!/usr/bin/env bash
#SBATCH --job-name=indigo-scaling-sweep
#SBATCH --output=job-outputs/indigo-scaling-sweep.%A_%a.out
#SBATCH --error=job-outputs/indigo-scaling-sweep.%A_%a.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# The grid is SIZED against this limit rather than merely hoping to fit it:
# src/scaling/configs.py rejects any config whose estimated wall exceeds 80%
# of the 3h that --qos=short enforces (--time above that is ignored), against
# a throughput refit on sweep v1's own 20 elapsed times. The default ladder's
# longest config is ~2.2h. A config that cannot finish is worse than one not
# attempted, because it silently removes a point from its IsoFLOP parabola and
# biases the fitted minimum.
#
# THIS 3h CAP IS NOW THE STUDY'S BINDING CONSTRAINT, and not because the top
# rung takes too long. At fixed C a smaller model needs MORE passes, so the
# wall clock sets a FLOOR on N, and that floor grows like C while the optimum
# N* grows only like sqrt(C). Past ~4e15 every size that fits 3h already sits
# above N*, the parabola goes one-sided, and the rung cannot locate its own
# minimum. That caps the ladder at 1e14-4e15, i.e. 1.6 decades of lever arm
# for alpha, well short of production's ~1e17.
#
# A longer QoS is worth far more here than any code change:
#    3h -> top usable budget 4e15   (1.6 decades)
#   12h -> top usable budget 2.5e16 (2.4 decades)
#   24h -> top usable budget 1.2e17 (3.1 decades, reaches production scale)
# To use one: set QOS/TIME below and pass MAX_WALL_HOURS to match, which is
# what re-sizes the grid. Check what this account can get with
#   sbatch --test-only --qos=long --time=12:00:00 slurms/scaling_sweep.sh
#SBATCH --time=03:00:00
#SBATCH --qos=short
# 4 budgets x 6 sizes + 1 repeat-seed arm per budget. Confirm with
# MODE=dry-run on slurms/scaling_fit.sh, or scripts/scaling_sweep.py
# --n-configs, and update this to 0-(N-1) if you change BUDGETS/SPAN/POINTS.
#SBATCH --array=0-27
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# IsoFLOP sweep: one array task per (budget, model size)
# ============================================================================
#
# Each task trains one config and writes val_de into its history.jsonl, which
# is what scripts/fit_scaling.py fits. The grid lives in
# src/scaling/configs.py; this wrapper asks it for config
# $SLURM_ARRAY_TASK_ID, so the submitted sweep and the analysed sweep cannot
# disagree about what the grid was.
#
# Env knobs (only DATA_DIR is required):
#   DATA_DIR    REQUIRED. Pretrain parquet shards.
#   OUT_ROOT    default: data/checkpoints/scaling_sweep
#   BUDGETS     default: "1e14 4e14 1.4e15 4e15". See the QoS note above for
#               why the top rung stops at 4e15.
#   SPAN        default: 10  (ratio of largest to smallest N within a budget).
#               v1 used 2.8x and every parabola came back monotone or
#               concave; curvature needs roughly an order of magnitude.
#   POINTS      default: 6   (sizes per budget)
#   REPEAT_SEED default: 43. Re-runs each rung's middle size under a second
#               init. Those four pairs are the only error bar on the fit;
#               v1 had no repeats, so a 1.04-unit spread at its top rung
#               could not be told apart from eval noise.
#   MAX_WALL_HOURS  unset. Set it (with QOS/TIME above) to re-size the grid
#               for a longer QoS. Confirmed available on this account:
#               sbatch --test-only --qos=long --time=12:00:00 was accepted.
#   DATA_LADDER unset. Set to 1 to run the FIXED-N data ladder instead of the
#               IsoFLOP grid: one shape (d128/se3, 0.97M params, the v1
#               standout at val_de 9.56), D swept log-spaced. Answers whether
#               a ~1M model saturates or keeps improving -- which the IsoFLOP
#               grid cannot, because at fixed C a smaller model needs more
#               passes, so the wall clock floors N. Use a SEPARATE OUT_ROOT:
#               these are not IsoFLOP rungs and fit_scaling.py must not see
#               them. At 3h the ladder spans only 9x and never reaches one
#               epoch; at 12h it spans 43x and reaches 4.3 epochs.
#   BATCH_SIZE  default: 256
#   SEED        default: 42  (baseline arms; repeat arms override it)
#
# Usage:
#   # 1. inspect the grid, feasibility and cost
#   MODE=dry-run sbatch slurms/scaling_fit.sh
#
#   # 2. run it
#   DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train \
#       sbatch slurms/scaling_sweep.sh
#
#   # 3. fit the curves
#   MODE=fit sbatch slurms/scaling_fit.sh
#
# STAGED SUBMISSION (preferred when SU budget matters)
# -----------------------------------------------------
# Each rung is an INDEPENDENT parabola and the power law simply gains a point
# per rung, so rungs can be submitted in waves into the SAME OUT_ROOT and
# re-fitted after each. Nothing is wasted if you stop early. The fit needs 3
# rungs minimum (fit_scaling.py refuses below that), which wave 1 supplies.
#
# Every wave MUST pass the same BUDGETS/SPAN/POINTS/REPEAT_SEED, because the
# array index is a position in that grid. Change any of them and the indices
# shift, so a later wave would train a different config into an earlier one's
# save dir.
#
#   export SW='BUDGETS=1e14 3.02e14 9.1e14 2.75e15 8.29e15 2.5e16'
#   COMMON="DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train \
#           MAX_WALL_HOURS=12 OUT_ROOT=data/checkpoints/scaling_sweep_12h"
#
#   # wave 1 -- 21 cfgs, 22 GPU-h, 1.0 decade, NO 12h exposure at all
#   env $COMMON BUDGETS="..." sbatch --array=0-20%8 slurms/scaling_sweep.sh
#   # wave 2 -- +7 cfgs, +15 GPU-h -> 1.4 decades
#   env $COMMON BUDGETS="..." sbatch --qos=long --time=06:00:00 \
#       --array=21-27%6 slurms/scaling_sweep.sh
#   # wave 3 -- +7 cfgs, +24 GPU-h -> 1.9 decades
#   env $COMMON BUDGETS="..." sbatch --qos=long --time=09:00:00 \
#       --array=28-34%6 slurms/scaling_sweep.sh
#   # wave 4 -- +7 cfgs, +29 GPU-h -> 2.4 decades
#   env $COMMON BUDGETS="..." sbatch --qos=long --time=12:00:00 \
#       --array=35-41%6 slurms/scaling_sweep.sh
#
# The %N suffix throttles simultaneous tasks. Use it. Sweep v1 ran all 20 at
# once against the same parquet shards on shared /ix1 and measured 246-2856
# ex/s, a 12x spread with a median of 2155 -- these models are input-bound,
# not GPU-bound, so concurrent jobs were competing for the same reads. The
# wall model is calibrated on that contended throughput, so throttling should
# make configs finish EARLY rather than late, which costs fewer SUs.
#
#   # all six rungs at once (90 GPU-h) if SU budget is not a concern:
#   DATA_DIR=... MAX_WALL_HOURS=12 \
#       BUDGETS="1e14 3.02e14 9.1e14 2.75e15 8.29e15 2.5e16" \
#       OUT_ROOT=data/checkpoints/scaling_sweep_12h \
#       sbatch --qos=long --time=12:00:00 --array=0-41%8 slurms/scaling_sweep.sh
#
#   # fixed-N data ladder (6 runs, separate out-root, NOT for fit_scaling):
#   DATA_DIR=... DATA_LADDER=1 MAX_WALL_HOURS=12 \
#       OUT_ROOT=data/checkpoints/data_ladder_n1M \
#       sbatch --qos=long --time=12:00:00 --array=0-5 slurms/scaling_sweep.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

DATA_DIR="${DATA_DIR:-}"
OUT_ROOT="${OUT_ROOT:-data/checkpoints/scaling_sweep}"
BUDGETS="${BUDGETS:-1e14 4e14 1.4e15 4e15}"
SPAN="${SPAN:-10}"
POINTS="${POINTS:-6}"
REPEAT_SEED="${REPEAT_SEED:-43}"
MAX_WALL_HOURS="${MAX_WALL_HOURS:-}"
DATA_LADDER="${DATA_LADDER:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"

GRID_ARGS="--budgets ${BUDGETS} --span ${SPAN} --points ${POINTS} --batch-size ${BATCH_SIZE}"
[[ -n "${REPEAT_SEED}" ]] && GRID_ARGS="${GRID_ARGS} --repeat-seed ${REPEAT_SEED}"
[[ -n "${MAX_WALL_HOURS}" ]] && GRID_ARGS="${GRID_ARGS} --max-wall-hours ${MAX_WALL_HOURS}"
[[ -n "${DATA_LADDER}" ]] && GRID_ARGS="${GRID_ARGS} --data-ladder"
LIMIT_DE_EXAMPLES="${LIMIT_DE_EXAMPLES:-2048}"
LIMIT_VAL_EXAMPLES="${LIMIT_VAL_EXAMPLES:-2000}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR is required, e.g." >&2
  echo "  DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train" >&2
  exit 1
fi

CFG_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "============================================================================"
echo "INDIGO ISOFLOP SWEEP - config ${CFG_ID} (job ${SLURM_ARRAY_JOB_ID:-local}_${CFG_ID})"
echo "============================================================================"
echo "Started:   $(date)"
echo "DATA_DIR:  ${DATA_DIR}"
echo "OUT_ROOT:  ${OUT_ROOT}"
echo "BUDGETS:   ${BUDGETS}"
echo "GRID_ARGS: ${GRID_ARGS}"
echo "SEED:      ${SEED}"
echo

python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Without the simulator there is no val_de, and val_de is the ONLY thing the
# IsoFLOP fit consumes -- fail in seconds rather than after hours of training.
python -c "
from src.delta_e_eval import OPTICAL_SIM_AVAILABLE
import sys
print(f'Optical sim available: {OPTICAL_SIM_AVAILABLE}')
if not OPTICAL_SIM_AVAILABLE:
    print('ERROR: no optical simulator, so this config would train for hours '
          'and produce no val_de. Fix the jaxlayerlumos install first.',
          file=sys.stderr)
    sys.exit(1)
"
echo

N_CONFIGS=$(python scripts/scaling_sweep.py --n-configs ${GRID_ARGS})
echo "Grid has ${N_CONFIGS} configs; this is config ${CFG_ID}."
if (( CFG_ID >= N_CONFIGS )); then
  echo "ERROR: config ${CFG_ID} does not exist (valid 0-$((N_CONFIGS-1)))." >&2
  echo "       Set --array=0-$((N_CONFIGS-1))." >&2
  exit 1
fi
echo

CFG_ARGS=$(python scripts/scaling_sweep.py --emit-config "${CFG_ID}" \
    --data-dir "${DATA_DIR}" --out-root "${OUT_ROOT}" ${GRID_ARGS} \
    --seed "${SEED}" --limit-de-examples "${LIMIT_DE_EXAMPLES}" \
    --limit-val-examples "${LIMIT_VAL_EXAMPLES}" --num-workers "${NUM_WORKERS}")

# Resume only from an epoch boundary. training.py replays a resumed epoch's
# loader from the start, so a mid-epoch resume would change WHICH examples the
# config sees -- and for an IsoFLOP point the (N, D) pair is the measurement.
CFG_SAVE_DIR=$(echo "${CFG_ARGS}" | tr ' ' '\n' \
    | grep -A1 -- '--save-dir' | tail -1 || true)
RESUME_ARGS=""
if [[ "${RESUME:-1}" == "1" && -f "${CFG_SAVE_DIR}/latest/model.pt" ]]; then
  RESUME_ARGS="--resume ${CFG_SAVE_DIR}/latest"
  echo "Resuming from ${CFG_SAVE_DIR}/latest"
fi

echo "Config command:"
echo "  python scripts/training.py ${CFG_ARGS} ${RESUME_ARGS}"
echo
echo "----------------------------------------------------------------------------"

START_TS=$(date +%s)
set +e
python -u scripts/training.py ${CFG_ARGS} ${RESUME_ARGS}
EXIT_CODE=$?
set -e
END_TS=$(date +%s)

echo "----------------------------------------------------------------------------"
echo "Finished:  $(date)"
echo "Elapsed:   $(( (END_TS-START_TS)/60 )) min $(( (END_TS-START_TS)%60 )) s"
echo "Exit code: ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo
  echo "When every config is done, fit the curves with:"
  echo "  MODE=fit OUT_ROOT=${OUT_ROOT} sbatch slurms/scaling_fit.sh"
fi
exit ${EXIT_CODE}
