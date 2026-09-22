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
# src/scaling/configs.py rejects any config whose estimated wall exceeds 85%
# of the 3h that --qos=short enforces (--time above that is ignored). The
# default ladder's longest config is ~2.2h. A config that cannot finish is
# worse than one not attempted, because it silently removes a point from its
# IsoFLOP parabola and biases the fitted minimum.
#SBATCH --time=03:00:00
#SBATCH --qos=short
# 4 budgets x 5 sizes. Confirm with MODE=dry-run on slurms/scaling_fit.sh,
# or scripts/scaling_sweep.py --n-configs, and update this to 0-(N-1) if you
# change BUDGETS or BRACKET.
#SBATCH --array=0-19
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
#   BUDGETS     default: "1e14 3.7e14 1.4e15 5e15". The top rung is set by
#               the 3h QoS, not by the epoch ceiling: the low-N corner of a
#               rung needs the most data, and beyond ~5.7e15 it no longer
#               fits. Raising it needs a longer QoS.
#   BRACKET     default: "0.6 0.8 1.0 1.3 1.7" (multiples of the prior N*)
#   BATCH_SIZE  default: 256
#   SEED        default: 42
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
BUDGETS="${BUDGETS:-1e14 3.7e14 1.4e15 5e15}"
BRACKET="${BRACKET:-0.6 0.8 1.0 1.3 1.7}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
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
echo "BRACKET:   ${BRACKET}"
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

N_CONFIGS=$(python scripts/scaling_sweep.py --n-configs \
    --budgets ${BUDGETS} --bracket ${BRACKET} --batch-size "${BATCH_SIZE}")
echo "Grid has ${N_CONFIGS} configs; this is config ${CFG_ID}."
if (( CFG_ID >= N_CONFIGS )); then
  echo "ERROR: config ${CFG_ID} does not exist (valid 0-$((N_CONFIGS-1)))." >&2
  echo "       Set --array=0-$((N_CONFIGS-1))." >&2
  exit 1
fi
echo

CFG_ARGS=$(python scripts/scaling_sweep.py --emit-config "${CFG_ID}" \
    --data-dir "${DATA_DIR}" --out-root "${OUT_ROOT}" \
    --budgets ${BUDGETS} --bracket ${BRACKET} --batch-size "${BATCH_SIZE}" \
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
