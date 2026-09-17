#!/usr/bin/env bash
#SBATCH --job-name=indigo-epoch-ceiling
#SBATCH --output=job-outputs/indigo-epoch-ceiling.%A_%a.out
#SBATCH --error=job-outputs/indigo-epoch-ceiling.%A_%a.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One arm needs ~31 min at the conservative 460 ex/s measured on the
# production run (22 min training + ~6 min of DeltaE evals + ~3 min of CE
# val, checkpointing and startup). 01:30:00 is ~2.9x that, comfortably past
# the 1.5x safety budget: over-requesting wall time on SLURM costs nothing
# but scheduling priority, whereas a TIME_LIMIT kill throws away the whole
# arm. Still well inside the 3h that --qos=short actually enforces
# (--time is NOT honoured above that; see CLAUDE.md).
#SBATCH --time=01:30:00
#SBATCH --qos=short
#SBATCH --array=0-11
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Epoch-ceiling probe (scripts/epoch_ceiling_probe.py)
# ============================================================================
#
# Measures how many passes over the corpus INDIGO tolerates before repeats
# start hurting, at MATCHED compute. That number sets the top rung of the
# compute-optimal scaling ladder, so it is the one measurement the whole
# sweep design is waiting on.
#
# Each array task runs ONE arm. Every arm sees identical total steps,
# identical example-passes and identical C; only the amount of DISTINCT data
# those passes draw from changes. The grid lives in
# scripts/epoch_ceiling_probe.py -- this wrapper just asks for arm
# $SLURM_ARRAY_TASK_ID and runs it, so there is one definition of the grid.
#
# Default grid: 3 model sizes (d_model 128/256/512, ~0.77M/2.9M/17.5M params)
# x 4 repeat depths (1/2/4/8 epochs) = 12 arms.
#
# IMPORTANT: --array must match the arm count. Check it with
#     python scripts/epoch_ceiling_probe.py --n-arms
# and if you change SIZES/DEPTHS below, update --array above to 0-(N-1).
#
# Env knobs (all optional except DATA_DIR):
#   DATA_DIR     REQUIRED. Parquet shards dir (the pretrain corpus).
#   OUT_ROOT     default: data/checkpoints/epoch_ceiling_probe
#   TOTAL_STEPS  default: 2400. Gradient steps per arm, identical across
#                arms. MUST be divisible by the LCM of DEPTHS or the script
#                refuses to run (unequal step counts would break the
#                fixed-C comparison). 2400 puts each arm at 614k
#                example-passes, roughly where the production run first had
#                valid generations -- go lower and arms risk producing no
#                scorable DeltaE at all.
#   BATCH_SIZE   default: 256. Held fixed across arms so the repeat effect is
#                not confounded. Note production used 512; halving it keeps
#                the probe affordable and does not affect the comparison.
#   LR           default: 6e-5 (production). Held FIXED across arms on
#                purpose -- this probe isolates the repeat effect, so LR must
#                not co-vary. Per-scale LR tuning is a separate step
#                (slurms/lr_tuning.sh).
#   DEPTHS       default: "1 2 4 8"
#   SIZES        default: "128:2 256:2 512:4"  (d_model:slot_encoder_layers)
#   LIMIT_DE_EXAMPLES  default: 512
#   LIMIT_VAL_EXAMPLES default: 2000
#
# Usage:
#   # 0. Sanity-check the grid, the fixed-C assertion and the time budget
#   python scripts/epoch_ceiling_probe.py --dry-run
#
#   # 1. Launch all 12 arms
#   DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train \
#       sbatch slurms/epoch_ceiling_probe.sh
#
#   # 2. Once they finish, read the ceiling off
#   python scripts/epoch_ceiling_probe.py --analyze \
#       --out-root data/checkpoints/epoch_ceiling_probe
#
# Re-running a single failed arm:
#   DATA_DIR=... sbatch --array=7 slurms/epoch_ceiling_probe.sh
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

# The optical simulator (jaxlayerlumos) is an unrolled small-matrix
# transfer-matrix calculation and is not GPU-amenable; pinning JAX to CPU
# keeps the GPU free for torch. Matches the convention in the other slurms.
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# -------------------- Configuration --------------------
DATA_DIR="${DATA_DIR:-}"
OUT_ROOT="${OUT_ROOT:-data/checkpoints/epoch_ceiling_probe}"
TOTAL_STEPS="${TOTAL_STEPS:-2400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-6e-5}"
DEPTHS="${DEPTHS:-1 2 4 8}"
SIZES="${SIZES:-128:2 256:2 512:4}"
LIMIT_DE_EXAMPLES="${LIMIT_DE_EXAMPLES:-512}"
LIMIT_VAL_EXAMPLES="${LIMIT_VAL_EXAMPLES:-2000}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR is required. Point it at the pretrain parquet shards," >&2
  echo "       e.g. DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train" >&2
  exit 1
fi

ARM_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "============================================================================"
echo "INDIGO EPOCH-CEILING PROBE - arm ${ARM_ID} (job ${SLURM_ARRAY_JOB_ID:-local}_${ARM_ID})"
echo "============================================================================"
echo "PWD:         $(pwd)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo "Started:     $(date)"
echo "DATA_DIR:    ${DATA_DIR}"
echo "OUT_ROOT:    ${OUT_ROOT}"
echo "TOTAL_STEPS: ${TOTAL_STEPS}  BATCH_SIZE: ${BATCH_SIZE}  LR: ${LR}"
echo "SIZES:       ${SIZES}"
echo "DEPTHS:      ${DEPTHS}"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# A missing optical simulator would silently cost us val_de -- the probe's
# only real output -- so fail loudly here rather than 30 minutes in.
python -c "
from src.delta_e_eval import OPTICAL_SIM_AVAILABLE
import sys
print(f'Optical sim available: {OPTICAL_SIM_AVAILABLE}')
if not OPTICAL_SIM_AVAILABLE:
    print('ERROR: the optical simulator is unavailable, so this probe would '
          'produce no val_de and measure nothing. Check the jaxlayerlumos '
          'install before resubmitting.', file=sys.stderr)
    sys.exit(1)
"
echo

# Guard the array bound: a mismatch between --array and the real arm count
# silently drops arms (or runs nonexistent ones), and a probe with a missing
# e=1 baseline cannot produce a ceiling at all.
N_ARMS=$(python scripts/epoch_ceiling_probe.py --n-arms \
    --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
    --depths ${DEPTHS} --sizes ${SIZES})
echo "Grid has ${N_ARMS} arms; this is arm ${ARM_ID}."
if (( ARM_ID >= N_ARMS )); then
  echo "ERROR: arm ${ARM_ID} does not exist (grid has ${N_ARMS}: valid 0-$((N_ARMS-1)))." >&2
  echo "       Fix --array in this script to 0-$((N_ARMS-1))." >&2
  exit 1
fi
echo

# Ask the probe script for this arm's flags, so the grid is defined once.
ARM_ARGS=$(python scripts/epoch_ceiling_probe.py \
    --emit-arm "${ARM_ID}" \
    --data-dir "${DATA_DIR}" \
    --out-root "${OUT_ROOT}" \
    --total-steps "${TOTAL_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --depths ${DEPTHS} \
    --sizes ${SIZES} \
    --limit-de-examples "${LIMIT_DE_EXAMPLES}" \
    --limit-val-examples "${LIMIT_VAL_EXAMPLES}")

echo "Arm command:"
echo "  python scripts/training.py ${ARM_ARGS}"
echo
echo "----------------------------------------------------------------------------"

START_TS=$(date +%s)
set +e
python -u scripts/training.py ${ARM_ARGS}
EXIT_CODE=$?
set -e
END_TS=$(date +%s)

echo "----------------------------------------------------------------------------"
echo "Finished: $(date)"
echo "Elapsed:  $(( (END_TS - START_TS) / 60 )) min $(( (END_TS - START_TS) % 60 )) s"
echo "Exit code: ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo
  echo "When every arm is done, read the ceiling with:"
  echo "  python scripts/epoch_ceiling_probe.py --analyze --out-root ${OUT_ROOT}"
fi
exit ${EXIT_CODE}
