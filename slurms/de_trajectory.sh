#!/usr/bin/env bash
#SBATCH --job-name=indigo-de-trajectory
#SBATCH --output=job-outputs/indigo-de-trajectory.%j.out
#SBATCH --error=job-outputs/indigo-de-trajectory.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Recover a finished run's DeltaE trajectory from its saved checkpoints
# ============================================================================
#
# NO RETRAINING. The scaling sweep saved every 2000 steps, so the trajectory
# the sweep declined to record (one DeltaE per run, at the final step) can be
# rebuilt after the fact for the cost of a forward pass plus an optical sim
# per checkpoint.
#
# This is the experiment that separates the two readings of the data ladder:
#
#   REPETITION  val_de peaks early and decays while the cosine LR is still
#               high -> passes past ~1 epoch genuinely hurt, the corpus is
#               the binding constraint, and reaching production-scale budgets
#               needs roughly 4x more data.
#   SCHEDULE    val_de tracks the LR decay and degrades in the tail -> the
#               ladder measured cosine death over a long horizon, the epoch
#               ceiling is not 1, and the current 10M corpus already reaches
#               ~2.4 decades.
#
# The sharpest comparison is the SAME pass count in two runs: the ladder's
# 9.49M-pass run finished there with a fully decayed LR, while the 42.5M-pass
# run only passes through 9.49M mid-flight at high LR. Both saw identical
# data, so a large gap there is a schedule effect by construction.
#
# Env knobs:
#   RUN_DIR    REQUIRED. One run directory containing step_* checkpoints.
#   DATA_DIR   REQUIRED.
#   EVERY_NTH  default 10. Cost is linear: a 166k-step run has 84
#              checkpoints, and DeltaE eval is ~146 s per 2048 examples, so
#              every-10th is ~22 min and every-1 is ~3.4 h.
#   SEED       default 42. Must match the run, or the eval slice differs and
#              the trajectory is not comparable to that run's recorded val_de.
#
# Usage -- the two ladder points that matter, deepest first:
#   RUN_DIR=data/checkpoints/data_ladder_n1M/sweep_C1.05e16_d128_se3_s42 \
#       DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train \
#       sbatch slurms/de_trajectory.sh
#   RUN_DIR=data/checkpoints/data_ladder_n1M/sweep_C4.95e15_d128_se3_s42 \
#       DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train \
#       sbatch slurms/de_trajectory.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p job-outputs

RUN_DIR="${RUN_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
EVERY_NTH="${EVERY_NTH:-10}"
SEED="${SEED:-42}"
LIMIT_DE_EXAMPLES="${LIMIT_DE_EXAMPLES:-2048}"

if [[ -z "${RUN_DIR}" || -z "${DATA_DIR}" ]]; then
  echo "ERROR: RUN_DIR and DATA_DIR are both required." >&2
  echo "  RUN_DIR=data/checkpoints/data_ladder_n1M/sweep_... \\" >&2
  echo "  DATA_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/train" >&2
  exit 1
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR '${RUN_DIR}' does not exist." >&2
  echo "       Check the name with: ls data/checkpoints/data_ladder_n1M/" >&2
  exit 1
fi

OUT_NAME="$(basename "${RUN_DIR}")"
OUTPUT="${OUTPUT:-analyses/scaling/results/de_trajectory_${OUT_NAME}.json}"

echo "============================================================================"
echo "INDIGO DELTA-E TRAJECTORY - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "Started:    $(date)"
echo "RUN_DIR:    ${RUN_DIR}"
echo "EVERY_NTH:  ${EVERY_NTH}"
echo "OUTPUT:     ${OUTPUT}"
echo
N_CKPT=$(find "${RUN_DIR}" -maxdepth 1 -name 'step_*' -type d | wc -l)
echo "step_* checkpoints present: ${N_CKPT}  ->  ~$(( N_CKPT / EVERY_NTH + 1 )) evals"
echo
echo "----------------------------------------------------------------------------"

mkdir -p "$(dirname "${OUTPUT}")"
python -u scripts/de_trajectory.py \
    --run-dir "${RUN_DIR}" \
    --data-dir "${DATA_DIR}" \
    --every-nth "${EVERY_NTH}" \
    --limit-de-examples "${LIMIT_DE_EXAMPLES}" \
    --seed "${SEED}" \
    --output "${OUTPUT}"

echo
echo "Finished: $(date)"
