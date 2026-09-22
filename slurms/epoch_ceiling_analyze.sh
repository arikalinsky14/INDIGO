#!/usr/bin/env bash
#SBATCH --job-name=indigo-epoch-ceiling-analyze
#SBATCH --output=job-outputs/indigo-epoch-ceiling-analyze.%j.out
#SBATCH --error=job-outputs/indigo-epoch-ceiling-analyze.%j.err

# CPU only: these modes read JSON and fit nothing on a GPU. Running them on
# the gpu cluster would idle an L40S.
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

#SBATCH --time=00:20:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Read-only companion to slurms/epoch_ceiling_probe.sh
# ============================================================================
#
# Python cannot be run directly on the CRC login nodes, so the probe's
# inspection modes need a batch wrapper too. All three are read-only -- none
# of them trains anything or writes into a checkpoint dir.
#
#   MODE=dry-run   Print the arm grid, the fixed-C assertion and the wall-time
#                  budget WITHOUT submitting anything. Run this first: it also
#                  prints the arm count that --array must match.
#   MODE=analyze   (default) Read the finished arms and report the epoch
#                  ceiling, per-chroma breakdown and the seed noise floor.
#   MODE=diagnose  Recover real throughput from partial or timed-out runs.
#
# The knobs below must MATCH the probe submission, or the grid this computes
# will not be the grid that ran.
#
# Usage:
#   # 1. before submitting the probe
#   MODE=dry-run TOTAL_STEPS=9600 SEEDS="42 43" \
#       SIZES="128:2:5.48e-4 256:2:5.48e-4 512:4:1e-4" \
#       sbatch slurms/epoch_ceiling_analyze.sh
#
#   # 2. after the probe finishes
#   MODE=analyze sbatch slurms/epoch_ceiling_analyze.sh
#
#   # 3. if arms time out
#   MODE=diagnose TOTAL_STEPS=9600 sbatch slurms/epoch_ceiling_analyze.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

MODE="${MODE:-analyze}"
OUT_ROOT="${OUT_ROOT:-data/checkpoints/epoch_ceiling_probe}"
TOTAL_STEPS="${TOTAL_STEPS:-9600}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEPTHS="${DEPTHS:-1 2 4 8}"
SIZES="${SIZES:-128:2 256:2 512:4}"
SEEDS="${SEEDS:-42}"

echo "============================================================================"
echo "INDIGO EPOCH-CEILING ${MODE^^} - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:         $(pwd)"
echo "Started:     $(date)"
echo "MODE:        ${MODE}"
echo "OUT_ROOT:    ${OUT_ROOT}"
echo "TOTAL_STEPS: ${TOTAL_STEPS}  BATCH_SIZE: ${BATCH_SIZE}"
echo "SIZES:       ${SIZES}"
echo "DEPTHS:      ${DEPTHS}"
echo "SEEDS:       ${SEEDS}"
echo

case "${MODE}" in
  dry-run)
    python -u scripts/epoch_ceiling_probe.py --dry-run \
        --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
        --depths ${DEPTHS} --sizes ${SIZES} --seeds ${SEEDS}
    echo
    N_ARMS=$(python scripts/epoch_ceiling_probe.py --n-arms \
        --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
        --depths ${DEPTHS} --sizes ${SIZES} --seeds ${SEEDS})
    echo "ARM COUNT: ${N_ARMS}"
    echo "Submit the probe with --array=0-$((N_ARMS-1)) and the SAME"
    echo "TOTAL_STEPS / BATCH_SIZE / DEPTHS / SIZES / SEEDS as above."
    ;;
  analyze)
    # TOTAL_STEPS/BATCH_SIZE/DEPTHS must match the submission: they select
    # WHICH runs under OUT_ROOT are analysed. Successive probe runs share the
    # directory and their corpora overlap, so without this the analysis
    # averages different experiments together.
    python -u scripts/epoch_ceiling_probe.py --analyze \
        --out-root "${OUT_ROOT}" \
        --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
        --depths ${DEPTHS}
    ;;
  diagnose)
    python -u scripts/epoch_ceiling_probe.py --diagnose \
        --out-root "${OUT_ROOT}" \
        --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}"
    ;;
  *)
    echo "ERROR: unknown MODE '${MODE}'. Expected dry-run, analyze or diagnose." >&2
    exit 1
    ;;
esac

echo
echo "Finished: $(date)"
