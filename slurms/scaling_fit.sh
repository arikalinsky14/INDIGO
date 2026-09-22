#!/usr/bin/env bash
#SBATCH --job-name=indigo-scaling-fit
#SBATCH --output=job-outputs/indigo-scaling-fit.%j.out
#SBATCH --error=job-outputs/indigo-scaling-fit.%j.err

# CPU only: both modes read JSON and fit small least-squares problems. Neither
# touches a GPU, so running this on the gpu cluster would idle an L40S.
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
# Read-only companion to slurms/scaling_sweep.sh
# ============================================================================
#
# Python cannot be run directly on the CRC login nodes, so the sweep's
# planning and fitting steps need a batch wrapper too. Both modes are
# read-only: neither trains anything nor writes into a checkpoint dir.
#
#   MODE=dry-run   Print the IsoFLOP grid -- every (budget, N, D) point, its
#                  LR, its estimated wall time, whether it fits --qos=short
#                  and the total GPU-hours. Run this FIRST: it also prints
#                  the config count that scaling_sweep.sh's --array must
#                  match, and it is the only place the bracket is checked to
#                  actually straddle each rung's prior N*.
#   MODE=fit       (default) Read the finished runs under OUT_ROOT and run
#                  the two-step IsoFLOP fit: per-budget parabola in log N ->
#                  N*(C_i), then a power law across budgets -> alpha, beta.
#                  Pooled and per-chroma-bucket.
#
# BUDGETS/BRACKET/BATCH_SIZE must MATCH the sweep submission in dry-run mode,
# or the grid this prints will not be the grid that ran. In fit mode they are
# unused: the fit reads N and C back out of each run's own config.json and
# history.jsonl, so it cannot silently inherit a stale grid.
#
# Usage:
#   MODE=dry-run sbatch slurms/scaling_fit.sh
#   MODE=fit sbatch slurms/scaling_fit.sh
#   MODE=fit OUT_ROOT=data/checkpoints/scaling_sweep sbatch slurms/scaling_fit.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p job-outputs

MODE="${MODE:-fit}"
OUT_ROOT="${OUT_ROOT:-data/checkpoints/scaling_sweep}"
BUDGETS="${BUDGETS:-1e14 3.7e14 1.4e15 5e15}"
BRACKET="${BRACKET:-0.6 0.8 1.0 1.3 1.7}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# The fit reads whichever val_de the run recorded. "final" is the honest
# default for a scaling law: "best" would select the minimum over a noisy
# eval trace, biasing every point downward by a size-dependent amount and
# therefore tilting alpha.
METRIC_SELECTION="${METRIC_SELECTION:-final}"
OUTPUT="${OUTPUT:-analyses/scaling/results/isoflop_fit.json}"

# Lets the report express each run's D as a fraction of an epoch. Purely
# presentational -- the fit itself uses absolute example-passes -- but it is
# how we check at a glance that no rung quietly ran past the repeat depth the
# epoch-ceiling probe covered.
CORPUS_EXAMPLES="${CORPUS_EXAMPLES:-$(python -c \
    'from src.scaling.configs import CORPUS_EXAMPLES; print(CORPUS_EXAMPLES)')}"

echo "============================================================================"
echo "INDIGO SCALING FIT - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:       $(pwd)"
echo "Started:   $(date)"
echo "MODE:      ${MODE}"
echo "OUT_ROOT:  ${OUT_ROOT}"
echo

case "${MODE}" in
  dry-run)
    python -u scripts/scaling_sweep.py --dry-run \
        --budgets ${BUDGETS} --bracket ${BRACKET} \
        --batch-size "${BATCH_SIZE}" --out-root "${OUT_ROOT}"
    echo
    echo "----------------------------------------------------------------------------"
    N_CONFIGS=$(python scripts/scaling_sweep.py --n-configs \
        --budgets ${BUDGETS} --bracket ${BRACKET} --batch-size "${BATCH_SIZE}")
    echo "Set slurms/scaling_sweep.sh to: #SBATCH --array=0-$((N_CONFIGS-1))"
    ;;

  fit)
    if [[ ! -d "${OUT_ROOT}" ]]; then
      echo "ERROR: OUT_ROOT '${OUT_ROOT}' does not exist -- nothing to fit." >&2
      echo "       Run the sweep first: sbatch slurms/scaling_sweep.sh" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${OUTPUT}")"
    python -u scripts/fit_scaling.py \
        --runs-root "${OUT_ROOT}" \
        --metric-selection "${METRIC_SELECTION}" \
        --corpus-examples "${CORPUS_EXAMPLES}" \
        --output "${OUTPUT}" \
        --plot
    echo
    echo "Wrote ${OUTPUT}"
    echo "Plot:  ${OUTPUT%.json}.png"
    ;;

  *)
    echo "ERROR: unknown MODE='${MODE}' (expected dry-run or fit)" >&2
    exit 1
    ;;
esac

echo
echo "Finished: $(date)"
