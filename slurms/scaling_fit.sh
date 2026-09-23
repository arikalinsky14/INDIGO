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

#SBATCH --time=00:30:00
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
#   MODE=ladder    Analyse the FIXED-N data ladder under LADDER_ROOT: val_de
#                  vs D at constant N, local slopes, and a saturation
#                  verdict. A separate mode because fit_scaling.py must not
#                  see these runs -- each ladder point is its own budget, so
#                  it would read them as one-point budgets and refuse.
#   MODE=all       fit, then ladder. What to run once every job is done.
#
# BUDGETS/SPAN/POINTS/REPEAT_SEED must MATCH the sweep submission in dry-run mode,
# or the grid this prints will not be the grid that ran. In fit mode they are
# unused: the fit reads N and C back out of each run's own config.json and
# history.jsonl, so it cannot silently inherit a stale grid.
#
# Usage:
#   MODE=dry-run sbatch slurms/scaling_fit.sh
#   MODE=fit OUT_ROOT=data/checkpoints/scaling_sweep_12h sbatch slurms/scaling_fit.sh
#
#   # once every sweep AND ladder job is done -- fit plus ladder in one go:
#   MODE=all OUT_ROOT=data/checkpoints/scaling_sweep_12h \
#       LADDER_ROOT=data/checkpoints/data_ladder_n1M \
#       sbatch slurms/scaling_fit.sh
#
#   # the ladder's saturation verdict needs a noise figure; take it from the
#   # repeat-seed pairs the fit reports, then re-run ladder mode alone:
#   MODE=ladder NOISE_DE=0.4 sbatch slurms/scaling_fit.sh
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
BUDGETS="${BUDGETS:-1e14 4e14 1.4e15 4e15}"
SPAN="${SPAN:-10}"
POINTS="${POINTS:-6}"
REPEAT_SEED="${REPEAT_SEED:-43}"
MAX_WALL_HOURS="${MAX_WALL_HOURS:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"

GRID_ARGS="--budgets ${BUDGETS} --span ${SPAN} --points ${POINTS} --batch-size ${BATCH_SIZE}"
[[ -n "${REPEAT_SEED}" ]] && GRID_ARGS="${GRID_ARGS} --repeat-seed ${REPEAT_SEED}"
[[ -n "${MAX_WALL_HOURS}" ]] && GRID_ARGS="${GRID_ARGS} --max-wall-hours ${MAX_WALL_HOURS}"

# The fit reads whichever val_de the run recorded. "final" is the honest
# default for a scaling law: "best" would select the minimum over a noisy
# eval trace, biasing every point downward by a size-dependent amount and
# therefore tilting alpha.
METRIC_SELECTION="${METRIC_SELECTION:-final}"
OUTPUT="${OUTPUT:-analyses/scaling/results/isoflop_fit.json}"
LADDER_ROOT="${LADDER_ROOT:-data/checkpoints/data_ladder_n1M}"
LADDER_OUTPUT="${LADDER_OUTPUT:-analyses/scaling/results/data_ladder.json}"
# Paired run-to-run val_de spread, for the ladder's saturation verdict. Left
# EMPTY on purpose: it should come from the IsoFLOP sweep's --repeat-seed
# arms, which measure it directly. Without it the ladder reports slopes and
# declines to call saturation, which is the honest default -- inventing a
# noise figure is how the epoch-ceiling probe first reported a ceiling that
# did not exist.
NOISE_DE="${NOISE_DE:-}"

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

run_fit() {
  if [[ ! -d "${OUT_ROOT}" ]]; then
    echo "ERROR: OUT_ROOT '${OUT_ROOT}' does not exist -- nothing to fit." >&2
    echo "       Run the sweep first: sbatch slurms/scaling_sweep.sh" >&2
    return 1
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
}

run_ladder() {
  if [[ ! -d "${LADDER_ROOT}" ]]; then
    echo "NOTE: LADDER_ROOT '${LADDER_ROOT}' does not exist; skipping the data"
    echo "      ladder. Run it with DATA_LADDER=1 on the sweep first."
    return 0
  fi
  mkdir -p "$(dirname "${LADDER_OUTPUT}")"
  local args=(--runs-root "${LADDER_ROOT}" --output "${LADDER_OUTPUT}")
  if [[ -n "${NOISE_DE}" ]]; then
    args+=(--noise-de "${NOISE_DE}")
  fi
  python -u scripts/analyse_data_ladder.py "${args[@]}"
  echo
  echo "Wrote ${LADDER_OUTPUT}"
}

run_dry_run() {
  python -u scripts/scaling_sweep.py --dry-run ${GRID_ARGS} \
      --out-root "${OUT_ROOT}"
  echo
  echo "----------------------------------------------------------------------------"
  local n
  n=$(python scripts/scaling_sweep.py --n-configs ${GRID_ARGS})
  echo "Set slurms/scaling_sweep.sh to: #SBATCH --array=0-$((n-1))"
}

case "${MODE}" in
  dry-run) run_dry_run ;;
  fit)     run_fit ;;
  ladder)  run_ladder ;;
  all)
    run_fit
    echo
    echo "============================================================================"
    run_ladder
    ;;
  *)
    echo "ERROR: unknown MODE='${MODE}'" >&2
    echo "       expected one of: dry-run, fit, ladder, all" >&2
    exit 1
    ;;
esac

echo
echo "Finished: $(date)"
