#!/usr/bin/env bash
#SBATCH --job-name=indigo-plot-train
#SBATCH --output=job-outputs/indigo-plot-train.%j.out
#SBATCH --error=job-outputs/indigo-plot-train.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

#SBATCH --time=00:15:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

# ============================================================================
# INDIGO training-curve plot from history.jsonl
# ============================================================================
#
# Reads scripts/training.py's per-save history.jsonl and produces a PNG
# with train+val loss (top) and LR schedule (bottom). Overlay multiple
# runs by passing multiple HISTORY files (space-separated).
#
# Env knobs:
#   HISTORY   required — space-separated list of history.jsonl paths.
#             Example: HISTORY="run_a/history.jsonl run_b/history.jsonl"
#   LABEL     optional — space-separated legend labels (same count as HISTORY).
#             Falls back to each history's parent directory name.
#   OUTPUT    optional — output PNG path (default: <first HISTORY dir>/training_curve.png)
#   TITLE     optional — plot title (default: "INDIGO training curve")
#
# Runs on smp (matplotlib only, no GPU) — cheap and never queues long.
#
# Example — single prod run:
#   HISTORY=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/flex_raw_spectrum_enc128-64_d1024_L8_do0.1_lr6e-05_bs512_ep1_cross_attnH8_se4_dec1/history.jsonl \
#       sbatch slurms/plot_training_curve.sh
#
# Example — overlay two runs:
#   HISTORY="path/to/run_a/history.jsonl path/to/run_b/history.jsonl" \
#   LABEL="baseline lr_up" \
#       sbatch slurms/plot_training_curve.sh
# ============================================================================

if [[ -z "${HISTORY:-}" ]]; then
    echo "ERROR: set HISTORY=<path/to/history.jsonl> (space-separated for overlay)" >&2
    exit 2
fi

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# Build --history repeated flags from the space-separated list.
ARGS=()
for h in ${HISTORY}; do
    ARGS+=(--history "$h")
done
if [[ -n "${LABEL:-}" ]]; then
    for l in ${LABEL}; do
        ARGS+=(--label "$l")
    done
fi
if [[ -n "${OUTPUT:-}" ]]; then
    ARGS+=(--output "${OUTPUT}")
fi
if [[ -n "${TITLE:-}" ]]; then
    ARGS+=(--title "${TITLE}")
fi

echo "======================================================================"
echo " INDIGO training-curve plot — Job ${SLURM_JOB_ID:-local}"
echo " HISTORY: ${HISTORY}"
echo " LABEL:   ${LABEL:-<default = parent dir names>}"
echo " OUTPUT:  ${OUTPUT:-<default = <first HISTORY dir>/training_curve.png>}"
echo "======================================================================"

python scripts/plot_training_curve.py "${ARGS[@]}"
