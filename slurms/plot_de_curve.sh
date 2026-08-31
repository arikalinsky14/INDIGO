#!/usr/bin/env bash
#SBATCH --job-name=indigo-plot-de
#SBATCH --output=job-outputs/indigo-plot-de.%j.out
#SBATCH --error=job-outputs/indigo-plot-de.%j.err

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
# INDIGO ΔE-curve plot from de_curve.sh's per-checkpoint eval_step_*.json
# ============================================================================
#
# The de_curve.sh SLURM auto-plots its own OUT_DIR on completion; use
# this wrapper only when you want to OVERLAY multiple splits (val +
# tier_a + tier_b), or re-plot an older run that predates the in-line
# plot step.
#
# Env knobs:
#   INPUT_DIR   required — space-separated list of OUT_DIRs from de_curve.sh.
#                          Each contributes one series (mean+median+p75).
#                          Example:
#                            INPUT_DIR="ckpt/de_curve ckpt/de_curve_tier_a ckpt/de_curve_tier_b"
#   LABEL       optional — space-separated legend labels (same count as INPUT_DIR).
#                          Falls back to each dir's basename.
#   OUTPUT      optional — output PNG path (default: <first INPUT_DIR>/de_curve.png).
#   TITLE       optional — plot title.
#
# Runs on smp (matplotlib only, no GPU). 15 min wall.
#
# Example — overlay val + tier_a + tier_b:
#   CKPT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/flex_raw_spectrum_enc128-64_d1024_L8_do0.1_lr6e-05_bs512_ep1_cross_attnH8_se4_dec1
#   INPUT_DIR="$CKPT/de_curve $CKPT/de_curve_tier_a $CKPT/de_curve_tier_b" \
#   LABEL="val tier_a tier_b" \
#   OUTPUT=$CKPT/de_curve_overlay.png \
#       sbatch slurms/plot_de_curve.sh
# ============================================================================

if [[ -z "${INPUT_DIR:-}" ]]; then
    echo "ERROR: set INPUT_DIR=<path/to/de_curve_dir> (space-separated for overlay)" >&2
    exit 2
fi

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

ARGS=()
for d in ${INPUT_DIR}; do
    ARGS+=(--input-dir "$d")
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
echo " INDIGO ΔE-curve plot — Job ${SLURM_JOB_ID:-local}"
echo " INPUT_DIR: ${INPUT_DIR}"
echo " LABEL:     ${LABEL:-<default = dir basenames>}"
echo " OUTPUT:    ${OUTPUT:-<default = <first INPUT_DIR>/de_curve.png>}"
echo "======================================================================"

python scripts/plot_de_curve.py "${ARGS[@]}"
