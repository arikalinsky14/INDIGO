#!/usr/bin/env bash
#SBATCH --job-name=indigo-thick-plot
#SBATCH --output=job-outputs/indigo-thick-plot.%j.out
#SBATCH --error=job-outputs/indigo-thick-plot.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# INDIGO Thickness Distribution Plot — wrapper
# ============================================================================
#
# Runs analyses/de_finetune/thickness_distribution_plot.py. Loads a
# finetune (or pretrain) checkpoint, runs ONE forward pass on N val
# examples, and produces a faceted grid of P(thickness | winning slot)
# for every layer position. Answers "are thickness predictions sharp,
# broad, or bimodal?" — see the script header for details.
#
# Cost: single forward pass, no simulator. ~1 min after cold module
# load. 30 min wall is a big safety margin.
#
# Required env vars:
#   CHECKPOINT     path to checkpoint dir (e.g. .../best/, .../step_N/)
#   OUTPUT_PATH    where to write the PNG
#
# Optional env vars:
#   DATA_DIR       default: data/finetune
#   SPLIT          default: validation
#   N_EXAMPLES     default: 32
#   GRID_COLS      default: 4
#   SEED           default: 42
#
# Usage — two plots (one per checkpoint), submit both:
#   CHECKPOINT=data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const/best \
#       OUTPUT_PATH=analyses/de_finetune/results/thickness_dist_best.png \
#       sbatch slurms/thickness_distribution_plot.sh
#
#   CHECKPOINT=data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#       OUTPUT_PATH=analyses/de_finetune/results/thickness_dist_pretrain.png \
#       sbatch slurms/thickness_distribution_plot.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# ---- Required inputs ----
if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "ERROR: set CHECKPOINT=<path/to/checkpoint_dir> before sbatch" >&2
    exit 2
fi
if [[ -z "${OUTPUT_PATH:-}" ]]; then
    echo "ERROR: set OUTPUT_PATH=<path/to/output.png> before sbatch" >&2
    exit 2
fi

# ---- Defaults ----
: "${DATA_DIR:=data/finetune}"
: "${SPLIT:=validation}"
: "${N_EXAMPLES:=32}"
: "${GRID_COLS:=4}"
: "${SEED:=42}"

# Ensure the output directory exists so matplotlib.savefig can write.
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "============================================================================"
echo "INDIGO THICKNESS DISTRIBUTION PLOT - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:         $(pwd)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo "Started:     $(date)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || \
    echo "[INFO] no GPU visible to this job"
echo
echo "----- Configuration -----"
echo "CHECKPOINT   : ${CHECKPOINT}"
echo "OUTPUT_PATH  : ${OUTPUT_PATH}"
echo "DATA_DIR     : ${DATA_DIR}"
echo "SPLIT        : ${SPLIT}"
echo "N_EXAMPLES   : ${N_EXAMPLES}"
echo "GRID_COLS    : ${GRID_COLS}"
echo "SEED         : ${SEED}"
echo "============================================================================"
echo

CMD=(
    python analyses/de_finetune/thickness_distribution_plot.py
        --checkpoint   "${CHECKPOINT}"
        --data-dir     "${DATA_DIR}"
        --split        "${SPLIT}"
        --n-examples   "${N_EXAMPLES}"
        --grid-cols    "${GRID_COLS}"
        --seed         "${SEED}"
        --output-path  "${OUTPUT_PATH}"
)

echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

EXIT_CODE=$?
echo
echo "============================================================================"
echo "THICKNESS PLOT COMPLETE   exit=${EXIT_CODE}   ended=$(date)"
echo "OUTPUT:  ${OUTPUT_PATH}"
echo "============================================================================"
exit ${EXIT_CODE}
