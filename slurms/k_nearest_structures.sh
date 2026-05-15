#!/usr/bin/env bash
#SBATCH --job-name=indigo-knn
#SBATCH --output=job-outputs/indigo-knn.%j.out
#SBATCH --error=job-outputs/indigo-knn.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# INDIGO k-Nearest-Structures
# ============================================================================
#
# CPU + I/O bound (no GPU): scans training-shard Lab targets, ranks by
# CIEDE2000 against a query color, re-simulates the top-k matches and
# writes a reflectance + swatch visualization. Runs on the SMP partition
# so it doesn't hog the login node.
#
# Rough wall time:
#   SCAN_ROWS=100000  (default)  ~1-2 min
#   SCAN_ROWS=1000000            ~2-4 min
#   full 10M dataset             ~10-20 min
# (Dominated by the per-shard footer enumeration + the CIEDE2000 loop,
#  not the k-match re-simulation.)
#
# Query is specified as EITHER sRGB or Lab (mutually exclusive):
#
#   # sRGB query
#   QUERY_RGB="220 30 30" sbatch slurms/k_nearest_structures.sh
#
#   # Lab query
#   QUERY_LAB="50 60 40" sbatch slurms/k_nearest_structures.sh
#
#   # Scan more of the dataset, more matches, custom output
#   QUERY_RGB="0 100 200" K=12 SCAN_ROWS=1000000 \
#       OUTPUT=outputs/blue_neighbors.png \
#       sbatch slurms/k_nearest_structures.sh
#
#   # Search the Tier-B test set instead of training
#   DATA_DIR=data/test/tier_b QUERY_LAB="70 -40 30" \
#       sbatch slurms/k_nearest_structures.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# No GPU work here; the k-match re-simulation uses jaxlayerlumos on CPU.
export JAX_PLATFORMS=cpu

echo "============================================================================"
echo "INDIGO K-NEAREST-STRUCTURES - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR="${DATA_DIR:-data/train}"
K="${K:-8}"
SCAN_ROWS="${SCAN_ROWS:-100000}"
OUTPUT="${OUTPUT:-outputs/k_nearest_structures.png}"

# Exactly one of QUERY_RGB / QUERY_LAB must be set (space-separated triple).
QUERY_RGB="${QUERY_RGB:-}"
QUERY_LAB="${QUERY_LAB:-}"

if [[ -n "${QUERY_RGB}" && -n "${QUERY_LAB}" ]]; then
    echo "[ERROR] Set only one of QUERY_RGB or QUERY_LAB, not both." >&2
    exit 1
fi
if [[ -z "${QUERY_RGB}" && -z "${QUERY_LAB}" ]]; then
    echo "[ERROR] Set one of QUERY_RGB or QUERY_LAB (space-separated triple)." >&2
    echo "        e.g.  QUERY_RGB=\"220 30 30\" sbatch slurms/k_nearest_structures.sh" >&2
    exit 1
fi

ARGS=(
    --data-dir "${DATA_DIR}"
    --k "${K}"
    --scan-rows "${SCAN_ROWS}"
    --output "${OUTPUT}"
    --verbose
)

if [[ -n "${QUERY_RGB}" ]]; then
    # shellcheck disable=SC2206
    RGB_ARR=(${QUERY_RGB})
    ARGS+=(--query-rgb "${RGB_ARR[@]}")
else
    # shellcheck disable=SC2206
    LAB_ARR=(${QUERY_LAB})
    ARGS+=(--query-lab "${LAB_ARR[@]}")
fi

# Any extra sbatch positional args flow through.
USER_ARGS=("$@")

CMD=(python scripts/k_nearest_structures.py "${ARGS[@]}" "${USER_ARGS[@]}")

echo "Configuration:"
echo "  Data dir:   ${DATA_DIR}"
echo "  Query RGB:  ${QUERY_RGB:-(unset)}"
echo "  Query Lab:  ${QUERY_LAB:-(unset)}"
echo "  k:          ${K}"
echo "  Scan rows:  ${SCAN_ROWS}"
echo "  Output:     ${OUTPUT}"
echo
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

"${CMD[@]}"

EXIT_CODE=$?
echo
echo "============================================================================"
echo "K-NEAREST-STRUCTURES COMPLETE"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"
exit ${EXIT_CODE}
