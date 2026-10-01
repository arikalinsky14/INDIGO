#!/usr/bin/env bash
#SBATCH --job-name=indigo-check-corpus
#SBATCH --output=job-outputs/indigo-check-corpus.%j.out
#SBATCH --error=job-outputs/indigo-check-corpus.%j.err

# CPU only, and light: the row-count check reads each parquet's FOOTER, not
# its row groups, so it is thousands of small metadata reads rather than a
# scan of 352 GiB.
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
# Corpus integrity + space check
# ============================================================================
#
# Exists as a SLURM job because the login node's default python has no
# pyarrow, so running scripts/check_corpus.py there silently skips the
# row-count check -- which is the half that catches a truncated shard. The
# sidecar check alone does not: a worker killed after the parquet was fully
# written but before the sidecar is rare compared with one killed mid-write,
# and only the footer read distinguishes a short shard from a healthy one.
#
# Run it BEFORE extending a corpus and AFTER the extension finishes.
#
#   DATA_DIR        default data/train
#   EXTEND_SHARDS   default 0; projects the disk cost of adding this many
#
# Usage:
#   sbatch slurms/check_corpus.sh
#   EXTEND_SHARDS=3000 sbatch slurms/check_corpus.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p job-outputs

DATA_DIR="${DATA_DIR:-data/train}"
EXTEND_SHARDS="${EXTEND_SHARDS:-0}"

echo "Started: $(date)"
python -c "import pyarrow; print(f'pyarrow {pyarrow.__version__} available')"
echo

python -u scripts/check_corpus.py \
    --data-dir "${DATA_DIR}" \
    --extend-shards "${EXTEND_SHARDS}"
RC=$?

echo
echo "Finished: $(date)"
echo "Exit code: ${RC}  (1 = problems found; see INTEGRITY above)"
exit ${RC}
