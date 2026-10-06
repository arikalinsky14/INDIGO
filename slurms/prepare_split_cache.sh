#!/usr/bin/env bash
#SBATCH --job-name=indigo-split-cache
#SBATCH --output=job-outputs/indigo-split-cache.%j.out
#SBATCH --error=job-outputs/indigo-split-cache.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=03:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Build the validation-split cache on a CPU node, before a GPU array starts
# ============================================================================
#
# Every lr_grid.sh task reads the same validation rows (10,000, a few from
# each of ~3,600 shards), and reading them was most of a task's ~28-minute
# startup. scripts/lr_tuning.py now keeps those rows in one parquet file under
# cache/splits/ after the first read (src/dataset.py load_split_cached). This
# job does that first read on smp, at 0.8 SU per core-hour instead of on a GPU,
# so no GPU task pays for it:
#
#   sbatch slurms/prepare_split_cache.sh          # wait for the END mail
#   STAGE=2 sbatch --array=... slurms/lr_grid.sh
#
# smp and gpu are separate clusters, so a --dependency between them is not
# something to rely on: submit the array once this has finished. If it has
# not, nothing breaks; the first GPU tasks build the cache themselves.
#
# SEEDS lists the run seeds to prepare (each seed has its own validation
# slice); stage 2 needs 42, stage 3 and the beta2 study 42, 43, 44. The
# arguments below must match what lr_grid.sh passes, or the cache key will not
# match and the GPU tasks will build their own (correct, just slower).
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"

DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
for SEED in ${SEEDS:-42}; do
    echo "=== seed ${SEED}"
    python -u scripts/lr_tuning.py --prepare-split-cache --epochs 1 \
        --data-dir "${DATA_DIR}" --seed "${SEED}" \
        --limit-examples 614400 --limit-shard-aligned \
        --limit-val-examples "${LIMIT_VAL_EXAMPLES:-10000}" \
        --split-cache-dir "${SPLIT_CACHE_DIR:-cache/splits}" \
        --streaming --head-mode cross_attn --output-dir /tmp/unused
done
