#!/usr/bin/env bash
#SBATCH --job-name=indigo-gen-data
#SBATCH --output=job-outputs/indigo-gen-data.%j.out
#SBATCH --error=job-outputs/indigo-gen-data.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# INDIGO Data Generation (parallel)
# ============================================================================
#
# Generates training (or test) shards in parallel using N CPU processes,
# where N defaults to ${SLURM_CPUS_PER_TASK}. The L40S GPU is allocated for
# queue access but unused by data generation:
#
#   * Each row's optical-sim call (stackrt_n_k) takes ~50-100ms on CPU.
#     There is no batching, so a GPU launch would not amortize and would
#     also serialize all parallel workers through one device.
#   * JAX is forced to CPU below for that reason.
#
# Each parallel worker calls compile_datasets.py with --skip-existing for
# one shard at a time. Re-running this script resumes from wherever the
# previous run stopped.
#
# Throughput (rough): a 5000-row shard takes ~5 min on one CPU core. With
# 32 workers, 2000 shards (10M rows) takes ~5 hours wall time.
#
# Examples
# --------
#
# 1. Production training set (2M rows, default settings):
#    sbatch slurms/generate_data.sh
#
# 2. Tier-B test set (held-out real materials, 50k rows, disjoint shard IDs):
#    TOTAL_ROWS=50000 START_SHARD_ID=2000000 \
#        OUTPUT_DIR=data/test/tier_b USE_HELD_OUT_REALS=1 \
#        sbatch slurms/generate_data.sh
#
# 3. Smaller / quicker test run:
#    TOTAL_ROWS=10000 ROWS_PER_SHARD=1000 OUTPUT_DIR=data/dryrun \
#        sbatch slurms/generate_data.sh
#
# 4. Fully-real dataset (zero synthetic — all JLL real materials).
#    Cap pool size at the real-material count to avoid duplicate slots
#    (~27 held-in; 8 held-out for the --use-held-out-reals case):
#    ALL_REAL=1 POOL_SIZE_MAX=27 OUTPUT_DIR=data/train_allreal \
#        sbatch slurms/generate_data.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# Force JAX/XLA to CPU so workers don't contend for the GPU. The optical
# simulator runs faster per-call on CPU than on GPU for unbatched calls.
export JAX_PLATFORMS=cpu

echo "============================================================================"
echo "INDIGO DATA GENERATION - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
nproc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Volume.
TOTAL_ROWS="${TOTAL_ROWS:-10000000}"             # Total rows across all shards
ROWS_PER_SHARD="${ROWS_PER_SHARD:-5000}"        # Rows per shard
START_SHARD_ID="${START_SHARD_ID:-0}"           # First shard id (use disjoint
                                                # ranges for test sets)

# Distribution.
LAYER_LAMBDA="${LAYER_LAMBDA:-4.5}"
LAYER_MIN="${LAYER_MIN:-2}"
LAYER_MAX="${LAYER_MAX:-10}"
GREYSCALE_THRESHOLD="${GREYSCALE_THRESHOLD:-8.0}"
GREYSCALE_KEEP_PROB="${GREYSCALE_KEEP_PROB:-0.2}"
P_REAL="${P_REAL:-0.15}"
ALL_REAL="${ALL_REAL:-0}"                        # 1 = fully-real dataset
                                                # (zero synthetic; forces p_real=1)
POOL_SIZE_MIN="${POOL_SIZE_MIN:-4}"
POOL_SIZE_MAX="${POOL_SIZE_MAX:-32}"

# Train vs test set.
INCIDENCE_ANGLE="${INCIDENCE_ANGLE:-0}"
USE_HELD_OUT_REALS="${USE_HELD_OUT_REALS:-0}"   # 1 for Tier-B test set

# IO.
OUTPUT_DIR="${OUTPUT_DIR:-data/train}"

# Parallelism.
PARALLEL_WORKERS="${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"

# Compute shard range.
N_SHARDS=$(( (TOTAL_ROWS + ROWS_PER_SHARD - 1) / ROWS_PER_SHARD ))
END_SHARD_ID=$(( START_SHARD_ID + N_SHARDS - 1 ))

HELD_OUT_FLAG=""
if [[ "${USE_HELD_OUT_REALS}" == "1" ]]; then
    HELD_OUT_FLAG="--use-held-out-reals"
fi

ALL_REAL_FLAG=""
if [[ "${ALL_REAL}" == "1" ]]; then
    ALL_REAL_FLAG="--all-real"
fi

echo
echo "Configuration:"
echo "  Total rows:        ${TOTAL_ROWS}"
echo "  Rows/shard:        ${ROWS_PER_SHARD}"
echo "  Shards:            ${N_SHARDS} (ids ${START_SHARD_ID}..${END_SHARD_ID})"
echo "  Parallel workers:  ${PARALLEL_WORKERS} (via xargs -P)"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Incidence angle:   ${INCIDENCE_ANGLE}"
echo "  Layer lambda:      ${LAYER_LAMBDA} (range [${LAYER_MIN}, ${LAYER_MAX}])"
echo "  Greyscale:         C* >= ${GREYSCALE_THRESHOLD} accepted, "
echo "                     C* < ${GREYSCALE_THRESHOLD} kept with prob ${GREYSCALE_KEEP_PROB}"
echo "  p_real:            ${P_REAL}"
echo "  All-real:          ${ALL_REAL} (1 = zero synthetic, p_real forced to 1)"
echo "  Pool size range:   [${POOL_SIZE_MIN}, ${POOL_SIZE_MAX}]"
echo "  Use held-out:      ${USE_HELD_OUT_REALS}"
echo "  JAX platforms:     ${JAX_PLATFORMS}"
echo

# Write a top-level run manifest before forking workers. Each shard also gets
# its own .manifest.json sidecar from compile_datasets.py.
mkdir -p "${OUTPUT_DIR}"
RUN_MANIFEST="${OUTPUT_DIR}/run_manifest.json"
cat > "${RUN_MANIFEST}" <<EOF
{
  "created_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "total_rows": ${TOTAL_ROWS},
  "rows_per_shard": ${ROWS_PER_SHARD},
  "n_shards": ${N_SHARDS},
  "shard_id_range": [${START_SHARD_ID}, ${END_SHARD_ID}],
  "incidence_angle": ${INCIDENCE_ANGLE},
  "layer_lambda": ${LAYER_LAMBDA},
  "layer_range": [${LAYER_MIN}, ${LAYER_MAX}],
  "greyscale_threshold": ${GREYSCALE_THRESHOLD},
  "greyscale_keep_prob": ${GREYSCALE_KEEP_PROB},
  "p_real": ${P_REAL},
  "all_real": ${ALL_REAL},
  "pool_size_range": [${POOL_SIZE_MIN}, ${POOL_SIZE_MAX}],
  "use_held_out_reals": ${USE_HELD_OUT_REALS},
  "parallel_workers": ${PARALLEL_WORKERS}
}
EOF
echo "Wrote ${RUN_MANIFEST}"
echo

# ============================================================================
# PARALLEL EXECUTION
# ============================================================================
#
# Pipe shard ids into xargs -P which spawns up to PARALLEL_WORKERS child
# processes. Each compile_datasets.py call is idempotent under --skip-existing,
# so the script is safe to re-run on a partial output directory.

echo "Launching workers..."
echo

seq "${START_SHARD_ID}" "${END_SHARD_ID}" | xargs -P "${PARALLEL_WORKERS}" -I '{}' \
    python create_dataset/src/compile_datasets.py \
        --target-rows "${ROWS_PER_SHARD}" \
        --shard-id '{}' \
        --incidence-angle "${INCIDENCE_ANGLE}" \
        --layer-lambda "${LAYER_LAMBDA}" \
        --layer-min "${LAYER_MIN}" \
        --layer-max "${LAYER_MAX}" \
        --greyscale-threshold "${GREYSCALE_THRESHOLD}" \
        --greyscale-keep-prob "${GREYSCALE_KEEP_PROB}" \
        --p-real "${P_REAL}" \
        --pool-size-min "${POOL_SIZE_MIN}" \
        --pool-size-max "${POOL_SIZE_MAX}" \
        --output-dir "${OUTPUT_DIR}" \
        --skip-existing \
        ${HELD_OUT_FLAG} ${ALL_REAL_FLAG}

EXIT_CODE=$?

echo
echo "============================================================================"
echo "DATA GENERATION COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"

# Final tally.
N_PRESENT=$(find "${OUTPUT_DIR}" -name "shard_*.parquet" 2>/dev/null | wc -l)
echo "Shards on disk: ${N_PRESENT} (expected ${N_SHARDS})"
echo "============================================================================"

exit ${EXIT_CODE}
