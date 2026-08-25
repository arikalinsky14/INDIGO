#!/usr/bin/env bash
#SBATCH --job-name=indigo-gen-data-gpu
#SBATCH --output=job-outputs/indigo-gen-data-gpu.%j.out
#SBATCH --error=job-outputs/indigo-gen-data-gpu.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# INDIGO Data Generation — GPU variant.
# ============================================================================
#
# Same as slurms/generate_data.sh but runs on the l40s partition with JAX
# allowed to see the GPU. Multiple xargs workers share one L40S. This works
# because stackrt_n_k inside JLL is NOT JIT-able (bool cast on a Tracer),
# so per-forward cost is launch overhead + a small kernel — the GPU is idle
# most of the wall clock, letting many workers interleave.
#
# UNMEASURED: xargs-P scaling on a single L40S. The verify_datagen_gpu
# throughput number (~1.64 s/row) was for ONE process. If K workers sharing
# one L40S give effective scaling of ε (0 < ε ≤ 1), per-row wall drops to
# ~1.64/(K·ε) s. Run the pilot below (10k rows) to measure ε before firing
# the 10M-row loop.
#
# Multi-worker CUDA sharing:
#   * XLA_PYTHON_CLIENT_PREALLOCATE=false  — required (each process only
#     grabs VRAM it uses; else the first process eats the whole L40S).
#   * XLA_PYTHON_CLIENT_MEM_FRACTION=0.05  — cap each worker at 5% of the
#     48 GB L40S. 16 workers × 5% = 80% ceiling, fragmentation headroom OK.
#
# Examples
# --------
#
# 1. Pilot to measure xargs-P scaling on GPU (fast — ~30 min max):
#    TOTAL_ROWS=10000 ROWS_PER_SHARD=1000 OUTPUT_DIR=data/pilot_gpu \
#    HIGH_CHROMA_PROB=0.2 START_SHARD_ID=9000000 \
#        sbatch slurms/generate_data_gpu.sh
#    # Then divide wall-clock by 10000 to get sec/row at PARALLEL_WORKERS=16.
#
# 2. Production 1M-row shard on GPU (one of 10):
#    TOTAL_ROWS=1000000 START_SHARD_ID=0 HIGH_CHROMA_PROB=0.2 \
#        sbatch slurms/generate_data_gpu.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# Let JAX see the GPU. See slurms/verify_datagen_gpu.sh header for why
# just setting JAX_PLATFORMS isn't sufficient on the pytorch module.
unset CUDA_VISIBLE_DEVICES || true
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05

echo "============================================================================"
echo "INDIGO DATA GENERATION (GPU) - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
nproc

python -c "import jax; d = jax.devices(); print(f'[env] JAX platform: {d[0].platform}, device_count={len(d)}, devices={d}')" \
    || echo "[env] WARNING: jax.devices() call failed"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Volume.
TOTAL_ROWS="${TOTAL_ROWS:-1000000}"             # Per-job default: 1M rows
ROWS_PER_SHARD="${ROWS_PER_SHARD:-5000}"        # Rows per shard
START_SHARD_ID="${START_SHARD_ID:-0}"           # First shard id (use disjoint
                                                # ranges across submissions)

# Distribution.
LAYER_LAMBDA="${LAYER_LAMBDA:-4.5}"
LAYER_MIN="${LAYER_MIN:-2}"
LAYER_MAX="${LAYER_MAX:-10}"
GREYSCALE_THRESHOLD="${GREYSCALE_THRESHOLD:-8.0}"
GREYSCALE_KEEP_PROB="${GREYSCALE_KEEP_PROB:-0.2}"

# High-chroma-search path.
HIGH_CHROMA_PROB="${HIGH_CHROMA_PROB:-0.0}"
HIGH_CHROMA_CANDIDATE_COUNT="${HIGH_CHROMA_CANDIDATE_COUNT:-24}"
HIGH_CHROMA_REFINE_ITERS="${HIGH_CHROMA_REFINE_ITERS:-12}"
HIGH_CHROMA_OPTIMIZER="${HIGH_CHROMA_OPTIMIZER:-dog}"
HIGH_CHROMA_CHROMA_MIN="${HIGH_CHROMA_CHROMA_MIN:-60.0}"
HIGH_CHROMA_CHROMA_MAX="${HIGH_CHROMA_CHROMA_MAX:-110.0}"
HIGH_CHROMA_LIGHTNESS_MIN="${HIGH_CHROMA_LIGHTNESS_MIN:-25.0}"
HIGH_CHROMA_LIGHTNESS_MAX="${HIGH_CHROMA_LIGHTNESS_MAX:-75.0}"
P_REAL="${P_REAL:-0.15}"
ALL_REAL="${ALL_REAL:-0}"
POOL_SIZE_MIN="${POOL_SIZE_MIN:-4}"
POOL_SIZE_MAX="${POOL_SIZE_MAX:-32}"

# Train vs test set.
INCIDENCE_ANGLE="${INCIDENCE_ANGLE:-0}"
USE_HELD_OUT_REALS="${USE_HELD_OUT_REALS:-0}"

# IO.
OUTPUT_DIR="${OUTPUT_DIR:-data/train_gpu}"

# Parallelism — CPUs available on the l40s allocation (16 by default here).
PARALLEL_WORKERS="${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

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
echo "  Parallel workers:  ${PARALLEL_WORKERS} (via xargs -P, all sharing 1 L40S)"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Incidence angle:   ${INCIDENCE_ANGLE}"
echo "  Layer lambda:      ${LAYER_LAMBDA} (range [${LAYER_MIN}, ${LAYER_MAX}])"
echo "  Greyscale:         C* >= ${GREYSCALE_THRESHOLD} accepted, "
echo "                     C* < ${GREYSCALE_THRESHOLD} kept with prob ${GREYSCALE_KEEP_PROB}"
echo "  High-chroma:       prob=${HIGH_CHROMA_PROB} (0 = disabled)"
if [ "$(printf '%s\n' "${HIGH_CHROMA_PROB}" | awk '{ print ($1 > 0) }')" = "1" ]; then
  echo "                     candidate_count=${HIGH_CHROMA_CANDIDATE_COUNT}, "
  echo "                     refine_iters=${HIGH_CHROMA_REFINE_ITERS}, "
  echo "                     optimizer=${HIGH_CHROMA_OPTIMIZER}, "
  echo "                     C*∈[${HIGH_CHROMA_CHROMA_MIN}, ${HIGH_CHROMA_CHROMA_MAX}], "
  echo "                     L*∈[${HIGH_CHROMA_LIGHTNESS_MIN}, ${HIGH_CHROMA_LIGHTNESS_MAX}]"
fi
echo "  p_real:            ${P_REAL}"
echo "  All-real:          ${ALL_REAL}"
echo "  Pool size range:   [${POOL_SIZE_MIN}, ${POOL_SIZE_MAX}]"
echo "  Use held-out:      ${USE_HELD_OUT_REALS}"
echo "  JAX platforms:     ${JAX_PLATFORMS}"
echo "  XLA mem/proc:      ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo

mkdir -p "${OUTPUT_DIR}"
RUN_MANIFEST="${OUTPUT_DIR}/run_manifest.gpu.${SLURM_JOB_ID:-local}.json"
cat > "${RUN_MANIFEST}" <<EOF
{
  "created_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "backend": "gpu:l40s",
  "total_rows": ${TOTAL_ROWS},
  "rows_per_shard": ${ROWS_PER_SHARD},
  "n_shards": ${N_SHARDS},
  "shard_id_range": [${START_SHARD_ID}, ${END_SHARD_ID}],
  "incidence_angle": ${INCIDENCE_ANGLE},
  "layer_lambda": ${LAYER_LAMBDA},
  "layer_range": [${LAYER_MIN}, ${LAYER_MAX}],
  "greyscale_threshold": ${GREYSCALE_THRESHOLD},
  "greyscale_keep_prob": ${GREYSCALE_KEEP_PROB},
  "high_chroma_prob": ${HIGH_CHROMA_PROB},
  "high_chroma_candidate_count": ${HIGH_CHROMA_CANDIDATE_COUNT},
  "high_chroma_refine_iters": ${HIGH_CHROMA_REFINE_ITERS},
  "high_chroma_optimizer": "${HIGH_CHROMA_OPTIMIZER}",
  "high_chroma_chroma_range": [${HIGH_CHROMA_CHROMA_MIN}, ${HIGH_CHROMA_CHROMA_MAX}],
  "high_chroma_lightness_range": [${HIGH_CHROMA_LIGHTNESS_MIN}, ${HIGH_CHROMA_LIGHTNESS_MAX}],
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
        --high-chroma-prob "${HIGH_CHROMA_PROB}" \
        --high-chroma-candidate-count "${HIGH_CHROMA_CANDIDATE_COUNT}" \
        --high-chroma-refine-iters "${HIGH_CHROMA_REFINE_ITERS}" \
        --high-chroma-optimizer "${HIGH_CHROMA_OPTIMIZER}" \
        --high-chroma-chroma-min "${HIGH_CHROMA_CHROMA_MIN}" \
        --high-chroma-chroma-max "${HIGH_CHROMA_CHROMA_MAX}" \
        --high-chroma-lightness-min "${HIGH_CHROMA_LIGHTNESS_MIN}" \
        --high-chroma-lightness-max "${HIGH_CHROMA_LIGHTNESS_MAX}" \
        --skip-existing \
        ${HELD_OUT_FLAG} ${ALL_REAL_FLAG}

EXIT_CODE=$?

echo
echo "============================================================================"
echo "DATA GENERATION COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"

N_PRESENT=$(find "${OUTPUT_DIR}" -name "shard_*.parquet" 2>/dev/null | wc -l)
echo "Shards on disk: ${N_PRESENT} (expected ${N_SHARDS})"
echo "============================================================================"

exit ${EXIT_CODE}
