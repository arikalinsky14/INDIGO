#!/usr/bin/env bash
#SBATCH --job-name=indigo-verify-datagen-gpu
#SBATCH --output=job-outputs/indigo-verify-datagen-gpu.%j.out
#SBATCH --error=job-outputs/indigo-verify-datagen-gpu.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# GPU variant of the data-generation throughput verifier.
# ============================================================================
#
# Same test as slurms/verify_datagen.sh but lets JAX see the L40S. Use it
# to A/B-benchmark whether the high-chroma-search path is faster on GPU
# than on CPU on your data-gen shape (per row cost ~4.5s on CPU last we
# measured; the ceiling has been dominated by 12x jax.grad calls that
# don't vmap because of a JLL assert — see inference/src/simulate.py).
#
# The stackrt_n_k call inside JLL is NOT jit'ed (a bool cast on a Tracer
# blocks it). So per-call XLA launch overhead is paid for every
# forward + backward. On small (MAX_LAYERS=10, NUM_LAMBDA=128) arrays,
# a discrete GPU sometimes LOSES to CPU because launch overhead dominates.
# The whole point of this script is to measure that empirically for our
# shapes with the NEW KK-consistent Lorentz-permutation materials.
#
# The Python script prints the JAX platform and device count at startup
# so you can confirm the run actually lands on GPU (`platform=gpu`,
# device_count=1).
#
# Env knobs — same as verify_datagen.sh:
#   OUTPUT_DIR                     default: data/verify_2h_gpu
#   N_ROWS                         default: 200
#   HIGH_CHROMA_PROB               default: 0.2
#   HIGH_CHROMA_CANDIDATE_COUNT    default: 24
#   HIGH_CHROMA_REFINE_ITERS       default: 12
#   PARALLEL_WORKERS               default: 32 (for 10M-row ETA extrapolation)
#   KK_SLACK_FACTOR                default: 2.5
#
# Example:
#   sbatch slurms/verify_datagen_gpu.sh
#   # A/B comparison against CPU: submit both, then diff the timing_reports.
# ============================================================================

: "${OUTPUT_DIR:=data/verify_2h_gpu}"
: "${N_ROWS:=200}"
: "${HIGH_CHROMA_PROB:=0.2}"
: "${HIGH_CHROMA_CANDIDATE_COUNT:=24}"
: "${HIGH_CHROMA_REFINE_ITERS:=12}"
: "${PARALLEL_WORKERS:=32}"
: "${KK_SLACK_FACTOR:=2.5}"

echo "======================================================================"
echo " INDIGO data-gen throughput verifier — GPU variant"
echo " OUTPUT_DIR                  : $OUTPUT_DIR"
echo " N_ROWS                      : $N_ROWS"
echo " HIGH_CHROMA_PROB            : $HIGH_CHROMA_PROB"
echo " HIGH_CHROMA_CANDIDATE_COUNT : $HIGH_CHROMA_CANDIDATE_COUNT"
echo " HIGH_CHROMA_REFINE_ITERS    : $HIGH_CHROMA_REFINE_ITERS"
echo " PARALLEL_WORKERS            : $PARALLEL_WORKERS   (for 10M-row ETA)"
echo " KK_SLACK_FACTOR             : $KK_SLACK_FACTOR"
echo "======================================================================"

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"

# Let JAX see the L40S — the whole point. If the environment's JAX build
# has a GPU plugin, JAX_PLATFORMS defaults to preferring GPU; being
# explicit avoids ambiguity.
unset CUDA_VISIBLE_DEVICES || true
export JAX_PLATFORMS=cuda,cpu

# Give TF/JAX room to allocate on the L40S but don't pre-block the full
# VRAM (leaves the mem-fragmentation escape hatch on).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[env] JAX_PLATFORMS=${JAX_PLATFORMS}"
python -c "import jax; d = jax.devices(); print(f'[env] JAX platform: {d[0].platform}, device_count={len(d)}, devices={d}')" \
    || echo "[env] WARNING: jax.devices() call failed — falling back may happen"

mkdir -p "$(dirname "$OUTPUT_DIR")"

python scripts/verify_datagen.py \
    --output-dir "$OUTPUT_DIR" \
    --n-rows "$N_ROWS" \
    --high-chroma-prob "$HIGH_CHROMA_PROB" \
    --high-chroma-candidate-count "$HIGH_CHROMA_CANDIDATE_COUNT" \
    --high-chroma-refine-iters "$HIGH_CHROMA_REFINE_ITERS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --kk-slack-factor "$KK_SLACK_FACTOR"

echo
echo "Artefacts:"
echo "  $OUTPUT_DIR/dryrun.parquet"
echo "  $OUTPUT_DIR/dryrun.manifest.json"
echo "  $OUTPUT_DIR/curves.png"
echo "  $OUTPUT_DIR/kk_report.json"
echo "  $OUTPUT_DIR/timing_report.json"
echo
echo "To A/B compare vs CPU baseline:"
echo "  jq '.sec_per_search, .per_row_avg_s' $OUTPUT_DIR/timing_report.json"
echo "  jq '.sec_per_search, .per_row_avg_s' data/verify_2h/timing_report.json"
