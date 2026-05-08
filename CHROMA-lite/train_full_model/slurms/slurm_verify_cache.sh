#!/usr/bin/env bash
#SBATCH --job-name=verify-cache
#SBATCH --output=job-outputs/slurm-verify-cache.%j.out
#SBATCH --error=job-outputs/slurm-verify-cache.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Verify Cached Pretrained Outputs
# ============================================================================
#
# USAGE:
#   # Verify test cache (default — quick, small):
#   sbatch train_full_model/slurms/slurm_verify_cache.sh
#
#   # Verify rest cache:
#   CACHE_DIR=train_full_model/data/cache_test \
#     sbatch train_full_model/slurms/slurm_verify_cache.sh
#
#   # Limit to first 5 shards:
#   MAX_SHARDS=5 sbatch train_full_model/slurms/slurm_verify_cache.sh
#
#   # With round-trip MLP check:
#   RGB_CKPT=pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
#     sbatch train_full_model/slurms/slurm_verify_cache.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

CACHE_DIR="${CACHE_DIR:-train_full_model/data/cache_train}"
MAX_SHARDS="${MAX_SHARDS:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5000}"
RGB_CKPT="${RGB_TO_STRUCT_CKPT:-/ihome/ohinder/ajk245/Desktop/Link to ajk245/Github/CHROMA-lite/pretrain_rgb_to_structure/data/checkpoints/mlp_d1024_L8_do0.1_lr0.000686_bs512_ep200/latest}"

echo "============================================================================"
echo "VERIFY CACHE - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "Cache dir:    ${CACHE_DIR}"
echo "Max shards:   ${MAX_SHARDS:-all}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "RGB MLP ckpt: ${RGB_CKPT:-none (round-trip check skipped)}"
echo "Started:      $(date)"
echo "============================================================================"
echo

ARGS=(
    python train_full_model/scripts/verify_cache.py
    --cache-dir "${CACHE_DIR}"
    --max-examples "${MAX_EXAMPLES}"
    --verbose
)

[ -n "${MAX_SHARDS}" ] && ARGS+=(--max-shards "${MAX_SHARDS}")
[ -n "${RGB_CKPT}" ] && ARGS+=(--rgb-to-structure-checkpoint "${RGB_CKPT}")

ARGS+=("$@")

echo "Command: ${ARGS[*]}"
echo

"${ARGS[@]}"
EXIT_CODE=$?

echo
echo "Done: $(date) — exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}