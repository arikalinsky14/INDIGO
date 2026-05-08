#!/usr/bin/env bash
#SBATCH --job-name=chroma-cache
#SBATCH --output=job-outputs/slurm-cache.%j.out
#SBATCH --error=job-outputs/slurm-cache.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=48:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Cache Pretrained Pipeline Outputs (Two-Phase)
# ============================================================================
#
# Phase 1: Generate all RGB predictions (GPU-bound LLM with KV-cache)
# Phase 2: Compute structure logits (tiny MLP, very fast)
#
# USAGE:
#   # Both phases (default):
#   sbatch train_full_model/slurms/slurm_cache.sh
#
#   # Phase 1 only (RGB prediction):
#   PHASE=1 sbatch train_full_model/slurms/slurm_cache.sh
#
#   # Phase 2 only (structure logits, after Phase 1 completes):
#   PHASE=2 sbatch train_full_model/slurms/slurm_cache.sh
#
#   # Quick test:
#   LIMIT_EXAMPLES=1000 SHARD_SIZE=500 \
#     sbatch train_full_model/slurms/slurm_cache.sh
#
#   # Larger text batch (if GPU memory allows):
#   TEXT_BATCH_SIZE=128 sbatch train_full_model/slurms/slurm_cache.sh
#
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CHROMA-LITE CACHE PRETRAINED - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

TEXT_TO_RGB_CKPT="${TEXT_TO_RGB_CKPT:-/ihome/ohinder/ajk245/Desktop/Link to ajk245/Github/CHROMA-lite/pretrain_text_to_rgb/data/checkpoints/constrained_TinyLlama-TinyLlama-1.1B-Chat-v1.0_lora_lr0.0007_bs4_ep1_wd0.01_gc1.0_wu0.03_r16_a32_q_proj-v_proj_lim2000000/latest}"
RGB_TO_STRUCT_CKPT="${RGB_TO_STRUCT_CKPT:-/ihome/ohinder/ajk245/Desktop/Link to ajk245/Github/CHROMA-lite/pretrain_rgb_to_structure/data/checkpoints/mlp_d1024_L8_do0.1_lr0.000686_bs512_ep200/latest}"

DATA_DIR="${DATA_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
SPLIT="${SPLIT:-train}"
SEED="${SEED:-42}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"
SHARD_SIZE="${SHARD_SIZE:-10000}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}"
PHASE="${PHASE:-0}"  # 0=both, 1=RGB only, 2=structure only

# ============================================================================
# BUILD AND RUN COMMAND
# ============================================================================

ARGS=()
ARGS+=(python train_full_model/scripts/cache_pretrained.py)
ARGS+=(--text-to-rgb-checkpoint "${TEXT_TO_RGB_CKPT}")
ARGS+=(--rgb-to-structure-checkpoint "${RGB_TO_STRUCT_CKPT}")
ARGS+=(--split "${SPLIT}")
ARGS+=(--seed "${SEED}")
ARGS+=(--shard-size "${SHARD_SIZE}")
ARGS+=(--text-batch-size "${TEXT_BATCH_SIZE}")
ARGS+=(--phase "${PHASE}")

[ -n "${DATA_DIR}" ] && ARGS+=(--data-dir "${DATA_DIR}")
[ -n "${CACHE_DIR}" ] && ARGS+=(--cache-dir "${CACHE_DIR}")
[ -n "${LIMIT_EXAMPLES}" ] && ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")

ARGS+=("$@")

echo "Command: ${ARGS[*]}"
echo "============================================================================"
echo

"${ARGS[@]}"
EXIT_CODE=$?

echo
echo "============================================================================"
echo "COMPLETE - $(date) — exit code: ${EXIT_CODE}"
echo "============================================================================"

exit ${EXIT_CODE}