#!/usr/bin/env bash
#SBATCH --job-name=chroma-bench-l40s
#SBATCH --output=job-outputs/slurm-bench-l40s.%j.out
#SBATCH --error=job-outputs/slurm-bench-l40s.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=04:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Throughput Benchmark — L40S (48 GB GDDR6, Ada Lovelace)
# ============================================================================
#
# Benchmarks training throughput for the scaled-up CHROMA-Lite full model
# (Llama 3.1 8B + 10-layer MixingMLP) with synthetic data.
#
# USAGE:
#   # Default (all standard batch sizes):
#   sbatch train_full_model/slurms/slurm_bench_l40s.sh
#
#   # Custom batch sizes:
#   BATCH_SIZES=1,2,4,8 sbatch train_full_model/slurms/slurm_bench_l40s.sh
#
#   # Quick test:
#   WARMUP_STEPS=1 MEASURE_STEPS=3 \
#     sbatch train_full_model/slurms/slurm_bench_l40s.sh
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "CHROMA-LITE THROUGHPUT BENCH (L40S) - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "[WARN] transformers not available"
python -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null || echo "[WARN] peft not available"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

GPU_TYPE="l40s"
ENCODER="${ENCODER:-meta-llama/Llama-3.1-8B-Instruct}"
BATCH_SIZES="${BATCH_SIZES:-}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ============================================================================
# BUILD AND RUN COMMAND
# ============================================================================

ARGS=()
ARGS+=(python train_full_model/scripts/throughput_bench.py)
ARGS+=(--gpu-type "${GPU_TYPE}")
ARGS+=(--encoder "${ENCODER}")
ARGS+=(--warmup-steps "${WARMUP_STEPS}")
ARGS+=(--measure-steps "${MEASURE_STEPS}")

[ -n "${BATCH_SIZES}" ] && ARGS+=(--batch-sizes "${BATCH_SIZES}")
[ -n "${OUTPUT_DIR}" ] && ARGS+=(--output-dir "${OUTPUT_DIR}")

# Append any extra args from sbatch command line
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
