#!/usr/bin/env bash
#SBATCH --job-name=indigo-thickness-sens
#SBATCH --output=job-outputs/indigo-thickness-sens.%j.out
#SBATCH --error=job-outputs/indigo-thickness-sens.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=03:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Thickness Sensitivity Study on High-Chroma Structures
# ============================================================================
#
# What: generate a batch of high-chroma structures via the existing directed-
# search path, then sweep each layer's thickness ± SWEEP_MAX_NM in
# SWEEP_STEP_NM increments and record ΔE_00 vs the base achieved Lab.
# Produces per-layer sensitivity curves, aggregate |dΔE/dnm| by base-thickness
# bin, a grid-comparison table, and 4 plots.
#
# See scripts/thickness_sensitivity.py for the full method.
#
# Env knobs (all optional):
#   OUTPUT_DIR                    default: data/thickness_sensitivity
#   N_STRUCTURES                  default: 30
#   SWEEP_MAX_NM                  default: 10 (±10 nm around base t)
#   SWEEP_STEP_NM                 default: 1  (integer nm resolution)
#   HIGH_CHROMA_CANDIDATE_COUNT   default: 24
#   HIGH_CHROMA_REFINE_ITERS      default: 12
#   SEED                          default: 0
#
# Example:
#   sbatch slurms/thickness_sensitivity.sh
#   N_STRUCTURES=100 SWEEP_MAX_NM=20 SWEEP_STEP_NM=0.5 \
#       sbatch slurms/thickness_sensitivity.sh
# ============================================================================

: "${OUTPUT_DIR:=data/thickness_sensitivity}"
: "${N_STRUCTURES:=60}"       # per source (high_chroma + random); doubles total
: "${SWEEP_MAX_NM:=10}"
: "${SWEEP_STEP_NM:=1}"
: "${HIGH_CHROMA_CANDIDATE_COUNT:=24}"
: "${HIGH_CHROMA_REFINE_ITERS:=12}"
: "${SEED:=0}"

echo "======================================================================"
echo " INDIGO thickness sensitivity study"
echo " OUTPUT_DIR                  : $OUTPUT_DIR"
echo " N_STRUCTURES                : $N_STRUCTURES"
echo " SWEEP_MAX_NM                : $SWEEP_MAX_NM"
echo " SWEEP_STEP_NM               : $SWEEP_STEP_NM"
echo " HIGH_CHROMA_CANDIDATE_COUNT : $HIGH_CHROMA_CANDIDATE_COUNT"
echo " HIGH_CHROMA_REFINE_ITERS    : $HIGH_CHROMA_REFINE_ITERS"
echo " SEED                        : $SEED"
echo "======================================================================"

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"

# Force JAX to CPU on smp queue.
#
# JAX_PLATFORMS alone isn't enough: the PyTorch module pulls in CUDA
# libraries, JAX detects CUDA at import time and tries to initialise its
# xla_cuda12 plugin BEFORE checking JAX_PLATFORMS, then fails with
# "operation cuInit(0) failed: Unknown CUDA error 303" because there's
# no driver on smp. Hiding all devices with CUDA_VISIBLE_DEVICES="" makes
# JAX skip the plugin init entirely and go straight to CPU.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

mkdir -p "$(dirname "$OUTPUT_DIR")"

python scripts/thickness_sensitivity.py \
    --output-dir "$OUTPUT_DIR" \
    --n-structures "$N_STRUCTURES" \
    --sweep-max-nm "$SWEEP_MAX_NM" \
    --sweep-step-nm "$SWEEP_STEP_NM" \
    --high-chroma-candidate-count "$HIGH_CHROMA_CANDIDATE_COUNT" \
    --high-chroma-refine-iters "$HIGH_CHROMA_REFINE_ITERS" \
    --seed "$SEED"

echo
echo "Artefacts:"
echo "  $OUTPUT_DIR/sensitivity.json         raw sweep data + aggregates"
echo "  $OUTPUT_DIR/curves_examples.png      per-layer ΔE(Δnm) sample"
echo "  $OUTPUT_DIR/sensitivity_by_bin.png   |dΔE/dnm| by base thickness"
echo "  $OUTPUT_DIR/delta_e_1_by_bin.png     Δnm needed for ΔE=1"
echo "  $OUTPUT_DIR/grid_comparison.png      snap-cost per candidate grid"
echo "  $OUTPUT_DIR/grid_comparison.txt      same table printed to stdout"
