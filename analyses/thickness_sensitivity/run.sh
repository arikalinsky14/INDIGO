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
# See analyses/thickness_sensitivity/thickness_sensitivity.py for the
# full method and analyses/thickness_sensitivity/README.md for the
# study rationale (why synthetic materials, why the outermost-air-side
# layer, etc.).
#
# Env knobs (all optional):
#   OUTPUT_DIR                    default: analyses/thickness_sensitivity/results
#   N_STRUCTURES                  default: 60 (per source; total = 2N)
#   SWEEP_MAX_NM                  default: 10 (±10 nm around base t)
#   SWEEP_STEP_NM                 default: 1  (integer nm resolution)
#   HIGH_CHROMA_CANDIDATE_COUNT   default: 24
#   HIGH_CHROMA_REFINE_ITERS      default: 12
#   SEED                          default: 0
#
# Example:
#   sbatch analyses/thickness_sensitivity/run.sh
#   # Large hand-off experiment: 500 structures per source, ±15 nm at
#   # 0.5 nm resolution.
#   N_STRUCTURES=500 SWEEP_MAX_NM=15 SWEEP_STEP_NM=0.5 \
#       OUTPUT_DIR=analyses/thickness_sensitivity/results_large \
#       sbatch analyses/thickness_sensitivity/run.sh
# ============================================================================

: "${OUTPUT_DIR:=analyses/thickness_sensitivity/results}"
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

python analyses/thickness_sensitivity/thickness_sensitivity.py \
    --output-dir "$OUTPUT_DIR" \
    --n-structures "$N_STRUCTURES" \
    --sweep-max-nm "$SWEEP_MAX_NM" \
    --sweep-step-nm "$SWEEP_STEP_NM" \
    --high-chroma-candidate-count "$HIGH_CHROMA_CANDIDATE_COUNT" \
    --high-chroma-refine-iters "$HIGH_CHROMA_REFINE_ITERS" \
    --seed "$SEED"

echo
echo "Artefacts (see README.md for the full layout):"
echo "  $OUTPUT_DIR/sensitivity.json           raw sweep data + aggregates (JSON)"
echo "  $OUTPUT_DIR/per_layer.csv              one row per swept layer (summary stats)"
echo "  $OUTPUT_DIR/sweeps_long.csv            one row per (layer, Δnm) probe point"
echo "  $OUTPUT_DIR/curves_examples.png        per-layer ΔE(Δnm) sample"
echo "  $OUTPUT_DIR/sensitivity_by_bin.png     |dΔE/dnm| by base thickness"
echo "  $OUTPUT_DIR/delta_e_2_by_bin.png       Δnm needed for ΔE=2"
echo "  $OUTPUT_DIR/delta_e_3_by_bin.png       Δnm needed for ΔE=3"
echo "  $OUTPUT_DIR/grid_comparison.png        snap-cost per candidate grid (all sources)"
echo "  $OUTPUT_DIR/grid_comparison_by_source.png  same, HC vs random"
echo "  $OUTPUT_DIR/grid_comparison.txt        printable tables"
