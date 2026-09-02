#!/usr/bin/env bash
#SBATCH --job-name=indigo-thickness-sens
#SBATCH --output=job-outputs/indigo-thickness-sens.%j.out
#SBATCH --error=job-outputs/indigo-thickness-sens.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# 12 h reservation covers the large hand-off run (N=2000 per source).
# Under adaptive probing (default), a full 4000-structure sweep runs in
# roughly 40-90 min on 4 CPUs, so this reservation has generous slack.
# The HC directed search dominates cost (~5-10 s per structure at
# candidate_count=24, refine_iters=12) → ~5 h for 2000 HC structures.
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Thickness Sensitivity Study on High-Chroma Structures
# ============================================================================
#
# What: for each structure, pick the layer whose thickness the achieved
# colour is MOST sensitive to (via a ±SLOPE_PROBE_NM finite-difference
# slope estimate across all layers), then run an ADAPTIVE probe on that
# layer:
#
#   doubling schedule 0.5 → 1 → 2 → 4 → 8 → 16 → 32 → 64 → CAP_NM,
#   stopping as soon as ΔE ≥ 5 (the highest reported threshold),
#   then bisecting the bracket to BISECT_TOL_NM precision for each
#   threshold (ΔE = 2, 3, 5) independently.
#
# Structures where NO threshold is reached within ±CAP_NM are kept and
# marked right-censored (cens_de*_* = 1 in per_layer.csv). Nothing is
# dropped for being "opaque" any more — a fully-opaque metal layer just
# shows up with near-zero local slope and all crossings pinned at CAP_NM.
#
# See analyses/thickness_sensitivity/thickness_sensitivity.py for the
# full method and analyses/thickness_sensitivity/README.md for the
# study rationale.
#
# Env knobs (all optional):
#   OUTPUT_DIR                    default: analyses/thickness_sensitivity/results
#   N_STRUCTURES                  default: 60 (per source; total = 2N)
#   CAP_NM                        default: 128 (right-censoring boundary)
#   SLOPE_PROBE_NM                default: 0.5 (finite-diff step for layer pick)
#   BISECT_TOL_NM                 default: 0.05 (Δnm tolerance for crossings)
#   N_JOBS                        default: 4  (matches --cpus-per-task)
#   HIGH_CHROMA_CANDIDATE_COUNT   default: 24
#   HIGH_CHROMA_REFINE_ITERS      default: 12
#   SEED                          default: 0
#
# Examples:
#   # Quick sanity run (~5 min on 4 CPUs, ~2 min per source):
#   sbatch analyses/thickness_sensitivity/run.sh
#
#   # Large hand-off experiment (~1 h on 4 CPUs, 12 h reservation):
#   N_STRUCTURES=2000 \
#       OUTPUT_DIR=analyses/thickness_sensitivity/results_large \
#       sbatch analyses/thickness_sensitivity/run.sh
#
#   # Fine-granularity experiment: tighter bisection tolerance for
#   # more precise crossings on very-sensitive layers.
#   N_STRUCTURES=2000 BISECT_TOL_NM=0.02 SLOPE_PROBE_NM=0.25 \
#       OUTPUT_DIR=analyses/thickness_sensitivity/results_fine \
#       sbatch analyses/thickness_sensitivity/run.sh
# ============================================================================

: "${OUTPUT_DIR:=analyses/thickness_sensitivity/results}"
: "${N_STRUCTURES:=60}"
: "${CAP_NM:=128}"
: "${SLOPE_PROBE_NM:=0.5}"
: "${BISECT_TOL_NM:=0.05}"
: "${N_JOBS:=4}"
: "${HIGH_CHROMA_CANDIDATE_COUNT:=24}"
: "${HIGH_CHROMA_REFINE_ITERS:=12}"
: "${SEED:=0}"

echo "======================================================================"
echo " INDIGO thickness sensitivity study (adaptive probe)"
echo " OUTPUT_DIR                  : $OUTPUT_DIR"
echo " N_STRUCTURES                : $N_STRUCTURES"
echo " CAP_NM                      : $CAP_NM"
echo " SLOPE_PROBE_NM              : $SLOPE_PROBE_NM"
echo " BISECT_TOL_NM               : $BISECT_TOL_NM"
echo " N_JOBS                      : $N_JOBS"
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
    --cap-nm "$CAP_NM" \
    --slope-probe-nm "$SLOPE_PROBE_NM" \
    --bisect-tol-nm "$BISECT_TOL_NM" \
    --n-jobs "$N_JOBS" \
    --high-chroma-candidate-count "$HIGH_CHROMA_CANDIDATE_COUNT" \
    --high-chroma-refine-iters "$HIGH_CHROMA_REFINE_ITERS" \
    --seed "$SEED"

echo
echo "Artefacts (see README.md for the full layout):"
echo "  $OUTPUT_DIR/sensitivity.json           raw probes + crossings + aggregates"
echo "  $OUTPUT_DIR/per_layer.csv              one row per chosen layer"
echo "  $OUTPUT_DIR/sweeps_long.csv            one row per (structure, probe) point"
echo "  $OUTPUT_DIR/curves_examples.png        adaptive ΔE(Δnm) sample"
echo "  $OUTPUT_DIR/sensitivity_by_bin.png     |dΔE/dnm| by base thickness"
echo "  $OUTPUT_DIR/delta_e_2_by_bin.png       Δnm needed for ΔE=2  (+% censored)"
echo "  $OUTPUT_DIR/delta_e_3_by_bin.png       Δnm needed for ΔE=3  (+% censored)"
echo "  $OUTPUT_DIR/grid_comparison.png        snap-cost per candidate grid"
echo "  $OUTPUT_DIR/grid_comparison_by_source.png  same, HC vs random"
echo "  $OUTPUT_DIR/grid_comparison.txt        printable tables"
