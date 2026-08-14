#!/usr/bin/env bash
#SBATCH --job-name=indigo-verify-2h
#SBATCH --output=job-outputs/indigo-verify-2h.%j.out
#SBATCH --error=job-outputs/indigo-verify-2h.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Data-Generation Verifier — MUST pass before generating 10M rows.
# ============================================================================
#
# What it does (see scripts/verify_two_head.py):
#   1. Generates one small end-to-end shard (default 200 rows) with the same
#      pipeline you'll run at scale, including the high-chroma-search path.
#   2. Kramers-Kronig residual check on every material, grouped by source.
#      Synthetic sources PASS if their p95 residual is within
#      KK_SLACK_FACTOR × the real-material p95 (real JLL materials are the
#      noise floor — they're causal by measurement, so any residual is
#      the band-limitation artefact from restricting to the visible band).
#   3. Plots n,k curves across generation strategies to a PNG grid.
#   4. Measures wall-clock per row for random / search paths, then projects
#      the 10M-row full run at PARALLEL_WORKERS SLURM concurrency.
#
# Env knobs (all optional):
#   OUTPUT_DIR                     default: data/verify_2h
#   N_ROWS                         default: 200
#   HIGH_CHROMA_PROB               default: 0.2
#   HIGH_CHROMA_CANDIDATE_COUNT    default: 24 (drop to 12 to halve search cost)
#   HIGH_CHROMA_REFINE_ITERS       default: 12 (drop to 6 to halve search cost)
#   PARALLEL_WORKERS               default: 32 (for ETA extrapolation)
#   KK_SLACK_FACTOR                default: 2.5
#
# Example:
#   sbatch slurms/verify_two_head.sh
#   # trim search cost for a faster projection
#   HIGH_CHROMA_CANDIDATE_COUNT=12 HIGH_CHROMA_REFINE_ITERS=6 \
#       sbatch slurms/verify_two_head.sh
# ============================================================================

: "${OUTPUT_DIR:=data/verify_2h}"
: "${N_ROWS:=200}"
: "${HIGH_CHROMA_PROB:=0.2}"
: "${HIGH_CHROMA_CANDIDATE_COUNT:=24}"
: "${HIGH_CHROMA_REFINE_ITERS:=12}"
: "${PARALLEL_WORKERS:=32}"
: "${KK_SLACK_FACTOR:=2.5}"

echo "======================================================================"
echo " INDIGO two-head transition verifier"
echo " OUTPUT_DIR                  : $OUTPUT_DIR"
echo " N_ROWS                      : $N_ROWS"
echo " HIGH_CHROMA_PROB            : $HIGH_CHROMA_PROB"
echo " HIGH_CHROMA_CANDIDATE_COUNT : $HIGH_CHROMA_CANDIDATE_COUNT"
echo " HIGH_CHROMA_REFINE_ITERS    : $HIGH_CHROMA_REFINE_ITERS"
echo " PARALLEL_WORKERS            : $PARALLEL_WORKERS   (for ETA)"
echo " KK_SLACK_FACTOR             : $KK_SLACK_FACTOR"
echo "======================================================================"

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"

# Force JAX to CPU — the search path uses jax.grad but the smp queue's
# mock GPU would just slow it down.
export JAX_PLATFORMS=cpu

mkdir -p "$(dirname "$OUTPUT_DIR")"

python scripts/verify_two_head.py \
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
