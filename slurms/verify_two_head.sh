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
# Two-Head Transition Verifier — MUST pass before generating 10M rows.
# ============================================================================
#
# What it does (see scripts/verify_two_head.py):
#   1. Generates one small end-to-end shard (default 200 rows) with the same
#      pipeline you'll run at scale, including the high-chroma-search path.
#   2. Independently checks every material for Kramers-Kronig consistency
#      via a Fourier-domain Hilbert transform residual. Perturb-real and
#      interpolate-real materials are constructed with an additive causal
#      correction; all should pass.
#   3. Plots n,k curves across generation strategies to a PNG grid.
#   4. Measures wall-clock per row for random / search paths, then projects
#      the 10M-row full run at PARALLEL_WORKERS SLURM concurrency.
#
# Env knobs (all optional):
#   OUTPUT_DIR          default: data/verify_2h
#   N_ROWS              default: 200
#   HIGH_CHROMA_PROB    default: 0.2
#   PARALLEL_WORKERS    default: 32 (matches the intended production slurm)
#   KK_TOLERANCE        default: 0.30
#
# Example:
#   sbatch slurms/verify_two_head.sh
#   N_ROWS=500 HIGH_CHROMA_PROB=0.3 sbatch slurms/verify_two_head.sh
# ============================================================================

: "${OUTPUT_DIR:=data/verify_2h}"
: "${N_ROWS:=200}"
: "${HIGH_CHROMA_PROB:=0.2}"
: "${PARALLEL_WORKERS:=32}"
: "${KK_TOLERANCE:=0.30}"

echo "======================================================================"
echo " INDIGO two-head transition verifier"
echo " OUTPUT_DIR       : $OUTPUT_DIR"
echo " N_ROWS           : $N_ROWS"
echo " HIGH_CHROMA_PROB : $HIGH_CHROMA_PROB"
echo " PARALLEL_WORKERS : $PARALLEL_WORKERS   (for ETA extrapolation)"
echo " KK_TOLERANCE     : $KK_TOLERANCE"
echo "======================================================================"

module load anaconda/2023.09-2 || true
source activate indigo || true

# Force JAX to CPU — the search path uses jax.grad but the mock GPU on the
# smp queue would just slow it down.
export JAX_PLATFORMS=cpu

mkdir -p "$(dirname "$OUTPUT_DIR")"

python scripts/verify_two_head.py \
    --output-dir "$OUTPUT_DIR" \
    --n-rows "$N_ROWS" \
    --high-chroma-prob "$HIGH_CHROMA_PROB" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --kk-tolerance "$KK_TOLERANCE"

echo
echo "Artefacts:"
echo "  $OUTPUT_DIR/dryrun.parquet"
echo "  $OUTPUT_DIR/dryrun.manifest.json"
echo "  $OUTPUT_DIR/curves.png"
echo "  $OUTPUT_DIR/kk_report.json"
echo "  $OUTPUT_DIR/timing_report.json"
