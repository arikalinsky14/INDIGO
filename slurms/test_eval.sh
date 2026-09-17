#!/usr/bin/env bash
#SBATCH --job-name=indigo-test-eval
#SBATCH --output=job-outputs/indigo-test-eval.%j.out
#SBATCH --error=job-outputs/indigo-test-eval.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=12:00:00
# NOTE: do not add `--qos=short` here — it caps wall time at 3h on Pitt
# CRC regardless of the --time value, so long-running evals (e.g.
# ensemble N=200 on 500 rows × 2 tiers ≈ 9h) will be cancelled at 3h.
# Omitting --qos lets SLURM use the account default, which respects
# the --time value above.
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# INDIGO test-set sweep evaluation
# ============================================================================
#
# Runs `inference/scripts/test_eval.py` against data/test/tier_a and
# data/test/tier_b: per row, extract Lab + per-row pool, run solve(),
# record ΔE_00, render PNGs for the first PLOT_EXAMPLES rows.
#
# Output (under inference/outputs/test_eval/<checkpoint_tag>/):
#   tier_a/
#     summary.json    aggregate ΔE distribution + buckets + runtime
#     rows.json       per-row outcomes (compact dicts)
#     examples/       row_NNNN.json + row_NNNN.png for the first N rows
#   tier_b/
#     ... same shape ...
#
# Time budget (defaults LIMIT=500, ENSEMBLE_N=200, TOLERANCE=0):
#   - Per-row solve: ~0.5-1.5 s (sequential JLL physics over ensemble)
#   - 500 rows × 2 tiers ≈ 15-30 min. The 03:00:00 wall is safety
#     padding for cold module loads and bigger LIMIT runs.
# ============================================================================

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
# Keep JAX off the GPU — the physics chain isn't GPU-amenable here, and
# leaving JAX on CPU saves the GPU memory for the torch model forward.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs inference/outputs/test_eval

echo "============================================================================"
echo "INDIGO TEST EVAL - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# ============================================================================
# CONFIGURATION (env-var overrideable)
# ============================================================================

# REQUIRED — checkpoint to evaluate.
CHECKPOINT="${CHECKPOINT:-}"

# Test data + output roots.
TEST_DIR="${TEST_DIR:-data/test}"
OUTPUT_DIR="${OUTPUT_DIR:-inference/outputs/test_eval}"

# Tier selection.
TIERS="${TIERS:-a b}"

# Per-tier knobs.
LIMIT="${LIMIT:-500}"
PLOT_EXAMPLES="${PLOT_EXAMPLES:-20}"
ENSEMBLE_N="${ENSEMBLE_N:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOLERANCE="${TOLERANCE:-0.0}"
WEIGHT_LAMBDA="${WEIGHT_LAMBDA:-1.0}"
TOP_K="${TOP_K:-1}"
REFINE_ITERS="${REFINE_ITERS:-80}"
MC_SAMPLES="${MC_SAMPLES:-0}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-25}"

# Pass-through positional args.
USER_ARGS=("$@")

# ============================================================================
# VALIDATION
# ============================================================================

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT env var is required." >&2
  echo "        Example: CHECKPOINT=data/checkpoints/<tag>/latest" >&2
  exit 2
fi

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --test-dir "${TEST_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --tiers ${TIERS}                 # intentional word-splitting
  --limit "${LIMIT}"
  --plot-examples "${PLOT_EXAMPLES}"
  --ensemble-n "${ENSEMBLE_N}"
  --temperature "${TEMPERATURE}"
  --tolerance "${TOLERANCE}"
  --lambda "${WEIGHT_LAMBDA}"
  --top-k "${TOP_K}"
  --refine-iters "${REFINE_ITERS}"
  --mc-samples "${MC_SAMPLES}"
  --seed "${SEED}"
  --log-every "${LOG_EVERY}"
)

CMD=(python inference/scripts/test_eval.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo "============================================================================"
echo "EVAL CONFIGURATION"
echo "============================================================================"
echo
echo "Model:"
echo "  Checkpoint:        ${CHECKPOINT}"
echo
echo "Data:"
echo "  Test dir:          ${TEST_DIR}"
echo "  Tiers:             ${TIERS}"
echo "  Limit per tier:    ${LIMIT}"
echo
echo "Knobs (sweep defaults — see test_eval.py docstring):"
echo "  ensemble_N:        ${ENSEMBLE_N}"
echo "  temperature:       ${TEMPERATURE}"
echo "  tolerance %:       ${TOLERANCE}"
echo "  weight λ:          ${WEIGHT_LAMBDA}"
echo "  top_k:             ${TOP_K}"
echo "  refine_iters:      ${REFINE_ITERS}"
echo "  mc_samples:        ${MC_SAMPLES}"
echo "  seed:              ${SEED}"
echo
echo "Output:"
echo "  Root:              ${OUTPUT_DIR}"
echo "  PNGs per tier:     ${PLOT_EXAMPLES}"
echo
echo "============================================================================"
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

# ============================================================================
# RUN
# ============================================================================

"${CMD[@]}"
EXIT_CODE=$?

echo
echo "============================================================================"
echo "TEST EVAL COMPLETE  exit=${EXIT_CODE}  ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE
# ============================================================================
#
# Minimal:
#   CHECKPOINT=data/checkpoints/<tag>/latest sbatch slurms/test_eval.sh
#
# Larger sweep, only tier_a, render 50 PNGs:
#   CHECKPOINT=<ckpt> TIERS="a" LIMIT=1000 PLOT_EXAMPLES=50 \
#     sbatch slurms/test_eval.sh
#
# Include robustness scoring (slower):
#   CHECKPOINT=<ckpt> TOLERANCE=5.0 MC_SAMPLES=16 \
#     sbatch slurms/test_eval.sh
#
# Custom output dir per experiment:
#   CHECKPOINT=<ckpt> OUTPUT_DIR=inference/outputs/test_eval_exp17 \
#     sbatch slurms/test_eval.sh
# ============================================================================
