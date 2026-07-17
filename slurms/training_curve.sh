#!/usr/bin/env bash
#SBATCH --job-name=indigo-training-curve
#SBATCH --output=job-outputs/indigo-training-curve.%j.out
#SBATCH --error=job-outputs/indigo-training-curve.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# INDIGO training-curve sweep — teacher-forcing loss + token accuracy at
# each saved checkpoint, on four datasets in one pass:
#
#   train    — subset of the train shards      (--train-examples)
#   val      — the 0.5% holdout of the train shards
#   test_a   — data/test/tier_a   (seen materials, novel structures)
#   test_b   — data/test/tier_b   (unseen materials, novel structures)
#
# The tier_a vs tier_b gap is the material-generalisation number — the
# reason the shared MaterialEncoder + slot-permutation invariance exist.
#
# Output: outputs/training_curves_<CHECKPOINT_TAG>.json plus a matching
# .png (2 panels: loss and accuracy) with one series per dataset.
#
# Speed model (post pre-collate + meta-json train-loss optimisation):
#   • One-time cost:  pre-collate val + tier_a + tier_b (~30-60 s total).
#   • Per checkpoint: forward-only inference over the cached batches
#     (~5-15 s at 5 k rows/dataset on an L40s). Train loss free (read
#     from meta.json).
#   • Default sweep (35 checkpoints): ~5-10 min total wall.
# The prior version's timeout came from re-reading the parquet shards
# AND re-spawning DataLoader workers 140 times (35 ckpts × 4 datasets).
# Both are now one-time costs. 24 h wall is padding for pathological
# NFS latency; a healthy run finishes in under an hour.
#
# Robust to timeouts: results are written incrementally after every
# checkpoint, and --resume (default on) skips already-completed steps.
# Re-sbatch the same job and it picks up where it left off.
# ============================================================================

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
# JAX not used by this script; safe default.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs outputs

echo "============================================================================"
echo "INDIGO TRAINING CURVE - Job ${SLURM_JOB_ID:-local}"
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

# REQUIRED — parent directory of step_<N> checkpoints, e.g.
#   data/checkpoints/flex_raw_spectrum_..._cross_attnH8_se4_dec1
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# Data roots.
DATA_DIR="${DATA_DIR:-data/train}"
TEST_A_DIR="${TEST_A_DIR:-data/test/tier_a}"
TEST_B_DIR="${TEST_B_DIR:-data/test/tier_b}"

# Checkpoint sweep bounds.
START_STEP="${START_STEP:-1000}"
END_STEP="${END_STEP:-174000}"
STEP_INTERVAL="${STEP_INTERVAL:-5000}"

# Per-dataset row caps.
EVAL_EXAMPLES="${EVAL_EXAMPLES:-5000}"       # val
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-30000}"    # train subset (fresh sample per ckpt)
TEST_EXAMPLES="${TEST_EXAMPLES:-5000}"       # each of tier_a / tier_b

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-15}"
SEED="${SEED:-42}"
STREAMING="${STREAMING:-1}"                  # 1 = shard-by-shard (recommended)

# Where to get the training loss. 'meta' (default) reads the running
# training loss from each checkpoint's meta.json — no re-eval, saves
# ~75%% of the sweep. 'eval' re-scores a train subset the same way val
# is scored (slower, but on identical scale).
TRAIN_LOSS_SOURCE="${TRAIN_LOSS_SOURCE:-meta}"

# Output path. Default: outputs/training_curves_<tag>.json.
OUTPUT="${OUTPUT:-}"

# Pass-through positional args.
USER_ARGS=("$@")

# ============================================================================
# VALIDATION
# ============================================================================

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] CHECKPOINT_DIR env var is required." >&2
  echo "        Example: CHECKPOINT_DIR=data/checkpoints/<tag>" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] CHECKPOINT_DIR '${CHECKPOINT_DIR}' does not exist." >&2
  exit 2
fi

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --data-dir "${DATA_DIR}"
  --start-step "${START_STEP}"
  --end-step "${END_STEP}"
  --step-interval "${STEP_INTERVAL}"
  --eval-examples "${EVAL_EXAMPLES}"
  --train-examples "${TRAIN_EXAMPLES}"
  --test-examples "${TEST_EXAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --smoothing-window "${SMOOTHING_WINDOW}"
  --seed "${SEED}"
  --train-loss-source "${TRAIN_LOSS_SOURCE}"
  --plot
)
# Include tier flags only if the directories actually exist — the eval
# script gracefully skips a missing tier, but this keeps the CLI clean.
if [[ -d "${TEST_A_DIR}" ]]; then
  ARGS+=(--test-a-dir "${TEST_A_DIR}")
else
  echo "[WARN] TEST_A_DIR '${TEST_A_DIR}' missing — skipping test_a curve."
fi
if [[ -d "${TEST_B_DIR}" ]]; then
  ARGS+=(--test-b-dir "${TEST_B_DIR}")
else
  echo "[WARN] TEST_B_DIR '${TEST_B_DIR}' missing — skipping test_b curve."
fi
if [[ "${STREAMING}" == "1" ]]; then
  ARGS+=(--streaming)
else
  ARGS+=(--no-streaming)
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

CMD=(python scripts/plot_training_curves.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo "============================================================================"
echo "TRAINING CURVE CONFIGURATION"
echo "============================================================================"
echo
echo "Checkpoints:"
echo "  Dir:            ${CHECKPOINT_DIR}"
echo "  Sweep:          step ${START_STEP} → ${END_STEP} every ${STEP_INTERVAL}"
echo
echo "Datasets:"
echo "  train shards:   ${DATA_DIR}   (${TRAIN_EXAMPLES} train / ${EVAL_EXAMPLES} val)"
echo "  test_a:         ${TEST_A_DIR}   (${TEST_EXAMPLES} rows if present)"
echo "  test_b:         ${TEST_B_DIR}   (${TEST_EXAMPLES} rows if present)"
echo
echo "Knobs:"
echo "  batch_size:     ${BATCH_SIZE}"
echo "  num_workers:    ${NUM_WORKERS}"
echo "  streaming:      ${STREAMING}"
echo "  seed:           ${SEED}"
echo
echo "Output:"
echo "  JSON+PNG:       ${OUTPUT:-outputs/training_curves_<checkpoint_tag>.[json|png]}"
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
echo "TRAINING CURVE COMPLETE  exit=${EXIT_CODE}  ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE
# ============================================================================
#
# Minimal (production checkpoint, default sweep + all four datasets):
#   CHECKPOINT_DIR=data/checkpoints/<tag> sbatch slurms/training_curve.sh
#
# Coarser sweep to burn less GPU time (every 10 k steps instead of 5 k):
#   CHECKPOINT_DIR=<dir> STEP_INTERVAL=10000 sbatch slurms/training_curve.sh
#
# Only up to step 100000 (early-stopping preview):
#   CHECKPOINT_DIR=<dir> END_STEP=100000 sbatch slurms/training_curve.sh
#
# Skip a tier (e.g. no test_b generated yet):
#   CHECKPOINT_DIR=<dir> TEST_B_DIR=/dev/null \
#     sbatch slurms/training_curve.sh
#
# LIGHT preset — fastest useful curve (~2-5 min wall on L40s):
#   CHECKPOINT_DIR=<dir> \
#     STEP_INTERVAL=10000 TEST_EXAMPLES=2000 EVAL_EXAMPLES=2000 \
#     sbatch slurms/training_curve.sh
#
# TRAIN-EVAL preset — re-scores a 30k-row train subset the same way
# val is scored (slower; use when train-vs-val on identical scale
# matters more than wall time):
#   CHECKPOINT_DIR=<dir> TRAIN_LOSS_SOURCE=eval \
#     sbatch slurms/training_curve.sh
#
# RESUME after timeout — just re-run the same command. Incremental
# saves persist every completed checkpoint; --resume (default on)
# picks up where the previous run left off:
#   CHECKPOINT_DIR=<dir> sbatch slurms/training_curve.sh
# ============================================================================
