#!/usr/bin/env bash
#SBATCH --job-name=indigo-inference
#SBATCH --output=job-outputs/indigo-inference.%j.out
#SBATCH --error=job-outputs/indigo-inference.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# INDIGO end-to-end inference
# ============================================================================
#
# Runs `inference/scripts/run_inference.py`:
#   pool (JLL by default) + target Lab + optional constraints  ──►
#   generate.py  ──►  select.py  ──►  refine.py  ──►  MC robustness  ──►
#   Result JSON in inference/outputs/
#
# Memory / time budget (cross_attn, N=500):
#   - Model forward (B=N once per decoding step): ~650 MB GPU.
#   - Physics chain (sequential per candidate, no jit/vmap): ~1–3 s/inference.
#   - MC robustness on top_k=5: ~0.5 s.
# Wall-clock is well under a minute on an L40s; the 01:00:00 SBATCH wall is
# only for safety against cold module loads / JIT compile / first-time JAX
# device probe.
# ============================================================================

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
# Force JAX onto CPU — the physics chain isn't gpu-amenable (jit/vmap blocked
# by the JLL assert; see inference/src/simulate.py). Saves GPU memory.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs inference/outputs

echo "============================================================================"
echo "INDIGO INFERENCE - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

# REQUIRED — the model to use.
CHECKPOINT="${CHECKPOINT:-}"

# REQUIRED — target color as three space-separated Lab values, e.g. "60 5 -8".
TARGET_LAB="${TARGET_LAB:-}"

# Pool: at most one of (POOL_JSON, POOL_DIR). If neither is set the CLI uses
# the installed jaxlayerlumos materials/ subdir (or $JLL_MATERIALS_DIR if
# you've exported one).
POOL_JSON="${POOL_JSON:-}"
POOL_DIR="${POOL_DIR:-}"

# Optional constraints JSON. See inference/scripts/run_inference.py docstring
# for the schema; one of:
#   {"kind":"allowed_subset", "allowed_names":[...]}, {"kind":"layer_count"...},
#   {"kind":"thickness_range"...}, ... etc.
CONSTRAINTS="${CONSTRAINTS:-}"

# Output path. Default: inference/outputs/result_seed<SEED>.json.
OUTPUT="${OUTPUT:-}"

# Knobs — see inference/src/schema.py:InferenceKnobs for the underlying
# defaults; these mirror them with env-var override surface.
ENSEMBLE_N="${ENSEMBLE_N:-500}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOLERANCE="${TOLERANCE:-5.0}"          # manufacturing tolerance %; 0 disables R
WEIGHT_LAMBDA="${WEIGHT_LAMBDA:-1.0}"   # J = ΔE + λ·R_l2
TOP_K="${TOP_K:-5}"
REFINE_ITERS="${REFINE_ITERS:-100}"
REFINE_STEP="${REFINE_STEP:-1.0}"       # nm; Adam initial step
MC_SAMPLES="${MC_SAMPLES:-32}"
SEED="${SEED:-42}"
RANDOM_RESTARTS="${RANDOM_RESTARTS:-0}"

# Anything you pass after the script name flows through to run_inference.py.
USER_ARGS=("$@")

# ============================================================================
# VALIDATION
# ============================================================================

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT env var is required." >&2
  echo "        Example: CHECKPOINT=data/checkpoints/flex_..._cross_attn.../latest" >&2
  exit 2
fi
if [[ -z "${TARGET_LAB}" ]]; then
  echo "[ERROR] TARGET_LAB env var is required." >&2
  echo "        Example: TARGET_LAB=\"60 5 -8\"" >&2
  exit 2
fi
if [[ -n "${POOL_JSON}" && -n "${POOL_DIR}" ]]; then
  echo "[ERROR] Set at most one of POOL_JSON and POOL_DIR." >&2
  exit 2
fi

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --target-lab ${TARGET_LAB}   # intentional word-splitting → three positional floats
  --ensemble-n "${ENSEMBLE_N}"
  --temperature "${TEMPERATURE}"
  --tolerance "${TOLERANCE}"
  --lambda "${WEIGHT_LAMBDA}"
  --top-k "${TOP_K}"
  --refine-iters "${REFINE_ITERS}"
  --refine-step "${REFINE_STEP}"
  --mc-samples "${MC_SAMPLES}"
  --seed "${SEED}"
  --random-restarts "${RANDOM_RESTARTS}"
)
if [[ -n "${POOL_JSON}" ]]; then
  ARGS+=(--pool "${POOL_JSON}")
fi
if [[ -n "${POOL_DIR}" ]]; then
  ARGS+=(--pool-dir "${POOL_DIR}")
fi
if [[ -n "${CONSTRAINTS}" ]]; then
  ARGS+=(--constraints "${CONSTRAINTS}")
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

CMD=(python inference/scripts/run_inference.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo "============================================================================"
echo "INFERENCE CONFIGURATION"
echo "============================================================================"
echo
echo "Model:"
echo "  Checkpoint:      ${CHECKPOINT}"
echo
echo "Target:"
echo "  Lab:             ${TARGET_LAB}"
echo
echo "Pool:"
if [[ -n "${POOL_JSON}" ]]; then
  echo "  Source:          json:${POOL_JSON}"
elif [[ -n "${POOL_DIR}" ]]; then
  echo "  Source:          jll:${POOL_DIR}"
else
  echo "  Source:          jll:(installed jaxlayerlumos / \$JLL_MATERIALS_DIR)"
fi
echo
echo "Constraints:"
echo "  File:            ${CONSTRAINTS:-(none)}"
echo
echo "Knobs:"
echo "  ensemble_N:      ${ENSEMBLE_N}"
echo "  temperature:     ${TEMPERATURE}"
echo "  tolerance %:     ${TOLERANCE}"
echo "  weight λ:        ${WEIGHT_LAMBDA}"
echo "  top_k:           ${TOP_K}"
echo "  refine_iters:    ${REFINE_ITERS}"
echo "  refine_step nm:  ${REFINE_STEP}"
echo "  mc_samples:      ${MC_SAMPLES}"
echo "  random_restarts: ${RANDOM_RESTARTS}"
echo "  seed:            ${SEED}"
echo
echo "Output:"
echo "  File:            ${OUTPUT:-(default: inference/outputs/result_seed${SEED}.json)}"
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
echo "INFERENCE COMPLETE  exit=${EXIT_CODE}  ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE
# ============================================================================
#
# Minimal:
#   CHECKPOINT=data/checkpoints/flex_..._cross_attn.../latest \
#     TARGET_LAB="60 5 -8" \
#     sbatch slurms/inference.sh
#
# With constraints + custom knobs:
#   CHECKPOINT=data/checkpoints/<tag>/latest \
#     TARGET_LAB="70 0 0" \
#     CONSTRAINTS=inference/example_constraints.json \
#     ENSEMBLE_N=1000 TOLERANCE=3.0 TOP_K=10 \
#     sbatch slurms/inference.sh
#
# Override the JLL pool path (otherwise auto-detected from installed package):
#   JLL_MATERIALS_DIR=/some/path \
#     CHECKPOINT=... TARGET_LAB="..." \
#     sbatch slurms/inference.sh
#
# Send extra positional CLI flags through (anything after the script name):
#   CHECKPOINT=... TARGET_LAB="..." \
#     sbatch slurms/inference.sh --top-k 3 --temperature 0.7
# ============================================================================
