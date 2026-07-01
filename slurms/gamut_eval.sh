#!/usr/bin/env bash
#SBATCH --job-name=indigo-gamut-eval
#SBATCH --output=job-outputs/indigo-gamut-eval.%j.out
#SBATCH --error=job-outputs/indigo-gamut-eval.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# INDIGO gamut evaluation — production ΔE across five test batteries
# ============================================================================
#
# Runs `inference/scripts/gamut_eval.py` on a checkpoint against a fixed
# set of Lab targets covering the sRGB gamut extremes and the L/a/b axes:
#
#   rgb_primaries      8 sRGB cube corners in Lab (R, G, B, C, M, Y,
#                      black, white).
#   lab_l_sweep        L in {5, 20, 40, 60, 80, 95} at a=b=0.
#   lab_a_sweep        L=50, b=0, a in {-80, -40, 0, 40, 80}.
#   lab_b_sweep        L=50, a=0, b in {-80, -40, 0, 40, 80}.
#   chromatic_corners  L=50, (a,b) in {(±80, ±80)}.
#
# Per target: solve() → record achieved Lab, ΔE_00, layer count, wall time.
# Per battery: median / mean / p95 / worst ΔE + pass rates at ΔE < {1, 2, 5, 10}.
# Aggregated JSON written under --output; stable enough for CI regression.
#
# Time budget (balanced preset, 28 targets total):
#   - Per-target solve: ~15-25 s on L40s (dominant cost is sequential JLL
#     physics for refine + score).
#   - OPTIMIZER=dog   → ~28 × 20 s ≈ 10 min.
#   - OPTIMIZER=adam  → same.
#   - OPTIMIZER=both  → ~2× because every target runs twice.
# The 02:00:00 wall is padding for cold module loads / bigger presets.
# ============================================================================

set -euo pipefail

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
# JAX physics chain isn't GPU-amenable (jit/vmap blocked by the JLL
# stackrt assert). Keep it on CPU so the GPU stays for the torch forward.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs inference/outputs/gamut_eval

echo "============================================================================"
echo "INDIGO GAMUT EVAL - Job ${SLURM_JOB_ID:-local}"
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

# One of {fast, balanced, best} — mirrors the frontend presets so the
# numbers here match what a user would see clicking the same preset.
PRESET="${PRESET:-balanced}"

# One of {dog, adam, both}. Default matches the InferenceKnobs default.
# 'both' doubles wall time by running every target twice, but produces a
# side-by-side A/B in the output JSON that answers "did we improve?".
OPTIMIZER="${OPTIMIZER:-dog}"

# Where the aggregate JSON lands. Defaults include the checkpoint tag and
# optimizer so re-runs don't clobber each other.
OUTPUT_DIR="${OUTPUT_DIR:-inference/outputs/gamut_eval}"
OUTPUT_NAME="${OUTPUT_NAME:-gamut_${PRESET}_${OPTIMIZER}.json}"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}"

# Base RNG seed. Per-battery seeds derive from it.
SEED="${SEED:-42}"

# Optional JLL materials override (defaults to the installed package or
# $JLL_MATERIALS_DIR if you've exported one).
POOL_DIR="${POOL_DIR:-}"

# Anything after the script name flows through to gamut_eval.py.
USER_ARGS=("$@")

# ============================================================================
# VALIDATION
# ============================================================================

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT env var is required." >&2
  echo "        Example: CHECKPOINT=data/checkpoints/<tag>/latest" >&2
  exit 2
fi
case "${PRESET}" in
  fast|balanced|best) ;;
  *) echo "[ERROR] PRESET must be one of fast/balanced/best (got ${PRESET})" >&2; exit 2 ;;
esac
case "${OPTIMIZER}" in
  dog|adam|both) ;;
  *) echo "[ERROR] OPTIMIZER must be one of dog/adam/both (got ${OPTIMIZER})" >&2; exit 2 ;;
esac

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --preset "${PRESET}"
  --optimizer "${OPTIMIZER}"
  --output "${OUTPUT}"
  --seed "${SEED}"
)
if [[ -n "${POOL_DIR}" ]]; then
  ARGS+=(--pool-dir "${POOL_DIR}")
fi

CMD=(python -m inference.scripts.gamut_eval "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY
# ============================================================================

echo "============================================================================"
echo "GAMUT EVAL CONFIGURATION"
echo "============================================================================"
echo
echo "Model:"
echo "  Checkpoint:        ${CHECKPOINT}"
echo
echo "Batteries (5 total, 28 targets):"
echo "  rgb_primaries       (8 targets — sRGB cube corners)"
echo "  lab_l_sweep         (6 targets — L axis at a=b=0)"
echo "  lab_a_sweep         (5 targets — a axis at L=50, b=0)"
echo "  lab_b_sweep         (5 targets — b axis at L=50, a=0)"
echo "  chromatic_corners   (4 targets — (±80, ±80) at L=50)"
echo
echo "Sweep knobs:"
echo "  preset:            ${PRESET}"
echo "  optimizer:         ${OPTIMIZER}"
echo "  seed:              ${SEED}"
if [[ -n "${POOL_DIR}" ]]; then
  echo "  pool_dir:          ${POOL_DIR}"
else
  echo "  pool_dir:          (installed JLL package)"
fi
echo
echo "Output:"
echo "  JSON:              ${OUTPUT}"
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
echo "GAMUT EVAL COMPLETE  exit=${EXIT_CODE}  ended=$(date)"
echo "============================================================================"
if [[ -f "${OUTPUT}" ]]; then
  echo "Output JSON: ${OUTPUT}"
  # Small human-readable summary tail for the mail notification.
  python - <<PY 2>/dev/null || true
import json, sys
data = json.load(open("${OUTPUT}"))
def summary(root, label):
    s = root.get("overall", {}).get("stats") or {}
    if s.get("median_de") is None:
        print(f"  {label}: all targets failed")
        return
    print(f"  {label}: n={s['n']} median={s['median_de']:.2f} "
          f"mean={s['mean_de']:.2f} p95={s['p95_de']:.2f} "
          f"worst={s['worst_de']:.2f} "
          f"<5={s['pass_rate_de_lt_5']*100:.0f}% "
          f"<10={s['pass_rate_de_lt_10']*100:.0f}%")
if "ab_comparison" in data:
    for opt in ("dog", "adam"):
        summary(data["ab_comparison"][opt], f"optimizer={opt}")
else:
    summary(data, f"optimizer={data.get('optimizer','?')}")
PY
fi
exit ${EXIT_CODE}

# ============================================================================
# USAGE
# ============================================================================
#
# Minimal (default = balanced preset, DoG optimizer):
#   CHECKPOINT=data/checkpoints/<tag>/latest sbatch slurms/gamut_eval.sh
#
# DoG vs Adam A/B on the balanced preset (recommended first run):
#   CHECKPOINT=<ckpt> OPTIMIZER=both sbatch slurms/gamut_eval.sh
#
# Quick smoke test with the fast preset (~5 min):
#   CHECKPOINT=<ckpt> PRESET=fast sbatch slurms/gamut_eval.sh
#
# High-quality reference number for a release (~30-60 min):
#   CHECKPOINT=<ckpt> PRESET=best OPTIMIZER=both \
#     sbatch slurms/gamut_eval.sh
#
# Custom pool + named output:
#   CHECKPOINT=<ckpt> POOL_DIR=/path/to/materials \
#     OUTPUT_NAME=gamut_ep17_release.json \
#     sbatch slurms/gamut_eval.sh
# ============================================================================
