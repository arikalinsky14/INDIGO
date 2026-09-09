#!/usr/bin/env bash
#SBATCH --job-name=indigo-ste-proj
#SBATCH --output=job-outputs/indigo-ste-proj.%j.out
#SBATCH --error=job-outputs/indigo-ste-proj.%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=03:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# STE Projection Quality Test — wrapper
# ============================================================================
#
# Runs analyses/de_finetune/ste_projection_quality.py to quantify how
# well the ΔE finetune's linear projection of the sim gradient onto every
# pool material approximates the true "swap and re-sim" ΔE.
#
# Two modes selected by BASE_MODE:
#
#   BASE_MODE=gt (default)
#     Anchors the linearization at the GT slot per position. base ΔE ~= 0
#     so this measures projection accuracy at the optimum. Runs on CPU
#     (smp partition) — no torch model needed.
#
#   BASE_MODE=model_argmax
#     Loads a pretrained checkpoint, runs forward with GT structure_matrix,
#     uses the model's argmax slot + argmax thickness per position as the
#     anchor. base ΔE > 0 (roughly the model's own greedy ΔE). Measures
#     projection accuracy at the *training-time* anchor points. Needs
#     PRETRAINED_CHECKPOINT and runs on GPU (l40s) so the torch forward
#     is fast; JAX stays on CPU throughout regardless.
#
# Verdict (rule of thumb, exact thresholds in the summary):
#   HOLDS   — projection is a useful gradient signal at these anchors
#   MIXED   — partial signal
#   NOISE   — projection is unreliable at these anchors
#
# Cost per run: N_EXAMPLES × avg_layers × avg_pool_size sims + JAX warmup.
# Ballpark: 500 examples × 4 layers × 25 materials = 50k sims at ~30 ms
# each = ~25 min after warmup. Time budget generous.
#
# Environment variables (all optional; sensible defaults):
#   BASE_MODE             default: gt   ('gt' | 'model_argmax')
#   PRETRAINED_CHECKPOINT required if BASE_MODE=model_argmax
#   DATA_DIR              default: data/finetune
#   SPLIT                 default: all
#   N_EXAMPLES            default: 500
#   LAYERS_PER_EXAMPLE    default: all   ('all', 'random', or an int)
#   INCIDENCE_ANGLE       default: 0
#   SEED                  default: 42
#   OUTPUT_DIR            default: analyses/de_finetune/results/ste_projection_<mode>_<JOBID>
#
# Model architecture (only used in model_argmax mode; defaults match
# the current prod checkpoint):
#   FEATURE_MODE, ENCODER_HIDDEN, ENCODER_OUT, ENCODER_DROPOUT,
#   D_MODEL, N_LAYERS, DROPOUT, HEAD_MODE, N_HEADS,
#   SLOT_ENCODER_LAYERS, DECODER_LAYERS
#
# Usage:
#   # GT-anchor sweep (default, cheap):
#   sbatch --clusters=smp --partition=smp slurms/ste_projection_quality.sh
#
#   # Model-argmax anchor sweep — the training-time diagnostic:
#   BASE_MODE=model_argmax \
#       PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#       sbatch --clusters=gpu --partition=l40s --gres=gpu:1 slurms/ste_projection_quality.sh
#
#   Quick smoke (50 examples, 1 layer each):
#     N_EXAMPLES=50 LAYERS_PER_EXAMPLE=random \
#         sbatch --time=00:30:00 --clusters=smp --partition=smp \
#         slurms/ste_projection_quality.sh
# ============================================================================

module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# JAX stays CPU-only for the sim (no XLA GPU compile hassles for our
# per-call vjp path); torch may use GPU if allocated for the model
# forward in model_argmax mode.
export JAX_PLATFORMS=cpu

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

# ---- Defaults ----
: "${BASE_MODE:=gt}"
: "${DATA_DIR:=data/finetune}"
: "${SPLIT:=all}"
: "${N_EXAMPLES:=500}"
: "${LAYERS_PER_EXAMPLE:=all}"
: "${INCIDENCE_ANGLE:=0}"
: "${SEED:=42}"
: "${OUTPUT_DIR:=analyses/de_finetune/results/ste_projection_${BASE_MODE}_${SLURM_JOB_ID:-local}}"

# Model architecture defaults (only used in model_argmax mode).
: "${FEATURE_MODE:=raw_spectrum}"
: "${ENCODER_HIDDEN:=128}"
: "${ENCODER_OUT:=64}"
: "${ENCODER_DROPOUT:=0.1}"
: "${D_MODEL:=1024}"
: "${N_LAYERS:=8}"
: "${DROPOUT:=0.1}"
: "${HEAD_MODE:=cross_attn}"
: "${N_HEADS:=8}"
: "${SLOT_ENCODER_LAYERS:=4}"
: "${DECODER_LAYERS:=1}"

# Validation.
if [[ "${BASE_MODE}" == "model_argmax" && -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
    echo "ERROR: BASE_MODE=model_argmax requires PRETRAINED_CHECKPOINT" >&2
    exit 2
fi

echo "============================================================================"
echo "INDIGO STE PROJECTION QUALITY - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:              $(pwd)"
echo "Node:             $(hostname)"
echo "Python:           $(which python)"
echo "Started:          $(date)"
echo
python --version
python -c "import jax, jaxlayerlumos; print(f'JAX: {jax.__version__}  jaxlayerlumos: {jaxlayerlumos.__version__}')" 2>/dev/null || echo "[WARN] JAX / jaxlayerlumos import check failed"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "[INFO] no GPU visible to this job"
fi
echo
echo "----- Configuration -----"
echo "BASE_MODE           : ${BASE_MODE}"
if [[ "${BASE_MODE}" == "model_argmax" ]]; then
    echo "PRETRAINED_CHECKPOINT: ${PRETRAINED_CHECKPOINT}"
fi
echo "DATA_DIR            : ${DATA_DIR}"
echo "SPLIT               : ${SPLIT}"
echo "N_EXAMPLES          : ${N_EXAMPLES}"
echo "LAYERS_PER_EXAMPLE  : ${LAYERS_PER_EXAMPLE}"
echo "INCIDENCE_ANGLE     : ${INCIDENCE_ANGLE}"
echo "SEED                : ${SEED}"
echo "OUTPUT_DIR          : ${OUTPUT_DIR}"
echo "============================================================================"
echo

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python analyses/de_finetune/ste_projection_quality.py
    --data-dir            "${DATA_DIR}"
    --split               "${SPLIT}"
    --n-examples          "${N_EXAMPLES}"
    --layers-per-example  "${LAYERS_PER_EXAMPLE}"
    --incidence-angle     "${INCIDENCE_ANGLE}"
    --seed                "${SEED}"
    --output-dir          "${OUTPUT_DIR}"
    --base-mode           "${BASE_MODE}"
)

if [[ "${BASE_MODE}" == "model_argmax" ]]; then
    CMD+=(
        --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
        --feature-mode          "${FEATURE_MODE}"
        --encoder-hidden        "${ENCODER_HIDDEN}"
        --encoder-out           "${ENCODER_OUT}"
        --encoder-dropout       "${ENCODER_DROPOUT}"
        --d-model               "${D_MODEL}"
        --n-layers              "${N_LAYERS}"
        --dropout               "${DROPOUT}"
        --head-mode             "${HEAD_MODE}"
        --n-heads               "${N_HEADS}"
        --slot-encoder-layers   "${SLOT_ENCODER_LAYERS}"
        --decoder-layers        "${DECODER_LAYERS}"
    )
fi

echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "STE PROJECTION QUALITY COMPLETE   exit=${EXIT_CODE}   ended=$(date)"
echo "OUTPUTS:  ${OUTPUT_DIR}/"
echo "  - results.json"
echo "  - summary.txt"
echo "  - projected_vs_true.png"
echo "  - spearman_hist.png"
echo "  - sign_agreement_hist.png"
echo "============================================================================"
exit ${EXIT_CODE}
