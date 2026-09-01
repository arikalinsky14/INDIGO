#!/usr/bin/env bash
#SBATCH --job-name=indigo-de-curve
#SBATCH --output=job-outputs/indigo-de-curve.%j.out
#SBATCH --error=job-outputs/indigo-de-curve.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# INDIGO ΔE Training-Curve Builder
# ============================================================================
#
# Loops over the per-step checkpoints in CKPT_DIR and runs full evaluate.py
# (autoregressive generation + optical simulation + CIEDE2000) against
# EVAL_DIR/EVAL_SPLIT, writing one JSON per checkpoint. Combine with the
# plotting snippet in the README of prod/de_curve/ to make a ΔE-vs-step
# training curve.
#
# ONE SLURM job — not one job per checkpoint — so module load, Python
# startup, and val pre-collation happen once. Sequential per checkpoint.
#
# Wall-time budget: ~5 min per 1000 val examples per checkpoint (autoreg
# gen + stackrt_n_k on L40S). 20 checkpoints × 1000 val = ~100 min.
#
# Env knobs (all optional):
#   MODE           default: both. One of {compute, plot, both}:
#                    * compute — run per-checkpoint evaluate.py, write JSONs,
#                                SKIP plot. Use when you'll plot later or
#                                overlay with other splits.
#                    * plot    — SKIP the compute loop, just re-plot from
#                                existing eval_step_*.json in OUT_DIR. Cheap;
#                                consider using slurms/plot_de_curve.sh
#                                (smp queue) instead to avoid burning a GPU
#                                allocation.
#                    * both    — compute JSONs then plot. Default.
#   CKPT_DIR       required — directory holding step_XXXX/ subdirs
#                  (e.g. outputs/prod/1ep_bs512_lr6e-5)
#   OUT_DIR        default: $CKPT_DIR/de_curve
#   EVAL_DIR       default: data/train (val split lives here — 5k rows)
#   EVAL_SPLIT     default: validation  (use tier_a/tier_b for those tiers,
#                                        pointed at their DATA_DIR)
#   LIMIT_EXAMPLES default: 1000 (per checkpoint — trade cost for tightness)
#   STEP_START     default: 1000
#   STEP_STOP      default: 100000 (inclusive; script skips missing steps)
#   STEP_STEP     default: 1000  (match SAVE_EVERY of the training run)
#   SAMPLE_PREDICTIONS default: 1 (stochastic sampling at TEMPERATURE — the
#                  more realistic reflection of inference behaviour than
#                  greedy argmax). Set to 0 for the greedy-argmax curve.
#                  Seed is 42 inside evaluate.py so runs across checkpoints
#                  stay comparable.
#   TEMPERATURE    default: 1.0. Only used when SAMPLE_PREDICTIONS=1.
#
# Model hyperparams inherit prod defaults (cross_attn, LR=6e-5, bs=512).
# Override if you're evaluating a non-prod checkpoint.
#
# The default OUT_DIR ($CKPT_DIR/de_curve) now holds SAMPLED-@-T=1 JSONs.
# When comparing sampled vs greedy on the same checkpoint, point the
# non-default decoder at a distinct OUT_DIR (convention:
# de_curve_greedy / de_curve_sample_t${TEMPERATURE}) so the sets don't
# collide.
#
# Examples:
#   # Full pipeline — compute + plot (default):
#   CKPT_DIR=<checkpoint_dir> sbatch slurms/de_curve.sh
#
#   # Compute-only — save the plot step for after you have all splits:
#   MODE=compute CKPT_DIR=<checkpoint_dir> sbatch slurms/de_curve.sh
#
#   # Plot-only — re-render from JSONs already on disk:
#   MODE=plot CKPT_DIR=<checkpoint_dir> sbatch slurms/de_curve.sh
#   # (or use slurms/plot_de_curve.sh on smp — no wasted GPU allocation)
# ============================================================================

: "${MODE:=both}"
case "${MODE}" in
    compute|plot|both) ;;
    *)  echo "ERROR: MODE must be compute|plot|both (got: ${MODE})" >&2
        exit 2 ;;
esac

if [[ -z "${CKPT_DIR:-}" ]]; then
    echo "ERROR: set CKPT_DIR=<path/to/save_dir> before sbatch" >&2
    exit 2
fi

: "${OUT_DIR:=${CKPT_DIR}/de_curve}"
: "${EVAL_DIR:=data/train}"
: "${EVAL_SPLIT:=validation}"
: "${LIMIT_EXAMPLES:=1000}"
: "${STEP_START:=1000}"
: "${STEP_STOP:=100000}"
: "${STEP_STEP:=1000}"

# Sampling knobs. Default: SAMPLE_PREDICTIONS=1 → stochastic sampling at
# TEMPERATURE (matches how inference actually runs; the sampled ΔE curve
# is a more faithful picture of end-user quality than the greedy one).
# SAMPLE_PREDICTIONS=0 → greedy argmax. Seed is 42 inside evaluate.py, so
# runs across checkpoints are apples-to-apples. When running BOTH decoders
# on the same checkpoint, point each at a distinct OUT_DIR so they don't
# clobber each other (convention: de_curve / de_curve_greedy).
: "${SAMPLE_PREDICTIONS:=1}"
: "${TEMPERATURE:=1.0}"

: "${HEAD_MODE:=cross_attn}"
: "${N_HEADS:=8}"
: "${SLOT_ENCODER_LAYERS:=4}"
: "${DECODER_LAYERS:=1}"
: "${FEATURE_MODE:=raw_spectrum}"
: "${ENCODER_HIDDEN:=128}"
: "${ENCODER_OUT:=64}"
: "${ENCODER_DROPOUT:=0.1}"
: "${D_MODEL:=1024}"
: "${N_LAYERS:=8}"
: "${DROPOUT:=0.1}"
: "${LR:=6e-5}"
: "${BATCH_SIZE:=512}"
: "${EPOCHS:=1}"

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs "${OUT_DIR}"

echo "======================================================================"
echo " INDIGO ΔE training-curve builder — Job ${SLURM_JOB_ID:-local}"
echo " MODE:              ${MODE}"
echo " CKPT_DIR:          ${CKPT_DIR}"
echo " OUT_DIR:           ${OUT_DIR}"
echo " EVAL_DIR:          ${EVAL_DIR}"
echo " EVAL_SPLIT:        ${EVAL_SPLIT}"
echo " LIMIT_EXAMPLES:    ${LIMIT_EXAMPLES}"
echo " Step range:        ${STEP_START}..${STEP_STOP} step ${STEP_STEP}"
echo " SAMPLE:            ${SAMPLE_PREDICTIONS} (temperature=${TEMPERATURE})"
echo "======================================================================"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

N_DONE=0
N_MISSING=0
if [[ "${MODE}" == "plot" ]]; then
    echo "[mode=plot] skipping per-checkpoint compute — plotting existing JSONs only"
    # For plot-only, count what's already on disk so the summary line makes sense.
    if [[ -d "${OUT_DIR}" ]]; then
        N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name 'eval_step_*.json' | wc -l)
    fi
    if [[ "${N_DONE}" -eq 0 ]]; then
        echo "ERROR: MODE=plot but no eval_step_*.json in ${OUT_DIR}." >&2
        echo "       Run MODE=compute (or MODE=both) first." >&2
        exit 3
    fi
else
    # Common evaluate.py args (identify architecture; no --checkpoint here —
    # each iteration adds its own).
    COMMON_ARGS=(
        --data-dir "${EVAL_DIR}"
        --split "${EVAL_SPLIT}"
        --limit-examples "${LIMIT_EXAMPLES}"
        --feature-mode "${FEATURE_MODE}"
        --encoder-hidden "${ENCODER_HIDDEN}"
        --encoder-out "${ENCODER_OUT}"
        --encoder-dropout "${ENCODER_DROPOUT}"
        --d-model "${D_MODEL}"
        --n-layers "${N_LAYERS}"
        --dropout "${DROPOUT}"
        --head-mode "${HEAD_MODE}"
        --n-heads "${N_HEADS}"
        --slot-encoder-layers "${SLOT_ENCODER_LAYERS}"
        --decoder-layers "${DECODER_LAYERS}"
        --lr "${LR}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${EPOCHS}"
        --num-workers 2
        --streaming
        --no-swatch
    )
    if [[ "${SAMPLE_PREDICTIONS}" == "1" ]]; then
        COMMON_ARGS+=(--sample-predictions --temperature "${TEMPERATURE}")
        echo "[sampling] stochastic sampling ON (temperature=${TEMPERATURE}, seed=42)"
    else
        echo "[sampling] greedy argmax (SAMPLE_PREDICTIONS=0)"
    fi

    for STEP in $(seq "${STEP_START}" "${STEP_STEP}" "${STEP_STOP}"); do
        CKPT="${CKPT_DIR}/step_${STEP}"
        if [[ ! -d "${CKPT}" ]]; then
            # Skip silently — the caller may set STEP_STOP > actual last save.
            N_MISSING=$((N_MISSING + 1))
            continue
        fi
        OUTFILE="${OUT_DIR}/eval_step_${STEP}.json"
        if [[ -f "${OUTFILE}" ]]; then
            echo "[skip] step_${STEP}: ${OUTFILE} already exists"
            N_DONE=$((N_DONE + 1))
            continue
        fi
        echo
        echo "---- step_${STEP} ----------------------------------------------"
        python scripts/evaluate.py \
            --checkpoint "${CKPT}" \
            --output "${OUTFILE}" \
            "${COMMON_ARGS[@]}"
        N_DONE=$((N_DONE + 1))
    done

    # Also evaluate the two roll-up checkpoints if present.
    for TAG in latest final; do
        CKPT="${CKPT_DIR}/${TAG}"
        if [[ -d "${CKPT}" ]]; then
            OUTFILE="${OUT_DIR}/eval_${TAG}.json"
            if [[ ! -f "${OUTFILE}" ]]; then
                echo
                echo "---- ${TAG} ---------------------------------------------"
                python scripts/evaluate.py \
                    --checkpoint "${CKPT}" \
                    --output "${OUTFILE}" \
                    "${COMMON_ARGS[@]}"
            fi
        fi
    done
fi

echo
echo "======================================================================"
echo " Per-checkpoint eval done  evaluated=${N_DONE}  missing_steps=${N_MISSING}"
echo " JSONs in: ${OUT_DIR}"
echo "======================================================================"

if [[ "${MODE}" == "compute" ]]; then
    echo
    echo "[mode=compute] skipping plot. Fire it later with:"
    echo "  MODE=plot CKPT_DIR=${CKPT_DIR} OUT_DIR=${OUT_DIR} sbatch slurms/de_curve.sh"
    echo "  (or, on smp: INPUT_DIR=${OUT_DIR} sbatch slurms/plot_de_curve.sh)"
    exit 0
fi

# Plot the ΔE-vs-step curve as the final step of the SAME SLURM job so
# nothing needs to be run interactively on the login node.
LABEL="${LABEL:-$(basename "${OUT_DIR}")}"
TITLE="${TITLE:-INDIGO ΔE₀₀ vs training step (${EVAL_SPLIT} @ limit=${LIMIT_EXAMPLES})}"

echo
echo "[plot] launching scripts/plot_de_curve.py..."
python scripts/plot_de_curve.py \
    --input-dir "${OUT_DIR}" \
    --label     "${LABEL}" \
    --title     "${TITLE}" \
    --output    "${OUT_DIR}/de_curve.png"

echo
echo "======================================================================"
echo " DONE  — plot at ${OUT_DIR}/de_curve.png"
echo "======================================================================"
