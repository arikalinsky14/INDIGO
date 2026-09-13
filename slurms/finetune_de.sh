#!/usr/bin/env bash
#SBATCH --job-name=indigo-finetune-de
#SBATCH --output=job-outputs/indigo-finetune-de.%j.out
#SBATCH --error=job-outputs/indigo-finetune-de.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# INDIGO Post-Training ΔE Finetune
# ============================================================================
#
# Companion to slurms/training.sh but for the post-training ΔE finetune
# stage. Full design is documented in analyses/de_finetune/README.md.
#
# Two experiments, selected by FREEZE_ENCODER:
#   FREEZE_ENCODER=1   Experiment A — decoder-only (freeze pool encoder).
#                      Default LR: 5e-6.
#   FREEZE_ENCODER=0   Experiment B — full-model. Default LR: 1e-6.
#
# Required env vars:
#   PRETRAINED_CHECKPOINT   path to the pretrained checkpoint dir
#                           (recommend the val-optimal step, e.g.
#                           .../step_13000/ — NOT .../latest/ if the
#                           pretrain overfit past that point)
#   SAVE_DIR                where finetune checkpoints land
#
# Optional env vars:
#   DATA_DIR                default: data/finetune (must have HC=0.30
#                           parquet shards — see slurms/generate_data.sh
#                           command in analyses/de_finetune/README.md)
#   FREEZE_ENCODER          default: 1
#   LR                      default: 5e-6 if FREEZE_ENCODER=1 else 1e-6
#   EPOCHS                  default: 3
#   BATCH_SIZE              default: 128
#   NUM_WORKERS             default: 4
#   SAVE_EVERY              default: 500
#   LOG_EVERY               default: 50
#   LIMIT_EXAMPLES          default: unset (full dataset)
#   LIMIT_VAL_EXAMPLES      default: 1000
#   SEED                    default: 42
#   RESUME                  optional; path to a finetune checkpoint to resume
#
# Model hyperparameters MUST match the pretrained checkpoint (defaults
# below match the current prod cross_attn config).
#
# ============================================================================

# -------------------- Environment --------------------
module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "INDIGO ΔE FINETUNE - Job ${SLURM_JOB_ID:-local}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"

python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# -------------------- Required inputs --------------------
if [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
    echo "ERROR: set PRETRAINED_CHECKPOINT=<path/to/step_N> before sbatch" >&2
    exit 2
fi
if [[ -z "${SAVE_DIR:-}" ]]; then
    echo "ERROR: set SAVE_DIR=<path/to/finetune/output> before sbatch" >&2
    exit 2
fi

# -------------------- Defaults --------------------
: "${DATA_DIR:=data/finetune}"
: "${FREEZE_ENCODER:=1}"
: "${EPOCHS:=3}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=4}"
: "${SAVE_EVERY:=500}"
: "${LOG_EVERY:=50}"
: "${LIMIT_VAL_EXAMPLES:=1000}"
: "${SEED:=42}"

if [[ -z "${LR:-}" ]]; then
    if [[ "${FREEZE_ENCODER}" == "1" ]]; then
        LR="5e-6"
    else
        LR="1e-6"
    fi
fi

# Model architecture (match pretrained checkpoint).
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

# Optimisation.
: "${WEIGHT_DECAY:=0.01}"
: "${GRAD_CLIP:=1.0}"
: "${WARMUP_FRACTION:=0.02}"
: "${INCIDENCE_ANGLE:=0.0}"

# CE anchor loss weight. 0 = pure ΔE (the original finetune); >0 blends
# in per-token CE against GT to keep the model near the pretrain
# manifold — needed because the STE gradient's linearization at model-
# argmax anchors has only ~41% top-1 accuracy and drifts unaanchored
# training away from the pretrained CE optimum. See
# analyses/de_finetune/GRADIENT_FLOW_EXPLAINER.md.
: "${CE_LOSS_WEIGHT:=0.0}"

# Top-K real-sim loss (C1). 0 = STE mode (original ΔE linearization,
# fast but the Sept 9 sweep showed it is net-harmful at every scale).
# K>0 replaces the STE primary loss with a listwise CE loss over K real
# sims per position — target dist is softmax(-SIM_TARGET_BETA · ΔE_real)
# so gradient flows into slot logits and pushes them toward the actual
# argmin-ΔE candidate. Sim cost scales linearly with K. Val ALWAYS runs
# the STE greedy path so val_loss_de stays comparable across K.
: "${REAL_SIM_TOPK:=0}"
: "${SIM_TARGET_BETA:=1.0}"

# Top-K mode.
#   slot         top-K over per-slot scores (max-over-thickness), each
#                candidate uses its argmax thickness.
#   joint        top-K over the flat (slot × thickness) grid. Sept 10
#                finding: concentrates on 1-2 slots' neighbor-thickness
#                bins, loss stalls at log(K). Not recommended.
#   hierarchical top-K slots AND top-N thicknesses per slot (N via
#                THICKNESS_TOPN). Total K·N sims/pos, all distinct
#                (slot, thick) pairs — material diversity AND
#                thickness training.
: "${TOPK_MODE:=slot}"
: "${THICKNESS_TOPN:=1}"

# LR schedule after warmup. 'cosine' decays to 0 by end (matches pretrain).
# 'constant' holds base LR flat — better for long runs where cosine decay
# to zero hurts (Sept 10 K=3 run peaked at step 600/1665 and got worse
# with continued cosine decay).
: "${LR_SCHEDULE:=cosine}"

# ε-exploration for the top-K real-sim loss. floor(K · ε) of the K sim
# slots come from uniform-random draws over the active grid; the rest
# from top-K by logit. ε anneals linearly from EPSILON_START to
# EPSILON_END over EPSILON_DECAY_FRACTION · total_steps, then holds at
# EPSILON_END. Both 0 = pure top-K (default). Typical schedule:
# START=0.3 END=0.0 DECAY_FRACTION=0.5 (30% random early, off by
# midpoint).
: "${EPSILON_START:=0.0}"
: "${EPSILON_END:=0.0}"
: "${EPSILON_DECAY_FRACTION:=1.0}"

echo "============================================================================"
echo "FINETUNE CONFIGURATION"
echo "============================================================================"
echo "PRETRAINED_CHECKPOINT : ${PRETRAINED_CHECKPOINT}"
echo "SAVE_DIR              : ${SAVE_DIR}"
echo "DATA_DIR              : ${DATA_DIR}"
echo "FREEZE_ENCODER        : ${FREEZE_ENCODER}  ($([ "${FREEZE_ENCODER}" = "1" ] && echo "Experiment A: decoder-only" || echo "Experiment B: full-model"))"
echo "LR                    : ${LR}"
echo "CE_LOSS_WEIGHT        : ${CE_LOSS_WEIGHT}"
echo "REAL_SIM_TOPK         : ${REAL_SIM_TOPK}  ($([ "${REAL_SIM_TOPK}" = "0" ] && echo "STE mode" || echo "top-K real-sim mode"))"
echo "SIM_TARGET_BETA       : ${SIM_TARGET_BETA}"
echo "TOPK_MODE             : ${TOPK_MODE}"
echo "THICKNESS_TOPN        : ${THICKNESS_TOPN}"
echo "LR_SCHEDULE           : ${LR_SCHEDULE}"
echo "EPSILON               : start=${EPSILON_START}  end=${EPSILON_END}  decay_frac=${EPSILON_DECAY_FRACTION}"
echo "EPOCHS                : ${EPOCHS}"
echo "BATCH_SIZE            : ${BATCH_SIZE}"
echo "NUM_WORKERS           : ${NUM_WORKERS}"
echo "SAVE_EVERY / LOG_EVERY: ${SAVE_EVERY} / ${LOG_EVERY}"
echo "SEED                  : ${SEED}"
echo "============================================================================"

# -------------------- Build command --------------------
ARGS=(
    --data-dir            "${DATA_DIR}"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --save-dir            "${SAVE_DIR}"
    --seed                "${SEED}"
    --limit-val-examples  "${LIMIT_VAL_EXAMPLES}"

    --feature-mode        "${FEATURE_MODE}"
    --encoder-hidden      "${ENCODER_HIDDEN}"
    --encoder-out         "${ENCODER_OUT}"
    --encoder-dropout     "${ENCODER_DROPOUT}"
    --d-model             "${D_MODEL}"
    --n-layers            "${N_LAYERS}"
    --dropout             "${DROPOUT}"
    --head-mode           "${HEAD_MODE}"
    --n-heads             "${N_HEADS}"
    --slot-encoder-layers "${SLOT_ENCODER_LAYERS}"
    --decoder-layers      "${DECODER_LAYERS}"

    --lr                  "${LR}"
    --weight-decay        "${WEIGHT_DECAY}"
    --grad-clip           "${GRAD_CLIP}"
    --warmup-fraction     "${WARMUP_FRACTION}"
    --epochs              "${EPOCHS}"
    --batch-size          "${BATCH_SIZE}"
    --num-workers         "${NUM_WORKERS}"

    --save-every          "${SAVE_EVERY}"
    --log-every           "${LOG_EVERY}"
    --incidence-angle     "${INCIDENCE_ANGLE}"
    --ce-loss-weight      "${CE_LOSS_WEIGHT}"
    --real-sim-topk       "${REAL_SIM_TOPK}"
    --sim-target-beta     "${SIM_TARGET_BETA}"
    --topk-mode           "${TOPK_MODE}"
    --thickness-topn      "${THICKNESS_TOPN}"
    --lr-schedule         "${LR_SCHEDULE}"
    --epsilon-start       "${EPSILON_START}"
    --epsilon-end         "${EPSILON_END}"
    --epsilon-decay-fraction "${EPSILON_DECAY_FRACTION}"
)

if [[ "${FREEZE_ENCODER}" == "1" ]]; then
    ARGS+=(--freeze-encoder)
fi
if [[ -n "${LIMIT_EXAMPLES:-}" ]]; then
    ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi
if [[ -n "${RESUME:-}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi

CMD=(python scripts/finetune_de.py "${ARGS[@]}")

echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

"${CMD[@]}"

EXIT_CODE=$?
echo
echo "============================================================================"
echo "FINETUNE COMPLETE   exit=${EXIT_CODE}   ended=$(date)"
echo "============================================================================"
exit ${EXIT_CODE}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# Experiment A (decoder-only) — start here:
#   PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#       SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_A_decoder_only \
#       FREEZE_ENCODER=1 LR=5e-6 \
#       sbatch slurms/finetune_de.sh
#
# Experiment B (full-model) — only if A shows lift:
#   PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#       SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_full_model \
#       FREEZE_ENCODER=0 LR=1e-6 \
#       sbatch slurms/finetune_de.sh
#
# Resume a finetune that timed out:
#   PRETRAINED_CHECKPOINT=<same as original> \
#       SAVE_DIR=<same as original> \
#       RESUME=<SAVE_DIR>/latest \
#       FREEZE_ENCODER=<same> LR=<same> \
#       sbatch slurms/finetune_de.sh
#
# CE-anchored finetune — after the Sept 8 diagnostic showed the STE
# gradient drifts an unanchored ΔE-only finetune off the pretrain
# manifold. λ = 1.0 is the balanced-anchor case; sweep {0.1, 1.0, 10.0}
# to bracket loose vs strong anchor.
#   for LAM in 0.1 1.0 10.0; do
#     PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#         SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_A_ceanchor_lam${LAM} \
#         FREEZE_ENCODER=1 LR=1e-6 CE_LOSS_WEIGHT=${LAM} \
#         EPOCHS=1 LIMIT_EXAMPLES=128000 LIMIT_VAL_EXAMPLES=1000 \
#         NUM_WORKERS=0 LOG_EVERY=25 SAVE_EVERY=250 \
#         sbatch --time=03:00:00 slurms/finetune_de.sh
#   done
#
# Top-K real-sim sweep (C1) — after the Sept 9 CE-anchor λ sweep
# confirmed the STE linearization is unusable at every scale. Uses real
# sim + listwise CE per position; sim cost is ~K× the STE baseline, so
# LIMIT_EXAMPLES is scaled ~1/K to keep wall clock at ~3h per job.
#
# CE_LOSS_WEIGHT=0.5 is included because the top-K slot loss only puts
# gradient on the argmax thickness cell for each picked slot — it can't
# discover a wrong thickness pick. The CE anchor at 0.5 trains the
# thickness head against GT tokens as a safety net (roughly balances
# gradient magnitudes with topk once topk drops from ~log(K) → 0).
# Compare val_loss_de across K at the same wall-time budget:
#
#   for K in 5 10 15; do
#     case $K in 5) LIM=40000;; 10) LIM=20000;; 15) LIM=13000;; esac
#     PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#         SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_A_topk${K}_ce0p5 \
#         FREEZE_ENCODER=1 LR=1e-6 REAL_SIM_TOPK=${K} SIM_TARGET_BETA=1.0 \
#         CE_LOSS_WEIGHT=0.5 \
#         EPOCHS=1 LIMIT_EXAMPLES=${LIM} LIMIT_VAL_EXAMPLES=500 \
#         NUM_WORKERS=0 LOG_EVERY=25 SAVE_EVERY=100 \
#         sbatch --time=03:00:00 slurms/finetune_de.sh
#   done
#
# If topk gradient looks dominated by CE (loss_topk not falling but
# loss_ce falling nicely), rerun with CE_LOSS_WEIGHT=0.2 to soften CE
# and let topk drive slot selection more.
#
# ---------------------------------------------------------------------------
# Diagnostic pair (Sept 10) — decide "does joint-K + const LR + ε
# help vs the K=3-slot baseline?" BEFORE committing multi-day compute.
# Sized to see 1000-1500 steps past the peak zone (Sept 10 K=3 peaked
# at step 600), which is what we need to distinguish "constant LR
# holds the peak" from "still degrades regardless of schedule."
#
#   A. Baseline reproduce with constant LR + best-checkpoint saving.
#      213k examples (matches the earlier winning run at 1665 steps),
#      ~2.5h at K=3 slot.
#        FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=3 CE_LOSS_WEIGHT=0.1 \
#            TOPK_MODE=slot LR_SCHEDULE=constant \
#            PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#            SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_slot3_ce0p1_lr1e5_213k_const \
#            EPOCHS=1 LIMIT_EXAMPLES=213000 LIMIT_VAL_EXAMPLES=1000 \
#            NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
#            sbatch --time=04:00:00 slurms/finetune_de.sh
#
#   B. Joint top-15 + ε=0.3→0.0 over first half + constant LR.
#      160k examples ≈ 1250 steps at K=15 joint (~5.5h at ~8 ex/s).
#      Tests joint mode + exploration + const LR in one shot.
#        FREEZE_ENCODER=0 LR=1e-5 REAL_SIM_TOPK=15 CE_LOSS_WEIGHT=0.1 \
#            TOPK_MODE=joint LR_SCHEDULE=constant \
#            EPSILON_START=0.3 EPSILON_END=0.0 EPSILON_DECAY_FRACTION=0.5 \
#            PRETRAINED_CHECKPOINT=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/prod_3ep_bs512_lr6e-5/step_13000 \
#            SAVE_DIR=/ix1/ohinder/ajk245/Github/INDIGO/data/checkpoints/finetune_de_B_joint15_ce0p1_lr1e5_160k_eps03 \
#            EPOCHS=1 LIMIT_EXAMPLES=160000 LIMIT_VAL_EXAMPLES=1000 \
#            NUM_WORKERS=0 LOG_EVERY=50 SAVE_EVERY=250 \
#            sbatch --time=07:00:00 slurms/finetune_de.sh
#
# Both auto-save `best/` on val_loss_de improvements. Compare best@step
# vs 7.978 baseline. Only then commit to a multi-day 1M-example run
# with the winning config.
#
# Fresh 1M finetune data (HC=0.30) — one-time smp job:
#   TOTAL_ROWS=1000000 START_SHARD_ID=3000000 \
#       OUTPUT_DIR=data/finetune \
#       HIGH_CHROMA_PROB=0.30 \
#       sbatch slurms/generate_data.sh
# ============================================================================
