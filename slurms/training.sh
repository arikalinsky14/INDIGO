#!/usr/bin/env bash
#SBATCH --job-name=indigo-train
#SBATCH --output=job-outputs/indigo-train.%j.out
#SBATCH --error=job-outputs/indigo-train.%j.err

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
# SLURM Training Script for INDIGO (scripts/training.py)
# ============================================================================
#
# This script trains the INDIGO FlexMaterialMLP with:
# - RGB + variable material pool conditioning
# - Shared per-slot material encoder + feedforward backbone
# - Learning rate schedule: linear warmup + cosine decay
# - Checkpoints saved to: data/checkpoints/<hparams_tag>/
#
# POST-TRAIN AUTO-CHAIN: on a successful exit, this wrapper submits
# TWO follow-up SLURM jobs so the training + ΔE curves land on disk
# with no manual step:
#   * plot_training_curve.sh — CE loss (train+val) from history.jsonl
#   * de_curve.sh            — per-checkpoint ΔE₀₀ on the val split
# Requires SAVE_DIR to be set on the outer sbatch (so we know where the
# checkpoints landed). Set AUTO_POST_TRAIN=0 to skip.
# The evaluate.sh SLURM stays reserved for the final tier-A / tier-B
# reports (loss + ΔE + color swatches).
#
# ARCHITECTURE:
#   MaterialEncoder (shared per slot): [2, NUM_LAMBDA=128] -> [encoder_out]
#   Backbone input  = RGB (3) + M_MAX*encoder_out + M_MAX*MAX_LAYERS + 1
#                  -> Dense(d_model) -> ReLU -> Dropout
#                  -> [Dense(d_model) -> ReLU -> Dropout] x (n_layers - 1)
#                  -> Dense(vocab_size=3201 = M_MAX*NUM_THICKNESSES + EOS)
#
# USAGE EXAMPLES:
#
# 1. Full production training (default settings):
#    sbatch slurms/training.sh
#
# 2. Quick validation run (limit to 1000 examples):
#    sbatch slurms/training.sh \
#      --limit-examples 1000
#
# 3. Custom hyperparameters:
#    sbatch slurms/training.sh \
#      --lr 1e-3 --d-model 512 --n-layers 6 --epochs 20
#
# ============================================================================

# -------------------- Environment Setup --------------------
module purge
module load python/pytorch_251_311_cu124

source "$HOME/envs/llm-env/bin/activate"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job-outputs

echo "============================================================================"
echo "INDIGO TRAINING RUN - Job ${SLURM_JOB_ID}"
echo "============================================================================"
echo "PWD:      $(pwd)"
echo "Node:     $(hostname)"
echo "Python:   $(which python)"
echo "Started:  $(date)"
echo

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ============================================================================
# HYPERPARAMETERS - Organized by Category
# ============================================================================

# -------------------- Data & Training Control --------------------
DATA_DIR="${DATA_DIR:-}"                   # Path to a parquet-shards directory;
                                           # empty -> training.py defaults to
                                           # <repo>/data/train
SPLIT="${SPLIT:-train}"                    # Dataset split (train/validation)
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"       # Limit to N examples (for testing)
LIMIT_VAL_EXAMPLES="${LIMIT_VAL_EXAMPLES:-5000}"  # Examples per val-loss (CE) eval.
                                           # 5000 = the full 0.05% val split.
                                           # 0 disables CE val entirely.
LIMIT_DE_EXAMPLES="${LIMIT_DE_EXAMPLES:-256}"     # Examples per DeltaE_00 eval.
                                           # DeltaE is the metric that matters
                                           # (CE/DeltaE decoupling is verified
                                           # on INDIGO), so this is on by
                                           # default. Costs ~50ms/example of
                                           # optical sim plus the decode: a few
                                           # percent at the default cadence.
                                           # 0 disables.
DE_EVERY="${DE_EVERY:-0}"                  # DeltaE cadence in steps. 0 = every
                                           # save tick (SAVE_EVERY). Use a
                                           # multiple of SAVE_EVERY on short
                                           # runs where the eval would
                                           # otherwise dominate wall time.
DE_SAMPLE="${DE_SAMPLE:-0}"                # 1 = temperature-sample the DeltaE
                                           # eval; 0 (default) = greedy, so the
                                           # metric is deterministic across
                                           # checkpoints. NOTE de_curve.sh
                                           # defaults the OTHER way
                                           # (SAMPLE_PREDICTIONS=1), so its
                                           # curves are on a different scale.
DE_TEMPERATURE="${DE_TEMPERATURE:-1.0}"    # Temperature when DE_SAMPLE=1.
SEED="${SEED:-42}"                         # Random seed

# -------------------- Model Architecture --------------------
FEATURE_MODE="${FEATURE_MODE:-raw_spectrum}"  # 'raw_spectrum' or 'compact'
ENCODER_HIDDEN="${ENCODER_HIDDEN:-128}"       # Material encoder hidden dim
ENCODER_OUT="${ENCODER_OUT:-64}"              # Material encoder output dim
ENCODER_DROPOUT="${ENCODER_DROPOUT:-0.1}"     # Material encoder dropout
D_MODEL="${D_MODEL:-1024}"                    # Backbone hidden dim
N_LAYERS="${N_LAYERS:-8}"                     # Number of backbone hidden layers
DROPOUT="${DROPOUT:-0.1}"                     # Backbone dropout
HEAD_MODE="${HEAD_MODE:-cross_attn}"          # PRODUCTION default: 'cross_attn'
                                              # (pointer head: per-slot transformer
                                              # + query cross-attention; permutation-
                                              # equivariant by construction). Matches
                                              # the last shipped prod checkpoint tag
                                              # flex_..._cross_attnH8_se4_dec1.
                                              # Override to 'mlp' for the ablation
                                              # baseline. Tune LR per head — optimum
                                              # differs across architectures.
N_HEADS="${N_HEADS:-8}"                       # Attention heads (cross_attn only)

# Cross-attn-specific depth knobs. Slot encoder of 4 is the recommended
# default — 8-layer self-attn over ≤32 set elements is overkill. Decoder
# stays at 1 layer (causal self-attn + cross-attn + FFN).
if [[ "${HEAD_MODE}" == "cross_attn" ]]; then
  SLOT_ENCODER_LAYERS="${SLOT_ENCODER_LAYERS:-4}"
else
  SLOT_ENCODER_LAYERS="${SLOT_ENCODER_LAYERS:-0}"  # 0 = use N_LAYERS (no-op for MLP)
fi
DECODER_LAYERS="${DECODER_LAYERS:-1}"

# -------------------- Performance --------------------
BF16="${BF16:-1}"                             # bf16 autocast (~1.8-2x on L40s/H100,
                                              # same dynamic range as fp32 → no
                                              # GradScaler needed). Set to 0 for
                                              # bit-identical fp32 reference runs.
PACKED_TF="${PACKED_TF:-}"                    # Packed teacher-forcing collate.
                                              # Empty = auto (on for cross_attn,
                                              # off for mlp). 0 = force off, 1 =
                                              # force on. cross_attn benefits ~5x
                                              # because the slot encoder runs once
                                              # per example instead of once per
                                              # decoding step.
COMPILE="${COMPILE:-0}"                       # torch.compile(model). First batch
                                              # is slow to trace; subsequent ~1.3x.

# -------------------- Optimization --------------------
LR="${LR:-6e-5}"                              # PRODUCTION default LR for the
                                              # cross_attn head. If you flip
                                              # HEAD_MODE=mlp, bump this to ~1.44e-3
                                              # (the LR tuning previously ran).
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"          # AdamW weight decay
GRAD_CLIP="${GRAD_CLIP:-1.0}"                 # Gradient clipping norm
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"    # Warmup fraction
EPOCHS="${EPOCHS:-10}"                        # PRODUCTION default (matches last
                                              # shipped checkpoint's ep=10 tag).

# -------------------- Data Loading --------------------
BATCH_SIZE="${BATCH_SIZE:-256}"                # Batch size
NUM_WORKERS="${NUM_WORKERS:-6}"               # DataLoader workers. 6 was
                                              # OOM'ing previously at
                                              # prefetch_factor=2 (PyTorch
                                              # default); now with PREFETCH=1
                                              # below each worker's prefetch
                                              # queue is ~half the memory, so
                                              # 6 fits at --mem=64G. If OOMs
                                              # return, drop to 4 OR bump --mem.
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"        # DataLoader prefetch_factor.
                                              # Default 1 (was PyTorch default 2).
                                              # Pipeline is producer-bound — GPU
                                              # is ~10x faster than workers — so
                                              # a deeper queue buys no throughput
                                              # and only costs memory. Bump
                                              # only if model/batch ever grow
                                              # enough to make the GPU the
                                              # bottleneck.
STREAMING="${STREAMING:-1}"                    # Dataset streaming mode.
                                              # 1 (default): each worker reads
                                              # shards one at a time, yields
                                              # rows directly — bounded memory
                                              # (~140 MB/worker for the current
                                              # parquet table). 0: legacy
                                              # global-order mode that buffers
                                              # the worker's full epoch in
                                              # memory (~40KB/example × millions
                                              # → OOMs at production scale).

# -------------------- Checkpointing --------------------
SAVE_DIR="${SAVE_DIR:-}"                   # Override checkpoint dir (default: auto-generated)
SAVE_EVERY="${SAVE_EVERY:-1000}"           # Save checkpoint every N steps
RESUME="${RESUME:-}"                       # Path to a checkpoint subdir to
                                           # resume from (e.g. .../step_91000/
                                           # or .../latest/). Loads model +
                                           # optimizer state + step; LR
                                           # schedule continues from there.
                                           # All other env vars MUST match the
                                           # original run.

# -------------------- Logging --------------------
LOG_EVERY="${LOG_EVERY:-100}"              # Print per-step loss every N steps
                                           # (python defaults to --verbose; pass
                                           # --no-verbose to silence)

# ============================================================================
# COMMAND LINE OVERRIDE HANDLING
# ============================================================================

# All sbatch positional args flow through to training.py.
USER_ARGS=("$@")

# ============================================================================
# BUILD COMMAND
# ============================================================================

ARGS=(
  # Data & training control
  --split "${SPLIT}"
  --seed "${SEED}"

  # Model architecture
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

  # Optimization
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --warmup-fraction "${WARMUP_FRACTION}"
  --epochs "${EPOCHS}"

  # Data loading
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
)

if [[ "${STREAMING}" == "1" ]]; then
  ARGS+=(--streaming)
else
  ARGS+=(--no-streaming)
fi

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
else
  ARGS+=(--no-bf16)
fi

ARGS+=(--limit-val-examples "${LIMIT_VAL_EXAMPLES}")
ARGS+=(--limit-de-examples "${LIMIT_DE_EXAMPLES}")
ARGS+=(--de-every "${DE_EVERY}")
ARGS+=(--de-temperature "${DE_TEMPERATURE}")
if [[ "${DE_SAMPLE}" == "1" ]]; then
  ARGS+=(--de-sample)
else
  ARGS+=(--no-de-sample)
fi

# PACKED_TF tri-state: "" = auto (python picks per head), "1" = force on,
# "0" = force off.
if [[ "${PACKED_TF}" == "1" ]]; then
  ARGS+=(--packed-tf)
elif [[ "${PACKED_TF}" == "0" ]]; then
  ARGS+=(--no-packed-tf)
fi

if [[ "${COMPILE}" == "1" ]]; then
  ARGS+=(--compile)
fi

ARGS+=(

  # Checkpointing
  --save-every "${SAVE_EVERY}"

  # Logging
  --log-every "${LOG_EVERY}"
)

# Add optional data-dir if specified
if [[ -n "${DATA_DIR}" ]]; then
  ARGS+=(--data-dir "${DATA_DIR}")
fi

# Add optional limit-examples if specified
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

# Add optional save-dir if specified
if [[ -n "${SAVE_DIR}" ]]; then
  ARGS+=(--save-dir "${SAVE_DIR}")
fi

if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

# User arguments append after ARGS, so they win for any duplicated flag
# (including --no-verbose to silence the default per-step logging).
CMD=(python scripts/training.py "${ARGS[@]}" "${USER_ARGS[@]}")

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "============================================================================"
echo "TRAINING CONFIGURATION"
echo "============================================================================"
echo
echo "Data:"
echo "  Split:           ${SPLIT}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  Num workers:     ${NUM_WORKERS}"
if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  echo "  Limited to:      ${LIMIT_EXAMPLES} examples"
fi
if [[ -n "${DATA_DIR}" ]]; then
  echo "  Data dir:        ${DATA_DIR}"
fi
echo
echo "Model Architecture:"
echo "  head_mode:       ${HEAD_MODE}"
if [[ "${HEAD_MODE}" == "cross_attn" ]]; then
  echo "  n_heads:         ${N_HEADS}"
  echo "  slot_encoder:    ${SLOT_ENCODER_LAYERS} layer(s)"
  echo "  decoder:         ${DECODER_LAYERS} layer(s)"
fi
echo "  feature mode:    ${FEATURE_MODE}"
echo "  encoder hidden:  ${ENCODER_HIDDEN}"
echo "  encoder out:     ${ENCODER_OUT}"
echo "  d_model:         ${D_MODEL}"
echo "  val (CE) examples:  ${LIMIT_VAL_EXAMPLES}"
echo "  val (dE) examples:  ${LIMIT_DE_EXAMPLES} (every ${DE_EVERY:-save_every} steps, sample=${DE_SAMPLE})"
echo "  n_layers:        ${N_LAYERS}"
echo "  dropout:         ${DROPOUT}"
echo "  Output dim:      3201 (M_MAX=32 * NUM_THICKNESSES=100 + EOS)"
echo
echo "Performance:"
echo "  bf16 autocast:   $([ "${BF16}" = "1" ] && echo on || echo off)"
echo "  packed TF:       ${PACKED_TF:-auto (on for cross_attn, off for mlp)}"
echo "  torch.compile:   $([ "${COMPILE}" = "1" ] && echo on || echo off)"
echo
echo "Optimization:"
echo "  Learning rate:   ${LR}"
echo "  Weight decay:    ${WEIGHT_DECAY}"
echo "  Grad clip:       ${GRAD_CLIP}"
echo "  Warmup:          ${WARMUP_FRACTION} (fraction of training)"
echo "  Epochs:          ${EPOCHS}"
echo
echo "Checkpointing:"
echo "  Save every:      ${SAVE_EVERY} steps"
if [[ -n "${SAVE_DIR}" ]]; then
  echo "  Save dir:        ${SAVE_DIR}"
else
  echo "  Save dir:        (auto-generated from hyperparameters)"
fi
echo
echo "============================================================================"
echo "COMMAND:"
printf '  %q ' "${CMD[@]}"
echo
echo "============================================================================"
echo

# ============================================================================
# RUN TRAINING
# ============================================================================

"${CMD[@]}"

EXIT_CODE=$?

echo
echo "============================================================================"
echo "TRAINING COMPLETE"
echo "============================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Ended:     $(date)"
echo "============================================================================"

# ============================================================================
# POST-TRAIN AUTO-CHAIN
# ============================================================================
# Fire the two curve-building SLURM jobs so the training + ΔE curves show up
# on disk without any manual step. Only runs on a successful exit.
#
# Knobs:
#   AUTO_POST_TRAIN  1 (default) = auto-fire follow-up plots.
#                    0           = skip; run the plot SLURMs by hand.
#
# Prerequisite: SAVE_DIR must be set on the outer sbatch invocation so we
# know where the checkpoints landed. If it's empty (auto-generated path
# case) the auto-chain is skipped with a clear message.
if [[ "${EXIT_CODE}" -eq 0 && "${AUTO_POST_TRAIN:-1}" == "1" ]]; then
    if [[ -n "${SAVE_DIR:-}" && -d "${SAVE_DIR}" ]]; then
        echo
        echo "[post-train] AUTO_POST_TRAIN=1  SAVE_DIR=${SAVE_DIR}"
        # 1. Training-loss curve from history.jsonl (smp, matplotlib only).
        HISTORY_PATH="${SAVE_DIR}/history.jsonl"
        if [[ -f "${HISTORY_PATH}" ]]; then
            echo "[post-train] queueing plot_training_curve on ${HISTORY_PATH}"
            HISTORY="${HISTORY_PATH}" \
                sbatch --export=ALL,HISTORY slurms/plot_training_curve.sh || true
        else
            echo "[post-train] no history.jsonl at ${HISTORY_PATH} — skipping loss plot"
        fi
        # 2. ΔE-vs-step curve (gpu, ~2 h). Uses de_curve.sh's compute+plot
        # pipeline against the same val split the training loop scored.
        echo "[post-train] queueing de_curve on ${SAVE_DIR}"
        CKPT_DIR="${SAVE_DIR}" \
            sbatch --export=ALL,CKPT_DIR slurms/de_curve.sh || true
    else
        echo
        echo "[post-train] SAVE_DIR unset or missing — skipping auto-chain."
        echo "[post-train] Set SAVE_DIR=<dir> on the sbatch call, or set"
        echo "[post-train] AUTO_POST_TRAIN=0 to silence this notice."
    fi
fi

exit ${EXIT_CODE}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. FULL PRODUCTION RUN (recommended settings):
#    sbatch slurms/training.sh --epochs 15
#
# 2. QUICK TEST (small subset):
#    sbatch slurms/training.sh \
#      --limit-examples 10000 --epochs 5
#
# 3. HYPERPARAMETER SWEEP - Learning Rate:
#    for lr in 1e-3 2e-3 5e-3; do
#      sbatch slurms/training.sh --lr $lr --epochs 10
#    done
#
# 4. HYPERPARAMETER SWEEP - Model Size:
#    sbatch slurms/training.sh --d-model 128 --n-layers 3
#    sbatch slurms/training.sh --d-model 256 --n-layers 4
#    sbatch slurms/training.sh --d-model 512 --n-layers 6
#
# 5. LARGER MODEL WITH MORE EPOCHS:
#    sbatch slurms/training.sh \
#      --d-model 512 --n-layers 6 --epochs 20 --lr 1e-3
#
# 6. HIGHER DROPOUT (regularization):
#    sbatch slurms/training.sh --dropout 0.2 --epochs 15
#
# 7. LOG CADENCE (verbose is on by default, every 100 steps):
#    LOG_EVERY=25 sbatch slurms/training.sh --epochs 10
#    Or silence: sbatch slurms/training.sh --no-verbose --epochs 10
#
# 8. ENVIRONMENT VARIABLE OVERRIDE:
#    LR=1e-3 EPOCHS=20 D_MODEL=512 \
#      sbatch slurms/training.sh
#
# 9. CUSTOM DATA DIRECTORY:
#    DATA_DIR=/path/to/data_prompts \
#      sbatch slurms/training.sh --epochs 15
#
# ============================================================================
