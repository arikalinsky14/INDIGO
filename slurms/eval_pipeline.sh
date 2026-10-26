#!/usr/bin/env bash
#SBATCH --job-name=indigo-eval-pipe
#SBATCH --output=job-outputs/indigo-eval-pipe.%A_%a.out
#SBATCH --error=job-outputs/indigo-eval-pipe.%A_%a.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# The third metric: DeltaE after a LIGHT inference pipeline, per saved model
# ============================================================================
#
# One array task per checkpoint that STAGE=confirm saved. Each runs
# inference/scripts/test_eval.py on the held-out test tiers with a light
# version of the production search:
#
#   ROW_OFFSET=0      rows [ROW_OFFSET, ROW_OFFSET + LIMIT) of each tier. 0 is
#                     the test set every figure reports; ROW_OFFSET=500 with
#                     its own PIPE_DIR is a dev set for choosing a rate.
#   ENSEMBLE_N=16     sample 16 stacks per target at TEMPERATURE=1.0
#   REFINE_ITERS=5    a very short thickness refinement on the best one
#   top-k 1, no Monte-Carlo robustness, tolerance 0
#
# Small N on purpose: at N=200 the ensemble was nearly model-agnostic
# (CLAUDE.md, Sept 15), which would flatten exactly the differences between
# model sizes a scaling curve needs. Output lands beside the weights' layout,
# <confirm dir>/pipeline/<cell tag>/lr<rate>/, which collect_confirm.py reads.
#
#   bash slurms/eval_pipeline.sh --list              # freezes the checkpoint list
#   sbatch --array=0-<n-1>%8 slurms/eval_pipeline.sh
#   # one checkpoint first, to time it:
#   sbatch --array=0 slurms/eval_pipeline.sh
#
# Done checkpoints are skipped, so a re-submitted array only runs what is
# missing. The physics runs on CPU (JAX_PLATFORMS=cpu); the GPU holds the model.
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p job-outputs

CONFIRM_DIR="${CONFIRM_DIR:-outputs/isoflop_confirm/cross_attn}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${CONFIRM_DIR}/weights}"
PIPE_DIR="${PIPE_DIR:-${CONFIRM_DIR}/pipeline}"
# A LIST given explicitly must exist: falling back to every checkpoint under
# WEIGHTS_DIR would score the wrong models (Oct 26: a missing production list
# scored the study models that sort first).
LIST_GIVEN="${LIST:+1}"
# Without a LIST, the checkpoints are every model.pt under WEIGHTS_DIR, sorted;
# CKPT_MATCH keeps only paths containing it, CKPT_EXCLUDE drops paths
# containing it. Each task derives the same list on its compute node, which
# avoids building a list on a login node that may not see the same files.
list_checkpoints() {
    find "${WEIGHTS_DIR}" -name model.pt -printf '%h\n' 2>/dev/null | sort \
        | { if [[ -n "${CKPT_MATCH:-}" ]]; then grep -F -- "${CKPT_MATCH}"; else cat; fi; } \
        | { if [[ -n "${CKPT_EXCLUDE:-}" ]]; then grep -vF -- "${CKPT_EXCLUDE}"; else cat; fi; }
}
LIST="${LIST:-${CONFIRM_DIR}/pipeline_checkpoints.txt}"

if [[ "${1:-}" == "--list" ]]; then
    if [[ ! -d "${WEIGHTS_DIR}" ]]; then
        echo "[ERROR] no ${WEIGHTS_DIR} under $(pwd); run STAGE=confirm first" >&2
        exit 1
    fi
    list_checkpoints > "${LIST}"
    N="$(grep -c . "${LIST}" || true)"
    echo "${N} checkpoint(s) listed in ${LIST}"
    (( N > 0 )) && echo "submit with: sbatch --array=0-$(( N - 1 ))%8 slurms/eval_pipeline.sh" \
        && echo "  or split by tier on CPU: SPLIT_TIERS=1 sbatch --array=0-$(( 2 * N - 1 )) slurms/eval_pipeline_cpu.sh"
    exit 0
fi

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export TOKENIZERS_PARALLELISM=false

TASK="${SLURM_ARRAY_TASK_ID:-0}"
# SPLIT_TIERS=1: two tasks per checkpoint, one per test tier (task 2k is
# checkpoint k's tier a, 2k+1 its tier b), so each finishes in half the time.
# Both tiers land in the same output folder, which collect_confirm.py reads.
if [[ "${SPLIT_TIERS:-0}" == 1 ]]; then
    TIERS=$([[ $(( TASK % 2 )) == 0 ]] && echo a || echo b)
    TASK=$(( TASK / 2 ))
fi
if [[ -s "${LIST}" ]]; then
    CKPT="$(sed -n "$(( TASK + 1 ))p" "${LIST}")"
elif [[ -n "${LIST_GIVEN}" ]]; then
    echo "[ERROR] LIST=${LIST} is missing or empty; not falling back to every" \
         "checkpoint under ${WEIGHTS_DIR}. Build the list first." >&2
    exit 1
else
    # No frozen list: derive the same sorted list on the fly. Stable as long
    # as no new weights land while the array runs.
    echo "[WARN] ${LIST} missing; listing ${WEIGHTS_DIR} directly"
    CKPT="$(list_checkpoints | sed -n "$(( TASK + 1 ))p")"
    echo "[INFO] checkpoint $(( TASK + 1 )) of $(list_checkpoints | grep -c .)" \
         "(match '${CKPT_MATCH:-}', exclude '${CKPT_EXCLUDE:-}')"
fi
if [[ -z "${CKPT}" ]]; then
    echo "[ERROR] no checkpoint at line $(( TASK + 1 )) of ${LIST}" >&2
    exit 1
fi
REL="${CKPT#${WEIGHTS_DIR}/}"
OUT="${PIPE_DIR}/${REL}"

DONE=1
for t in ${TIERS:-a b}; do
    compgen -G "${OUT}/*/tier_${t}/rows.json" > /dev/null || DONE=0
done
if (( DONE )); then
    echo "[INFO] ${OUT} already scored (tiers ${TIERS:-a b}); nothing to do"
    exit 0
fi

echo "== checkpoint ${CKPT}"
echo "   -> ${OUT}"
echo "   ensemble ${ENSEMBLE_N:-16} at T=${TEMPERATURE:-1.0}, refine ${REFINE_ITERS:-5} steps, ${LIMIT:-500} rows per tier"
mkdir -p "${OUT}"
python -u inference/scripts/test_eval.py \
    --checkpoint "${CKPT}" \
    --test-dir "${TEST_DIR:-data/test}" --tiers ${TIERS:-a b} \
    --limit "${LIMIT:-500}" --offset "${ROW_OFFSET:-0}" --plot-examples 0 \
    --ensemble-n "${ENSEMBLE_N:-16}" --temperature "${TEMPERATURE:-1.0}" \
    --tolerance 0 --top-k 1 --refine-iters "${REFINE_ITERS:-5}" \
    --mc-samples 0 --seed 42 \
    --output-dir "${OUT}"
