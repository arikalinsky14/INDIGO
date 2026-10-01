#!/usr/bin/env bash
#SBATCH --job-name=indigo-wall-model
#SBATCH --output=job-outputs/indigo-wall-model.%j.out
#SBATCH --error=job-outputs/indigo-wall-model.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

#SBATCH --time=00:15:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

# ============================================================================
# Does INDIGO's wall clock track examples, FLOPs, or both?
# ============================================================================
#
# Reads finished sweep runs and fits
#
#     seconds per example = 1 / rate + c * F(N)
#
# against each run's own history.jsonl. No GPU, no training, no model is built:
# this reads JSON and solves a two-parameter least squares. It is on smp rather
# than gpu for that reason, and it needs no torch at all, so it would run on a
# login node too. It is here because CLAUDE.md says analyses go through SLURM,
# and because a 15-minute smp job costs less than finding out otherwise.
#
# What it settles: src/scaling/configs.py:estimate_wall_sec has NO model-size
# term. That assumption sizes every config in the ladder against the wall cap,
# and it is why the service-unit axis compresses the sweep to 1.4x while the
# FLOP axis spans 250x. Forward cost per example spans 164x across the ladder,
# so if the pipeline were GPU-bound throughput would vary by that much; sweep v1
# measured 12x and blamed shared-filesystem contention. This measures it.
#
# Usage
# -----
#   RUNS_ROOT=data/checkpoints/scaling_sweep_12h sbatch slurms/fit_wall_model.sh
#
#   # or against a CSV of passes,d_model,slot_encoder_layers,seconds
#   OBSERVATIONS=my_timings.csv sbatch slurms/fit_wall_model.sh
#
# Each run directory needs history.jsonl with two or more entries carrying
# step and wall_time_utc, plus a checkpoint's config.json. Intervals are taken
# between consecutive checkpoints, so one run yields several observations and
# the startup cost drops out of all of them. Time spent inside a DeltaE eval is
# subtracted, since that is not training.
# ============================================================================

RUNS_ROOT="${RUNS_ROOT:-}"
OBSERVATIONS="${OBSERVATIONS:-}"

if [[ -z "${RUNS_ROOT}" && -z "${OBSERVATIONS}" ]]; then
    echo "set RUNS_ROOT=<dir of finished runs> or OBSERVATIONS=<csv>" >&2
    exit 1
fi

# -------------------- Environment Setup --------------------
# Same two lines every other slurm in this repo uses. An earlier version of
# this script invented a conda activation that does not exist on this cluster
# and swallowed the failure with `|| true`, so the job ran against the system
# python and died on `import torch` after the scheduler had already given it a
# GPU. Failures here are fatal and loud.
if command -v module >/dev/null 2>&1; then
    module purge
    module load python/pytorch_251_311_cu124
fi
if [[ -f "$HOME/envs/llm-env/bin/activate" ]]; then
    source "$HOME/envs/llm-env/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs

if [[ -n "${RUNS_ROOT}" ]]; then
    srun python scripts/fit_wall_model.py --runs-root "${RUNS_ROOT}" \
        ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"}
else
    srun python scripts/fit_wall_model.py --observations "${OBSERVATIONS}" \
        ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"}
fi
