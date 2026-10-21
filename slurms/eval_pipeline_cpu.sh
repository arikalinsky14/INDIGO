#!/usr/bin/env bash
#SBATCH --job-name=indigo-eval-pipe-cpu
#SBATCH --output=job-outputs/indigo-eval-pipe-cpu.%A_%a.out
#SBATCH --error=job-outputs/indigo-eval-pipe-cpu.%A_%a.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

# slurms/eval_pipeline.sh on the CPU cluster: the light pipeline is mostly the
# CPU-side physics simulation, and the models are small, so a GPU adds little
# while the GPU queue is the bottleneck. Same environment variables (CONFIRM_DIR,
# SPLIT_TIERS, LIMIT, ...), same outputs, same skip rule.
#
#   CONFIRM_DIR=outputs/lr_curves/cross_attn SPLIT_TIERS=1 \
#       sbatch --array=12-49 slurms/eval_pipeline_cpu.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
exec bash slurms/eval_pipeline.sh "$@"
