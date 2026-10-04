#!/usr/bin/env bash
#SBATCH --job-name=indigo-analyze
#SBATCH --output=job-outputs/indigo-analyze.%j.out
#SBATCH --error=job-outputs/indigo-analyze.%j.err

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Run any analysis script on a compute node, and bundle what it wrote
# ============================================================================
#
# Python does not run on the login nodes, and every analysis in this repo is a
# CPU job that reads JSON and writes JSON and figures. This is the one wrapper
# for all of them: pass the script and its arguments after the sbatch line,
# exactly as you would type them.
#
#   sbatch slurms/analyze.sh scripts/fit_beta2.py \
#       --results-dir outputs/lr_search/beta2/cross_attn
#
#   sbatch slurms/analyze.sh scripts/fit_lr_law.py \
#       --results-dir outputs/lr_search/cross_attn \
#       --coverage-from analyses/scaling/results/isoflop_fit.json
#
#   sbatch slurms/analyze.sh scripts/collect_isoflop.py --beta2 0.99 --rungs 3
#
# The script's printed report lands in job-outputs/indigo-analyze.<job>.out.
#
# BUNDLE. Every file under analyses/ and outputs/ that the job created or
# changed, plus the job's own .out and .err, is zipped into
# job-outputs/analysis_<job>.zip: one file to download and send on. Add
# INCLUDE=<dir> to put a directory's inputs in too (the beta2 study's result
# JSONs, say), so whoever reads the zip can re-run the analysis from it:
#
#   INCLUDE=outputs/lr_search/beta2/cross_attn \
#       sbatch slurms/analyze.sh scripts/fit_beta2.py \
#       --results-dir outputs/lr_search/beta2/cross_attn
#
# smp, not gpu: nothing here builds a model, and the analysis stack is kept
# importable without torch (tests/ enforce it). An hour covers every analysis
# in the repo; the beta2 fit takes about a minute.
# ============================================================================

if (( $# < 1 )); then
    echo "usage: sbatch slurms/analyze.sh <script.py> [args...]" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs

# -------------------- Environment Setup --------------------
# The cluster's own two lines, as in every slurm here. Failures are fatal.
if command -v module >/dev/null 2>&1; then
    module purge
    module load python/pytorch_251_311_cu124
fi
if [[ -f "$HOME/envs/llm-env/bin/activate" ]]; then
    source "$HOME/envs/llm-env/bin/activate"
fi

SCRIPT="$1"
shift
if [[ ! -f "${SCRIPT}" ]]; then
    echo "no such script: ${SCRIPT} (run sbatch from the INDIGO checkout)" >&2
    exit 1
fi

JOB="${SLURM_JOB_ID:-local}"
MARKER="job-outputs/.analyze_start_${JOB}"
touch "${MARKER}"

echo "=================================================================="
echo " ${SCRIPT} $*"
echo " commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " started $(date)"
echo "=================================================================="

set +e
python -u "${SCRIPT}" "$@"
STATUS=$?
set -e

# -------------------- Bundle --------------------
ZIP="job-outputs/analysis_${JOB}.zip"
python - "${MARKER}" "${ZIP}" "${JOB}" "${INCLUDE:-}" <<'PYEOF'
import os, sys, zipfile
from pathlib import Path

marker, out, job, include = sys.argv[1:5]
since = os.path.getmtime(marker)
files = []
for root in ("analyses", "outputs"):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.stat().st_mtime >= since:
            files.append(p)
if include:
    files += [p for p in Path(include).rglob("*") if p.is_file()]
for ext in ("out", "err"):
    log = Path(f"job-outputs/indigo-analyze.{job}.{ext}")
    if log.is_file():
        files.append(log)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        z.write(p)
print(f"\n[INFO] bundled {len(set(files))} file(s) into {out}")
PYEOF
rm -f "${MARKER}"

echo "finished $(date), exit ${STATUS}"
exit "${STATUS}"
