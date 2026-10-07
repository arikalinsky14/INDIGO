#!/usr/bin/env bash
#SBATCH --job-name=indigo-chain3
#SBATCH --output=job-outputs/indigo-chain3.%j.out
#SBATCH --error=job-outputs/indigo-chain3.%j.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================================
# Stage 2 finished -> learning-rate law -> stage 3 + check -> final figure
# ============================================================================
#
# Submit it to wait on the last stage-2 array (same cluster, so the
# dependency holds):
#
#   sbatch --dependency=afterany:<stage-2 array job id> slurms/chain_stage3.sh
#
# When that array is done it
#   1. fits lr(N, D) on stage 2 (scripts/fit_lr_law.py, which writes
#      analyses/scaling/results/lr_law_fit.json, the law stage 3 reads) with
#      the leave-one-rung-out check;
#   2. GATES on that check: if the law, fitted without the top tuned rung,
#      misses that rung's tuned optima by more than MAX_TOP_RATIO (default
#      1.85, one stage-2 grid step) at its worst point, it stops here: stage 3
#      would spend its compute on rates the law cannot reach. FORCE_STAGE3=1
#      overrides;
#   3. submits STAGE=3 and STAGE=check (in parallel: the check no longer
#      gates stage 3, to fit a deadline; if it fails, re-run stage 3's
#      upper-rung points);
#   4. submits the final analysis to run once both are done, and once any
#      job ids in AFTER (colon-separated, e.g. a 3seeds or 2pick array) are
#      done too, so their points land in the final figure:
#      scripts/stage2_preview.py --final, figures and fits in
#      analyses/scaling/results/tuned/, zipped into
#      job-outputs/analysis_<job>.zip.
#
# Sits on a GPU node for a minute or two (the dependency has to stay on the
# gpu cluster); nothing here uses the GPU.
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$HOME/Github/INDIGO}"
mkdir -p job-outputs

module purge
module load python/pytorch_251_311_cu124
source "$HOME/envs/llm-env/bin/activate"

echo "== 1. learning-rate law from stage 2"
python -u scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
    --coverage-from analyses/scaling/results/isoflop_fit.json

echo "== 2. extrapolation gate"
set +e
python - "${MAX_TOP_RATIO:-1.85}" <<'PYEOF'
import json, math, sys
limit = float(sys.argv[1])
law = json.load(open("analyses/scaling/results/lr_law_fit.json"))
if law.get("lr_vs_n_and_d", {}).get("status") != "ok":
    print(f"[GATE] no lr(N, D) law: {law.get('lr_vs_n_and_d')}"); sys.exit(2)
top = [r for r in law.get("leave_one_rung_out", {}).get("rungs", [])
       if str(r.get("held_out", "")).startswith("top")]
if not top or "worst_ratio" not in top[0]:
    print("[GATE] no leave-one-rung-out result for the top rung"); sys.exit(2)
w = top[0]["worst_ratio"]
ok = abs(math.log(w)) <= math.log(limit)
print(f"[GATE] top tuned rung held out: worst point {w:.2f}x, median "
      f"{top[0]['median_ratio']:.2f}x (limit {limit:g}x): "
      + ("pass" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PYEOF
GATE=$?
set -e
if (( GATE != 0 )) && [[ "${FORCE_STAGE3:-0}" != "1" ]]; then
    echo "[STOP] the law does not reach its own top rung; stage 3 not submitted."
    echo "       Look at the leave-one-rung-out lines above. FORCE_STAGE3=1 to run anyway."
    exit 1
fi

count() {   # tasks a stage would submit
    STAGE="$1" bash slurms/lr_grid.sh --list | sed -n 's/.*--array=0-\([0-9]*\)%.*/\1/p' | tail -1
}

echo "== 3. submitting stage 3 and the check"
N3="$(count 3)"; NC="$(count check)"
J3="$(STAGE=3 sbatch --parsable --array=0-${N3}%${THROTTLE:-8} --time=10:00:00 slurms/lr_grid.sh)"
JC="$(STAGE=check sbatch --parsable --array=0-${NC}%${THROTTLE:-8} --time=10:00:00 slurms/lr_grid.sh)"
J3="${J3%%;*}"; JC="${JC%%;*}"
echo "   stage 3: job ${J3} (tasks 0-${N3});  check: job ${JC} (tasks 0-${NC})"

echo "== 4. final analysis after both"
JF="$(sbatch --parsable --clusters=gpu --partition=l40s --gres=gpu:1 \
      --dependency=afterany:${J3}:${JC}${AFTER:+:${AFTER}} slurms/analyze.sh \
      scripts/stage2_preview.py --final)"
echo "   final analysis: job ${JF%%;*} -> job-outputs/analysis_${JF%%;*}.zip"
