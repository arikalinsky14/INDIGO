#!/usr/bin/env bash
# ============================================================================
# Rerun the rows that crashed on the sampler's token-3201 bug (fixed Nov 18,
# commit 1113878), so every paper number counts them as real results.
# ============================================================================
#
# Run on a CRC login node (this only calls sbatch):
#
#   bash slurms/rerun_crashed_rows.sh            # submit
#   bash slurms/rerun_crashed_rows.sh --list     # show what it would submit
#
# Seven rows, one GPU job each (~5 min of compute; --time=01:00:00):
#
#   paper headline, random proposals : tier A rows 300, 496; tier B 129, 411, 442
#   paper headline, INDIGO N = 250   : tier B row 173
#   NEUTRON, any-material arm        : row 173 (Great Wall)
#
# Every setting is copied from the crashed run's own summary.json args.
# Row seeds are seed + absolute row index (test_eval.py), so a one-row job
# gives exactly the row the original chunk would have produced.
#
# Each writes into its own rerun_* folder INSIDE the run's folder. Both
# scripts/paper_numbers.py and analyses/neutron/neutron.py rescore read every
# rows.json under the run folder, replace a crashed copy with its rerun, and
# stop on any other duplicate. A job whose folder already holds a rows.json
# is skipped, so running this twice submits nothing new.
#
# After all seven finish, fold them into their chunks (Nov 23; a run must not
# hold a row twice, and sync_results.sh refuses to commit one that does):
#   scripts/merge_reruns.py outputs/paper/strong_model outputs/paper/strong_random outputs/neutron_final/full
#   (from an srun shell on smp, as for the sync), then bash scripts/sync_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LIST=0
[[ "${1:-}" == "--list" ]] && LIST=1

# Inputs must exist before anything is submitted. (The NEUTRON run did not
# record its test folder; data/neutron_final_full is inferred from its
# row_constraints path. Each rerun's target_lab is checked against the
# crashed row's before any paper number is regenerated.)
for p in data/test/tier_a data/test/tier_b data/neutron_final_full/tier_a \
         data/neutron_final_full/tier_a/row_constraints.json \
         data/checkpoints/production/current; do
    if [[ ! -e "${p}" ]]; then
        echo "[ERROR] missing ${p} (from $(pwd)); nothing submitted." >&2
        exit 2
    fi
done

# Settings shared by every row (the paper's strong search setting).
COMMON=(ENSEMBLE_N=250 TOP_K=10 REFINE_ITERS=100 SEED=42 TEMPERATURE=1.0
        TOLERANCE=0.0 WEIGHT_LAMBDA=1.0 MC_SAMPLES=0 PLOT_EXAMPLES=0 LIMIT=1)

submit() {   # submit <output dir> <env assignments...>
    local out="$1"; shift
    if compgen -G "${out}/*/tier_*/rows.json" > /dev/null; then
        echo "[skip] ${out} already has results"
        return
    fi
    echo "[submit] ${out}  $*"
    if [[ ${LIST} -eq 0 ]]; then
        env "${COMMON[@]}" "$@" OUTPUT_DIR="${out}" \
            sbatch --time=01:00:00 --job-name=indigo-rerun slurms/test_eval.sh
    fi
}

# Paper headline runs: test tiers, train split, free layer count.
for spec in "strong_random random a 300" "strong_random random a 496" \
            "strong_random random b 129" "strong_random random b 411" \
            "strong_random random b 442" "strong_model model b 173"; do
    read -r run proposer tier row <<< "${spec}"
    submit "outputs/paper/${run}/rerun_tier${tier}_row$(printf %03d "${row}")" \
        PROPOSER="${proposer}" TIERS="${tier}" ROW_OFFSET="${row}" \
        TEST_DIR=data/test SPLIT=train LAYER_COUNT=0
done

# NEUTRON any-material arm: their 5-layer task, per-row constraints.
submit "outputs/neutron_final/full/rerun_row0173" \
    PROPOSER=model TIERS=a ROW_OFFSET=173 \
    TEST_DIR=data/neutron_final_full SPLIT=all LAYER_COUNT=5 \
    ROW_CONSTRAINTS=data/neutron_final_full/tier_a/row_constraints.json
