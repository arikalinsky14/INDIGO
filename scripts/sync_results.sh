#!/usr/bin/env bash
# Put the final results into git, without the weights. Run on CRC, from the
# repo, on the branch you want them on:
#
#   bash scripts/sync_results.sh            # copy, commit and push
#   DRY_RUN=1 bash scripts/sync_results.sh  # list what would be copied, change nothing
#   NO_PUSH=1 bash scripts/sync_results.sh  # commit only
#
# outputs/ and data/ are git-ignored (weights, shards, caches), so the result
# files under them are mirrored into results/, which is tracked:
#
#   outputs/prod_eval/               -> results/prod_eval/        production evals and figures
#   outputs/lr_curves/cross_attn/    -> results/lr_curves/        learning-rate curves (prod included)
#   outputs/lr_search/               -> results/lr_search/        the scaling study's tuning runs
#   outputs/isoflop_confirm/cross_attn/ -> results/isoflop_confirm/
#   data/checkpoints/production/     -> results/production/       PRODUCTION.json, config, meta; no model.pt
#
# and analyses/*/results/ (already tracked) is added in place. Only result
# types are copied (json, csv, png, pdf, txt, md, docx), each at most
# MAX_FILE_MB (default 10); weights/ folders, *.pt and *.parquet never are.
# Safe to re-run: unchanged files are not recopied, and nothing is deleted
# from results/ when a source file goes away.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MAX_FILE_MB="${MAX_FILE_MB:-10}"
MAX_TOTAL_MB="${MAX_TOTAL_MB:-300}"
PATTERNS=(--include='*/' --include='*.json' --include='*.csv' --include='*.png'
          --include='*.pdf' --include='*.txt' --include='*.md' --include='*.docx'
          --exclude='*')
SKIP=(--exclude='weights/' --exclude='*.pt' --exclude='*.parquet' --exclude='cache/'
      --exclude='.*.tmp/')

pairs=(
  "outputs/prod_eval/:results/prod_eval/"
  "outputs/lr_curves/cross_attn/:results/lr_curves/"
  "outputs/lr_search/:results/lr_search/"
  "outputs/isoflop_confirm/cross_attn/:results/isoflop_confirm/"
  "data/checkpoints/production/:results/production/"
)

RSYNC=(rsync -a --prune-empty-dirs --max-size="${MAX_FILE_MB}m" "${SKIP[@]}" "${PATTERNS[@]}")
[[ "${DRY_RUN:-0}" == "1" ]] && RSYNC+=(--dry-run --out-format='  %n (%l bytes)')

for pair in "${pairs[@]}"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    if [[ ! -d "${src}" ]]; then
        echo "[skip] ${src} (not here)"
        continue
    fi
    echo "[sync] ${src} -> ${dst}"
    mkdir -p "${dst}"
    # -L follows production/current and copies the version folder's files.
    "${RSYNC[@]}" -L "${src}" "${dst}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry run] nothing copied or committed."
    exit 0
fi

# Files over the cap were skipped by rsync; say which, so nothing goes missing silently.
for pair in "${pairs[@]}"; do
    src="${pair%%:*}"
    [[ -d "${src}" ]] && find -L "${src}" -type f -size +"${MAX_FILE_MB}"M \
        \( -name '*.json' -o -name '*.csv' -o -name '*.png' -o -name '*.pdf' \) \
        -not -path '*/weights/*' -printf "[not copied, over ${MAX_FILE_MB} MB] %p\n" || true
done

TOTAL_MB=$(du -sm results | cut -f1)
echo "[size] results/ is ${TOTAL_MB} MB"
if (( TOTAL_MB > MAX_TOTAL_MB )); then
    echo "[ERROR] over MAX_TOTAL_MB=${MAX_TOTAL_MB}; nothing committed. Look at" >&2
    echo "        du -sh results/* and raise the cap or narrow the sources." >&2
    exit 1
fi

shopt -s nullglob
ANALYSES=(analyses/*/results)
shopt -u nullglob
git add results "${ANALYSES[@]}"
if git diff --cached --quiet -- results analyses; then
    echo "[git] no new results to commit."
    exit 0
fi
git diff --cached --stat -- results analyses | tail -3
git commit -q -m "Results from CRC ($(date +%Y-%m-%d)): eval outputs, LR curves, scaling runs, production metadata" \
    -- results analyses
echo "[git] committed $(git rev-parse --short HEAD)"

if [[ "${NO_PUSH:-0}" == "1" ]]; then
    echo "[git] NO_PUSH=1: not pushed."
    exit 0
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git pull -q --no-rebase origin "${BRANCH}"
git push -q origin "${BRANCH}"
echo "[git] pushed to origin/${BRANCH}"
