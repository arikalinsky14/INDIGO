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
# Needs only bash, find and cp. Safe to re-run: unchanged files are not recopied, and nothing is deleted
# from results/ when a source file goes away.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MAX_FILE_MB="${MAX_FILE_MB:-10}"
MAX_TOTAL_MB="${MAX_TOTAL_MB:-300}"
MAX_KB=$(( MAX_FILE_MB * 1024 ))
# Plain find + cp (CRC's login nodes have no rsync). -L follows symlinks, so
# production/current is copied as the version folder's files.
TYPES=( \( -name '*.json' -o -name '*.csv' -o -name '*.png' -o -name '*.pdf'
        -o -name '*.txt' -o -name '*.md' -o -name '*.docx' \) )
PRUNE=( \( -name weights -o -name cache -o -name '.*.tmp' \) -prune )

pairs=(
  "outputs/prod_eval/:results/prod_eval/"
  "outputs/lr_curves/cross_attn/:results/lr_curves/"
  "outputs/lr_search/:results/lr_search/"
  "outputs/isoflop_confirm/cross_attn/:results/isoflop_confirm/"
  "data/checkpoints/production/:results/production/"
)

REPO="$(pwd)"
n_copy=0
for pair in "${pairs[@]}"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    if [[ ! -d "${src}" ]]; then
        echo "[skip] ${src} (not here)"
        continue
    fi
    t0=$SECONDS
    # One find, then cp in batches (xargs): a process per file is slow on /ix1.
    # cp -u copies only files that are new or newer than the copy already in
    # results/, and -p keeps the source's mtime so the next run can tell.
    list="$(mktemp)"
    (cd "${src}" && find -L . -mindepth 1 -type d "${PRUNE[@]}" -o -type f "${TYPES[@]}" \
        -size -"${MAX_KB}"k -print0) > "${list}"
    n_found=$(tr -cd '\0' < "${list}" | wc -c)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[sync] ${src} -> ${dst}: ${n_found} result file(s) found"
        tr '\0' '\n' < "${list}" | head -5 | sed 's/^/    /'
        (( n_found > 5 )) && echo "    ..."
    else
        mkdir -p "${dst}"
        n=$( (cd "${src}" && xargs -0 -r cp -pLuv --parents -t "${REPO}/${dst}" < "${list}") \
             | grep -cE "\.(json|csv|png|pdf|txt|md|docx)' -> " || true)
        n_copy=$(( n_copy + n ))
        echo "[sync] ${src} -> ${dst}: ${n} of ${n_found} copied ($(( SECONDS - t0 )) s)"
    fi
    rm -f "${list}"
    # Result files over the cap are skipped; say which, so none goes missing silently.
    find -L "${src}" -mindepth 1 -type d "${PRUNE[@]}" -o -type f "${TYPES[@]}" \
        -size +"$(( MAX_KB - 1 ))"k -printf "    [not copied, over ${MAX_FILE_MB} MB] %p\n"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry run] nothing copied or committed."
    exit 0
fi
echo "[sync] ${n_copy} file(s) copied in total"

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
    -- results "${ANALYSES[@]}"
echo "[git] committed $(git rev-parse --short HEAD)"

if [[ "${NO_PUSH:-0}" == "1" ]]; then
    echo "[git] NO_PUSH=1: not pushed."
    exit 0
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
# Take in whatever reached the branch meanwhile (a merge, never a rebase:
# others may have the branch checked out), then push.
if git ls-remote --exit-code --heads origin "${BRANCH}" >/dev/null 2>&1; then
    git pull -q --no-rebase --no-edit origin "${BRANCH}"
fi
git push -q -u origin "${BRANCH}"
echo "[git] pushed to origin/${BRANCH}"
