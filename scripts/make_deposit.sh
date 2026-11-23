#!/usr/bin/env bash
# Build the files for the paper's Zenodo deposit. Run on CRC from the repo, on
# a compute node (it hashes gigabytes):
#
#   srun -M smp -p smp -n1 -c2 --mem=8G -t 4:00:00 bash scripts/make_deposit.sh
#   HASH_CORPUS=1 ... bash scripts/make_deposit.sh    # also sha256 every training shard (long)
#
# Writes deposit/<YYYY-MM-DD>/ (git-ignored):
#   indigo-code-<commit>.tar.gz     git archive of HEAD (the code, results/, analyses/)
#   indigo-checkpoint.tar.gz        data/checkpoints/production/current, files not
#                                   the symlink: model.pt, config.json, meta.json,
#                                   PRODUCTION.json, PROVENANCE.md
#   indigo-test-tiers.tar.gz        data/test/tier_a and tier_b, shards and manifests
#   indigo-corpus-manifest.json     the training corpus as a recipe: every shard's
#                                   generator manifest (create_dataset/src/
#                                   compile_datasets.py writes one per shard: all
#                                   arguments, the six seeds, the code commit), its
#                                   row count and size, and with HASH_CORPUS=1 its
#                                   sha256. The corpus (~1.4 TB) is far over Zenodo's
#                                   per-record limit, so the deposit carries the
#                                   generator (in the code archive) and this file.
#   SHA256SUMS                      for everything above
#   DEPOSIT.md                      what each file is, for the Zenodo description
#
# It uploads nothing: create the Zenodo record, upload the files, then put the
# DOI in the paper's \ph{} placeholders.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

COMMIT="$(git rev-parse --short HEAD)"
if ! git diff --quiet HEAD -- . ':!results' ':!job-outputs'; then
    echo "[deposit] uncommitted code changes; commit first so the archive is a real commit" >&2
    exit 1
fi
OUT="deposit/$(date +%Y-%m-%d)"
mkdir -p "${OUT}"
echo "[deposit] ${OUT}, commit ${COMMIT}"

python3 scripts/check_production_config.py
python3 scripts/merge_reruns.py --check results/paper_runs results/neutron_final_runs

git archive --format=tar.gz --prefix="indigo-${COMMIT}/" -o "${OUT}/indigo-code-${COMMIT}.tar.gz" HEAD
echo "[deposit] code archive done"

CK=data/checkpoints/production/current
tar -czhf "${OUT}/indigo-checkpoint.tar.gz" \
    --transform "s|^${CK}|indigo-checkpoint|" \
    --transform "s|^results/production/current|indigo-checkpoint|" \
    "${CK}/model.pt" "${CK}/config.json" "${CK}/meta.json" "${CK}/PRODUCTION.json" \
    results/production/current/PROVENANCE.md
echo "[deposit] checkpoint done"

tar -czf "${OUT}/indigo-test-tiers.tar.gz" data/test/tier_a data/test/tier_b
echo "[deposit] test tiers done"

HASH_CORPUS="${HASH_CORPUS:-0}" python3 - "${OUT}/indigo-corpus-manifest.json" "${COMMIT}" <<'PYEOF'
import glob, hashlib, json, os, sys
from pathlib import Path
out, commit = sys.argv[1], sys.argv[2]
hash_it = os.environ.get("HASH_CORPUS") == "1"
shards = []
for pq in sorted(glob.glob("data/train/**/shard_*.parquet", recursive=True)):
    man = Path(pq).with_suffix(".manifest.json")
    ent = {"path": pq, "bytes": os.path.getsize(pq),
           "manifest": json.load(open(man)) if man.is_file() else None}
    if hash_it:
        h = hashlib.sha256()
        with open(pq, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        ent["sha256"] = h.hexdigest()
    shards.append(ent)
missing = sum(1 for s in shards if s["manifest"] is None)
json.dump({"generator": "create_dataset/src/compile_datasets.py (in the code archive)",
           "deposit_commit": commit, "n_shards": len(shards),
           "n_rows": sum((s["manifest"] or {}).get("rows_written", 0) for s in shards),
           "shards_without_manifest": missing, "sha256_included": hash_it,
           "shards": shards}, open(out, "w"), indent=1)
print(f"[deposit] corpus manifest: {len(shards)} shards, {missing} without a manifest")
PYEOF

cat > "${OUT}/DEPOSIT.md" <<EOF
# INDIGO deposit (commit ${COMMIT})

| file | contents |
|---|---|
| indigo-code-${COMMIT}.tar.gz | code, tracked results and analyses at commit ${COMMIT} |
| indigo-checkpoint.tar.gz | production model (d352/se6, 11,314,405 parameters, sha256 $(python3 -c "import json;print(json.load(open('results/production/current/PRODUCTION.json'))['model_sha256'][:16])")); PROVENANCE.md explains its hyperparameter record |
| indigo-test-tiers.tar.gz | test tiers A and B: parquet shards and generator manifests |
| indigo-corpus-manifest.json | the 40M-row training corpus as a recipe: each shard's generator arguments and seeds (regenerate with create_dataset/src/compile_datasets.py) |
| SHA256SUMS | checksums of the above |

Demo: inference/frontend/ in the code archive; its Paper preset is the paper's search
(check with scripts/check_ui_matches_solve.py).
EOF

(cd "${OUT}" && sha256sum indigo-* DEPOSIT.md > SHA256SUMS)
ls -lh "${OUT}"
echo "[deposit] done. Upload ${OUT}/* to Zenodo; the archive is commit ${COMMIT}."
