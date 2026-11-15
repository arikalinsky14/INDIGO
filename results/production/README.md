# Production model

`current/` is the INDIGO production model. Load it like any checkpoint:

    CHECKPOINT=data/checkpoints/production/current sbatch slurms/gamut_eval.sh

Every other folder here is one promoted model, never overwritten. Its
`PRODUCTION.json` says which training run it was copied from, the rule that
chose it, its metrics and the sha256 of `model.pt` (the first 16 characters are
the `model_sha256` every eval output records).

Written by scripts/promote_production.py; see that script to promote another.
