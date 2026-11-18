# Production checkpoint: provenance of its hyperparameters

Reconciled 2026-11-18. Weights: `model.pt`, sha256 `4fd024434fcc0c02...`
(full hash in `PRODUCTION.json`), unchanged by this note.

## What the run used

| | value | source |
|---|---|---|
| learning rate | **3.104e-4** | the AdamW rate `lr_tuning.py` trained this trial with (`AdamW(lr=lr)`); `meta.json` and the LR-curve record `results/lr_curves/lr_search_ep1_lim39959808_d352_se6_bs256_b20.999_r5.json` (`optimal_lr`) |
| batch size | **256** | the same LR-curve record (`batch_size`, written from the `--batch-size` the data loader used); `slurms/lr_grid.sh` passes `configs.DEFAULT_BATCH_SIZE` = 256; the run's weights folder `..._bs256_...` |
| architecture | d_model 352, 8 layers, 11 heads, slot encoder 6, decoder 1 | `config.json` and the LR-curve record agree |

## What was wrong

Until 2026-11-18, `config.json` said `learning_rate 4.42e-05`, `batch_size 64`.
Those are `ModelConfig`'s defaults: `lr_tuning.py` built one `ModelConfig` per
cell without either field and saved it beside every trial's weights. The
proof: all five learning-rate runs of this cell (4.85e-6 to 3.104e-4), which
trained to different losses, carry the identical `lr4.42e-05_bs64` tag.
`lr_tuning.py` and `promote_production.py` now write the trial's own values,
and `scripts/check_production_config.py` (with `tests/test_production_config.py`)
fails if `PRODUCTION.json` and `config.json` disagree again.

Stale tag: `flex_raw_spectrum_enc128-64_d352_L8_do0.1_lr4.42e-05_bs64_ep1_cross_attnH11_se6_dec1`
Corrected tag: `flex_raw_spectrum_enc128-64_d352_L8_do0.1_lr0.0003104_bs256_ep1_cross_attnH11_se6_dec1`

Evaluation directories created before the correction embed the stale tag
(`ModelConfig.tag()` builds names from `config.json`), and are kept as they
are because results, analyses and figure scripts reference them: those names
are not evidence of the run's settings; the `model_sha256` each eval records is.
