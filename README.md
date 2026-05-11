# INDIGO: Flexible-Material RGB → Structure

INDIGO adapts CHROMA-Lite for a flexible-material thin-film design task. The
model takes a user-supplied pool of materials (specified by their n,k
refractive-index spectra) and selects which of them to use in a multilayer
structure to hit a target sRGB color. There are no preset materials and no
constraint awareness in the model — constraints are filtered post-hoc.

## What's in here

```
INDIGO/
├── src/                                 # shared model + utility code
│   ├── material_features.py             # JLL CSV loader, canonical grid, featurizer
│   ├── synthetic_materials.py           # 3 strategies for synthetic n,k generation
│   ├── materials_vocab.py               # Slot-indexed token encoding (M_MAX × NUM_THICKNESSES + EOS)
│   ├── optical_sim.py                   # Optical simulator (accepts arbitrary n,k)
│   ├── dataset.py                       # FlexThinFilmDataset (streaming parquet reader)
│   └── model.py                         # FlexMaterialMLP (shared encoder + masked output)
│
├── create_dataset/
│   ├── src/
│   │   ├── pool_sampler.py              # Build per-row material pools with slot-shuffling
│   │   ├── random_layer.py              # Sample structures from held-in real + synthetic
│   │   └── compile_datasets.py          # Drive end-to-end parquet generation
│   └── data_prompts/                    # Output dir, gitignored
│
├── scripts/
│   ├── training.py                      # Training loop (AdamW + 2% warmup + cosine decay)
│   ├── evaluate.py                      # Teacher-forcing + autoregressive eval, CIEDE2000
│   ├── lr_tuning.py                     # LR grid search (re-run for new model size)
│   ├── plot_training_curves.py          # Train/val loss curves across checkpoints
│   ├── visualize_synthetic.py           # Sanity-check synthetic material distributions
│   └── verify_optical_sim.py            # Cross-check new vs original optical sim
│
├── slurms/
│   ├── training.sh
│   ├── evaluate.sh
│   └── lr_tuning.sh
│
└── data/                                # gitignored, auto-created
    └── checkpoints/
```

All modules use **absolute imports from `src.*`** per project convention.

## How the pieces fit together

```
                  user provides
                  ┌──────────────────┐
                  │  Material pool   │  list of (name, n[128], k[128])
                  │  (≤ M_MAX = 32)  │  on the canonical wavelength grid
                  └────────┬─────────┘
                           │ featurize_pool + pad_pool_features
                           ▼
              ┌──────────────────────────┐
              │ pool_features [M_MAX,2,L]│
              │ pool_mask     [M_MAX]    │
              └──────────┬───────────────┘
                         │
       target RGB        │     structure-so-far
       [3]               ▼              [M_MAX, MAX_LAYERS]
        │      MaterialEncoder (shared)        │
        │      → [M_MAX, encoder_out]          │
        │      *= pool_mask                    │
        │                                      │
        └──────────┬──── concat ───────────────┘
                   ▼
          Backbone MLP (RGB-to-Structure-style)
                   │
                   ▼
              logits [VOCAB_SIZE]
                   │
                   │ + output_mask (–∞ for invalid slots)
                   ▼
              softmax → next-token distribution

         token_id = slot_idx * NUM_THICKNESSES + thickness_idx
                   or EOS

         decode using user's pool: pool[slot_idx].name and the thickness
```

`VOCAB_SIZE = M_MAX * NUM_THICKNESSES + 1 = 32 * 40 + 1 = 1281`
(vs. 1001 in original CHROMA-Lite).

## Why this prevents overfitting to specific materials

1. **Shared material encoder** — the encoder MLP is applied to every slot
   with the same weights. The model has no slot-specific capacity, so it
   cannot learn "slot 3 = Ag". Anything it learns about a slot must be a
   function of n, k features.

2. **Slot indices are arbitrary per example** — when slot ordering is
   randomized for the same underlying structure, the model gets the same
   RGB target with completely different "slot 3 = Ag at 100nm" → "slot 7
   = Ag at 100nm" tokenizations. The only feature shared across these is
   the n,k spectrum at the chosen slot. This is the strongest pressure
   toward feature-based generalization.

3. **Synthetic materials in `synthetic_materials.py`** — by generating
   ~80% of training materials from Lorentz oscillators, perturbations of
   real materials, and pair interpolations, the model never sees a
   stable list of distinct material identities to memorize. The JLL
   library becomes a rare special case rather than the dominant training
   signal.

4. **Held-out real materials** — by partitioning the JLL library into
   training and validation halves, you get a direct measurement of
   generalization to materials never seen during training.

5. **Output masking** — invalid slots get `-inf` logits, so the model
   cannot leak probability mass to slots that don't exist for the
   current example. This forces honest behaviour at variable pool sizes.

## Validation strategy (two-tier)

Hold these out from training entirely:
- **Tier A** (seen materials, unseen structures): pools drawn from the
  same distributions as training, but the (RGB, structure) pair is novel.
  Measures generalization to new color targets.
- **Tier B** (unseen materials, unseen structures): pools include held-out
  real materials and a held-out synthetic distribution (e.g. only Lorentz
  with oscillator counts the model wasn't trained on). Measures
  generalization to *new* materials.

The Tier-A vs Tier-B gap quantifies how much of model performance comes
from material memorization vs. genuine n,k generalization.

## Quick start

```bash
# Generate a small dataset (≈100 rows for smoke testing)
python create_dataset/src/compile_datasets.py \
    --num_layers 4 --incidence_angle 0 --structure_seed 42 \
    --num_structures 100 --pool_size_min 4 --pool_size_max 16

# Train
python scripts/training.py \
    --data-dir create_dataset/data_prompts --epochs 1

# Evaluate
python scripts/evaluate.py \
    --checkpoint data/checkpoints/<auto-generated-tag>/latest \
    --limit-examples 20
```

The default training hyperparameters (encoded in `scripts/training.py`)
land at `--lr 4.42e-5 --batch-size 64 --epochs 1 --d-model 1024
--n-layers 8 --dropout 0.1`. Re-run `scripts/lr_tuning.py` before any
production training — the LR power-law fit from CHROMA-Lite does not
transfer because the model size and input dimensionality are different.
