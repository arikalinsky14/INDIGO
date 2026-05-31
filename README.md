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
│   └── model.py                         # FlexMaterialMLP + FlexMaterialCrossAttn
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
          Backbone (head_mode = 'mlp' or 'cross_attn')
                   │   • mlp:        flatten + n-layer MLP
                   │   • cross_attn: per-slot transformer encoder
                   │                 + query cross-attention pointer
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

## Backbone architectures — `head_mode`

Two backbones produce the same `[VOCAB_SIZE]` logits over the same vocab.
Select via `ModelConfig.head_mode` (CLI: `--head-mode {mlp,cross_attn}`,
SLURM: `HEAD_MODE=...`). The default is `mlp` so existing checkpoints
load unchanged; `cross_attn` is opt-in and appends `_cross_attnH<N>` to
the checkpoint tag so it never collides with an MLP run.

### `head_mode=mlp` (default — `FlexMaterialMLP`)

Encoded pool is flattened (`[B, M_MAX, encoder_out] → [B, M_MAX*encoder_out]`)
and concatenated with Lab and the flattened structure matrix. A
feed-forward stack of `n_layers` `Linear(d_model)+ReLU+Dropout` blocks
produces the logits. Permutation invariance over slots is learned as
augmentation via pool_sampler's per-row slot shuffle.

### `head_mode=cross_attn` (`FlexMaterialCrossAttn`)

A pointer-style head that keeps permutation-equivariance by construction:

```
            pool_features [B,M_MAX,2,L]      structure_matrix [B,M_MAX,MAX_LAYERS]
                    │                                  │
              MaterialEncoder                     (per-slot stripe)
                    │                                  │
               [B,M_MAX,E]   ───── concat ────► [B,M_MAX,E+MAX_LAYERS]
                                                       │
                                                  slot_proj
                                                       ▼
                                              [B,M_MAX,d_model]
                                                       │
                                  TransformerEncoder ×n_layers   ← SELF-ATTN over SLOTS
                                  (src_key_padding_mask=~pool_mask)
                                                       ▼
                                              slot_tokens [B,M_MAX,d_model]
                                                       │
   lab[B,3] + pool_size_norm[B,1] + struct_summary[B,MAX_LAYERS]
        │                                              │
    query_proj                                         │
        ▼                                              │
   query [B,1,d_model] ─── TransformerDecoder ×1 ─────►│   ← CROSS-ATTN query→slots
                          (memory_key_padding_mask)
        │                                              │
        │    ┌─── concat(slot_token, broadcast(query)) ┘
        ▼    ▼
   eos_head        thickness_head (Linear→GELU→Linear)
        │                │
   eos_logit       [B,M_MAX,NUM_THICKNESSES]   ─reshape→ [B, M_MAX*NUM_THICKNESSES]
        └────────────────┴────── concat ──────────────────► logits [B, VOCAB_SIZE]
                                                                + output_mask
```

**What attention runs over (be explicit):**

| Attention                                            | Used? | Layers              | Operates over                                                                                |
|------------------------------------------------------|-------|---------------------|----------------------------------------------------------------------------------------------|
| Self-attention over **pool slots**                   | YES   | `n_layers`          | `M_MAX` slot tokens (each = encoded n,k + that slot's per-layer thickness stripe)            |
| Cross-attention **query → slots**                    | YES   | 1                   | Query (1 token) attends to all `M_MAX` slot tokens, with padded slots masked out             |
| Causal self-attention over a **sequence of past tokens** | NO  | —                   | Not used: the partial structure is *not* tokenised as a sequence of past (slot, thickness) decisions |

**How the cross-attn head encodes previously generated tokens.**
The partial structure (everything decoded so far) enters in two places, both
derived from the `structure_matrix` `[M_MAX, MAX_LAYERS]` where
`matrix[s, l] = normalized_thickness` iff layer `l` used slot `s`:

1. **Per-slot stripe** → folded into each slot token *before* self-attention.
   Slot `s`'s row holds its thickness at every layer position where it was
   chosen (zero elsewhere). The self-attention over slots therefore reasons
   about each slot's full usage history alongside its n,k features.

2. **Per-layer thickness summary** → folded into the query *before*
   cross-attention. Since each layer position is filled by exactly one slot,
   `structure_matrix.sum(dim=slots)` collapses to the per-layer thickness
   sequence (with trailing zeros revealing how many layers remain). The
   query carries the chronological trajectory; the slot tokens do not.

So there *is* self-attention — but over the **set of slots**, not over a
generated-token sequence. The chronological order of decisions is preserved
implicitly (column index of `structure_matrix` = layer step) and exposed to
the query as the per-layer thickness summary, but never attended over
causally. This is a deliberate trade-off: it preserves permutation-equivariance
over slots by construction while keeping the model O(M_MAX²) instead of
O((M_MAX + step) · step) per forward pass.

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
# Inspect the synthetic-material families before committing GPU time
python scripts/visualize_synthetic.py
# -> outputs/synthetic_visual_check.png (2x4 panel: small/large/interp/lorentz)

# Cross-check the optical sim against the original CHROMA-Lite sim
python scripts/verify_optical_sim.py

# Generate a small training dataset (~1000 rows, 5 shards × 200 rows)
python scripts/generate_dataset.py \
    --total-rows 1000 --rows-per-shard 200 --start-shard-id 0 \
    --output-dir data/train

# Tier-A + Tier-B test sets (held-out materials in Tier-B)
python scripts/generate_test_sets.py \
    --output-root data/test \
    --tier-a-rows 500 --tier-b-rows 500 --rows-per-shard 250

# Train
python scripts/training.py \
    --data-dir data/train --epochs 1

# Evaluate
python scripts/evaluate.py \
    --checkpoint data/checkpoints/<auto-generated-tag>/latest \
    --limit-examples 20
```

For production-scale data generation, the canonical command is:

```bash
# 2M training rows, 5000 rows/shard → 400 shards. Resumable via --skip-existing.
python scripts/generate_dataset.py \
    --total-rows 2000000 --rows-per-shard 5000 --start-shard-id 0 \
    --output-dir data/train --skip-existing
```

Each shard is fully determined by its `shard_id`, so re-running with the
same arguments yields bit-identical data and extending the dataset is
just picking up where the previous run left off.

The default training hyperparameters (encoded in `scripts/training.py`)
land at `--lr 4.42e-5 --batch-size 64 --epochs 1 --d-model 1024
--n-layers 8 --dropout 0.1`. Re-run `scripts/lr_tuning.py` before any
production training — the LR power-law fit from CHROMA-Lite does not
transfer because the model size and input dimensionality are different.

## LR tuning (recommended commands)

`slurms/lr_tuning.sh` defaults are now aligned with `slurms/training.sh`
(`BATCH_SIZE=256`, `WEIGHT_DECAY=0.01`, `DROPOUT=0.1`, `WARMUP_FRACTION=0.02`,
`STREAMING=1`, `NUM_WORKERS=6`, `PREFETCH_FACTOR=1`). Override any of these
via env var when sbatch'ing to keep the LR fit meaningful for the
architecture you'll actually train.

### Sweep across 1, 2, 3 epochs at the same training budget

Each job writes `outputs/lr_search/lr_search_ep<E>_lim<N>.json`, so the
three runs never clobber each other. Streaming is on by default.

```bash
# 6-LR sweep at each epoch count on a 1 M-row subset (one GPU each, in parallel)
for EP in 1 2 3; do
    EPOCHS=${EP} LIMIT_EXAMPLES=1000000 sbatch slurms/lr_tuning.sh
done
```

Use this when you want to compare LR optima across epoch counts at a fixed
data budget — the cosine schedule reshapes with `EPOCHS`, so the optimum
moves. Production-training the model with `EPOCHS=K` should use the LR found
by the sweep with the matching `EPOCHS=K`.

### Sweep at the production scale (full training pool)

```bash
# Single sweep, one epoch, on the entire training set (~hours)
EPOCHS=1 sbatch slurms/lr_tuning.sh

# Same, but with the cross-attention pointer head — re-tune LR per head
HEAD_MODE=cross_attn EPOCHS=1 sbatch slurms/lr_tuning.sh
```

### Multi-N scaling-law fit (preferred for production-LR derivation)

For a single-pass production run on `N` rows, `lr_opt` scales as a power
law in `N`. Submit at several `N` values with `EPOCHS=1`, then extrapolate
to your target with `scripts/fit_lr_scaling.py`:

```bash
for N in 500000 1000000 2000000; do
    EPOCHS=1 LIMIT_EXAMPLES=${N} sbatch slurms/lr_tuning.sh
done

# After all three complete:
python scripts/fit_lr_scaling.py \
    --results-dir outputs/lr_search \
    --target-examples 10000000 --plot
```

### Narrow-range follow-up

Once you know the rough scale of the optimum, narrow `LR_MIN`/`LR_MAX` and
bump `N_LRS` for a finer sweep:

```bash
LR_MIN=5e-5 LR_MAX=5e-4 N_LRS=8 LIMIT_EXAMPLES=1000000 \
    sbatch slurms/lr_tuning.sh
```
