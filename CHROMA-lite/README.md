# CHROMA-Lite

**Modular ML pipeline for text-guided structural color design using thin-film optics.**

The system decomposes the full text→structure problem into independently pretrained stages:

```
"Make a blue coating with SiO2"   →   [pretrain_text_to_rgb]   →   RGB [0.1, 0.2, 0.8]
                                                                          │
RGB [0.1, 0.2, 0.8]              →   [pretrain_rgb_to_structure] →  [SiO2/100nm, Ag/50nm, ...]
```

After pretraining, `train_full_model` combines them for end-to-end fine-tuning.

---

## Repository Structure

```
chroma-lite/
├── src/                                    # SHARED modules (used by all subprojects)
│   ├── materials_vocab.py                  #   Materials/thickness token encoding (25 materials × 40 thicknesses)
│   ├── dataset.py                          #   Parquet reading, train/validation splitting, dataset classes
│   └── optical_sim.py                      #   Thin-film optical simulation (jaxlayerlumos)
│
├── create_dataset/                         # Dataset generation (parquet files)
│   └── data_prompts/                       #   layers_N_angle_A_substrate_S/seed_X.parquet
│
├── pretrain_rgb_to_structure/              # Stage 1: RGB → thin-film structure (MLP)
│   ├── src/
│   │   └── model.py                        #   ThinFilmMLP architecture + ModelConfig
│   ├── scripts/
│   │   ├── training.py                     #   Training loop (autoregressive, multi-step)
│   │   ├── evaluate.py                     #   Eval: teacher forcing + autoregressive + CIEDE2000
│   │   ├── lr_tuning.py                    #   Learning rate grid search
│   │   ├── plot_training_curves.py         #   Plot train/val curves across checkpoints
│   │   └── investigate_data.py             #   Data leakage investigation
│   ├── slurms/                             #   SLURM job scripts
│   └── data/                               #   Checkpoints (auto-created)
│
├── pretrain_text_to_rgb/                   # Stage 2: Text → RGB (frozen LLM + MLP head)
│   ├── src/
│   │   └── model.py                        #   TextToRGBHead (MLP) + TextToRGBModel (w/ encoder)
│   ├── scripts/
│   │   ├── training.py                     #   Two-phase: encode texts → train MLP head
│   │   └── evaluate.py                     #   Eval: MSE, MAE, CIEDE2000
│   ├── slurms/                             #   SLURM job scripts
│   └── data/                               #   Checkpoints + embedding caches (auto-created)
│
└── train_full_model/                       # Stage 3: End-to-end pipeline (TODO)
```

---

## Shared `src/` Modules

All subprojects import from the top-level `src/`:

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `materials_vocab.py` | Token encoding for 25 materials × 40 thicknesses = 1000 tokens + EOS | `encode_layer`, `decode_token`, `build_structure_matrix`, `VOCAB_SIZE=1001` |
| `dataset.py` | Parquet scanning, deterministic splitting, dataset classes | `ThinFilmDataset` (RGB→structure), `TextThinFilmDataset` (text→RGB), `scan_files`, `make_permutation` |
| `optical_sim.py` | Transfer matrix method → sRGB color computation | `OpticalSimulator`, `is_available()` |

### Data Format

Parquet files in `create_dataset/data_prompts/layers_N_angle_A_substrate_S/seed_X.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Natural language prompt |
| `materials` | string (JSON) | `'["SiO2", "Ag"]'` |
| `thicknesses` | string (JSON) | `'[100, 50]'` |
| `sRGB_R` | string (JSON) | `'[128, 64, 200]'` ground truth RGB |
| `num_layers` | int | 1-8 |

### Train/Validation Split

99.5% train / 0.5% validation, deterministic via `seed=42`.

---

## pretrain_rgb_to_structure

**Task:** Given a target RGB color, predict the thin-film layer stack that produces it.

**Architecture:** Autoregressive MLP. At each step, takes RGB + partial structure matrix → predicts next layer token.

```
Input (203) = RGB (3) + structure_matrix (25×8 flattened)
→ Linear(d_model) → ReLU → Dropout → ... → Linear(1001)
```

**Current best config:** `d_model=1024, n_layers=8, dropout=0.1, lr=6.86e-4`

### Quick Start

```bash
# Train
python pretrain_rgb_to_structure/scripts/training.py \
    --data-dir create_dataset/data_prompts --epochs 200

# Evaluate (teacher forcing only - fast)
python pretrain_rgb_to_structure/scripts/evaluate.py --low-compute

# Evaluate (full: autoregressive + optical sim + CIEDE2000)
python pretrain_rgb_to_structure/scripts/evaluate.py \
    --checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest
```

---

## pretrain_text_to_rgb

**Task:** Given a natural language prompt, predict the target RGB color.

**Architecture:** Frozen LLM encoder (TinyLlama-1.1B) + trainable MLP head.

```
Text → Frozen TinyLlama → last hidden state [2048]
→ Linear(512) → ReLU → Dropout → ... → Linear(3) → Sigmoid → RGB [0,1]
```

**Training is two-phase for efficiency:**
1. Pre-compute LLM embeddings for all training prompts (one-time)
2. Train MLP head on cached embeddings (fast, many epochs)

### Quick Start

```bash
# Train (will auto-cache embeddings)
python pretrain_text_to_rgb/scripts/training.py \
    --data-dir create_dataset/data_prompts \
    --encoder TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 50

# Train with pre-cached embeddings (skip encoding)
python pretrain_text_to_rgb/scripts/training.py \
    --embeddings-cache pretrain_text_to_rgb/data/embeddings_cache_train_*.pt \
    --epochs 100

# Evaluate
python pretrain_text_to_rgb/scripts/evaluate.py \
    --checkpoint pretrain_text_to_rgb/data/checkpoints/<tag>/latest
```

### SLURM

```bash
sbatch pretrain_text_to_rgb/slurms/slurm_training.sh
sbatch pretrain_text_to_rgb/slurms/slurm_training.sh --epochs 100
CHECKPOINT=pretrain_text_to_rgb/data/checkpoints/<tag>/latest \
    sbatch pretrain_text_to_rgb/slurms/slurm_evaluate.sh
```

---

## Dependencies

```
torch>=2.0
pyarrow>=12.0
numpy>=1.24
transformers>=4.30          # for pretrain_text_to_rgb
jaxlayerlumos               # optional, for optical simulation
matplotlib                   # optional, for visualizations
```

---

## Migration from Previous Layout

The old `finetune_model/` has been split:
- `finetune_model/src/` → shared modules in `src/` + local model in `pretrain_rgb_to_structure/src/`
- `finetune_model/scripts/` → `pretrain_rgb_to_structure/scripts/`
- `finetune_model/slurms/` → `pretrain_rgb_to_structure/slurms/`

All existing checkpoints are compatible — just update the path.
