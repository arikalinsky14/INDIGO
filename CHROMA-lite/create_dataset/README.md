# Dataset Creation

Scripts for generating the CHROMA-Lite training data and constraint test set.

## Directory Structure

```
create_dataset/
├── data_prompts/           # Generated data files (parquets + test set CSV)
│   └── test_set.csv        # Constraint test set (1200 examples)
├── src/
│   ├── random_layer.py             # Random thin-film structure generation
│   ├── procedural_template.py      # Natural language prompt templates
│   ├── constraint_aware_template.py # Template wrapper that also captures structured constraints
│   ├── generate_test_set.py        # Generates the constraint test set
│   ├── compile_datasets.py         # Compiles training parquets
│   ├── compile_datasets_batch.py   # Batch compilation with OpenAI rephrasing
│   ├── rephraser.py                # LLM-based prompt rephrasing
│   └── ...
├── batch_manager.py        # OpenAI batch API manager
└── requirements.txt
```

## Constraint Test Set

The constraint test set (`data_prompts/test_set.csv`) contains **1200 examples** designed to evaluate whether the model respects natural language constraints in its predictions.

### Composition

- **400 examples** with 4 layers
- **400 examples** with 6 layers
- **400 examples** with 8 layers

Each example uses raw procedural templates (no LLM rephrasing) with higher constraint density than training data, ensuring every example has meaningful checkable constraints.

### CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Natural language prompt with embedded constraints |
| `materials` | JSON string | Ground truth materials, e.g. `["SiO2", "Ag", "SiO2", "TiO2"]` |
| `thicknesses` | JSON string | Ground truth thicknesses (nm), e.g. `[100, 50, 75, 120]` |
| `sRGB_R` | JSON string | Ground truth RGB color [0-255], e.g. `[128, 64, 200]` |
| `num_layers` | int | Number of layers (4, 6, or 8) |
| `constraints` | JSON string | Structured constraint annotations (see below) |

### Structured Constraints

The `constraints` column contains a JSON object with the following fields:

```json
{
    "layer_count_type": "exact",
    "layer_count_value": 4,
    "materials_type": "strict",
    "allowed_materials": ["SiO2", "Ag"],
    "forbidden_materials": [],
    "additional": [
        {"type": "individual_thickness", "material": "SiO2", "bound": "max", "value_nm": 150},
        {"type": "adjacency", "material_a": "SiO2", "material_b": "Ag", "must_be_adjacent": true}
    ]
}
```

**Constraint types:**

| Type | Fields | Description |
|------|--------|-------------|
| Layer count | `layer_count_type` (`exact`/`min`/`max`/`none`), `layer_count_value` | Number of layers |
| Materials | `materials_type` (`strict`/`helpful`/`extra`/`restrict`/`any`) | Which materials to use |
| `adjacency` | `material_a`, `material_b`, `must_be_adjacent` | Two materials must/must not be adjacent |
| `relative_thickness` | `material_a`, `material_b`, `relation` (`thicker`/`thinner`/`equal`) | Relative thickness between materials |
| `individual_thickness` | `material`, `bound` (`min`/`max`/`exact`), `value_nm` | Per-material thickness bound |
| `total_thickness_sum` | `bound` (`min`/`max`), `value_nm` | Total stack thickness bound |
| `total_thickness_each` | `bound` (`min`/`max`), `value_nm` | Per-layer thickness bound |
| `layer_identity` | `position` (`first`/`last`), `material`, `must_be` | First/last layer material |

### Regenerating the Test Set

```bash
python create_dataset/src/generate_test_set.py
```

### Evaluating Against the Test Set

```bash
python train_full_model/scripts/evaluate.py \
    --checkpoint train_full_model/data/checkpoints/<tag>/latest \
    --rgb-to-structure-checkpoint pretrain_rgb_to_structure/data/checkpoints/<tag>/latest \
    --constraint-test
```

No `--cache-dir` is needed — constraint test mode skips teacher forcing and only runs autoregressive generation + constraint checking. See `train_full_model/scripts/evaluate.py --help` for all options.
