# Paper figures

Every figure in the INDIGO paper is made by one script here. Outputs go to
`results/paper/figures/`, and each script writes its own one-line provenance
(input files and fields) to `results/paper/figures/PROVENANCE.md`.

| Figure | Script | Output | Data |
|---|---|---|---|
| Fig 1, pipeline | `fig1_pipeline.py` | `fig1_pipeline.{pdf,png}` | none (schematic) |
| TOC graphic | `fig1_pipeline.py` | `toc_graphic.{tiff,pdf}` (8 x 4 cm) | none |
| Fig 2, material and target diversity | `fig2_diversity.py` | `fig2_diversity.{pdf,png}` | JaxLayerLumos materials, `results/prod_eval/eval_tier_{a,b}_greedy.json` |
| Fig 3, architecture | `fig3_architecture.py` | `fig3_architecture.{pdf,png}` | sizes from `src/` and `results/production/current/config.json` |
| Fig 8, scaling | `fig8_scaling.py` | `fig8_scaling.{pdf,png}` | `analyses/scaling/results/tuned/` |

`_style.py` holds the shared rules: 8.3 cm or 17 cm wide, 8 pt sans-serif,
Okabe-Ito colors, no titles, vector PDF with editable text plus 600 dpi PNG.
Figure wording lives in a `LABELS` dict at the top of each script.

Rules for anything added here: only settled result files, never a number
from a pending run; target metadata only where a model-dependent field
would bias the figure.

```bash
python scripts/paper_figures/fig1_pipeline.py
python scripts/paper_figures/fig2_diversity.py   # --jll-materials-dir DIR if jaxlayerlumos is not installed
python scripts/paper_figures/fig3_architecture.py
python scripts/paper_figures/fig8_scaling.py
```

Needs numpy, matplotlib, scipy and pillow; not torch.
