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
| Fig 4, INDIGO vs random (headline) | `fig4_headline.py` | `fig4_headline.{pdf,png}` | `results/paper/headline.json`, `results/paper_runs/{strong_model, strong_random, budget50_model}` |
| Fig 5, paired per-target comparison | `fig5_paired.py` | `fig5_paired.{pdf,png}` (8.3 cm) | `results/paper_runs/{strong_model, strong_random}` |
| Fig 6, example targets | `fig6_examples.py` | `fig6_examples.{pdf,png}` | `results/paper_runs/{strong_model, strong_random}` |
| Fig 7, gamut batteries (out-of-distribution) | `fig7_gamut.py` | `fig7_gamut.{pdf,png}` | `results/prod_eval/gamut/gamut_strong_dog{,_random}.json` |
| NEUTRON head-to-head (number TBD) | `fig_neutron.py` | `fig_neutron.{pdf,png}`; caption `results/paper/neutron_caption.tex` (generated) | `results/neutron/rescore_final_{fixed,full}.json` |
| LR law (number TBD) | `fig_lr_law.py` | `fig_lr_law.{pdf,png}`; paragraph `results/paper/lr_law_paragraph.tex`, its numbers `results/paper/lr_law_numbers.json` | `analyses/scaling/results/lr_law_fit_cliff.json`, `tuned/isoflop_tuned.json`, production LR sweep in `results/lr_curves/` |
| Fig 8, scaling | `fig8_scaling.py` | `fig8_scaling.{pdf,png}` | `analyses/scaling/results/tuned/` |

`_runs.py` loads the headline runs with `scripts/paper_numbers.py`'s own loader
(failed rows count as failures) and stops if the rows disagree with
`headline.json`.

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
python scripts/paper_figures/fig4_headline.py
python scripts/paper_figures/fig5_paired.py
python scripts/paper_figures/fig6_examples.py
python scripts/paper_figures/fig7_gamut.py
python scripts/paper_figures/fig8_scaling.py
python scripts/paper_figures/fig_lr_law.py
python scripts/paper_figures/fig_neutron.py
```

Needs numpy, matplotlib, scipy and pillow; not torch.
