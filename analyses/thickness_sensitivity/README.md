# Thickness Sensitivity Study

Diagnostic study that measures **how ΔE₀₀ responds to nanometre-scale
perturbations of the topmost colour-determining layer** in an INDIGO
thin-film stack. The output drove the token-grid decision (5 nm →
2 nm) documented in `src/materials_vocab.py`.

The full method lives in
[`thickness_sensitivity.py`](thickness_sensitivity.py); this README is
the hand-off document — what the study is asking, why it is set up the
way it is, and where every artefact lives.

---

## What we are trying to answer

The model emits per-layer thicknesses as **classification tokens** over
a discrete grid. Two questions decide that grid:

1. **What Δnm perturbation moves the achieved colour by
   ΔE ≈ 2–3?** ΔE ≈ 2–3 is the "clearly perceptible" band that colour
   scientists cite — anything a person would notice. ΔE = 1 was too
   close to numerical noise from the optical simulator to be a useful
   threshold, so we report 2 and 3.

2. **Does sensitivity vary systematically with the layer's base
   thickness?** If yes, a log-nm or piecewise grid with finer spacing
   at small t would spend tokens where they matter. If not, a plain
   linear grid at the right pitch is enough.

The output — a "snap ΔE" table per candidate grid — makes the token
choice honest instead of picking a round number by feel.

---

## Why the materials are sampled the way they are

The study samples structures from the same
`create_dataset/src/pool_sampler.RandomLayerSimulation` used at data
generation time, with the production defaults:

- `p_real = 0.15` — 15 % of pool slots draw a **real** JLL measured
  material; 85 % draw a fresh **synthetic** Lorentz-parameterised
  material.
- Real materials come from `src.material_features.load_jll_directory`
  (the held-in split from `create_dataset/src/pool_sampler.split_jll_real`).
- Synthetic materials are causal by construction (positive damping in
  the Lorentz model) and Kramers-Kronig consistent, so they are
  physically valid even though they were never measured.

Three reasons the study weights synthetics so heavily:

1. **Coverage.** The JLL library contains ~35 dielectrics and metals.
   A sensitivity study needs hundreds of layer-sweeps to bin
   meaningfully by base thickness *and* material class — real-only
   would starve every bin.
2. **Distribution match.** The training set is 15 % real / 85 %
   synthetic. If the sensitivity study used a different mix, the
   token-grid choice would optimise for a distribution the model
   never sees. Keeping the same mix means the grid recommendation
   transfers directly to production.
3. **Diversity of dispersion.** Lorentz permutations shift the
   resonance frequency and damping across a much wider range than any
   fixed measured library, which stress-tests the grid against every
   plausible optical response — including combinations that don't
   have a real-world exemplar yet.

The `structure_source` column in every output file records whether
each swept structure came from the **directed high-chroma search**
(`high_chroma_search`) or the **undirected random sampler**
(`random`). Both are reported separately so we can see whether high-
chroma structures — the ones where thickness precision matters most
— cluster on more sensitive operating points than random ones.

---

## Why we probe the topmost colour-determining layer

**First-pass sweeps of every layer showed most layers barely move the
colour.** The reason is optical: interior layers sit behind the top
layer and can only affect what light reaches them; light that gets
absorbed or reflected at the surface never sees them at all. So a
naive "sweep every layer" study is dominated by interior layers whose
sweep is a flat line, dragging the aggregate |dΔE/dnm| toward zero
and making every grid look fine. That would give a false OK to
grids that are actually too coarse for the layers that *do* move the
colour.

Restricting to the **outermost** layer (the air-side one) fixed most
of the flat-curve dilution, but not all: some structures have an
outermost layer that is an **opaque metal** thicker than a few
skin depths. In that regime, adding more metal thickness just piles
more onto a stack that already reflects like a bulk metal — the
optical response saturates, and the sweep is flat again.

So the study walks from the air side inward until it finds a layer
whose ±sweep produces measurable ΔE (currently
`max ΔE ≥ OPAQUE_MAX_DE = 0.05`), and probes THAT layer. The output
JSON records:

- `n_walked_inward` — how many structures needed to skip an opaque
  top layer to find a colour-determining one below.
- `n_fully_opaque` — how many structures were fully opaque and were
  dropped from the analysis entirely.
- The `layer_depth_from_top` column (in every output row) records
  which layer was actually probed (0 = the air-side surface layer).

---

## Files in this folder

### Code

| File | What it does |
|---|---|
| `thickness_sensitivity.py` | The whole study: material pool, structure generation, layer walk, thickness sweep, aggregation, plotting, JSON+CSV export. Self-contained (imports project modules but is not imported anywhere). |
| `run.sh` | SLURM wrapper for Pitt CRC's `smp` queue. Exposes every knob (`N_STRUCTURES`, `SWEEP_MAX_NM`, `SWEEP_STEP_NM`, seed, output dir). Handles the `CUDA_VISIBLE_DEVICES=""` trick required to keep JAX from crashing on the CUDA-less smp nodes. |
| `README.md` | This file. |

### Outputs (`results/` — small default run; `results_large/` — hand-off run)

Every run writes to a single `OUTPUT_DIR`, no subdirectories. The
default run (60 structures per source, ±10 nm, 1 nm step, ~5 min
CPU) produces the same artefact list as the large hand-off run
(500 per source, ±15 nm, 0.5 nm step, ~30 min CPU) — only the
sample sizes differ.

**Aggregate charts.** Show the grid-choice story at a glance —
these are the ones for the write-up.

| File | Shows |
|---|---|
| `curves_examples.png` | A sample of individual per-layer ΔE(Δnm) sweeps, one panel each, colour-coded by source. Face-validity check — you should see V-shaped curves around Δnm=0. |
| `sensitivity_by_bin.png` | `|dΔE/dnm|` (local slope at Δnm=0) binned by base thickness. If this were monotone in t, a log-nm grid would be defensible; the observed shape is not monotone → linear grid stays. |
| `delta_e_2_by_bin.png` | Δnm needed to reach ΔE₀₀ = 2 on either side of the base, binned by base thickness, split by source. The distance a nominal design has to be "off" before a person would notice. |
| `delta_e_3_by_bin.png` | Same, threshold ΔE = 3 ("clearly perceptible"). |
| `grid_comparison.png` | Per candidate token grid: expected snap ΔE₀₀ (mean, p95, max, fraction of layers > threshold). One panel per statistic; six grids compared (5 nm / 2 nm / 1 nm linear, 1/2/5 piecewise, log-nm 1.10, log-nm 1.05). |
| `grid_comparison_by_source.png` | Same as above but split HC vs random — shows whether the grid choice is dominated by directed-search structures. |
| `grid_comparison.txt` | Printable tables (same numbers as the plot) for pasting into notebooks / slides. |

**Data files.** Everything you need to re-analyse, re-plot, or
merge with other studies. All three describe the same sweep; use
whichever format is convenient.

| File | Grain | Columns |
|---|---|---|
| `per_layer.csv` | One row per probed layer (`≈ 2 × N_STRUCTURES` rows, minus dropped opaques) | `structure_id, structure_source, layer_idx, layer_depth_from_top, material_name, material_category, base_thickness_nm, L_base, a_base, b_base, local_slope_dE_per_nm, dnm_de2_{neg,pos}, dnm_de3_{neg,pos}, dnm_de5_{neg,pos}` |
| `sweeps_long.csv` | One row per (structure, layer, Δnm) probe point (`≈ 2 × N_STRUCTURES × (2·SWEEP_MAX_NM/SWEEP_STEP_NM + 1)` rows) | `structure_id, structure_source, layer_idx, material_name, material_category, base_thickness_nm, delta_nm, delta_e` — long-format, ready for `pd.read_csv` + `groupby` |
| `sensitivity.json` | Everything above plus the run config and the six per-grid aggregate tables. Source of truth. | `config, grids_all_sources, grids_high_chroma_search, grids_random, per_layer, sweeps` (schema mirrors the CSVs) |

The two CSVs are convenience mirrors of the JSON — no information is
in one that isn't in the other. Pick the file that matches your
downstream tool.

---

## Running the study

The environment setup (module + venv activate + JAX-on-CPU trick)
is baked into `run.sh`. All knobs are env vars.

### Default run — quick sanity check (~5 min on CPU)

```bash
sbatch analyses/thickness_sensitivity/run.sh
```

Produces `results/` in the repo (60 structures per source, ±10 nm
at 1 nm resolution). Enough to see the qualitative shape; not
enough for tight per-bin CIs.

### Large hand-off experiment (~6-8 h on CPU, 12 h SBATCH window)

```bash
N_STRUCTURES=2000 SWEEP_MAX_NM=15 SWEEP_STEP_NM=0.5 \
    OUTPUT_DIR=analyses/thickness_sensitivity/results_large \
    sbatch analyses/thickness_sensitivity/run.sh
```

- **2000 per source** — statistically-meaningful bin counts for both
  the HC and random subsets, and for the by-material splits.
- **±15 nm** — captures ΔE₅ crossings even for less-sensitive layers,
  so `dnm_de5_{neg,pos}` is populated more often than at ±10 nm.
- **0.5 nm step** — twice the resolution of the default; makes the
  local-slope estimate at Δnm=0 (central finite difference over ±0.5
  nm) less noisy.

Approximate cost:

- **HC directed search dominates** — ~5-10 s per structure at
  `candidate_count=24`, `refine_iters=12` → ~5 h for the 2000 HC
  half (random path is free).
- **Sweep sims** — `2000 × 2 × (2·15/0.5 + 1) = 244,000` stack sims.
  At ~30 ms per sim on smp CPU, ~2 h.
- **Total** ~6-8 h. The 12 h `#SBATCH --time` in `run.sh` covers
  variance and any HC-search retries.

### Overriding individual knobs

Any env var listed in the `run.sh` header can be set at the sbatch
line — see `sbatch analyses/thickness_sensitivity/run.sh` header
for the full list. Common ones:

```bash
# Just measure sensitivity on high-chroma structures (skip random) —
# not currently a flag; the way to do this is post-hoc: filter the
# per_layer.csv on structure_source == "high_chroma_search".

# Reproduce a specific run — same seed gives the exact same
# structures and sweeps:
SEED=42 sbatch analyses/thickness_sensitivity/run.sh
```

---

## Interpreting the outputs

**The grid decision.** Look at `grid_comparison.png` → the "p95 snap
ΔE" panel. That is the ΔE₀₀ the model incurs on the 95th-percentile
worst-snapped layer under each grid. A grid whose p95 exceeds the
"just perceptible" ΔE = 2 line is too coarse.

**The sensitivity shape.** Look at `sensitivity_by_bin.png`. If the
curve is roughly flat, linear grids are efficient. If it slopes up
toward small t, a piecewise-fine-at-small-t grid saves tokens for
the same p95.

**Source dominance.** Look at `grid_comparison_by_source.png`. If
the HC and random panels diverge (HC needing a finer grid than
random), the grid choice is dominated by high-chroma structures —
consistent with intuition that directed search finds thin-layer
"interference-critical" operating points.

**Sanity check.** Read `sensitivity.json`'s `config` block. Confirm
`layer_probed == "outermost (air-side)"`, and that
`n_walked_inward + n_fully_opaque` is a small fraction of the total
— if it isn't, the study is measuring something other than what the
paper describes.

---

## Provenance

- Original driver of the 5 nm → 2 nm token-grid decision. See the
  vocab-size docstring in `src/materials_vocab.py` for the resulting
  code invariants.
- Kept as a diagnostic — every time the pool sampler or the optical
  simulator changes (new synthetic strategy, different substrate,
  non-normal incidence), re-run and check whether the grid choice
  still holds.
