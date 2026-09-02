# Thickness Sensitivity Study

Diagnostic study that measures **how ΔE₀₀ responds to nanometre-scale
perturbations of the layer whose thickness most drives the achieved
colour** in an INDIGO thin-film stack. The output drove the token-grid
decision (5 nm → 2 nm) documented in `src/materials_vocab.py`.

The full method lives in
[`thickness_sensitivity.py`](thickness_sensitivity.py); this README is
the hand-off document — what the study is asking, why it is set up the
way it is, and where every artefact lives.

---

## What we are trying to answer

The model emits per-layer thicknesses as **classification tokens** over
a discrete grid. Two questions decide that grid:

1. **What Δnm perturbation moves the achieved colour by ΔE ≈ 2–3?**
   ΔE ≈ 2–3 is the "clearly perceptible" band that colour scientists
   cite — anything a person would notice. ΔE = 1 was too close to
   numerical noise from the optical simulator to be a useful
   threshold, so we report 2, 3, and 5.

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
   never sees.
3. **Diversity of dispersion.** Lorentz permutations shift the
   resonance frequency and damping across a much wider range than any
   fixed measured library, stress-testing the grid against every
   plausible optical response.

The `structure_source` column in every output file records whether
each swept structure came from the **directed high-chroma search**
(`high_chroma_search`) or the **undirected random sampler**
(`random`). Both are reported separately so we can see whether high-
chroma structures — the ones where thickness precision matters most
— cluster on more sensitive operating points than random ones.

---

## Which layer we probe (and how the adaptive probe works)

### Layer selection — most sensitive by finite-difference slope

For each structure we probe every layer at Δnm = ±SLOPE_PROBE_NM
(default 0.5 nm) — two simulator calls per layer — and compute a peak
slope:

```
slope_i = max(|ΔE(+0.5 nm)|, |ΔE(-0.5 nm)|) / 0.5   for each layer i
```

The layer with the largest slope is the one whose thickness the
achieved colour is most sensitive to, and therefore the one whose
grid resolution actually matters. The chosen layer's index and depth
from the air side are recorded in every output row
(`chosen_layer_idx`, `layer_depth_from_top`), along with the full
`all_layer_slopes` vector in `sensitivity.json` for downstream
whole-stack analysis.

This replaces the previous "walk from the air side inward past opaque
layers" heuristic, which could pick a mildly-sensitive top layer
while a highly-sensitive interior layer was ignored, and which
dropped fully opaque structures entirely — biasing the distribution
of "min Δnm needed" toward already-sensitive layers.

### Adaptive probe on the chosen layer

For each side (+ and − direction) independently:

1. **Doubling phase.** Probe outward through the schedule
   `0.5 → 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 nm`, stopping as soon as
   ΔE reaches 5.0 (the highest reported threshold) OR the perturbed
   thickness would fall outside the physical bound [1 nm, 300 nm].
2. **Bisection phase.** For each threshold (ΔE = 2, 3, 5)
   independently, find the smallest probe with ΔE ≥ threshold and
   bisect the interval [previous, that probe] until it is narrower
   than BISECT_TOL_NM (default 0.05 nm). The upper endpoint is the
   reported Δnm crossing.

Typical cost: ~7–9 doubling probes + ~4–6 bisection probes per
threshold, sharing probes across thresholds where possible.
**15–25 simulator calls per side, ~30–50 per structure** — vs
~121 for a fixed ±5 nm / 0.25 nm sweep grid.

### Right-censoring — no structure is dropped

If NO probe on a side reached a given threshold — a fully opaque
metal layer, say, whose ±128 nm perturbation still moves colour by
less than ΔE = 2 — that side's crossing is set to CAP_NM (128 nm) and
its `cens_de*_*` flag is set to 1. The structure is kept and
counted, and its local slope is still valid (measured at ±0.5 nm).

The `sensitivity_by_bin.png` plot uses the slope (always defined) and
so is unaffected by censoring. The `delta_e_2_by_bin.png` /
`delta_e_3_by_bin.png` boxplots **exclude the censored rows from the
box body** and annotate the fraction censored per bin above each box
(e.g. `⌐12%` = 12 % of that bin were still under ΔE = 2 at ±128 nm).
This surfaces the "how many stacks are essentially insensitive?"
question the previous version hid by dropping those structures.

### Cross-structure parallelism

The sweep phase is embarrassingly parallel: each structure's sweep
depends only on its own materials + thicknesses. A
`ProcessPoolExecutor` with N_JOBS workers (default 4, matching
`--cpus-per-task`) gives an ~N_JOBS× speedup after per-worker JAX
warmup. The HC directed search and random-structure generation stay
serial — both consume the main-process RNG state, and parallelising
them would change which structures come out.

---

## Files in this folder

### Code

| File | What it does |
|---|---|
| `thickness_sensitivity.py` | Full study: material pool, structure generation, adaptive layer selection + probe, aggregation, plotting, JSON+CSV export. Self-contained. |
| `run.sh` | SLURM wrapper for Pitt CRC's `smp` queue. Exposes every knob (`N_STRUCTURES`, `CAP_NM`, `SLOPE_PROBE_NM`, `BISECT_TOL_NM`, `N_JOBS`, seed, output dir). Handles the `CUDA_VISIBLE_DEVICES=""` trick required to keep JAX from crashing on the CUDA-less smp nodes. |
| `README.md` | This file. |

### Outputs

Every run writes to a single `OUTPUT_DIR`, no subdirectories.
Convention:

- `results/` — small default run (60 structures per source, ~5 min on
  4 CPUs).
- `results_large/` — the large hand-off run (2000 structures per
  source, ~1 h on 4 CPUs).
- `results_fine/` — tighter bisection tolerance and finer slope probe
  for very-sensitive layers.

**Aggregate charts.** Show the grid-choice story at a glance — these
are the ones for the write-up.

| File | Shows |
|---|---|
| `curves_examples.png` | A sample of individual per-structure ΔE(Δnm) probes, one panel each, colour-coded by material category. Points are the actual doubling + bisection probes, so the density is higher near Δnm = 0 and near each threshold crossing. Face-validity check — expect roughly V-shaped curves around Δnm = 0. |
| `sensitivity_by_bin.png` | Peak `|dΔE/dnm|` at Δnm = 0 (from the ±SLOPE_PROBE_NM finite-difference), binned by base thickness. Every structure contributes exactly one point — including previously-dropped opaque ones (which cluster near zero). |
| `delta_e_2_by_bin.png` | Δnm needed to reach ΔE₀₀ = 2, binned by base thickness, split by source. Right-censored structures (both sides still under ΔE=2 at CAP_NM) are annotated above each box as `⌐N%` and excluded from the box body. |
| `delta_e_3_by_bin.png` | Same, threshold ΔE = 3 ("clearly perceptible"). |
| `grid_comparison.png` | Per candidate token grid: expected snap ΔE₀₀ (mean, p95, max, fraction of layers > threshold). One panel per statistic; six grids compared (5 nm / 2 nm / 1 nm linear, 1/2/5 piecewise, log-nm 1.10, log-nm 1.05). |
| `grid_comparison_by_source.png` | Same as above but split HC vs random — shows whether the grid choice is dominated by directed-search structures. |
| `grid_comparison.txt` | Printable tables (same numbers as the plot) for pasting into notebooks / slides. |

**Data files.** Everything you need to re-analyse, re-plot, or merge
with other studies. All three describe the same sweep; use whichever
format is convenient.

| File | Grain | Columns |
|---|---|---|
| `per_layer.csv` | One row per chosen layer (`≈ 2 × N_STRUCTURES` rows) | `structure_id, structure_source, chosen_layer_idx, layer_depth_from_top, material_name, material_category, base_thickness_nm, L_base, a_base, b_base, local_slope_dE_per_nm, max_de_seen_{neg,pos}, dnm_de{2,3,5}_{neg,pos}, cens_de{2,3,5}_{neg,pos}` |
| `sweeps_long.csv` | One row per (structure, probe) point — variable count per structure since the adaptive probe visits different Δnm's | `structure_id, structure_source, chosen_layer_idx, material_name, material_category, base_thickness_nm, delta_nm, delta_e` |
| `sensitivity.json` | Everything above plus run config, the six per-grid aggregate tables, and the full `all_layer_slopes` vector per structure. Source of truth. | `config, grids_all_sources, grids_high_chroma_search, grids_random, per_layer, sweeps` |

The two CSVs are convenience mirrors of the JSON — no information is
in one that isn't in the other. Pick the file that matches your
downstream tool.

---

## Running the study

The environment setup (module + venv activate + JAX-on-CPU trick) is
baked into `run.sh`. All knobs are env vars.

### Default run — quick sanity check (~5 min on 4 CPUs)

```bash
sbatch analyses/thickness_sensitivity/run.sh
```

Produces `results/` in the repo (60 structures per source, CAP_NM=128,
SLOPE_PROBE_NM=0.5, BISECT_TOL_NM=0.05, N_JOBS=4). Enough to see the
qualitative shape; not enough for tight per-bin CIs.

### Large hand-off experiment (~1 h on 4 CPUs, 12 h reservation)

```bash
N_STRUCTURES=2000 \
    OUTPUT_DIR=analyses/thickness_sensitivity/results_large \
    sbatch analyses/thickness_sensitivity/run.sh
```

- **2000 per source** — statistically-meaningful bin counts for both
  the HC and random subsets, and for the by-material splits.
- Default `CAP_NM = 128 nm` catches even weakly-sensitive layers
  before censoring; anything more insensitive than that is honestly
  described as "grid resolution doesn't matter for this stack".
- Default `BISECT_TOL_NM = 0.05 nm` gives 20× finer crossings than
  the previous 1 nm sweep step.

Approximate cost:

- **HC directed search dominates** — ~5-10 s per structure at
  `candidate_count=24`, `refine_iters=12` → ~5 h for 2000 HC.
- **Adaptive sweep** — ~30-50 sims per structure × 4000 structures ≈
  120-200k sims. At ~30 ms per sim on smp CPU with 4 workers, ~30-60
  min.
- **Total** ~1-2 h wallclock. The 12 h `#SBATCH --time` in `run.sh`
  covers HC-search variance.

### Fine-granularity experiment (~1 h on 4 CPUs)

Tighter bisection tolerance and finer slope probe for very-sensitive
layers (sub-nm crossings resolved more precisely):

```bash
N_STRUCTURES=2000 BISECT_TOL_NM=0.02 SLOPE_PROBE_NM=0.25 \
    OUTPUT_DIR=analyses/thickness_sensitivity/results_fine \
    sbatch analyses/thickness_sensitivity/run.sh
```

Costs about the same as the large run (bisection depth adds ~2-3
sims per side per threshold).

### Overriding individual knobs

```bash
# Reproduce a specific run — same seed gives the exact same
# structures and probes:
SEED=42 sbatch analyses/thickness_sensitivity/run.sh

# Serial mode (useful for profiling, or when SLURM only gives 1 CPU):
N_JOBS=1 sbatch analyses/thickness_sensitivity/run.sh
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

**Perceptibility distance + censoring.** Look at
`delta_e_2_by_bin.png` and read the `⌐N%` labels above each box.
A high fraction censored in a bin means "many stacks in this bin are
essentially insensitive — their grid resolution barely matters".
A low fraction censored means the boxplot's spread is representative.

**Source dominance.** Look at `grid_comparison_by_source.png`. If
the HC and random panels diverge (HC needing a finer grid than
random), the grid choice is dominated by high-chroma structures —
consistent with intuition that directed search finds thin-layer
"interference-critical" operating points.

**Sanity check.** Read `sensitivity.json`'s `config` block. Confirm
`layer_selection == "most_sensitive_by_slope"`, `cap_nm`,
`bisect_tol_nm`, and `doubling_schedule_nm` match what you asked
for.

---

## Provenance

- Original driver of the 5 nm → 2 nm token-grid decision. See the
  vocab-size docstring in `src/materials_vocab.py` for the resulting
  code invariants.
- Kept as a diagnostic — every time the pool sampler or the optical
  simulator changes (new synthetic strategy, different substrate,
  non-normal incidence), re-run and check whether the grid choice
  still holds.
