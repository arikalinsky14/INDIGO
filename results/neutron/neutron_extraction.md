# NEUTRON extraction for the INDIGO head-to-head

Source paper: Wang & Guo, "NEUTRON: Neural particle swarm optimization for material-aware inverse design of structural color," iScience 25(5), 104339 (2022), doi 10.1016/j.isci.2022.104339. Read from the PMC full-text PDF (PMC9117888). Page numbers are that PDF's.
Source code: github.com/hammer-wang/NEUTRON (master, cloned Nov 16, 2026). File and cell references below are to that repo.

**Supplement not read.** The supplemental PDF (Figures S1 to S4) could not be retrieved: the Cell/Elsevier and Europe PMC links failed or were rate limited. Everything below attributed to S1 to S4 comes from how the main text describes them.

**Method labels used below**
- NEUTRON: M3DN material ranking + MDN-initialized PSO
- MDN only: M3DN samples, no PSO
- PSO random-init: PSO with random starting positions
- Data-gen PSO: the PSO used to create training labels

Where paper and code disagree, both are given.

---

## Read this first: three facts that change the comparison plan

1. **The paper reports no test-set ΔE00 ≤ 2 success rate for any method.** The only test-set numbers are an AUC of 0.91 for the material classifier and "average test CIEDE2000 values … slightly above ten" for MDN only. Its quantitative method comparisons are 5 pictures (Table 2) and 1 chrome target (Table 1). A head-to-head on their test set must be run by us. It cannot be read from the paper.
2. **Their test rows are not "reachable by construction."** Each row is a random sRGB target paired with a random material combination. The label records whether data-gen PSO reached ΔE00 ≤ 2. The paper says only about 25% did. Our test sets are reachable by construction; theirs are not. Any pass rate on their test set must be interpreted with that in mind.
3. **Data and model are public.** The README links the train/val/test pickles on Mendeley Data (dataset 54d84d88p8) and a trained checkpoint on Google Drive. So their exact test set, and their model, can be rerun.

---

## 1. Test-set results

| Item | Value | Method | Quote / source | Location |
|---|---|---|---|---|
| Success rate at ΔE00 ≤ 2 | not reported | any | No such number in text or tables | none |
| Success rate during data generation | about 25% | Data-gen PSO (random init, random material combination) | "the average success ratio of obtaining the accurate color following the random data generation process is about 25%." | Dataset generation, p. 5 |
| Mean ΔE00, test | "slightly above ten" (two values, not given exactly) | MDN only (M3DN and an ablation; not stated which two) | "Both average test CIEDE2000 values are slightly above ten and can be further improved by finetuning with the PSO." | STAR Methods, M3DN, p. 16; plotted in Fig. S2C (not read) |
| Median ΔE00, test | not reported | any | none | none |
| Classifier test AUC | 0.91 | M3DN classification head | "The material classification network achieves excellent performance with an AUC of 0.91." | STAR Methods, p. 16; Fig. S2B |
| Dataset size and split | 400,000 total; 380,000 / 10,000 / 10,000 | n/a | "The entire dataset is randomly split into 380,000/10,000/10,000 data points for training/validation/testing." | Dataset generation, p. 5 |
| Number of test targets | 10,000 rows (each row is one target paired with one material combination) | n/a | Same quote as above | p. 5 |
| Val/test composition | primitive materials only (no blended n,k) | n/a | "Importantly, both validation and test sets are based on primitive materials only" | p. 5 |
| Which split headline numbers use | test | n/a | "Unless stated otherwise, all results reported in the paper are based on the test sets." | p. 5 |
| PSO settings for a test-set NEUTRON evaluation | not reported (no such evaluation exists) | NEUTRON | none | none |
| Runtime per test target | not reported | any | none | none |
| Hardware | NVIDIA RTX 3090 (24 GB) for training. CPU count not reported. | M3DN training | "We train the M3DN with PyTorch (Paszke et al., 2017) on NVIDIA RTX 3090 GPUs with 24 GB of internal memory." | STAR Methods, p. 16 |
| Hardware (code) | PSO objective runs in a CPU `multiprocessing.Pool(32)` for pictures, `Pool(100)` for chrome, `Pool(16)` in the data generator | all PSO | `pool = multiprocessing.Pool(args.pop_size)` (pop_size default 32) | `pso_baseline_mdn_init_multitask.py`; notebook cell 48; `simulation/multilayer_sRGB.py` |

**PSO settings, general (paper).**
- "we randomly sampled 32 thickness designs from the MDN to be used as the initial positions" (STAR Methods, p. 17)
- "optimize until convergence with a tolerance level of 1 × 10⁻⁵" (p. 17; the exponent is garbled in the PDF text, and the code uses `ftol=1e-5`)
- "w = 0.9; c1 = 0.5; c2 = 0.3" (p. 17)
- Particle count and maximum iterations: not reported in the paper. The code values are below.

---

## 2. Picture reconstruction

**Table 2 (p. 10), verbatim values.** The method columns are ΔE00 averaged over the image.

| Picture | Unique target colours | Material combination (chosen by M3DN ranking) | MDN only | PSO random-init | NEUTRON | Time (s) |
|---|---|---|---|---|---|---|
| White Orchard | not reported | Al2O3 / Ge / Al / Al2O3 / Zn | 18.55 | 5.16 | 3.76 | 9,917 |
| Tulip Field | not reported | Ta2O5 / Au / Ti / Al2O3 / Au | 19.45 | 4.34 | 3.48 | 9,246 |
| Great Wall | not reported | HfO2 / Au / Ti / SiO2 / Au | 21.46 | 6.45 | 5.69 | 6,835 |
| Flower (style-transferred) | not reported | Ta2O5 / Au / Ti / Al2O3 / Au | 19.37 | 5.00 | 3.93 | 11,249 |
| Street (style-transferred) | not reported | ZnO / Ti / Au / MgF2 / Au | 21.60 | 4.45 | 3.28 | 7,349 |

- **ΔE ≤ 2 rate per picture:** not reported; only image-average ΔE00 is given (Fig. 4C caption: "Average CIEDE2000 computed over entire images before and after particle swarm optimization fine-tuning").
- **Which method the time belongs to:** not stated in the table. In the code, the timer wraps only the per-colour PSO loop of the MDN-initialized run (`start_time` to `end_time` in `pso_baseline_mdn_init_multitask.py`). That makes it the NEUTRON PSO stage, excluding the 100,000-combination material screening.
- **"PSO" column means random-init PSO on the same chosen material combination.** In code this is the `--random_init` flag of the same script. The material choice still comes from the M3DN classifier.
- **Per-pixel cost:** "the particle swarm optimization process for finetuning a single pixel requires ~10 seconds" (Picture quantization details, p. 17; the symbol is garbled).
- **Material search space:** "for all 100,000 allowed material combinations" (Fig. 4B caption, p. 8). In code, `itertools.product(D, M, M, D, M)` = 10⁵, which allows the same metal in both absorber layers. Training data forbade that ("the two adjacent absorber layers must be composed of different materials", p. 4).

**Preprocessing**

| Item | Paper | Code |
|---|---|---|
| Thumbnail size | not reported ("pictures with more than 200,000 pixels", p. 6) | `image.thumbnail((512, 512), Image.ANTIALIAS)` (max side 512) |
| Quantization step | 10. "we quantize the pixel values and find the quantization step size of 10" (p. 6); "s = 10 led to the best tradeoff" (p. 17) | `step_size = 10` |
| Rounding vs truncation | "group similar pixels values to the nearest integer values that can be divided by a quantization step size s" (p. 17), which implies rounding | **Truncation (floor):** `RGB_quantize = (RGB / step_size).astype(int) * step_size`. The paper and code disagree. |
| Compression ratio | ">50X" (Results, p. 6) vs ">40X" (STAR Methods, p. 17), inconsistent within the paper | n/a |
| Unique colours per picture | not reported | computed at run time (`np.unique` after quantization) |
| PSO per colour | 32 MDN samples as initial positions (p. 17) | `GlobalBestPSO(n_particles=32, bounds=([5,5,5,5],[250,15,15,250]), ftol=1e-5, ftol_iter=5)`, `optimize(..., 100)` (32 particles, at most 100 iterations) |
| Thickness to simulator | integers ("we constrain the thicknesses of all layers to be integers", p. 5) | `.astype(int)` (truncation) |
| Hardware for timings | not reported | CPU pool of 32 processes; machine not stated |

---

## 3. Cr (chrome) replacement

**Target definition**

| Source | What it says |
|---|---|
| Paper | No numeric target. Goal is "an appearance similar to chrome" (p. 6). Fig. 3D compares spectra to "a 100 nm thick Cr film" (Fig. 3 caption, p. 7). Cr is removed from the metal list: "we exclude Cr from the available metals" (p. 7). |
| Code (notebook cell 43) | `get_color(nk_dict_cr, [20, 20, 20, 20, 20])`: five Cr layers of 20 nm. In the simulator, that is 4 coherent Cr layers of 20 nm plus a 20 nm incoherent Cr "back metal", on 5 µm glass (`5e3` in `multilayer_thin_film.py`), not a single 100 nm coherent film. |
| Code (cell 48) | Target Lab hard-coded: `[84.96137356, -0.24662783, -3.42720451]` |

**Search and PSO settings (notebook cells 44 to 48)**
- **Screening:** all `product(D, M, M, D, M)` with Cr removed, giving 10 × 9 × 9 × 10 × 9 = 72,900 combinations. The paper says 720,000 (p. 7), which looks like a typo or a different count.
- **Top-K kept:** the top 100 by predicted probability. Paper: "fine-tune the designs with PSO for the top 100 combinations" (p. 6).
- **Per combination:** 100 MDN samples as initial positions; PSO with `n_particles=100`, `ftol=1e-5`, `optimize(..., 1000)` (100 particles, at most 1,000 iterations); thicknesses rounded up (`np.ceil`) from MDN samples; not cast to int inside the PSO objective.

**Table 1 (p. 7).** Each cell is M3DN only → after PSO. Caption: "Chrome color designs based on M3DN only and the fine-tuned design by PSO (bold)."

| Design | Layers (top to bottom) | Thickness, nm | ΔE00 |
|---|---|---|---|
| I | HfO2 / W / Ge / SiO2 / Zn | 120→105, 13→15, 15→10, 122→83, 100 | 7.6 → **0.8** |
| II | HfO2 / Zn / Ge / Al2O3 / Ni | 12→7, 14→14, 8→13, 160→98, 100 | 15.8 → **1.2** |
| III | SiO2 / Ag / Ge / SiO2 / Au | 273→250, 14→14, 15→15, 147→130, 100 | 9.8 → **1.4** |
| IV | HfO2 / Zn / Ge / Al2O3 / Au | 61→5, 14→14, 15→15, 202→207, 100 | 25.0 → **1.6** |
| V | HfO2 / Zn / Ge / MgF2 / Au | 97→93, 15→15, 12→11, 249→218, 100 | 5.4 → **1.6** |

Text: "All five final designs obtained CIEDE2000 lower than two" (p. 7).

**Checking your notebook reading (HfO2 / W / Ge / SiO2 / Zn at 105 / 15 / 10 / 83 / 100, ΔE 2.35): partly correct.**
- **Structure and thicknesses:** confirmed. They match Table 1 design I after PSO, and notebook cell 55 simulates exactly `[105, 15, 10, 83, 100]`.
- **ΔE00 = 2.35 is a different number.** Notebook cell 54 computes ΔE00 between the Cr target Lab and the Lab of the sRGB triple `[199, 205, 214]`, an 8-bit rounded colour, not the simulated design. The paper's value for this design is **0.8**.
- **Notebook printout differs.** Cell 53 prints the design-I PSO result as `106 / 14 / 12 / 86 / 100`, not `105 / 15 / 10 / 83`. The published thicknesses may come from a different run.
- **Rescoring:** score `105 / 15 / 10 / 83 / 100` in their physics against the cell-48 target Lab, and expect about 0.8.

**Baselines on this target:** only M3DN-only vs M3DN + PSO (Table 1). There is no random-init PSO or other method. "PSO significantly improved the initial designs predicted by the M3DN in terms of the CIEDE2000" (p. 7).

---

## 4. Simulation settings

Code: `simulation/multilayer_sRGB.py` (data generator), `simulation/multilayer_thin_film.py`, and the notebook.

| Setting | Paper | Code |
|---|---|---|
| Stack | "five-layer optical thin films with two absorbing layers sandwiched by two dielectric layers" (p. 3); "fixing the bottom reflective layer at 100 nm" (p. 4) | Air / D / M / M / D (coherent) / back metal (incoherent) / glass (incoherent) / air: `inc_list = ['i','c','c','c','c','i','i','i']` |
| Back metal incoherent? | not reported | **Yes.** 5th layer `'i'` |
| Glass | not reported | n = 1.5 constant (`[1.5, 1]` appended). **Thickness: 50 µm in the data generator** (`5e4` in `multilayer_sRGB.py`), **5 µm in `multilayer_thin_film.py` and the notebook** (`5e3`), which the picture and chrome runs use via `get_color`. |
| Thickness bounds | dielectrics "[5, 250] nm"; absorbers "[5, 15] nm" (p. 5); back metal fixed 100 nm (p. 4) | `bounds=([5,5,5,5],[250,15,15,250])`; back metal 100 |
| Integer thickness | "we constrain the thicknesses of all layers to be integers" (p. 5) | `.astype(int)` (truncate) in data gen and pictures; chrome uses `np.ceil` on init, float during PSO |
| Wavelengths | "for the wavelength range [400, 700] nm with a step size of 10 nm" (p. 4); 31 points, "62-dimensional vector" per material (p. 4) | `np.arange(0.4, 0.701, 0.01)` µm = 400 to 700 nm, 31 points; n,k via `scipy interp1d` from the database |
| Illuminant | not reported (Colour package cited, p. 4) | D65: `SDS_ILLUMINANTS['D65']` |
| Observer | not reported | CIE 1931 2°: `MSDS_CMFS['CIE 1931 2 Degree Standard Observer']` |
| XYZ to Lab | not reported | `colour.XYZ_to_Lab(XYZ / 100)` (Colour default white point, D65) |
| Target sRGB to Lab | not reported | `colour.XYZ_to_Lab(colour.sRGB_to_XYZ(RGB / 255))` |
| Polarization | not reported | average of s and p reflectance: `(res['R'] + res_t['R']) / 2` |
| Incidence angle | not reported | normal (0) |
| Quantity | reflection (p. 4) | `R` only |
| TMM package | "transfer-matrix method (Byrnes, 2016)" (p. 4) | `tmm.inc_tmm` |
| ΔE formula | CIEDE2000 | `colour.delta_E_CIE2000` (default kL = kC = kH = 1) |

---

## 5. Dataset generation

| Item | Paper | Code (`simulation/multilayer_sRGB.py`) |
|---|---|---|
| Target colours | "we randomly sample the sRGB color target from the 3D sRGB space with a value range [0, 255] for each dimension" (p. 4) | `RGB = (rng.uniform(0, 1, size=3) * 256).astype(int)`: uniform integer sRGB 0 to 255 |
| Materials | "randomly sampled materials from ten candidate metal materials" and ten dielectrics (p. 3); "the two adjacent absorber layers must be composed of different materials" (p. 4) | dielectrics with replacement; the two absorbers without replacement; back metal free |
| Blending | "randomly mixing two dielectrics or two metals" by "convex combinations of their complex refractive indices" (Eq. 1, p. 5); mixing probability not reported | **50%:** `if rng.binomial(1, 0.5)`; per-layer weight `alphas = rng.uniform(0, 1, size=5)`; linear mix of complex n directly (not in ε) |
| Blended rows in data | "adding 50,000 additional data points from mixture materials leads to greater performance improvement" (algorithmic choices, p. 8) | n/a |
| Labels | "we use PSO to optimize the thickness of the top four layers" (p. 4); success label if ΔE00 ≤ 2 (p. 5) | `GlobalBestPSO(n_particles=32, ..., ftol=1e-4, ftol_iter=10)`, `optimize(..., args.iters)` with default `iters=100`; random init; label `acc = 1 if deltaE <= 2` |
| Success rate of labels | "about 25%" (p. 5) | n/a |
| Dataset link | Mendeley Data (Key Resources) | README: data.mendeley.com/datasets/54d84d88p8; trained model on Google Drive |

---

## 6. Background

**Why ΔE ≤ 2:** "CIEDE2000 lower than or equal to two is almost imperceivable by untrained eyes" (Dataset generation, p. 5). No citation is given for the threshold in that sentence.

**Prior methods discussed.** None are compared numerically against NEUTRON.

| Method | What the paper says | Numbers |
|---|---|---|
| Unni et al. (tandem/MDN) | "their method fixes the material selections for the alternating high refractive index" (p. 3) | not reported |
| Gao et al.; Dai et al. | "both works can only output a single design given a user-specified color input" (p. 2) | not reported |
| Wang et al. (RL) | "this method only designs a single color at a time" (p. 2) | not reported |
| Shokooh-Saremi & Magnusson (PSO/GA) | "Both optimization methods took a few hours to converge for a single design target" (p. 2) | qualitative only |
| Liu et al. (GA, metasurfaces) | "the achievable area in the CIE diagram is expanded by 106%" (p. 2) | 106% gamut area |
| Single-layer PSO ablation | "adjusting a single layer's thickness leads to a deteriorated average CIEDE2000 of 13.36 (see Figure S1) compared to 3.48 when tuning the top four layers" | Tulip Field, Fig. S1 (not read), p. 8 |

---

## Implications for our rescoring and benchmark

1. **Match their physics exactly when rescoring:**
   - Normal incidence; s and p averaged.
   - D65 illuminant, CIE 1931 2° observer, 31 points from 400 to 700 nm.
   - 4 coherent layers on a 100 nm incoherent back metal, on incoherent glass (n = 1.5), then air.
   - Integer thicknesses, truncated.
   - **Glass thickness:** 5 µm for chrome and pictures, 50 µm for the dataset. With both glass and back metal incoherent, glass thickness should have little effect when k = 0; check it once.
2. **Test-set head-to-head.** Use their `test.pkl` (10,000 rows, primitive only). Two ways to report:
   - (a) Per row, with INDIGO constrained to that row's material combination. Compare to the data-gen label success of about 25%, and to their MDN-only checkpoint plus PSO if we rerun it.
   - (b) INDIGO choosing materials from the 20-material pool. This is not their task; report it separately.
3. **Chrome.** Use target Lab `[84.96, -0.25, -3.43]`, Cr excluded, 5 layers with absorbers 5 to 15 nm and dielectrics 5 to 250 nm. Their best is 0.8 (design I), and all top-5 are below 2.
4. **Runtime.** Their per-picture times are the PSO stage only, on a 32-process CPU pool, with unreported unique-colour counts.

## Our recount of unique colours (not from the paper)

Recomputed from the repo's `photo_data/` with their preprocessing: `thumbnail((512, 512))` and step-10 truncation. Current Pillow's default resampling stands in for the deprecated `ANTIALIAS`, so counts may differ slightly.

| File | Original size | After thumbnail | Unique colours, raw | Unique after quantization | Table 2 time (s) | Implied s per colour |
|---|---|---|---|---|---|---|
| the_white_orchard.jpg | 800×589 | 512×377 | 152,692 | 1,962 | 9,917 | about 5.1 |
| tulip_fields.jpg | 800×631 | 512×404 | 95,570 | 2,073 | 9,246 | about 4.5 |
| great_wall.jpg | 1920×1275 | 512×340 | 62,970 | 1,283 | 6,835 | about 5.3 |
| flower_guo_group.jpg (probably "Flower") | 512×680 | 386×512 | 162,521 | 3,216 | 11,249 | about 3.5 |
| "Street" | file not identified (`bridge_guo_group.jpg` is referenced in scripts but absent; `san_juan.jpg` is a candidate) | | | | 7,349 | |

The ~5 s per colour on a 32-process pool fits the paper's "~10 seconds" per pixel only roughly. Treat these mappings and counts as our reconstruction, not reported values.
