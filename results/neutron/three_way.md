# NEUTRON comparison, scored in NEUTRON's physics (Nov 18)

Image-averaged CIEDE2000, lower is better. INDIGO: production checkpoint
(sha256 4fd024434fcc0c02), 250 candidates, top 10 refined for 100 steps, 5 layers,
thicknesses snapped to whole nm by our simulator, then scored with NEUTRON's
inc_tmm geometry and n,k data. Each picture: 150 colours drawn with probability
proportional to pixel count (seed 1, same colours in both INDIGO arms), so the
plain mean estimates the image average; 95% bootstrap interval.

| image | NEUTRON (Table 2) | PSO, random init | INDIGO, NEUTRON's 5 materials | INDIGO, all 20, any order, back metal | 
|---|---|---|---|---|
| White Orchard | 3.76 | 5.16 | 1.53 [1.28, 1.84] | 0.56 [0.49, 0.63] |
| Tulip Field | 3.48 | 4.34 | 1.46 [1.27, 1.66] | 0.41 [0.35, 0.46] |
| Great Wall | 5.69 | 6.45 | 2.12 [1.85, 2.43] | 3.08 [2.58, 3.61] |
| Flower | 3.93 | 5.00 | 1.12 [0.98, 1.27] | 1.05 [0.89, 1.22] |

Table 1 chrome designs (their materials, ΔE to the Cr target): INDIGO 0.74, 0.20,
1.33, 0.20, 0.67 against their 0.8, 1.2, 1.4, 1.6, 1.6.

- Fixed-material arm: every row used the row's materials in order within their
  bounds. All-materials arm: layer 5 one of their metals at 100 nm (opaque, so
  their glass backside drops out); 0 violations. Their physics and ours agree to
  a median 0.02 to 0.04 ΔE on these stacks.
- Great Wall, all materials: the 69 colours over ΔE 2 are the dark ones (mean
  L* 17 vs 49 for the rest, same chroma 16). The fixed Au/Ti/Au template reaches
  them; the free arm, which picks Ge as the back metal in 55 of 150, does not.
- One row (173, all-materials arm) was rerun after the sampler crash fixed Nov 18.
- Not compared: run time (their per-image seconds cover an unknown number of
  colours) and their test split (not reachable).

Sources: rescore_final_fixed.json, rescore_final_full.json,
results/neutron_final_runs/{fixed,full}.
