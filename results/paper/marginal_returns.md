# Marginal returns: pretraining compute vs search

Median DeltaE00. Greedy rows are on a validation slice, built the same way for the study and the production run; search rows are on test rows 0-499 (failed rows count as +inf).

| system | compute (FLOPs) | decoding | median ΔE₀₀ |
|---|---|---|---|
| study, lowest usable budget | 1.00e+14 | greedy (best size, best LR) | 11.89 |
| study, highest usable budget | 8.29e+15 | greedy (best size, best LR) | 8.98 |
| production, best greedy LR (9.7e-06) | 1.00e+17 | greedy | 9.12 |
| production, shipped LR (0.00031) | 1.00e+17 | greedy | 12.12 |
| production, shipped LR | 1.00e+17 | search, light: 16 candidates, top-1, 5 steps | A 2.81 / B 2.18 |
| production, shipped LR | 1.00e+17 | search, mid: 50 candidates, top-1, 20 steps | A 1.20 / B 0.96 |
| production, shipped LR | 1.00e+17 | search, high: 250 candidates, top-1, 40 steps | A 0.64 / B 0.50 |

Bridge, same decoding and statistic on other rows: eval_tier_a_greedy 12.57 (n=2000); eval_tier_b_greedy 7.98 (n=2000); eval_val_greedy 12.25 (n=2000).
