"""scripts/fit_beta2.py must recover a beta2 optimum that is planted.

Synthetic cells in lr_tuning.py's results format: a quadratic in
log10(1 - beta2) with its minimum at 0.97, seed offsets shared within a block,
an LR optimum that moves with beta2, and divergence at high rates. The
analysis has to find 0.97 inside its interval, keep the grid winner's close
neighbours in the "indistinguishable" set, and see no beta2 effect when none
is planted.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fit_beta2 as F  # noqa: E402

BETAS = (0.9, 0.95, 0.98, 0.99, 0.999)
SIZES = [(40, 1, 100_493, 3_454_000, 2.28e-3, 12.4),
         (128, 3, 965_285, 3_566_000, 6e-4, 11.0),
         (288, 5, 6_606_789, 6_140_000, 2.12e-4, 10.4)]


def make(root: Path, effect: float, rng_seed: int, beta_star: float = 0.97):
    rng = np.random.default_rng(rng_seed)
    root.mkdir(parents=True, exist_ok=True)
    xs = math.log10(1 - beta_star)
    for d, se, n, D, prior, base in SIZES:
        for seed in (42, 43, 44):
            block = rng.normal(0, 0.6)
            for b in BETAS:
                g = effect * (math.log10(1 - b) - xs) ** 2 + rng.normal(0, 0.35)
                lr_opt = prior * 1.3 * ((1 - b) / 0.02) ** 0.15
                res = []
                for lr in prior * 2 ** np.arange(-1.5, 1.51, 0.5):
                    v = base + block + g + 2 * np.log2(lr / lr_opt) ** 2 \
                        + rng.normal(0, 0.15)
                    div = lr > prior * (2.3 if b == 0.999 else 3.5)
                    res.append({"lr": float(lr),
                                "best_val_loss": 80.0 if div else 6.1,
                                "final_val_de": 27.0 if div else float(v),
                                "de_result": {"per_example_de":
                                              list(rng.normal(v, 8, 64))}})
                sfx = "" if seed == 42 else f"_s{seed}"
                json.dump({"d_model": d, "slot_encoder_layers": se,
                           "n_params": n, "train_examples": D, "epochs": 1,
                           "beta2": b, "seed": seed, "results": res},
                          open(root / f"lr_search_ep1_lim{D}_d{d}_se{se}"
                                      f"_bs256_b2{b:g}{sfx}.json", "w"))


def test_recovers_planted_optimum(tmp_path):
    make(tmp_path, effect=1.5, rng_seed=1)
    res = F.analyse(F.load_cells([str(tmp_path)]), n_boot=50, seed=0)
    m = res["by_metric"]["pooled"]
    q = m["pooled"]["quadratic"]
    assert q["beta2_star"] is not None
    lo, hi = q["beta2_ci"]
    assert lo <= 0.97 <= hi, (lo, hi)
    assert m["anova"]["p_beta2"] < 1e-6
    # 0.999 is far from the optimum and must be distinguishable from it.
    assert 0.999 not in m["pooled"]["indistinguishable_from_winner"]
    assert res["decision"]["verdict"] == "single"
    # Every alternative was compared on the same examples, paired.
    assert res["example_bootstrap"]


def test_no_effect_is_not_reported_as_one(tmp_path):
    make(tmp_path, effect=0.0, rng_seed=3)
    res = F.analyse(F.load_cells([str(tmp_path)]), n_boot=50, seed=0)
    assert res["by_metric"]["pooled"]["anova"]["p_beta2"] > 0.05


def test_extension_rates_merge_into_their_cell(tmp_path):
    """beta2x writes extra rates for a cell to a second directory; the
    analysis must treat them as more points on the same curve, keep the
    prior from the main grid, and not double-count a cell."""
    main, ext = tmp_path / "main", tmp_path / "ext"
    make(main, effect=1.5, rng_seed=1)
    ext.mkdir()
    src = sorted(main.glob("*d40_se1*_b20.999.json"))[0]
    d = json.load(open(src))
    lrs = [t["lr"] for t in d["results"]]
    prior = float(np.exp(np.mean(np.log(lrs))))
    d["results"] = [{"lr": prior * f, "best_val_loss": 80.0,
                     "final_val_de": 27.0, "de_result": {}}
                    for f in (4.0, 4.0 * 2 ** 0.5)]
    json.dump(d, open(ext / src.name, "w"))

    cells = F.load_cells([str(main), str(ext)])
    assert len(cells) == 45
    c = next(c for c in cells if c["size"] == (40, 1) and c["seed"] == 42
             and c["beta2"] == 0.999)
    assert len(c["lrs"]) == 9 and list(c["lrs"]) == sorted(c["lrs"])
    assert abs(c["prior"] / prior - 1) < 1e-9
    assert c["diverged"][-2:].all()
