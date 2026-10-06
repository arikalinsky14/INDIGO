"""Which tuning cells and IsoFLOP rungs did not bracket their optimum.

Two kinds of edge, both decided from results on disk:

LEARNING RATE (`lr_extension`). A cell's optimum is bracketed when, among the
rates that did not diverge, the Akima-interpolated argmin falls strictly
inside the second and second-to-last rates: src/scaling/porian.py
`tuned_optimum`, the rule the fit itself applies. When it does not:

  * best at the LOWEST rate: two more rates below, continuing the grid's step.
  * best at the HIGHEST stable rate, nothing tried above it: two more above.
  * best at the highest stable rate, and the next rate up DIVERGED: one rate
    at their geometric midpoint. The optimum sits somewhere on the way to the
    divergence cliff and the grid's 1.85x step cannot say where; repeated
    rounds halve the gap each time.

Extension trials are written next to the cell's file as
`<cell>_ext<k>.json` (lr_tuning.py --output-suffix), and `cell_trials`
merges them back, so the fit and the collector see one cell with more rates.

IsoFLOP RUNG (`rung_extension_sizes`). A rung whose ΔE minimum is its
smallest or largest model has no interior N*. Two more models beyond that
end, from the same shape-bounded ladder the sweep uses, trained at the
rung's FLOPs with a full stage-2 grid.

Torch-free, like the rest of the analysis stack.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.materials_vocab import VOCAB_SIZE  # noqa: E402
from src.scaling.porian import tuned_optimum  # noqa: E402

DIVERGENCE_VAL_LOSS = 2.0 * math.log(VOCAB_SIZE)
EXT_RE = re.compile(r"_ext(\d+)\.json$")


def ext_paths(main: Path) -> List[Path]:
    """The extension files of a cell, in round order."""
    stem = main.name[:-len(".json")]
    return sorted(main.parent.glob(f"{stem}_ext*.json"),
                  key=lambda p: int(EXT_RE.search(p.name).group(1))
                  if EXT_RE.search(p.name) else 0)


def cell_trials(main: Path) -> Tuple[Optional[dict], List[dict]]:
    """The cell's results file and every trial, its extensions' included,
    sorted by rate. (None, []) when the cell has not run."""
    if not main.is_file():
        return None, []
    d = json.load(open(main))
    trials = list(d.get("results", []))
    for p in ext_paths(main):
        trials += json.load(open(p)).get("results", [])
    return d, sorted(trials, key=lambda t: t["lr"])


def diverged(t: dict) -> bool:
    vl = t.get("best_val_loss")
    return vl is None or not math.isfinite(vl) or vl > DIVERGENCE_VAL_LOSS


def lr_extension(trials: List[dict], metric: str = "final_val_de"
                 ) -> Optional[Tuple[str, List[float]]]:
    """(kind, new rates) for a cell that did not bracket, else None."""
    ok = [t for t in trials if not diverged(t) and t.get(metric) is not None
          and math.isfinite(t[metric])]
    if len(ok) < 2:
        return None
    xs = [t["lr"] for t in ok]
    best, on_edge = tuned_optimum(xs, [t[metric] for t in ok])
    if not on_edge:
        return None
    step = (xs[-1] / xs[0]) ** (1 / (len(xs) - 1))
    if best < xs[1]:
        return "down", [xs[0] / step ** 2, xs[0] / step]
    cliff = [t["lr"] for t in trials if diverged(t) and t["lr"] > xs[-1]]
    if cliff:
        return "cliff", [math.sqrt(xs[-1] * min(cliff))]
    return "up", [xs[-1] * step, xs[-1] * step ** 2]


def next_ext_suffix(main: Path) -> str:
    rounds = [int(EXT_RE.search(p.name).group(1)) for p in ext_paths(main)]
    return f"_ext{max(rounds, default=0) + 1}"


def rung_extension_sizes(rung_ns: List[int], argmin_n: float,
                         ladder: List[Tuple[int, int, int]], k: int = 2,
                         min_ratio: float = 1.25) -> Tuple[str, list]:
    """Side ('left'/'right'/'') and up to k ladder sizes beyond that side.

    Sizes are taken at least `min_ratio` apart from the rung's end and each
    other, so the extension adds reach rather than near-duplicates; when the
    ladder runs out it says so by returning fewer than k.
    """
    lo, hi = min(rung_ns), max(rung_ns)
    if argmin_n <= lo * 1.0001:
        side = "left"
    elif argmin_n >= hi * 0.9999:
        side = "right"
    else:
        return "", []
    return side, sizes_beyond(rung_ns, side, ladder, k, min_ratio)


def soft_edge_sides(values: dict, sigma: float, k_sigma: float = 1.0) -> List[str]:
    """Ends of a rung that sit within k_sigma * sigma of its minimum.

    A rung can pass the bracketing rule (its argmin is interior) while an end
    point is as good as the minimum within seed noise; then the minimum has
    not really been located on that side. `values` maps N to the metric.
    """
    ns = sorted(values)
    vmin = min(values.values())
    out = []
    if values[ns[0]] - vmin < k_sigma * sigma:
        out.append("left")
    if values[ns[-1]] - vmin < k_sigma * sigma:
        out.append("right")
    return out


def sizes_beyond(rung_ns: List[int], side: str,
                 ladder: List[Tuple[int, int, int]], k: int = 2,
                 min_ratio: float = 1.25) -> list:
    """Up to k ladder sizes beyond one end of a rung, >= min_ratio apart."""
    lo, hi = min(rung_ns), max(rung_ns)
    pool = (sorted((s for s in ladder if s[0] < lo), key=lambda s: -s[0])
            if side == "left" else
            sorted((s for s in ladder if s[0] > hi), key=lambda s: s[0]))
    picked, edge = [], lo if side == "left" else hi
    for s in pool:
        far = (s[0] <= edge / min_ratio) if side == "left" else (s[0] >= edge * min_ratio)
        if far:
            picked.append(s)
            edge = s[0]
        if len(picked) == k:
            break
    return picked


CELL_RE = re.compile(r"^lr_search_ep(\d+)_lim(\d+)_d(\d+)_se(\d+)_bs(\d+)"
                     r"_b2([0-9.e-]+)(?:_s(\d+))?\.json$")


def arch_for(d_model: int, se: int):
    from src.scaling.configs import n_heads_for
    from src.scaling.flops import ArchSpec
    return ArchSpec(d_model=d_model, n_heads=n_heads_for(d_model),
                    head_mode="cross_attn", slot_encoder_layers=se,
                    decoder_layers=1)


def discover_cells(stage2_dir: str, budgets: List[float], beta2: float,
                   seed: int = 42, tol: float = 0.02) -> List[dict]:
    """Every finished seed-`seed` cell in `stage2_dir` whose training FLOPs
    land on one of `budgets`, as lr_grid_cells cell dicts. This is how
    STAGE=2edge points join their rung without anything keeping a list."""
    from src.scaling.flops import n_params, train_flops_per_example
    out = []
    for path in sorted(Path(stage2_dir).glob("lr_search_*.json")):
        m = CELL_RE.match(path.name)
        if not m:
            continue
        ep, lim, d, se, _bs, b2, s = m.groups()
        if abs(float(b2) - beta2) > 1e-12 or int(s or 42) != seed:
            continue
        arch = arch_for(int(d), int(se))
        flops = train_flops_per_example(arch) * int(lim) * int(ep)
        b = min(budgets, key=lambda x: abs(x - flops))
        if abs(flops - b) / b > tol:
            continue
        n = int(n_params(arch))
        out.append({"n_params": n, "d_model": int(d), "se": int(se),
                    "D": int(lim) * int(ep), "epochs": int(ep),
                    "limit": int(lim), "beta2": beta2, "seed": seed,
                    "lr": None, "budget": b, "M": int(lim) * int(ep) / n})
    return out
