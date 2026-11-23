"""Shared loader for the headline runs (INDIGO vs random).

Reads per-row results with scripts/paper_numbers.py's own loader, so every
figure counts rows exactly as results/paper/headline.json does: rows 0-499
of each tier, a failed or missing row is DeltaE = inf (a failure, never
dropped), chroma buckets C < 8, 8 <= C < 30, C >= 30 on the TARGET color.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import _style as S

sys.path.insert(0, str(S.REPO / "scripts"))
import paper_numbers as PN  # noqa: E402

HEADLINE = S.REPO / "results/paper/headline.json"

# headline.json run key -> display name and color
RUNS = {
    "random": ("Baseline, N = 250", S.GREY),
    "INDIGO_N50": ("INDIGO, N = 50", S.OI["sky"]),
    "INDIGO": ("INDIGO, N = 250", S.OI["blue"]),
}
BUCKETS = [b[0] for b in PN.BUCKETS]
BUCKET_LABEL = {"C<8": "$C^*$ < 8", "8<=C<30": "8 ≤ $C^*$ < 30", "C>=30": "$C^*$ ≥ 30"}


def headline() -> dict:
    return json.load(open(HEADLINE))


def load(run_key: str, h: dict | None = None):
    """Per-row arrays over tiers a then b (1000 rows):
    de (inf = failed/missing), bucket, tier, row index, and the raw row dicts."""
    h = h or headline()
    folder = S.REPO / h["runs"][run_key]["folder"]
    rows, settings, _ = PN.load_run(folder)
    n = h["rows_per_tier"]
    de, bk, tier, idx, raw = [], [], [], [], []
    for t in PN.TIERS:
        for i in range(n):
            r = rows[t].get(i)
            de.append(PN.de_of(r))
            lab = (r or {}).get("target_lab")
            bk.append(PN.bucket_of(PN.chroma(lab)) if lab else None)
            tier.append(t.upper()); idx.append(i); raw.append(r)
    return dict(de=np.array(de), bucket=np.array(bk, dtype=object), tier=np.array(tier),
                idx=np.array(idx), raw=raw, settings=settings,
                folder=h["runs"][run_key]["folder"])


def check_against_headline(run_key: str, d: dict, h: dict) -> None:
    """Refuse to draw if the rows disagree with headline.json."""
    for tset, tiers in (("A", ["A"]), ("B", ["B"]), ("A+B", ["A", "B"])):
        m = np.isin(d["tier"], tiers)
        for thr in PN.THRESHOLDS:
            got = float(np.mean(d["de"][m] < thr))
            want = h["runs"][run_key]["tiers"][tset]["all"][f"pass_lt_{thr:g}"]["est"]
            if abs(got - want) > 1e-9:
                sys.exit(f"{run_key} {tset} pass<{thr:g}: rows give {got}, headline.json {want}")
