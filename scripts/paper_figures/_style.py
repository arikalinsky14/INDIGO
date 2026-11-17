"""Shared style for INDIGO paper figures.

Every figure script imports this module so that fonts, sizes, colors and
export settings are identical across the paper.

  * Widths: single column 8.3 cm, double column 17 cm.
  * Sans-serif at 8 pt (Arial, else its metric twin Liberation Sans).
  * Okabe-Ito palette (colorblind-safe).
  * Export: vector PDF with TrueType (type 42) fonts, so every label stays
    editable text in Illustrator/Inkscape, plus a 600 dpi PNG.
  * No figure titles: captions live in the paper.

Text shown in a figure is kept in a LABELS dict at the top of each script,
so wording can be changed without touching drawing code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "results" / "paper" / "figures"

CM = 1 / 2.54
SINGLE = 8.3 * CM
DOUBLE = 17.0 * CM
FONT_PT = 8
SMALL_PT = 7

# Okabe & Ito (2008). Names are the published ones.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}
GREY = "#8C8C8C"
LIGHT = "#E6E6E6"
INK = "#1A1A1A"


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": SMALL_PT,
        "ytick.labelsize": SMALL_PT,
        "legend.fontsize": SMALL_PT,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "Liberation Sans",
        "mathtext.sf": "Liberation Sans",
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,     # TrueType: text stays editable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save(fig, stem: str, *, tiff_size_cm=None) -> list[Path]:
    """Write <stem>.pdf and <stem>.png (600 dpi). Figure size is kept exact
    (no bbox_inches='tight') so the width matches the column."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=600, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    return paths


def save_tiff(fig, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import io
    from PIL import Image
    p = OUT_DIR / f"{stem}.tiff"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(p, dpi=(600, 600), compression="tiff_lzw")
    return p


def write_provenance(fig_id: str, line: str) -> None:
    """Keep one provenance line per figure in PROVENANCE.md (idempotent)."""
    p = OUT_DIR / "PROVENANCE.md"
    head = "# Paper figure provenance\n\nOne line per figure: script, inputs, fields.\n\n"
    lines = p.read_text().splitlines()[4:] if p.exists() else []
    lines = [l for l in lines if l and not l.startswith(f"- **{fig_id}**")]
    lines.append(f"- **{fig_id}** {line}")
    lines.sort()
    p.write_text(head + "\n".join(lines) + "\n")


def lab_in_srgb(lab, tol=0.002) -> bool:
    """True if the Lab color is inside the sRGB gamut (no clipping needed)."""
    return lab_to_srgb(lab, clip=False) is not None


def lab_to_srgb(lab, clip=True):
    """CIE Lab (D65) -> sRGB in [0, 1], clipped. Used only to paint swatches.
    With clip=False, returns None for a color outside the sRGB gamut."""
    import numpy as np
    L, a, b = (float(v) for v in lab)
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    finv = lambda t: t ** 3 if t > d else 3 * d * d * (t - 4 / 29)
    X, Y, Z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z])
    if not clip and (rgb.min() < -0.002 or rgb.max() > 1.002):
        return None
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return tuple(np.clip(rgb, 0, 1))
