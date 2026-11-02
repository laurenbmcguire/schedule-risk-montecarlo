"""Single matplotlib style helper for every chart in the repo."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# DejaVu Sans has no semibold face; matplotlib falls back to bold and logs a warning each time.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BG = "#FFFBF6"
INK = "#241A18"
MUTED = "#5E4C46"
GRID = "#EAE0D5"
CRIMSON = "#BE1E3C"
DEEP_CRIMSON = "#8E1630"
GOLD = "#B08A4E"
SAND = "#E3D2B4"

FIGSIZE = (10, 5.625)  # 1600 x 900 px at dpi=160
DPI = 160


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 15,
            "axes.titlelocation": "left",
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 10.5,
            "legend.frameon": False,
            "axes.prop_cycle": matplotlib.cycler(color=[CRIMSON, GOLD, DEEP_CRIMSON, SAND, MUTED]),
        }
    )


def new_figure(figsize=FIGSIZE):
    apply()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path
