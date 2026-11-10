"""One shared figure style: vector PDF, Okabe-Ito colour-blind-safe palette."""
from pathlib import Path

import matplotlib
from cycler import cycler

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def apply() -> None:
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, no Type 3 fonts
        "font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.figsize": (3.4, 2.4), "savefig.bbox": "tight",
    })


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    return path
