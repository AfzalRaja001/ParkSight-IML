"""Methodology diagram -> results/figures/F0_method.pdf.  Usage: PYTHONPATH=. python experiments/make_method_diagram.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from parksight.research import style

COLS = {"data": "#56B4E9", "pred": "#009E73", "bench": "#E69F00", "eval": "#CC79A7"}


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=COLS[kind], ec="black", lw=0.6, alpha=0.35))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.7))


def main():
    style.apply()
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_xlim(-0.1, 10.0); ax.set_ylim(0, 4.3); ax.axis("off")
    w, h, g = 2.1, 0.95, 0.5
    xs = [i * (w + g) for i in range(4)]
    y1, y2, ym = 2.95, 0.15, 1.55
    row1 = ["298k tickets, 54 stations,\n151 days (IST, H3,\nseverity weights)", "Station-day grid (T1),\nhex-week grid (T2),\nleak-free features",
            "Baselines, GLM, ridge,\nlasso, random forest,\nLightGBM, SVM", "4 time-blocked folds;\ntuned on last block\nof the training window"]
    for x, t, k in zip(xs, row1, ["data", "data", "pred", "pred"]):
        box(ax, x, y1, w, h, t, k)
    for x in xs[:3]:
        arrow(ax, (x + w, y1 + h / 2), (x + w + g, y1 + h / 2))
    box(ax, xs[3], ym, w, h, "macro MASE, PR-AUC,\nbootstrap CI;\nconformal bands (E5)", "eval")
    arrow(ax, (xs[3] + w / 2, y1), (xs[3] + w / 2, ym + h))
    row2 = ["Pseudo-truth: tickets\n02:00-11:59 (240k)", "Thin each ticket with\nprob. e*(station, hour);\n2 families x 2 severities",
            "Naive count, HT oracle\n(cap 2 to none), HT with\nestimated e (cap 12)", "Bias, MAE, Kendall tau\nvs. truth, 20 seeds"]
    for x, t, k in zip(xs, row2, ["data", "bench", "bench", "eval"]):
        box(ax, x, y2, w, h, t, k)
    for x in xs[:3]:
        arrow(ax, (x + w, y2 + h / 2), (x + w + g, y2 + h / 2))
    arrow(ax, (xs[0] + w / 2, y1), (xs[0] + w / 2, y2 + h))
    ax.text(xs[2], 4.25, "Forecasting (E3, E4, E5)", fontsize=7, weight="bold", va="top", ha="center")
    ax.text(xs[1] + 0.6, 1.5, "Censoring benchmark (E2)", fontsize=7, weight="bold", va="center")
    style.savefig(fig, "F0_method")


if __name__ == "__main__":
    main()
