"""Report-ready LaTeX tables built from the saved result CSVs. Owner: Afzal.

pandas' raw to_latex keeps column names like `mase_macro`, whose underscores break LaTeX, and it
cannot show a confidence interval next to its estimate. These helpers write clean tables (escaped
model names, "estimate [low, high]" cells) that report/sections/results_*.tex include.

Run on its own to rebuild the .tex files from the CSVs without re-running any experiment:
    python -m parksight.research.report_tables
"""
import pandas as pd

from parksight.research import paths


def _tex(name: str) -> str:
    return str(name).replace("_", r"\_")


def _ci(v, lo, hi, nd=3) -> str:
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _write(df: pd.DataFrame, name: str, column_format: str) -> None:
    (paths.TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=False, column_format=column_format), encoding="utf-8")


def write_e3() -> None:
    s = pd.read_csv(paths.TABLES / "E3_counts.csv")
    t = pd.DataFrame({
        "Model": s["model"].map(_tex),
        r"Macro MASE [95\% CI]": [_ci(a, b, c) for a, b, c in zip(s["mase_macro"], s["mase_lo"], s["mase_hi"])],
        "Pooled MASE": s["mase_pooled"].map("{:.3f}".format),
        "RMSE": s["rmse"].map("{:.1f}".format),
        "Poisson dev.": s["deviance"].map("{:.1f}".format),
    })
    _write(t, "E3_counts", "lllrr")

    a = pd.read_csv(paths.TABLES / "E3_ablation.csv")
    ta = pd.DataFrame({
        "Variant": a["variant"].map(_tex),
        r"Macro MASE [95\% CI]": [_ci(x, y, z) for x, y, z in zip(a["mase_macro"], a["mase_lo"], a["mase_hi"])],
        r"$\Delta$ vs full": a["delta_vs_full"].map("{:+.3f}".format),
    })
    _write(ta, "E3_ablation", "llr")


def write_e4() -> None:
    h = pd.read_csv(paths.TABLES / "E4_hotspots.csv")
    allr = h[h["subset"] == "all"].set_index("model")
    emr = h[h["subset"] == "emerging"].set_index("model")
    ref = "persistence_impact"
    rows = []
    for m in allr.sort_values("pr_auc", ascending=False).index:
        a, e = allr.loc[m], emr.loc[m]
        rows.append({
            "Model": _tex(m),
            r"PR-AUC, all [95\% CI]": _ci(a["pr_auc"], a["pr_auc_lo"], a["pr_auc_hi"]),
            r"PR-AUC, emerging [95\% CI]": _ci(e["pr_auc"], e["pr_auc_lo"], e["pr_auc_hi"]),
            r"$\Delta$ vs rank baseline, all": "--" if m == ref else f"{a['delta_vs_ref']:+.3f}",
            r"$\Delta$ vs rank baseline, emerging": "--" if m == ref else f"{e['delta_vs_ref']:+.3f}",
            "F1, all": f"{a['f1']:.3f}",
            "Brier, all": "--" if m == ref else f"{a['brier']:.3f}",   # rank scores are not probabilities
        })
    _write(pd.DataFrame(rows), "E4_hotspots", "lllrrrr")


def write_all() -> None:
    write_e3()
    write_e4()


if __name__ == "__main__":
    write_all()
    print("wrote E3_counts.tex, E3_ablation.tex, E4_hotspots.tex")
