"""Compact one-column LaTeX tables for the IEEE report, built only from results/tables/*.csv.
Usage: PYTHONPATH=. python experiments/make_ieee_tables.py   -> results/tables/ieee_*.tex"""
import pandas as pd

from parksight import config as C

T = C.ROOT / "results" / "tables"
esc = lambda s: s.replace("_", r"\_")


def save(name, header, rows, cols):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    (T / f"ieee_{name}.tex").write_text(
        "\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    c = pd.read_csv(T / "E3_counts.csv")
    save("counts", r"Model & Macro MASE [95\% CI] & RMSE & Dev.",
         [(esc(r.model), f"{r.mase_macro:.3f} [{r.mase_lo:.3f}, {r.mase_hi:.3f}]", f"{r.rmse:.1f}", f"{r.deviance:.1f}")
          for r in c.itertuples()], "lccc")

    h = pd.read_csv(T / "E4_hotspots.csv")
    a, e = h[h.subset == "all"].set_index("model"), h[h.subset == "emerging"].set_index("model")
    order = ["persistence", "persistence_impact", "logreg", "svm", "random_forest", "lightgbm"]
    save("hot", "Model & PR-AUC all & PR-AUC emerg. & F1 emerg.",
         [(esc(m), f"{a.pr_auc[m]:.3f}", f"{e.pr_auc[m]:.3f}", f"{e.f1[m]:.2f}") for m in order], "lccc")

    b = pd.read_csv(T / "E2_benchmark.csv")
    b = b[b.severity == 1.0]
    meths = ["naive", "ht_oracle_cap12", "ht_oracle_cap50", "ht_oracle_uncapped", "ht_estimated_cap12"]
    rows = []
    for m in meths:
        r = {f: b[(b.family == f) & (b.method == m)].iloc[0] for f in ("same_for_all", "station_hetero")}
        rows.append((esc(m), *[f"{100 * r[f].rel_bias:.1f} / {r[f].mae:.0f} / {r[f].kendall_tau_station:.2f}"
                               for f in r]))
    save("bench", "Method & Same curve for all & Station-specific curves", rows, "lcc")

    k = pd.read_csv(T / "E5_conformal.csv")
    rows = [(m, *[f"{100 * k[(k.method == m) & (k.tier == t)].coverage.iloc[0]:.0f} / {k[(k.method == m) & (k.tier == t)].width.iloc[0]:.0f}"
                  for t in ("low", "mid", "high", "all")]) for m in ("global", "tier")]
    save("conf", "Interval & Low & Mid & High & All", rows, "lcccc")


if __name__ == "__main__":
    main()
