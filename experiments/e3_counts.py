"""E3: next-day ticket-count forecasting, classical model comparison (T1). Owner: Afzal.

Writes (all under results/):
  tables/E3_counts.csv / .tex        the ladder: macro MASE (+95% CI over stations), pooled MASE, RMSE, deviance
  tables/E3_wilcoxon.csv             paired Wilcoxon p-values on per-station MASE
  tables/E3_ablation.csv             best model with one feature group removed at a time
  tables/E3_chosen_params.csv        which grid values the inner validation picked (edge-of-grid check)
  predictions/E3_counts_predictions.parquet   every test prediction
  predictions/t1_best.parquet        hand-off to E5 (conformal): best model, calibration + test blocks
  figures/E3_counts.pdf              macro MASE with intervals

Run:  python experiments/e3_counts.py
"""
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from parksight.research import data as D  # noqa: E402
from parksight.research import features as F  # noqa: E402
from parksight.research import metrics as M  # noqa: E402
from parksight.research import models as MD  # noqa: E402
from parksight.research import paths, report_tables, style  # noqa: E402


def ladder(feat):
    pred = MD.run_cv("t1", MD.T1_MODELS, feat=feat)
    summary = M.summarise_t1(pred)
    summary.to_csv(paths.TABLES / "E3_counts.csv", index=False)
    pred.to_parquet(paths.PREDICTIONS / "E3_counts_predictions.parquet", index=False)
    return pred, summary


def chosen_params(pred):
    """How often each model's inner validation picked the first or last grid value.

    A model that always sits on an edge needs a wider grid, so this is printed and saved."""
    fits = (pred[~pred["model"].isin(MD.T1_BASELINES)]
            .groupby(["model", "fold", "seed"])["params"].first().reset_index())
    rows = []
    for model, g in fits.groupby("model"):
        grid = [list(p.values())[0] for p in MD.GRIDS[model]]
        picked = [list(ast.literal_eval(s).values())[0] for s in g["params"]]
        rows.append(dict(model=model, knob=list(MD.GRIDS[model][0].keys())[0], grid=str(grid),
                         n_fits=len(picked), frac_at_min=sum(v == min(grid) for v in picked) / len(picked),
                         frac_at_max=sum(v == max(grid) for v in picked) / len(picked),
                         picked=str(sorted(set(picked)))))
    out = pd.DataFrame(rows)
    out["edge_warning"] = (out["frac_at_min"] >= 0.75) | (out["frac_at_max"] >= 0.75)
    out.to_csv(paths.TABLES / "E3_chosen_params.csv", index=False)
    return out


def significance(pred):
    rows = []
    for m in MD.T1_MODELS:
        for ref in ("seasonal_naive", "random_forest"):
            if m != ref:
                rows.append(dict(model=m, versus=ref, wilcoxon_p=M.wilcoxon_vs(pred, m, ref)))
    out = pd.DataFrame(rows)
    out.to_csv(paths.TABLES / "E3_wilcoxon.csv", index=False)
    return out


def best_model(summary):
    return summary[~summary["model"].isin(MD.T1_BASELINES)].iloc[0]["model"]


def ablation(feat, best):
    full = F.FEATURES_T1
    variants = {"full": full}
    for name, cols in MD.FEATURE_GROUPS.items():
        variants[f"no_{name}_features"] = [f for f in full if f not in cols]
    rows = []
    for name, feats in variants.items():
        print(f"[E3 ablation] {best} with {name}", flush=True)
        s = M.summarise_t1(MD.run_cv("t1", [best], features=feats, feat=feat, verbose=False)).iloc[0]
        rows.append(dict(model=best, variant=name, mase_macro=s["mase_macro"], mase_lo=s["mase_lo"],
                         mase_hi=s["mase_hi"], rmse=s["rmse"]))
    out = pd.DataFrame(rows)
    out["delta_vs_full"] = out["mase_macro"] - out.loc[out["variant"] == "full", "mase_macro"].iloc[0]
    out.to_csv(paths.TABLES / "E3_ablation.csv", index=False)
    return out


def figure(summary):
    style.apply()
    s = summary.sort_values("mase_macro", ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    err = [s["mase_macro"] - s["mase_lo"], s["mase_hi"] - s["mase_macro"]]
    ax.barh(s["model"], s["mase_macro"], xerr=err, color=style.PALETTE[0], capsize=3)
    ax.axvline(1.0, color="#888888", lw=1, ls="--")
    ax.set_xlabel("macro MASE (lower is better; 1.0 = last-week forecast)")
    style.savefig(fig, "E3_counts")
    plt.close(fig)


def main():
    paths.ensure_dirs()
    t0 = time.time()
    feat = F.t1_features(D.load_station_day())
    print(f"[E3] {len(feat):,} feature rows, {feat['date'].min().date()} to {feat['date'].max().date()}",
          flush=True)
    pred, summary = ladder(feat)
    print(summary.round(3).to_string(), flush=True)
    print(chosen_params(pred).to_string(), flush=True)
    print(significance(pred).round(4).to_string(), flush=True)
    best = best_model(summary)
    print(ablation(feat, best).round(3).to_string(), flush=True)
    figure(summary)
    report_tables.write_e3()
    h = MD.conformal_handoff(best, feat=feat)
    h.to_parquet(paths.PREDICTIONS / "t1_best.parquet", index=False)
    print(f"[E3] hand-off for E5: {best}, {len(h):,} rows -> predictions/t1_best.parquet", flush=True)
    print(f"[E3] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
