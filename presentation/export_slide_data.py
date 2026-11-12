"""Chart-ready data and figures for the presentation (Claude Design upload pack). Owner: Afzal.

Reads only committed results (results/tables, results/predictions) and re-fits the two LightGBM training curves
(the per-round loss numbers were never saved, only the figure). Writes presentation/attach/:
  data/*.csv      one tidy CSV per chart, column names written for humans
  figures/*.pdf   the report's existing figures (diagram, training curve, ...)

Run:  python presentation/export_slide_data.py        (about 2 minutes; the training curves take most of it)
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_poisson_deviance, precision_recall_curve  # noqa: E402

from parksight.research import data as D  # noqa: E402
from parksight.research import features as F  # noqa: E402
from parksight.research import models as Mo  # noqa: E402

TABLES, PRED, FIGS = ROOT / "results" / "tables", ROOT / "results" / "predictions", ROOT / "results" / "figures"
OUT = ROOT / "presentation" / "attach"
DATA = OUT / "data"


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA / name, index=False)
    print(f"  {name:36s} {len(df):4d} rows  columns: {', '.join(df.columns)}")


def evening_gap() -> pd.DataFrame:
    e1 = pd.read_csv(TABLES / "E1_eda.csv", comment="#")
    h = e1[e1["section"] == "by_hour"].pivot(index="hour", columns="metric", values="value").reset_index()
    h["hour"] = h["hour"].astype(int)
    h = h.rename(columns={"n_tickets": "tickets", "n_devices": "active_devices"})
    h["tickets_pct_of_peak"] = 100 * h["tickets"] / h["tickets"].max()
    h["devices_pct_of_peak"] = 100 * h["active_devices"] / h["active_devices"].max()
    return h[["hour", "tickets", "active_devices", "tickets_pct_of_peak", "devices_pct_of_peak"]]


def benchmark() -> pd.DataFrame:
    b = pd.read_csv(TABLES / "E2_benchmark.csv")
    b = b[b["family"] == "same_for_all"].copy()
    b["bias_pct"] = 100 * b["rel_bias"]
    b["cap_label"] = b["cap"].map(lambda c: "none" if pd.isna(c) else str(int(c)))
    names = {"naive": "Naive count", "ht_oracle_cap2": "Weighted, cap 2", "ht_oracle_cap5": "Weighted, cap 5",
             "ht_oracle_cap12": "Weighted, cap 12 (hackathon)", "ht_oracle_cap50": "Weighted, cap 50",
             "ht_oracle_uncapped": "Weighted, no cap", "ht_estimated_cap12": "Weighted, estimated exposure"}
    b["method_label"] = b["method"].map(names)
    return b[["severity", "method", "method_label", "cap_label", "bias_pct", "mae", "kendall_tau_station",
              "bias2", "variance"]].rename(columns={"kendall_tau_station": "station_ranking_tau"})


def exposure_mechanism() -> pd.DataFrame:
    d = pd.read_csv(TABLES / "E2_exposure_mechanism.csv")
    d = d[d["severity"] == 1.0]
    d["series"] = d["mechanism"] + " / " + d["estimator"]
    w = d.pivot(index="hour", columns="series", values="e_hat").reset_index()
    truth = d.drop_duplicates("hour").set_index("hour")["e_true"]
    w.insert(1, "true_exposure", w["hour"].map(truth))
    return w.rename(columns={
        "ticket_level / distinct_devices": "est_tickets_dropped_distinct_devices (E2 original)",
        "ticket_level / device_days": "est_tickets_dropped_device_days",
        "device_level / distinct_devices": "est_devices_censored_distinct_devices",
        "device_level / device_days": "est_devices_censored_device_days (best)"})


def counts() -> pd.DataFrame:
    c = pd.read_csv(TABLES / "E3_counts.csv")
    return c[["model", "mase_macro", "mase_lo", "mase_hi", "mase_pooled", "rmse", "deviance"]]


def ablation() -> pd.DataFrame:
    a = pd.read_csv(TABLES / "E3_ablation.csv")
    return a[["variant", "mase_macro", "mase_lo", "mase_hi", "delta_vs_full"]]


def hotspots() -> pd.DataFrame:
    h = pd.read_csv(TABLES / "E4_hotspots.csv")
    return h[["model", "subset", "pr_auc", "pr_auc_lo", "pr_auc_hi", "delta_vs_ref", "f1", "brier", "n"]]


def pr_curves() -> pd.DataFrame:
    p = pd.read_parquet(PRED / "E4_hotspots_predictions.parquet")
    keys = ["fold", "h3_r8", "week"]
    avg = p.groupby(["model"] + keys, as_index=False).agg(pred=("pred", "mean"), y=("y", "first"), prev=("prev", "first"))
    grid = np.round(np.arange(0, 1.0001, 0.02), 2)
    rows = []
    for subset in ("all", "emerging"):
        d = avg if subset == "all" else avg[avg["prev"] == 0]
        for model, g in d.groupby("model"):
            if model == "persistence":          # binary scores give a degenerate curve
                continue
            prec, rec, _ = precision_recall_curve(g["y"], g["pred"])
            for r in grid:                      # interpolated precision: best precision at recall >= r
                ok = prec[rec >= r]
                rows.append(dict(subset=subset, model=model, recall=r, precision=float(ok.max()) if len(ok) else 0.0))
    return pd.DataFrame(rows)


def conformal() -> pd.DataFrame:
    c = pd.read_csv(TABLES / "E5_conformal.csv")
    c["coverage_pct"] = 100 * c["coverage"]
    c["method"] = c["method"].map({"global": "One global interval", "tier": "One interval per volume tier"})
    c["tier"] = pd.Categorical(c["tier"], ["low", "mid", "high", "all"], ordered=True)
    return c.sort_values(["tier", "method"])[["tier", "method", "coverage_pct", "width"]].assign(target_pct=80)


def training_curves() -> pd.DataFrame:
    """Per-round loss of the tuned-setting LightGBM on the last fold of each task (same code as E7)."""
    rows = []
    f = D.day_folds()[-1]
    feat = F.t1_features(D.load_station_day())
    tr = feat[feat["date"] < f.train_end]
    vs = tr["date"].max() - pd.Timedelta(days=13)
    a, b = tr[tr["date"] < vs], tr[tr["date"] >= vs]
    est = lgb.LGBMRegressor(objective="poisson", random_state=0, n_jobs=4, n_estimators=800, **Mo.LGB_BASE)
    est.fit(a[F.FEATURES_T1], a["y"])
    for k in range(1, 801):
        rows.append(dict(task="Next-day counts (Poisson deviance)", round=k,
                         train_loss=mean_poisson_deviance(a["y"], est.predict(a[F.FEATURES_T1], num_iteration=k).clip(1e-6)),
                         validation_loss=mean_poisson_deviance(b["y"], est.predict(b[F.FEATURES_T1], num_iteration=k).clip(1e-6))))
    f = D.week_folds()[-1]
    hw = F.t2_features(D.load_hex_week())
    tr = hw[hw["week"] < f.train_end]
    vs = tr["week"].max() - 1
    a, b = tr[tr["week"] < vs], tr[tr["week"] >= vs]
    feats = [c for c in F.FEATURES_T2 if c in tr.columns]
    rec = {}
    est = Mo._make_t2("lightgbm", {"n_estimators": 200}, 0)
    est.fit(a[feats], a["y_next"].astype(int), eval_set=[(a[feats], a["y_next"].astype(int)), (b[feats], b["y_next"].astype(int))],
            eval_names=["train", "val"], eval_metric="binary_logloss", callbacks=[lgb.record_evaluation(rec)])
    for k, (t, v) in enumerate(zip(rec["train"]["binary_logloss"], rec["val"]["binary_logloss"]), 1):
        rows.append(dict(task="Next-week hotspots (log loss)", round=k, train_loss=t, validation_loss=v))
    return pd.DataFrame(rows)


def stations_pca() -> pd.DataFrame:
    e6 = pd.read_csv(TABLES / "E6_unsup.csv", comment="#")
    s = e6[e6["section"] == "station"].pivot(index="key", columns="metric", values="value").reset_index()
    s = s.rename(columns={"key": "police_station"})
    s["cluster"] = s["cluster"].astype(int)
    ev = e6[e6["section"] == "explained_variance"].set_index("key")["value"]
    s["pc1_variance_pct"], s["pc2_variance_pct"] = 100 * ev["PC1"], 100 * ev["PC2"]
    return s


def folds() -> pd.DataFrame:
    rows = []
    for f in D.day_folds():
        rows.append(dict(task="Counts (days)", fold=f.idx + 1, train_start="2023-12-08", train_end=str((f.train_end - pd.Timedelta(days=1)).date()),
                         test_start=str(f.test_start.date()), test_end=str(f.test_end.date())))
    for f in D.week_folds():
        rows.append(dict(task="Hotspots (weeks)", fold=f.idx + 1, train_start="week 4", train_end=f"week {f.train_end - 1}",
                         test_start=f"week {f.test_start}", test_end=f"week {f.test_end}"))
    return pd.DataFrame(rows)


def key_numbers() -> pd.DataFrame:
    meta = json.loads((ROOT / "parksight/data/processed/meta.json").read_text(encoding="utf-8"))
    g, b = evening_gap().set_index("hour"), pd.read_csv(TABLES / "E2_benchmark.csv")
    s1 = b[(b.family == "same_for_all") & (b.severity == 1.0)].set_index("method")
    mech = pd.read_csv(TABLES / "E2_exposure_mechanism_summary.csv").query("severity == 1.0")
    m = mech.set_index(["mechanism", "estimator"])
    c, h = pd.read_csv(TABLES / "E3_counts.csv").set_index("model"), pd.read_csv(TABLES / "E4_hotspots.csv")
    h = h.set_index(["model", "subset"])
    e5 = pd.read_csv(TABLES / "E5_conformal.csv").set_index(["method", "tier"])
    rows = [
        ("tickets", meta["total_violations"], "", "dataset"),
        ("police stations", meta["n_stations"], "", "dataset"),
        ("days of data", 151, "10 Nov 2023 to 8 Apr 2024", "dataset"),
        ("enforcement devices", meta["n_devices"], "", "dataset"),
        ("active devices at 11:00", int(g.loc[11, "active_devices"]), "peak", "E1"),
        ("active devices at 19:00", int(g.loc[19, "active_devices"]), "evening rush", "E1"),
        ("tickets in the 10:00 hour", int(g.loc[10, "tickets"]), "peak hour", "E1"),
        ("tickets in the 19:00 hour", int(g.loc[19, "tickets"]), "", "E1"),
        ("share of tickets 17:00-21:00 (%)", 0.2, "596 of 298,445", "E1"),
        ("share of tickets 00:00-06:00 (%)", 34, "timestamp caveat", "E1"),
        ("naive count bias, real severity (%)", round(100 * s1.loc["naive", "rel_bias"], 1), "", "E2"),
        ("weighted, cap 12 bias (%)", round(100 * s1.loc["ht_oracle_cap12", "rel_bias"], 1), "hackathon cap", "E2"),
        ("weighted, cap 50 bias (%)", round(100 * s1.loc["ht_oracle_cap50", "rel_bias"], 1), "", "E2"),
        ("weighted, no cap bias (%)", round(100 * s1.loc["ht_oracle_uncapped", "rel_bias"], 1), "", "E2"),
        ("variance, no cap", round(s1.loc["ht_oracle_uncapped", "variance"]), "tickets^2 per cell", "E2"),
        ("variance, cap 12", round(s1.loc["ht_oracle_cap12", "variance"]), "tickets^2 per cell", "E2"),
        ("estimated-exposure bias (%)", round(100 * s1.loc["ht_estimated_cap12", "rel_bias"], 1), "distinct devices, ticket thinning", "E2"),
        ("exposure overestimate at 09:00, original (x)", round(m.loc[("ticket_level", "distinct_devices"), "ratio_at_09"], 1), "", "E2b"),
        ("exposure overestimate at 09:00, device-days + device-level censoring (x)", round(m.loc[("device_level", "device_days"), "ratio_at_09"], 1), "", "E2b"),
        ("macro MASE, last-week forecast", round(c.loc["seasonal_naive", "mase_macro"], 3), "1.0 = this forecast", "E3"),
        ("macro MASE, lasso (best)", round(c.loc["lasso", "mase_macro"], 3), "", "E3"),
        ("macro MASE, random forest", round(c.loc["random_forest", "mase_macro"], 3), "", "E3"),
        ("macro MASE, LightGBM", round(c.loc["lightgbm", "mase_macro"], 3), "", "E3"),
        ("PR-AUC all, rank-by-last-week baseline", round(h.loc[("persistence_impact", "all"), "pr_auc"], 3), "", "E4"),
        ("PR-AUC all, best classifier (SVM)", round(h.loc[("svm", "all"), "pr_auc"], 3), "", "E4"),
        ("PR-AUC emerging, baseline", round(h.loc[("persistence_impact", "emerging"), "pr_auc"], 3), "", "E4"),
        ("PR-AUC emerging, logistic regression", round(h.loc[("logreg", "emerging"), "pr_auc"], 3), "", "E4"),
        ("global interval coverage, busiest stations (%)", round(100 * e5.loc[("global", "high"), "coverage"], 1), "target 80", "E5"),
        ("per-tier interval coverage, busiest stations (%)", round(100 * e5.loc[("tier", "high"), "coverage"], 1), "target 80", "E5"),
        ("PCA PC1 variance (%)", 30.3, "when, not how much, stations record", "E6"),
        ("full pipeline reproduction time (minutes)", 11, "run_all.py --force", "reproducibility check"),
    ]
    out = pd.DataFrame(rows, columns=["what", "value", "note", "source"])
    out["value"] = pd.Series([int(v) if float(v).is_integer() else v for v in out["value"]], dtype=object)
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    print("writing", DATA)
    save(evening_gap(), "c01_evening_gap.csv")
    save(benchmark(), "c02_benchmark_bias_variance.csv")
    save(exposure_mechanism(), "c03_exposure_curves.csv")
    save(counts(), "c04_count_models.csv")
    save(ablation(), "c05_count_ablation.csv")
    save(hotspots(), "c06_hotspot_models.csv")
    save(pr_curves(), "c07_pr_curves.csv")
    save(conformal(), "c08_conformal_coverage.csv")
    save(training_curves(), "c09_training_curves.csv")
    save(stations_pca(), "c10_station_pca.csv")
    save(folds(), "c11_folds.csv")
    save(key_numbers(), "key_numbers.csv")
    for f in ("F0_method", "F1_evening_gap", "F2_benchmark", "E3_counts", "E4_pr_curves", "F5_training_curve", "F4_pca_kmeans"):
        shutil.copy2(FIGS / f"{f}.pdf", OUT / "figures" / f"{f}.pdf")
    print("copied 7 report figures to", OUT / "figures")


if __name__ == "__main__":
    main()
