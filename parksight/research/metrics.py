"""Evaluation metrics for T1 (station-day counts) and T2 (hotspots).

Prediction frames (from models.run_cv):
  T1: [model, fold, seed, police_station, date, y, pred, naive_mae]
      naive_mae = that station's training-window seasonal-naive MAE (see add_naive_mae).
  T2: [model, fold, seed, h3_r8, week, y, pred, prev]
      y = next-week hotspot label, pred = score in [0, 1], prev = this week's label
      (emerging subset = prev == 0). Optional thr = per-row threshold from the
      inner validation split; without it F1 uses 0.5.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

EPS = 1e-6


def mase(y, pred, naive_mae) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))) / naive_mae)


def add_naive_mae(pred: pd.DataFrame, sd: pd.DataFrame, folds) -> pd.DataFrame:
    """Add naive_mae: mean |n_t - n_{t-7}| per station over days before fold.train_end."""
    sd = sd.sort_values(["police_station", "date"]).copy()
    sd["err"] = (sd["n"] - sd.groupby("police_station")["n"].shift(7)).abs()
    parts = [
        sd[sd["date"] < f.train_end].groupby("police_station")["err"].mean()
        .rename("naive_mae").reset_index().assign(fold=f.idx)
        for f in folds
    ]
    return pred.merge(pd.concat(parts), on=["fold", "police_station"], how="left")


def bootstrap_ci(values, n: int = 2000, seed: int = 12345) -> tuple[float, float]:
    """95% CI of the mean, resampling the given values (one value per station)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def _station_scores(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, station): MASE, mean squared error, Poisson deviance.
    Seeds are averaged first; a (near) zero denominator is floored and flagged."""
    d = pred.copy()
    d["flag"] = d["naive_mae"] < EPS
    mu = d["pred"].clip(lower=EPS)
    y = d["y"]
    d["ae"] = (y - d["pred"]).abs()
    d["se"] = (y - d["pred"]) ** 2
    d["dev"] = 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))
    d["den"] = d["naive_mae"].clip(lower=EPS)
    # mean over seeds within each (model, station, fold, date), then over rows
    d = d.groupby(["model", "police_station", "fold", "date"], as_index=False).agg(
        ae=("ae", "mean"), se=("se", "mean"), dev=("dev", "mean"),
        den=("den", "first"), flag=("flag", "first"))
    d["mase"] = d["ae"] / d["den"]
    return d.groupby(["model", "police_station"], as_index=False).agg(
        mase=("mase", "mean"), mse=("se", "mean"), dev=("dev", "mean"), flag=("flag", "max"))


def summarise_t1(pred: pd.DataFrame) -> pd.DataFrame:
    st = _station_scores(pred)
    d = pred.copy()
    d["ae"] = (d["y"] - d["pred"]).abs()
    mae = d.groupby("model")["ae"].mean()
    base = mae.get("seasonal_naive", np.nan)  # pooled MASE: test-window denominator
    rows = []
    for m, g in st.groupby("model"):
        lo, hi = bootstrap_ci(g["mase"])
        rlo, rhi = bootstrap_ci(g["mse"])
        dlo, dhi = bootstrap_ci(g["dev"])
        rows.append(dict(
            model=m, mase_macro=g["mase"].mean(), mase_lo=lo, mase_hi=hi,
            mase_pooled=mae[m] / base, rmse=np.sqrt(g["mse"].mean()),
            rmse_lo=np.sqrt(rlo), rmse_hi=np.sqrt(rhi),
            deviance=g["dev"].mean(), dev_lo=dlo, dev_hi=dhi,
            n_flagged_stations=int(g["flag"].sum())))
    return pd.DataFrame(rows).sort_values("mase_macro").reset_index(drop=True)


def wilcoxon_vs(pred: pd.DataFrame, model_a: str, model_b: str) -> float:
    """Two-sided Wilcoxon p-value on per-station MASE, paired over stations."""
    s = _station_scores(pred).pivot(index="police_station", columns="model", values="mase")
    return float(wilcoxon(s[model_a], s[model_b]).pvalue)


def _t2_scores(g: pd.DataFrame) -> dict:
    if g["y"].nunique() < 2:
        return dict(pr_auc=np.nan, f1=np.nan, brier=np.nan, n=len(g))
    thr = g["thr"] if "thr" in g else 0.5
    return dict(pr_auc=average_precision_score(g["y"], g["pred"]),
                f1=f1_score(g["y"], g["pred"] >= thr, zero_division=0),
                brier=brier_score_loss(g["y"], g["pred"]), n=len(g))


def summarise_t2(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in pred.groupby("model"):
        # average seeds so a row is scored once
        keys = [c for c in ("fold", "h3_r8", "week") if c in g]
        agg = {c: "mean" if c in ("pred", "thr") else "first"
               for c in g.columns if c not in keys + ["model", "seed"]}
        g = g.groupby(keys, as_index=False).agg(agg)
        for subset, h in (("all", g), ("emerging", g[g["prev"] == 0])):
            rows.append(dict(model=m, subset=subset, **_t2_scores(h)))
    return pd.DataFrame(rows)
