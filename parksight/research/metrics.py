"""Metrics for T1 and summary tables.

FIRST DRAFT written by Afzal so that E3 could run. Abdul owns this file.

Headline = MACRO MASE: for each station, MAE of the model divided by that station's in-sample
seasonal-naive (lag-7) MAE on the TRAINING window of the same fold, averaged over folds and seeds;
then the mean over stations. One station has 34k tickets, so pooled numbers would be decided by it.
Pooled MASE (test-window denominator = what the hackathon's 0.73 used) is reported next to it, and
the two are NOT comparable with each other.
"""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6
BOOT_SEED = 12345


def mase(y, pred, scale) -> float:
    """Plain MASE: mean |y - pred| divided by a naive-forecast scale."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(y - pred)) / max(float(scale), EPS))


def poisson_deviance(y, pred) -> float:
    y, mu = np.asarray(y, float), np.clip(np.asarray(pred, float), EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def bootstrap_ci(values, n: int = 2000, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling the given values (stations)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def station_mase(pred: pd.DataFrame) -> pd.DataFrame:
    """Per (model, station) MASE averaged over folds and seeds. `pred` needs the run_cv columns."""
    d = pred.copy()
    d["flag_scale"] = d["scale"] < EPS
    d["ae"] = (d["y"] - d["pred"]).abs() / d["scale"].clip(lower=EPS)
    per = (d.groupby(["model", "police_station", "seed", "fold"], as_index=False)
             .agg(mase=("ae", "mean"), flagged=("flag_scale", "max")))
    return (per.groupby(["model", "police_station"], as_index=False)
               .agg(mase=("mase", "mean"), flagged=("flagged", "max")))


def summarise_t1(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per model: macro MASE with 95% CI over stations, pooled MASE, RMSE, Poisson deviance."""
    sm = station_mase(pred)
    rows = []
    for model, sub in sm.groupby("model"):
        lo, hi = bootstrap_ci(sub["mase"].values)
        p = pred[pred["model"] == model]
        per_seed = []
        for _, s in p.groupby("seed"):
            per_seed.append(dict(
                pooled=np.abs(s["y"] - s["pred"]).sum() / max(np.abs(s["y"] - s["naive"]).sum(), EPS),
                rmse=float(np.sqrt(np.mean((s["y"] - s["pred"]) ** 2))),
                dev=poisson_deviance(s["y"], s["pred"])))
        ps = pd.DataFrame(per_seed).mean()
        rows.append(dict(model=model, macro_mase=sub["mase"].mean(), mase_lo=lo, mase_hi=hi,
                         pooled_mase=ps["pooled"], rmse=ps["rmse"], poisson_deviance=ps["dev"],
                         n_seeds=p["seed"].nunique(), flagged_stations=int(sub["flagged"].sum())))
    return pd.DataFrame(rows).sort_values("macro_mase").reset_index(drop=True)


def wilcoxon_vs(pred: pd.DataFrame, model_a: str, model_b: str) -> float:
    """Paired Wilcoxon signed-rank p-value on per-station MASE of two models."""
    sm = station_mase(pred).pivot(index="police_station", columns="model", values="mase")
    diff = (sm[model_a] - sm[model_b]).dropna()
    if (diff == 0).all():
        return 1.0
    return float(stats.wilcoxon(diff).pvalue)
