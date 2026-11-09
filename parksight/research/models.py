"""T1 model registry, tiny inner-validation tuning, and the cross-validation runner. Owner: Afzal.

Models (all classical ML):
  baselines      seasonal_naive (same weekday last week), ma7 (mean of the last 7 days)
  linear         poisson_glm (log link), ridge and lasso on log1p(y); count features enter as log1p
  trees          random_forest (Poisson criterion; the Gao et al. 2019 replication baseline),
                 lightgbm (hackathon parameters, Poisson objective)

Tuning is deliberately tiny: one knob per model, 4-5 values, scored on the LAST 14 DAYS OF THE TRAINING
WINDOW (an inner time-blocked split), never on the test block.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, PoissonRegressor, Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from parksight.research import data as D
from parksight.research import features as F

T1_MODELS = ["seasonal_naive", "ma7", "poisson_glm", "ridge", "lasso", "random_forest", "lightgbm"]
BASELINES = {"seasonal_naive", "ma7"}
LINEAR = {"poisson_glm", "ridge", "lasso"}
LOG_TARGET = {"ridge", "lasso"}
STOCHASTIC = {"random_forest", "lightgbm"}      # the only models whose result depends on the seed
EPS = 1e-6

# One hyperparameter per model, 4-5 values. A first version used 3 values and every model that has
# a capacity knob picked the edge of its grid, so the grids were widened once (see
# E3_chosen_params.csv, which flags any model that still sits on an edge).
GRIDS = {
    "poisson_glm": [{"alpha": a} for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0)],
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "lasso": [{"alpha": a} for a in (1e-4, 1e-3, 1e-2, 1e-1)],
    "random_forest": [{"min_samples_leaf": m} for m in (5, 10, 20, 40)],
    "lightgbm": [{"n_estimators": n} for n in (100, 200, 400, 800)],
}
# parameters copied from the hackathon forecast (parksight/models/train_forecast.py BASE_PARAMS)
LGB_BASE = dict(learning_rate=0.04, num_leaves=24, subsample=0.85, colsample_bytree=0.85,
                min_child_samples=30, reg_lambda=1.0, reg_alpha=0.2, verbose=-1)


def _design(model, train, test, features):
    """Feature matrices for one model type. Everything is fitted on `train` only."""
    num = [f for f in features if f != F.STATION]
    use_stn = F.STATION in features
    if model in LINEAR:
        tr, te = F.log_counts(train), F.log_counts(test)
        parts = []
        if num:
            parts.append(("num", StandardScaler(), num))
        if use_stn:
            parts.append(("stn", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [F.STATION]))
        ct = ColumnTransformer(parts, sparse_threshold=0)
        return ct.fit_transform(tr), ct.transform(te)
    Xtr, Xte = train[num].copy(), test[num].copy()
    if use_stn:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train[[F.STATION]])
        Xtr["station_code"] = enc.transform(train[[F.STATION]]).ravel()
        Xte["station_code"] = enc.transform(test[[F.STATION]]).ravel()
    return Xtr, Xte


def _make(model, params, seed):
    if model == "poisson_glm":
        return PoissonRegressor(alpha=params["alpha"], max_iter=3000)
    if model == "ridge":
        return Ridge(alpha=params["alpha"])
    if model == "lasso":
        return Lasso(alpha=params["alpha"], max_iter=50000)
    if model == "random_forest":
        return RandomForestRegressor(n_estimators=300, criterion="poisson", n_jobs=-1,
                                     random_state=seed, **params)
    if model == "lightgbm":
        return lgb.LGBMRegressor(objective="poisson", random_state=seed, n_jobs=4, **{**LGB_BASE, **params})
    raise ValueError(f"unknown model {model}")


def _fit_predict_once(model, params, train, test, features, seed):
    Xtr, Xte = _design(model, train, test, features)
    y = train["y"].values
    est = _make(model, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model in LOG_TARGET:
            est.fit(Xtr, np.log1p(y))
            pred = np.expm1(est.predict(Xte))
        else:
            est.fit(Xtr, y)
            pred = est.predict(Xte)
    return np.clip(pred, 0.0, None)


def station_scale(train: pd.DataFrame) -> pd.Series:
    """Per-station in-sample seasonal-naive (lag-7) MAE on the training rows: the MASE denominator."""
    a = (train["y"] - train["lag7"]).abs()
    return a.groupby(train[F.STATION]).mean().clip(lower=EPS)


def _choose(model, train, features, seed):
    """Pick the grid value with the lowest scaled MAE on the last 14 days of the training window."""
    grid = GRIDS[model]
    if len(grid) == 1:
        return grid[0]
    val_start = train["date"].max() - pd.Timedelta(days=13)
    itr, iva = train[train["date"] < val_start], train[train["date"] >= val_start]
    scale = iva[F.STATION].map(station_scale(itr)).values
    best, best_score = grid[0], np.inf
    for p in grid:
        pred = _fit_predict_once(model, p, itr, iva, features, seed)
        score = float(np.mean(np.abs(iva["y"].values - pred) / scale))
        if score < best_score:
            best, best_score = p, score
    return best


def _fit_predict(model, train, test, seed, features):
    if model == "seasonal_naive":
        return test["lag7"].values.astype(float), {}
    if model == "ma7":
        return test["roll7"].values.astype(float), {}
    params = _choose(model, train, features, seed)
    return _fit_predict_once(model, params, train, test, features, seed), params


def fit_predict(task: str, model: str, train: pd.DataFrame, test: pd.DataFrame,
                seed: int = 0, features: list[str] | None = None) -> np.ndarray:
    if task != "t1":
        raise NotImplementedError("only T1 (counts) is implemented; T2 hotspots come with E4")
    return _fit_predict(model, train, test, seed, features or F.FEATURES_T1)[0]


def fold_frames(feat: pd.DataFrame, fold: D.Fold):
    """Train = every row strictly before the test block; test = the 14-day block."""
    train = feat[feat["date"] < fold.train_end]
    test = feat[(feat["date"] >= fold.test_start) & (feat["date"] <= fold.test_end)]
    return train, test


def run_cv(task: str, models: list[str], seeds=(0, 1, 2), features: list[str] | None = None,
           feat: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """One row per (model, seed, fold, test row): police_station, date, y, pred, naive, scale, params."""
    if task != "t1":
        raise NotImplementedError("only T1 (counts) is implemented; T2 hotspots come with E4")
    feat = F.t1_features(D.load_station_day()) if feat is None else feat
    features = features or F.FEATURES_T1
    out = []
    for fold in D.day_folds():
        train, test = fold_frames(feat, fold)
        scale = station_scale(train)
        base = test[[F.STATION, "date", "y"]].assign(
            fold=fold.idx, naive=test["lag7"].values, scale=test[F.STATION].map(scale).values)
        for model in models:
            for seed in (seeds if model in STOCHASTIC else seeds[:1]):
                t0 = time.time()
                pred, params = _fit_predict(model, train, test, seed, features)
                out.append(base.assign(model=model, seed=seed, pred=pred, params=str(params)))
                if verbose:
                    print(f"[run_cv] fold {fold.idx} {model:15s} seed {seed} "
                          f"{time.time() - t0:5.1f}s {params}", flush=True)
    return pd.concat(out, ignore_index=True)
