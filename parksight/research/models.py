"""Model registries, tiny inner-validation tuning, and the cross-validation runners. Owner: Afzal.

T1 (next-day ticket count per station), all classical ML:
  baselines   seasonal_naive (same weekday last week), ma7 (mean of the last 7 days)
  linear      poisson_glm (log link), ridge and lasso on log1p(y); count features enter as log1p
  trees       random_forest (Poisson criterion; the Gao et al. 2019 replication baseline),
              lightgbm (hackathon parameters, Poisson objective)

T2 (will a hexagon be a hotspot next week?), probabilities in [0, 1]:
  baselines   persistence (last week's label), persistence_impact (percentile rank of last week's impact)
  models      logreg, svm (linear SVM + sigmoid calibration), random_forest, lightgbm

Tuning is deliberately tiny: ONE knob per model, 3-5 values, scored on the LAST block of the training
window (14 days for T1, 2 weeks for T2): an inner time-blocked split, never the test block.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from parksight.research import data as D
from parksight.research import features as F
from parksight.research import metrics as M

EPS = 1e-6

# ----------------------------------------------------------------------------------- T1 (counts)
T1_MODELS = ["seasonal_naive", "ma7", "poisson_glm", "ridge", "lasso", "random_forest", "lightgbm"]
T1_BASELINES = {"seasonal_naive", "ma7"}
LINEAR = {"poisson_glm", "ridge", "lasso"}
LOG_TARGET = {"ridge", "lasso"}

LAG = ["lag1", "lag7", "lag14", "roll7", "roll28", "samedow_mean"]
CAL = ["weekday", "is_weekend", "is_holiday"]
STATION = ["station_code"]
FEATURE_GROUPS = {"lag": LAG, "calendar": CAL, "station": STATION}

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

# ----------------------------------------------------------------------------------- T2 (hotspots)
T2_MODELS = ["persistence", "persistence_impact", "logreg", "svm", "random_forest", "lightgbm"]
T2_BASELINES = {"persistence", "persistence_impact"}
T2_LINEAR = {"logreg", "svm"}
T2_IMPACT_COLS = ["impact_lag1", "impact_lag2", "impact_lag3", "impact_lag4", "impact_roll4", "neighbor_impact"]
T2_GRIDS = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "svm": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (5, 10, 20, 40)],
    "lightgbm": [{"n_estimators": n} for n in (25, 50, 100, 200)],
}
# (T2 grids were widened once for the same reason as the T1 grids: the first version, leaf sizes
#  1-10 and 100-400 trees, had the forest on its largest leaf and LightGBM near its fewest trees.)

STOCHASTIC = {"random_forest", "lightgbm"}      # the only models whose result depends on the seed


def models_for(task):
    return T1_MODELS if task == "t1" else T2_MODELS


# =================================================================================== T1 internals
def _design(model, train, test, features):
    """Feature matrices for one T1 model type. Everything is fitted on `train` only."""
    num = [f for f in features if f != "station_code"]
    use_stn = "station_code" in features
    if model in LINEAR:
        def logged(X):
            X = X.copy()
            for c in LAG:
                if c in X.columns:
                    X[c] = np.log1p(X[c])
            return X
        tr, te = logged(train), logged(test)
        parts = []
        if num:
            parts.append(("num", StandardScaler(), num))
        if use_stn:     # station enters the linear models as a category, not as an arbitrary number
            parts.append(("stn", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["police_station"]))
        ct = ColumnTransformer(parts, sparse_threshold=0)
        return ct.fit_transform(tr), ct.transform(te)
    return train[features], test[features]


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
    """Per-station in-sample seasonal-naive (lag-7) MAE on the training rows (used for tuning only)."""
    a = (train["y"] - train["lag7"]).abs()
    return a.groupby(train["police_station"]).mean().clip(lower=EPS)


def _choose(model, train, features, seed):
    """Pick the grid value with the lowest scaled MAE on the last 14 days of the training window."""
    grid = GRIDS[model]
    val_start = train["date"].max() - pd.Timedelta(days=13)
    itr, iva = train[train["date"] < val_start], train[train["date"] >= val_start]
    scale = iva["police_station"].map(station_scale(itr)).values
    best, best_score = grid[0], np.inf
    for p in grid:
        pred = _fit_predict_once(model, p, itr, iva, features, seed)
        score = float(np.mean(np.abs(iva["y"].values - pred) / scale))
        if score < best_score:
            best, best_score = p, score
    return best


def _fit_predict_t1(model, train, test, seed, features):
    if model == "seasonal_naive":
        return test["lag7"].values.astype(float), {}
    if model == "ma7":
        return test["roll7"].values.astype(float), {}
    params = _choose(model, train, features, seed)
    return _fit_predict_once(model, params, train, test, features, seed), params


def fold_frames(feat: pd.DataFrame, fold: D.Fold):
    """T1: train = every row strictly before the test block; test = the 14-day block."""
    train = feat[feat["date"] < fold.train_end]
    test = feat[(feat["date"] >= fold.test_start) & (feat["date"] <= fold.test_end)]
    return train, test


# =================================================================================== T2 internals
def _design_t2(model, train, test, features):
    Xtr, Xte = train[features].copy(), test[features].copy()
    if model in T2_LINEAR:
        for c in T2_IMPACT_COLS:
            if c in features:
                Xtr[c], Xte[c] = np.log1p(Xtr[c]), np.log1p(Xte[c])
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    return Xtr, Xte


def _make_t2(model, params, seed):
    if model == "logreg":
        return LogisticRegression(C=params["C"], max_iter=3000)
    if model == "svm":
        return CalibratedClassifierCV(LinearSVC(C=params["C"], max_iter=50000), cv=3, method="sigmoid")
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, **params)
    if model == "lightgbm":
        return lgb.LGBMClassifier(learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4,
                                  random_state=seed, **params)
    raise ValueError(f"unknown model {model}")


def _fit_predict_t2_once(model, params, train, test, features, seed):
    Xtr, Xte = _design_t2(model, train, test, features)
    est = _make_t2(model, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(Xtr, train["y_next"].astype(int).values)
        return est.predict_proba(Xte)[:, 1]


def _best_f1_threshold(y, p) -> float:
    """Score threshold with the best F1 (chosen on the inner validation block only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], EPS, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def _fit_predict_t2(model, train, test, seed, features):
    """Returns (score in [0, 1], decision threshold, chosen params)."""
    if model == "persistence":          # strict definition: this week's label is next week's prediction
        return test["y"].astype(float).values, 0.5, {}
    if model == "persistence_impact":   # rank by last week's impact; the hotspot rule is "top 10%"
        return test.groupby("week")["impact"].rank(pct=True).values, 0.90, {}
    grid = T2_GRIDS[model]
    val_start = train["week"].max() - 1                     # the last 2 weeks of the training rows
    itr, iva = train[train["week"] < val_start], train[train["week"] >= val_start]
    best, best_score, best_thr = grid[0], -np.inf, 0.5
    for p in grid:
        s = _fit_predict_t2_once(model, p, itr, iva, features, seed)
        ap = average_precision_score(iva["y_next"].astype(int), s)
        if ap > best_score:
            best, best_score = p, ap
            best_thr = _best_f1_threshold(iva["y_next"].astype(int), s)
    return _fit_predict_t2_once(model, best, train, test, features, seed), best_thr, best


def fold_frames_t2(feat: pd.DataFrame, fold: D.Fold):
    """T2 rows are keyed by the ORIGIN week w (target = label of week w+1).

    Train = origin weeks before the test block; test = the 2 origin weeks of the block."""
    train = feat[feat["week"] < fold.train_end]
    test = feat[(feat["week"] >= fold.test_start) & (feat["week"] <= fold.test_end)]
    return train, test


# =================================================================================== public API
def fit_predict(task: str, model: str, train: pd.DataFrame, test: pd.DataFrame,
                seed: int = 0, features: list[str] | None = None) -> np.ndarray:
    if task == "t1":
        return _fit_predict_t1(model, train, test, seed, features or F.FEATURES_T1)[0]
    return _fit_predict_t2(model, train, test, seed, features or F.FEATURES_T2)[0]


def run_cv(task: str, models: list[str] | None = None, seeds=(0, 1, 2), features: list[str] | None = None,
           feat: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Cross-validated predictions, one row per (model, seed, fold, test row).

    T1 columns: model, fold, seed, police_station, date, y, pred, naive_mae, params
    T2 columns: model, fold, seed, h3_r8, week, y (= next week's label), pred, thr, prev, params
    These are exactly the frames metrics.summarise_t1 / summarise_t2 expect."""
    models = models or models_for(task)
    if task == "t1":
        return _run_cv_t1(models, seeds, features or F.FEATURES_T1, feat, verbose)
    if task == "t2":
        return _run_cv_t2(models, seeds, features or F.FEATURES_T2, feat, verbose)
    raise ValueError(f"unknown task {task}")


def _log(verbose, fold, model, seed, t0, params):
    if verbose:
        print(f"[run_cv] fold {fold} {model:19s} seed {seed} {time.time() - t0:5.1f}s {params}", flush=True)


def _run_cv_t1(models, seeds, features, feat, verbose):
    sd = D.load_station_day()
    feat = F.t1_features(sd) if feat is None else feat
    folds = D.day_folds()
    out = []
    for fold in folds:
        train, test = fold_frames(feat, fold)
        base = test[["police_station", "date", "y"]].assign(fold=fold.idx)
        for model in models:
            for seed in (seeds if model in STOCHASTIC else seeds[:1]):
                t0 = time.time()
                pred, params = _fit_predict_t1(model, train, test, seed, features)
                out.append(base.assign(model=model, seed=seed, pred=pred, params=str(params)))
                _log(verbose, fold.idx, model, seed, t0, params)
    return M.add_naive_mae(pd.concat(out, ignore_index=True), sd, folds)


def _run_cv_t2(models, seeds, features, feat, verbose):
    feat = F.t2_features(D.load_hex_week()) if feat is None else feat
    out = []
    for fold in D.week_folds():
        train, test = fold_frames_t2(feat, fold)
        base = test[["h3_r8", "week"]].assign(y=test["y_next"].astype(int).values,
                                              prev=test["y"].astype(int).values, fold=fold.idx)
        for model in models:
            for seed in (seeds if model in STOCHASTIC else seeds[:1]):
                t0 = time.time()
                pred, thr, params = _fit_predict_t2(model, train, test, seed, features)
                out.append(base.assign(model=model, seed=seed, pred=pred, thr=thr, params=str(params)))
                _log(verbose, fold.idx, model, seed, t0, params)
    return pd.concat(out, ignore_index=True)


def conformal_handoff(model: str, feat: pd.DataFrame | None = None, seed: int = 0,
                      features: list[str] | None = None, cal_days: int = 21) -> pd.DataFrame:
    """Predictions for E5 (conformal intervals): columns fold, police_station, date, y, pred, split.

    Per fold the model is trained ONLY on days before the calibration block, then predicts both the
    calibration block (the `cal_days` days right before the fold's train_end, split='cal') and the
    14-day test block (split='test'). Calibration residuals and test residuals therefore come from
    the same fitted model, which is what split-conformal prediction needs."""
    features = features or F.FEATURES_T1
    feat = F.t1_features(D.load_station_day()) if feat is None else feat
    out = []
    for fold in D.day_folds():
        cal_start = fold.train_end - pd.Timedelta(days=cal_days)
        train = feat[feat["date"] < cal_start]
        cal = feat[(feat["date"] >= cal_start) & (feat["date"] < fold.train_end)]
        test = feat[(feat["date"] >= fold.test_start) & (feat["date"] <= fold.test_end)]
        both = pd.concat([cal.assign(split="cal"), test.assign(split="test")])
        pred, _ = _fit_predict_t1(model, train, both, seed, features)      # one fit, both blocks
        out.append(both[["police_station", "date", "y", "split"]].assign(fold=fold.idx, pred=pred))
    return pd.concat(out, ignore_index=True)[["fold", "police_station", "date", "y", "pred", "split"]]
