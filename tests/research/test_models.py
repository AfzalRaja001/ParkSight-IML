import numpy as np

from parksight.research import data as D
from parksight.research import features as F
from parksight.research import metrics as M
from parksight.research import models as MD


def test_t1_no_test_row_in_train_and_dates_strictly_ordered():
    feat = F.t1_features(D.load_station_day())
    for fold in D.day_folds():
        train, test = MD.fold_frames(feat, fold)
        assert train["date"].max() < test["date"].min()
        assert len(set(train.index) & set(test.index)) == 0
        assert len(test) == 54 * 14


def test_t1_baselines_are_exact_and_seasonal_naive_pooled_mase_is_one():
    feat = F.t1_features(D.load_station_day())
    pred = MD.run_cv("t1", ["seasonal_naive", "ma7"], feat=feat, verbose=False)
    assert len(pred) == 2 * 4 * 54 * 14
    assert {"model", "fold", "seed", "police_station", "date", "y", "pred", "naive_mae"} <= set(pred.columns)
    assert pred["naive_mae"].notna().all()
    s = M.summarise_t1(pred).set_index("model")
    assert round(s.loc["seasonal_naive", "mase_pooled"], 3) == 1.0
    # the baseline really is "same weekday last week": its error equals the MASE denominator's definition
    assert s.loc["seasonal_naive", "mase_macro"] < 1.3


def test_t1_fit_predict_is_deterministic_for_a_given_seed():
    feat = F.t1_features(D.load_station_day())
    train, test = MD.fold_frames(feat, D.day_folds()[0])
    a = MD.fit_predict("t1", "lightgbm", train, test, seed=1)
    b = MD.fit_predict("t1", "lightgbm", train, test, seed=1)
    assert (a == b).all() and (a >= 0).all()


def test_t2_folds_never_train_on_or_after_the_test_block():
    feat = F.t2_features(D.load_hex_week())
    for fold in D.week_folds():
        train, test = MD.fold_frames_t2(feat, fold)
        assert train["week"].max() < test["week"].min()
        assert set(test["week"].unique()) == {fold.test_start, fold.test_end}


def test_t2_persistence_rows_are_present_and_exact():
    feat = F.t2_features(D.load_hex_week())
    pred = MD.run_cv("t2", ["persistence", "persistence_impact"], feat=feat, verbose=False)
    assert {"persistence", "persistence_impact"} == set(pred["model"])
    assert {"model", "fold", "seed", "h3_r8", "week", "y", "pred", "thr", "prev"} <= set(pred.columns)
    p = pred[pred["model"] == "persistence"]
    assert (p["pred"] == p["prev"]).all()
    assert p["pred"].isin([0.0, 1.0]).all()


def test_t2_classifier_probabilities_are_valid():
    feat = F.t2_features(D.load_hex_week())
    train, test = MD.fold_frames_t2(feat, D.week_folds()[0])
    p = MD.fit_predict("t2", "logreg", train, test)
    assert np.all((p >= 0) & (p <= 1)) and len(p) == len(test)


def test_conformal_handoff_has_the_columns_and_blocks_E5_expects():
    h = MD.conformal_handoff("lasso")
    assert list(h.columns) == ["fold", "police_station", "date", "y", "pred", "split"]
    for fold, g in h.groupby("fold"):
        cal, test = g[g["split"] == "cal"], g[g["split"] == "test"]
        assert cal["date"].nunique() == 21 and test["date"].nunique() == 14
        assert cal["date"].max() < test["date"].min()
