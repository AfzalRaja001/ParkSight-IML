from parksight.research import data as D
from parksight.research import features as F
from parksight.research import metrics as M
from parksight.research import models as MD


def test_no_test_row_in_train_and_dates_strictly_ordered():
    feat = F.t1_features(D.load_station_day())
    for fold in D.day_folds():
        train, test = MD.fold_frames(feat, fold)
        assert train["date"].max() < test["date"].min()
        assert len(set(train.index) & set(test.index)) == 0
        assert len(test) == 54 * 14


def test_baselines_are_exact_and_seasonal_naive_pooled_mase_is_one():
    feat = F.t1_features(D.load_station_day())
    pred = MD.run_cv("t1", ["seasonal_naive", "ma7"], feat=feat, verbose=False)
    assert len(pred) == 2 * 4 * 54 * 14
    sn = pred[pred["model"] == "seasonal_naive"]
    assert (sn["pred"] == sn["naive"]).all()
    s = M.summarise_t1(pred).set_index("model")
    assert round(s.loc["seasonal_naive", "pooled_mase"], 3) == 1.0


def test_fit_predict_is_deterministic_for_a_given_seed():
    feat = F.t1_features(D.load_station_day())
    train, test = MD.fold_frames(feat, D.day_folds()[0])
    a = MD.fit_predict("t1", "lightgbm", train, test, seed=1)
    b = MD.fit_predict("t1", "lightgbm", train, test, seed=1)
    assert (a == b).all() and (a >= 0).all()
