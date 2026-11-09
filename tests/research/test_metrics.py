import numpy as np
import pandas as pd

from parksight.research import metrics as M


def test_mase_hand_example():
    actual, model, naive = [10, 12, 9, 14], [11, 11, 10, 12], [8, 15, 12, 10]
    scale = np.mean(np.abs(np.array(actual) - np.array(naive)))      # 3.0
    assert scale == 3.0
    assert round(M.mase(actual, model, scale), 3) == 0.417


def test_poisson_deviance_survives_zero_prediction():
    d = M.poisson_deviance([0, 3, 5], [0.0, 2.0, 5.0])
    assert np.isfinite(d) and d >= 0


def test_bootstrap_ci_brackets_the_mean():
    v = np.random.default_rng(1).normal(2.0, 1.0, 1000)
    lo, hi = M.bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_zero_scale_is_flagged_not_dropped():
    pred = pd.DataFrame(dict(model="m", police_station="S", seed=0, fold=0, y=[1.0, 2.0],
                             pred=[1.0, 3.0], naive=[1.0, 2.0], scale=[0.0, 0.0],
                             date=pd.to_datetime(["2024-01-01", "2024-01-02"])))
    sm = M.station_mase(pred)
    assert len(sm) == 1 and bool(sm["flagged"].iloc[0]) and np.isfinite(sm["mase"].iloc[0])
