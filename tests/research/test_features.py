import numpy as np
import pandas as pd

from parksight.research.features import FEATURES_T1, t1_features


def _toy_station_day(n_days=60):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    return pd.concat([
        pd.DataFrame({"police_station": s, "date": dates, "n": rng.integers(0, 50, n_days)})
        for s in ("A", "B")
    ], ignore_index=True)


def test_no_future_leak():
    sd = _toy_station_day()
    cutoff = sd["date"].iloc[40]
    mutated = sd.copy()
    mutated.loc[mutated["date"] >= cutoff, "n"] = 9999

    keep_cols = ["police_station", "date"] + FEATURES_T1
    before = t1_features(sd)
    after = t1_features(mutated)
    b = before.loc[before["date"] <= cutoff, keep_cols].reset_index(drop=True)
    a = after.loc[after["date"] <= cutoff, keep_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(a, b)
