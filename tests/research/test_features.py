import numpy as np
import pandas as pd

from parksight.research import features as F


def _grid(seed=0, days=70, stations=("A", "B")):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = [(s, d, int(rng.poisson(20 if s == "A" else 5))) for s in stations for d in dates]
    return pd.DataFrame(rows, columns=["police_station", "date", "n"])


def test_no_future_leak():
    """Changing every count from day D on must not change any feature on or before day D."""
    sd = _grid()
    D = pd.Timestamp("2024-02-20")
    changed = sd.copy()
    changed.loc[changed["date"] >= D, "n"] = 9999
    a, b = F.t1_features(sd), F.t1_features(changed)
    cols = F.LAG_FEATURES + F.CAL_FEATURES + ["police_station", "date"]
    a = a[a["date"] <= D].reset_index(drop=True)
    b = b[b["date"] <= D].reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_lags_mean_what_they_say():
    sd = _grid()
    f = F.t1_features(sd)
    row = f[f["police_station"] == "A"].iloc[0]
    a = sd[sd["police_station"] == "A"].set_index("date")["n"]
    d = row["date"]
    day = pd.Timedelta(days=1)
    assert row["lag1"] == a[d - day]
    assert row["lag7"] == a[d - 7 * day]
    assert np.isclose(row["roll7"], a[d - 7 * day: d - day].mean())
    assert np.isclose(row["roll28"], a[d - 28 * day: d - day].mean())
    assert row["y"] == a[d]


def test_rejects_grid_with_gaps():
    sd = _grid()
    gap = sd.drop(sd.index[10])
    try:
        F.t1_features(gap)
    except ValueError:
        return
    raise AssertionError("a gap in the daily grid must raise")
