"""Regression tests for the T1 row convention: row (station, d) has y = n(d) and features from days < d.

An earlier version set y = n(d+1) while features still stopped at d-1, which skipped day d, made lag7
point 8 days before the target, and attached the weekday/holiday of d to the count of d+1."""
import pandas as pd

from parksight.research import data as D
from parksight.research import features as F


def _real():
    sd = D.load_station_day()
    return sd, F.t1_features(sd)


def test_target_is_the_rows_own_day_count():
    sd, f = _real()
    m = f.merge(sd.rename(columns={"n": "n_check"}), on=["police_station", "date"])
    assert len(m) == len(f)
    assert (m["y"] == m["n_check"]).all()


def test_history_features_come_from_earlier_days_with_the_right_offsets():
    sd, f = _real()
    s = sd.set_index(["police_station", "date"])["n"]
    row = f[f["police_station"] == "Upparpet"].iloc[10]
    d, st, day = row["date"], "Upparpet", pd.Timedelta(days=1)
    assert row["lag1"] == s[(st, d - day)]
    assert row["lag7"] == s[(st, d - 7 * day)]
    assert (d - 7 * day).weekday() == d.weekday()       # lag7 really is "same weekday last week"
    assert row["lag14"] == s[(st, d - 14 * day)]


def test_calendar_flags_describe_the_target_day():
    _, f = _real()
    assert (f["weekday"] == f["date"].dt.weekday).all()
    assert (f["is_weekend"] == (f["date"].dt.weekday >= 5).astype(int)).all()


def test_rows_cover_every_day_after_the_28_day_warmup_through_the_last_day():
    sd, f = _real()
    assert f["date"].min() == sd["date"].min() + pd.Timedelta(days=28)
    assert f["date"].max() == sd["date"].max() == pd.Timestamp("2024-04-08")
    assert len(f) == 54 * (151 - 28)
