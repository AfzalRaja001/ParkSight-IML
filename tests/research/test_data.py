import pandas as pd

from parksight.research import data as D


def test_station_day_grid_is_complete():
    sd = D.load_station_day()
    assert sd["police_station"].nunique() == 54
    assert sd["date"].nunique() == 151
    assert len(sd) == 54 * 151
    assert sd["n"].sum() == 298_445


def test_folds_no_overlap_and_pinned_dates():
    folds = D.day_folds()
    assert [f.test_start.strftime("%Y-%m-%d") for f in folds] == [
        "2024-02-13", "2024-02-27", "2024-03-12", "2024-03-26"]
    for f in folds:
        assert f.train_end <= f.test_start
        assert (f.test_end - f.test_start).days == 13
    assert folds[-1].test_end == pd.Timestamp("2024-04-08")
    sd = D.load_station_day()
    assert folds[-1].test_end == sd["date"].max()
