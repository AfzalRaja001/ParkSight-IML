import pandas as pd

from parksight.research.data import day_folds, week_folds


def test_folds_no_overlap():
    for f in day_folds() + week_folds():
        assert f.train_end <= f.test_start <= f.test_end

    starts = [f.test_start for f in day_folds()]
    assert starts == [pd.Timestamp(d) for d in
                      ["2024-02-13", "2024-02-27", "2024-03-12", "2024-03-26"]]
    assert day_folds()[-1].test_end == pd.Timestamp("2024-04-08")

    assert [f.test_start for f in week_folds()] == [13, 15, 17, 19]
