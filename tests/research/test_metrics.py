import numpy as np
import pandas as pd

from parksight.research.metrics import (bootstrap_ci, coverage, mase,
                                        summarise_t1, summarise_t2)


def test_mase_hand_example():
    assert round(mase([10, 12, 9, 14], [11, 11, 10, 12], 3.0), 3) == 0.417


def test_summarise_t1_flags_zero_denominator():
    rows = [dict(model="m", fold=0, seed=0, police_station=s, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                 y=10.0, pred=11.0, naive_mae=0.0 if s == "B" else 2.0)
            for s in "ABC" for i in range(3)]
    out = summarise_t1(pd.DataFrame(rows))
    assert out.loc[0, "n_flagged_stations"] == 1  # flagged, not dropped
    assert np.isfinite(out.loc[0, "mase_macro"])


def test_bootstrap_and_coverage():
    lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi
    assert coverage([1, 5], [0, 0], [2, 2]) == 0.5


def test_summarise_t2_emerging_subset():
    d = pd.DataFrame(dict(model="p", fold=0, seed=0, h3_r8=range(8), week=1,
                          y=[1, 0, 1, 0, 1, 0, 0, 1], prev=[1, 1, 0, 0, 1, 0, 0, 0],
                          pred=[.9, .2, .6, .1, .8, .3, .2, .7]))
    out = summarise_t2(d).set_index("subset")
    assert out.loc["all", "n"] == 8 and out.loc["emerging", "n"] == 5
