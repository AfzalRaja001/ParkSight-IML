import numpy as np
import pandas as pd

from parksight.research.conformal import global_interval, make_tiers, tier_interval
from parksight.research.metrics import coverage


def _frame(rng, n, scale):
    pred = rng.uniform(5, 50, n)
    return pd.DataFrame(dict(police_station=rng.choice(list("abcdef"), n), pred=pred,
                             y=pred + rng.normal(0, scale, n)))


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    cal, test = _frame(rng, 2000, 4), _frame(rng, 4000, 4)
    out = global_interval(cal, test)
    assert abs(coverage(out["y"], out["lo"], out["hi"]) - 0.80) < 0.04


def test_tier_interval_adapts_to_noise():
    rng = np.random.default_rng(1)
    tiers = pd.Series(dict(zip("abcdef", ["low"] * 3 + ["high"] * 3)))
    def mk(n):
        f = _frame(rng, n, 1)
        f["y"] = f["pred"] + rng.normal(0, np.where(f["police_station"].isin(list("abc")), 1, 10))
        return f
    out = tier_interval(mk(3000), mk(3000), tiers)
    low = out["police_station"].isin(list("abc"))
    assert (out.hi - out.lo)[low].mean() < (out.hi - out.lo)[~low].mean()
    assert abs(coverage(out.y[~low], out.lo[~low], out.hi[~low]) - 0.80) < 0.05


def test_make_tiers():
    assert set(make_tiers(pd.Series(range(9)))) == {"low", "mid", "high"}
