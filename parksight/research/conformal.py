"""Split-conformal intervals for T1 counts: one global quantile, or one per volume tier."""
import numpy as np
import pandas as pd


def _quantile(abs_resid, alpha: float) -> float:
    n = len(abs_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_resid, level, method="higher"))


def _band(test: pd.DataFrame, q) -> pd.DataFrame:
    out = test.copy()
    out["lo"] = (out["pred"] - q).clip(lower=0)  # counts cannot be negative
    out["hi"] = out["pred"] + q
    return out


def global_interval(cal_resid: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.20) -> pd.DataFrame:
    """cal_resid needs [y, pred]; test needs [pred]. Returns test + lo, hi."""
    return _band(test, _quantile((cal_resid["y"] - cal_resid["pred"]).abs(), alpha))


def tier_interval(cal_resid: pd.DataFrame, test: pd.DataFrame, tiers: pd.Series,
                  alpha: float = 0.20) -> pd.DataFrame:
    """tiers maps police_station -> tier. Each tier gets its own quantile."""
    cal_t = cal_resid["police_station"].map(tiers)
    test_t = test["police_station"].map(tiers)
    q = {t: _quantile((g["y"] - g["pred"]).abs(), alpha)
         for t, g in cal_resid.groupby(cal_t)}
    return _band(test, test_t.map(q))


def make_tiers(train_totals: pd.Series) -> pd.Series:
    """Terciles of TRAINING ticket totals per station -> low / mid / high."""
    return pd.qcut(train_totals, 3, labels=["low", "mid", "high"]).astype(str)
