"""Leak-free features for T1 (next-day ticket count per police station).

FIRST DRAFT written by Afzal so that E3 could run. Shaurya owns this file.

Convention: one row per (station, date d). The target `y` is the ticket count ON day d. Every
history feature uses only days strictly before d (shift(1) first), so a row can never see its own
day or the future. Calendar features (weekday, holiday) are known in advance, so they are allowed.
This is a one-step-ahead forecast: inside a test block each day is predicted from the actual
counts up to the previous day.
"""
import numpy as np
import pandas as pd

LAG_FEATURES = ["lag1", "lag7", "lag14", "roll7", "roll28", "samedow_mean"]
CAL_FEATURES = ["dow", "is_weekend", "is_holiday"]
STATION = "police_station"
FEATURES_T1 = LAG_FEATURES + CAL_FEATURES + [STATION]
WARMUP_DAYS = 28   # rows without a full 28-day history are dropped, not imputed


def _holiday_dates(years):
    try:
        import holidays
        return set(pd.Timestamp(d) for d in holidays.country_holidays("IN", years=years).keys())
    except Exception:       # library missing: no holiday signal rather than a crash
        return set()


def t1_features(sd: pd.DataFrame) -> pd.DataFrame:
    """sd = [police_station, date, n] on a complete daily grid. Returns sd plus features and `y`."""
    df = sd.sort_values([STATION, "date"]).reset_index(drop=True).copy()
    gaps = df.groupby(STATION)["date"].diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError("t1_features needs a complete daily grid per station (shift = days)")

    g = df.groupby(STATION, sort=False)["n"]
    df["lag1"] = g.shift(1)
    df["lag7"] = g.shift(7)
    df["lag14"] = g.shift(14)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(WARMUP_DAYS, min_periods=WARMUP_DAYS).mean())

    df["dow"] = df["date"].dt.weekday
    df["samedow_mean"] = (df.groupby([STATION, "dow"], sort=False)["n"]
                            .transform(lambda s: s.shift(1).expanding().mean()))
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    hol = _holiday_dates(sorted(df["date"].dt.year.unique()))
    df["is_holiday"] = df["date"].isin(hol).astype(int)

    df["y"] = df["n"].astype(float)
    return df.dropna(subset=LAG_FEATURES).reset_index(drop=True)


def log_counts(X: pd.DataFrame) -> pd.DataFrame:
    """log1p of the count-type features (used by the linear models, which have a log scale)."""
    X = X.copy()
    for c in LAG_FEATURES:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    return X
