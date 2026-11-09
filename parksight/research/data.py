"""Station-day and hexagon-week grids, and the pinned time-blocked folds.

Reads the committed violations_clean.parquet; no raw CSV needed.
"""
from dataclasses import dataclass

import pandas as pd

from parksight import config as C

TOP_DECILE = 0.10
T1_TEST_STARTS = ["2024-02-13", "2024-02-27", "2024-03-12", "2024-03-26"]
T1_TEST_DAYS = 14
T2_TEST_STARTS = [13, 15, 17, 19]  # 1-indexed week numbers, out of the 21 full weeks
T2_TEST_WEEKS = 2


@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: object    # exclusive: a date (T1) or a week number (T2)
    test_start: object   # inclusive
    test_end: object     # inclusive


def _read(columns):
    return pd.read_parquet(C.PROCESSED / "violations_clean.parquet", columns=columns)


def load_station_day() -> pd.DataFrame:
    """[police_station, date, n]; every station x every day in the data's span, zeros filled."""
    df = _read(["police_station", "created_ist"])
    df["date"] = df["created_ist"].dt.normalize()
    counts = df.groupby(["police_station", "date"]).size().rename("n").reset_index()
    stations = sorted(counts["police_station"].unique())
    days = pd.date_range(counts["date"].min(), counts["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([stations, days], names=["police_station", "date"]).to_frame(index=False)
    out = grid.merge(counts, on=["police_station", "date"], how="left")
    out["n"] = out["n"].fillna(0).astype("int64")
    return out


def _full_weeks(ts: pd.Series) -> list:
    """Monday-start weeks whose 7 days all lie inside the data's date span."""
    start, end = ts.min().normalize(), ts.max().normalize()
    weeks = sorted(ts.dt.to_period("W").unique())
    return [w for w in weeks if w.start_time.normalize() >= start and w.end_time.normalize() <= end]


def week_numbers(ts: pd.Series) -> pd.Series:
    """Maps each timestamp's Monday-start week to a 1..21 week number.

    Timestamps outside the 21 full weeks (the partial first/last week) map to NaN.
    """
    full = _full_weeks(ts)
    num = {w: i + 1 for i, w in enumerate(full)}
    return ts.dt.to_period("W").map(num)


def load_hex_week() -> pd.DataFrame:
    """[h3_r8, week, n, impact, y]; every hexagon x each of the 21 full weeks, zeros filled.

    y = 1 if that hexagon is in the top 10% of ITS OWN week's hexagons by impact
    (impact = sum(severity * peak_w)), else 0.
    """
    df = _read(["h3_r8", "created_ist", "severity", "peak_w"])
    df["week"] = week_numbers(df["created_ist"])
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype("int64")
    df["impact_term"] = df["severity"] * df["peak_w"]

    agg = (df.groupby(["h3_r8", "week"])
          .agg(n=("severity", "size"), impact=("impact_term", "sum"))
          .reset_index())
    hexes = sorted(df["h3_r8"].unique())
    weeks = sorted(df["week"].unique())
    grid = pd.MultiIndex.from_product([hexes, weeks], names=["h3_r8", "week"]).to_frame(index=False)
    out = grid.merge(agg, on=["h3_r8", "week"], how="left")
    out["n"] = out["n"].fillna(0).astype("int64")
    out["impact"] = out["impact"].fillna(0.0)

    def label(g):
        k = max(1, round(TOP_DECILE * len(g)))
        cutoff = g["impact"].nlargest(k).min()
        return (g["impact"] >= cutoff).astype(int)

    out["y"] = out.groupby("week", group_keys=False).apply(label)
    return out


def day_folds() -> list:
    """4 expanding-window folds, 14-day test blocks; train uses everything before test_start."""
    folds = []
    for i, start in enumerate(T1_TEST_STARTS):
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(days=T1_TEST_DAYS - 1)
        folds.append(Fold(idx=i, train_end=start, test_start=start, test_end=end))
    return folds


def week_folds() -> list:
    """4 expanding-window folds, 2-week test blocks; train uses everything before test_start."""
    folds = []
    for i, start in enumerate(T2_TEST_STARTS):
        end = start + T2_TEST_WEEKS - 1
        folds.append(Fold(idx=i, train_end=start, test_start=start, test_end=end))
    return folds
