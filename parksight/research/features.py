"""Leak-free feature builders for T1 (station-day counts) and T2 (hex-week hotspots).

Every feature uses only data from strictly before the row's own date/week (via shift);
the target column (y for T1, y_next for T2) is the only forward-looking (shift(-1)) value.
"""
import h3
import numpy as np
import pandas as pd

from parksight import config as C
from parksight.models.train_forecast import indian_holidays
from parksight.research.data import week_numbers

FEATURES_T1 = ["lag1", "lag7", "lag14", "roll7", "roll28", "samedow_mean",
              "weekday", "is_weekend", "is_holiday", "station_code"]
FEATURES_T2 = ["impact_lag1", "impact_lag2", "impact_lag3", "impact_lag4", "impact_roll4",
              "evening_share", "mean_severity", "junction_share", "neighbor_impact"]


def t1_features(sd: pd.DataFrame) -> pd.DataFrame:
    df = sd.sort_values(["police_station", "date"]).reset_index(drop=True).copy()
    g = df.groupby("police_station")["n"]
    df["lag1"] = g.shift(1)
    df["lag7"] = g.shift(7)
    df["lag14"] = g.shift(14)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    years = sorted({df["date"].min().year, df["date"].max().year})
    hol = indian_holidays(years)
    df["is_holiday"] = df["date"].dt.date.isin(hol).astype(int)
    df["samedow_mean"] = (df.groupby(["police_station", "weekday"])["n"]
                          .transform(lambda s: s.shift(1).expanding().mean()))
    df["station_code"] = df["police_station"].astype("category").cat.codes
    df["y"] = g.shift(-1)
    return df.dropna(subset=FEATURES_T1 + ["y"]).reset_index(drop=True)


def _neighbor_lag1_impact(hw: pd.DataFrame) -> pd.Series:
    """Mean impact of a hexagon's 6 ring-1 neighbours, in the row's own week

    (i.e. the same 1-week-before-the-target offset as impact_lag1 for the hexagon itself).
    """
    pivot = hw.pivot(index="h3_r8", columns="week", values="impact")
    neighbours = {cell: [n for n in h3.grid_disk(cell, 1) if n != cell and n in pivot.index]
                 for cell in pivot.index}
    out = np.full(len(hw), np.nan)
    for i, (cell, week) in enumerate(zip(hw["h3_r8"], hw["week"])):
        ns = neighbours[cell]
        if ns and week in pivot.columns:
            out[i] = pivot.loc[ns, week].mean()
    return pd.Series(out, index=hw.index)


def _raw_hex_week_rates(hw: pd.DataFrame) -> pd.DataFrame:
    """evening_share, mean_severity, junction_share per (h3_r8, week); 0 where n == 0."""
    df = pd.read_parquet(C.PROCESSED / "violations_clean.parquet",
                         columns=["h3_r8", "created_ist", "hour", "severity", "at_junction"])
    df["week"] = week_numbers(df["created_ist"])
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype("int64")
    lo, hi, _ = C.PEAK_WINDOWS["evening"]
    df["is_evening"] = (df["hour"] >= lo) & (df["hour"] < hi)
    agg = (df.groupby(["h3_r8", "week"])
          .agg(evening_share=("is_evening", "mean"), mean_severity=("severity", "mean"),
               junction_share=("at_junction", "mean"))
          .reset_index())
    out = hw[["h3_r8", "week"]].merge(agg, on=["h3_r8", "week"], how="left")
    return out[["evening_share", "mean_severity", "junction_share"]].fillna(0.0)


def t2_features(hw: pd.DataFrame) -> pd.DataFrame:
    df = hw.sort_values(["h3_r8", "week"]).reset_index(drop=True).copy()
    g = df.groupby("h3_r8")["impact"]
    df["impact_lag1"] = df["impact"]
    df["impact_lag2"] = g.shift(1)
    df["impact_lag3"] = g.shift(2)
    df["impact_lag4"] = g.shift(3)
    df["impact_roll4"] = g.transform(lambda s: s.rolling(4).mean())
    df["neighbor_impact"] = _neighbor_lag1_impact(df)
    df[["evening_share", "mean_severity", "junction_share"]] = _raw_hex_week_rates(df)
    df["y_next"] = df.groupby("h3_r8")["y"].shift(-1)
    return df.dropna(subset=FEATURES_T2 + ["y_next"]).reset_index(drop=True)
