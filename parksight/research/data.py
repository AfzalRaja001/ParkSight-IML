"""Datasets and time-blocked folds.

FIRST DRAFT written by Afzal so that E3 (count forecasting) could run end to end. Signatures
follow IMPLEMENTATION_PLAN_V2.md section 5. Shaurya owns this file: review, extend (hex-week
dataset for E4) and replace as needed.
"""
from dataclasses import dataclass

import pandas as pd

from parksight.research import paths

# pinned in IMPLEMENTATION_PLAN_V2.md section 3
T1_TEST_STARTS = ("2024-02-13", "2024-02-27", "2024-03-12", "2024-03-26")
T1_TEST_DAYS = 14


@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: pd.Timestamp     # exclusive
    test_start: pd.Timestamp    # inclusive
    test_end: pd.Timestamp      # inclusive


def _clean_tickets(columns):
    return pd.read_parquet(paths.PROCESSED / "violations_clean.parquet", columns=columns)


def load_station_day() -> pd.DataFrame:
    """[police_station, date, n] on the full station x day grid (zeros filled).

    A zero means "no ticket was recorded", which can mean no violation or no officer."""
    v = _clean_tickets(["police_station", "created_ist"])
    v["date"] = v["created_ist"].dt.normalize()
    counts = v.groupby(["police_station", "date"]).size().rename("n")
    days = pd.date_range(v["date"].min(), v["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([sorted(v["police_station"].unique()), days],
                                      names=["police_station", "date"])
    return counts.reindex(grid, fill_value=0).reset_index()


def day_folds() -> list[Fold]:
    """4 expanding-window folds, 14-day test blocks, last block ends on the last date."""
    folds = []
    for i, s in enumerate(T1_TEST_STARTS):
        start = pd.Timestamp(s)
        folds.append(Fold(i, start, start, start + pd.Timedelta(days=T1_TEST_DAYS - 1)))
    return folds


def load_hex_week() -> pd.DataFrame:
    raise NotImplementedError("hex x week hotspot dataset (E4) is not written yet")


def week_folds() -> list[Fold]:
    raise NotImplementedError("week folds (E4) are not written yet")


def build_device_file() -> pd.DataFrame:
    """Write research_data/tickets_device.parquet: [id, police_station, date, hour, weekday, device_id].

    Joins the raw CSV's device_id onto the cleaned tickets by id, so the 298,445 rows are exactly
    the ones the hackathon pipeline kept. For E1 and E2 only; never commit this file."""
    paths.ensure_dirs()
    raw = pd.read_csv(paths.RAW_CSV, usecols=["id", "device_id"], low_memory=False)
    v = _clean_tickets(["id", "police_station", "created_ist", "hour", "weekday"])
    out = v.merge(raw, on="id", how="left", validate="one_to_one")
    out["date"] = out["created_ist"].dt.normalize()
    out = out[["id", "police_station", "date", "hour", "weekday", "device_id"]]
    out.to_parquet(paths.DEVICE_FILE, index=False)
    return out
