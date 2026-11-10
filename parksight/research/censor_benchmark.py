"""Semi-synthetic censoring benchmark (E2): thin a well-covered pseudo-truth with a known exposure
curve, then score how well each estimator recovers it. 

  truth      every ticket from 02:00 to 11:59 IST (itself slightly censored, so we measure recovery
             of this reference, not of an absolute truth)
  e_star     known exposure per (station, hour): the shape of the real device curve over hours 12-21
             (the evening dip) mapped onto the 10 truth hours, divided by its first value so the best
             hour keeps everything, raised to `severity`
  censor     keeps each ticket independently with probability e_star(station, hour)
  estimators naive (count what is left), HT with the true e (oracle), HT with e re-estimated from the
             thinned tickets by the hackathon method (distinct devices per hour / max)
"""
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from parksight import config as C

TRUTH_HOURS = range(2, 12)
DIP_HOURS = range(12, 22)          # the real dip whose shape becomes e_star
HETERO_POWER = (0.5, 1.5)          # station_hetero: each station's exponent is severity * U(0.5, 1.5)
KEYS = ["police_station", "hour"]


def truth_window(tickets: pd.DataFrame, hours=TRUTH_HOURS) -> pd.DataFrame:
    """Tickets in the pseudo-truth hours. Needs [police_station, hour]; other columns are kept."""
    return tickets[tickets["hour"].isin(list(hours))].reset_index(drop=True)


def truth_counts(truth: pd.DataFrame) -> pd.DataFrame:
    """[police_station, hour, n] for the pseudo-truth."""
    return truth.groupby(KEYS).size().rename("n").reset_index()


def dip_curve() -> np.ndarray:
    """Real distinct-device counts for hours 12-21, divided by hour 12 (so the first value is 1)."""
    dev = pd.read_parquet(C.PROCESSED / "enforcement_exposure.parquet").set_index("hour")["devices"]
    curve = dev.reindex(list(DIP_HOURS)).to_numpy(dtype=float)
    return curve / curve[0]


def e_star(family: Literal["same_for_all", "station_hetero"], severity: float, seed: int,
           stations=None) -> pd.DataFrame:
    """[police_station, hour, e], max e = 1. `stations` defaults to the 54 stations in the data."""
    if stations is None:
        stations = sorted(pd.read_parquet(C.PROCESSED / "violations_clean.parquet",
                                          columns=["police_station"])["police_station"].unique())
    base = dip_curve()
    if family == "same_for_all":
        power = np.full(len(stations), float(severity))
    elif family == "station_hetero":
        power = severity * np.random.default_rng(seed).uniform(*HETERO_POWER, len(stations))
    else:
        raise ValueError(f"unknown family {family}")
    e = base[None, :] ** power[:, None]
    return pd.DataFrame({"police_station": np.repeat(stations, len(base)),
                         "hour": np.tile(list(TRUTH_HOURS), len(stations)),
                         "e": e.ravel()})


def censor(truth: pd.DataFrame, e: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Bernoulli thinning: keep each ticket with probability e(station, hour). Keeps every column
    of `truth` (including device_id if present)."""
    p = truth[KEYS].merge(e, on=KEYS, how="left")["e"].to_numpy()
    if np.isnan(p).any():
        raise ValueError("e is missing some (station, hour) cells of the truth")
    keep = np.random.default_rng(seed).random(len(truth)) < p
    return truth[keep].reset_index(drop=True)


def estimate_exposure_x1(thinned: pd.DataFrame) -> pd.Series:
    """The hackathon exposure proxy, refitted on the THINNED tickets only: distinct devices per hour
    divided by the busiest hour (one citywide curve, indexed by hour)."""
    if "device_id" not in thinned.columns:
        raise ValueError("estimate_exposure_x1 needs device_id (research_data/tickets_device.parquet)")
    active = thinned.groupby("hour")["device_id"].nunique().reindex(list(TRUTH_HOURS), fill_value=0)
    return (active / active.max()).astype(float).rename("e")


def ht_counts(thinned: pd.DataFrame, e, cap: float | None, floor: float | None = None) -> pd.DataFrame:
    """Per (station, hour): n_obs and the Horvitz-Thompson estimate n_hat = sum of min(1/e, cap).

    e is either a [police_station, hour, e] frame or a Series indexed by hour (one curve for all
    stations). `floor` clips e from below first, as the hackathon does with EXPOSURE_FLOOR. Every
    cell of a frame-valued e appears in the output, with zeros where no ticket survived."""
    obs = thinned.groupby(KEYS).size().rename("n_obs").reset_index()
    if isinstance(e, pd.Series):
        obs["e"] = obs["hour"].map(e)
    else:
        obs = e[KEYS + ["e"]].merge(obs, on=KEYS, how="left").fillna({"n_obs": 0})
    eff = obs["e"] if floor is None else obs["e"].clip(lower=floor)
    w = 1.0 / eff.clip(lower=1e-12)
    if cap is not None:
        w = w.clip(upper=cap)
    obs["n_obs"] = obs["n_obs"].astype("int64")
    obs["n_hat"] = obs["n_obs"] * w
    return obs[KEYS + ["n_obs", "n_hat"]]


def recovery_scores(truth_counts: pd.DataFrame, hat_counts: pd.DataFrame) -> dict:
    """rel_bias of the total, mean absolute error over station-hour cells, and Kendall tau between
    the true and estimated station rankings (by total count)."""
    m = truth_counts.merge(hat_counts[KEYS + ["n_hat"]], on=KEYS, how="left").fillna({"n_hat": 0.0})
    st = m.groupby("police_station")[["n", "n_hat"]].sum()
    return dict(rel_bias=float(m["n_hat"].sum() / m["n"].sum() - 1),
                mae=float((m["n_hat"] - m["n"]).abs().mean()),
                kendall_tau_station=float(kendalltau(st["n"], st["n_hat"]).statistic))
