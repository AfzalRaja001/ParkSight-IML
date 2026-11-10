import numpy as np
import pandas as pd

from parksight.research.censor_benchmark import TRUTH_HOURS, censor, e_star, ht_counts

STATIONS = list("ABCDE")


def _tickets(per_cell):
    """Synthetic truth window: per_cell[i] tickets in each (station, hour) cell, one row per ticket."""
    cells = [(s, h) for s in STATIONS for h in TRUTH_HOURS]
    reps = np.broadcast_to(per_cell, len(cells))
    return pd.DataFrame([(s, h) for (s, h), k in zip(cells, reps) for _ in range(int(k))],
                        columns=["police_station", "hour"])


def test_censor_retention():
    truth = _tickets(400)
    e = e_star("station_hetero", 1.0, seed=0, stations=STATIONS)
    kept = censor(truth, e, seed=0)
    m = (truth.groupby(["police_station", "hour"]).size().rename("n").reset_index()
         .merge(kept.groupby(["police_station", "hour"]).size().rename("k").reset_index(),
                on=["police_station", "hour"], how="left")
         .merge(e, on=["police_station", "hour"]))
    m["k"] = m["k"].fillna(0)
    se = np.sqrt(m["e"] * (1 - m["e"]) / m["n"])
    within = (m["k"] / m["n"] - m["e"]).abs() <= 3 * se.clip(lower=1 / m["n"])
    assert within.mean() >= 0.95


def test_ht_unbiased():
    rng = np.random.default_rng(1)
    truth = _tickets(rng.poisson(50, len(STATIONS) * len(TRUTH_HOURS)))
    e = pd.DataFrame([(s, h, 0.3) for s in STATIONS for h in TRUTH_HOURS],
                     columns=["police_station", "hour", "e"])
    totals = [ht_counts(censor(truth, e, seed=s), e, cap=None)["n_hat"].sum() for s in range(200)]
    assert abs(np.mean(totals) / len(truth) - 1) < 0.03
