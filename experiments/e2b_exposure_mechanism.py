"""E2b: is the exposure-estimator failure in E2 caused by the estimator or by how E2 censors? Owner: Afzal.

E2 thins the pseudo-truth ticket by ticket, and its estimator counts DISTINCT DEVICES per hour over all 151
days. Real censoring works through officers being on or off duty, and a device that is active in an hour on
even a few days is counted in full by the distinct-device estimator, so that count saturates. This script
crosses two censoring mechanisms with two estimators and compares each estimate with the known exposure:

  mechanism   ticket_level  keep each ticket with prob e*(hour)                        (what E2 does)
              device_level  keep each (device, date, hour) block with prob e*(hour), and all its tickets
  estimator   distinct_devices  distinct devices per hour / max                         (the hackathon method)
              device_days       distinct (device, date) pairs per hour / max

Same truth window, same e* curve (same_for_all) and the same seeds 0-19 as E2. It scores the exposure CURVE
only; the weighting is not re-run end to end with these estimates.

Needs research_data/tickets_device.parquet (gitignored; build it with scripts/build_tickets_device.py).
Writes results/tables/E2_exposure_mechanism.csv (per hour) and E2_exposure_mechanism_summary.csv.
Run:  python experiments/e2b_exposure_mechanism.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from parksight.research import censor_benchmark as B  # noqa: E402
from parksight.research import paths  # noqa: E402

SEVERITIES = (0.5, 1.0)
N_SEEDS = 20
AT_HOUR = 9
HOURS = list(B.TRUTH_HOURS)
ESTIMATORS = ("distinct_devices", "device_days")


def estimate(kept: pd.DataFrame, estimator: str) -> np.ndarray:
    """Exposure curve over the truth hours, divided by its maximum (the best hour counts as 1)."""
    if estimator == "distinct_devices":
        act = kept.groupby("hour")["device_id"].nunique()
    else:
        act = kept.drop_duplicates(["device_id", "date", "hour"]).groupby("hour").size()
    act = act.reindex(HOURS, fill_value=0).astype(float)
    return (act / act.max()).to_numpy()


def main():
    paths.ensure_dirs()
    if not paths.DEVICE_FILE.exists():
        raise SystemExit(f"{paths.DEVICE_FILE} is missing; run scripts/build_tickets_device.py first")
    dev = pd.read_parquet(paths.DEVICE_FILE, columns=["police_station", "date", "hour", "device_id"])
    truth = B.truth_window(dev)
    blocks = truth[["device_id", "date", "hour"]].drop_duplicates().reset_index(drop=True)
    print(f"[E2b] truth: {len(truth):,} tickets in {len(blocks):,} device-hour blocks", flush=True)

    detail, summary = [], []
    for severity in SEVERITIES:
        e = B.e_star("same_for_all", severity, 0)
        e_hour = e.groupby("hour")["e"].mean().reindex(HOURS)          # one curve for every station
        e_true = (e_hour / e_hour.max()).to_numpy()
        p_ticket = truth["hour"].map(e_hour).to_numpy()
        p_block = blocks["hour"].map(e_hour).to_numpy()

        acc = {(m, est): [] for m in ("ticket_level", "device_level") for est in ESTIMATORS}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            kept_ticket = truth[rng.random(len(truth)) < p_ticket]
            kept_block = blocks[rng.random(len(blocks)) < p_block]
            for est in ESTIMATORS:
                acc[("ticket_level", est)].append(estimate(kept_ticket, est))
                acc[("device_level", est)].append(estimate(kept_block, est))

        for (mechanism, est), runs in acc.items():
            e_hat = np.mean(runs, axis=0)
            ratio = e_hat / e_true
            for h, t, a, r in zip(HOURS, e_true, e_hat, ratio):
                detail.append(dict(severity=severity, mechanism=mechanism, estimator=est, hour=h,
                                   e_true=t, e_hat=a, ratio=r))
            summary.append(dict(severity=severity, mechanism=mechanism, estimator=est, n_seeds=N_SEEDS,
                                mean_abs_log_ratio=float(np.abs(np.log(ratio)).mean()),
                                ratio_at_09=float(ratio[HOURS.index(AT_HOUR)])))

    pd.DataFrame(detail).to_csv(paths.TABLES / "E2_exposure_mechanism.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(paths.TABLES / "E2_exposure_mechanism_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(s.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
