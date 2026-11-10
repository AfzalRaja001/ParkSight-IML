"""E5: global vs per-tier conformal intervals, plus stratified error.

Input (hand-off from Afzal): parquet with [fold, police_station, date, y, pred, split],
split in {"cal", "test"}; cal = last 21 days before the fold's train_end, test = the
14-day test block, all from the best E3 model.
Usage: PYTHONPATH=. python experiments/e5_conformal.py [predictions.parquet]
"""
import sys
from pathlib import Path

import pandas as pd

from parksight.research.conformal import global_interval, make_tiers, tier_interval
from parksight.research.metrics import coverage

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.20


def main(path):
    p = pd.read_parquet(path)
    sd = pd.read_parquet(ROOT / "parksight/data/processed/daily_station.parquet")
    rows, strat = [], []
    for fold, f in p.groupby("fold"):
        cal, test = f[f.split == "cal"], f[f.split == "test"]
        # tiers from TRAINING totals only: days before this fold's calibration block
        tiers = make_tiers(sd[sd.date < cal.date.min()].groupby("police_station")["n"].sum())
        for name, out in (("global", global_interval(cal, test, ALPHA)),
                          ("tier", tier_interval(cal, test, tiers, ALPHA))):
            out = out.assign(tier=out.police_station.map(tiers), width=out.hi - out.lo)
            for tier, g in [("all", out), *out.groupby("tier")]:
                rows.append(dict(fold=fold, method=name, tier=tier, n=len(g),
                                 coverage=coverage(g.y, g.lo, g.hi), width=g.width.mean()))
        t = test.assign(tier=test.police_station.map(tiers),
                        day=test.date.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday"),
                        ae=(test.y - test.pred).abs())
        for col in ("tier", "day"):
            strat += [dict(fold=fold, by=col, group=k, mae=v) for k, v in t.groupby(col).ae.mean().items()]
    cov = pd.DataFrame(rows).groupby(["method", "tier"]).apply(
        lambda g: pd.Series(dict(coverage=(g.coverage * g.n).sum() / g.n.sum(),  # row-weighted
                                 width=(g.width * g.n).sum() / g.n.sum(), n=g.n.sum())),
        include_groups=False).reset_index()
    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out_dir / "E5_conformal.csv", index=False)
    st = pd.DataFrame(strat).groupby(["by", "group"], as_index=False).mae.mean()
    st.to_csv(out_dir / "E5_stratified_error.csv", index=False)
    (out_dir / "E5_conformal.tex").write_text(cov.to_latex(index=False, float_format="%.3f"))
    print(cov.to_string(index=False), "\n", st.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/predictions/t1_best.parquet")
