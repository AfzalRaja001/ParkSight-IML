"""E7: LightGBM training loss vs boosting round (the "epoch" of a gradient-boosted model).

Last fold of each task, tuned-model settings; validation = the last 14 days (T1) / 2 weeks (T2)
of the training window, exactly the inner split used for tuning. Output: results/figures/F5_training_curve.pdf
and results/tables/E7_training_curve.csv.   Usage: PYTHONPATH=. python experiments/e7_training_curve.py
"""
import matplotlib
matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from parksight.research import data as D
from parksight.research import features as F
from parksight.research import models as Mo
from parksight.research import style
from parksight import config as C

OUT = C.ROOT / "results"


def deviance_curve(est, Xtr, ytr, Xva, yva):
    """Mean Poisson deviance after each boosting round (the same loss as the E3 scores)."""
    est.fit(Xtr, ytr)
    f = lambda X, y: [mean_poisson_deviance(y, est.predict(X, num_iteration=k).clip(1e-6)) for k in range(1, est.n_estimators + 1)]
    return f(Xtr, ytr), f(Xva, yva)


def curve(est, Xtr, ytr, Xva, yva, metric):
    rec = {}
    est.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "val"], eval_metric=metric,
            callbacks=[lgb.record_evaluation(rec)])
    return rec["train"][metric], rec["val"][metric]


def main():
    style.apply()
    rows, curves = [], {}

    f = D.day_folds()[-1]
    feat = F.t1_features(D.load_station_day())
    tr = feat[feat["date"] < f.train_end]
    vs = tr["date"].max() - pd.Timedelta(days=13)
    a, b = tr[tr["date"] < vs], tr[tr["date"] >= vs]
    est = lgb.LGBMRegressor(objective="poisson", random_state=0, n_jobs=4, n_estimators=800, **Mo.LGB_BASE)
    curves["T1: next-day counts (Poisson deviance)"] = deviance_curve(est, a[F.FEATURES_T1], a["y"], b[F.FEATURES_T1], b["y"])

    f = D.week_folds()[-1]
    hw = F.t2_features(D.load_hex_week())
    tr = hw[hw["week"] < f.train_end]
    vs = tr["week"].max() - 1
    a, b = tr[tr["week"] < vs], tr[tr["week"] >= vs]
    feats = Mo.F.FEATURES_T2 + ["y"] if "y" in Mo.F.FEATURES_T2 else Mo.F.FEATURES_T2
    feats = [c for c in feats if c in tr.columns]
    est = Mo._make_t2("lightgbm", {"n_estimators": 200}, 0)
    curves["T2: next-week hotspots (log loss)"] = curve(est, a[feats], a["y_next"].astype(int), b[feats], b["y_next"].astype(int), "binary_logloss")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, (name, (t, v)) in zip(axes, curves.items()):
        ax.plot(range(1, len(t) + 1), t, label="train")
        ax.plot(range(1, len(v) + 1), v, label="validation")
        best = int(pd.Series(v).idxmin()) + 1
        ax.axvline(best, color="gray", ls=":", lw=1)
        ax.set(title=name, xlabel="boosting round (epoch)", ylabel="loss")
        ax.legend()
        rows.append(dict(task=name, best_round=best, train_loss_best=t[best - 1], val_loss_best=v[best - 1],
                         train_loss_final=t[-1], val_loss_final=v[-1], rounds=len(t)))
    style.savefig(fig, "F5_training_curve")
    pd.DataFrame(rows).to_csv(OUT / "tables" / "E7_training_curve.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
