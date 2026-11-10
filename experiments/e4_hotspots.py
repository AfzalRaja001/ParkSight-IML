"""E4: hotspot classification (T2) against the persistence baselines. Owner: Afzal (labels: Shaurya).

Task: for each H3-r8 hexagon and origin week w, predict whether it is a hotspot (top 10% of hexagons
by weekly impact) in week w+1, from weeks up to and including w. 4 expanding-window folds, 2-week test blocks.

Every table reports ALL hexagons and the EMERGING subset (hexagons that were NOT hotspots in week w),
because "last week's hotspot is this week's hotspot" already scores PR-AUC ~0.75 and the interesting
question is whether a model can anticipate new hotspots.

Writes (under results/):
  tables/E4_hotspots.csv / .tex      PR-AUC (+95% cluster-bootstrap CI over hexagons), delta vs the strong
                                     baseline, F1 at an inner-validation threshold, Brier, row counts
  tables/E4_chosen_params.csv        edge-of-grid check for the tuned knobs
  predictions/E4_hotspots_predictions.parquet
  figures/E4_pr_curves.pdf           precision-recall curves, all hexagons and emerging subset

Run:  python experiments/e4_hotspots.py
"""
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from parksight.research import data as D  # noqa: E402
from parksight.research import features as F  # noqa: E402
from parksight.research import metrics as M  # noqa: E402
from parksight.research import models as MD  # noqa: E402
from parksight.research import paths, report_tables, style  # noqa: E402

REFERENCE = "persistence_impact"     # the strong baseline every model is compared with
N_BOOT, BOOT_SEED = 1000, 12345
KEYS = ["fold", "h3_r8", "week"]


def seed_average(pred):
    """One row per (fold, hexagon, week) with the seed-averaged score of every model side by side."""
    d = pred.groupby(["model"] + KEYS, as_index=False).agg(pred=("pred", "mean"), y=("y", "first"),
                                                          prev=("prev", "first"))
    P = d.pivot(index=KEYS, columns="model", values="pred")
    meta = (d.drop_duplicates(KEYS).set_index(KEYS)[["y", "prev"]]).loc[P.index]
    return P, meta


def _ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else np.nan


def cluster_bootstrap(P, meta, n=N_BOOT, seed=BOOT_SEED):
    """Resample whole hexagons (all their weeks together) so repeated weeks of one place are not
    treated as independent. Returns dict[(model, subset)] -> array of PR-AUC over resamples."""
    hexes = meta.index.get_level_values("h3_r8")
    uniq = hexes.unique()
    by_hex = {h: np.flatnonzero(hexes == h) for h in uniq}
    y_all, prev_all = meta["y"].values, meta["prev"].values
    rng = np.random.default_rng(seed)
    out = {(m, s): np.empty(n) for m in P.columns for s in ("all", "emerging")}
    for b in range(n):
        idx = np.concatenate([by_hex[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        y, em = y_all[idx], prev_all[idx] == 0
        for m in P.columns:
            p = P[m].values[idx]
            out[(m, "all")][b] = _ap(y, p)
            out[(m, "emerging")][b] = _ap(y[em], p[em])
    return out


def build_table(pred):
    base = M.summarise_t2(pred)                       # Abdul's metrics: PR-AUC, F1, Brier, n
    P, meta = seed_average(pred)
    boots = cluster_bootstrap(P, meta)
    rows = []
    for _, r in base.iterrows():
        m, s = r["model"], r["subset"]
        b = boots[(m, s)]
        ref = boots[(REFERENCE, s)]
        delta = b - ref
        rows.append(dict(model=m, subset=s, pr_auc=r["pr_auc"],
                         pr_auc_lo=np.nanpercentile(b, 2.5), pr_auc_hi=np.nanpercentile(b, 97.5),
                         delta_vs_ref=r["pr_auc"] - base[(base["model"] == REFERENCE) & (base["subset"] == s)]["pr_auc"].iloc[0],
                         delta_lo=np.nanpercentile(delta, 2.5), delta_hi=np.nanpercentile(delta, 97.5),
                         f1=r["f1"], brier=r["brier"], n=int(r["n"])))
    out = pd.DataFrame(rows)
    out["subset"] = pd.Categorical(out["subset"], ["all", "emerging"])
    return out.sort_values(["subset", "pr_auc"], ascending=[True, False]).reset_index(drop=True)


def chosen_params(pred):
    fits = (pred[~pred["model"].isin(MD.T2_BASELINES)]
            .groupby(["model", "fold", "seed"])["params"].first().reset_index())
    rows = []
    for model, g in fits.groupby("model"):
        grid = [list(p.values())[0] for p in MD.T2_GRIDS[model]]
        picked = [list(ast.literal_eval(s).values())[0] for s in g["params"]]
        rows.append(dict(model=model, knob=list(MD.T2_GRIDS[model][0].keys())[0], grid=str(grid),
                         n_fits=len(picked), frac_at_min=sum(v == min(grid) for v in picked) / len(picked),
                         frac_at_max=sum(v == max(grid) for v in picked) / len(picked),
                         picked=str(sorted(set(picked)))))
    out = pd.DataFrame(rows)
    out["edge_warning"] = (out["frac_at_min"] >= 0.75) | (out["frac_at_max"] >= 0.75)
    out.to_csv(paths.TABLES / "E4_chosen_params.csv", index=False)
    return out


def figure(pred):
    style.apply()
    P, meta = seed_average(pred)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, (title, mask) in zip(axes, (("all hexagons", np.ones(len(meta), bool)),
                                        ("emerging only (not hot last week)", meta["prev"].values == 0))):
        y = meta["y"].values[mask]
        for i, m in enumerate(P.columns):
            if m == "persistence":      # binary scores give a degenerate curve; the impact version is shown
                continue
            pr, rc, _ = precision_recall_curve(y, P[m].values[mask])
            ax.plot(rc, pr, lw=2.2 if m == REFERENCE else 1.2, color="#777777" if m == REFERENCE else None,
                    ls="--" if m == REFERENCE else "-", label=m)
        ax.axhline(y.mean(), color="#bbbbbb", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[1].legend(loc="upper right", fontsize=6.5)
    style.savefig(fig, "E4_pr_curves")
    plt.close(fig)


def main():
    paths.ensure_dirs()
    t0 = time.time()
    feat = F.t2_features(D.load_hex_week())
    print(f"[E4] {len(feat):,} rows, origin weeks {feat['week'].min()}-{feat['week'].max()}, "
          f"{feat['h3_r8'].nunique()} hexagons", flush=True)
    pred = MD.run_cv("t2", MD.T2_MODELS, feat=feat)
    present = set(pred["model"])
    assert {"persistence", REFERENCE} <= present, f"baselines missing from E4 output: {present}"
    pred.to_parquet(paths.PREDICTIONS / "E4_hotspots_predictions.parquet", index=False)

    table = build_table(pred)
    table.to_csv(paths.TABLES / "E4_hotspots.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(), flush=True)
    print(chosen_params(pred).to_string(), flush=True)
    figure(pred)
    report_tables.write_e4()
    print(f"[E4] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
