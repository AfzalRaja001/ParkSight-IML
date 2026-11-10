"""E2: censoring benchmark. Naive vs Horvitz-Thompson weighting, cap sweep, estimated exposure.

Grid: 2 families (same_for_all, station_hetero) x 2 severities (0.5, 1.0) x 20 censoring seeds x
estimators: naive; HT-oracle (true e) with cap in {2, 5, 12, 50, none}; HT-estimated (e refitted on
the thinned tickets by the hackathon method, floor 0.05, cap 12, the hackathon setting).

The exposure curve of each (family, severity) is drawn once (seed 0) and only the thinning varies
over seeds 0-19, so the spread across seeds is the estimator's own variance.

Output (all under results/):
  tables/E2_benchmark.csv / .tex   per (family, severity, method): rel_bias, MAE, Kendall tau with 95%
                                   bootstrap CIs over seeds, and bias^2 / variance over station-hour cells
                                   (bias^2 has the var/n_seeds noise of the seed mean subtracted)
  tables/E2_runs.csv               one row per (family, severity, seed, method)
  tables/E2_exposure_drift.csv     true vs re-estimated exposure by hour (only with device data)
  figures/F2_benchmark.pdf         Figure 2: relative bias vs severity per method, and bias^2 vs variance by cap

HT-estimated needs research_data/tickets_device.parquet (gitignored). Without it that arm is dropped
and the oracle benchmark still runs (plan section 5, fallback).
Run:  python experiments/e2_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from parksight import config as C  # noqa: E402
from parksight.research import censor_benchmark as B  # noqa: E402
from parksight.research import paths, style  # noqa: E402
from parksight.research.metrics import bootstrap_ci  # noqa: E402

FAMILIES = ["same_for_all", "station_hetero"]
SEVERITIES = [0.5, 1.0]
SEEDS = range(20)
CAPS = [2, 5, 12, 50, None]
EST_CAP = C.IPW_WEIGHT_CAP
METRICS = ["rel_bias", "mae", "kendall_tau_station"]


def method_name(kind, cap):
    return kind if kind == "naive" else f"{kind}_cap{cap:g}" if cap is not None else f"{kind}_uncapped"


def load_truth():
    """Pseudo-truth tickets [id, police_station, hour (+ device_id)]; device_id joined on id if available."""
    t = pd.read_parquet(C.PROCESSED / "violations_clean.parquet", columns=["id", "police_station", "hour"])
    t = B.truth_window(t)
    if paths.DEVICE_FILE.exists():
        dev = pd.read_parquet(paths.DEVICE_FILE, columns=["id", "device_id"]).drop_duplicates("id")
        t = t.merge(dev, on="id", how="left")
        print(f"device_id joined for {t['device_id'].notna().mean():.1%} of truth tickets")
    else:
        print(f"{paths.DEVICE_FILE} missing: HT-estimated arm dropped (oracle-only fallback)")
    return t


def estimators(thinned, e, has_device):
    """(method, cap, hat_counts) for every estimator; also returns the re-estimated curve (or None)."""
    out = [("naive", None, B.ht_counts(thinned, e.assign(e=1.0), cap=None))]
    out += [(method_name("ht_oracle", c), c, B.ht_counts(thinned, e, cap=c)) for c in CAPS]
    e_hat = None
    if has_device:
        e_hat = B.estimate_exposure_x1(thinned)
        out.append((method_name("ht_estimated", EST_CAP), EST_CAP,
                    B.ht_counts(thinned, e_hat, cap=EST_CAP, floor=C.EXPOSURE_FLOOR)))
    return out, e_hat


def run(truth):
    has_device = "device_id" in truth.columns
    tc = B.truth_counts(truth)
    runs, cells, drift = [], {}, []
    stations = sorted(truth["police_station"].unique())
    for family in FAMILIES:
        for sev in SEVERITIES:
            e = B.e_star(family, sev, seed=0, stations=stations)
            true_e = e.groupby("hour")["e"].mean()       # citywide average, comparable to the one-curve estimate
            for seed in SEEDS:
                thinned = B.censor(truth, e, seed=seed)
                ests, e_hat = estimators(thinned, e, has_device)
                for method, cap, hat in ests:
                    runs.append(dict(family=family, severity=sev, seed=seed, method=method,
                                     cap=cap, **B.recovery_scores(tc, hat)))
                    n_hat = tc[B.KEYS].merge(hat, on=B.KEYS, how="left")["n_hat"].fillna(0).to_numpy()
                    cells.setdefault((family, sev, method), []).append(n_hat)
                if e_hat is not None:
                    drift += [dict(family=family, severity=sev, seed=seed, hour=h, e_true=true_e[h], e_hat=v)
                              for h, v in e_hat.items()]
    return pd.DataFrame(runs), cells, tc["n"].to_numpy(dtype=float), pd.DataFrame(drift)


def summarise(runs, cells, truth_n):
    rows = []
    for (family, sev, method), g in runs.groupby(["family", "severity", "method"], sort=False):
        row = dict(family=family, severity=sev, method=method, cap=g["cap"].iloc[0], n_seeds=len(g))
        for m in METRICS:
            lo, hi = bootstrap_ci(g[m])
            row.update({m: g[m].mean(), f"{m}_lo": lo, f"{m}_hi": hi})
        est = np.vstack(cells[(family, sev, method)])          # seeds x station-hour cells
        var = est.var(axis=0, ddof=1)
        # the squared error of a mean over n seeds carries var/n of noise; remove it so bias2 is unbiased
        row["bias2"] = float(np.clip((est.mean(axis=0) - truth_n) ** 2 - var / len(est), 0, None).mean())
        row["variance"] = float(var.mean())
        row["mse"] = float(((est - truth_n) ** 2).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_tex(s):
    def ci(v, lo, hi, nd):
        return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    t = pd.DataFrame({
        "Family": s["family"].str.replace("_", r"\_"),
        "Severity": s["severity"].map("{:.1f}".format),
        "Method": s["method"].str.replace("_", r"\_"),
        r"Rel. bias [95\% CI]": [ci(a, b, c, 3) for a, b, c in zip(s["rel_bias"], s["rel_bias_lo"], s["rel_bias_hi"])],
        "MAE": s["mae"].map("{:.1f}".format),
        r"Kendall $\tau$": s["kendall_tau_station"].map("{:.3f}".format),
        r"Bias$^2$": s["bias2"].map("{:.1f}".format),
        "Variance": s["variance"].map("{:.1f}".format),
    })
    (paths.TABLES / "E2_benchmark.tex").write_text(
        t.to_latex(index=False, escape=False, column_format="lllrrrrr"), encoding="utf-8")


def figure2(s):
    """(a, b) relative bias vs severity per method, one panel per family; (c) bias^2 vs variance by cap."""
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    methods = list(dict.fromkeys(s["method"]))
    colors = style.PALETTE[:6] + style.PALETTE[7:]      # skip the pale yellow, hard to read as a line
    for ax, family in zip(axes[:2], FAMILIES):
        f = s[s["family"] == family]
        for i, m in enumerate(methods):
            g = f[f["method"] == m].sort_values("severity")
            ax.errorbar(g["severity"], 100 * g["rel_bias"],
                        yerr=[100 * (g["rel_bias"] - g["rel_bias_lo"]), 100 * (g["rel_bias_hi"] - g["rel_bias"])],
                        marker="o", ms=3, lw=1.2, capsize=2, color=colors[i % len(colors)],
                        label=m.replace("_", " "))
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks(SEVERITIES)
        ax.set_xlabel("censoring severity")
        ax.set_title(family.replace("_", " "))
    axes[0].set_ylabel("relative bias of total (%)")

    sweep = s[(s["family"] == "same_for_all") & (s["severity"] == 1.0) & s["method"].str.startswith("ht_oracle")]
    labels = ["none" if pd.isna(c) else f"{c:g}" for c in sweep["cap"]]
    x = np.arange(len(sweep))
    axes[2].bar(x - 0.2, sweep["bias2"], 0.4, label="bias$^2$", color=style.PALETTE[0])
    axes[2].bar(x + 0.2, sweep["variance"], 0.4, label="variance", color=style.PALETTE[1])
    axes[2].set_yscale("log")
    axes[2].set_ylim(top=sweep[["bias2", "variance"]].to_numpy().max() * 50)   # headroom for the legend
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("weight cap (oracle e, severity 1.0)")
    axes[2].set_ylabel("per cell (tickets$^2$)")
    axes[2].set_title("cap trade-off")
    axes[2].legend(frameon=False, ncol=2, loc="upper center")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return style.savefig(fig, "F2_benchmark")


def main():
    paths.ensure_dirs()
    truth = load_truth()
    print(f"truth window: {len(truth):,} tickets, {truth['police_station'].nunique()} stations")
    runs, cells, truth_n, drift = run(truth)
    summary = summarise(runs, cells, truth_n)
    runs.to_csv(paths.TABLES / "E2_runs.csv", index=False)
    summary.to_csv(paths.TABLES / "E2_benchmark.csv", index=False)
    write_tex(summary)
    if len(drift):
        (drift.groupby(["family", "severity", "hour"], as_index=False)
         .agg(e_true=("e_true", "first"), e_hat=("e_hat", "mean"), e_hat_sd=("e_hat", "std"))
         .to_csv(paths.TABLES / "E2_exposure_drift.csv", index=False))
    print(summary[["family", "severity", "method", "rel_bias", "mae", "kendall_tau_station",
                   "bias2", "variance"]].to_string(index=False))
    print(f"\nFigure 2 -> {figure2(summary)}")


if __name__ == "__main__":
    main()
