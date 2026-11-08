# ParkSight for IML: Lean Implementation Plan (V2)

**Written:** Sat 7 Nov 2026. **Deadline:** Poster + Final Report, Sun 15 Nov. **Target submission:** Sat 14 Nov (Sunday is buffer).
**This replaces** `IMPLEMENTATION_PLAN.md` (V1). V1 was sized for 17 days, and 8 remain. Everything that is not here is cut, and section 7 lists exactly what was cut and what to say about it in the report.
**Companion:** `TEAM_WORK_SPLIT_V2.md` (who does what, day by day). Background: `PARKSIGHT_EXPLAINED.md`.

> **If you already built something from V1**, keep it. V2 is a strict subset, so nothing needed here conflicts with it. Only stop working on things that V2 cut.

---

## 0. The idea in one paragraph

Tickets only exist where an officer was, and officers almost vanish in the evening. We show three things with numbers: **(1)** the evening gap is real in this data, **(2)** on a test where we know the truth, naive models under-count badly and inverse-propensity weighting helps but becomes unstable when exposure is tiny, **(3)** which classical models forecast tickets and hotspots best, with honest uncertainty intervals. That is a complete, defensible story for a 4 to 5 page report.

**The 6 pieces (E1 to E6).** Each is one script, one results CSV, one or two figures.

| ID | Piece | Question it answers |
|---|---|---|
| E1 | Evening-gap evidence + time sanity check | Is the censoring real, and are the timestamps trustworthy? |
| E2 | **Censoring benchmark** (naive vs weighting, cap sweep) | How much does censoring hurt, and how well does weighting recover? **The core contribution.** |
| E3 | Count forecasting: classical model comparison | Which classical model forecasts station-day tickets best? |
| E4 | Hotspot classification vs the persistence baseline | Can a model beat "last week's hotspot is this week's hotspot"? |
| E5 | Conformal intervals: global vs per-tier | Are our uncertainty bands honest? |
| E6 | PCA and K-Means on hourly profiles | Do stations differ in *when* they get tickets? |

---

## 1. Global rules (keep these, they protect the result)

1. **Classical ML only.** No neural networks.
2. **Time-blocked evaluation only.** Train strictly before test. Never shuffle.
3. **No fitting on test data.** Features are built from earlier days only (shifted), so leakage is impossible by construction. Anything fitted (hyperparameters, tiers, exposure curves) uses the training part only.
4. **Fixed seeds.** Models: `0, 1, 2`. Benchmark: `0` to `19`. Bootstrap: `12345`.
5. **Numbers come from CSVs.** Every number in the report is in `results/tables/*.csv`. Each experiment script also writes the LaTeX table (`df.to_latex`), so nothing is retyped.
6. **Hackathon code stays frozen.** New code goes in `parksight/research/`. It may import from the old code.
7. **Disclose the origin.** The report says ParkSight began as Afzal's hackathon project before the course and lists what was added. Methods we did not invent (inverse-propensity weighting, conformal prediction, LightGBM, DBSCAN, H3) are cited.
8. **Ticket-level data with `device_id` is not committed.** Only aggregates and results go in git.
9. **Docs style.** Plain hyphen, never an em dash. Plain commit messages with no co-author lines.

---

## 2. Facts from the real data that shape the plan

| Fact | Value | What it means |
|---|---|---|
| Date span | 10 Nov 2023 to 8 Apr 2024 (151 days) | 4 folds of 14 days |
| Station x day grid | 54 x 151, 6.1% zeros | Count forecasting is done here |
| Station volume | 339 to 34,468 tickets | One station is huge, so headline metrics are per-station (macro) |
| Full weeks | 21 (first and last are partial and dropped) | Hotspot folds |
| Hotspot label (top 10% by weekly impact) | prevalence 10.1%, about 78 per week | Imbalanced, so PR-AUC is the headline |
| **Persistence baseline** | **PR-AUC 0.749** | Hotspots barely move week to week. A model must beat this, and we also score the **emerging hotspots** (hexagons not hot last week) |
| Benchmark truth window 02:00 to 11:59 | 240,352 tickets, median 226 per station-hour | Plenty of data to thin and recover |
| Real exposure at 19:00 | 0.0068 (weight would be 147; the hackathon code caps it at 12) | The cap is what the benchmark tests |

---

## 3. The two prediction tasks (definitions)

**T1, count forecasting.** For each police station and day, predict **next day's** ticket count. Grain: station x day. (Horizon 1 only.)

**T2, hotspot classification.** For each hexagon (H3 resolution 8) and week, predict whether it will be a hotspot **next week**. Hotspot means: in the top 10% of that week's hexagons by **weekly impact**, where `impact = sum(severity * peak_w)` over the week's tickets. This deliberately excludes the PCIS trend term, because the repo's trend is computed over the whole period and would leak the future.

**Folds (pinned).**
- T1: expanding window, 4 folds, test blocks of 14 days starting `2024-02-13`, `2024-02-27`, `2024-03-12`, `2024-03-26` (the last ends `2024-04-08`).
- T2: expanding window, 4 folds, test blocks of 2 weeks over weeks `[13,14]`, `[15,16]`, `[17,18]`, `[19,20]` of the 21 full weeks.

**Honesty rule.** On real data there is no ground truth for the evening. Accuracy on real data is measured on observed counts only. Any claim about *recovering* the unobserved evening comes only from E2, where the truth is known by construction.

---

## 4. Files (small on purpose)

```
parksight/research/
  data.py               load_station_day, load_hex_week, day_folds, week_folds         (Shaurya)
  features.py           t1_features, t2_features                                        (Shaurya)
  models.py             model lists, fit_predict, run_cv                                (Afzal)
  metrics.py            all metrics, bootstrap CI, Wilcoxon                             (Abdul)
  conformal.py          global and per-tier intervals                                   (Abdul)
  censor_benchmark.py   e_star, censor, HT weighting, recovery scores                   (Khushi)
  unsup.py              PCA and K-Means                                                 (Shaurya)
experiments/
  e1_eda.py  e2_benchmark.py  e3_counts.py  e4_hotspots.py  e5_conformal.py  e6_unsup.py  run_all.py
tests/research/         six tests only (section 6)
research_data/          GITIGNORED (device-level file)
results/tables/*.csv  results/figures/*.pdf
```

---

## 5. What each file must do (interfaces)

Signatures are fixed so everyone can code against them from day one.

### `data.py` (Shaurya)
```python
def load_station_day() -> pd.DataFrame      # [police_station, date, n]; full 54 x 151 grid, zeros filled
def load_hex_week() -> pd.DataFrame         # [h3_r8, week, n, impact, y]; 21 full weeks; y = hotspot label of THAT week
def day_folds() -> list[Fold]               # 4 pinned folds
def week_folds() -> list[Fold]              # 4 pinned folds
# Fold = dataclass(idx, train_end (exclusive), test_start, test_end (inclusive))
```
Reads the committed `violations_clean.parquet`. No raw file needed for these.

### `features.py` (Shaurya)
```python
FEATURES_T1: list[str]
FEATURES_T2: list[str]
def t1_features(sd: pd.DataFrame) -> pd.DataFrame   # adds columns; target column is y = n of the NEXT day
def t2_features(hw: pd.DataFrame) -> pd.DataFrame   # adds columns; target column is y_next = label of the NEXT week
```
- T1 features: counts 1, 7 and 14 days ago; means over the last 7 and 28 days; expanding same-weekday mean; weekday; weekend flag; holiday flag (country IN). Station as a category.
- T2 features: weekly impact 1 to 4 weeks ago; mean of the last 4 weeks; share of evening tickets; mean severity; junction share; mean of last week's impact over the six neighbouring hexagons (`h3.grid_disk(cell, 1)` minus the centre).
- Every feature uses only data from before the predicted day or week (use shifts). Rows with no history are dropped.

### `models.py` (Afzal)
```python
T1_MODELS = ["seasonal_naive", "ma7", "poisson_glm", "ridge", "lasso", "random_forest", "lightgbm"]   # + "svr_rbf" if time
T2_MODELS = ["persistence", "logreg", "random_forest", "lightgbm"]                                       # + "linear_svm" if time

def fit_predict(task: str, model: str, train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray
def run_cv(task: str, models: list[str], seeds=(0, 1, 2)) -> pd.DataFrame
# run_cv returns one row per test row: [model, fold, seed, key columns, y, pred]
```
Model choices:
- `seasonal_naive` (same weekday last week) and `ma7` are the baselines.
- `poisson_glm`: `PoissonRegressor`, standardised numeric features, one-hot station.
- `ridge`, `lasso`: on `log1p(y)`; Lasso also prints its selected features.
- `random_forest`: Poisson criterion for counts. **This is the Gao et al. replication baseline.**
- `lightgbm`: the hackathon parameters (`BASE_PARAMS` in `train_forecast.py`) with a Poisson objective.
- T2: `persistence` (last week's label), `logreg` (L2, balanced class weights), `random_forest`, `lightgbm`.
- **Tuning is kept tiny:** a 3-value grid per model, scored on the **last 14 days of the training window** (an inner time-blocked split), never on the test block.

### `metrics.py` (Abdul)
```python
def summarise_t1(pred: pd.DataFrame) -> pd.DataFrame   # per model: macro MASE, pooled MASE, RMSE, Poisson deviance, with bootstrap CIs
def summarise_t2(pred: pd.DataFrame) -> pd.DataFrame   # per model: PR-AUC, F1 at the best inner-validation threshold, Brier; on ALL hexagons and on the EMERGING subset
def bootstrap_ci(values, n: int = 2000, seed: int = 12345) -> tuple[float, float]   # resamples stations
def wilcoxon_vs(pred: pd.DataFrame, model_a: str, model_b: str) -> float            # paired over stations, per-station MASE
def coverage(y, lo, hi) -> float
```
Definitions:
- **Headline MASE is per station, macro-averaged**, using each station's own training-window seasonal-naive MAE as denominator. Also report a **pooled MASE** with the test-window denominator, because that is what the hackathon's 0.73 used. The two are not comparable, so state which is which.
- Poisson deviance clips predictions at `1e-6`.
- A station with a (near) zero denominator gets an epsilon floor `1e-6` and is **flagged in the output**, never silently dropped.
- Emerging subset = hexagons that were **not** hotspots in the previous week.

### `conformal.py` (Abdul)
```python
def global_interval(cal_resid: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.20) -> pd.DataFrame        # adds lo, hi
def tier_interval(cal_resid: pd.DataFrame, test: pd.DataFrame, tiers: pd.Series, alpha: float = 0.20) -> pd.DataFrame
```
Use the quantile rule `ceil((n + 1)(1 - alpha)) / n`. Tiers are low, mid, high station volume (terciles of **training** ticket totals). The calibration block is the last 21 days of each training window.

### `censor_benchmark.py` (Khushi)
```python
def truth_window(tickets: pd.DataFrame, hours=range(2, 12)) -> pd.DataFrame
def e_star(family: Literal["same_for_all", "station_hetero"], severity: float, seed: int) -> pd.DataFrame   # [police_station, hour, e], max e = 1
def censor(truth: pd.DataFrame, e: pd.DataFrame, seed: int) -> pd.DataFrame                              # Bernoulli thinning, keeps device_id if present
def estimate_exposure_x1(thinned: pd.DataFrame) -> pd.Series                                              # distinct devices per hour / max (the hackathon method)
def ht_counts(thinned: pd.DataFrame, e, cap: float | None) -> pd.DataFrame                                # per station-hour: n_obs, n_hat = sum of min(1/e, cap)
def recovery_scores(truth_counts, hat_counts) -> dict                                                    # rel_bias, mae, kendall_tau_station
```
How the benchmark works:
1. **Pseudo-truth** = every ticket from 02:00 to 11:59. It is itself slightly censored in reality, so we say we measure recovery of this reference, not an absolute truth.
2. **Censoring curve `e_star`** = the shape of the real exposure curve over hours 12 to 21 (the dip), mapped onto the 10 truth hours and divided by its first value so the best hour keeps everything. `severity` is an exponent: `1.0` is the real severity, `0.5` is milder. `station_hetero` raises each station's curve to its own random power, so a single hourly curve is deliberately wrong.
3. **Thin** each ticket with probability `e_star(station, hour)`.
4. **Estimate** by: `naive` (count what is left), `HT-oracle` (weight by the true `e`, cap in `{2, 5, 12, 50, none}`), and `HT-estimated` (weight by `estimate_exposure_x1` fitted on the thinned data, cap 12, the hackathon setting).
5. **Score** against the pseudo-truth on station-hour cells: relative bias of the total, mean absolute error, Kendall tau of the station ranking. Average over 20 seeds with bootstrap intervals, and for the cap sweep split error into bias squared and variance across seeds.

**Fallback.** If device-level data cannot be shared, drop `HT-estimated` and keep `HT-oracle`. The benchmark still stands.

### `unsup.py` (Shaurya)
```python
def hourly_profile_matrix(tickets) -> pd.DataFrame          # 54 x 24, rows sum to 1
def run_pca(profile) -> tuple[pd.DataFrame, np.ndarray]     # scores, explained variance
def run_kmeans(profile, k_range=range(2, 7), seed=0) -> dict   # silhouette per k, chosen k, labels
```

---

## 6. Tests (six only, each protects against a real risk)

| Test | File | Protects against |
|---|---|---|
| `test_no_future_leak`: change every value from day `D` on; features for rows before `D` and for `D` do not change | `test_features.py` | Leakage, the most likely silent error |
| `test_folds_no_overlap`: `train_end <= test_start` for every fold, and the four T1 test starts equal the pinned dates | `test_data.py` | Wrong splits |
| `test_mase_hand_example`: `actual=[10,12,9,14]`, `model=[11,11,10,12]`, naive MAE 3.0 gives 0.417 | `test_metrics.py` | Wrong metric |
| `test_censor_retention`: retained fraction per station-hour is within 3 standard errors of `e_star` for 95% of cells | `test_benchmark.py` | Broken thinning |
| `test_ht_unbiased`: simulate Poisson counts with `e = 0.3`, 200 seeds, uncapped HT mean is within 3% of the truth | `test_benchmark.py` | Wrong weighting |
| `test_conformal_coverage`: on synthetic exchangeable data, mean coverage is within 0.04 of 0.80 | `test_conformal.py` | Wrong intervals |

Plus one assert inside `e4_hotspots.py`: the output must contain `persistence` rows, or the script fails.

---

## 7. The experiments

### E1: Evening gap + time sanity (Shaurya)
- **Figure 1:** tickets per hour and distinct active devices per hour, on the same axis (shows the evening collapse: 1,773 devices at 11:00, 12 at 19:00).
- **Sanity check (about one hour):** is the timestamp trustworthy? Every timestamp has seconds equal to 46, and 34% of tickets fall between 00:00 and 06:00 IST. Using device IDs, compute each device's first and last ticket per day and look at shift start times and lengths; compare weekdays and weekends. Write 3 sentences on what the data supports. If the offset looks wrong, tell the group the same day, because it moves the evening window.
- **Output:** `E1_eda.csv`, Figure 1.

### E2: Censoring benchmark (Khushi)
- **Grid:** 2 families x 2 severities (`0.5`, `1.0`) x 20 seeds x estimators (naive; HT-oracle with 5 caps; HT-estimated cap 12).
- **Expected:** naive under-counts, more as severity rises. HT with a big cap has high variance where `e < 0.05`. Capping lowers variance but leaves bias. An estimated exposure curve drifts from the true one when tickets are sparse (the circularity), and `station_hetero` hurts every single-curve method.
- **Output:** `E2_benchmark.csv`, **Figure 2 (the central figure):** recovery vs severity with one line per method, and a small panel of bias squared vs variance by cap.

### E3: Count forecasting (Afzal)
- `run_cv("t1", T1_MODELS)`. Report macro MASE with 95% intervals, pooled MASE, RMSE, Poisson deviance. Wilcoxon of each model against `seasonal_naive` and against `random_forest`.
- **Small ablation (best model only):** drop the lag features, drop the calendar features, drop the station feature. Three runs.
- **Expected:** tree ensembles lead on accuracy, Poisson GLM wins on interpretability, a plain moving average stays close.
- **Output:** `E3_counts.csv`, `E3_ablation.csv`, Table 1.

### E4: Hotspots (Afzal; labels from Shaurya)
- `run_cv("t2", T2_MODELS)`. Report PR-AUC, F1, Brier on **all hexagons** and on the **emerging subset**. `persistence` must appear in every table.
- **Output:** `E4_hotspots.csv`, Table 2, Figure 4 (precision-recall curves with persistence).

### E5: Conformal (Abdul)
- For the best E3 model, per fold: train on days before the calibration block, calibrate on the 21 days before `train_end`, predict the test block. Compare global and per-tier intervals at `alpha = 0.20`. Report coverage and mean width by tier. Target 0.80.
- **Output:** `E5_conformal.csv`, Table 3.
- **Also (15 minutes):** stratified error of the best model by station tier and by weekday vs weekend.

### E6: PCA and K-Means (Shaurya)
- PCA on the 54 x 24 hourly-profile matrix: scree plot and PC1 loadings by hour. **Hypothesis:** PC1 separates stations by *when* they record, not how much. If it does not, say so. K-Means for `k = 2..6` with silhouette, then name the archetypes.
- **Output:** `E6_unsup.csv`, Figure 5.

---

## 8. What was cut from V1, and what to write in the report

Say these honestly in the report's Limitations and Future Work. A clear, stated scope beats a half-finished extra.

| Cut from V1 | Why it is safe | Report wording |
|---|---|---|
| Exposure estimators X2 to X5, empirical-Bayes estimator | E2 already tests the hackathon estimator and the cap | "Alternative exposure estimators and shrinkage are future work." |
| Spatial-scale replication (station vs hex-r8 vs hex-r9) | Station x day and hex x week already cover two scales | "We did not replicate the full scale comparison." |
| Direct horizon-7 forecasts | Horizon 1 is enough to rank models | "We evaluate next-day forecasts only." |
| 6-hour-block grain and the by-hour error analysis | E1 covers the hour pattern with device counts | "Error analysis covers station tier and weekday." |
| Weighted conformal, rank-stability study, PCIS weight sensitivity | Stretch goals | "Left as future work." |
| Corrected target on real data (E12) | Cannot be scored without ground truth anyway | "Applying the correction to real data is future work." |
| Cluster-label ablation, MLP control, OSM features | Not needed for the main claim | omit |
| Heavy test suite, many modules, fit/transform feature builders | Features are shift-based, so leakage is impossible by construction | n/a |

**Proposal promises now covered lightly or dropped.** Poisson regression, tree, forest and boosting ladder, hotspot classification, PCA and K-Means, inverse-propensity weights, Mondrian conformal, rolling-origin evaluation, MASE, RMSE, Poisson deviance, PR-AUC, F1, Brier, coverage, the benchmark and a small ablation are all kept. Dropped: the full spatial-scale comparison, by-hour and by-vehicle error analysis, the cluster-label ablation. The report should mention these in one sentence.

---

## 9. Schedule

| Date | What | Gate |
|---|---|---|
| **Sat 7** | Kick-off call (30 min). Setup. Afzal: tag + branch + folder skeleton (30 min). Shaurya: `data.py`. Khushi: `e_star`, `censor`. Abdul: Overleaf, `metrics.py`. Afzal: `models.py` on synthetic data. | **G0 tonight:** decisions answered (section 11), skeleton merged |
| **Sun 8** | `features.py`, baselines and linear models on real data, `censor` and HT tested, metrics tested, figure style | **G1 (night):** `seasonal_naive`, `ma7`, `poisson_glm` run end to end with real metrics; `censor` tested |
| **Mon 9 to Tue 10** | All experiments produce a v1 table. Report sections that need no results get written (Intro, Related Work, Data, Method). | |
| **Wed 11** | Finish E2 and E3, send best-model predictions for E5, draft the Results text | **G2 (night):** all six E-tables exist, central figure drafted |
| **Thu 12** | Final runs in the morning. **Code freeze at 12:00.** `run_all.py` reproduces everything. Poster panels. | **G3 (noon): freeze** |
| **Fri 13** | Full draft in Overleaf, read-through call (45 min), poster draft | **G4 (evening):** complete draft |
| **Sat 14** | Final proofread, PDF check (fonts, page limit, references), submit. One person submits on Google Classroom and tags the other three. | Submitted |
| Sun 15 | Buffer only | |

**Cut order if time runs short (cut from the top first):** SVM and SVR, K-Means, the ablation, the stratified error, `station_hetero`, the Wilcoxon tests. **Never cut:** the E3 ladder with baselines and intervals, E4 with persistence, the E2 cap sweep, E5, Figure 1.

---

## 10. Report plan (4 to 5 pages, ICML or NeurIPS style, references excluded)

| Section | Pages |
|---|---|
| Abstract | 0.15 |
| Introduction (problem, three questions, contributions, hackathon disclosure) | 0.45 |
| Related work (Gao 2019, Ensign 2018, Schnabel 2016, Liu and Chen 2025, Inam 2022, Grinsztajn 2022) | 0.3 |
| Data and tasks (dataset, T1, T2, label, folds) | 0.5 |
| Methods (models, benchmark, conformal, metrics) | 0.9 |
| Results (Table 1 counts, Table 2 hotspots, Figure 2 benchmark, Table 3 coverage, Figures 1 and 5) | 1.4 |
| Discussion and limitations | 0.5 |
| Conclusion | 0.1 |
| Contributions | 0.15 |

**Figures:** F1 evening gap, **F2 benchmark (the main one)**, F3 hotspot PR curves, F4 PCA. **Tables:** counts ladder, hotspot ladder, coverage.

**Limitations to state:** five months and one city; no congestion ground truth (PCIS is a proxy); a zero count may mean "no violation" or "no officer"; the benchmark's truth is itself slightly censored; timestamps were transformed (E1 check); the hackathon's MASE and ours use different denominators.

---

## 11. Decisions needed tonight (Sat 7)

1. **Device data.** E2's circularity check and E1's shift check need `device_id`. Afzal builds one small file `research_data/tickets_device.parquet` (columns: id, police_station, date, hour, device_id) from the raw CSV and shares it only with Shaurya and Khushi. If hackathon terms forbid it, use the fallback in section 5.
2. **Grain change.** Counts at station x day and hotspots at hex-r8 x week, instead of the proposal's "hexagon by hour" (that grid is 99% empty).
3. **Scope.** Confirm section 8 is acceptable to everyone.
4. **Submitter.** Pick the one person who submits on Google Classroom.
5. **Poster format.** Ask the instructor now if it has not been shared.

---

## 12. Self-review

- **Coverage:** every piece in section 0 has an owner, an interface, a test or an assert, and an output file.
- **Consistency:** `Fold`, `fit_predict`, `run_cv`, `summarise_t1`, `summarise_t2`, `ht_counts` and `recovery_scores` keep the same name and meaning wherever they appear.
- **Size:** 7 modules (V1 had 17), 6 experiments (V1 had 13), 6 tests (V1 had about 45), 4 to 5 report pages.
- **Assumption to confirm:** this plan assumes the V1 work is mostly not started. If parts exist, reuse them and reduce the time estimates in the split document.
