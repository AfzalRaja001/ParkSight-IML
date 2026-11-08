# ParkSight for IML: Lean Team Work Split (V2)

**Written:** Sat 7 Nov 2026. **Deadline:** Sun 15 Nov. **Target submission:** Sat 14 Nov.
**This replaces** `TEAM_WORK_SPLIT.md` (V1). Technical details are in `IMPLEMENTATION_PLAN_V2.md`. IDs below (E1 to E6) are the V2 ones, not the old E00 to E12.

The four roles from the submitted proposal are unchanged, so the report's "Contributions" section stays true. Each person now has **one main deliverable, one smaller one, and one report section group**, about 22 to 24 hours over 8 days.

---

## 1. Rules (short)

1. **One owner per file.** Only the owner edits it. Need a change? Ask the owner.
2. **Signatures are fixed** (see the plan, section 5), so everyone can start today using stubs.
3. **Own branch, quick merge.** Branch `feat/<name>-<topic>`, merge into `iml-research` when your test passes. The three critical files (`data.py`, `metrics.py`, `censor_benchmark.py`) get a second pair of eyes before merging (section 5). Nobody touches `main`.
4. **Every number comes from a CSV** in `results/tables/`.
5. **Device-level data is never committed.**
6. **Daily message** in the group chat: yesterday, today, blocked. Blocked for more than half a day? Say so immediately.
7. **Freeze is Thu 12 Nov, 12:00.** After that, only text and figures change, not results.

---

## 2. At a glance

| | **Shaurya** | **Afzal** | **Khushi** | **Abdul** |
|---|---|---|---|---|
| **Main deliverable** | Data layer and features | Model comparison (E3, E4) | Censoring benchmark (E2) | Metrics, conformal (E5), LaTeX |
| **Smaller deliverable** | E1 evening gap + time check, E6 PCA and K-Means | Repo setup, `run_all.py` | Report on de-biasing | Figure style, stratified error |
| **Files owned** | `data.py`, `features.py`, `unsup.py` | `models.py`, `run_all.py` | `censor_benchmark.py` | `metrics.py`, `conformal.py` |
| **Report** | Data and tasks, Limitations | Introduction, Methods (models), counts and hotspot results, Conclusion | De-biasing method and results, Related work (Ensign, Schnabel), Discussion | Abstract, Methods (metrics, conformal), conformal results, LaTeX assembly |
| **Poster panel** | Data and the evening gap | Model results | The benchmark (largest) | Layout, uncertainty |
| **Second pair of eyes on** | gets: Afzal | gets: Abdul | gets: Abdul | gets: Khushi |
| **Backup for** | Afzal | Shaurya | Abdul | Khushi |

**Fairness note.** ParkSight began as Afzal's hackathon project and is the foundation. The report says so. Afzal's course contribution is the new work above, not the old code. That keeps credit for Shaurya, Khushi and Abdul clear.

---

## 3. Person cards

### Shaurya: data, features, evidence

**Mission.** Make the datasets and features correct and leak-free, then show with data that the evening gap is real.

| Date | Do | Done when |
|---|---|---|
| **Sat 7** (about 3 h) | Setup. Read `PARKSIGHT_EXPLAINED.md` and plan sections 2 to 5. Write `data.py`: `load_station_day`, `load_hex_week` (with the hotspot label), `day_folds`, `week_folds`. Write `test_folds_no_overlap`. | 54 x 151 grid builds; 21 weeks; the four pinned fold dates match |
| **Sun 8** (about 2.5 h) | `features.py`: `t1_features`, `t2_features`. Write `test_no_future_leak`. Hand off to Afzal. | Leak test passes. **G1** |
| **Mon 9** (about 3 h) | E1: Figure 1 (tickets and devices by hour) and the time sanity check using the device file. | Figure 1 + 3 sentences on timestamps |
| **Tue 10** (about 3 h) | E6: PCA and K-Means (`unsup.py`), Figure 4. | `E6_unsup.csv`, figure |
| **Wed 11** (about 2 h) | Write **Data and tasks** and **Limitations**. Send Figure 1 to Abdul. | Draft in Overleaf |
| **Thu 12** (about 2 h) | Freeze at noon. Poster data panel. | Panel done |
| **Fri 13** (about 1.5 h) | Read-through, fix text. | |
| **Sat 14** (about 1 h) | Final proofread. | |

**Watch for.** The first and last weeks are partial and must be dropped. A zero station-day is a real zero, not missing. If the timestamp check finds a problem, tell the group the same day.

**Needs from others.** The device file (Afzal, Sat).
**Others need from you.** `data.py` by Sat night, `features.py` by Sun noon.

---

### Afzal: model comparison and setup

**Mission.** Run the main model comparison fairly and keep the repo organised.

| Date | Do | Done when |
|---|---|---|
| **Sat 7** (about 3.5 h) | Tag `hackathon-baseline`, create `iml-research`, add the three teammates as collaborators, create the folder skeleton. Build `tickets_device.parquet` and share it with Shaurya and Khushi. Start `models.py` and `run_cv` on synthetic data. | Everyone can clone and branch. **G0** |
| **Sun 8** (about 3 h) | Baselines (`seasonal_naive`, `ma7`), `poisson_glm`, `ridge`, `lasso` on real data. | Real rows for the baselines. **G1** |
| **Mon 9** (about 3 h) | `random_forest`, `lightgbm`, tiny grids. First full E3 run. | `E3_counts.csv` v1 |
| **Tue 10** (about 4 h) | T2 models and E4 (assert `persistence` is present). The three-run ablation. Send Abdul predictions from the best model by **Tue night**. | `E4_hotspots.csv` v1 |
| **Wed 11** (about 2.5 h) | Finalise E3 and E4. Write **Introduction** and **Methods (models)**. | **G2** |
| **Thu 12** (about 3 h) | Final runs in the morning. Freeze at noon. `run_all.py`. Write the counts and hotspot **Results** text and **Conclusion**. | **G3** |
| **Fri 13** (about 2 h) | Read-through. Poster model panel. | |
| **Sat 14** (about 1 h) | Submission checks. | |

**Watch for.** Always report `persistence` (PR-AUC about 0.75) and the emerging subset. Our MASE is macro per-station, the hackathon's 0.73 was pooled with a different denominator, so do not compare them directly.

**Needs from others.** Data and folds (Shaurya, Sat night), features (Shaurya, Sun noon), metrics (Abdul, Sun).
**Others need from you.** Skeleton and access (Sat), `fit_predict` and predictions (Tue night).

---

### Khushi: the censoring benchmark

**Mission.** Show with numbers how much censoring hurts and how well weighting fixes it. This is the report's central contribution.

| Date | Do | Done when |
|---|---|---|
| **Sat 7** (about 4 h) | Read `parksight/models/debias.py` (170 lines) and Ensign et al. 2018. Write `test_censor_retention` first, then `truth_window`, `e_star`, `censor`. | Retention test passes |
| **Sun 8** (about 3 h) | `ht_counts` and `recovery_scores`. Write `test_ht_unbiased`. | Both tests pass. **G1** |
| **Mon 9** (about 3 h) | `estimate_exposure_x1` on the thinned data (the circularity check). First E2 run. | v1 table |
| **Tue 10** (about 3.5 h) | Full E2: 2 families x 2 severities x 20 seeds x all estimators. Build the central figure with Abdul's style. | `E2_benchmark.csv`, Figure 2 draft |
| **Wed 11** (about 3 h) | Write **De-biasing method**, **E2 results**, and the Ensign and Schnabel paragraphs of **Related work**. | **G2** |
| **Thu 12** (about 3 h) | Freeze at noon. Poster benchmark panel (the largest). | Panel done |
| **Fri 13** (about 2 h) | Write the **Discussion** (the cap trade-off). Read-through. | |
| **Sat 14** (about 1 h) | Final proofread. | |

**Watch for.** (1) The benchmark measures recovery of a pseudo-truth, so say so. (2) Any exposure curve must be fitted on the **thinned** data only, never on the truth. (3) The hackathon cap of 12 binds for every hour from 16:00 to 23:00, so the cap sweep is the key experiment. (4) If the device file is not allowed, drop the estimated-exposure arm and keep the oracle one.

**Needs from others.** The device file (Afzal, Sat), plotting style (Abdul, Sun).
**Others need from you.** E2 table and figure by Wed.

---

### Abdul: metrics, uncertainty, LaTeX

**Mission.** Make sure everything is measured correctly and the report looks clean.

| Date | Do | Done when |
|---|---|---|
| **Sat 7** (about 3 h) | Create the Overleaf project from the ICML or NeurIPS template and invite the team. Write `test_mase_hand_example`, then `metrics.py` (`summarise_t1`, `summarise_t2`). Ask the instructor for the poster format. | Template compiles; hand-example test passes |
| **Sun 8** (about 2 h) | `bootstrap_ci`, `wilcoxon_vs`, `coverage`, and one shared figure style (vector PDF, colour-blind-safe palette). | Others can import metrics and style. **G1** |
| **Mon 9** (about 3 h) | `conformal.py` (`global_interval`, `tier_interval`). Write `test_conformal_coverage`. | Test passes |
| **Tue 10** (about 3 h) | E5 using Afzal's predictions. Stratified error by station tier and weekday. | `E5_conformal.csv` v1 |
| **Wed 11** (about 2 h) | Report skeleton: one `.tex` file per section. Draft the abstract. Optional 15-minute check: Spearman of the volume term alone against the 12 expert junctions (the repo reports 0.92 for the full PCIS). | **G2** |
| **Thu 12** (about 3 h) | Freeze at noon. Assemble tables and figures. Poster layout. | |
| **Fri 13** (about 3 h) | LaTeX assembly, proofread, PDF checks (fonts, page limit, references). | Complete draft. **G4** |
| **Sat 14** (about 1 h) | Final PDF. | |

**Watch for.** Headline MASE is per-station macro, with pooled beside it. Bootstrap over stations, not rows. A station with a near-zero denominator is flagged, not dropped.

**Needs from others.** Predictions (Afzal, Tue night), Figure 1 (Shaurya, Wed), Figure 2 (Khushi, Wed).
**Others need from you.** Metrics and style by Sun, Overleaf by Sat.

---

## 4. Hand-offs

| From | To | What | By |
|---|---|---|---|
| Afzal | Shaurya, Khushi | `tickets_device.parquet`, repo access | Sat 7 |
| Shaurya | all | `data.py` | Sat 7 night |
| Shaurya | Afzal | `features.py` | Sun 8 noon |
| Abdul | Afzal, Khushi | `metrics.py`, figure style | Sun 8 |
| Afzal | Abdul | Best-model predictions | Tue 10 night |
| Shaurya, Khushi | Abdul | Figures 1 and 2 | Wed 11 |
| Afzal | all | `run_all.py` | Thu 12 |

**Critical path.** Shaurya's `data.py` and `features.py` unblock the models. If they slip, Afzal keeps building on synthetic data and Abdul tests metrics on a hand-made fixture.

---

## 5. Second pair of eyes (replaces formal reviews)

Before merging, one other person reads the file and runs its test. This is the minimum that still catches silent errors.

| File | Written by | Checked by |
|---|---|---|
| `data.py` | Shaurya | Afzal |
| `metrics.py` | Abdul | Khushi |
| `censor_benchmark.py` | Khushi | Abdul |

Others merge without review but must run their own test first.

---

## 6. Hours (estimates)

| | Sat 7 to Sun 8 | Mon 9 to Wed 11 | Thu 12 to Sat 14 | Total |
|---|---|---|---|---|
| Shaurya | 5.5 | 8 | 4.5 | 18 |
| Afzal | 6.5 | 9.5 | 6 | 22 |
| Khushi | 7 | 9.5 | 6 | 22.5 |
| Abdul | 5 | 8 | 7 | 20 |

Add about 2 hours each for calls and messages. If anyone is overloaded, use the backups below.

---

## 7. Backups and cut order

| If this slips | Backup |
|---|---|
| Data and features (Shaurya) | Afzal |
| Models (Afzal) | Shaurya takes `ridge`, `lasso`, `random_forest`; Afzal keeps Poisson GLM and LightGBM |
| Benchmark (Khushi) | Abdul takes the cap sweep; Khushi keeps the rest |
| Metrics, conformal (Abdul) | Khushi takes conformal; Afzal takes the figure style |

**Cut from the top first:** SVM and SVR, K-Means, the ablation, stratified error, `station_hetero`, Wilcoxon tests. **Never cut:** the E3 ladder with baselines and intervals, E4 with persistence, the E2 cap sweep, E5, Figure 1.

---

## 8. Report and poster writing

| Section | Writer | Reads it |
|---|---|---|
| Abstract | Abdul | Afzal |
| Introduction | Afzal | Khushi |
| Related work | Khushi (Ensign, Schnabel), Shaurya (Gao, Liu and Chen, Inam, Grinsztajn) | Abdul |
| Data and tasks | Shaurya | Afzal |
| Methods | Afzal (models), Khushi (benchmark), Abdul (metrics, conformal) | Shaurya |
| Results | Each owner for their experiments | Abdul checks consistency |
| Discussion and limitations | Khushi and Shaurya | All |
| Conclusion | Afzal | All |
| Contributions | Each person writes one line, Abdul assembles | All |

**Overleaf** is owned by Abdul, with one `.tex` file per section. Figures come only from `results/figures/*.pdf`. Sections that need no results (Introduction, Related work, Data, Methods) can be written from Mon 9, so do not wait for the freeze.

**Contributions paragraph (edit to match what really happened):**
> Shaurya built the data layer, leak-free features, the evening-gap evidence and the unsupervised study. Afzal built the model comparison and reproducibility pipeline, and provided the ParkSight hackathon baseline on which this work builds. Khushi designed and ran the censoring benchmark. Abdul built the metrics, conformal intervals, stratified error analysis and figures, and assembled the report. Report and poster were written jointly.

---

## 9. Meetings

| When | What | Length |
|---|---|---|
| **Sat 7** | Kick-off: confirm decisions, assign files, answer questions | 30 min |
| Every day | Written update in the group chat | 2 min |
| **Sun 8 night** | G1: baselines run end to end | 15 min |
| **Wed 11 night** | G2: look at v1 tables and Figure 2, decide any cuts | 30 min |
| **Fri 13 evening** | G4: read the full draft together | 45 min |

---

## 10. First hour today (Sat 7)

**Everyone.** Read `PARKSIGHT_EXPLAINED.md` and `IMPLEMENTATION_PLAN_V2.md` sections 0 to 5. Answer the five decisions in plan section 11. Install Python 3.12 and wait for Afzal's repo invitation.

**Afzal.** Tag, branch, invitations, folder skeleton, device file, decisions.
**Shaurya.** Start `data.py` and the fold test.
**Khushi.** Read `debias.py` and Ensign et al., then write the censor test.
**Abdul.** Overleaf project, first metric test, ask for the poster format.
