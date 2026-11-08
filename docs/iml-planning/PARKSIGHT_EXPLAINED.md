# ParkSight Explained: The Whole Project, From Zero

**Who this is for:** everyone on Group 22 (Afzal, Shaurya, Abdul, Khushi). You do not need to have seen the code. Read it top to bottom once. Every term is explained before it is used.

**Where the code lives:** https://github.com/Susan5504R/ParkSight
**Course:** Machine Learning project, deadlines 20 Oct (milestone) and 15 Nov 2026 (poster + final report).

---

## 0. The whole project in five sentences

Bengaluru police issue parking tickets, and we have 298,445 of them. A ticket only exists if an officer was standing there to write it, so the tickets show where officers were, not just where cars parked illegally. This matters because officers almost vanish in the evening, exactly when traffic is worst. ParkSight scores how much each place hurts traffic, corrects for the missing evening data, forecasts next week's workload, and shows it all in a web app. Our course project takes this system and turns it into a research study: how badly does the bias hurt a model, and how well can classical machine learning fix it?

---

## 1. The real-world problem

Cars parked on main roads, near junctions, markets and metro stations squeeze the road. The traffic behind them slows down. One badly parked car at a busy junction can cost far more time than ten cars on a quiet lane.

Enforcement today is **reactive**: officers patrol the usual routes and write tickets. Nobody has a map that says "these places cause the most congestion, so go there first". The data to build that map exists, but it has not been turned into one.

That is the gap ParkSight fills. It is **decision support**, which means it advises officers. It does not enforce anything automatically.

---

## 2. The data

The dataset is 298,445 anonymised parking violation records from the Bengaluru traffic police, covering 10 Nov 2023 to 8 Apr 2024 (151 days). It was provided for the Flipkart Gridlock 2.0 hackathon (Round 2, Theme 1), and the raw file is not redistributed.

Each row has:
- where it happened (latitude, longitude, a text address, a police station, sometimes a junction name),
- what vehicle it was (scooter, car, motorcycle, auto, and so on),
- what the violation was (tags like `PARKING IN A MAIN ROAD`, `DOUBLE PARKING`, `NO PARKING`),
- when it was recorded,
- which enforcement device recorded it (3,070 distinct devices).

The most important fact about this data: **it has no column for traffic speed, delay or congestion.** We can never measure congestion directly. Everything we say about congestion is inferred from the tickets. Keep this in mind, because it limits every claim we make.

Two housekeeping facts:
- The raw CSV is 109 MB and is not in the repo. The repo ships small precomputed files instead.
- The cleaned file in the repo drops the `device_id` column. To do any work that needs device information, you need the raw CSV from Afzal. Check the hackathon terms before sharing it.

---

## 3. How the pipeline flows

Before the details, here is the shape of the whole system:

```
Raw CSV  ->  Clean + place on a map grid  ->  Impact score (PCIS)
                                         ->  Hotspot zones + repeat offenders
         ->  De-biasing engine (fix the missing evening data)
         ->  Forecast (next 7 days per police station)
         ->  Small saved files (parquet)  ->  Streamlit web app (10 pages)
```

The heavy computing runs once, offline, and writes small files. The web app only reads those files, which is why it opens instantly. A Java comparison: it is like a batch job that fills a read-only cache, with a thin web layer on top.

---

## 4. Step 1: cleaning and putting tickets on a map grid

**Cleaning** (`parksight/etl/build_artifacts.py`). The main steps:
- Convert timestamps from UTC to IST (Indian Standard Time, 5 hours 30 minutes ahead). Without this, every time-of-day finding is wrong.
- Parse each ticket's violation tags and give it a severity.
- Drop columns that are 100% empty.

**The map grid: H3.** To say "this area is bad" we must cut the city into areas. We use **H3**, a system by Uber that tiles the whole planet with hexagons at several sizes. Each hexagon has a short text ID, so "which area is this ticket in" becomes a simple lookup.

Why hexagons and not squares? Every neighbour of a hexagon is the same distance from its centre. Squares have diagonal neighbours that are further away, which distorts distance-based analysis.

We use two sizes:
- **Resolution 8**, about 0.74 square km, for the city-wide heat layer (776 hexagons with tickets).
- **Resolution 9**, about 0.1 square km, for street-level hotspots (2,534 hexagons with tickets).

---

## 5. Step 2: the impact score (PCIS)

**The problem.** Ticket count alone is a poor measure of harm. 100 tickets for `NO PARKING` on a side street hurt traffic less than 20 tickets for `PARKING IN A MAIN ROAD`. And we cannot measure delay.

**The solution.** Build an explainable score from ingredients that we know drive congestion. We call it the **Parking Congestion Impact Score (PCIS)**:

```
PCIS = 100 x (0.30 V + 0.20 S + 0.20 L + 0.20 P + 0.10 T)
```

| Letter | Name | What it captures | How it is built |
|---|---|---|---|
| V | Volume | how many tickets | log of the count, so 1,000 tickets is not 10 times worse than 100 |
| S | Severity | how badly it blocks the road | main road 1.0, near a crossing 0.95, double parking 0.90, footpath 0.85, ... no parking 0.40 |
| L | Location | how critical the place is | share of tickets at junctions, plus keywords like market, metro, hospital |
| P | Peak overlap | does it happen at rush hour | tickets at 08:00-11:00 and 17:00-21:00 count more |
| T | Trend | is it getting worse | only worsening trends count; improving areas get 0 |

Each ingredient is rescaled to run from 0 to 1, combined with the weights above, and then stretched so the final scores run from 0 to 100. Scores are split into Low (up to 33), Medium (34 to 66) and High (67 to 100).

**Two honest things about PCIS.**
1. The weights (0.30, 0.20, ...) are **chosen by us, not learned**. There is no ground truth to learn them from. The app lets you drag the weights to see how rankings change.
2. There is a sanity check. The repo contains 12 Bengaluru junctions ranked by local knowledge (6 famous bottlenecks like KR Market, 6 quiet ones). PCIS ranks them with Spearman correlation 0.92, where 1.0 would be a perfect match. That is encouraging, but it is only 12 points. The famous bottlenecks are also the most ticketed places, so the check may mostly confirm that volume matters.

PCIS is computed for three kinds of places: hexagons, junctions (168) and police stations (54). Top station: Upparpet (PCIS 100, 34,468 tickets). 234 hexagons land in the High tier.

---

## 6. Step 3: hotspot zones and repeat offenders

**Zones with DBSCAN.** Hexagons are good for maps but awkward for humans ("hexagon 8961892..."). **DBSCAN** is a clustering algorithm: it groups points that are close to each other and calls isolated points noise. We run it on the high-impact hexagons, with a radius of about 400 metres and at least 3 hexagons per cluster. The result is 140 named zones, such as "BTP179 - 9th Block, 28th Main, Jayanagar" (the top zone, PCIS 85).

**Repeat offenders.** Any vehicle ticketed more than once is flagged. Result: about 15% of vehicles account for about 34% of all violations. That is a small, targetable group, and it is a useful policy insight.

---

## 7. Step 4: the key idea, tickets are not violations

This section is the heart of the project, and it is also the heart of our research.

### 7.1 The problem

A ticket is only created if an officer with a device was present. So:

> **observed tickets = real violations x chance that an officer was there to record them**

That chance is called the **exposure** (or recording probability). In plain words: how likely was a violation, at this hour, to end up in our data?

The data shows exposure is wildly uneven across the day. Counting distinct devices that wrote at least one ticket in each hour:

| Hour (IST) | Active devices | Tickets |
|---|---|---|
| 11:00 | 1,773 (the peak) | 32,176 |
| 15:00 | 271 | 1,224 |
| 19:00 | 12 | 27 |

Only 0.2% of all tickets fall between 17:00 and 21:00, which is the evening rush hour.

### 7.2 Why a naive model fails

Imagine training any model on raw ticket counts. It would learn that 7 PM is quiet, because the data says so. It would then recommend sending officers elsewhere at 7 PM. Fewer officers means even fewer tickets. The data gets emptier, and the model gets more sure. This is a **feedback loop**, and a research paper by Ensign et al. (2018) proves it happens for any system trained on officer-discovered events.

### 7.3 The two corrections in the repo (`parksight/models/debias.py`)

**Correction 1: inverse-propensity weighting (IPW).** Think of a survey where only 1 in 10 people in one group answered. To estimate the group, you count each answer 10 times. Same idea: a ticket recorded when exposure was low is counted as more than one ticket. The weight is 1 divided by exposure, with the exposure floored at 0.05 and the weight capped at 12. Real numbers from our data:

```
hour 11: tickets= 32176 exposure=1.0000 uncapped_weight=   1.0 capped_weight= 1.0 weighted_tickets=    32176
hour 15: tickets=  1224 exposure=0.1528 uncapped_weight=   6.5 capped_weight= 6.5 weighted_tickets=     8010
hour 19: tickets=    27 exposure=0.0068 uncapped_weight= 147.1 capped_weight=12.0 weighted_tickets=      324
```

Look at the last line. The ideal weight at 19:00 would be 147, but the code stops at 12. Even after correction, 19:00 looks about 100 times smaller than 11:00. We come back to this in section 12, because it is one of our best research questions.

**Correction 2: an outside prior.** We build an expected congestion level that does not use ticket timing at all: a road-type weight (from address keywords such as "main road", "junction", "metro") multiplied by a standard textbook rush-hour curve (low at night, peaks near 09:00 and 18:00 to 19:00). Since it never looks at tickets, the missing-ticket loop cannot erase it.

**The output: the Blind-Spot score.** For each place, the engine compares expected evening congestion with observed evening ticket activity. A big positive gap means "high expected congestion, very little data". The final **blindspot_risk** (0 to 100) multiplies the corrected demand by that gap. It pushes the recommendations toward places where data is missing but congestion is likely. This is the default sort on the Prioritize page.

---

## 8. Step 5: forecasting next week's tickets

### 8.1 What is predicted, and what is not

The forecast predicts **how many tickets each police station will issue on each of the next 7 days** (54 stations). It does **not** predict violations. The target is the observed count, so it carries the bias from section 7. The repo's own documents say this: the forecast is for **workload and staffing**, and the blind-spot score is for **deployment**.

Why station by day? Because that is where the signal is. If you cut the data into hexagon-by-hour cells, 99% are empty. If you use station-by-day cells, 94% have tickets.

### 8.2 The model: LightGBM

A **decision tree** is a chain of yes/no questions ending in a number, like nested `if` statements in Java. **Gradient boosting** builds many small trees in sequence, each one trained to fix the mistakes of the trees before it. **LightGBM** is a fast library for this. It is classical machine learning, not deep learning.

The model sees 15 features, all built from a station's own history:
- **lag features:** the count 1, 7 and 14 days ago,
- **rolling averages:** the mean of the last 3, 7, 14 and 28 days,
- the mean for the same weekday in the past,
- calendar information: weekday, weekend, month, day of year, Indian holidays, days to the next holiday, and the station itself.

The 7-day forecast is **recursive**: it predicts tomorrow, treats that as data, and uses it to predict the day after. Errors can pile up on later days.

### 8.3 Judging it honestly

A forecast is meaningless without something to beat. The repo uses:
- a **baseline**: predict each station's average for that weekday,
- **rolling-origin cross-validation**: train on the past, test on the following block, slide forward, repeat 4 times. Never test on days that come before the training days, or the model peeks at the future,
- **MASE**: the model's error divided by the error of a naive guess ("same weekday last week"). Below 1 means you beat the naive guess.

A tiny example you can run (numbers invented):

```python
import numpy as np
actual = np.array([10, 12, 9, 14])
model  = np.array([11, 11, 10, 12])
naive  = np.array([8, 15, 12, 10])
mase = np.abs(actual - model).sum() / np.abs(actual - naive).sum()
print("model MAE:", np.abs(actual - model).mean(), "| naive MAE:", np.abs(actual - naive).mean(), "| MASE:", round(mase, 3))
```
```
model MAE: 1.25 | naive MAE: 3.0 | MASE: 0.417
```

Real results (`parksight/models/metrics.json`):

| Measure | Value |
|---|---|
| Model error (MAE) | 15.9 tickets per station per day |
| Baseline error | 18.5 |
| Improvement | 14.2% |
| MASE | 0.73 |
| Plain 7-day moving average | 17.2 |

The simple moving average is not far behind. The first version of LightGBM actually lost to it, so the final model is a **blend** of LightGBM and the moving average. Two classical alternatives were also tested and rejected, and the repo says why (an exponential smoothing model was unstable on sparse counts).

A flag for the report: the model comparison table in `metrics.json` shows plain LightGBM at 15.63, slightly better than the chosen blend at 15.90. Selection used a score that also penalises variance, so this may be fine, but we should reproduce these numbers ourselves before putting them in a paper.

### 8.4 Uncertainty: conformal prediction

"52 tickets" is less useful than "52, probably between 40 and 65". **Conformal prediction** gives such a range without assuming any distribution:
1. Hold out a calibration set the model never saw.
2. Measure its errors.
3. Find the error that 80% of them fall below.
4. Put that width around every future forecast.

Busy stations make big errors and quiet stations make tiny ones, so a single width fits neither. **Mondrian conformal** uses a separate width per volume group. A toy version:

```python
alpha = 0.20
def q(scores):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))

low  = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
high = np.array([4, 9, 6, 12, 3, 8, 15, 5, 7, 10])
print("one global band :", q(np.concatenate([low, high])))
print("low-tier band   :", q(low))
print("high-tier band  :", q(high))
```
```
one global band : 9.149999999999999
low-tier band   : 1.0
high-tier band  : 12.299999999999999
```

In the repo, the widths are about 0.35 for low-volume stations and 5.49 for high-volume ones. Coverage rose from 74.7% to 80.4% against an 80% target.

---

## 9. The app: what a user can do

The app uses **Streamlit**, a Python library where each page is a script that re-runs from top to bottom on every click. It has 10 pages:

1. **Home (Executive Dashboard):** headline numbers, the evening blind spot chart, vehicle mix, monthly trend.
2. **AI Hotspot Map:** hexagons coloured by PCIS, a time filter, DBSCAN zones, drill-down into one junction, a month-by-month time-lapse.
3. **Impact (PCIS):** explains the formula, lets you drag the five weights, and shows the expert-ranking check.
4. **Forecast:** the 7-day station forecast with its confidence band against the baseline, tomorrow or the weekly total.
5. **Prioritize:** a ranked deployment plan. It can sort by PCIS, by enforcement gap, or by blind-spot score. It gives a reason for each place and downloads a one-page **PDF briefing** for an officer.
6. **Simulator:** pick stations and an enforcement increase and see estimated violations prevented. This uses an assumed "deterrence elasticity" of 0.4. That number is an assumption, not learned from data.
7. **AI Copilot:** ask questions in plain English. A rule-based router maps each question to one of a fixed list of functions (like a Java `switch` over an enum), so answers always come from the data. An optional Claude or Gemini key improves question understanding.
8. **Offenders:** the repeat-vehicle watchlist.
9. **About:** the methodology and limitations inside the app.
10. **Data Refresh:** upload a new CSV and the whole pipeline reruns in the app.

**To run it:**
```bash
pip install -r requirements.txt
streamlit run parksight/app/Home.py
```
It works out of the box because the processed files are committed. You only need the raw CSV to rebuild them. A headless test (`python scripts/smoke_test_app.py`) checks that all pages load.

---

## 10. Where the code lives

```
parksight/
  config.py                 paths, PCIS weights, severity table, peak windows, IPW cap
  scoring.py                PCIS recompute, tiers, enforcement gap (shared by app + reports)
  etl/build_artifacts.py    clean, IST, H3, PCIS, zones, offenders, summaries
  models/debias.py          exposure, inverse-propensity weights, blind-spot score
  models/train_forecast.py  LightGBM forecast, cross-validation, conformal intervals
  copilot/engine.py         the question router
  reports/                  PDF briefing, figures, pitch deck
  app/Home.py + pages/      the Streamlit app
  data/processed/*.parquet  committed result tables (about 15 MB)
scripts/                    smoke test, model experiments, full rebuild
DECISIONS.md                why each design choice was made
```

`DECISIONS.md` is worth reading: it records the experiments that did not work (for example, the first LightGBM lost to a moving average) and why.

---

## 11. Honest limitations (say these out loud in the report)

1. **No congestion ground truth.** PCIS is a designed proxy, not measured delay.
2. **Five months, one city.** No yearly seasonality, no claim that this works elsewhere.
3. **Tickets reflect enforcement, not violations.** The de-biasing is our answer, but it is untested (see below).
4. **The weight cap of 12 binds** for every hour from 16:00 to 23:00, so the evening is under-corrected. The README's "up to 2.8x" is an average across a station's tickets, not the hourly correction.
5. **The exposure measure is circular.** It counts distinct devices seen in the tickets themselves, so fewer tickets automatically means fewer devices. It is also one citywide curve, and it assumes everything is recorded at 11:00.
6. **Timestamps are an assumption.** Every timestamp has seconds equal to 46, so the field was transformed. About 34% of tickets sit between 00:00 and 06:00 IST, which is unusual for patrol enforcement. The "evening blind spot" depends on the UTC to IST step being right. We found no evidence it is wrong, but we should check it with a day of work.

---

## 12. From hackathon project to course research project

### 12.1 What we are doing

The course wants a research project: identify a problem, use papers, make a contribution. The hackathon built a working product. We keep the working parts as the foundation and add the research layer on top.

**Our title:** De-biasing Enforcement Data for Parking Congestion Prediction: A Classical Machine Learning Study on Bengaluru Violation Records.

**Base papers:**
- **Gao et al. (2019), Annals of GIS.** Predicted parking violation counts and legality from NYC tickets with classical models. Random Forest won. This is our application template.
- **Ensign et al. (2018), FAT\*.** Proved that training on officer-discovered events creates runaway feedback loops. This is our theory for why correction is needed.

Our contribution in one sentence: Gao's framework has not been run with an explicit bias correction, and not on an Indian city.

### 12.2 What is ours and what we must cite

| Ours | Cite, not ours |
|---|---|
| ETL and H3 pipeline on this dataset | Inverse-propensity weighting (Horvitz and Thompson 1952; Schnabel et al. 2016) |
| PCIS, Blind-Spot Divergence, Enforcement-Gap | The feedback-loop argument (Ensign et al. 2018) |
| Device-based exposure measure and its use here | Conformal prediction, Mondrian version (Vovk et al.) |
| The censoring benchmark (below) | LightGBM, Optuna, DBSCAN, H3 |
| Findings on Bengaluru data | The method we replicate (Gao et al. 2019) |

Also: the code predates the course. We will tag the current commit `hackathon-baseline`, work on a new branch, and put new code under `parksight/research/`, so credit for new work is clear. The report must say this openly.

### 12.3 The research questions

- **RQ1 (core):** how much do naive models under-predict hours with low exposure, and how well does IPW recover the truth as exposure shrinks? Includes the cost of the weight cap.
- **RQ2 (core):** does the device-based exposure measure track real exposure, and does an exposure model that varies by station and weekday beat the single hourly curve?
- **RQ3 (core):** which classical model is best for station-by-day counts and for weekly hotspot classification?
- **RQ4 (stretch):** does weighted conformal prediction keep its coverage under censoring where the plain version does not?
- **RQ5 (stretch):** do de-biased rankings change who gets officers, and are rankings stable when the PCIS weights are perturbed?

### 12.4 The key experiment: a censoring benchmark

We have no ground truth for the evening, so we manufacture one:
1. Take hours with strong, stable coverage (roughly 02:00 to 12:00) as the pretend truth.
2. Delete tickets using a known exposure curve shaped like the real evening dip. Delete at ticket level so device counts fall too.
3. Re-estimate exposure with the repo's device-count method and compare it to the known curve (tests the circularity problem).
4. Train with and without IPW on the thinned data. Sweep the censoring severity, the cap (2, 5, 12, 50, none), and wrong exposure assumptions.
5. Measure how close each approach gets to the withheld truth, with bootstrap confidence intervals.

This turns every de-biasing claim into a number with a known failure mode.

### 12.5 The model ladder (what is new and does not exist yet)

The repo has only LightGBM, a moving average, the baseline, and a failed exponential smoothing test. For the course we need a proper comparison:

seasonal-naive baseline, then Poisson regression (the correct simple model for count data), Ridge and Lasso, a single decision tree, Random Forest, LightGBM, SVR, and a small neural network included on purpose so we can show on our own data that trees win (as Grinsztajn et al. 2022 report). We also add a second task, **hotspot classification** ("will this hexagon be in the top 10% next week?"), with logistic regression, SVM, Random Forest and boosting, judged by precision-recall AUC, F1 and Brier score. And an unsupervised part: PCA on hourly profiles and K-Means archetypes.

One grid decision: the proposal said "hexagon by hour", but that grid is 99% empty. We will run count regression at station by day and hotspot classification at hexagon (resolution 8) by week, and state this change in the Method section.

### 12.6 Who does what

| Person | Track |
|---|---|
| Shaurya | Data pipeline, features, exploratory analysis, time-semantics check |
| Afzal | Model ladder, tuning, cross-validation, repo integration |
| Khushi | De-biasing, exposure models, the censoring benchmark |
| Abdul | Evaluation, error analysis, conformal intervals, figures |

The report and poster are written jointly, in the ICML or NeurIPS LaTeX style (4 to 6 pages). One person submits on Google Classroom and must tag the other three.

### 12.7 What to leave out of the report

The Copilot, Simulator, PDF briefing and pitch deck are product features, not machine learning. Keep them to one line or an appendix. Lead with the tables, the ablations and the benchmark.

### 12.8 Time left

As of 29 Oct, 17 days remain until 15 Nov. The dates in `PHASE2_RESEARCH_PLAN.md` started on 22 Oct, so the first checkpoints have already passed. Re-confirm the schedule as a team. Also confirm whether the 20 Oct milestone was submitted.

---

## 13. Check yourself

1. Why would a model trained on raw ticket counts send fewer officers out in the evening?
2. At 19:00 the ideal IPW weight is 147 but the code uses 12. What does that do to the evening estimate, and why might a lower cap still be a good idea?
3. Why is the forecast the wrong tool for choosing evening deployments, even if its MASE is 0.73?
4. Why can a single conformal band fail when stations differ a lot in volume?
5. What does the censoring benchmark let us measure that the original ParkSight never could?

**Answers.**
1. The data says evening is quiet because officers are absent. The model learns that, recommends fewer officers, and the data gets emptier. This is the feedback loop.
2. The cap stops a few tickets from being multiplied by a huge number, which would make the estimate swing wildly (high variance). But it leaves the evening under-corrected (some bias). Choosing the cap is a bias versus variance trade-off, and we will measure it.
3. It predicts observed tickets, which already carry the enforcement bias. It is good for staffing workload, not for finding where unrecorded violations are.
4. Busy stations have big errors and quiet ones have tiny errors. One width is too wide for quiet stations and too narrow for busy ones. Per-group widths fix this.
5. The true answer. By deleting tickets in a known way, we know what the data should have looked like and can measure how well each correction recovers it.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| PCIS | Parking Congestion Impact Score, a 0 to 100 explainable score built from volume, severity, location, peak overlap and trend. |
| H3 | Uber's system that tiles the Earth with hexagons of several sizes. |
| DBSCAN | A clustering algorithm that groups close points and treats isolated ones as noise. |
| Exposure | The chance that a violation actually gets recorded as a ticket at a given time. |
| IPW | Inverse-propensity weighting: counting each recorded event as 1 divided by its recording chance. |
| Feedback loop | A model's output changes the data it is later trained on, reinforcing its own mistakes. |
| Blind-spot score | Expected congestion minus observed evening activity, highlighting places with high stakes and little data. |
| Decision tree | A chain of yes/no questions on features that ends in a predicted number. |
| Gradient boosting | Many small trees built in sequence, each correcting the errors of the earlier ones. |
| LightGBM | A fast gradient boosting library; classical machine learning, not deep learning. |
| Lag feature | A value from an earlier time step, such as the count 7 days ago. |
| Rolling-origin cross-validation | Train on the past, test on the next block, slide forward, so the model never sees the future. |
| MASE | Model error divided by a naive same-weekday-last-week error; below 1 beats the naive guess. |
| Conformal prediction | A way to turn held-out errors into prediction intervals without distribution assumptions. |
| Mondrian conformal | Conformal prediction with a separate interval width per group. |
| Streamlit | A Python library for web apps where each page is a script re-run on every interaction. |
| Parquet | A compact column-based file format for tables. |
| Semi-synthetic benchmark | Real data deliberately damaged in a known way, so we know the truth and can score a correction. |

---

## 15. Reading order for teammates

1. This file.
2. `PHASE1_PROPOSAL_PLAN.md` (paper shortlist, metrics, model choices) and `PHASE2_RESEARCH_PLAN.md` (the plan and risks).
3. `DECISIONS.md` and `README.md` in the ParkSight repo.
4. Run the app once (`streamlit run parksight/app/Home.py`) and click through Forecast, Prioritize and Impact.
5. Read `parksight/models/debias.py` (170 lines) and `parksight/models/train_forecast.py`. These two files are the technical core.
