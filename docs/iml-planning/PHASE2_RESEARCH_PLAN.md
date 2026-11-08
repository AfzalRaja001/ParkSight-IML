# ParkSight for IML - Research Plan After the Proposal

**Written:** 22 Oct 2026. **Hard deadline:** Poster + Final Report, 15 Nov 2026 (24 days).
**Heads up:** the guidelines list the Milestone (max 3 pages) as due 20 Oct. If it is not submitted yet, ask the instructor today. If it is, the plan below is the "next steps" section of it.

---

## 1. Where we stand (checked against the repo, not assumed)

I cloned ParkSight and ran numbers on the committed `violations_clean.parquet` (298,445 rows, 151 days, 54 stations, 2,534 H3-r9 cells). Four findings change the plan.

**Finding 1 - The proposal's grid does not match the repo.** The proposal says "H3 hexagon by hour cells". The repo forecasts police station x day (D8/D17 in DECISIONS.md), for a good reason:

| Grid | Share of cells with at least one ticket |
|---|---|
| H3-r9 x hour | 0.9% (82k of 9.2M) |
| H3-r8 x hour | 2.3% |
| H3-r8 x day | 22.5% |
| Station x hour | 23.0% |
| Station x day | 93.9% |

A count regression on a 99%-empty grid is mostly a zero-inflation problem, and Poisson GLM, Ridge, and SVR will all look bad for reasons unrelated to our contribution. **Decision needed:** run count regression at station x day (and station x 4-hour block), and run hotspot classification at hex-r8 x week. The proposal said we would "compare spatial scales", so this is consistent with it. State the change in the Method section.

**Finding 2 - The evening correction is capped, not estimated.** From `enforcement_exposure.parquet`: exposure at 19:00 is 0.0068, so the inverse weight would be 147. The code caps it at 12, and the cap binds for every hour from 16:00 to 23:00. Only 596 tickets (0.20%) fall in 17:00-21:00. The README's "up to 2.8x" is a station-level average uplift, not the hour-level correction. So with the current code, IPW cannot recover evening demand; it is information-starved. This is not a flaw to hide. It is the research question (see RQ1).

**Finding 3 - The exposure proxy is circular.** Exposure is "distinct active devices per hour", computed from the ticket rows themselves (`estimate_exposure` in `parksight/models/debias.py`). Fewer tickets mechanically means fewer distinct devices seen. It is also normalised so that 11:00 has exposure 1, which assumes full detection at the daytime peak, and it is one curve for the whole city (no spatial variation). We need to test whether the proxy tracks real exposure.

**Finding 4 - Two premises are unvalidated.**
- PCIS weights and the congestion prior (hand-set road keywords x a synthetic rush-hour curve) have no sensor ground truth. The repo does have a face-validity check: 12 expert-ranked Bengaluru junctions (`PCIS_GROUND_TRUTH` in `config.py`), and PCIS ranks them with Spearman rho = 0.92. That is encouraging but small, and the known bottlenecks are also the most heavily ticketed junctions, so it may mostly be confirming volume. Treat PCIS as a designed proxy and test sensitivity, do not claim accuracy.
- 100,444 tickets (33.7%) are stamped 00:00-06:00 IST, which is unusual for patrol enforcement, and every timestamp has seconds = 46, so the field was clearly transformed. The "evening collapse" depends on the UTC to IST step (D7). Spend half a day checking time semantics (shift structure per device, weekday pattern). If the offset is wrong, the blind-spot window moves.

I did not find evidence that the evening gap is an artefact. I only could not rule it out from the processed file.

---

## 2. What is ours and what we must cite

You said everything in ParkSight is our contribution. Mostly true for the system, but graders check this, so be precise.

| Ours (claim it) | Not ours (build on, cite) |
|---|---|
| ETL and H3 pipeline on this dataset | Inverse-propensity / Horvitz-Thompson estimator (Horvitz & Thompson 1952; Schnabel et al. 2016) |
| PCIS, Blind-Spot Divergence, Enforcement-Gap scores | The feedback-loop argument (Ensign et al. 2018) |
| Device-activity exposure proxy and its application here | Conformal prediction; Mondrian variant (Vovk et al.) |
| Semi-synthetic censoring benchmark (new, below) | LightGBM, Optuna, DBSCAN, H3 |
| Evaluation findings on Bengaluru data | The replication target (Gao et al. 2019) |

Two practical points:
1. **Disclose the origin.** The code began as a hackathon project built before this course. Say so in the report, and make the diff auditable: `git tag hackathon-baseline` on the current commit, then work on a branch `iml-research`, with all new code under `parksight/research/`. Course credit rests on what is new after the tag.
2. **Data provenance.** The raw CSV (with `device_id`) is not in the repo and the clean parquet drops `device_id`. Teammates cannot do exposure work without the raw file. Confirm you are allowed to share it with the team, then share it.

---

## 3. Research framing

**Problem.** Observed ticket count = true violations x probability that an officer was present to record them. Predicting observed counts is not predicting violations, and deployment decisions need the latter.

**Research questions**

- **RQ1 (core).** How much does naive training underestimate censored periods, and how well does inverse-propensity weighting recover the truth as the exposure gets small? Includes the bias-variance tradeoff of the weight cap M.
- **RQ2 (core).** Does the distinct-device exposure proxy recover known exposure, and does a spatio-temporal exposure model (station x hour x weekday) beat the hour-only curve?
- **RQ3 (core).** Which classical learner is best for station x day counts and for weekly hotspot classification? Do trees lead, as Gao et al. and Grinsztajn et al. predict?
- **RQ4 (stretch).** Does weighted conformal prediction (Tibshirani et al. 2019) keep coverage under censoring where plain Mondrian conformal does not?
- **RQ5 (stretch).** How much do de-biased rankings change deployment priorities (Kendall tau, top-k overlap) compared with naive ones, and are the rankings stable under PCIS weight perturbation?

**Hypotheses.** H1: naive models under-predict censored hours by an amount that grows as exposure falls. H2: IPW reduces this bias but variance explodes below roughly e < 0.05, so an intermediate cap wins. H3: the circular proxy drifts away from the known exposure when tickets are sparse, because distinct devices seen depends on ticket volume. H4: tree ensembles lead on accuracy, Poisson GLM wins on interpretability.

---

## 4. The experiment that makes this a research project

**Semi-synthetic censoring benchmark.** We have no ground truth for evening violations, so we manufacture one.

1. Take hours where coverage is high and stable (exposure above 0.75, about 02:00-12:00) as pseudo-truth, `y_true`.
2. Thin the tickets with a known exposure curve `e*(station, hour)` shaped like the real evening dip. Thin at ticket level, so each device's activity falls too.
3. Re-estimate exposure from the thinned data using the repo's own distinct-device proxy. Compare it with the known `e*`. This tests Finding 3 directly.
4. Train with and without IPW on the thinned data, sweeping: censoring severity, cap M in {2, 5, 12, 50, none}, and exposure misspecification (correct, hour-only, wrong shape).
5. Report recovery error against `y_true`, with bootstrap confidence intervals.

This gives every claim about de-biasing a number and a failure mode.

---

## 5. Work packages

| WP | What | Owner | Done by |
|---|---|---|---|
| 0 | Tag baseline, branch, share raw CSV, validate time semantics, fix grains (Finding 1) | Afzal + Shaurya | 24 Oct |
| 1 | Feature table (station x day, hex-r8 x week), time-blocked CV harness, EDA notebook | Shaurya | 29 Oct |
| 2 | Model ladder: seasonal-naive, Poisson GLM, Ridge/Lasso, tree, RF, LightGBM, SVR, small MLP; hotspot classifiers | Afzal | 3 Nov |
| 3 | Semi-synthetic benchmark, cap sweep, proxy test, spatio-temporal exposure model (RQ1, RQ2) | Khushi | 5 Nov |
| 4 | Metrics, error analysis (hour, road class, vehicle), conformal coverage, PCA/K-Means, figures | Abdul | 6 Nov |
| 5 | Stretch: weighted conformal (RQ4), ranking stability and PCIS sensitivity (RQ5) | Abdul + Khushi | 8 Nov, only if WP2 and WP3 are done |
| 6 | Freeze experiments, `run_all.py` reproduces every table | all | 9 Nov |
| 7 | Report (ICML template on Overleaf, set up now), poster | all | 13 Nov draft, 15 Nov final |

**Priority if time slips:** P0 = WP0-4. P1 = WP5. Cut the Copilot, Simulator, PDF briefing, and PPTX from the academic deliverables entirely.

**Rules:** time-blocked splits only; fixed seeds; one results CSV per experiment; the report reads numbers from those CSVs; absolute paths in configs.

**Why four people can work in parallel:** the ETL, PCIS, LightGBM, rolling CV, and Mondrian conformal already exist. Teammates build the research layer on top of them, they do not rebuild them.

---

## 6. Final report map (4-6 pages)

Abstract. Introduction: censored enforcement data, RQs, contribution list. Related work: Gao 2019, Ensign 2018, Liu & Chen 2025, Inam 2022, Schnabel 2016, Grinsztajn 2022. Methodology: data, grids, exposure model, IPW, ladder, conformal. Experiments: ladder table, ablations, benchmark curves. Analysis: where each method fails, the cap tradeoff, proxy bias, ranking stability. Limitations: five months, one city, no ground-truth congestion, timestamp assumptions, hand-set PCIS. Contributions per member, plus a sentence on the hackathon baseline.

---

## 7. Open questions for the team

1. Milestone: submitted, late, or extended?
2. Are we allowed to share the raw hackathon CSV with the team?
3. Agree to the grid change (station x day, hex-r8 x week) and to amend the Method wording?
4. Does everyone accept that RQ1-RQ3 are the committed scope and RQ4-RQ5 are stretch?
5. Who owns Overleaf and the final submission on Google Classroom (one submitter, tag the other three)?
