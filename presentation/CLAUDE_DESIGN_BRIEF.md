# Presentation brief for Claude Design (Group 22)

**What this is.** Everything you need to build the deck in Claude Design: a step-by-step workflow, a paste-ready master prompt, the slide-by-slide plan, who speaks when, and Q&A prep. **No slides have been made.** The design is left to Claude Design.

**Course rules (from the instructor):** 10 to 12 slides maximum, a mandatory **10-minute** slot, and these slides: Title (1), Introduction (1), Related Work / Base paper (1), Research Problem and Motivation (1), Proposed Methodology with diagram and training curves (2), Experimental Setup (1), Results and Discussion (2), Conclusion and Future Work (1), References (1). That is 11 slides. We use a 12th for team contributions and Q&A.

---

## 1. Workflow

1. **Open Claude Design** and start a new presentation. (I have not used its interface in this session, so button names may differ from what is written here. The steps are the same either way.)
2. **Upload** the folder `presentation/attach/` from the repo: 12 CSV files (one per chart) and 7 report figures. Also upload this file. Do **not** upload `report/ieee/main.pdf`, because it has not been recompiled since the last report fixes.
3. **Paste the master prompt** from section 2.
4. **Let it draft all 12 slides**, then fix them **one slide at a time** with the follow-up prompts in section 5. Do not ask for a full redo of the deck, because every redo can change numbers or styles you already liked.
5. **Check every number** against `presentation/attach/data/key_numbers.csv`. Design tools sometimes round or reword numbers. This takes 10 minutes and is the most important step.
6. **Export** to PowerPoint and also to PDF as a backup. Open the PowerPoint file and check fonts, that charts are editable, and that nothing overflows.
7. **Rehearse with a timer** (section 4). Target 9:15 spoken, so Q&A and slips fit inside 10:00.

**Where each chart's data lives:** every chart below names its CSV in `presentation/attach/data/`. Ask Claude Design to draw native charts from those files (not screenshots), so you can edit colours and labels.

---

## 2. Master prompt (paste this)

```text
Create a 12-slide presentation for a university Machine Learning course project, to be presented by 4 people in a strict 10-minute slot. Audience: the instructor and classmates who know basic ML but not this topic. Goal: clear, good-looking, and presentable, with MANY charts and little text.

PROJECT: "ParkSight: Forecasting Parking Violations from Enforcement Data That Is Censored by Where Officers Were". Bengaluru traffic-police parking tickets (298,445 tickets, 54 police stations, 151 days). Core idea: a ticket exists only where an officer was, and officers almost vanish in the evening rush, so ticket counts say little about the evening. We (1) show the evening gap is real, (2) test on a semi-synthetic benchmark with a known truth how much this censoring hurts and whether inverse-propensity weighting recovers the truth, and (3) compare classical ML models for next-day station counts and next-week hotspots, with honest uncertainty intervals.

SLIDES, in this exact order (action titles that state the takeaway, not topic labels):
1 Title | 2 Introduction | 3 Related Work / Base paper | 4 Research Problem and Motivation | 5 Methodology 1 (pipeline diagram + benchmark design) | 6 Methodology 2 (models, tuning, training curves) | 7 Experimental Setup | 8 Results 1 (censoring benchmark) | 9 Results 2 (forecasting, hotspots, uncertainty) | 10 Conclusion and Future Work | 11 References | 12 Team contributions and Q&A.
The per-slide content, chart-to-file mapping and speaker for each slide is in the uploaded file CLAUDE_DESIGN_BRIEF.md (section 3). Follow it.

DATA: draw every chart as a NATIVE, editable chart from the uploaded CSV files (data/c01 to c11). Use numbers exactly as in the files and in key_numbers.csv; never round differently or invent numbers. Show confidence intervals where the CSV has them. Use the uploaded PDF figures only for the methodology diagram (F0_method.pdf) if you cannot redraw it natively.

DESIGN: modern, clean, high contrast for a lecture hall. Light content slides with a dark title and closing slide. One accent colour family, colour-blind safe (Okabe-Ito: #0072B2, #E69F00, #009E73, #D55E00, #CC79A7, #56B4E9). Body text at least 22 pt, chart labels at least 16 pt, at most 3 short bullets per slide, no paragraphs. Every chart: a headline takeaway, labelled axes, a small source note (the CSV name), and the one number worth remembering highlighted. Slide number and the speaker's first name in the footer. Consistent colours: the same method or model has the same colour on every slide.

SPEAKER NOTES: for every slide write a spoken script of the time given in the brief, in plain language, ending with a one-sentence handoff to the next speaker. Total spoken time must be at most 9 minutes 15 seconds.

HONESTY RULES: show uncertainty, do not claim we recovered the real evening (we cannot, there is no ground truth), and keep the limitations on the conclusion slide.
```

---

## 3. Slide-by-slide plan

Speaker times add up to **9:15**. Put the speaker's first name in each footer.

| # | Slide (action title) | Speaker | Time |
|---|---|---|---|
| 1 | ParkSight: forecasting parking violations when tickets only exist where officers were | Abdul | 0:15 |
| 2 | Parking enforcement runs on tickets, and tickets are a biased record | Shaurya | 0:45 |
| 3 | Prior work predicts violations or fixes bias, but not both on enforcement data | Abdul | 0:45 |
| 4 | Officers vanish in the evening, so ticket counts say little about the evening | Shaurya | 0:55 |
| 5 | One clean data layer feeds a forecasting pipeline and a censoring benchmark | Khushi | 0:55 |
| 6 | Classical models only, tuned on the last block of training data, never on test | Afzal | 1:00 |
| 7 | Time-blocked evaluation: always train on the past, test on the future | Shaurya | 0:50 |
| 8 | Censoring hides most tickets, and the cap that looks safe hides the bias | Khushi | 1:20 |
| 9 | Simple models forecast well, and per-tier intervals make uncertainty honest | Afzal | 1:30 |
| 10 | Conclusion: where the signal is, where it is not, and what is next | Abdul | 0:45 |
| 11 | References | (shown, not spoken) | 0:00 |
| 12 | Team contributions and questions | Abdul | 0:15 |

### Slide 1, Title (Abdul)
- Title and subtitle above; "Group 22, Machine Learning course".
- Team: Shaurya Raj Mallick (IIT2024250), Afzal Raja (IIB2024031), Khushi Shorey (IIT2024244), Abdul Azeem (IIT2024243). **Confirm the spelling of Shaurya's surname first** (the proposal says "Malick", the report says "Mallick").
- Visual: a subtle hero graphic from `c01_evening_gap.csv` (devices collapsing across the day).

### Slide 2, Introduction (Shaurya)
- Cars parked on main roads and junctions slow traffic; enforcement is reactive: officers patrol usual routes.
- The data is a record of tickets, and a ticket records an officer as much as an offence.
- **Visual:** four big number tiles from `key_numbers.csv`: **298,445** tickets, **54** stations, **151** days, **3,070** devices.
- Bottom strip: the three research questions, as short chips: *Is the evening gap real? What does censoring cost, and can weighting fix it? Which classical models forecast best, with honest intervals?*

### Slide 3, Related Work / Base paper (Abdul)
A compact table, not paragraphs. Columns: **Paper | What it did | What we take from it**.
- **Gao et al. 2019 (base paper):** random forest on NYC parking tickets for counts and legality; **we replicate that model family on Bengaluru and add censoring.**
- **Ensign et al. 2018:** proves runaway feedback loops when training on officer-discovered events; **our motivation.**
- **Schnabel et al. 2016 and Horvitz-Thompson 1952:** inverse-propensity weighting for data seen with unequal probability; **our correction.**
- **Liu and Chen 2025:** deep model for parking-violation demand, notes ticket data is biased in space and time; **same premise, different method.**
- **Inam et al. 2022 and Grinsztajn et al. 2022:** simple classical models match or beat deep ones on tabular parking data; **why we use classical ML.**
- Footer chip: conformal prediction (Lei et al. 2018, Vovk et al. 2005) for the uncertainty intervals.

### Slide 4, Research Problem and Motivation (Shaurya)
- **Main chart:** two lines by hour of day, each indexed to its own peak (`c01_evening_gap.csv`, columns `tickets_pct_of_peak` and `devices_pct_of_peak`). Annotate: **1,773 devices at 11:00 to 12 at 19:00**; **32,580 tickets in the 10:00 hour to 27 at 19:00**.
- Callout: only **0.2%** of tickets fall between 17:00 and 21:00, the evening rush.
- One line: a model trained on counts learns where officers went, then sends them there again (the feedback loop).
- Small caveat tag: 34% of tickets are stamped 00:00 to 06:00, so the clock may carry an offset.

### Slide 5, Methodology 1: pipeline and benchmark (Khushi)
- **Visual:** the methodology diagram (`figures/F0_method.pdf`; redraw natively if possible). Top lane: tickets, two leak-free grids (station by day for counts, hexagon by week for hotspots), model ladder, metrics and conformal bands. Bottom lane: **pseudo-truth** (tickets 02:00 to 11:59, 240,352 of them), thin each ticket with a known exposure curve, then compare three estimators.
- Three estimator chips: **naive count**, **weighted with true exposure** (caps 2, 5, 12, 50, none), **weighted with estimated exposure** (the hackathon method).
- One line: weighting counts a kept ticket as 1 divided by its keep-probability.

### Slide 6, Methodology 2: models, tuning, training curves (Afzal)
- **Left:** model ladder as chips. *Counts:* last-week forecast, 7-day average, Poisson regression, ridge, lasso, random forest, LightGBM. *Hotspots:* persistence, logistic regression, SVM, random forest, LightGBM.
- Rule box: one hyperparameter per model, chosen on the last block of the training window, **never on test data**.
- **Right, training curves:** two small charts from `c09_training_curves.csv` (column `task` splits them): train versus validation loss per boosting round. Mark the best round (**223** for counts, **104** for hotspots). Message: training loss keeps falling, validation turns up, so models overfit slowly, not sharply.

### Slide 7, Experimental Setup (Shaurya)
- **Visual:** a timeline of the four expanding training windows with the 14-day test blocks (`c11_folds.csv`, rows where `task` is "Counts (days)"): test blocks start 13 Feb, 27 Feb, 12 Mar and 26 Mar 2024.
- Three small facts: station-day grid 54 by 151 (6.1% zeros); hexagon-week grid 16,275 cells with 10.1% hotspots; features use only earlier data (a unit test enforces this).
- Metric chips: **macro MASE**, RMSE, Poisson deviance; **PR-AUC**, F1, Brier; **80% conformal coverage**. Add a note: macro MASE averages per station so one busy station cannot decide the result.
- Footer: CPU laptop, no neural network; seeds 0 to 2 (benchmark: 20 seeds); the whole pipeline reproduces in about 11 minutes.

### Slide 8, Results 1: the censoring benchmark (Khushi)
Three charts (one slide, one story):
- **(a) Bias by method** at real severity: bar chart of `bias_pct` from `c02_benchmark_bias_variance.csv` (keep rows with `severity` 1.0). Naive **-77.8%**, cap 2 **-67.3%**, cap 5 **-54.9%**, cap 12 **-41.8%**, cap 50 **-14.5%**, no cap **+0.1%**, estimated exposure **-70.9%**.
- **(b) The trade-off:** bias squared and variance by cap, log scale (same file, `bias2` and `variance`). No cap removes the bias but variance is **17,817** against **1,416** at cap 12.
- **(c) Why the estimated exposure fails:** line chart from `c03_exposure_curves.csv`, true exposure against the four estimates by hour. At 09:00 the original estimate is **14.4 times** too high; counting device-days and censoring whole devices brings it to **1.3 times**.
- Takeaway banner: *the cap that looks safe (12) hides a -42% bias.*

### Slide 9, Results 2: forecasting and uncertainty (Afzal)
Three charts:
- **(a) Counts:** macro MASE per model with 95% intervals and a reference line at 1.0 (`c04_count_models.csv`). Lasso **0.697**, ridge 0.699, random forest 0.730, LightGBM 0.733, last-week forecast **0.967**.
- **(b) Hotspots:** grouped bars of PR-AUC, "all hexagons" versus "emerging only" (`c06_hotspot_models.csv`): baseline **0.743 / 0.248**, classifiers **0.80 to 0.81 / 0.36 to 0.37**.
- **(c) Intervals:** coverage by volume tier, global versus per-tier, with an 80% target line (`c08_conformal_coverage.csv`). Global covers the busiest stations only **61%**; per-tier **82%**.
- Discussion strip, three lines: *linear models match trees; the lag features carry the signal (+0.048 when removed, `c05_count_ablation.csv`); hotspots persist, so we always compare to "last week's hotspot".*
- Spare charts if a slide needs swapping: `c07_pr_curves.csv` (precision-recall curves), `c10_station_pca.csv` (PCA scatter of stations).

### Slide 10, Conclusion and Future Work (Abdul)
- **Three takeaways:** (1) tickets record officers, and the evening is the least observed period; (2) the weight cap that looks safe hides a -42% bias, and the exposure estimate must not reuse the censored tickets; (3) simple classical models forecast next-day counts (macro MASE 0.697 versus 0.967) and rank next-week hotspots (PR-AUC about 0.80 versus 0.743), with intervals that need per-tier calibration.
- **Limits (small, honest):** five months of one city; four folds; no congestion ground truth; a zero count can mean "no violation" or "no officer"; the benchmark's truth window is itself slightly censored.
- **Future work:** a device-days exposure estimate run end to end with weighting; apply the weighted target to real data; a longer record; spatial-scale comparisons.

### Slide 11, References (shown, not spoken)
Format from `report/ieee/refs.bib`, numbered in this order: Ensign 2018, Gao 2019, Liu and Chen 2025, Inam 2022, Horvitz and Thompson 1952, Schnabel 2016, Lei 2018, Vovk 2005, Grinsztajn 2022, Hoerl and Kennard 1970, Tibshirani 1996, Breiman 2001, Ke 2017, Hyndman and Koehler 2006. Small type, two columns.

### Slide 12, Team contributions and Q&A (Abdul)
Four cards, one line each: **Shaurya** data layer, leak-free features, evening-gap evidence, PCA study. **Afzal** model comparison, reproducibility pipeline, exposure-mechanism check, and the ParkSight hackathon baseline. **Khushi** the censoring benchmark. **Abdul** metrics, conformal intervals, error analysis, report and figures. Large "Questions?".

---

## 4. Timing and handoffs

| Speaker | Slides | Speaking time |
|---|---|---|
| Abdul | 1, 3, 10, 12 | 2:00 |
| Shaurya | 2, 4, 7 | 2:30 |
| Khushi | 5, 8 | 2:15 |
| Afzal | 6, 9 | 2:30 |
| **Total** | | **9:15** |

Handoff lines (end each segment with one):
- Abdul to Shaurya: "Shaurya will start with the problem and the data."
- Shaurya to Abdul (after slide 2): "Abdul will cover what others have done."
- Abdul to Shaurya (slide 3 to 4): "Shaurya, why does this matter here?"
- Shaurya to Khushi (slide 4 to 5): "Khushi will show how we test this."
- Khushi to Afzal (slide 5 to 6): "Afzal covers the models."
- Afzal to Shaurya (slide 6 to 7): "Shaurya will explain the setup."
- Shaurya to Khushi (slide 7 to 8): "Khushi shows what censoring costs."
- Khushi to Afzal (slide 8 to 9): "Afzal shows what forecasting achieves."
- Afzal to Abdul (slide 9 to 10): "Abdul will conclude."

Swap freely if someone is unavailable: the only slides that need domain knowledge are 5 and 8 (the benchmark) and 6 and 9 (models).

---

## 5. Follow-up prompts for Claude Design

- "On slide 8, keep all three charts but reduce text to the takeaway banner only. Make chart (a) the largest."
- "Make every chart use native editable chart objects from the CSVs, not images, and use the same colour for the same model on every slide."
- "Slide 7 has too much text. Turn the metric list into six small chips and keep only the timeline and three facts."
- "Increase all chart labels to at least 16 pt and body text to at least 22 pt. Check nothing overflows."
- "Add speaker notes to every slide, written to be spoken in the time shown, ending with the handoff line."
- "Check every number on every slide against key_numbers.csv and list any that differ."
- "Redraw the methodology diagram natively with the same two lanes, so it matches the deck's style."

---

## 6. Presentable, not just pretty

- **One message per slide**, stated in the title. If a slide needs more than 20 seconds to read, cut text.
- **Charts over text:** slides 4, 6, 7, 8 and 9 are chart slides; slides 2, 3, 10 use short chips and tables.
- **Same story arc as the talk:** problem, test, models, results, honesty.
- **Never show a result without its comparison:** every model chart has the baseline (last-week forecast, last-week hotspot).
- **Colour-blind safe**, and the same method keeps the same colour throughout.

---

## 7. Likely questions, with short answers

1. **Why classical ML and not deep learning?** The data is small tabular data (151 days), and the literature we cite finds simpler models match or beat deep ones there. Our results agree: lasso beat the random forest on macro MASE.
2. **Why macro MASE?** One station has 34,468 tickets; macro MASE gives every station equal weight, and 1.0 means "no better than last week's count".
3. **What is inverse-propensity weighting?** Count a ticket we observed with probability e as 1/e tickets. It removes bias but gets noisy when e is tiny, so a cap trades bias for variance.
4. **Is the evening gap real, or a timestamp problem?** Active devices fall with tickets, which supports a real drop in enforcement. But 34% of tickets are stamped 00:00 to 06:00 and all timestamps have seconds equal to 46, so we cannot rule out a fixed offset; we say so.
5. **Can you recover the real evening?** No: there is no ground truth. The benchmark tests recovery of a known pseudo-truth, and applying it to the real evening is future work.
6. **What does 80% coverage mean?** The interval contains the true count 80% of the time. One global interval is right on average but covers the busiest stations only 61% of the time; per-tier intervals fix that.
7. **Why does lasso win on MASE but not on RMSE?** RMSE is dominated by the few busiest stations; MASE weights every station equally. We have not tested this explanation, so we report macro MASE as the headline.
8. **Why did the estimated exposure fail?** Partly the estimator (distinct devices over 151 days saturates) and partly the benchmark (tickets dropped one by one, not whole officers). Counting device-days under device-level censoring brought the error from 14 times to 1.3 times.
9. **What is new compared with the hackathon project?** The two prediction tasks, folds, metrics, model comparison, benchmark and conformal analysis were written for this course; the hackathon state is kept in git.
10. **How do you know there is no leakage?** Every feature uses only earlier data, and a unit test changes all values from a day onward and checks no earlier feature moves.

---

## 8. Confirm before you present

- Spelling of Shaurya's surname (Malick or Mallick) and whether to show enrollment numbers on the title slide.
- Your slot date and time, and that the deck is at most 12 slides.
- The instructor's presentation template or file format, if any.
- Final numbers: if anyone changes the report or reruns experiments, regenerate the data pack with `python presentation/export_slide_data.py` and re-upload.
