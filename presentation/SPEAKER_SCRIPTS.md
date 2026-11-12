# Speaker scripts and Q&A (Group 22)

**Time budget.** The slot is a mandatory **10 minutes for the whole group**. These scripts are written for about **9 minutes 15 seconds** and measure about **8 minutes 45 seconds** at a calm pace (about 1,200 words), which leaves over a minute for slips, laughter and slide changes. Do not add more material: going over is worse than finishing early.

| Speaker | Slides | Time budget | Words |
|---|---|---|---|
| Abdul | 1, 3, 10, 12 | 2:00 | 288 |
| Shaurya | 2, 4, 7 | 2:30 | 338 |
| Khushi | 5, 8 | 2:15 | 298 |
| Afzal | 6, 9 | 2:30 | 286 |

**Slide order:** 1 Title, 2 Introduction, 3 Related work, 4 Problem and motivation, 5 Methodology 1, 6 Methodology 2, 7 Experimental setup, 8 Results 1, 9 Results 2, 10 Conclusion, 11 References, 12 Contributions and Q&A.

**How to use this.** Do not memorise word for word. Learn the **bold numbers** and the **one idea per slide**, read the script aloud twice, then talk from the slide. Cues in [brackets] are what to do. Lines marked **(cut if late)** can be dropped without losing the story. If the deck from Claude Design changes the order or the chart layout, adjust only the [cues], not the numbers.

---

## 1. Abdul

### Slide 1, Title (0:15)
> Good morning. We are Group 22: Shaurya, Afzal, Khushi and Abdul. Our project asks what parking tickets can tell us when a ticket only exists where an officer was standing. Shaurya will start with the problem.

### Slide 3, Related work (0:45)
[Point at each row of the table as you name it.]
> Our base paper is Gao and colleagues, 2019: random forests on New York parking tickets. We replicate that model family in Bengaluru and add a treatment of censoring. Ensign and colleagues proved that training on officer-discovered events creates runaway feedback loops, which is our motivation. To correct uneven observation we use inverse-propensity weighting, from Horvitz and Thompson and Schnabel. Liu and Chen share our premise that ticket data is biased but use a deep model, while Inam and Grinsztajn found that simple models match deep ones on tabular data, which is why we stay classical. **(cut if late: the last clause.)** Shaurya, why does this matter here?

### Slide 10, Conclusion and future work (0:45)
[Slow down. This is the slide people remember.]
> Three takeaways. One: tickets record where officers were, and the evening is the least observed period. Two: a weight cap that looks safe hides a minus 42 percent bias, and the exposure estimate must not reuse the censored tickets. Three: simple classical models forecast next-day counts and next-week hotspots well, but their intervals need per-tier calibration. The limits are honest ones: five months of one city, four folds, no congestion ground truth, and a zero count can mean no violation or no officer. Next, we want to run the device-days estimate end to end with weighting, apply the weighted target to real data, and use a longer record. Thank you.

### Slide 12, Contributions and questions (0:15)
[Point at the four cards.]
> Shaurya built the data layer and the evening-gap evidence, Afzal the model comparison and reproducible pipeline, Khushi the censoring benchmark, and I the metrics, intervals and report. Thank you; we are happy to take questions.

**Delivery tips for Abdul.** You open and close, so keep the first line warm and slow. On slide 3, name each paper while pointing; do not read the table. On slide 10, pause for one second after each takeaway number.

---

## 2. Shaurya

### Slide 2, Introduction (0:45)
[Point at the four number tiles, then the three questions.]
> Cars parked on main roads and junctions slow everyone behind them, yet enforcement is reactive: officers patrol usual routes and write tickets, and those tickets are our only large record. Our data is **298,445** tickets from the Bengaluru traffic police, **54** police stations, **151** days and **3,070** enforcement devices. The catch: a ticket records an officer as much as an offence. So we ask three questions. Is the evening gap real? How much does that censoring hurt, and can weighting fix it? And which classical models forecast best, with honest uncertainty? Abdul will cover prior work.

### Slide 4, Problem and motivation (0:55)
[Point at the two lines. Say the four numbers slowly.]
> This chart shows the problem. Both lines are by hour of the day, each scaled to its own peak: tickets, and active enforcement devices. At 11 in the morning, **1,773** devices are active. At 7 in the evening, just **12**. Tickets fall from **32,580** in the 10 o'clock hour to **27** at 7 pm. Only **0.2 percent** of all tickets fall in the evening rush, between 5 and 9. So a model trained on ticket counts learns where officers went, not where cars were parked illegally, and then sends officers there again. That is a feedback loop. One honest caveat: 34 percent of tickets are stamped between midnight and 6 am, so the clock may be offset, and we define evening on the recorded clock. Khushi will show how we test this.

### Slide 7, Experimental setup (0:50)
[Point at the timeline, then the chips.]
> Evaluation is time-blocked: always train on the past, test on the future. Four expanding windows, each testing a 14-day block starting 13 February, 27 February, 12 March and 26 March. Every feature uses only earlier data, and a unit test checks it. The count table is 54 stations by 151 days; the hotspot table has **16,275** hexagon-weeks, **10.1 percent** of them hotspots. Our headline metric is macro MASE: error per station relative to last week's forecast, averaged so one busy station cannot decide the result. For hotspots, PR-AUC; for uncertainty, 80 percent conformal intervals. It runs on a laptop CPU in about 11 minutes. Khushi will show what censoring costs.

**Delivery tips for Shaurya.** On slide 4, the contrast of 1,773 to 12 is your hook: pause after it. Do not explain the clock caveat for longer than one sentence. On slide 7, if you are running late, drop the reproducibility sentence.

---

## 3. Khushi

### Slide 5, Methodology 1: pipeline and benchmark (0:55)
[Trace the top lane left to right, then the bottom lane.]
> Two lanes share one clean data layer. Top lane, forecasting: tickets become two leak-free tables, station by day for counts and hexagon by week for hotspots, and a ladder of classical models is scored on time-blocked folds with per-station metrics and uncertainty bands. Bottom lane, the censoring benchmark. The real evening has no ground truth, so we take tickets from 2 am to noon, **240,352** of them, as a pseudo-truth. We thin each ticket with a known exposure curve shaped like the real evening dip, and test three estimators: the naive count, weighting with the true exposure at different caps, and weighting with exposure estimated from the thinned tickets. Weighting just counts a kept ticket as one divided by its chance of being kept. Afzal covers the models.

### Slide 8, Results 1: the censoring benchmark (1:20)
[Three charts: say "left", "middle", "right" as you move.]
> Three charts, one story. Left: bias in the total number of tickets at real severity. The naive count misses **77.8 percent**. Weighting with the true exposure helps only if the cap is large: the hackathon cap of 12 still leaves **minus 41.8 percent**, cap 50 minus 14.5, and no cap removes the bias, plus 0.1. Middle, the price: without a cap the variance is **17,817**, against 1,416 at cap 12. Capping trades variance for bias, and the cap that looks safe hides the bias. Right: if we must estimate exposure from the thinned tickets, the hackathon method stays almost as biased as the naive count, **minus 70.9 percent**, because at 9 in the morning it is **14 times** too high. Part of that is our benchmark thinning tickets one by one, and an estimator that saturates. Counting device-days and censoring whole devices brings it down to **1.3 times**, though we did not rerun the weighting with it. Afzal will show forecasting. **(cut if late: the sentence starting 'Part of that'.)**

**Delivery tips for Khushi.** Slide 8 is the core result of the project, so do not rush it. Take the three charts in order, one idea each: bias, price, and the estimator. If someone interrupts, the one sentence to keep is: *the cap that looks safe is the one that hides the bias.*

---

## 4. Afzal

### Slide 6, Methodology 2: models, tuning, training curves (1:00)
[Point at the ladder, then the two curves.]
> The models are classical only, no neural networks. For counts we compare two baselines, last week's count and a seven-day average, against Poisson regression, ridge, lasso, a random forest and LightGBM. For hotspots we use persistence, logistic regression, a linear SVM, a random forest and LightGBM. Each model has one setting that we tune on the last block of its training window, never on the test data. These two charts are LightGBM's training curves, where a boosting round plays the role of an epoch. Training loss keeps falling, but validation loss bottoms out around round **223** for counts and **104** for hotspots, then rises slowly. So the models overfit gently, not sharply, and choosing the number of rounds matters. Shaurya will explain the setup.

### Slide 9, Results 2: forecasting and uncertainty (1:30)
[Three charts: left, middle, right.]
> Now the forecasting results. Left, counts: macro MASE, lower is better, where 1.0 means last week's forecast. Every model beats it. Last week scores **0.967**, lasso **0.697**, ridge 0.699, random forest 0.730 and LightGBM 0.733. So linear models are best on this metric, and tree ensembles do not dominate. Removing the history features costs 0.048, while calendar and station features add nothing. Middle, hotspots: hotspots persist, so the fair baseline is ranking by last week's impact, which scores a PR-AUC of **0.743**. All four classifiers reach about **0.80 to 0.81**, a modest but real gain, with an interval that excludes zero. On the harder emerging hotspots, the ones that were not hot last week, the baseline gets **0.248** and the classifiers **0.36 to 0.37**. Right, uncertainty: with 80 percent intervals, one global interval is right on average but covers the busiest stations only **61 percent** of the time. Per-tier intervals cover **82 percent**. Abdul will conclude. **(cut if late: the ablation sentence.)**

**Delivery tips for Afzal.** On slide 9, say the three baselines first (0.967, 0.743, 0.248) so the audience has something to compare against. Do not claim the models are 'significantly better' in general: say 'a modest but real gain'. If asked which model is best, answer: *on this data, simple models match the complicated ones.*

---

## 5. Questions and answers

**Rule for everyone:** answer in two or three sentences, say 'we did not test that' when it is true, and hand the question to the person who owns that slide. Never defend a claim the results do not support.

### Who answers what
| Topic | Answers |
|---|---|
| Data, evening gap, timestamps, folds, features | Shaurya |
| Benchmark, weighting, caps, exposure estimates | Khushi |
| Models, tuning, MASE vs RMSE, hotspots | Afzal |
| Related work, conformal intervals, metrics, limits | Abdul |

### Questions for Shaurya
1. **Is the evening gap real, or a timestamp problem?** Active devices fall together with tickets, which supports a real drop in enforcement. But 34 percent of tickets are stamped between midnight and 6 am, and every timestamp has seconds equal to 46, so we cannot rule out a fixed offset. We define evening on the recorded clock and say so.
2. **How do you know there is no data leakage?** Every feature uses only earlier data, and a unit test changes all values from one day onward and checks that no earlier feature moves. We also never shuffle: training rows are always earlier than test rows.
3. **Why only four folds?** We have 151 days (about 21 weeks), and each fold needs a long training window and a 14-day or two-week test block. More folds would shrink the training window. Our intervals reflect variation across stations or hexagons, not across folds, and we say differences below about 0.05 PR-AUC are not evidence.
4. **How is a hotspot defined, and why the top 10 percent?** A hexagon in a week is a hotspot if it is in the top 10 percent by weekly impact, the sum of severity times a peak-hour weight. The 10 percent is a design choice, so prevalence is fixed by construction at about 10 percent. We deliberately excluded the trend score used in the hackathon because it would leak the future.
5. **What do the zeros in your table mean?** A zero ticket count can mean no violation or no officer, and the data cannot tell them apart. That is the core problem of the project, not a defect of the table.
6. **What is in the PCA result?** PCA on each station's 24-hour profile: the first component explains 30.3 percent of the variance and separates stations by when they record, not how much. It is supporting evidence for the evening gap, not a main result.

### Questions for Khushi
7. **What is inverse-propensity weighting?** If a ticket is observed with probability e, we count it as 1 divided by e tickets. It removes the bias, but gets noisy when e is very small, so a cap trades bias for variance.
8. **Why is your 'truth' valid if it is itself censored?** It is not an absolute truth. We take hours with many active officers, so we measure recovery of a reference, and we say so. It lets us ask whether a method recovers what was removed, which real data cannot.
9. **Why did the estimated exposure fail?** Two reasons we tested: our benchmark drops tickets one by one while real censoring removes officers, and the estimator counts distinct devices over 151 days, which saturates. Counting device-days and censoring whole devices cut the error at 9 am from 14 times to 1.3 times. We did not rerun the weighting end to end with it.
10. **Why cap 12?** That was the hackathon default, so we tested it. The sweep shows it still leaves a minus 42 percent bias at real severity, though about zero at milder severity.
11. **Can you recover the real evening on real data?** No, there is no ground truth. The benchmark shows what recovery looks like when the truth is known; applying the weighted target to real data is future work.
12. **How many random runs?** 20 seeds for the benchmark, and the interval around each point is drawn, though narrow.

### Questions for Afzal
13. **Why classical machine learning and not deep learning?** The data is small and tabular (151 days), the literature we cite finds simple models match or beat deep ones there, and our results agree: lasso beat the random forest on macro MASE.
14. **Why macro MASE?** One station has 34,468 tickets. Macro MASE scores each station against its own last-week forecast and averages, so every station counts equally, and 1.0 means no better than last week.
15. **What does a MASE of 0.697 mean in practice?** About 28 percent lower error than the last-week forecast (0.697 against 0.967), averaged over stations.
16. **Why does lasso win on MASE but the random forest on RMSE?** RMSE is dominated by the few busiest stations, MASE weights every station equally. We have not tested that explanation, so we report macro MASE as the headline and show the others beside it.
17. **Are these differences significant?** Lasso and ridge beat the forest station by station (p about 0.003 and 0.002), LightGBM and the forest are indistinguishable (p 0.72), and the intervals of the top five models overlap. Our 12 tests are not corrected for multiple comparisons, so the order inside that group should not be over-read.
18. **Why Poisson, ridge and lasso on a log scale?** Counts are non-negative and skewed. A Poisson model with a log link predicts positive counts, and ridge and lasso on log(1+y) handle skew and, for lasso, drop unhelpful features.
19. **What did hotspot models add over the baseline?** A modest but real gain: PR-AUC 0.80 to 0.81 against 0.743 over all hexagons, and 0.36 to 0.37 against 0.248 on emerging hotspots. Absolute F1 on emerging hotspots stays between 0.24 and 0.31, so new hotspots are ranked better, not found reliably.
20. **How did you tune, and could that leak?** One setting per model, chosen on the last block of the training window (14 days or two weeks), never on the test block.

### Questions for Abdul
21. **What does 80 percent coverage mean?** The interval contains the true count 80 percent of the time. One global interval gets 80 percent overall but only 61 percent for the busiest stations and 95 percent for the quietest. Per-tier intervals give 79, 80 and 82 percent.
22. **Why do per-tier intervals help?** Busy stations make large errors (mean absolute error 30 tickets) and quiet ones tiny errors (6), so one width cannot fit both. The cost is wider bands where counts are large (86 tickets against 44 for the busiest tier).
23. **What is new compared with the hackathon project?** The two prediction tasks, the folds, metrics, model comparison, benchmark and conformal analysis were written for this course. The hackathon state is preserved in git as a baseline, and we say so in the report.
24. **What is the impact score (PCIS) and did you validate it?** It was the hackathon's designed score from volume, severity, location, peak overlap and trend, not a measured delay. Against 12 expert-ranked junctions, the volume term alone has Spearman 0.93 and the full score 0.92, so the extra terms add no agreement. We treat it as a proxy.
25. **What are the main limitations?** Five months of one city, four folds, no congestion ground truth, a zero can mean no violation or no officer, the benchmark truth is slightly censored, and timestamps may carry an offset.

### Questions anyone may get (answer, then hand over)
26. **Does your forecast tell police where to deploy?** No. It predicts where tickets will be written, which reflects enforcement workload, not where violations occur. The evening cannot be recovered from these models.
27. **What would you do with more time or data?** Run the device-days exposure estimate end to end with weighting, apply the weighted target to real data, use a longer record with a full year, and compare spatial scales as the base paper does.
28. **Why is this novel?** Gao et al. compare models on tickets; we add an explicit test of how censoring distorts them, a benchmark with a known truth, and per-tier intervals, in a different city.

---

## 6. Rehearsal checklist

- **Full run-through with a timer, twice.** Target 9:15, and stop at 10:00 even mid-sentence in rehearsal.
- **Practise the handoffs.** Each speaker ends with their one-line handoff; the next speaker starts walking to the screen on the last sentence.
- **One person controls the slides.** Agree who clicks, and that the speaker says 'next slide' only if the clicker is not a speaker.
- **Practise the answers aloud:** each person should answer their own questions in under 20 seconds.
- **Have the PDF backup open** in case the PowerPoint file fails, and check every number against `presentation/attach/data/key_numbers.csv`.
