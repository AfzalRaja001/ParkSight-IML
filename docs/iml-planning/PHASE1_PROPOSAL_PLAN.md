# ParkSight for the ML Course - Phase 1 (Proposal) Working Doc

**Status:** draft for team review
**Source repo:** https://github.com/Susan5504R/ParkSight (built for Flipkart hackathon, being adapted)
**Course deliverable 1:** Proposal, PDF, 300 to 500 words, submitted on Google Classroom by ONE member who tags the other three.

> Heads up on dates: the guidelines PDF lists the proposal deadline as **15 September**, and today is 17 September. Somebody should confirm with the instructor whether the deadline moved or whether we are already late. Everything below assumes we are still submitting.

---

## 0. First, the vocabulary you were trying to remember

You said "impact score / confidence score". Those are not the standard terms. Here is what actually exists, so we use the right words in front of the professor.

**Citation count.** How many other papers have cited this specific paper. This is a property of the *paper*. Google Scholar gives the most generous count, Scopus and Semantic Scholar give smaller and stricter ones. This is the single most useful number for judging whether a paper is a reasonable thing to build on.

**Impact Factor (IF).** A property of a *journal*, not a paper. Published yearly by Clarivate in the Journal Citation Reports. Roughly: the average number of citations received in one year by articles that journal published in the previous two years. An IF of 4.8 means the average recent article in that journal picked up about 4.8 citations per year.

**Quartile (Q1, Q2, Q3, Q4).** Where a journal sits inside its own subject category. Q1 means top 25 percent of journals in that field. This matters more than the raw IF number, because IF values are not comparable across fields (medicine journals have far higher IFs than CS journals for reasons that have nothing to do with quality).

**CiteScore and SJR.** Scopus's competing metrics to IF. CiteScore is a 4 year citation average. SJR (SCImago Journal Rank) weights citations by the prestige of the citing journal, similar in spirit to PageRank.

**h5-index.** The metric for *conferences*, from Google Scholar Metrics. A venue has h5-index = h if it published h papers in the last 5 years that each got at least h citations. NeurIPS is 371, ICLR 362, ICML 272. Journals do not usually get judged by this, conferences do, because top ML conferences do not have impact factors.

**One important caveat for our case:** in machine learning, the best work is published at *conferences* (NeurIPS, ICML, ICLR, CVPR, FAccT), and conferences have no impact factor at all. So when we report metrics in the proposal we should report the right metric for the right venue type: IF plus quartile for journal papers, h5-index or citation count for conference papers. Reporting "impact factor: N/A" for a NeurIPS paper is correct and shows we understand the difference.

---

## 1. Paper shortlist (4 papers)

I searched for work on three things at once, because our project sits at their intersection: (a) predicting parking violations from ticket data, (b) using classical ML on tabular spatio temporal data, and (c) correcting for the fact that enforcement data records where police *were*, not where violations *happened*.

### P1. Gao, Li, Liang, Marks, Kang, Li (2019) - "Predicting the spatiotemporal legality of on-street parking using open data and machine learning"

| Field | Value |
|---|---|
| Venue | *Annals of GIS*, vol. 25, pp. 299 to 312 (Taylor & Francis) |
| DOI | 10.1080/19475683.2019.1679882 |
| Journal Impact Factor | 4.8 (2025 JCR) |
| Quartile | **Q1** |
| CiteScore (Scopus) | 8.2, Q1 |
| SJR | 1.014 |
| Citations | 64 (Google Scholar), 41 (Semantic Scholar) |

**What it does.** Takes NYC parking ticket open data, joins it with Points of Interest data and human mobility data, and predicts (i) the count of violation tickets and (ii) a binary "is parking here legal right now" label. Crucially, it runs the same models at four different spatial aggregation levels (point, street segment, census tract, and grid cell) to measure how much the choice of spatial unit changes results. Random Forest wins on both the regression (lowest RMSE) and the classification (highest accuracy) at every spatial scale. It also reports which POI categories are associated with more violations (retail, healthcare, accommodation, food services).

**Why it matters to us.** This is almost exactly our problem, in another city, already solved with classical ML. It gives us a ready made baseline to replicate and beat, a feature engineering recipe (POI plus mobility), and an experimental design we can copy (the spatial scale sweep). Our H3 hexagons are one more spatial unit we can add to their comparison.

---

### P2. Ensign, Friedler, Neville, Scheidegger, Venkatasubramanian (2018) - "Runaway Feedback Loops in Predictive Policing"

| Field | Value |
|---|---|
| Venue | *Conference on Fairness, Accountability and Transparency* (FAT*, now ACM FAccT), PMLR vol. 81, pp. 160 to 171 |
| arXiv | 1706.09847 |
| Journal Impact Factor | Not applicable, this is a conference (see section 0) |
| Citations | **741** (Google Scholar), 324 (Semantic Scholar), 22 "highly influential" citations |
| Standing | FAccT is the flagship ACM venue for this research area; PMLR is the same proceedings series that publishes ICML |

**What it does.** Proves mathematically why predictive policing systems trained on *discovered* crime (arrests made by officers the system dispatched) collapse into a feedback loop: the model sends police to neighbourhood A, police in A find more crime, the model becomes even more sure about A, and the true crime rate in B is never learned. They build an urn model (a Polya urn process), prove the loop occurs, show it empirically, and then show a black box fix: reweight the update so that each discovered incident is counted relative to how much it was *expected* given the policing effort already spent there. They also show that citizen reported crime attenuates the loop but does not remove it.

**Why it matters to us.** This is the theoretical backbone for ParkSight's de biasing engine. Right now our repo justifies inverse propensity weighting informally ("evening enforcement blind spot"). Ensign et al. give us the formal argument for *why* naively training on ticket counts is wrong, plus a principled correction. Without a paper like this, our de biasing engine is a heuristic we invented; with it, it is an application of an established result.

---

### P3. Liu & Chen (2025) - "Short-term parking violations demand dynamic prediction considering spatio-temporal heterogeneous effects of the built environment"

| Field | Value |
|---|---|
| Venue | *Transportation* (Springer) |
| DOI | 10.1007/s11116-025-10632-7 |
| Journal Impact Factor | 4.1 (2025 JCR) |
| Quartile | **Q1** |
| Citations | 1 (published May 2025, too recent to accumulate) |

**What it does.** Uses 80,572 electronic police parking violation records over 6 months in Chongqing. Builds a feature system of built environment factors (land use via AOI polygons rather than POI points, transport supply, road design), natural environment factors, and historical violation lags. Fits a Multiscale Geographically and Temporally Weighted Regression (MGTWR) model, then feeds MGTWR's spatio temporal coefficients into a deep learning stack (ResNet + Graph Attention Network + attention LSTM, called MGTWR-RGA). Average prediction error under 1 violation.

**Why we shortlisted it but are NOT making it a base paper.** Two reasons it is valuable: first, its opening line is literally our problem statement, it says parking violation management is hampered by "the spatio-temporal bias of traditional manual parking ticket data", which is independent confirmation that our blind spot premise is a real, publishable concern. Second, its feature system is the best checklist we have for what to engineer. But it is a deep learning paper, and we are deliberately going classical, so replicating it is not our goal. We will cite it as recent state of the art and as our motivation, and note that we deliberately trade a little accuracy for interpretability and for a working bias correction, which their model does not have.

---

### P4. Inam, Mahmood, Khatoon, Alshamari, Nawaz (2022) - "Multisource data integration and comparative analysis of machine learning models for on-street parking prediction"

| Field | Value |
|---|---|
| Venue | *Sustainability* (MDPI), 14(12), 7317 |
| DOI | 10.3390/su14127317 |
| Journal Impact Factor | 4.1 (2025 JCR) |
| Quartile | Q2 |
| CiteScore | 8.9 |
| Citations | 27 (Google Scholar) |

**What it does.** Integrates parking occupancy, pedestrian counts, weather, and traffic data, then runs a head to head comparison of MLP, Random Forest, Decision Tree, kNN, Gradient Boosting, AdaBoost, and linear SVC for predicting on street parking availability. Random Forest wins at 81 percent average accuracy. The headline finding is the one we care about: **the simpler algorithms (RF, DT, kNN) beat the more complex one (MLP) on prediction accuracy.**

**Why we shortlisted it but are NOT making it a base paper.** It is a solid, directly quotable empirical justification for our decision to go classical, and its comparison table is a template for ours. But it is a Q2 journal, the methodology is a straightforward model bake off with no novel contribution, and it studies parking *availability* rather than *violations*. It is a supporting citation, not a foundation.

---

### Additional supporting reference (not part of the shortlist, but we should cite it)

**Grinsztajn, Oyallon, Varoquaux (2022) - "Why do tree-based models still outperform deep learning on typical tabular data?"**, NeurIPS 2022 Datasets and Benchmarks Track. **4,194 Google Scholar citations.** NeurIPS h5-index 371.

45 datasets, 20,000 compute hours of hyperparameter search, conclusion: tree ensembles (XGBoost, Random Forest) remain state of the art on medium sized tabular data, and the gap survives hyperparameter tuning. They explain why in terms of inductive bias: neural networks are hurt by uninformative features, are rotation invariant when the data is not, and struggle to fit irregular (non smooth) target functions. Our data is medium sized tabular spatio temporal data with many weak features, which is precisely the regime this paper describes. This is our one line defence if anyone asks "why not deep learning".

---

## 2. The two base papers, and why these two

**Base Paper 1: Gao et al. (2019), Annals of GIS, Q1, IF 4.8, 64 citations.**
**Base Paper 2: Ensign et al. (2018), FAT* / PMLR 81, 741 citations.**

The reason to pick this specific pair is that they are *complementary rather than redundant*, and together they define both halves of our project:

- Gao et al. gives us the **application half**: the task definition (predict violation counts and a binary legality/hotspot label from ticket data), the feature sources (POI, mobility), the model class (classical, Random Forest wins), the evaluation metric (RMSE and accuracy), and the spatial scale experiment. We replicate it, then extend it.
- Ensign et al. gives us the **methodological half**: the proof that training on enforcement discovered events without correction is systematically wrong, and the correction to apply. This is what makes our project more than a model bake off.

And the combination is the originality claim the course explicitly asks for. The guidelines say: *"Replication of prior work is allowed, but must be extended with new applications or analysis."* Our extension is exactly that: **Gao et al.'s framework has never been run with Ensign et al.'s bias correction, and never on an Indian city.** That one sentence is our contribution statement.

If someone prefers a pure recommender systems style formal treatment of inverse propensity weighting instead of Ensign, the alternative is Schnabel et al. (2016), *"Recommendations as Treatments: Debiasing Learning and Evaluation"*, ICML 2016, PMLR 48, several thousand citations. It gives cleaner estimator math (the Horvitz Thompson / IPS estimator we already use in the repo) but it is set in recommender systems, so the story is less direct. My call: go with Ensign, keep Schnabel as a cited reference for the estimator itself.

---

## 3. The model question: should we drop LightGBM?

Short, honest answer first, because this affects how we word the proposal.

**LightGBM is already classical machine learning.** It is a gradient boosted decision tree ensemble. It contains no neural network, no backpropagation, no representation learning. "Classical ML" conventionally means everything that is not deep learning: linear and logistic regression, SVMs, decision trees, ensembles (bagging, boosting, random forests), kNN, naive Bayes, clustering, PCA. Boosting belongs squarely in that list. So the guidelines' note about deep learning projects needing classical baselines does not even apply to us, our project has no deep learning in it at all.

**But your instinct is still right, for a different reason.** The problem with our current repo is not that LightGBM is too fancy. The problem is that *one tuned model is a thin story for an ML course*. The course grades Technical Quality ("sound methodology, appropriate algorithms, meaningful results") and Originality. A single Optuna tuned LightGBM with a MASE number is a good product result and a weak coursework result. What gets marks is a **model ladder with ablations, error analysis, and a reason for every step.**

So the proposal is: **keep LightGBM, but demote it from "the model" to "the top rung of a ladder"**, and add the rungs below it. Concretely:

### 3.1 The regression ladder (predicting violation counts per hex per hour)

| Rung | Model | What it demonstrates |
|---|---|---|
| 0 | Seasonal naive / climatology (same hex, same hour, same weekday, last week) | The floor. Any model that cannot beat this is worthless. Already in the repo as the MASE denominator. |
| 1 | **Poisson GLM** (generalised linear model with a log link) | The statistically correct linear baseline. Our target is a *count*, so Poisson regression is the textbook right answer, not ordinary least squares. Gives interpretable multiplicative coefficients: "a commercial hex has e^β times more violations". |
| 2 | **Ridge / Lasso regression** on the same features | Regularisation, bias variance tradeoff, and Lasso gives us automatic feature selection we can report. |
| 3 | **Decision Tree** (single, depth limited) | The interpretable nonlinear model. We can literally print the tree in the poster. |
| 4 | **Random Forest** | Bagging. This is the model Gao et al. and Inam et al. both found best, so it is our direct comparison to the base paper. |
| 5 | **Gradient Boosting (LightGBM / HistGradientBoostingRegressor)** | Boosting. Our current model, now with something to compare against. |
| 6 | **SVR with RBF kernel** on a reduced feature set | Kernel methods. Will probably lose, and that is fine, a negative result with an explanation is worth marks. |
| 7 | *(optional)* small MLP | Deliberately included to *empirically reproduce* Grinsztajn et al.'s finding on our own data. "We tried deep learning, trees won, here is the table" is a stronger statement than "we did not try deep learning". |

### 3.2 Reframe part of the task as classification

This is the biggest single upgrade available to us. Right now everything is regression, which limits the metrics we can show. Add a binary task:

> **Will this hex be a high impact hotspot (top decile PCIS) during the next 7 day window?**

That one change unlocks Logistic Regression, Linear/RBF SVM, Naive Bayes, RF and GBM classifiers, and with them: ROC curves, precision recall curves (essential here because hotspots are rare, so the classes are imbalanced), confusion matrices, F1, Brier score, and calibration plots. It also matches Base Paper 1, which does exactly this (they classify spatiotemporal legality alongside regressing ticket counts). And operationally it is the more useful output anyway, enforcement teams want a ranked list, not a decimal count.

### 3.3 Add an unsupervised component

The guidelines explicitly name unsupervised methods. We already use DBSCAN for map clustering, but that is presentation, not analysis. Two genuinely useful additions:

- **PCA on the hourly profile matrix.** Represent each station or hex as a 24 dimensional vector of its average hourly violation counts, normalised. Run PCA. The hypothesis, which we can then test, is that the evening enforcement blind spot shows up as a dominant principal component separating stations by *when they record*, not by *how much* they record. If that comes out, it is a striking, visual, defensible result and it validates the entire premise of the de biasing engine using an unsupervised method that cannot be accused of circularity.
- **K-Means or Gaussian Mixture clustering** of hexes into behavioural archetypes (commercial daytime, transit peak, nightlife, chronic all day). Choose k with the elbow method and silhouette score, both of which are teachable and gradeable. Then feed the cluster label back in as a categorical feature to the supervised models and report whether it helps, which is a clean, self contained ablation.

### 3.4 Keep conformal prediction

Mondrian conformal prediction for the ~80 percent intervals is distribution free, has finite sample coverage guarantees, and is completely orthogonal to the choice of base model, so it works with a Poisson GLM just as well as with LightGBM. It is also genuinely fashionable in current ML teaching. Keep it, and report empirical coverage versus nominal coverage as a results table. That is honest uncertainty quantification and it is exactly the kind of thing that reads as methodological maturity.

---

## 4. The experiment that turns this from a project into a contribution

This is the part I would push hardest for.

Our current de biasing claim has a hole in it: **we have no ground truth.** We assert that inverse propensity weighting recovers the true violation rate in the evening, but we cannot prove it, because the true evening violation rate is exactly the thing that was never recorded. A grader will notice this immediately. The repo's own limitations section already admits it.

**The fix: build a semi synthetic bias benchmark.**

1. Take a slice of data from hours where enforcement coverage is high and roughly uniform (the morning window). Call the observed counts there our pseudo ground truth, `y_true`.
2. Apply a *known*, artificial censoring function to it. For example, drop each record with probability `1 - exposure(h)` where `exposure(h)` is a curve we chose ourselves. Now we have biased data `y_obs` AND we know the exact bias that produced it.
3. Train the full pipeline on `y_obs`, once with inverse propensity weighting and once without.
4. Measure how close each recovers `y_true`.

Now the de biasing engine has a number attached to it instead of a story. We can report: "without correction, the model underestimates censored period counts by X percent; with inverse propensity weighting, error drops to Y percent." We can sweep the severity of the censoring and plot the recovery curve. We can test the failure mode (what happens when our propensity estimate is itself wrong, which Schnabel et al. warn about).

This single experiment does more for Technical Quality and Originality than any amount of extra model tuning, because it is a real, falsifiable, quantitative claim that nobody in the four shortlisted papers has made for this problem.

---

## 5. What changes in the repo

The repo is currently a hackathon product. The course wants a research artifact. These are different shapes, so we should separate them rather than trying to make one thing be both.

**Keep and freeze (demo layer, do not spend time here):**
- The Streamlit app, pydeck maps, Plotly charts. Good for the poster and any demo. Zero further development.
- The ETL pipeline (CSV cleaning, UTC to IST conversion, H3 hexagon assignment). This is real work and it is done.

**Cut or park (not machine learning, will not earn marks, may invite awkward questions):**
- The AI Copilot / Claude integration. It is a nice product feature and it is not ML. If a grader asks "which part of this did you learn from data", pointing at an LLM API call is a bad moment. Leave the code, remove it from the report and poster.
- PDF briefing generation, python-pptx export. Product polish, not results.
- The Simulator page, unless we reframe it as sensitivity analysis on the PCIS weights, which is actually worth doing (see below).

**Build new (this is where the marks are):**

```
parksight/
  ml/
    datasets.py      # loads processed parquet, builds the (X, y) matrices,
                     # returns time-blocked splits. No leakage, ever.
    features.py      # every engineered feature, each with a docstring saying
                     # WHY it should predict violations
    models.py        # all ladder rungs as sklearn-compatible estimators so
                     # they share one fit/predict interface
    weighting.py     # the IPW / propensity estimation, isolated and testable
    conformal.py     # Mondrian conformal wrapper, model-agnostic
    evaluate.py      # MAE, RMSE, MASE, Poisson deviance, PR-AUC, Brier,
                     # empirical coverage
  experiments/
    exp01_model_ladder.py
    exp02_ablations.py          # +/- IPW, +/- spatial, +/- lags, +/- clusters
    exp03_spatial_scale.py      # replicate Gao et al.'s four spatial units
    exp04_synthetic_bias.py     # the section 4 benchmark
    exp05_unsupervised.py       # PCA + KMeans/GMM
    run_all.py                  # regenerates every table in the report
  notebooks/
    01_eda.ipynb                # distributions, missingness, the blind spot
    02_error_analysis.ipynb     # residuals by hour, road class, vehicle type
results/
  tables/*.csv
  figures/*.png
```

**Non negotiable engineering rules for this rewrite:**

1. **Time blocked splits only.** Never `train_test_split(shuffle=True)` on time series. Train on the past, validate on the future, always. The repo already does rolling CV, keep that discipline everywhere.
2. **Fixed seeds everywhere**, and record them. Every number in the report must be reproducible by running `run_all.py`.
3. **One results CSV per experiment**, committed. The report reads from those files, we never retype numbers by hand.
4. **Absolute paths in configs**, per our usual convention.

**One more thing worth doing: PCIS sensitivity analysis.**
PCIS is currently a hand chosen weighted sum (0.30 volume + 0.20 severity + 0.20 location + 0.20 peak overlap + 0.10 trend). Those weights were picked by us, not learned, and a grader will ask about it. The honest and interesting answer is a sensitivity analysis: perturb the weights (Dirichlet sampling, or a one at a time sweep) and measure how much the top 20 hotspot ranking changes (rank correlation, Jaccard overlap of the top 20 sets). If the ranking is stable under perturbation, the arbitrariness of the weights does not matter and we can prove it. If it is unstable, that is an honest finding worth reporting. Either outcome is a good section. This is also the honest way to reuse the Simulator page.

---

## 6. Draft proposal text (435 words, within the 300 to 500 limit)

Copy this into a doc, fill in the names and enrolment numbers, export as PDF.

---

**Title:** De-biasing Enforcement Data for Parking Congestion Prediction: A Classical Machine Learning Study on Bengaluru Violation Records

**Category:** Application Project (with a methodological extension)

**Team:** [Name 1, Enrolment No.], [Name 2, Enrolment No.], [Name 3, Enrolment No.], [Name 4, Enrolment No.]

**Motivation.** Urban parking enforcement is reactive. Authorities know where tickets were issued, not where violations actually occur, because a ticket records the presence of an officer as much as the presence of an offence. In 298,445 anonymised Bengaluru traffic police parking violation records (November 2023 to April 2024), recorded violations collapse during the evening peak (15:00 to 24:00 IST), precisely when congestion is worst. Training a predictor directly on these counts therefore learns enforcement habits and labels underpoliced evening zones as safe. Ensign et al. (2018) prove this feedback loop is intrinsic to models trained on officer-discovered events. We tackle the resulting question: can classical machine learning predict genuine parking congestion impact from systematically biased enforcement data?

**Method.** We extend the framework of Gao et al. (2019), who predicted violation counts and parking legality from NYC ticket data using classical models, by adding an explicit bias correction. Records are aggregated into H3 hexagon by hour cells with temporal, spatial, road-hierarchy and lag features. We solve two tasks: Poisson regression for violation counts, and binary classification of high-impact hotspots. We compare a model ladder of Poisson GLM, Ridge and Lasso regression, decision trees, Random Forest, gradient boosting, and SVM, plus unsupervised analysis (PCA on hourly profiles, K-Means archetypes). Exposure-corrected inverse-propensity weights re-weight training samples, and Mondrian conformal prediction supplies distribution-free intervals.

**Intended experiments.** Baselines are seasonal-naive climatology and an unweighted Random Forest replicating Gao et al. Evaluation uses rolling-origin cross-validation with strictly time-blocked splits, reporting MASE, RMSE and Poisson deviance for counts, PR-AUC, F1 and Brier score for hotspot classification, and empirical versus nominal coverage for intervals. Ablations isolate the contribution of inverse-propensity weighting, spatial features, lag features and cluster labels, and we replicate the base paper's comparison across spatial aggregation scales. Because true evening violation rates are unobservable, we construct a semi-synthetic benchmark: we censor a well-covered period using a known exposure function and measure how accurately weighting recovers the withheld counts. We expect tree ensembles to lead on accuracy, consistent with Grinsztajn et al. (2022) on tabular data, and weighting to substantially reduce underestimation in censored periods. Error analysis by hour, road class and vehicle type follows.

**Contributions.** Member 1: data pipeline, feature engineering, exploratory analysis. Member 2: model ladder, tuning, cross-validation. Member 3: de-biasing, propensity estimation, synthetic benchmark. Member 4: evaluation, error analysis, conformal intervals, visualisation. Report and poster are written jointly.

---

*Word count: 394 words for the body (Motivation through Contributions), 435 words including the title, category and team block. Comfortably inside the 300 to 500 limit, with room to add detail if we want it.*

---

## 7. Suggested division of work

Four members, four tracks that can proceed in parallel after week 1.

| Member | Track | Owns |
|---|---|---|
| 1 | **Data** | ETL, H3 aggregation, feature engineering, `features.py`, the EDA notebook, data quality section |
| 2 | **Models** | The model ladder, hyperparameter search, `models.py`, rolling CV harness, the main results table |
| 3 | **De-biasing** | Propensity/exposure estimation, IPW, the semi-synthetic benchmark (section 4), the PCIS sensitivity analysis |
| 4 | **Evaluation** | Metrics, error analysis, conformal intervals, unsupervised analysis (PCA, clustering), all figures |

Report writing, poster, and code review are joint. One person (decide who) is the single submitter on Google Classroom and must tag the other three, or we lose a point.

---

## 8. Timeline against the course deadlines

| Date | Deliverable | What must be true by then |
|---|---|---|
| Now | **Proposal** (300 to 500 words, PDF) | Section 6 text finalised, names and enrolment numbers filled in, submitted |
| ~1 week after | Internal checkpoint | Repo restructured into `parksight/ml/`, time-blocked splitting working, Poisson GLM and RF baselines running end to end |
| ~3 weeks after | Internal checkpoint | Full model ladder done, ablation table populated, synthetic bias benchmark producing numbers |
| **20 October** | **Milestone** (max 3 pages) | Motivation restated, methods tried and justified, preliminary results with baselines, error analysis, next steps |
| Early November | Internal checkpoint | All experiments frozen, `run_all.py` reproduces every table, figures finalised |
| **15 November** | **Poster + Final Report** (4 to 6 pages, ICML/NeurIPS LaTeX style) | Full report: abstract, intro, related work (our 4 papers), methodology, experiments, analysis, conclusion, references, per member contributions |

Note that the final report must use the ICML or NeurIPS LaTeX template. Download it early and set up Overleaf in week 1 rather than the night before.

---

## 9. Risks and honest weaknesses to pre-empt

These are the questions a grader will ask. Better that we have answers ready than get surprised.

1. **"PCIS is a formula you invented, not something you learned."** True. Answer: PCIS is an explicitly declared proxy target, fixed before modelling, and we quantify how much our conclusions depend on its weights via sensitivity analysis (section 5). We are honest that no ground truth congestion data exists for Bengaluru.
2. **"How do you know the de-biasing works?"** This is the hole. Answer: the semi-synthetic benchmark in section 4. Without it we have no answer, which is exactly why it is the highest priority experiment.
3. **"Your propensity/exposure estimate could itself be wrong."** Correct, and Schnabel et al. (2016) document exactly this failure mode for IPS estimators. We should test sensitivity to a misspecified exposure curve and report it rather than hide it.
4. **"Five months of data in one city."** Cannot model annual seasonality, cannot claim generalisation. State it plainly in limitations. It is a fine limitation for a course project as long as we do not overclaim.
5. **"This looks like a hackathon dashboard."** Which is why the report and poster must lead with the ML, the tables, and the ablations, and treat the Streamlit app as a one slide appendix.
6. **Reusing prior work.** The guidelines allow replication if extended. Our extension is real (new city, new bias correction, new benchmark), but we should say openly in the report that the codebase originated from a hackathon project and describe precisely what was added for this course. Being upfront costs nothing and looks far better than being asked about it.

---

## 10. Open questions for the team

1. Is the 15 September deadline still live? Someone please confirm with the instructor today.
2. Who submits on Google Classroom and tags the rest?
3. Do we have POI or road network data for Bengaluru to match Gao et al.'s feature set? OpenStreetMap is the obvious free source. If not, which features from Liu & Chen (2025) can we actually construct?
4. Does everyone agree on dropping the AI Copilot from the academic deliverable?
5. Are we all comfortable with scikit-learn, or do we need a shared session on the sklearn estimator API before the pipeline rewrite?
6. Overleaf account and ICML/NeurIPS template: who sets it up?

---

## Sources

- [Gao et al. 2019, Annals of GIS (Taylor & Francis)](https://www.tandfonline.com/doi/full/10.1080/19475683.2019.1679882)
- [Ensign et al. 2018, PMLR v81 (FAT*)](https://proceedings.mlr.press/v81/ensign18a.html) / [arXiv:1706.09847](https://arxiv.org/pdf/1706.09847)
- [Liu & Chen 2025, Transportation (Springer)](https://link.springer.com/article/10.1007/s11116-025-10632-7)
- [Inam et al. 2022, Sustainability 14(12) 7317 (MDPI)](https://www.mdpi.com/2071-1050/14/12/7317)
- [Grinsztajn et al. 2022, NeurIPS Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
- [Schnabel et al. 2016, ICML / PMLR v48](https://proceedings.mlr.press/v48/schnabel16.html)
- [Annals of GIS journal metrics](https://researcher.life/journal/annals-of-gis/8053)
- [Transportation journal metrics](https://researcher.life/journal/transportation/668)
- [Google Scholar Metrics, Artificial Intelligence venues](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence)
