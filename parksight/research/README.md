# parksight/research: the IML research layer

New code for the Machine Learning course project. The hackathon code under `parksight/` (app, ETL,
forecast, de-biasing engine) is left as it was at the `hackathon-baseline` git tag. Plans live in
`docs/iml-planning/` (start with `IMPLEMENTATION_PLAN_V2.md` and `TEAM_WORK_SPLIT_V2.md`).

## Setup (once)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-research.txt   # Windows
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt -r requirements-research.txt
```

The cleaned tickets in `parksight/data/processed/violations_clean.parquet` are committed, so E3, E4,
E5 and the tests run without the raw CSV. The raw CSV is only needed for the device-level file used by
E1 and E2 (it is never committed):

```bash
# put "jan to may police violation_anonymized791b166.csv" in the project root, then
PYTHONPATH=. .venv/Scripts/python.exe scripts/build_tickets_device.py
```

## Run

```bash
.venv/Scripts/python.exe -m pytest -q                       # all tests
.venv/Scripts/python.exe experiments/run_all.py             # every experiment that exists, skipping finished ones
.venv/Scripts/python.exe experiments/run_all.py --force --tests   # tests first, then re-run everything
.venv/Scripts/python.exe experiments/e3_counts.py           # or one experiment on its own (e1, e3, e4, e5)
```

## Layout and owners

| File | What | Owner |
|---|---|---|
| `paths.py` | absolute paths for results (override the raw CSV with `PARKSIGHT_RAW_CSV`) | Afzal |
| `data.py` | station x day and hexagon x week grids, hotspot label, pinned time-blocked folds | Shaurya |
| `features.py` | leak-free features for T1 (counts) and T2 (hotspots) | Shaurya |
| `models.py` | T1 and T2 model lists, tiny tuning, `run_cv`, `conformal_handoff` | Afzal |
| `metrics.py` | macro and pooled MASE, deviance, PR-AUC, F1, Brier, bootstrap CI, Wilcoxon | Abdul |
| `conformal.py`, `style.py` | split-conformal intervals; shared figure style | Abdul |
| `censor_benchmark.py`, `unsup.py` | not in the repo yet (E2, E6) | Khushi, Shaurya |

| Experiment | Script | Table |
|---|---|---|
| E1 evening gap | `experiments/e1_eda.py` | `results/tables/E1_eda.csv` |
| E3 count forecasting | `experiments/e3_counts.py` | `E3_counts.csv`, `E3_ablation.csv`, `E3_wilcoxon.csv` |
| E4 hotspot classification | `experiments/e4_hotspots.py` | `E4_hotspots.csv` |
| E5 conformal intervals | `experiments/e5_conformal.py` (reads `results/predictions/t1_best.parquet` from E3) | `E5_conformal.csv` |

## Conventions

- **Time-blocked only.** Training rows are strictly earlier than test rows. Never shuffle.
- **T1 row convention:** one row per (station, day d); `y` is the count on day d and every feature uses
  only days before d. `tests/research/test_alignment.py` pins this.
- **T2 row convention:** one row per (hexagon, origin week w); features use weeks up to w and `y_next`
  is the hotspot label of week w+1.
- **Headline metric is macro MASE** (per station, then averaged). Pooled MASE is reported beside it
  and is not comparable to it, nor to the hackathon's 0.73 (different denominator).
- **Hotspot results always include the persistence baselines and the emerging subset.**
- **Every number in the report comes from a CSV in `results/tables/`.**
- **Ticket-level files with `device_id` stay in `research_data/` (gitignored).**
