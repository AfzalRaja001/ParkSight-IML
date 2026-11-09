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

The cleaned tickets in `parksight/data/processed/violations_clean.parquet` are committed, so E3 and
the tests run without the raw CSV. The raw CSV is only needed to build the device-level file for E1
and E2 (it is never committed):

```bash
# put "jan to may police violation_anonymized791b166.csv" in the project root, then
.venv/Scripts/python.exe -c "from parksight.research import data; data.build_device_file()"
```

## Run

```bash
.venv/Scripts/python.exe -m pytest tests/research -q      # tests
.venv/Scripts/python.exe experiments/e3_counts.py         # E3, writes results/tables, predictions, figures
```

## Layout and owners

| File | What | Owner |
|---|---|---|
| `paths.py` | absolute paths (override with `PARKSIGHT_RAW_CSV`, `PARKSIGHT_RESEARCH_DATA`) | Afzal |
| `data.py` | station x day grid, pinned time-blocked folds, device file | Shaurya |
| `features.py` | leak-free features for T1 | Shaurya |
| `models.py` | model list, tiny tuning, `run_cv` | Afzal |
| `metrics.py` | macro and pooled MASE, deviance, bootstrap CI, Wilcoxon | Abdul |
| `censor_benchmark.py`, `conformal.py`, `unsup.py` | not written yet (E2, E5, E6) | Khushi, Abdul, Shaurya |

`data.py`, `features.py` and `metrics.py` are first drafts written by Afzal so that E3 could run
end to end. Their owners should review, extend and replace them as needed. The function signatures
follow the plan, so nothing downstream depends on the drafts' internals.

## Conventions

- **Time-blocked only.** Training rows are strictly earlier than test rows. Never shuffle.
- **Features use only earlier days** (`shift(1)` first), so leakage is impossible by construction.
  `tests/research/test_features.py::test_no_future_leak` enforces this.
- **Headline metric is macro MASE** (per station, then averaged). Pooled MASE is reported beside it
  and is not comparable to it, nor to the hackathon's 0.73 (different denominator).
- **Every number in the report comes from a CSV in `results/tables/`.**
- **Ticket-level files with `device_id` stay in `research_data/` (gitignored).**
