"""Absolute paths for the research layer. Override with environment variables if needed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "parksight" / "data" / "processed"

# ticket-level data with device_id: NEVER committed (gitignored)
RESEARCH_DATA = Path(os.environ.get("PARKSIGHT_RESEARCH_DATA", ROOT / "research_data")).resolve()
RAW_CSV = Path(os.environ.get(
    "PARKSIGHT_RAW_CSV", ROOT / "jan to may police violation_anonymized791b166.csv")).resolve()
DEVICE_FILE = RESEARCH_DATA / "tickets_device.parquet"

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"


def ensure_dirs():
    for p in (RESEARCH_DATA, TABLES, FIGURES, PREDICTIONS):
        p.mkdir(parents=True, exist_ok=True)
