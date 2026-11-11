"""Run every experiment that exists in the repo, in order, and check the outputs. Owner: Afzal.

  python experiments/run_all.py            run what is missing (skips experiments whose table exists)
  python experiments/run_all.py --force    re-run everything
  python experiments/run_all.py --only e3 e4
  python experiments/run_all.py --tests    run pytest first and stop if it fails

Experiments (script -> main table):  E1 e1_eda.py, E2 e2_benchmark.py, E3 e3_counts.py, E4 e4_hotspots.py,
E5 e5_conformal.py, E6 e6_unsup.py, E7 e7_training_curve.py, plus the method diagram (F0) and, last, the
IEEE report tables (make_ieee_tables.py), which are rebuilt whenever any experiment ran. A script that is
not in the repo yet is reported, not an error.
E5 needs E3's hand-off file; E1 needs research_data/tickets_device.parquet (build it with
scripts/build_tickets_device.py and the raw CSV).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [   # (id, script, output that proves it ran, optional prerequisite file)
    ("e1", "e1_eda.py", "results/tables/E1_eda.csv", "research_data/tickets_device.parquet"),
    ("e2", "e2_benchmark.py", "results/tables/E2_benchmark.csv", None),
    ("e3", "e3_counts.py", "results/tables/E3_counts.csv", None),
    ("e4", "e4_hotspots.py", "results/tables/E4_hotspots.csv", None),
    ("e5", "e5_conformal.py", "results/tables/E5_conformal.csv", "results/predictions/t1_best.parquet"),
    ("e6", "e6_unsup.py", "results/tables/E6_unsup.csv", None),
    ("e7", "e7_training_curve.py", "results/tables/E7_training_curve.csv", None),
    ("f0", "make_method_diagram.py", "results/figures/F0_method.pdf", None),
    ("pcis", "pcis_volume_check.py", "results/tables/PCIS_volume_check.csv", None),
]
TABLES_SCRIPT = "make_ieee_tables.py"      # reads the E2-E5 CSVs; always re-run after any experiment ran
TABLES_OUT = "results/tables/ieee_counts.tex"


def run(script, extra_env):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", **extra_env}
    return subprocess.run([sys.executable, str(ROOT / "experiments" / script)], cwd=ROOT, env=env).returncode


def check_e4():
    """E4 must always report the persistence baselines, or the hotspot results mean nothing."""
    t = pd.read_csv(ROOT / "results/tables/E4_hotspots.csv")
    missing = {"persistence", "persistence_impact"} - set(t["model"])
    if missing:
        raise SystemExit(f"E4 table is missing baselines: {sorted(missing)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tests", action="store_true")
    args = ap.parse_args()

    if args.tests:
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode
        if rc:
            raise SystemExit("tests failed; not running experiments")

    summary = []
    for eid, script, out, prereq in EXPERIMENTS:
        if args.only and eid not in args.only:
            continue
        if not (ROOT / "experiments" / script).exists():
            summary.append((eid, "not in repo yet")); continue
        if prereq and not (ROOT / prereq).exists():
            summary.append((eid, f"skipped, missing {prereq}")); continue
        if (ROOT / out).exists() and not args.force:
            summary.append((eid, "skipped, output exists (use --force)")); continue
        t0 = time.time()
        rc = run(script, {})
        summary.append((eid, f"{'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f}s"))
        if rc:
            break
        if eid == "e4":
            check_e4()
    ran = any(msg.startswith(("ok", "FAILED")) for _, msg in summary)
    if (ROOT / "experiments" / TABLES_SCRIPT).exists() and not any("FAILED" in m for _, m in summary):
        if ran or args.force or not (ROOT / TABLES_OUT).exists():
            t0 = time.time()
            rc = run(TABLES_SCRIPT, {})
            summary.append(("ieee tables", f"{'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f}s"))
        else:
            summary.append(("ieee tables", "skipped, nothing re-ran"))
    print("\n== run_all summary")
    for eid, msg in summary:
        print(f"  {eid.upper()}: {msg}")
    if any("FAILED" in m for _, m in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
