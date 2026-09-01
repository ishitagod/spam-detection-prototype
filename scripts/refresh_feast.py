"""
End-to-end Feast refresh: rebuild the per-sender snapshot, then `feast
apply` (pick up any schema/definition change) and `feast materialize`
(push the fresh snapshot into the online store). Run this after any
pipeline.py run that touches messages_with_behavioral.csv, or on a
recurring cadence in production - see feature_repo/definitions.py's
module docstring for the overall design and
features/behavioral_snapshot.py's for why "now" matters.

Uses `feast materialize <start> <end>` (an explicit range) rather than
`feast materialize-incremental`: incremental mode tracks its own
last-materialized-to watermark in the registry and refuses to move it
backwards, which breaks the moment `--now` is used to replay/demo against
this prototype's fixed-date historical sample (see
features/behavioral_snapshot.py's --now flag) - an explicit range has no
such state to get stuck.

Usage:
    python scripts/refresh_feast.py
    python scripts/refresh_feast.py --now 2026-08-03T23:59:59   # replay against the fixed-date sample data
"""
import argparse
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO_DIR = REPO_ROOT / "feature_repo"

sys.path.insert(0, str(REPO_ROOT))
from features.behavioral_snapshot import (  # noqa: E402
    DEFAULT_IMSI_SNAPSHOT_PATH,
    DEFAULT_MESSAGES_PATHS,
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_SS7_MESSAGES_PATH,
    run_behavioral_snapshot,
    run_imsi_snapshot,
)

# Feast CLI lives next to whichever Python this script is run with -
# venv\Scripts\feast.exe on Windows, venv/bin/feast on Unix - resolve it
# relative to sys.executable rather than assuming "feast" is on PATH.
_FEAST_EXE = Path(sys.executable).parent / ("feast.exe" if sys.platform == "win32" else "feast")


def _run_feast(args: list[str]) -> None:
    cmd = [str(_FEAST_EXE)] + args
    print(f"  $ {' '.join(cmd)}  (cwd={FEATURE_REPO_DIR})")
    subprocess.run(cmd, cwd=FEATURE_REPO_DIR, check=True)


def refresh(now: pd.Timestamp | None = None) -> None:
    now = now if now is not None else pd.Timestamp.now()

    print("=== 1/4: rebuild sender behavioral snapshot ===")
    run_behavioral_snapshot(DEFAULT_MESSAGES_PATHS, DEFAULT_SNAPSHOT_PATH, now=now)

    print("\n=== 2/4: rebuild IMSI behavioral snapshot (SS7-only) ===")
    run_imsi_snapshot(DEFAULT_SS7_MESSAGES_PATH, DEFAULT_IMSI_SNAPSHOT_PATH, now=now)

    print("\n=== 3/4: feast apply (registry) ===")
    _run_feast(["apply"])

    print("\n=== 4/4: feast materialize (online store) ===")
    # Wide, fixed start so a full re-snapshot always lands regardless of
    # what "now" is this run - this is a prototype-scale (~1k senders)
    # table rewritten in full each refresh, not an incremental stream, so
    # there's no cost to always covering the whole range.
    start = "2000-01-01T00:00:00"
    # datetime.timedelta, not pd.Timedelta - constructing a pd.Timedelta
    # here trips the same pandas==2.3.3/numpy==2.5.2 internal
    # "generic unit" DeprecationWarning documented in
    # features/behavioral.py's _window_timedelta64 - datetime.timedelta
    # adds to a pd.Timestamp just fine without it.
    end = (now + timedelta(days=1)).isoformat()
    _run_feast(["materialize", start, end])

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--now", type=str, default=None,
        help="Reference 'now' for the snapshot - see features/behavioral_snapshot.py's --now.",
    )
    args = parser.parse_args()
    now = pd.Timestamp(args.now) if args.now else None
    refresh(now=now)


if __name__ == "__main__":
    main()
