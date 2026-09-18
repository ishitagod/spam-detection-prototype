"""
Reports whether a source's raw data has grown enough since it was last
trained/clustered to be worth re-running that stage - a scheduled CHECK,
not a continuous/live daemon (CLAUDE.md's "Next" notes point at Airflow
later; this composes into cron/Task Scheduler today via its exit code).

TWO INDEPENDENT STAGES, two independent thresholds (config/settings.py):
  --stage retrain  compares against RETRAIN_MIN_NEW_ROWS - the expensive
                    trigger (full pipeline + LightGBM/Isolation Forest
                    retrain + compare_versions.py promotion gate).
  --stage dbscan   compares against DBSCAN_MIN_NEW_ANOMALY_ROWS - much
                    cheaper (DBSCAN re-clusters frozen pretrained MiniLM
                    embeddings, no retrain needed - see the architecture
                    plan's Section 1), so this can fire far more often.
Never conflated into one threshold - see the architecture plan's Section 7
for why these two are genuinely decoupled cadences.

STATE: a small JSON manifest per (source, stage) - data/processed/<source>/
last_trained_manifest.json (retrain) / last_dbscan_manifest.json (dbscan) -
recording {row_count, max_timestamp} as of the last time that stage
actually ran. No manifest yet = "always due" (a stage that's never run is
trivially due to run). `--mark_trained` updates the manifest to the
CURRENT state - call this from the scheduler/operator AFTER the real
retrain/dbscan-run actually completed, not automatically here; this script
only checks and reports, it never triggers training itself
(CLAUDE.md: "Champion/challenger promotion is explicit in code" - the same
explicitness applies to triggering the run that produces a challenger).

EXIT CODE: 0 = due, 1 = not due - composes into a scheduler's own
conditional (`if check_retrain_trigger.py --stage retrain; then ...`)
without extra glue.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from config.settings import DBSCAN_MIN_NEW_ANOMALY_ROWS, RETRAIN_MIN_NEW_ROWS

STAGE_THRESHOLDS = {
    "retrain": RETRAIN_MIN_NEW_ROWS,
    "dbscan": DBSCAN_MIN_NEW_ANOMALY_ROWS,
}
STAGE_MANIFEST_NAMES = {
    "retrain": "last_trained_manifest.json",
    "dbscan": "last_dbscan_manifest.json",
}


def _manifest_path(source_dir: Path, stage: str) -> Path:
    return source_dir / STAGE_MANIFEST_NAMES[stage]


def _current_state(source_dir: Path) -> dict:
    messages_path = source_dir / "messages_with_behavioral.csv"
    if not messages_path.exists():
        raise FileNotFoundError(f"No messages_with_behavioral.csv found in {source_dir}")
    df = pd.read_csv(messages_path, low_memory=False, usecols=["timestamp"])
    return {
        "row_count": int(len(df)),
        "max_timestamp": str(pd.to_datetime(df["timestamp"], format="mixed").max()),
    }


def check_due(source_dir: Path, stage: str) -> tuple[bool, dict]:
    """Returns (is_due, current_state). is_due is True when no manifest
    exists yet, or current_state's row_count has grown by at least
    STAGE_THRESHOLDS[stage] rows since the manifest was last written."""
    manifest_path = _manifest_path(source_dir, stage)
    current = _current_state(source_dir)

    if not manifest_path.exists():
        return True, current

    last = json.loads(manifest_path.read_text())
    new_rows = current["row_count"] - last.get("row_count", 0)
    return new_rows >= STAGE_THRESHOLDS[stage], current


def mark_trained(source_dir: Path, stage: str, current: dict) -> None:
    manifest_path = _manifest_path(source_dir, stage)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2))
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="SMPP or SS7")
    parser.add_argument("--stage", type=str, required=True, choices=sorted(STAGE_THRESHOLDS))
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--mark_trained", action="store_true",
        help="Update the manifest to the current state - call AFTER the "
        "real retrain/dbscan-run for this stage actually completed, not "
        "as a substitute for running it.",
    )
    args = parser.parse_args()

    source_dir = Path(args.data_dir) / args.source
    due, current = check_due(source_dir, args.stage)

    threshold = STAGE_THRESHOLDS[args.stage]
    print(f"source={args.source} stage={args.stage} rows={current['row_count']} "
          f"max_timestamp={current['max_timestamp']} threshold={threshold}")
    print("DUE" if due else "NOT DUE")

    if args.mark_trained:
        mark_trained(source_dir, args.stage, current)

    raise SystemExit(0 if due else 1)


if __name__ == "__main__":
    main()
