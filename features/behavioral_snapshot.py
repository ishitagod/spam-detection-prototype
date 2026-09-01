"""
Per-sender CURRENT-STATE snapshot, built for Feast to materialize into its
online store (feature_repo/definitions.py's `sender_behavioral_stats`
FeatureView) so inference-time lookups are a fast key-value read instead
of a fresh full-history scan per request.

DIFFERENT FROM features/behavioral.py: behavioral.py answers "as of just
BEFORE this message, what was this sender's history" - a value per ROW,
needed for point-in-time-correct training data (see that module's
docstring). This module answers "as of RIGHT NOW, what is each sender's
current state" - a single value PER SENDER, computed once and refreshed
by re-running this + `feast materialize` on a schedule. Same window
definitions and sender key (config/settings.py, source|originator) as
behavioral.py, but a genuinely simpler computation: there's no "row's
position in its own timeline" to track here, only "how much of this
sender's history falls within window of now" - so this is a plain
filter + groupby-aggregate, not behavioral.py's two-pointer sliding
window (that machinery exists specifically to make the per-ROW version
cheap; it isn't needed for a per-SENDER snapshot).

"NOW" IS WALL-CLOCK AT SNAPSHOT-BUILD TIME, NOT THE LAST MESSAGE'S
TIMESTAMP: a sender whose last real message was 3 days ago must show 0
messages in the trailing hour once this snapshot is rebuilt today, not
whatever count they had 3 days ago - passing `now` explicitly (rather
than deriving it from the data) keeps that ageing-out correct regardless
of how stale the input file is or how long since the last materialize
run.

REPEAT-CONTENT RATIO IS NOT PRECOMPUTED HERE: `sender_repeat_content_ratio_1hr`
needs the INCOMING message's text to compare against - that text doesn't
exist yet at snapshot-build time (it's the very thing about to be
scored). Instead this stores, per sender, `recent_text_counts_json` (the
top-K most frequent texts in the trailing long window, JSON-encoded
{text: count}) plus the existing `sender_msgs_last_1hr` as the ratio's
denominator - feature_repo/definitions.py's on-demand feature view
combines those two stored features with the request's candidate text at
serving time. Capping to top-K (not the full per-sender text set) is a
deliberate prototype approximation: a candidate that matches a prior
message outside the top-K undercounts to 0 instead of its true (small)
frequency - acceptable here because this ratio's job is catching
bulk-repeated blasts, which are exactly the high-frequency case top-K
always retains.

SENDER_AGE_DAYS: unlike the three window-based fields above, this is an
ALL-TIME aggregate (first-ever timestamp per sender, no window filter at
all) - "how long has this sender existed as of `now`". Mirrors
features/behavioral.py's point-in-time version exactly (same "first
message is its own reference point, gets 0.0" convention), just computed
once per sender instead of once per row.

IMSI_DISTINCT_ORIGINATORS_1HR: compute_imsi_snapshots() below is the
current-state counterpart to features/behavioral.py's training-time
(point-in-time, per-row) version of this SIM-farming signal - keyed on
`imsi`, not `sender_id`, a genuinely different entity (the physical SIM
vs. the apparent sender identity, see that module's IMSI-LINKAGE note).
Deliberately a SEPARATE function/output/entity from compute_sender_snapshots()
above, not a column bolted onto the sender_id-keyed schema - feature_repo/
definitions.py wires it to its own Feast entity (`imsi`) and FeatureView
(`imsi_behavioral_stats`). SS7-only: SMPP has no IMSI concept, and rows
with a null imsi are dropped entirely (not given a shared "<NA>" identity)
- same reasoning as the training-time version.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.behavioral import REQUIRED_COLS, _window_timedelta64
from config.settings import BEHAVIORAL_LONG_WINDOW, BEHAVIORAL_SHORT_WINDOW

REPO_ROOT = Path(__file__).resolve().parent.parent

# Single source of truth for where the snapshot lives: this module is the
# WRITER (run_behavioral_snapshot below), feature_repo/definitions.py (the
# Feast FileSource declaration) and scripts/refresh_feast.py (the refresh
# driver) both import DEFAULT_SNAPSHOT_PATH from here rather than
# hardcoding the path a second/third time - three independently-edited
# copies of the same path is exactly how a Feast FileSource silently
# points at a stale or nonexistent file.
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "data" / "processed" / "feast_sources" / "sender_behavioral_snapshot.parquet"
DEFAULT_MESSAGES_PATHS = [
    REPO_ROOT / "data" / "processed" / "SMPP" / "messages_with_behavioral.csv",
    REPO_ROOT / "data" / "processed" / "SS7" / "messages_with_behavioral.csv",
]
# IMSI snapshot is SS7-only (see module docstring) - one messages file,
# not a list, unlike DEFAULT_MESSAGES_PATHS above.
DEFAULT_IMSI_SNAPSHOT_PATH = REPO_ROOT / "data" / "processed" / "feast_sources" / "imsi_behavioral_snapshot.parquet"
DEFAULT_SS7_MESSAGES_PATH = REPO_ROOT / "data" / "processed" / "SS7" / "messages_with_behavioral.csv"

DEFAULT_TOP_K_TEXTS = 20

_SHORT_WINDOW = _window_timedelta64(BEHAVIORAL_SHORT_WINDOW)
_LONG_WINDOW = _window_timedelta64(BEHAVIORAL_LONG_WINDOW)

COL_SENDER_ID = "sender_id"
COL_EVENT_TS = "event_timestamp"
COL_MSGS_SHORT = "sender_msgs_last_5min"
COL_MSGS_LONG = "sender_msgs_last_1hr"
COL_UNIQUE_DEST_LONG = "sender_unique_destinations_1hr"
COL_RECENT_TEXTS_JSON = "recent_text_counts_json"
COL_SENDER_AGE_DAYS = "sender_age_days"

SNAPSHOT_COLUMNS = [
    COL_SENDER_ID, "source", "originator", COL_EVENT_TS,
    COL_MSGS_SHORT, COL_MSGS_LONG, COL_UNIQUE_DEST_LONG, COL_RECENT_TEXTS_JSON,
    COL_SENDER_AGE_DAYS,
]

COL_IMSI = "imsi"
COL_IMSI_DISTINCT_ORIG_LONG = "imsi_distinct_originators_1hr"
IMSI_SNAPSHOT_COLUMNS = [COL_IMSI, COL_EVENT_TS, COL_IMSI_DISTINCT_ORIG_LONG]


def compute_sender_snapshots(
    messages: pd.DataFrame,
    now: pd.Timestamp,
    top_k_texts: int = DEFAULT_TOP_K_TEXTS,
) -> pd.DataFrame:
    """
    One row per distinct (source, originator) sender seen anywhere in
    `messages` (even senders with zero activity inside either window still
    get a row, all-zero/empty - Feast should be able to serve a defined
    "no recent activity" state for every known sender, not just busy
    ones). `now` is the reference point every window is measured back
    from - always pass it explicitly (see module docstring).
    """
    missing = [c for c in REQUIRED_COLS if c not in messages.columns]
    if missing:
        raise ValueError(f"messages is missing required column(s): {missing}")

    df = messages.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    # Cast source/originator to str up front (not just when building the
    # compound key): SMPP originators are pure-text sender IDs but SS7
    # originators are MSISDNs that pandas can infer as int64 in one
    # source's CSV and str in another - concatenating both source frames
    # before this cast produces a mixed str/int "object" column that
    # pyarrow's parquet writer rejects outright (ArrowTypeError). Casting
    # here keeps `originator` uniformly str everywhere downstream,
    # including the snapshot's own output column.
    df["source"] = df["source"].astype(str)
    df["originator"] = df["originator"].fillna("<NA>").astype(str)
    df[COL_SENDER_ID] = df["source"] + "|" + df["originator"]
    df["destination"] = df["destination"].fillna("<NA>").astype(str)
    df["text"] = df["text"].fillna("")

    now64 = np.datetime64(now)
    age = now64 - df["timestamp"].to_numpy()
    zero = np.timedelta64(0, "ns")
    # closed-left/open-right relative to "now", matching behavioral.py: a
    # message exactly `window` old still counts; a message with negative
    # age (timestamp after `now`, e.g. clock skew in the source data) is
    # not yet history and is excluded from both windows.
    in_short = (age >= zero) & (age <= _SHORT_WINDOW)
    in_long = (age >= zero) & (age <= _LONG_WINDOW)

    all_senders = df[[COL_SENDER_ID, "source", "originator"]].drop_duplicates(
        subset=COL_SENDER_ID
    )

    msgs_short = df[in_short].groupby(COL_SENDER_ID).size()
    long_df = df[in_long]
    long_grouped = long_df.groupby(COL_SENDER_ID)
    msgs_long = long_grouped.size()
    uniq_dest_long = long_grouped["destination"].nunique()

    def _top_k_json(texts: pd.Series) -> str:
        counts = texts.value_counts().head(top_k_texts)
        return json.dumps({str(t): int(c) for t, c in counts.items()})

    recent_texts_json = long_grouped["text"].apply(_top_k_json)

    # ALL-TIME (no window filter) - every sender in `all_senders` has at
    # least one row by construction, so this can never be NaN the way the
    # window-based aggregates above can. See module docstring's
    # SENDER_AGE_DAYS note.
    first_seen_all_time = df.groupby(COL_SENDER_ID)["timestamp"].min()

    snapshot = all_senders.set_index(COL_SENDER_ID)
    snapshot[COL_MSGS_SHORT] = msgs_short
    snapshot[COL_MSGS_LONG] = msgs_long
    snapshot[COL_UNIQUE_DEST_LONG] = uniq_dest_long
    snapshot[COL_RECENT_TEXTS_JSON] = recent_texts_json
    snapshot[COL_SENDER_AGE_DAYS] = first_seen_all_time
    snapshot[COL_MSGS_SHORT] = snapshot[COL_MSGS_SHORT].fillna(0).astype(np.int64)
    snapshot[COL_MSGS_LONG] = snapshot[COL_MSGS_LONG].fillna(0).astype(np.int64)
    snapshot[COL_UNIQUE_DEST_LONG] = snapshot[COL_UNIQUE_DEST_LONG].fillna(0).astype(np.int64)
    snapshot[COL_RECENT_TEXTS_JSON] = snapshot[COL_RECENT_TEXTS_JSON].fillna("{}")
    # clip(lower=0): a sender whose only message(s) are all after `now`
    # (clock skew, same edge case the window features guard against via
    # their age>=0 check) would otherwise get a nonsensical negative age.
    snapshot[COL_SENDER_AGE_DAYS] = (
        (now64 - snapshot[COL_SENDER_AGE_DAYS].to_numpy()) / np.timedelta64(1, "D")
    ).clip(min=0)
    snapshot[COL_EVENT_TS] = now

    return snapshot.reset_index()[SNAPSHOT_COLUMNS]


def run_behavioral_snapshot(
    messages_paths: list[Path],
    out_path: Path,
    now: pd.Timestamp | None = None,
    top_k_texts: int = DEFAULT_TOP_K_TEXTS,
) -> pd.DataFrame:
    """
    Reads one messages(_with_behavioral).csv per source, concatenates, and
    writes ONE combined snapshot parquet - the sender_id key already
    encodes source, so SMPP and SS7 senders share one Feast entity/table
    without collision risk (see features/behavioral.py's sender-key note).
    """
    now = now if now is not None else pd.Timestamp.now()

    frames = []
    for p in messages_paths:
        p = Path(p)
        if not p.exists():
            print(f"  (skipping, not found) {p}")
            continue
        print(f"  loading {p} ...")
        frames.append(pd.read_csv(p, low_memory=False, usecols=lambda c: c in {
            "source", "originator", "destination", "timestamp", "text",
        }))

    if not frames:
        print("No messages files found - nothing to snapshot.")
        return pd.DataFrame()

    messages = pd.concat(frames, ignore_index=True)
    print(f"Computing sender snapshots for {len(messages)} message(s) as of {now} ...")
    snapshot = compute_sender_snapshots(messages, now=now, top_k_texts=top_k_texts)
    print(f"  {len(snapshot)} distinct sender(s)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(out_path, index=False)
    print(f"Wrote {len(snapshot)} rows to {out_path}")
    return snapshot


def compute_imsi_snapshots(messages: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """
    One row per distinct non-null `imsi` seen in `messages` (SS7 only -
    see module docstring), each showing the count of DISTINCT originators
    seen behind that imsi in the trailing 1hr as of `now` - the
    current-state counterpart to features/behavioral.py's per-row
    imsi_distinct_originators_1hr, same "current state, not per-row
    history" reframing as compute_sender_snapshots() above. Rows whose OWN
    imsi is null are dropped entirely, not given a shared "<NA>" identity
    - a null imsi means the physical SIM behind that message genuinely
    isn't known, matching the training-time version's exact treatment
    (features/behavioral.py's IMSI-LINKAGE note).
    """
    df = messages[messages[COL_IMSI].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=IMSI_SNAPSHOT_COLUMNS)

    df[COL_IMSI] = df[COL_IMSI].astype(str)
    df["originator"] = df["originator"].fillna("<NA>").astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    now64 = np.datetime64(now)
    age = now64 - df["timestamp"].to_numpy()
    zero = np.timedelta64(0, "ns")
    # Same closed-left/open-right window as compute_sender_snapshots().
    in_long = (age >= zero) & (age <= _LONG_WINDOW)

    all_imsis = df[[COL_IMSI]].drop_duplicates()
    distinct_orig = df[in_long].groupby(COL_IMSI)["originator"].nunique()

    snapshot = all_imsis.set_index(COL_IMSI)
    snapshot[COL_IMSI_DISTINCT_ORIG_LONG] = distinct_orig
    snapshot[COL_IMSI_DISTINCT_ORIG_LONG] = (
        snapshot[COL_IMSI_DISTINCT_ORIG_LONG].fillna(0).astype(np.int64)
    )
    snapshot[COL_EVENT_TS] = now

    return snapshot.reset_index()[IMSI_SNAPSHOT_COLUMNS]


def run_imsi_snapshot(
    messages_path: Path, out_path: Path, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """SS7-only counterpart to run_behavioral_snapshot() above - one
    messages file in, one imsi-keyed snapshot parquet out."""
    now = now if now is not None else pd.Timestamp.now()

    messages_path = Path(messages_path)
    if not messages_path.exists():
        print(f"  (skipping, not found) {messages_path}")
        return pd.DataFrame()

    print(f"  loading {messages_path} ...")
    messages = pd.read_csv(
        messages_path, low_memory=False,
        usecols=lambda c: c in {"imsi", "originator", "timestamp"},
    )
    print(f"Computing IMSI snapshots for {len(messages)} row(s) as of {now} ...")
    snapshot = compute_imsi_snapshots(messages, now=now)
    print(f"  {len(snapshot)} distinct imsi(s)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(out_path, index=False)
    print(f"Wrote {len(snapshot)} rows to {out_path}")
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--messages_paths", type=str, nargs="+",
        default=[str(p) for p in DEFAULT_MESSAGES_PATHS],
    )
    parser.add_argument(
        "--out_path", type=str, default=str(DEFAULT_SNAPSHOT_PATH),
    )
    parser.add_argument(
        "--ss7_messages_path", type=str, default=str(DEFAULT_SS7_MESSAGES_PATH),
        help="Input for the IMSI snapshot (SS7-only, see module docstring).",
    )
    parser.add_argument(
        "--imsi_out_path", type=str, default=str(DEFAULT_IMSI_SNAPSHOT_PATH),
    )
    parser.add_argument(
        "--now", type=str, default=None,
        help=(
            "Reference 'now' timestamp (ISO format) every window is measured "
            "back from. Defaults to actual wall-clock time - the correct "
            "choice once this runs on a live refresh cadence against live "
            "data. Override this only to replay/demo against this "
            "prototype's fixed-date historical CDR sample, where real "
            "wall-clock 'now' is weeks past the data and would correctly "
            "but unhelpfully zero out every window."
        ),
    )
    args = parser.parse_args()
    now = pd.Timestamp(args.now) if args.now else None
    run_behavioral_snapshot([Path(p) for p in args.messages_paths], Path(args.out_path), now=now)
    run_imsi_snapshot(Path(args.ss7_messages_path), Path(args.imsi_out_path), now=now)


if __name__ == "__main__":
    main()
