"""
Sender-behavioral (velocity/repeat-content) features, computed per message
from features.message_reassembly's OUTPUT (one row per logical message),
never from raw per-part ingestion features - see message_reassembly.py's
module docstring for why that ordering matters: a multipart message must
count as ONE message in a velocity window, not N (one per physical part).

Features produced (names/windows fixed by config/settings.py -
BEHAVIORAL_VERY_SHORT_WINDOW / BEHAVIORAL_SHORT_WINDOW / BEHAVIORAL_LONG_WINDOW):
  sender_msgs_last_1min             - # messages this sender sent in the
                                       trailing 1 minute (flooding signal -
                                       see VELOCITY note below)
  sender_msgs_last_5min             - # messages this sender sent in the
                                       trailing 5 minutes
  sender_msgs_last_1hr              - # messages this sender sent in the
                                       trailing 1 hour
  sender_unique_destinations_5min   - # distinct destinations this sender
                                       messaged in the trailing 5 minutes
  sender_unique_destinations_1hr    - # distinct destinations this sender
                                       messaged in the trailing 1 hour
  sender_recipient_diversity_ratio_5min / _1hr
                                     - unique destinations / message count,
                                       same window (0.0 when the window has
                                       no prior messages - see POINT-IN-TIME
                                       below). New destinations every time
                                       -> broad/scattered traffic; near-1.0
                                       repeatedly -> either normal diverse
                                       real-user traffic or a flooding blast
                                       hitting fresh numbers each time - this
                                       ratio alone can't tell those apart,
                                       it's a component feature, not a verdict.
  sender_repeat_content_ratio_1hr   - of this sender's messages in the
                                       trailing 1 hour, what fraction had
                                       the SAME text as the current message
                                       (0.0 when there's no prior history -
                                       see POINT-IN-TIME below)
  sender_velocity_zscore_5min       - see VELOCITY note below
  sender_age_days                   - days (float) since this sender's very
                                       first-ever message, as of the current
                                       one. NOT windowed and NOT strictly-
                                       before like the features above: a
                                       sender's own first message IS its own
                                       reference point, so it correctly
                                       gets 0.0 (honest cold-start, not
                                       excluded to NaN/missing).
  imsi_distinct_originators_1hr     - SS7 ONLY. See IMSI-LINKAGE below.

VELOCITY (sender_velocity_zscore_5min): how anomalous THIS message's
sender_msgs_last_5min reading is relative to what's typical for THIS
SAME sender, not a global/cross-sender baseline - a naturally bursty
enterprise sender_id and a naturally quiet one need different "normal"
baselines, so comparing either to a fixed global threshold would
misjudge one of them. Computed as (msgs_last_5min - running_mean) /
running_stdev, where running_mean/running_stdev are Welford's-algorithm
running statistics over THIS sender's own PRIOR msgs_last_5min readings
only (point-in-time, same discipline as everything else here) - each
reading is sampled once per message this sender sends (event-sampled),
not on fixed wall-clock 5-minute buckets, a deliberate simplification
that avoids needing to backfill buckets with zero messages.
EXPANDING (all-time-to-date), not a bounded rolling window - simpler,
avoids a second window-size parameter to pick with no data yet to justify
one; revisit only if evidence shows a sender's very old history should
stop counting toward "typical" for them.
NaN, not 0.0 or a fabricated value, when there isn't yet a meaningful
baseline to compare against: fewer than 2 prior readings (stdev
undefined), or the prior readings have zero variance (division by zero -
notably this includes the "always quiet, one sudden burst" case, which a
zero-variance guard can't safely distinguish from "no information yet"
without inventing an uncalibrated magic value - same "disclose, don't
fake certainty" principle as sender_age_days's cold start, applied to a
case where a fabricated large number would be actively misleading rather
than just uninformative).

SENDER KEY: (source, originator), not originator alone. SMPP originators
are business sender-ID strings, SS7 originators are real MSISDNs - two
different namespaces that could coincidentally collide on the same string,
and `source` is already a canonical, always-present column, so keying on
both costs nothing and removes a real (if unlikely) cross-source leakage
risk.

IMSI-LINKAGE (imsi_distinct_originators_1hr): the SIM-farming signal
CLAUDE.md's next-steps section flagged as not-yet-built - one physical
IMSI cycling through many apparent MSISDNs. Deliberately keyed on `imsi`,
a genuinely different entity from the sender key above: `source`+
`originator` identifies the apparent sender identity a message claims,
`imsi` identifies the physical SIM behind it - the whole point of this
feature is catching cases where those two diverge. Only computed when the
input has an `imsi` column at all (SS7's raw data - SMPP has no IMSI
concept); when absent, the column is left OFF the output entirely, not
NaN-filled - same convention as SS7's `message_type` being its own
source-specific column rather than a shared field forced onto SMPP rows
that have no equivalent. Same point-in-time discipline as the sender-keyed
features: counts distinct originators seen behind this IMSI in the
trailing 1hr, using only strictly-earlier rows - mirrors
sender_unique_destinations_1hr's exact window/mechanism, just swapping
the (sender -> distinct destination) relationship for (imsi -> distinct
originator). A row whose OWN imsi is null gets NA here (nullable Int64),
not 0 and not lumped into a shared fake identity - checked against real
data: 31.7% of a real SS7 sample has a null imsi, so treating them as one
shared "<NA>" sender would fabricate a high-variance signal out of rows
where the identity this feature measures isn't actually known. NOTE: this
is a training-time (point-in-time, per-row)
feature only so far - a live-serving version needs its own Feast entity
(imsi, not sender_id) and FeatureView, not yet built; see
features/behavioral_snapshot.py's docstring.

POINT-IN-TIME CORRECTNESS (see CLAUDE.md - this caused real bugs in the
earlier fraud-detection project's schema, worth restating every time a
behavioral feature is added): every feature here is computed from messages
STRICTLY BEFORE the current one (same sender, timestamp < current
timestamp) - the current message is never counted as its own history. A
sender's first-ever message therefore gets all-zero behavioral features
(0 prior messages, 0 unique destinations, 0.0 repeat ratio) - this is the
correct cold-start state, not a bug to paper over (same "disclose, don't
fake certainty" principle as the architecture plan's `confidence` field).
All three windows (1-minute, 5-minute, 1-hour) are closed-left/open-right
relative to "now": a prior message exactly `window` old (t_now - t_prior
== window) IS still counted; anything strictly older is not.

ALGORITHM / PERFORMANCE: this is NOT a simple groupby-aggregate the way
message_reassembly.py's multipart stats are (those collapse a whole group
to one row; this needs a value PER ROW that depends on that row's
position in its sender's timeline) and pandas has no built-in vectorized
"rolling nunique" or "rolling value-match-ratio". A naive per-row rescan
(for each message, filter all of that sender's messages within the window)
is O(n^2) per sender - checked against the real ingested SMPP data
(data/processed/SMPP/messages.csv): only 107 distinct senders, but the
busiest carries 756,612 messages alone (median 2,433) - an O(n^2) scan on
that one sender would be ~5.7*10^11 comparisons, never finishing. Instead:
sort once by (source, originator, timestamp), then for each sender run a
single forward two-pointer sliding window (one pointer per window size)
over that sender's already-sorted slice, maintaining running Counters of
destination/text seen in the current 1-hour window. Each pointer only ever
moves forward, so total work is O(n) per sender / O(N) overall regardless
of how skewed the per-sender message counts are - the property that
actually matters here, not raw group count.
"""
import argparse
import warnings
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from config.settings import (
    BEHAVIORAL_LONG_WINDOW,
    BEHAVIORAL_SHORT_WINDOW,
    BEHAVIORAL_VERY_SHORT_WINDOW,
)

REQUIRED_COLS = ["source", "originator", "destination", "timestamp", "text"]


def _window_timedelta64(spec: str) -> np.timedelta64:
    """
    Parses a pandas Timedelta literal ("5min", "1h", ...) into an explicit
    timedelta64[ns] - matching the datetime64[ns] dtype the timestamp
    arrays below use, so window comparisons never hit numpy's ambiguous
    "generic unit" timedelta path.

    The warnings.catch_warnings() suppression here is narrowly scoped to
    pd.Timedelta's OWN construction, not this module's logic: on this
    project's pinned pandas==2.3.3 / numpy==2.5.2 (requirements.txt),
    merely constructing a pd.Timedelta from a string emits numpy's
    DeprecationWarning about generic-unit timedelta64 internally - verified
    by isolating it down to the bare `pd.Timedelta("5min")` call with no
    further operations, so it's a pandas/numpy version-compat detail, not
    something this code can address by using a different accessor.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ns = pd.Timedelta(spec) // pd.Timedelta(1, "ns")
    return np.timedelta64(ns, "ns")


_VERY_SHORT_WINDOW = _window_timedelta64(BEHAVIORAL_VERY_SHORT_WINDOW)
_SHORT_WINDOW = _window_timedelta64(BEHAVIORAL_SHORT_WINDOW)
_LONG_WINDOW = _window_timedelta64(BEHAVIORAL_LONG_WINDOW)

# Below this, a sender's own historical msgs_last_5min readings are treated
# as having no real variance (all effectively identical) - see module
# docstring's VELOCITY note on why that's NaN, not a divide-by-zero crash
# or a fabricated large number.
_VELOCITY_ZSCORE_STDEV_EPSILON = 1e-9

# Column names are the exact ones CLAUDE.md/config/settings.py already
# name (sender_msgs_last_5min, sender_msgs_last_1hr, ...) - spelled out
# explicitly here rather than derived from BEHAVIORAL_*_WINDOW's Timedelta
# strings (e.g. via string-replacing "h" -> "hr"), since that string is a
# pandas Timedelta literal ("1h", "5min") governing duration, not a naming
# convention - conflating the two would silently rename every output
# column the moment someone tunes a window value in settings.py.
COL_MSGS_VERY_SHORT = "sender_msgs_last_1min"
COL_MSGS_SHORT = "sender_msgs_last_5min"
COL_MSGS_LONG = "sender_msgs_last_1hr"
COL_UNIQUE_DEST_SHORT = "sender_unique_destinations_5min"
COL_UNIQUE_DEST_LONG = "sender_unique_destinations_1hr"
COL_RECIPIENT_DIVERSITY_SHORT = "sender_recipient_diversity_ratio_5min"
COL_RECIPIENT_DIVERSITY_LONG = "sender_recipient_diversity_ratio_1hr"
COL_REPEAT_RATIO_LONG = "sender_repeat_content_ratio_1hr"
COL_VELOCITY_ZSCORE_SHORT = "sender_velocity_zscore_5min"
COL_SENDER_AGE_DAYS = "sender_age_days"
COL_IMSI_DISTINCT_ORIG_LONG = "imsi_distinct_originators_1hr"


class _WindowFeatures(NamedTuple):
    """Named, not a plain tuple - _sliding_window_features grew past the
    point where positional unpacking stays readable."""
    msgs_very_short: np.ndarray
    msgs_short: np.ndarray
    msgs_long: np.ndarray
    uniq_dest_short: np.ndarray
    uniq_dest_long: np.ndarray
    repeat_ratio_long: np.ndarray
    velocity_zscore_short: np.ndarray


def _sliding_window_features(
    ts: np.ndarray, dest: np.ndarray, text: np.ndarray
) -> _WindowFeatures:
    """
    Single sender's messages, already sorted ascending by `ts`. Returns a
    _WindowFeatures of one array per feature, one value per row, computed
    using only strictly-earlier rows (see module docstring's POINT-IN-TIME
    note). Three independent eviction pointers (1min/5min/1hr), each
    forward-only - same O(n)-per-sender argument as the module docstring's
    ALGORITHM note, just one more pointer than before.
    """
    n = len(ts)
    msgs_very_short = np.zeros(n, dtype=np.int64)
    msgs_short = np.zeros(n, dtype=np.int64)
    msgs_long = np.zeros(n, dtype=np.int64)
    uniq_dest_short = np.zeros(n, dtype=np.int64)
    uniq_dest_long = np.zeros(n, dtype=np.int64)
    repeat_ratio_long = np.zeros(n, dtype=np.float64)
    velocity_zscore_short = np.full(n, np.nan, dtype=np.float64)

    left_very_short = 0
    left_short = 0
    left_long = 0
    short_dest_counts: Counter = Counter()
    long_dest_counts: Counter = Counter()
    text_counts: Counter = Counter()

    # Welford's online algorithm for this sender's own running mean/
    # variance of msgs_last_5min readings - see module docstring's
    # VELOCITY note.
    baseline_count = 0
    baseline_mean = 0.0
    baseline_m2 = 0.0

    for i in range(n):
        t = ts[i]

        # Drop entries older than the long window from the long-window
        # counters before reading this row's features.
        while left_long < i and t - ts[left_long] > _LONG_WINDOW:
            old_dest, old_text = dest[left_long], text[left_long]
            long_dest_counts[old_dest] -= 1
            if long_dest_counts[old_dest] == 0:
                del long_dest_counts[old_dest]
            text_counts[old_text] -= 1
            if text_counts[old_text] == 0:
                del text_counts[old_text]
            left_long += 1

        # Short window needs its OWN destination counter, separate from the
        # long window's - it evicts on a different (faster) schedule, so it
        # can't just read a subset of long_dest_counts.
        while left_short < i and t - ts[left_short] > _SHORT_WINDOW:
            old_dest = dest[left_short]
            short_dest_counts[old_dest] -= 1
            if short_dest_counts[old_dest] == 0:
                del short_dest_counts[old_dest]
            left_short += 1

        # Very-short window shares the same forward-only pointer discipline
        # but needs no counters - it's a pure count of prior messages.
        while left_very_short < i and t - ts[left_very_short] > _VERY_SHORT_WINDOW:
            left_very_short += 1

        prior_long = i - left_long
        prior_short = i - left_short
        msgs_very_short[i] = i - left_very_short
        msgs_short[i] = prior_short
        msgs_long[i] = prior_long
        uniq_dest_short[i] = len(short_dest_counts)
        uniq_dest_long[i] = len(long_dest_counts)
        repeat_ratio_long[i] = (
            text_counts.get(text[i], 0) / prior_long if prior_long > 0 else 0.0
        )

        # velocity_zscore: compare THIS row's prior_short reading against
        # the running baseline built from EARLIER rows only (baseline is
        # updated below, after this). Left as NaN (see array init) unless
        # there are >=2 prior readings AND they show real variance.
        if baseline_count >= 2:
            baseline_variance = baseline_m2 / (baseline_count - 1)
            baseline_stdev = baseline_variance ** 0.5
            if baseline_stdev > _VELOCITY_ZSCORE_STDEV_EPSILON:
                velocity_zscore_short[i] = (
                    prior_short - baseline_mean
                ) / baseline_stdev

        # Only now does row i become part of its own sender's future
        # history - see POINT-IN-TIME note.
        short_dest_counts[dest[i]] += 1
        long_dest_counts[dest[i]] += 1
        text_counts[text[i]] += 1

        # Feed this row's own msgs_last_5min reading (prior_short - same
        # value as msgs_short[i]) into the running baseline for future
        # rows' z-scores.
        baseline_count += 1
        delta = prior_short - baseline_mean
        baseline_mean += delta / baseline_count
        baseline_m2 += delta * (prior_short - baseline_mean)

    return _WindowFeatures(
        msgs_very_short, msgs_short, msgs_long,
        uniq_dest_short, uniq_dest_long, repeat_ratio_long,
        velocity_zscore_short,
    )


def _imsi_window_originator_counts(ts: np.ndarray, originator: np.ndarray) -> np.ndarray:
    """
    Single IMSI's rows, already sorted ascending by `ts`. Returns, per row,
    the count of DISTINCT originators (apparent MSISDNs) this IMSI was seen
    behind in the trailing long window, using only strictly-earlier rows -
    same two-pointer/point-in-time discipline as _sliding_window_features
    above, just a single window/single counter (see module docstring's
    IMSI-LINKAGE note for why this is a separate, simpler function rather
    than a third case bolted onto _sliding_window_features).
    """
    n = len(ts)
    out = np.zeros(n, dtype=np.int64)
    left = 0
    counts: Counter = Counter()
    for i in range(n):
        t = ts[i]
        while left < i and t - ts[left] > _LONG_WINDOW:
            old = originator[left]
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
            left += 1
        out[i] = len(counts)
        # Only now does row i become part of its own IMSI's future history.
        counts[originator[i]] += 1
    return out


def compute_behavioral_features(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Returns `messages` with the sender-behavioral columns added (see module
    docstring for the full list), in the SAME row order as the input. Does
    not modify `messages` in place.
    """
    missing = [c for c in REQUIRED_COLS if c not in messages.columns]
    if missing:
        raise ValueError(f"messages is missing required column(s): {missing}")

    df = messages.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # NaN originator/destination/text are real possibilities in raw data
    # (see ingestion/*.py) - stringify with an explicit placeholder rather
    # than letting NaN collapse into pandas' float NaN-as-a-group-key
    # behavior (which silently groups all NaN senders together). This is a
    # known, accepted approximation for the rare NaN case, not a claim that
    # every genuinely-missing sender/destination is the same sender.
    sender_key = (
        df["source"].astype(str) + "|" + df["originator"].fillna("<NA>").astype(str)
    )
    dest_key = df["destination"].fillna("<NA>").astype(str)
    text_key = df["text"].fillna("")

    sort_cols = pd.DataFrame({"sender": sender_key, "ts": df["timestamp"]})
    # mergesort (stable) so rows with identical (sender, timestamp) keep
    # their original relative order - deterministic re-runs.
    order = sort_cols.sort_values(["sender", "ts"], kind="mergesort").index.to_numpy()

    sender_sorted = sender_key.to_numpy()[order]
    ts_sorted = df["timestamp"].to_numpy()[order]
    dest_sorted = dest_key.to_numpy()[order]
    text_sorted = text_key.to_numpy()[order]

    n = len(df)
    msgs_very_short = np.zeros(n, dtype=np.int64)
    msgs_short = np.zeros(n, dtype=np.int64)
    msgs_long = np.zeros(n, dtype=np.int64)
    uniq_dest_short = np.zeros(n, dtype=np.int64)
    uniq_dest_long = np.zeros(n, dtype=np.int64)
    repeat_ratio_long = np.zeros(n, dtype=np.float64)
    velocity_zscore_short = np.full(n, np.nan, dtype=np.float64)

    # Sorted by sender first, so every sender's rows are one contiguous
    # run - find each run's [start, end) bounds and process independently.
    group_change = np.empty(n, dtype=bool)
    group_change[0] = True
    if n > 1:
        group_change[1:] = sender_sorted[1:] != sender_sorted[:-1]
    group_starts = np.flatnonzero(group_change)
    group_ends = np.append(group_starts[1:], n)

    print(f"  behavioral: {len(group_starts)} distinct sender(s), {n} message(s)")
    for start, end in zip(group_starts, group_ends):
        wf = _sliding_window_features(
            ts_sorted[start:end], dest_sorted[start:end], text_sorted[start:end]
        )
        msgs_very_short[start:end] = wf.msgs_very_short
        msgs_short[start:end] = wf.msgs_short
        msgs_long[start:end] = wf.msgs_long
        uniq_dest_short[start:end] = wf.uniq_dest_short
        uniq_dest_long[start:end] = wf.uniq_dest_long
        repeat_ratio_long[start:end] = wf.repeat_ratio_long
        velocity_zscore_short[start:end] = wf.velocity_zscore_short

    # Scatter back into original row order.
    out = df.copy()
    out.loc[order, COL_MSGS_VERY_SHORT] = msgs_very_short
    out.loc[order, COL_MSGS_SHORT] = msgs_short
    out.loc[order, COL_MSGS_LONG] = msgs_long
    out.loc[order, COL_UNIQUE_DEST_SHORT] = uniq_dest_short
    out.loc[order, COL_UNIQUE_DEST_LONG] = uniq_dest_long
    out.loc[order, COL_REPEAT_RATIO_LONG] = repeat_ratio_long
    out.loc[order, COL_VELOCITY_ZSCORE_SHORT] = velocity_zscore_short
    out[COL_MSGS_VERY_SHORT] = out[COL_MSGS_VERY_SHORT].astype(np.int64)
    out[COL_MSGS_SHORT] = out[COL_MSGS_SHORT].astype(np.int64)
    out[COL_MSGS_LONG] = out[COL_MSGS_LONG].astype(np.int64)
    out[COL_UNIQUE_DEST_SHORT] = out[COL_UNIQUE_DEST_SHORT].astype(np.int64)
    out[COL_UNIQUE_DEST_LONG] = out[COL_UNIQUE_DEST_LONG].astype(np.int64)

    # recipient_diversity_ratio: unique destinations / message count, same
    # window - 0.0 when the window has no prior messages (same cold-start
    # convention as repeat_content_ratio, not NaN/undefined - "no history"
    # and "history but zero diversity" are different states elsewhere in
    # this module but degenerate to the same 0.0 here since 0 messages
    # trivially has 0 unique destinations too).
    #
    # Uses out[...] (already scattered back to ORIGINAL row order above),
    # not the msgs_short/uniq_dest_short local arrays - those are still in
    # SORTED (by sender) order at this point, and dividing them directly
    # here would silently misalign against `out`'s row order.
    msgs_short_out = out[COL_MSGS_SHORT].to_numpy()
    msgs_long_out = out[COL_MSGS_LONG].to_numpy()
    out[COL_RECIPIENT_DIVERSITY_SHORT] = np.where(
        msgs_short_out > 0,
        out[COL_UNIQUE_DEST_SHORT].to_numpy() / np.maximum(msgs_short_out, 1),
        0.0,
    )
    out[COL_RECIPIENT_DIVERSITY_LONG] = np.where(
        msgs_long_out > 0,
        out[COL_UNIQUE_DEST_LONG].to_numpy() / np.maximum(msgs_long_out, 1),
        0.0,
    )

    # sender_age_days: NOT windowed, NOT the two-pointer machinery above -
    # just "how long has this sender existed as of this row", a plain
    # groupby-min. A sender's own first message correctly gets 0.0 (that
    # first row IS its own reference point) - see module docstring.
    first_seen = df.groupby(sender_key)["timestamp"].transform("min")
    # np.timedelta64(1, "D") rather than pd.Timedelta(days=1) - the latter
    # trips the same numpy "generic unit" DeprecationWarning documented on
    # _window_timedelta64() above, on this project's pinned pandas/numpy.
    out[COL_SENDER_AGE_DAYS] = (
        (df["timestamp"] - first_seen).to_numpy() / np.timedelta64(1, "D")
    ).astype(np.float64)

    # imsi_distinct_originators_1hr: SS7-only, see module docstring's
    # IMSI-LINKAGE note. Column is left OFF the output entirely when the
    # input has no `imsi` column (SMPP case) - not NaN-filled.
    #
    # Rows where THIS row's own imsi is null get NA here, not 0 or a
    # shared "<NA>" bucket - checked against real data: 31.7% of a real
    # SS7 sample has a null imsi, so lumping every null-imsi row into one
    # fake shared identity (an earlier version of this code did exactly
    # that via fillna("<NA>")) would fabricate a high-variance signal out
    # of rows where the IMSI - the whole thing this feature measures -
    # isn't actually known. Uses pandas' nullable "Int64" so NA is a real,
    # distinct value (same convention as rule_flagged's nullable bool -
    # see common/schemas.py).
    if "imsi" in df.columns:
        imsi_known = df["imsi"].notna().to_numpy()
        imsi_distinct_orig = np.full(n, np.nan)

        if imsi_known.any():
            sub_pos = np.flatnonzero(imsi_known)
            imsi_sub_key = df["imsi"].astype(str).to_numpy()[sub_pos]
            orig_sub_key = df["originator"].fillna("<NA>").astype(str).to_numpy()[sub_pos]
            ts_sub = df["timestamp"].to_numpy()[sub_pos]

            sub_order = np.lexsort((ts_sub.view("i8"), imsi_sub_key))
            ordered_pos = sub_pos[sub_order]
            imsi_sorted = imsi_sub_key[sub_order]
            ts_sorted2 = ts_sub[sub_order]
            orig_sorted2 = orig_sub_key[sub_order]

            m = len(ordered_pos)
            imsi_group_change = np.empty(m, dtype=bool)
            imsi_group_change[0] = True
            if m > 1:
                imsi_group_change[1:] = imsi_sorted[1:] != imsi_sorted[:-1]
            imsi_starts = np.flatnonzero(imsi_group_change)
            imsi_ends = np.append(imsi_starts[1:], m)

            computed = np.zeros(m, dtype=np.int64)
            for start, end in zip(imsi_starts, imsi_ends):
                computed[start:end] = _imsi_window_originator_counts(
                    ts_sorted2[start:end], orig_sorted2[start:end]
                )
            imsi_distinct_orig[ordered_pos] = computed

        out[COL_IMSI_DISTINCT_ORIG_LONG] = pd.Series(
            imsi_distinct_orig, index=out.index
        ).astype("Int64")

    return out


def run_behavioral(messages_path: Path, out_path: Path) -> pd.DataFrame:
    messages_path = Path(messages_path)
    if not messages_path.exists():
        print(f"No messages file found at {messages_path}")
        return pd.DataFrame()

    print(f"Loading {messages_path} ...")
    messages = pd.read_csv(messages_path, low_memory=False)
    print(f"Computing behavioral features for {len(messages)} message(s)...")
    enriched = compute_behavioral_features(messages)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote {len(enriched)} rows to {out_path}")
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages_path", type=str, default="data/processed/SMPP/messages.csv")
    parser.add_argument("--out_path", type=str, default="data/processed/SMPP/messages_with_behavioral.csv")
    args = parser.parse_args()
    run_behavioral(Path(args.messages_path), Path(args.out_path))


if __name__ == "__main__":
    main()
