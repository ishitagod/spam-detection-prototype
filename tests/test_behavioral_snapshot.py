"""
pytest suite for features.behavioral_snapshot.compute_sender_snapshots().

Run:
    pytest tests/test_behavioral_snapshot.py -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.behavioral_snapshot import compute_imsi_snapshots, compute_sender_snapshots

NOW = pd.Timestamp("2026-08-19T10:00:00")

BASE = {
    "source": "SMPP", "originator": "SENDER1", "destination": "9198765001",
    "timestamp": "2026-08-19T09:00:00", "text": "hello",
}


def msg(**overrides) -> dict:
    return {**BASE, **overrides}


def run(rows, now=NOW, top_k_texts=20):
    return compute_sender_snapshots(pd.DataFrame(rows), now=now, top_k_texts=top_k_texts)


def row_for(result, sender_id):
    match = result[result["sender_id"] == sender_id]
    assert len(match) == 1, f"expected exactly one row for {sender_id}, got {len(match)}"
    return match.iloc[0]


def test_every_sender_seen_anywhere_gets_a_row_even_with_no_recent_activity():
    """A sender whose only message is long outside every window still
    gets a snapshot row - all-zero, not missing (see module docstring)."""
    rows = [msg(timestamp="2020-01-01T00:00:00")]
    result = run(rows)
    assert len(result) == 1
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_msgs_last_5min"] == 0
    assert row["sender_msgs_last_1hr"] == 0
    assert row["sender_unique_destinations_1hr"] == 0
    assert row["recent_text_counts_json"] == "{}"


def test_now_is_the_explicit_reference_point_not_the_data():
    """Ageing is measured from the passed-in `now`, not derived from the
    data's own max timestamp - a message 30min before `now` counts in the
    1hr window regardless of what today's wall-clock date is."""
    rows = [msg(timestamp="2026-08-19T09:30:00")]  # 30min before NOW
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_msgs_last_1hr"] == 1
    assert row["sender_msgs_last_5min"] == 0


def test_window_boundary_is_inclusive():
    """A message exactly `window` old is still counted (closed-left) -
    same convention as features/behavioral.py."""
    rows = [msg(timestamp="2026-08-19T09:00:00")]  # exactly 1hr before NOW
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_msgs_last_1hr"] == 1


def test_message_after_now_is_excluded_not_counted_as_negative_age():
    """A timestamp in the future relative to `now` (e.g. clock skew) must
    not count as history."""
    rows = [msg(timestamp="2026-08-19T10:30:00")]  # after NOW
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_msgs_last_1hr"] == 0


def test_unique_destinations_counts_distinct_within_long_window():
    rows = [
        msg(timestamp="2026-08-19T09:10:00", destination="A"),
        msg(timestamp="2026-08-19T09:20:00", destination="A"),
        msg(timestamp="2026-08-19T09:30:00", destination="B"),
    ]
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_unique_destinations_1hr"] == 2


def test_recent_text_counts_json_reflects_frequency_within_long_window():
    rows = [
        msg(timestamp="2026-08-19T09:10:00", text="WIN A PRIZE"),
        msg(timestamp="2026-08-19T09:20:00", text="WIN A PRIZE"),
        msg(timestamp="2026-08-19T09:30:00", text="different text"),
    ]
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    counts = json.loads(row["recent_text_counts_json"])
    assert counts["WIN A PRIZE"] == 2
    assert counts["different text"] == 1


def test_recent_text_counts_excludes_texts_outside_long_window():
    rows = [
        msg(timestamp="2020-01-01T00:00:00", text="ancient text"),
        msg(timestamp="2026-08-19T09:30:00", text="recent text"),
    ]
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    counts = json.loads(row["recent_text_counts_json"])
    assert "ancient text" not in counts
    assert counts["recent text"] == 1


def test_top_k_caps_stored_text_variety():
    rows = [
        msg(timestamp="2026-08-19T09:30:00", text=f"text_{i}")
        for i in range(5)
    ]
    result = run(rows, top_k_texts=2)
    row = row_for(result, "SMPP|SENDER1")
    counts = json.loads(row["recent_text_counts_json"])
    assert len(counts) == 2


def test_different_senders_are_isolated():
    """Each sender's own message legitimately counts toward its own
    snapshot (unlike behavioral.py's per-row features, this snapshot
    represents state INCLUDING everything known so far - see module
    docstring) - what must NOT happen is one sender's count including the
    other sender's message."""
    rows = [
        msg(originator="SENDER_A", timestamp="2026-08-19T09:30:00"),
        msg(originator="SENDER_B", timestamp="2026-08-19T09:31:00"),
    ]
    result = run(rows)
    assert len(result) == 2
    assert row_for(result, "SMPP|SENDER_A")["sender_msgs_last_1hr"] == 1
    assert row_for(result, "SMPP|SENDER_B")["sender_msgs_last_1hr"] == 1


def test_same_originator_different_source_are_different_senders():
    rows = [
        msg(source="SMPP", originator="12345", timestamp="2026-08-19T09:30:00"),
        msg(source="SS7", originator="12345", timestamp="2026-08-19T09:35:00"),
    ]
    result = run(rows)
    assert len(result) == 2
    assert set(result["sender_id"]) == {"SMPP|12345", "SS7|12345"}


def test_raises_on_missing_required_column():
    bad = pd.DataFrame([{"source": "SMPP"}])
    with pytest.raises(ValueError):
        compute_sender_snapshots(bad, now=NOW)


def test_sender_age_days_measures_time_since_first_ever_message():
    """ALL-TIME aggregate, not restricted to either window - a message
    3 days before `now` still sets the age even though it's outside both
    the 5min and 1hr windows (see module docstring's SENDER_AGE_DAYS
    note)."""
    rows = [msg(timestamp="2026-08-16T10:00:00")]  # 3 days before NOW
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_age_days"] == pytest.approx(3.0)


def test_sender_age_days_uses_earliest_message_not_latest():
    rows = [
        msg(timestamp="2026-08-17T10:00:00"),  # 2 days before NOW
        msg(timestamp="2026-08-19T09:00:00"),  # 1hr before NOW
    ]
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_age_days"] == pytest.approx(2.0)


def test_sender_age_days_clips_negative_age_from_future_timestamps():
    """A sender whose only message is after `now` (clock skew) must not
    get a nonsensical negative age."""
    rows = [msg(timestamp="2026-08-19T11:00:00")]  # 1hr after NOW
    result = run(rows)
    row = row_for(result, "SMPP|SENDER1")
    assert row["sender_age_days"] == 0.0


def test_mixed_numeric_and_string_originators_do_not_break_output_dtype():
    """SS7 originators can be inferred as int64 by pandas for one file and
    str for another before concatenation - the snapshot must normalize to
    str (a real bug hit while building this: pyarrow's parquet writer
    rejects a mixed str/int object column outright)."""
    rows = [
        msg(source="SS7", originator=12345, timestamp="2026-08-19T09:30:00"),
        msg(source="SS7", originator="67890", timestamp="2026-08-19T09:31:00"),
    ]
    result = run(rows)
    assert result["originator"].map(type).eq(str).all()


IMSI_BASE = {"imsi": "IMSI1", "originator": "SENDER1", "timestamp": "2026-08-19T09:00:00"}


def imsi_msg(**overrides) -> dict:
    return {**IMSI_BASE, **overrides}


def run_imsi(rows, now=NOW):
    return compute_imsi_snapshots(pd.DataFrame(rows), now=now)


def imsi_row_for(result, imsi):
    match = result[result["imsi"] == imsi]
    assert len(match) == 1, f"expected exactly one row for {imsi}, got {len(match)}"
    return match.iloc[0]


def test_imsi_null_rows_are_dropped_not_given_a_shared_identity():
    """A row whose own imsi is null must not produce ANY imsi snapshot
    row - not lumped into a shared '<NA>' identity (see module docstring
    - same convention as features/behavioral.py's per-row version)."""
    rows = [imsi_msg(imsi=None), imsi_msg(imsi="IMSI1")]
    result = run_imsi(rows)
    assert len(result) == 1
    assert result.iloc[0]["imsi"] == "IMSI1"


def test_imsi_distinct_originators_counts_within_long_window_only():
    rows = [
        imsi_msg(originator="A", timestamp="2026-08-19T09:10:00"),
        imsi_msg(originator="B", timestamp="2026-08-19T09:20:00"),
        imsi_msg(originator="A", timestamp="2020-01-01T00:00:00"),  # outside window
    ]
    result = run_imsi(rows)
    row = imsi_row_for(result, "IMSI1")
    assert row["imsi_distinct_originators_1hr"] == 2


def test_imsi_with_no_recent_activity_still_gets_a_zeroed_row():
    """Same 'every known entity gets a defined row' convention as
    compute_sender_snapshots() - an imsi seen only outside the window
    still gets a row, 0 not missing."""
    rows = [imsi_msg(timestamp="2020-01-01T00:00:00")]
    result = run_imsi(rows)
    row = imsi_row_for(result, "IMSI1")
    assert row["imsi_distinct_originators_1hr"] == 0


def test_different_imsis_are_isolated():
    rows = [
        imsi_msg(imsi="IMSI1", originator="A", timestamp="2026-08-19T09:30:00"),
        imsi_msg(imsi="IMSI2", originator="B", timestamp="2026-08-19T09:31:00"),
    ]
    result = run_imsi(rows)
    assert len(result) == 2
    assert imsi_row_for(result, "IMSI1")["imsi_distinct_originators_1hr"] == 1
    assert imsi_row_for(result, "IMSI2")["imsi_distinct_originators_1hr"] == 1


def test_all_null_imsi_returns_empty_frame_with_expected_columns():
    rows = [imsi_msg(imsi=None)]
    result = run_imsi(rows)
    assert len(result) == 0
    assert list(result.columns) == ["imsi", "event_timestamp", "imsi_distinct_originators_1hr"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
