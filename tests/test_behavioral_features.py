"""
pytest suite for features.behavioral.compute_behavioral_features().

Run:
    pytest tests/test_behavioral_features.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.behavioral import compute_behavioral_features

BASE = {
    "source": "SMPP", "originator": "SENDER1", "destination": "9198765001",
    "timestamp": "2026-08-19T10:00:00", "text": "hello",
}


def msg(**overrides) -> dict:
    return {**BASE, **overrides}


def run(rows):
    return compute_behavioral_features(pd.DataFrame(rows))


def test_first_message_from_a_sender_is_all_zero_cold_start():
    result = run([msg()])
    row = result.iloc[0]
    assert row["sender_msgs_last_5min"] == 0
    assert row["sender_msgs_last_1hr"] == 0
    assert row["sender_unique_destinations_1hr"] == 0
    assert row["sender_repeat_content_ratio_1hr"] == 0.0


def test_current_message_never_counts_as_its_own_history():
    """A single message must not see itself in its own window."""
    result = run([msg(record_id="only_one")])
    assert result.iloc[0]["sender_msgs_last_5min"] == 0


def test_messages_within_5min_are_counted_in_short_window():
    rows = [
        msg(timestamp="2026-08-19T10:00:00"),
        msg(timestamp="2026-08-19T10:02:00"),
        msg(timestamp="2026-08-19T10:04:00"),
    ]
    result = run(rows)
    assert list(result["sender_msgs_last_5min"]) == [0, 1, 2]


def test_message_outside_5min_window_is_excluded_from_short_but_counts_in_long():
    rows = [
        msg(timestamp="2026-08-19T10:00:00"),
        msg(timestamp="2026-08-19T10:10:00"),  # 10min later - outside 5min, inside 1hr
    ]
    result = run(rows)
    assert result.iloc[1]["sender_msgs_last_5min"] == 0
    assert result.iloc[1]["sender_msgs_last_1hr"] == 1


def test_window_boundary_is_inclusive():
    """A prior message exactly `window` old is still counted (closed-left)."""
    rows = [
        msg(timestamp="2026-08-19T10:00:00"),
        msg(timestamp="2026-08-19T10:05:00"),  # exactly 5min later
    ]
    result = run(rows)
    assert result.iloc[1]["sender_msgs_last_5min"] == 1


def test_message_older_than_1hr_ages_out_of_every_window():
    rows = [
        msg(timestamp="2026-08-19T09:00:00"),
        msg(timestamp="2026-08-19T10:00:01"),  # just over 1hr later
    ]
    result = run(rows)
    assert result.iloc[1]["sender_msgs_last_1hr"] == 0
    assert result.iloc[1]["sender_msgs_last_5min"] == 0


def test_unique_destinations_counts_distinct_not_total():
    rows = [
        msg(timestamp="2026-08-19T10:00:00", destination="A"),
        msg(timestamp="2026-08-19T10:01:00", destination="A"),
        msg(timestamp="2026-08-19T10:02:00", destination="B"),
        msg(timestamp="2026-08-19T10:03:00", destination="C"),
    ]
    result = run(rows)
    # 4th message: prior destinations seen = {A, A, B} -> 2 unique
    assert result.iloc[3]["sender_unique_destinations_1hr"] == 2


def test_repeat_content_ratio_matches_current_text_against_prior_window():
    rows = [
        msg(timestamp="2026-08-19T10:00:00", text="WIN A PRIZE"),
        msg(timestamp="2026-08-19T10:01:00", text="WIN A PRIZE"),
        msg(timestamp="2026-08-19T10:02:00", text="different text"),
        msg(timestamp="2026-08-19T10:03:00", text="WIN A PRIZE"),
    ]
    result = run(rows)
    # 4th message ("WIN A PRIZE"): 2 of the 3 prior messages match it
    assert result.iloc[3]["sender_repeat_content_ratio_1hr"] == pytest.approx(2 / 3)
    # 3rd message ("different text"): 0 of the 2 prior messages match it
    assert result.iloc[2]["sender_repeat_content_ratio_1hr"] == 0.0


def test_different_senders_do_not_share_history():
    rows = [
        msg(originator="SENDER_A", timestamp="2026-08-19T10:00:00"),
        msg(originator="SENDER_B", timestamp="2026-08-19T10:00:01"),
    ]
    result = run(rows)
    assert result.iloc[1]["sender_msgs_last_5min"] == 0


def test_same_originator_different_source_are_different_senders():
    """originator string could coincidentally collide across SMPP/SS7 - the
    sender key must include source too."""
    rows = [
        msg(source="SMPP", originator="12345", timestamp="2026-08-19T10:00:00"),
        msg(source="SS7", originator="12345", timestamp="2026-08-19T10:00:01"),
    ]
    result = run(rows)
    assert result.iloc[1]["sender_msgs_last_5min"] == 0


def test_output_preserves_original_row_order():
    """Internally sorted by (sender, timestamp) for processing - output
    must be scattered back to the caller's original row order."""
    rows = [
        msg(originator="SENDER_B", timestamp="2026-08-19T10:00:00", text="b"),
        msg(originator="SENDER_A", timestamp="2026-08-19T09:00:00", text="a"),
    ]
    result = run(rows)
    assert list(result["text"]) == ["b", "a"]


def test_raises_on_missing_required_column():
    bad = pd.DataFrame([{"source": "SMPP"}])
    with pytest.raises(ValueError):
        compute_behavioral_features(bad)


def test_sender_age_days_is_zero_on_first_message():
    result = run([msg()])
    assert result.iloc[0]["sender_age_days"] == 0.0


def test_sender_age_days_measures_days_since_first_message():
    rows = [
        msg(timestamp="2026-08-19T10:00:00"),
        msg(timestamp="2026-08-20T10:00:00"),  # exactly 1 day later
        msg(timestamp="2026-08-19T22:00:00"),  # 12hr later (inserted out of
        # chronological order on purpose - sorted internally by timestamp,
        # not by input row order)
    ]
    result = run(rows)
    assert result.iloc[0]["sender_age_days"] == pytest.approx(0.0)
    assert result.iloc[1]["sender_age_days"] == pytest.approx(1.0)
    assert result.iloc[2]["sender_age_days"] == pytest.approx(0.5)


def test_sender_age_days_isolated_per_sender():
    rows = [
        msg(originator="SENDER_A", timestamp="2026-08-19T10:00:00"),
        msg(originator="SENDER_B", timestamp="2020-01-01T00:00:00"),
        msg(originator="SENDER_A", timestamp="2026-08-20T10:00:00"),
    ]
    result = run(rows)
    # SENDER_B's ancient first message must not affect SENDER_A's age.
    assert result.iloc[2]["sender_age_days"] == pytest.approx(1.0)


def test_imsi_column_absent_from_output_when_not_in_input():
    """SMPP-shaped input has no imsi column at all - the SS7-only feature
    must not appear (NaN-filled or otherwise), same convention as
    message_type."""
    result = run([msg()])
    assert "imsi_distinct_originators_1hr" not in result.columns


def test_imsi_distinct_originators_counts_distinct_msisdns_behind_one_imsi():
    rows = [
        msg(source="SS7", originator="60111111111", imsi="IMSI_A", timestamp="2026-08-19T10:00:00"),
        msg(source="SS7", originator="60122222222", imsi="IMSI_A", timestamp="2026-08-19T10:05:00"),
        msg(source="SS7", originator="60111111111", imsi="IMSI_A", timestamp="2026-08-19T10:10:00"),
    ]
    result = run(rows)
    # 3rd row: prior originators behind IMSI_A = {60111111111, 60122222222} -> 2 distinct
    assert result.iloc[2]["imsi_distinct_originators_1hr"] == 2


def test_imsi_distinct_originators_excludes_current_row_and_stale_history():
    rows = [
        msg(source="SS7", originator="A", imsi="IMSI_X", timestamp="2026-08-19T08:00:00"),  # >1hr stale
        msg(source="SS7", originator="B", imsi="IMSI_X", timestamp="2026-08-19T09:59:00"),
        msg(source="SS7", originator="C", imsi="IMSI_X", timestamp="2026-08-19T10:00:00"),
    ]
    result = run(rows)
    # 3rd row: only B is within the trailing 1hr; the 08:00 row aged out,
    # and C (this row's own originator) never counts as its own history.
    assert result.iloc[2]["imsi_distinct_originators_1hr"] == 1


def test_imsi_distinct_originators_is_na_not_zero_when_this_rows_imsi_is_null():
    """A row with no imsi of its own must not get 0 (a real computed
    value) and must not be lumped into a shared '<NA>' bucket with every
    other null-imsi row - real SS7 data has null imsi on ~32% of rows, so
    that would fabricate signal. It should be a genuine NA."""
    rows = [
        msg(source="SS7", originator="A", imsi="IMSI_1", timestamp="2026-08-19T10:00:00"),
        msg(source="SS7", originator="B", imsi=None, timestamp="2026-08-19T10:01:00"),
    ]
    result = run(rows)
    assert pd.isna(result.iloc[1]["imsi_distinct_originators_1hr"])
    assert result["imsi_distinct_originators_1hr"].dtype == "Int64"


def test_imsi_distinct_originators_null_imsi_rows_do_not_pollute_each_other():
    """Two different null-imsi rows must not be treated as the same
    (fake shared) sender - each just gets NA independently."""
    rows = [
        msg(source="SS7", originator="A", imsi=None, timestamp="2026-08-19T10:00:00"),
        msg(source="SS7", originator="B", imsi=None, timestamp="2026-08-19T10:01:00"),
        msg(source="SS7", originator="C", imsi="IMSI_1", timestamp="2026-08-19T10:02:00"),
    ]
    result = run(rows)
    assert pd.isna(result.iloc[0]["imsi_distinct_originators_1hr"])
    assert pd.isna(result.iloc[1]["imsi_distinct_originators_1hr"])
    assert result.iloc[2]["imsi_distinct_originators_1hr"] == 0  # IMSI_1's own first message


def test_imsi_distinct_originators_isolated_per_imsi():
    rows = [
        msg(source="SS7", originator="A", imsi="IMSI_1", timestamp="2026-08-19T10:00:00"),
        msg(source="SS7", originator="Z", imsi="IMSI_2", timestamp="2026-08-19T10:01:00"),
        msg(source="SS7", originator="B", imsi="IMSI_1", timestamp="2026-08-19T10:02:00"),
    ]
    result = run(rows)
    # 3rd row (IMSI_1): only A is prior history for IMSI_1, IMSI_2's Z
    # must not leak in.
    assert result.iloc[2]["imsi_distinct_originators_1hr"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
