"""
pytest suite for features.message_reassembly.reassemble_messages().

Run:
    pytest tests/test_message_reassembly.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.message_reassembly import reassemble_messages

BASE_FEATURE = {
    "record_id": None, "originator": "SENDER1", "destination": "9198765001",
    "timestamp": "2026-08-19T10:00:00", "text": "", "text_decode_failed": False,
    "concat_ref": None, "concat_total_parts": 1, "concat_part_num": 1,
    "source": "SMPP",
}
BASE_LABEL = {"record_id": None, "rule_evaluated": False, "rule_flagged": None, "fraud_type": None}


def feat(**overrides) -> dict:
    return {**BASE_FEATURE, **overrides}


def lbl(**overrides) -> dict:
    return {**BASE_LABEL, **overrides}


def run(feature_rows, label_rows=None):
    if label_rows is None:
        label_rows = [lbl(record_id=r["record_id"]) for r in feature_rows]
    return reassemble_messages(pd.DataFrame(feature_rows), pd.DataFrame(label_rows))


def test_single_part_message_passes_through_unchanged():
    result = run([feat(record_id="r1", text="Hello World")])
    assert len(result) == 1
    assert result.iloc[0]["text"] == "Hello World"
    assert result.iloc[0]["message_partial"] == False
    assert result.iloc[0]["message_part_count"] == 1


def test_multipart_message_concatenated_in_part_order_even_if_input_is_out_of_order():
    """Part 2 listed BEFORE part 1 in the input - must still concatenate as
    part1+part2, not input order."""
    rows = [
        feat(record_id="r2", text="World", concat_ref=57, concat_total_parts=2, concat_part_num=2,
             timestamp="2026-08-19T10:00:05"),
        feat(record_id="r1", text="Hello ", concat_ref=57, concat_total_parts=2, concat_part_num=1,
             timestamp="2026-08-19T10:00:00"),
    ]
    result = run(rows)
    assert len(result) == 1
    assert result.iloc[0]["text"] == "Hello World"
    assert result.iloc[0]["message_partial"] == False
    assert result.iloc[0]["message_part_count"] == 2


def test_message_timestamp_is_earliest_part_not_latest():
    """Point-in-time correctness: the message's effective timestamp is when
    part 1 arrived, not the last part."""
    rows = [
        feat(record_id="r1", text="A", concat_ref=1, concat_total_parts=2, concat_part_num=1,
             timestamp="2026-08-19T10:00:00"),
        feat(record_id="r2", text="B", concat_ref=1, concat_total_parts=2, concat_part_num=2,
             timestamp="2026-08-19T10:05:00"),
    ]
    result = run(rows)
    assert pd.Timestamp(result.iloc[0]["timestamp"]) == pd.Timestamp("2026-08-19T10:00:00")


def test_incomplete_group_flagged_partial_not_dropped():
    """total_parts=3 but only 2 arrived (part 3 missing, e.g. network
    failure) - kept, not dropped, flagged via message_partial."""
    rows = [
        feat(record_id="r1", text="one ", concat_ref=9, concat_total_parts=3, concat_part_num=1),
        feat(record_id="r2", text="two", concat_ref=9, concat_total_parts=3, concat_part_num=2),
    ]
    result = run(rows)
    assert len(result) == 1
    assert result.iloc[0]["message_partial"] == True
    assert result.iloc[0]["message_part_count"] == 2
    assert result.iloc[0]["text"] == "one two"


def test_different_originators_with_same_ref_are_not_merged():
    """concat_ref is only an 8-bit value - two unrelated senders can
    legitimately share the same ref. Grouping key must include originator/
    destination, or these would get wrongly concatenated together."""
    rows = [
        feat(record_id="r1", originator="SENDER_A", text="fromA",
             concat_ref=5, concat_total_parts=1, concat_part_num=1),
        feat(record_id="r2", originator="SENDER_B", text="fromB",
             concat_ref=5, concat_total_parts=1, concat_part_num=1),
    ]
    result = run(rows)
    assert len(result) == 2
    assert set(result["text"]) == {"fromA", "fromB"}


def test_text_decode_failed_on_one_part_propagates_to_merged_message():
    rows = [
        feat(record_id="r1", text="ok part ", concat_ref=2, concat_total_parts=2, concat_part_num=1),
        feat(record_id="r2", text="", text_decode_failed=True,
             concat_ref=2, concat_total_parts=2, concat_part_num=2),
    ]
    result = run(rows)
    assert result.iloc[0]["text_decode_failed"] == True
    assert result.iloc[0]["text"] == "ok part "


def test_label_flagged_if_any_evaluated_part_is_flagged():
    rows = [
        feat(record_id="r1", concat_ref=3, concat_total_parts=2, concat_part_num=1),
        feat(record_id="r2", concat_ref=3, concat_total_parts=2, concat_part_num=2),
    ]
    labels = [
        lbl(record_id="r1", rule_evaluated=True, rule_flagged=False),
        lbl(record_id="r2", rule_evaluated=True, rule_flagged=True, fraud_type="spam_burst"),
    ]
    result = run(rows, labels)
    assert result.iloc[0]["rule_evaluated"] == True
    assert result.iloc[0]["rule_flagged"] == True
    assert result.iloc[0]["fraud_type"] == "spam_burst"


def test_label_unevaluated_when_no_part_was_evaluated():
    rows = [feat(record_id="r1")]
    labels = [lbl(record_id="r1", rule_evaluated=False, rule_flagged=None)]
    result = run(rows, labels)
    assert result.iloc[0]["rule_evaluated"] == False
    assert pd.isna(result.iloc[0]["rule_flagged"])


def test_raises_on_missing_required_feature_column():
    bad = pd.DataFrame([{"record_id": "r1"}])
    labels = pd.DataFrame([lbl(record_id="r1")])
    with pytest.raises(ValueError):
        reassemble_messages(bad, labels)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
