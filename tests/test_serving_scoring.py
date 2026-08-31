"""
pytest suite for serving/scoring.py's feature-row construction
(build_rule_pattern_row) - the piece that must stay byte-for-byte aligned
with models/rule_pattern/data.py's _base_feature_frame() column names.
Model loading itself (_load_champion) needs a real MLflow registry with a
promoted champion - not exercised here, covered by
tests/test_serving_app.py's mocked-scoring path instead.

Run:
    pytest tests/test_serving_scoring.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.canonical import CanonicalRow
from serving.scoring import BEHAVIORAL_COLS, build_rule_pattern_row


def _row(**overrides) -> CanonicalRow:
    defaults = dict(
        source="SMPP", record_id="r1", originator="123", destination="456",
        text="WIN A PRIZE", timestamp="2026-08-20T10:00:00Z", dcs=0.0,
        text_decode_failed=False,
    )
    defaults.update(overrides)
    return CanonicalRow(**defaults)


def test_known_sender_uses_real_behavioral_values():
    canonical = _row()
    behavioral = {
        "sender_msgs_last_5min": 3, "sender_msgs_last_1hr": 40,
        "sender_unique_destinations_1hr": 12, "sender_repeat_content_ratio_1hr": 0.75,
    }
    row = build_rule_pattern_row(canonical, behavioral)

    for col in BEHAVIORAL_COLS:
        assert row[col] == behavioral[col]
    assert row["dcs"] == 0.0
    assert row["text_decode_failed"] == 0
    assert row["text_length"] == len("WIN A PRIZE")
    assert row["source_SMPP"] == 1
    assert row["source_SS7"] == 0


def test_cold_start_sender_none_values_become_zero():
    canonical = _row()
    behavioral = {c: None for c in BEHAVIORAL_COLS}
    row = build_rule_pattern_row(canonical, behavioral)
    for col in BEHAVIORAL_COLS:
        assert row[col] == 0


def test_missing_dcs_becomes_nan_not_zero():
    """LightGBM has native missing-value handling (models/rule_pattern/
    data.py's _base_feature_frame() docstring) - a genuinely missing dcs
    must stay NaN, not silently become 0 (a real DCS value)."""
    canonical = _row(dcs=None)
    row = build_rule_pattern_row(canonical, {c: 1 for c in BEHAVIORAL_COLS})
    assert math.isnan(row["dcs"])


def test_source_one_hot_is_mutually_exclusive():
    ss7_row = build_rule_pattern_row(_row(source="SS7"), {c: 0 for c in BEHAVIORAL_COLS})
    assert ss7_row["source_SMPP"] == 0
    assert ss7_row["source_SS7"] == 1


def test_text_decode_failed_flag_carries_through():
    canonical = _row(text="", text_decode_failed=True)
    row = build_rule_pattern_row(canonical, {c: 0 for c in BEHAVIORAL_COLS})
    assert row["text_decode_failed"] == 1
    assert row["text_length"] == 0
