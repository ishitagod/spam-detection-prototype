"""
pytest suite for serving/anomaly_scoring.py's feature-row construction
(build_anomaly_row) - the anomaly_score counterpart to
tests/test_serving_scoring.py's build_rule_pattern_row tests. Model/corpus
loading (_load_champion/_load_corpus) needs a real MLflow registry and a
real embeddings/faiss corpus - not exercised here.

Run:
    pytest tests/test_serving_anomaly_scoring.py -v
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.data import BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL
from serving.anomaly_scoring import build_anomaly_row
from serving.canonical import CanonicalRow

NEAR_DUP_FEATURES = {
    "near_dup_match_count_1hr": 0, "near_dup_max_similarity_1hr": 0.0, "near_dup_distinct_senders_1hr": 0,
    "near_dup_match_count_24hr": 0, "near_dup_max_similarity_24hr": 0.0, "near_dup_distinct_senders_24hr": 0,
}
EMBEDDING = np.zeros((1, 4), dtype=np.float32)


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
        "sender_age_days": 5.5, "sender_recipient_diversity_ratio_5min": 0.4,
        "sender_recipient_diversity_ratio_1hr": 0.6,
    }
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    for col in BEHAVIORAL_COLS:
        assert row[col] == behavioral[col]


def test_cold_start_sender_none_values_become_zero():
    canonical = _row()
    behavioral = {c: None for c in BEHAVIORAL_COLS}
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    for col in BEHAVIORAL_COLS:
        assert row[col] == 0


def test_missing_imsi_becomes_nan_not_zero():
    """build_combined_frame() needs the real None-vs-real distinction to
    build imsi_distinct_originators_1hr's _known indicator correctly - a
    0-fill here would fabricate a known value for a cold-start/SMPP row."""
    canonical = _row()
    behavioral = {c: 0 for c in BEHAVIORAL_COLS}  # IMSI_DISTINCT_ORIG_COL deliberately absent
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    assert math.isnan(row[IMSI_DISTINCT_ORIG_COL])


def test_missing_velocity_zscore_becomes_nan_not_zero():
    """Same reasoning as IMSI above, for SENDER_VELOCITY_ZSCORE_COL - kept
    OUT of BEHAVIORAL_COLS deliberately (models/anomaly/data.py's comment)
    so build_combined_frame() can build its own _known indicator from a
    real None-vs-real distinction, not a fabricated 0."""
    canonical = _row()
    behavioral = {c: 0 for c in BEHAVIORAL_COLS}  # SENDER_VELOCITY_ZSCORE_COL deliberately absent
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    assert math.isnan(row[SENDER_VELOCITY_ZSCORE_COL])


def test_real_velocity_zscore_value_carries_through():
    canonical = _row()
    behavioral = {**{c: 0 for c in BEHAVIORAL_COLS}, SENDER_VELOCITY_ZSCORE_COL: -1.7}
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    assert row[SENDER_VELOCITY_ZSCORE_COL] == -1.7


def test_near_dup_source_and_embedding_are_included():
    canonical = _row(source="SS7")
    behavioral = {c: 0 for c in BEHAVIORAL_COLS}
    row = build_anomaly_row(canonical, behavioral, NEAR_DUP_FEATURES, EMBEDDING)
    for col, value in NEAR_DUP_FEATURES.items():
        assert row[col] == value
    assert row["source"] == "SS7"
    assert row["emb_0"] == 0.0
    assert "emb_3" in row and "emb_4" not in row


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
