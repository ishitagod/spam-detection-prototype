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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.data import SENDER_VELOCITY_ZSCORE_COL
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
        "sender_age_days": 5.5, "sender_recipient_diversity_ratio_5min": 0.4,
        "sender_recipient_diversity_ratio_1hr": 0.6,
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


def test_missing_velocity_zscore_becomes_nan_not_zero():
    """Same reasoning as test_missing_dcs_becomes_nan_not_zero() above,
    for SENDER_VELOCITY_ZSCORE_COL - a cold-start sender (or one Feast
    doesn't recognize) must stay NaN, not silently become 0 (which would
    fabricate 'exactly average burst size')."""
    canonical = _row()
    behavioral = {c: 1 for c in BEHAVIORAL_COLS}  # SENDER_VELOCITY_ZSCORE_COL deliberately absent
    row = build_rule_pattern_row(canonical, behavioral)
    assert math.isnan(row[SENDER_VELOCITY_ZSCORE_COL])


def test_real_velocity_zscore_value_carries_through():
    canonical = _row()
    behavioral = {**{c: 1 for c in BEHAVIORAL_COLS}, SENDER_VELOCITY_ZSCORE_COL: -1.7}
    row = build_rule_pattern_row(canonical, behavioral)
    assert row[SENDER_VELOCITY_ZSCORE_COL] == -1.7


def test_source_one_hot_is_mutually_exclusive():
    ss7_row = build_rule_pattern_row(_row(source="SS7"), {c: 0 for c in BEHAVIORAL_COLS})
    assert ss7_row["source_SMPP"] == 0
    assert ss7_row["source_SS7"] == 1


def test_text_decode_failed_flag_carries_through():
    canonical = _row(text="", text_decode_failed=True)
    row = build_rule_pattern_row(canonical, {c: 0 for c in BEHAVIORAL_COLS})
    assert row["text_decode_failed"] == 1
    assert row["text_length"] == 0


class _FakeTfidfVectorizer:
    """Minimal stand-in for a fitted sklearn TfidfVectorizer - real
    vocabulary/IDF weights don't matter here, only that
    build_rule_pattern_row() calls .transform()/.get_feature_names_out()
    the same way models/rule_pattern/data.py's build_feature_matrix() did
    at training time and names columns tfidf_<token> from the result."""

    def get_feature_names_out(self):
        return np.array(["win", "prize"])

    def transform(self, texts):
        class _Sparse:
            def toarray(self_):
                return np.array([[0.6, 0.8]])
        return _Sparse()


class _FakeEmbeddingPcaPipeline:
    """Minimal stand-in for a fitted (StandardScaler -> PCA) pipeline -
    real 384-dim MiniLM input isn't needed here, only that
    build_rule_pattern_row() feeds it embed_texts()'s raw output and
    names the reduced columns emb_pca_<i>."""

    def transform(self, raw_embeddings):
        assert raw_embeddings.shape == (1, 4)  # matches the fake embed_texts below
        return np.array([[1.5, -2.5]])


def test_tfidf_columns_are_named_and_ordered_from_the_vectorizer(monkeypatch):
    """Champion trained --with_tfidf: tfidf_* columns must appear, named
    from the FITTED vectorizer's own vocabulary (get_feature_names_out()),
    not assumed/hardcoded - a vocabulary change in training must show up
    here automatically, not require a matching edit in this module."""
    canonical = _row(text="win a prize")
    row = build_rule_pattern_row(
        canonical, {c: 0 for c in BEHAVIORAL_COLS},
        tfidf_vectorizer=_FakeTfidfVectorizer(),
    )
    assert row["tfidf_win"] == 0.6
    assert row["tfidf_prize"] == 0.8
    assert "emb_pca_0" not in row  # no embedding_pca_pipeline given - not added


def test_embedding_columns_use_champions_own_pca_pipeline(monkeypatch):
    """Champion trained --with_embeddings: emb_pca_* columns must appear,
    built by running the REAL live text through features/
    text_embeddings.py's embed_texts() (mocked here to avoid loading real
    MiniLM weights in a unit test) then the champion's own fitted PCA
    pipeline - never a fresh/refit PCA."""
    def fake_embed_texts(texts, **kwargs):
        assert list(texts) == ["win a prize"]
        return np.zeros((1, 4), dtype=np.float32)

    monkeypatch.setattr("features.text_embeddings.embed_texts", fake_embed_texts)

    canonical = _row(text="win a prize")
    row = build_rule_pattern_row(
        canonical, {c: 0 for c in BEHAVIORAL_COLS},
        embedding_pca_pipeline=_FakeEmbeddingPcaPipeline(),
    )
    assert row["emb_pca_0"] == 1.5
    assert row["emb_pca_1"] == -2.5
    assert "tfidf_win" not in row  # no tfidf_vectorizer given - not added
