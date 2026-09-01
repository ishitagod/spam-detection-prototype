"""
pytest suite for models.anomaly.explain's pure helper logic
(build_interpretable_frame, make_predict_fn) - the wiring that has to be
right for LIME's perturbed samples to reach the real pipeline correctly.
resolve_run_id/rebuild_dataset/main() need a real MLflow run + trained
pipeline and aren't unit-tested here - same convention as
models/rule_pattern/explain.py (untested end-to-end; run by hand against
a real promoted run).

Run:
    pytest tests/test_anomaly_explain.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.explain import build_interpretable_frame, make_predict_fn


def _sample_df(sources=("SMPP", "SMPP", "SS7", "SS7")) -> pd.DataFrame:
    n = len(sources)
    return pd.DataFrame({
        "source": list(sources),
        "sender_msgs_last_5min": [1, 2, 3, 4][:n],
        "sender_msgs_last_1hr": [10, 20, 30, 40][:n],
        "sender_unique_destinations_1hr": [1, 2, 3, 4][:n],
        "sender_repeat_content_ratio_1hr": [0.1, 0.2, 0.3, 0.4][:n],
        "near_dup_match_count_1hr": [0, 1, 2, 3][:n],
        "near_dup_max_similarity_1hr": [0.0, 0.5, 0.6, 0.7][:n],
        "near_dup_distinct_senders_1hr": [0, 1, 1, 2][:n],
        "near_dup_match_count_24hr": [0, 1, 2, 3][:n],
        "near_dup_max_similarity_24hr": [0.0, 0.5, 0.6, 0.7][:n],
        "near_dup_distinct_senders_24hr": [0, 1, 1, 2][:n],
    })


class _FakePipeline:
    """decision_function that's a known, simple function of one
    interpretable column and one embedding column - lets tests assert the
    exact numeric output rather than just shape/type."""

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return X["sender_msgs_last_1hr"].to_numpy(dtype=float) * 0.01 + X["emb_0"].to_numpy(dtype=float)


def test_interpretable_frame_excludes_embeddings_and_one_hots_multi_source():
    df = _sample_df()
    frame = build_interpretable_frame(df)
    assert "source_SMPP" in frame.columns and "source_SS7" in frame.columns
    assert not any(c.startswith("emb_") for c in frame.columns)
    assert len(frame) == len(df)


def test_interpretable_frame_drops_source_dummy_for_single_source():
    """Mirrors models/anomaly/data.py's build_feature_matrix(): a
    single-source rebuild must not produce a constant, information-free
    source dummy column."""
    df = _sample_df(sources=("SMPP", "SMPP", "SMPP", "SMPP"))
    frame = build_interpretable_frame(df)
    assert not any(c.startswith("source_") for c in frame.columns)


def test_interpretable_frame_log1ps_count_cols_not_ratios():
    df = _sample_df()
    frame = build_interpretable_frame(df)
    assert frame["sender_msgs_last_1hr"].iloc[0] == pytest.approx(np.log1p(10))
    # a ratio/similarity column - untouched by log1p
    assert frame["sender_repeat_content_ratio_1hr"].iloc[0] == pytest.approx(0.1)


def test_predict_fn_reattaches_fixed_embedding_and_flips_sign():
    df = _sample_df()
    frame = build_interpretable_frame(df)
    expected_cols = list(frame.columns) + ["emb_0", "emb_1"]

    build = make_predict_fn(_FakePipeline(), expected_cols, ["emb_0", "emb_1"], fixed_embedding=np.array([1.0, 2.0]))
    predict_fn = build(list(frame.columns))

    perturbed = frame.to_numpy(dtype=float)[:2]
    out = predict_fn(perturbed)

    assert out.shape == (2,)
    # sign is flipped (anomaly_score convention: higher = more anomalous,
    # negated decision_function - see models/anomaly/train.py) and emb_0
    # is the FIXED value (1.0), not anything from `perturbed`.
    expected = -(frame["sender_msgs_last_1hr"].to_numpy(dtype=float)[:2] * 0.01 + 1.0)
    assert out == pytest.approx(expected)


def test_predict_fn_fills_missing_expected_columns_with_zero():
    """A perturbed frame missing a column the fitted preprocessor expects
    (e.g. explaining a single-source run against a multi-source-trained
    model's expected_cols) must fill 0, not raise a KeyError."""
    df = _sample_df(sources=("SMPP", "SMPP", "SMPP", "SMPP"))
    frame = build_interpretable_frame(df)  # no source_* columns
    expected_cols = list(frame.columns) + ["source_SS7", "emb_0"]

    build = make_predict_fn(_FakePipeline(), expected_cols, ["emb_0"], fixed_embedding=np.array([0.0]))
    predict_fn = build(list(frame.columns))
    out = predict_fn(frame.to_numpy(dtype=float)[:1])
    assert out.shape == (1,)
