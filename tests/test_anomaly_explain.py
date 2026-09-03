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

from models.anomaly.explain import (
    build_interpretable_frame,
    compute_shap_contributions,
    make_predict_fn,
    summarize_shap_importance,
)


def _sample_df(sources=("SMPP", "SMPP", "SS7", "SS7")) -> pd.DataFrame:
    n = len(sources)
    return pd.DataFrame({
        "source": list(sources),
        "sender_msgs_last_5min": [1, 2, 3, 4][:n],
        "sender_msgs_last_1hr": [10, 20, 30, 40][:n],
        "sender_unique_destinations_1hr": [1, 2, 3, 4][:n],
        "sender_repeat_content_ratio_1hr": [0.1, 0.2, 0.3, 0.4][:n],
        # Tier 0 additions - see models/anomaly/data.py's BEHAVIORAL_COLS
        # comment. All three: 0-1 ratios / small day-counts, no log1p.
        "sender_age_days": [0.0, 0.5, 1.0, 1.5][:n],
        "sender_recipient_diversity_ratio_5min": [0.0, 0.25, 0.5, 0.75][:n],
        "sender_recipient_diversity_ratio_1hr": [0.0, 0.2, 0.4, 0.6][:n],
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


def test_compute_shap_contributions_negates_tree_explainer_output():
    """compute_shap_contributions() must return the SIGN-FLIPPED raw
    shap.TreeExplainer output (module docstring's SHAP SIGN CONVENTION) -
    checked here against shap.TreeExplainer called directly on the same
    fitted model/data, not re-derived from anomaly_score (no preprocessing
    pipeline needed for this - just the negation itself)."""
    shap = pytest.importorskip("shap")
    from sklearn.ensemble import IsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    iforest = IsolationForest(n_estimators=10, random_state=0).fit(X)

    raw = shap.TreeExplainer(iforest).shap_values(X)
    contributions = compute_shap_contributions(iforest, X)

    assert contributions.shape == raw.shape
    np.testing.assert_allclose(contributions, -raw)


def test_summarize_shap_importance_collapses_embedding_columns():
    feature_names = ["sender_msgs_last_1hr", "emb_pca_0", "emb_pca_1", "source_SMPP"]
    # row 0: emb_pca_0=0.1, emb_pca_1=0.2 -> bucket contribution 0.3
    # row 1: emb_pca_0=0.4, emb_pca_1=0.0 -> bucket contribution 0.4
    contributions = np.array([
        [1.0, 0.1, 0.2, 0.0],
        [2.0, 0.4, 0.0, 1.0],
    ])
    summary = summarize_shap_importance(contributions, feature_names)

    assert set(summary["feature"]) == {
        "sender_msgs_last_1hr", "source_SMPP",
        "content_embedding (sum of emb_pca_* |contribution|)",
    }
    bucket_row = summary[summary["feature"] == "content_embedding (sum of emb_pca_* |contribution|)"]
    assert bucket_row["mean_abs_shap_contribution"].iloc[0] == pytest.approx((0.3 + 0.4) / 2)
    # sorted descending by importance
    assert summary["mean_abs_shap_contribution"].is_monotonic_decreasing


def test_summarize_shap_importance_no_embedding_columns():
    """No emb_pca_* columns present (e.g. a hypothetical no-embedding run)
    - no bucket row should be added, every feature reported individually."""
    feature_names = ["sender_msgs_last_1hr", "source_SMPP"]
    contributions = np.array([[1.0, 0.0], [2.0, 1.0]])
    summary = summarize_shap_importance(contributions, feature_names)
    assert set(summary["feature"]) == {"sender_msgs_last_1hr", "source_SMPP"}


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
