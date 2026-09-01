"""
pytest suite for models.rule_pattern.data.

Run:
    pytest tests/test_rule_pattern_data.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.data import BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL
from models.rule_pattern.data import (
    build_feature_matrix,
    load_labelled_messages,
    load_labelled_messages_with_embeddings,
)


def _sample_df(n=5):
    return pd.DataFrame({
        "source": ["SMPP"] * n,
        "record_id": [str(i) for i in range(n)],
        "text": ["hello world"] * n,
        "dcs": [0.0, 8.0, np.nan, 0.0, 8.0],
        "text_decode_failed": [False, False, True, False, False],
        "sender_msgs_last_5min": [0, 1, 2, 3, 4],
        "sender_msgs_last_1hr": [0, 10, 100, 1000, 16971],
        "sender_unique_destinations_1hr": [0, 1, 5, 20, 100],
        "sender_repeat_content_ratio_1hr": [0.0, 0.1, 0.5, 0.9, 1.0],
        "rule_evaluated": [True, True, True, False, False],
        "rule_flagged": [True, False, True, None, None],
    })


def test_load_labelled_messages_keeps_only_rule_evaluated_rows(tmp_path):
    path = tmp_path / "messages_with_behavioral.csv"
    _sample_df().to_csv(path, index=False)
    result = load_labelled_messages(path)
    assert len(result) == 3  # only the 3 rule_evaluated==True rows
    assert set(result["record_id"]) == {"0", "1", "2"}


def test_label_is_one_for_flagged_zero_for_confirmed_clean():
    df = _sample_df().iloc[:3]  # the 3 rule_evaluated rows
    _, y, _, _ = build_feature_matrix(df)
    assert list(y) == [1, 0, 1]  # matches rule_flagged: True, False, True


def test_behavioral_columns_are_included_unscaled():
    """Unlike models/anomaly/data.py, no StandardScaler here - tree
    splits are scale-invariant, so raw values should pass through
    exactly, including real observed extremes like 16,971."""
    df = _sample_df()
    X, _, feature_names, _ = build_feature_matrix(df)
    idx = feature_names.index("sender_msgs_last_1hr")
    assert list(X[:, idx]) == [0, 10, 100, 1000, 16971]


def test_nan_dcs_does_not_crash_left_for_lightgbm_to_handle():
    df = _sample_df()
    X, _, feature_names, _ = build_feature_matrix(df)
    idx = feature_names.index("dcs")
    assert np.isnan(X[2, idx])  # the NaN row's dcs stays NaN, not imputed


def test_text_length_is_derived_correctly():
    df = _sample_df()
    df["text"] = ["hi", "hello", "hey there", "", "a"]
    X, _, feature_names, _ = build_feature_matrix(df)
    idx = feature_names.index("text_length")
    assert list(X[:, idx]) == [2, 5, 9, 0, 1]


def test_source_gets_one_hot_encoded():
    df = _sample_df()
    df["source"] = ["SMPP", "SMPP", "SS7", "SS7", "SS7"]
    _, _, feature_names, _ = build_feature_matrix(df)
    assert "source_SMPP" in feature_names
    assert "source_SS7" in feature_names


def test_all_behavioral_cols_present():
    df = _sample_df()
    _, _, feature_names, _ = build_feature_matrix(df)
    for col in BEHAVIORAL_COLS:
        assert col in feature_names


def test_imsi_distinct_orig_col_present_when_available_nan_otherwise():
    """SS7-only feature (see models/anomaly/data.py's comment) - must
    still show up as a NaN-filled column when the source df doesn't have
    it at all (SMPP's real case, and this test's df), not be silently
    dropped from the feature matrix."""
    df = _sample_df()
    assert IMSI_DISTINCT_ORIG_COL not in df.columns
    X, _, feature_names, _ = build_feature_matrix(df)
    assert IMSI_DISTINCT_ORIG_COL in feature_names
    idx = feature_names.index(IMSI_DISTINCT_ORIG_COL)
    assert np.isnan(X[:, idx]).all()

    df[IMSI_DISTINCT_ORIG_COL] = [1.0, np.nan, 3.0, 0.0, 2.0]
    X, _, feature_names, _ = build_feature_matrix(df)
    idx = feature_names.index(IMSI_DISTINCT_ORIG_COL)
    assert np.array_equal(X[:, idx], df[IMSI_DISTINCT_ORIG_COL].to_numpy(), equal_nan=True)


def test_fitted_is_empty_when_no_optional_features_requested():
    """Neither use_embeddings nor use_tfidf - nothing corpus-dependent was
    fit, so there's nothing to serialize for inference-time reuse."""
    df = _sample_df()
    _, _, _, fitted = build_feature_matrix(df)
    assert fitted == {}


def _write_labelled_messages_with_embeddings(tmp_path, n_rows=6, n_evaluated=4, n_with_embedding=3):
    """n_rows total, first n_evaluated are rule_evaluated=True, first
    n_with_embedding have an embedding - deliberately n_with_embedding <
    n_evaluated so at least one row is evaluated but embedding-less,
    exercising the inner-join exclusion."""
    source_dir = tmp_path / "SMPP"
    source_dir.mkdir()
    messages_path = source_dir / "messages_with_behavioral.csv"

    pd.DataFrame({
        "source": ["SMPP"] * n_rows,
        "record_id": [str(i) for i in range(n_rows)],
        "text": ["hello world"] * n_rows,
        "dcs": [0.0] * n_rows,
        "text_decode_failed": [False] * n_rows,
        **{c: [0] * n_rows for c in BEHAVIORAL_COLS},
        "rule_evaluated": [True] * n_evaluated + [False] * (n_rows - n_evaluated),
        "rule_flagged": [True, False] * (n_evaluated // 2) + [None] * (n_rows - n_evaluated),
    }).to_csv(messages_path, index=False)

    embeddings = np.random.RandomState(0).rand(n_with_embedding, 4).astype(np.float32)
    np.save(source_dir / "embeddings.npy", embeddings)
    pd.DataFrame({
        "message_key": [f"SMPP|{i}" for i in range(n_with_embedding)],
    }).to_parquet(source_dir / "embeddings_id_map.parquet")

    return source_dir, messages_path


def test_load_with_embeddings_keeps_only_rule_evaluated_and_has_embedding(tmp_path):
    source_dir, messages_path = _write_labelled_messages_with_embeddings(
        tmp_path, n_rows=6, n_evaluated=4, n_with_embedding=3,
    )
    result = load_labelled_messages_with_embeddings(source_dir, messages_path)
    # record_id "3" is rule_evaluated but has no embedding (only 0,1,2 do) -
    # record_ids "4","5" are not rule_evaluated at all - only "0","1","2" survive both filters
    assert set(result["record_id"]) == {"0", "1", "2"}
    assert "emb_0" in result.columns


def test_build_feature_matrix_with_embeddings_reduces_to_pca_components():
    source_dir_cols = [f"emb_{i}" for i in range(4)]
    df = pd.DataFrame({
        "source": ["SMPP"] * 5, "record_id": [str(i) for i in range(5)],
        "text": ["hi"] * 5, "dcs": [0.0] * 5, "text_decode_failed": [False] * 5,
        **{c: [0] * 5 for c in BEHAVIORAL_COLS},
        "rule_flagged": [True, False, True, False, True],
        **{col: np.random.RandomState(i).randn(5) for i, col in enumerate(source_dir_cols)},
    })
    X, y, feature_names, fitted = build_feature_matrix(df, use_embeddings=True, n_embedding_components=2)
    assert "emb_0" not in feature_names
    assert "emb_pca_0" in feature_names and "emb_pca_1" in feature_names
    assert "emb_pca_2" not in feature_names
    assert list(y) == [1, 0, 1, 0, 1]
    assert "embedding_pca_pipeline" in fitted


def test_build_feature_matrix_with_embeddings_base_features_stay_unscaled():
    df = pd.DataFrame({
        "source": ["SMPP"] * 5, "record_id": [str(i) for i in range(5)],
        "text": ["hi"] * 5, "dcs": [0.0] * 5, "text_decode_failed": [False] * 5,
        "sender_msgs_last_5min": [0, 1, 2, 3, 4], "sender_msgs_last_1hr": [0, 10, 100, 1000, 16971],
        "sender_unique_destinations_1hr": [0] * 5, "sender_repeat_content_ratio_1hr": [0.0] * 5,
        "rule_flagged": [True, False, True, False, True],
        **{f"emb_{i}": np.random.RandomState(i).randn(5) for i in range(4)},
    })
    X, _, feature_names, _ = build_feature_matrix(df, use_embeddings=True, n_embedding_components=2)
    idx = feature_names.index("sender_msgs_last_1hr")
    assert list(X[:, idx]) == [0, 10, 100, 1000, 16971]  # raw, not scaled


def test_embedding_pca_fit_only_on_train_mask():
    """PCA must be FIT on train_mask rows only - a component axis fit on a
    test-only outlier the train fold never saw would be real leakage into
    featurization itself, not just into the model."""
    n = 20
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "source": ["SMPP"] * n, "record_id": [str(i) for i in range(n)],
        "text": ["hi"] * n, "dcs": [0.0] * n, "text_decode_failed": [False] * n,
        **{c: [0] * n for c in BEHAVIORAL_COLS},
        "rule_flagged": [True, False] * (n // 2),
        **{f"emb_{i}": rng.randn(n) for i in range(4)},
    })
    train_mask = np.array([True] * 15 + [False] * 5)
    # A PCA fit on ONLY the train rows must differ from one fit on everyone -
    # confirms train_mask actually changed what gets fit, not silently ignored.
    _, _, _, fitted_train_only = build_feature_matrix(
        df, train_mask=train_mask, use_embeddings=True, n_embedding_components=2,
    )
    _, _, _, fitted_full = build_feature_matrix(
        df, train_mask=np.ones(n, dtype=bool), use_embeddings=True, n_embedding_components=2,
    )
    pca_train_only = fitted_train_only["embedding_pca_pipeline"].named_steps["pca"]
    pca_full = fitted_full["embedding_pca_pipeline"].named_steps["pca"]
    assert not np.allclose(pca_train_only.components_, pca_full.components_)


def test_build_feature_matrix_with_tfidf_adds_ngram_columns():
    df = pd.DataFrame({
        "source": ["SMPP"] * 6, "record_id": [str(i) for i in range(6)],
        "text": ["win a free prize now", "win a free prize now", "win a free prize now",
                 "hello how are you today", "hello how are you today", "meeting at noon tomorrow"],
        "dcs": [0.0] * 6, "text_decode_failed": [False] * 6,
        **{c: [0] * 6 for c in BEHAVIORAL_COLS},
        "rule_flagged": [True, True, True, False, False, False],
    })
    X, y, feature_names, fitted = build_feature_matrix(
        df, use_tfidf=True, tfidf_max_features=20, tfidf_min_df=1,
    )
    tfidf_cols = [f for f in feature_names if f.startswith("tfidf_")]
    assert len(tfidf_cols) > 0
    assert "tfidf_vectorizer" in fitted
    assert list(y) == [1, 1, 1, 0, 0, 0]


def test_tfidf_vocabulary_fit_only_on_train_mask():
    """A word that appears ONLY in a test-only row must not enter the
    vocabulary - fitting on the full pool would leak test-set vocabulary
    into featurization itself (the exact concern that motivated
    train_mask in the first place, see module docstring)."""
    df = pd.DataFrame({
        "source": ["SMPP"] * 4, "record_id": [str(i) for i in range(4)],
        "text": ["alpha beta", "alpha beta", "alpha beta", "onlyintestrow uniqueword"],
        "dcs": [0.0] * 4, "text_decode_failed": [False] * 4,
        **{c: [0] * 4 for c in BEHAVIORAL_COLS},
        "rule_flagged": [True, True, False, False],
    })
    train_mask = np.array([True, True, True, False])  # row 3 (the unique-vocab row) is test-only
    _, _, feature_names, fitted = build_feature_matrix(
        df, train_mask=train_mask, use_tfidf=True, tfidf_min_df=1,
    )
    vocab = set(fitted["tfidf_vectorizer"].vocabulary_.keys())
    assert "onlyintestrow" not in vocab
    assert "alpha" in vocab


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
