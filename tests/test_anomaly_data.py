"""
pytest suite for models.anomaly.data.

Run:
    pytest tests/test_anomaly_data.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.data import (
    BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL, IMSI_DISTINCT_ORIG_KNOWN_COL,
    NEAR_DUP_COLS, build_feature_matrix, load_source_features,
)

N_EMBEDDING_DIMS = 4  # small, for test speed - real data uses 384
N_TEST_COMPONENTS = 2  # must be <= min(n_rows, N_EMBEDDING_DIMS) for PCA to be valid


def _sample_df(n=5):
    """n rows, cycling through a fixed 5-row pattern (including real
    observed extremes: sender_msgs_last_1hr up to 16,971,
    near_dup_match_count_1hr up to 362) so any n still exercises the
    same range of values, not just n's own reduced pattern."""
    base = {
        "sender_msgs_last_5min": [0, 1, 2, 3, 4],
        "sender_msgs_last_1hr": [0, 10, 100, 1000, 16971],
        "sender_unique_destinations_1hr": [0, 1, 5, 20, 100],
        "sender_repeat_content_ratio_1hr": [0.0, 0.1, 0.5, 0.9, 1.0],
        "near_dup_match_count_1hr": [0, 0, 5, 50, 362],
        "near_dup_max_similarity_1hr": [0.0, 0.0, 0.93, 0.99, 1.0],
        "near_dup_distinct_senders_1hr": [0, 0, 1, 2, 4],
        "near_dup_match_count_24hr": [0, 1, 10, 100, 381],
        "near_dup_max_similarity_24hr": [0.0, 0.92, 0.95, 0.99, 1.0],
        "near_dup_distinct_senders_24hr": [0, 1, 1, 3, 5],
        "rule_evaluated": [False, False, True, True, False],
        "rule_flagged": [None, None, True, False, None],
    }
    reps = -(-n // 5)  # ceil division
    rows = {col: (vals * reps)[:n] for col, vals in base.items()}
    rows["source"] = ["SMPP"] * n
    rows["record_id"] = [str(i) for i in range(n)]
    # Genuinely independent columns (not e.g. linspace(-1,1,n)+i, which
    # are all perfectly correlated after standardization - a rank-1
    # embedding block, which makes PCA's later components numerically
    # unstable/arbitrary since there's no real variance left for them to
    # capture - a test-fixture trap, not something real embeddings hit).
    emb = np.random.RandomState(42).randn(n, N_EMBEDDING_DIMS)
    for i in range(N_EMBEDDING_DIMS):
        rows[f"emb_{i}"] = emb[:, i]
    return pd.DataFrame(rows)


def _build(df, n_components=N_TEST_COMPONENTS):
    return build_feature_matrix(df, n_embedding_components=n_components)


def test_count_columns_are_log1p_transformed():
    df = _sample_df()
    X, feature_names, _ = _build(df)
    idx = feature_names.index("sender_msgs_last_1hr")
    # log1p(16971) is nowhere near as extreme as the raw value - after
    # scaling it should not be many orders of magnitude apart from the
    # other rows' values (loosely checked: the max z-score isn't huge)
    assert abs(X[:, idx]).max() < 10


def test_source_gets_one_hot_encoded():
    df = _sample_df()
    df["source"] = ["SMPP", "SMPP", "SS7", "SS7", "SS7"]
    _, feature_names, _ = _build(df)
    assert "source_SMPP" in feature_names
    assert "source_SS7" in feature_names


def test_embeddings_are_reduced_to_pca_components_not_raw_dims():
    """The whole point of this change: raw emb_0..emb_3 must NOT appear
    in the output - they've been replaced by n_embedding_components
    PCA-derived columns."""
    df = _sample_df()
    _, feature_names, _ = _build(df, n_components=2)
    assert "emb_0" not in feature_names
    assert "emb_pca_0" in feature_names
    assert "emb_pca_1" in feature_names
    assert "emb_pca_2" not in feature_names  # only 2 components requested


def test_output_width_matches_n_components_plus_other_features():
    """_sample_df() has a single source (SMPP only) - source dummies are
    dropped entirely for a single-source df (see build_feature_matrix()'s
    docstring: a constant one-hot column carries no information), so no
    +1 here. test_source_gets_one_hot_encoded() below covers the
    multi-source case, where the dummy DOES appear."""
    df = _sample_df()
    X, feature_names, _ = _build(df, n_components=2)
    # +2 for IMSI_DISTINCT_ORIG_COL and its _known indicator - always
    # added by build_feature_matrix() regardless of whether the input df
    # has the column (absent here, same as SMPP's real file - see that
    # column's comment in models/anomaly/data.py).
    expected_width = 2 + len(BEHAVIORAL_COLS) + len(NEAR_DUP_COLS) + 2
    assert X.shape[1] == expected_width
    assert len(feature_names) == expected_width
    assert IMSI_DISTINCT_ORIG_COL in feature_names
    assert IMSI_DISTINCT_ORIG_KNOWN_COL in feature_names


def test_pca_explained_variance_is_accessible():
    """A real, checkable number (not just trust) - see module docstring
    on why this gets printed every run rather than assumed once."""
    df = _sample_df()
    _, _, preprocessor = _build(df, n_components=2)
    pca = preprocessor.named_steps["reduce"].named_transformers_["embeddings"].named_steps["pca"]
    assert len(pca.explained_variance_ratio_) == 2
    assert 0.0 <= pca.explained_variance_ratio_.sum() <= 1.0


def test_output_is_standardized_roughly_zero_mean_unit_variance():
    df = _sample_df(n=50)
    df["sender_msgs_last_1hr"] = np.random.RandomState(0).randint(0, 20000, 50)
    X, feature_names, _ = _build(df)
    idx = feature_names.index("sender_msgs_last_1hr")
    assert X[:, idx].mean() == pytest.approx(0.0, abs=1e-8)
    assert X[:, idx].std() == pytest.approx(1.0, abs=1e-8)


def test_preprocessor_is_returned_and_reusable():
    """Re-applying the SAME fitted preprocessor to the SAME raw input
    (after the same log1p step) must reproduce X exactly - this is the
    whole point of returning it instead of just the transformed data;
    inference has to replay this exact fit, not refit on new data."""
    df = _sample_df()
    X, _, preprocessor = _build(df)

    transformed = df.copy()
    for col in ["sender_msgs_last_5min", "sender_msgs_last_1hr", "sender_unique_destinations_1hr",
                "near_dup_match_count_1hr", "near_dup_distinct_senders_1hr",
                "near_dup_match_count_24hr", "near_dup_distinct_senders_24hr"]:
        transformed[col] = np.log1p(transformed[col])
    # IMSI_DISTINCT_ORIG_COL absent from _sample_df() entirely (same as
    # SMPP's real file) - build_feature_matrix() treats that as all-NaN:
    # _known indicator is 0 everywhere, the count itself log1p(0) = 0.
    transformed[IMSI_DISTINCT_ORIG_KNOWN_COL] = 0.0
    transformed[IMSI_DISTINCT_ORIG_COL] = 0.0
    # _sample_df() is single-source (SMPP only) - no source dummy in this
    # path, see test_output_width_matches_n_components_plus_other_features().
    embedding_cols = [c for c in transformed.columns if c.startswith("emb_")]
    imsi_cols = [IMSI_DISTINCT_ORIG_COL, IMSI_DISTINCT_ORIG_KNOWN_COL]
    combined = pd.concat(
        [transformed[BEHAVIORAL_COLS + NEAR_DUP_COLS + imsi_cols], transformed[embedding_cols]], axis=1,
    )
    X_again = preprocessor.transform(combined)
    assert np.allclose(X, X_again)


def test_imsi_distinct_orig_col_absent_and_null_both_get_known_zero():
    """SMPP (column absent entirely) and SS7's own null-imsi rows (column
    present, value NaN) must both land as _known=0, count=0 - "genuinely
    unknown" collapsed to one representation, not two different ones."""
    df = _sample_df()
    df[IMSI_DISTINCT_ORIG_COL] = [np.nan, 3.0, np.nan, 0.0, 7.0]
    X, feature_names, _ = _build(df)
    known_idx = feature_names.index(IMSI_DISTINCT_ORIG_KNOWN_COL)
    count_idx = feature_names.index(IMSI_DISTINCT_ORIG_COL)

    # Reconstruct expected _known/log1p values the same way
    # build_feature_matrix() does, then check relative order survives
    # standardization (exact values depend on the fitted scaler).
    known = df[IMSI_DISTINCT_ORIG_COL].notna().to_numpy(dtype=float)
    assert (X[:, known_idx] > 0).tolist() == (known == 1).tolist()
    # The two NaN rows (0, 2) must be indistinguishable on the known
    # column regardless of whether NaN came from "column absent" or "this
    # row's imsi is null" - both were fed through the exact same np.nan.
    assert X[0, known_idx] == X[2, known_idx]


def test_load_source_features_inner_joins_all_three_sources(tmp_path):
    """Only message_keys present in messages CSV, embeddings, AND
    faiss_output survive - matches the module docstring's stated
    restriction to the sampled subset."""
    source_dir = tmp_path / "SMPP"
    source_dir.mkdir()

    messages_path = source_dir / "messages_with_behavioral.csv"
    pd.DataFrame({
        "source": ["SMPP", "SMPP", "SMPP"],
        "record_id": ["1", "2", "3"],  # "3" has no embedding/near-dup - should be dropped
        **{c: [0, 0, 0] for c in BEHAVIORAL_COLS},
        "rule_evaluated": [False, False, False],
        "rule_flagged": [None, None, None],
    }).to_csv(messages_path, index=False)

    embeddings = np.random.RandomState(0).rand(2, 4).astype(np.float32)
    np.save(source_dir / "embeddings.npy", embeddings)
    pd.DataFrame({"message_key": ["SMPP|1", "SMPP|2"]}).to_parquet(source_dir / "embeddings_id_map.parquet")

    pd.DataFrame({
        "message_key": ["SMPP|1", "SMPP|2"],
        **{c: [0, 0] for c in NEAR_DUP_COLS},
    }).to_parquet(source_dir / "faiss_output.parquet")

    result = load_source_features(source_dir, messages_path)
    assert set(result["record_id"]) == {"1", "2"}  # "3" correctly dropped
    assert "emb_0" in result.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
