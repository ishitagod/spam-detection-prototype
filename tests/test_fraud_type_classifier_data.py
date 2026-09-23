"""
pytest suite for models.fraud_type_classifier.data - independent of any
real cluster-discovery/labeling output (small synthetic frames).

Run:
    pytest tests/test_fraud_type_classifier_data.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.data import BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL
from models.fraud_type_classifier.data import LABEL_COL, build_feature_matrix, load_cluster_labeled_messages


def _labelled_df(n=6) -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["SMPP"] * n,
        "record_id": [str(i) for i in range(n)],
        "text": ["hello world"] * n,
        "dcs": [0.0] * n,
        "text_decode_failed": [False] * n,
        **{c: [0] * n for c in BEHAVIORAL_COLS},
        IMSI_DISTINCT_ORIG_COL: [np.nan] * n,
        SENDER_VELOCITY_ZSCORE_COL: [0.0] * n,
        LABEL_COL: ["flooding_burst"] * 3 + ["bank_transaction_alert"] * 3,
    })


def test_build_feature_matrix_shapes_and_label_passthrough():
    df = _labelled_df()
    X, y, feature_names = build_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert list(y) == df[LABEL_COL].tolist()
    assert "text_length" in feature_names
    assert "text_decode_failed" in feature_names


def test_build_feature_matrix_text_length_is_derived_correctly():
    df = _labelled_df()
    df["text"] = ["hi", "hello", "hey there", "", "a", "ok"]
    X, _, feature_names = build_feature_matrix(df)
    idx = feature_names.index("text_length")
    assert list(X[:, idx]) == [2, 5, 9, 0, 1, 2]


def test_build_feature_matrix_single_source_drops_source_dummy():
    df = _labelled_df()  # single source (SMPP)
    _, _, feature_names = build_feature_matrix(df)
    assert not any(f.startswith("source_") for f in feature_names)


def test_build_feature_matrix_multi_source_one_hots():
    df = _labelled_df()
    df["source"] = ["SMPP", "SMPP", "SMPP", "SS7", "SS7", "SS7"]
    _, _, feature_names = build_feature_matrix(df)
    assert "source_SMPP" in feature_names and "source_SS7" in feature_names


def test_load_cluster_labeled_messages_suggested_source_joins_correctly(tmp_path):
    source_dir = tmp_path / "SMPP"
    source_dir.mkdir()

    pd.DataFrame({
        "message_key": ["SMPP|1", "SMPP|2", "SMPP|3"],
        "cluster_label": [0, 0, 1],
    }).to_parquet(source_dir / "fraud_type_clusters.parquet")

    pd.DataFrame({
        "cluster_label": [0, 1, -1],
        "suggested_fraud_type_label": ["flooding_burst", "bank_transaction_alert", "incoherent_no_pattern"],
    }).to_csv(source_dir / "cluster_labels_suggested.csv", index=False)

    pd.DataFrame({
        "record_id": ["1", "2", "3", "4"],  # "4" has no cluster - must not appear in output
        "source": ["SMPP"] * 4,
        "text": ["a", "b", "c", "d"],
        "dcs": [0.0] * 4,
        "text_decode_failed": [False] * 4,
        **{c: [0] * 4 for c in BEHAVIORAL_COLS},
    }).to_csv(source_dir / "messages_with_behavioral.csv", index=False)

    df = load_cluster_labeled_messages("SMPP", tmp_path, label_source="suggested")
    assert set(df["record_id"]) == {"1", "2", "3"}
    assert df.set_index("record_id")[LABEL_COL].to_dict() == {
        "1": "flooding_burst", "2": "flooding_burst", "3": "bank_transaction_alert",
    }


def test_load_cluster_labeled_messages_excludes_not_fraud_rows(tmp_path):
    source_dir = tmp_path / "SMPP"
    source_dir.mkdir()

    pd.DataFrame({
        "message_key": ["SMPP|1", "SMPP|2", "SMPP|3"],
        "cluster_label": [0, 1, 2],
    }).to_parquet(source_dir / "fraud_type_clusters.parquet")

    pd.DataFrame({
        "cluster_label": [0, 1, 2],
        "suggested_fraud_type_label": ["flooding_burst", "not_fraud", " Not_Fraud "],
    }).to_csv(source_dir / "cluster_labels_suggested.csv", index=False)

    pd.DataFrame({
        "record_id": ["1", "2", "3"],
        "source": ["SMPP"] * 3,
        "text": ["a", "b", "c"],
        "dcs": [0.0] * 3,
        "text_decode_failed": [False] * 3,
        **{c: [0] * 3 for c in BEHAVIORAL_COLS},
    }).to_csv(source_dir / "messages_with_behavioral.csv", index=False)

    df = load_cluster_labeled_messages("SMPP", tmp_path, label_source="suggested")
    assert set(df["record_id"]) == {"1"}


def test_load_cluster_labeled_messages_raises_clear_error_when_missing(tmp_path):
    (tmp_path / "SMPP").mkdir()
    with pytest.raises(FileNotFoundError):
        load_cluster_labeled_messages("SMPP", tmp_path, label_source="suggested")


def test_load_cluster_labeled_messages_invalid_label_source_raises():
    with pytest.raises(ValueError):
        load_cluster_labeled_messages("SMPP", Path("."), label_source="bogus")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
