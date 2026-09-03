"""
pytest suite for labels.cluster_labels - is_cluster_confirmed() /
build_cluster_labels(). Operates directly on small synthetic
clusters_df/template_df-shaped frames, independent of any real
cluster_discovery.py/inspect_clusters.py output.

Run:
    pytest tests/test_cluster_labels.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labels.cluster_labels import build_cluster_labels, is_cluster_confirmed


@pytest.fixture
def clusters_df() -> pd.DataFrame:
    """message_key/cluster_label pairs as models/anomaly/cluster_discovery.py
    writes them - two messages in cluster 0, two in cluster 1, one in noise
    (-1)."""
    return pd.DataFrame([
        {"message_key": "SS7|1", "cluster_label": 0, "anomaly_score": 0.9},
        {"message_key": "SS7|2", "cluster_label": 0, "anomaly_score": 0.91},
        {"message_key": "SS7|3", "cluster_label": 1, "anomaly_score": 0.95},
        {"message_key": "SS7|4", "cluster_label": 1, "anomaly_score": 0.96},
        {"message_key": "SS7|5", "cluster_label": -1, "anomaly_score": 0.99},
    ])


@pytest.fixture
def template_df() -> pd.DataFrame:
    """models/anomaly/inspect_clusters.py's cluster_labels_template.csv
    shape - one row per cluster (noise included), fraud_type_label blank
    by default."""
    return pd.DataFrame([
        {"cluster_label": 0, "n_rows": 2, "fraud_type_label": "flooding_burst"},
        {"cluster_label": 1, "n_rows": 2, "fraud_type_label": ""},  # reviewed, left blank
        {"cluster_label": -1, "n_rows": 1, "fraud_type_label": None},  # never reviewed
    ])


def test_is_cluster_confirmed_true_for_real_name(template_df):
    assert is_cluster_confirmed(template_df).tolist() == [True, False, False]


def test_is_cluster_confirmed_false_for_nan():
    df = pd.DataFrame({"fraud_type_label": [None]})
    assert is_cluster_confirmed(df).tolist() == [False]


def test_is_cluster_confirmed_false_for_empty_string():
    df = pd.DataFrame({"fraud_type_label": [""]})
    assert is_cluster_confirmed(df).tolist() == [False]


def test_is_cluster_confirmed_false_for_whitespace_only():
    """A cell containing only spaces (e.g. accidentally typed, or how some
    spreadsheet tools represent a cleared cell) must be treated the same
    as truly empty - not confirmed."""
    df = pd.DataFrame({"fraud_type_label": ["   "]})
    assert is_cluster_confirmed(df).tolist() == [False]


def test_build_cluster_labels_excludes_blank_and_nan_rows(clusters_df, template_df):
    """Only cluster 0's messages are confirmed - cluster 1 (blank) and
    cluster -1 (never reviewed/NaN) must not appear at all, never
    fabricated."""
    result = build_cluster_labels(clusters_df, template_df)
    assert set(result["message_key"]) == {"SS7|1", "SS7|2"}


def test_build_cluster_labels_joins_correct_label_by_cluster(clusters_df, template_df):
    result = build_cluster_labels(clusters_df, template_df)
    assert (result["cluster_fraud_type_label"] == "flooding_burst").all()
    assert set(result.columns) == {"message_key", "cluster_label", "cluster_fraud_type_label"}


def test_build_cluster_labels_noise_confirmed_flows_through_like_any_cluster(clusters_df):
    """Cluster -1 (DBSCAN noise) is not special-cased or excluded just
    because it's noise - if a human confirmed it, it's a real label."""
    template = pd.DataFrame([
        {"cluster_label": 0, "fraud_type_label": ""},
        {"cluster_label": 1, "fraud_type_label": ""},
        {"cluster_label": -1, "fraud_type_label": "genuine_one_off_scam"},
    ])
    result = build_cluster_labels(clusters_df, template)
    assert set(result["message_key"]) == {"SS7|5"}
    assert result["cluster_fraud_type_label"].iloc[0] == "genuine_one_off_scam"


def test_build_cluster_labels_multiple_confirmed_clusters_each_get_own_label(clusters_df):
    """Two different cluster_labels, each with their own distinct confirmed
    name, both flow through correctly - not just the first one found."""
    template = pd.DataFrame([
        {"cluster_label": 0, "fraud_type_label": "flooding_burst"},
        {"cluster_label": 1, "fraud_type_label": "phishing_template_rotating_url"},
        {"cluster_label": -1, "fraud_type_label": ""},
    ])
    result = build_cluster_labels(clusters_df, template)
    by_message = result.set_index("message_key")["cluster_fraud_type_label"]
    assert by_message["SS7|1"] == "flooding_burst"
    assert by_message["SS7|2"] == "flooding_burst"
    assert by_message["SS7|3"] == "phishing_template_rotating_url"
    assert by_message["SS7|4"] == "phishing_template_rotating_url"
    assert "SS7|5" not in by_message.index


def test_build_cluster_labels_raises_on_mismatched_cluster_universe(clusters_df):
    """template_df confirms cluster_label 7, which doesn't exist anywhere
    in clusters_df - a strong signal the two inputs come from different
    cluster_discovery.py runs (cluster ids aren't stable across reruns).
    Must raise, not silently drop those confirmed rows out of the join."""
    template = pd.DataFrame([
        {"cluster_label": 7, "fraud_type_label": "smishing_otp_bait"},
    ])
    with pytest.raises(ValueError):
        build_cluster_labels(clusters_df, template)


def test_build_cluster_labels_unconfirmed_unknown_cluster_does_not_raise(clusters_df):
    """The mismatch guard only fires for CONFIRMED rows - an unconfirmed
    (blank) row referencing a cluster_label absent from clusters_df is not
    itself proof of a run mismatch (e.g. a cluster that produced zero
    candidate rows in a filtered view) and must not block otherwise-valid
    confirmed labels elsewhere in the same template."""
    template = pd.DataFrame([
        {"cluster_label": 0, "fraud_type_label": "flooding_burst"},
        {"cluster_label": 99, "fraud_type_label": ""},  # unconfirmed, unknown - fine
    ])
    result = build_cluster_labels(clusters_df, template)
    assert set(result["message_key"]) == {"SS7|1", "SS7|2"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
