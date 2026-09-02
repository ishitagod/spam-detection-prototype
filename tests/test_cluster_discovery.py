"""
pytest suite for models.anomaly.cluster_discovery's pure-logic functions
(suggest_eps, select_anomalous_subset, run_dbscan, summarize_clusters).

Mirrors tests/test_anomaly_train.py's convention: the MLflow-logging/file-
I/O orchestration in run() is NOT unit-tested here (same as train.py's own
run() isn't) - it's covered by a manual real-data smoke test instead. Only
the functions with real, checkable logic get synthetic-fixture tests.

Run:
    pytest tests/test_cluster_discovery.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.cluster_discovery import (
    run_dbscan,
    select_anomalous_subset,
    suggest_eps,
    summarize_clusters,
)


def _two_clusters_plus_outlier() -> np.ndarray:
    """
    Two well-separated, tight 2D blobs (around (0,0) and (10,10), 6
    points each) plus one point far from both (100,100) - deterministic,
    easy to reason about expected DBSCAN output on, unlike real
    PCA-reduced embeddings.
    """
    rng = np.random.RandomState(0)
    blob_a = rng.normal(loc=(0, 0), scale=0.1, size=(6, 2))
    blob_b = rng.normal(loc=(10, 10), scale=0.1, size=(6, 2))
    outlier = np.array([[100.0, 100.0]])
    return np.vstack([blob_a, blob_b, outlier])


def test_run_dbscan_groups_each_tight_blob_into_its_own_cluster():
    X = _two_clusters_plus_outlier()
    labels = run_dbscan(X, eps=1.0, min_samples=3, n_jobs=1)
    blob_a_labels = set(labels[:6])
    blob_b_labels = set(labels[6:12])
    assert len(blob_a_labels) == 1  # all of blob A shares one label
    assert len(blob_b_labels) == 1  # all of blob B shares one label
    assert blob_a_labels != blob_b_labels  # but not the SAME label as each other


def test_run_dbscan_far_outlier_is_noise():
    X = _two_clusters_plus_outlier()
    labels = run_dbscan(X, eps=1.0, min_samples=3, n_jobs=1)
    assert labels[-1] == -1


def test_suggest_eps_is_positive_and_smaller_for_tighter_data():
    """Not just 'returns a number' - actually measuring density: a
    tighter blob must suggest a smaller eps than a more spread-out one."""
    rng = np.random.RandomState(0)
    tight = rng.normal(loc=(0, 0), scale=0.1, size=(20, 2))
    spread = rng.normal(loc=(0, 0), scale=5.0, size=(20, 2))
    eps_tight = suggest_eps(tight, min_samples=5)
    eps_spread = suggest_eps(spread, min_samples=5)
    assert eps_tight > 0
    assert eps_tight < eps_spread


def _percentile_test_df() -> tuple[pd.DataFrame, np.ndarray]:
    # SMPP's own 90th percentile would be ~9 (per-source); SS7's would be
    # ~109. A COMBINED percentile (the actual design) sits between them,
    # not equal to either source's own cutoff - this is what distinguishes
    # "combined" from "per-source" in a test rather than by inspection.
    df = pd.DataFrame({
        "source": ["SMPP"] * 10 + ["SS7"] * 10,
        "record_id": [str(i) for i in range(20)],
        "anomaly_score": list(range(10)) + list(range(100, 110)),
    })
    X = np.arange(20).reshape(20, 1).astype(np.float64)
    return df, X


def test_select_anomalous_subset_uses_combined_not_per_source_percentile():
    df, X = _percentile_test_df()
    subset, X_subset = select_anomalous_subset(df, X, percentile=90)
    # Combined 90th percentile of [0..9, 100..109] is ~99.1 - only the top
    # SS7 row(s) clear it; NO SMPP rows should survive (their own max is 9,
    # far below the combined cutoff) - proves the threshold isn't
    # recomputed separately per source.
    assert (subset["source"] == "SMPP").sum() == 0
    assert (subset["source"] == "SS7").sum() > 0


def test_select_anomalous_subset_x_rows_stay_aligned_with_df_rows():
    df, X = _percentile_test_df()
    subset, X_subset = select_anomalous_subset(df, X, percentile=90)
    assert len(subset) == len(X_subset)
    # X was built as arange(20) (X[i] == i), and record_id was str(i) - so
    # X_subset's values must equal subset's OWN record_id, proving each
    # X row travelled with the correct df row through the filter, not
    # just that the counts happen to match.
    assert np.array_equal(X_subset.flatten(), subset["record_id"].astype(float).to_numpy())


def _summary_df_and_labels():
    df = pd.DataFrame({
        "source": ["SMPP", "SMPP", "SS7", "SS7", "SS7"],
        "anomaly_score": [0.5, 0.6, 0.9, 0.95, 0.99],
        "sender_msgs_last_5min": [1, 2, 100, 110, 105],
        "sender_msgs_last_1hr": [1, 2, 100, 110, 105],
        "sender_unique_destinations_1hr": [1, 1, 50, 55, 52],
        "sender_repeat_content_ratio_1hr": [0.0, 0.0, 0.9, 0.95, 0.92],
        "near_dup_match_count_1hr": [0, 0, 40, 45, 42],
        "near_dup_max_similarity_1hr": [0.0, 0.0, 0.98, 0.99, 0.97],
        "near_dup_distinct_senders_1hr": [0, 0, 3, 4, 3],
        "near_dup_match_count_24hr": [0, 0, 40, 45, 42],
        "near_dup_max_similarity_24hr": [0.0, 0.0, 0.98, 0.99, 0.97],
        "near_dup_distinct_senders_24hr": [0, 0, 3, 4, 3],
    })
    labels = np.array([-1, -1, 0, 0, 0])
    return df, labels


def test_summarize_clusters_reports_noise_as_its_own_meaningful_key():
    df, labels = _summary_df_and_labels()
    summary = summarize_clusters(df, labels)
    assert "noise" in summary
    assert summary["noise"]["n_rows"] == 2
    assert "cluster_0" in summary
    assert summary["cluster_0"]["n_rows"] == 3


def test_summarize_clusters_mean_values_are_correct():
    df, labels = _summary_df_and_labels()
    summary = summarize_clusters(df, labels)
    assert summary["cluster_0"]["mean_anomaly_score"] == pytest.approx((0.9 + 0.95 + 0.99) / 3)
    assert summary["noise"]["mean_anomaly_score"] == pytest.approx((0.5 + 0.6) / 2)


def test_summarize_clusters_source_breakdown():
    df, labels = _summary_df_and_labels()
    summary = summarize_clusters(df, labels)
    assert summary["cluster_0"]["source_counts"] == {"SS7": 3}
    assert summary["noise"]["source_counts"] == {"SMPP": 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
