"""
pytest suite for models.anomaly.ingest_cluster_labels's file-I/O/dedup
logic (accumulate_labels). Uses tmp_path so real data/processed/ files are
never touched - same pattern as tests/test_compare_versions.py's
tmp_path-scoped mlflow store.

Run:
    pytest tests/test_ingest_cluster_labels.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.ingest_cluster_labels import accumulate_labels


def _labels(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["message_key", "cluster_label", "cluster_fraud_type_label"])


def test_accumulate_labels_creates_file_when_none_exists(tmp_path):
    out_path = tmp_path / "cluster_labels.parquet"
    new = _labels([["SS7|1", 0, "flooding_burst"], ["SS7|2", 0, "flooding_burst"]])
    combined, n_new = accumulate_labels(new, out_path)
    assert n_new == 2
    assert len(combined) == 2


def test_accumulate_labels_appends_new_messages_across_runs(tmp_path):
    """A later run's confirmed labels for DIFFERENT messages must ADD to
    the accumulated file, not replace it - simulates two separate
    cluster_discovery.py -> inspect_clusters.py -> ingest sessions."""
    out_path = tmp_path / "cluster_labels.parquet"
    run1 = _labels([["SS7|1", 0, "flooding_burst"]])
    combined1, n_new1 = accumulate_labels(run1, out_path)
    combined1.to_parquet(out_path, index=False)
    assert n_new1 == 1

    run2 = _labels([["SS7|2", 5, "phishing_template_rotating_url"]])
    combined2, n_new2 = accumulate_labels(run2, out_path)
    assert n_new2 == 1
    assert set(combined2["message_key"]) == {"SS7|1", "SS7|2"}


def test_accumulate_labels_dedups_same_message_key_keeps_newest(tmp_path):
    """The same message_key confirmed in an earlier run AND a later run
    must end up as exactly one row - and the later run's label wins (see
    accumulate_labels' KEEP-NEWEST docstring reasoning)."""
    out_path = tmp_path / "cluster_labels.parquet"
    run1 = _labels([["SS7|1", 0, "flooding_burst"]])
    combined1, _ = accumulate_labels(run1, out_path)
    combined1.to_parquet(out_path, index=False)

    # Re-clustered later, same message now lands in a differently-numbered
    # cluster, hand-labeled with a different (presumably corrected) name.
    run2 = _labels([["SS7|1", 3, "smishing_otp_bait"]])
    combined2, n_new2 = accumulate_labels(run2, out_path)
    assert n_new2 == 0  # not a NEW message_key, just a re-confirmation
    assert len(combined2) == 1
    row = combined2.iloc[0]
    assert row["cluster_fraud_type_label"] == "smishing_otp_bait"
    assert row["cluster_label"] == 3


def test_accumulate_labels_n_new_counts_only_first_time_message_keys(tmp_path):
    out_path = tmp_path / "cluster_labels.parquet"
    run1 = _labels([["SS7|1", 0, "flooding_burst"], ["SS7|2", 0, "flooding_burst"]])
    combined1, _ = accumulate_labels(run1, out_path)
    combined1.to_parquet(out_path, index=False)

    # One re-confirmed message_key (SS7|1) + one genuinely new one (SS7|3).
    run2 = _labels([["SS7|1", 3, "flooding_burst"], ["SS7|3", 3, "flooding_burst"]])
    combined2, n_new2 = accumulate_labels(run2, out_path)
    assert n_new2 == 1
    assert len(combined2) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
