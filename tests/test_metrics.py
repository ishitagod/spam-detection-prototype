"""
pytest suite for models.metrics - shared by models/anomaly/train.py and
models/rule_pattern/train.py.

Run:
    pytest tests/test_metrics.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.metrics import evaluate_overall_and_per_source, pr_auc_and_log_loss


def test_pr_auc_returns_none_when_only_one_class_present():
    """The exact real-world case: SMPP has zero confirmed-clean labels -
    PR-AUC is mathematically undefined with only one class, must be
    skipped, not silently computed wrong."""
    y_true = np.array([1, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.3, 0.9])
    assert pr_auc_and_log_loss(y_true, scores) is None


def test_pr_auc_computes_when_both_classes_present():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])  # perfectly separable
    result = pr_auc_and_log_loss(y_true, scores)
    assert result is not None
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["n"] == 4
    assert result["n_positive"] == 2


def test_pr_auc_is_high_when_score_correctly_ranks_positives_above_negatives():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    result = pr_auc_and_log_loss(y_true, scores)
    assert result["pr_auc"] > 0.9


def test_pr_auc_is_low_when_score_ranks_backwards():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])  # inverted
    result = pr_auc_and_log_loss(y_true, scores)
    assert result["pr_auc"] < 0.5


def _labelled_df(source_flagged, source_clean, source_name):
    n = source_flagged + source_clean
    return pd.DataFrame({"source": [source_name] * n})


def test_evaluate_skips_a_source_with_only_one_class_but_computes_the_other():
    """Mirrors the real SMPP (all-flagged, no clean) vs SS7 (both
    classes) situation exactly."""
    smpp = _labelled_df(source_flagged=5, source_clean=0, source_name="SMPP")
    ss7 = _labelled_df(source_flagged=5, source_clean=5, source_name="SS7")
    df = pd.concat([smpp, ss7], ignore_index=True)

    y_true = np.array([1, 1, 1, 1, 1,  # SMPP - all positive
                        1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # SS7 - both classes
    smpp_scores = np.random.RandomState(0).rand(5)  # arbitrary - SMPP has no clean class to rank against
    ss7_scores = np.array([0.8, 0.9, 0.85, 0.75, 0.7,  # 5 flagged - high scores
                            0.1, 0.2, 0.15, 0.25, 0.05])  # 5 clean - low scores
    score = np.concatenate([smpp_scores, ss7_scores])

    result = evaluate_overall_and_per_source(df, "source", y_true, score)

    assert not any(k.startswith("SMPP_") for k in result)  # skipped - only one class
    assert "SS7_pr_auc" in result
    assert result["SS7_pr_auc"] > 0.9  # SS7's scores were constructed to rank correctly


def test_evaluate_overall_combines_all_sources():
    smpp = _labelled_df(source_flagged=3, source_clean=0, source_name="SMPP")
    ss7 = _labelled_df(source_flagged=3, source_clean=3, source_name="SS7")
    df = pd.concat([smpp, ss7], ignore_index=True)
    y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
    score = np.array([0.6, 0.7, 0.65, 0.8, 0.9, 0.85, 0.1, 0.2, 0.15])

    result = evaluate_overall_and_per_source(df, "source", y_true, score)
    # overall combines all 9 rows - both classes present overall (via
    # SS7's clean rows), so this must compute even though SMPP alone
    # wouldn't.
    assert "overall_pr_auc" in result


def test_prefix_namespaces_metric_keys():
    df = _labelled_df(source_flagged=2, source_clean=2, source_name="SS7")
    y_true = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2])
    result = evaluate_overall_and_per_source(df, "source", y_true, score, prefix="train_")
    assert "train_overall_pr_auc" in result
    assert "overall_pr_auc" not in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
