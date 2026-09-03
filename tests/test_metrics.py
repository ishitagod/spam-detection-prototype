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

from models.metrics import (
    evaluate_overall_and_per_source, evaluate_precision_at_k, precision_at_k,
    precision_at_k_percentiles, pr_auc_and_log_loss,
)


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


def test_precision_at_k_returns_none_outside_valid_range():
    y_true = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    assert precision_at_k(y_true, scores, k=0) is None
    assert precision_at_k(y_true, scores, k=5) is None  # > len(y_true)


def test_precision_at_k_computes_over_top_k_by_score():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.15])  # top 3 by score: idx 1,3,4 -> all positive
    result = precision_at_k(y_true, scores, k=3)
    assert result["precision"] == pytest.approx(1.0)
    assert result["k"] == 3
    assert result["n_positive_in_k"] == 3


def test_precision_at_k_ignores_score_order_beyond_k():
    """Only the top k by score matter - a low-scoring positive outside
    the cutoff must not affect precision@k."""
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])  # top 2 by score: idx 0 (pos), idx 2 (neg)
    result = precision_at_k(y_true, scores, k=2)
    assert result["precision"] == pytest.approx(0.5)


def test_precision_at_k_percentiles_rounds_percentile_to_k_and_skips_out_of_range():
    y_true = np.array([1] * 10 + [0] * 90)  # n=100
    scores = np.concatenate([np.full(10, 0.9), np.full(90, 0.1)])
    result = precision_at_k_percentiles(y_true, scores, percentiles=(1.0, 50.0))
    # top 1% of 100 = k=1, top 50% of 100 = k=50
    assert result["precision_at_top_1_0pct_k"] == 1
    assert result["precision_at_top_1_0pct_precision"] == pytest.approx(1.0)
    assert result["precision_at_top_50_0pct_k"] == 50


def test_precision_at_k_percentiles_skips_percentile_that_rounds_to_zero():
    y_true = np.array([1, 0, 1, 0])  # n=4 - 0.1% of 4 rounds to 0
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    result = precision_at_k_percentiles(y_true, scores, percentiles=(0.1,))
    assert result == {}  # skipped, not a fabricated k=0 or k=1 reading


def test_evaluate_precision_at_k_reports_overall_and_per_source():
    smpp = _labelled_df(source_flagged=1, source_clean=0, source_name="SMPP")
    ss7 = _labelled_df(source_flagged=5, source_clean=5, source_name="SS7")
    df = pd.concat([smpp, ss7], ignore_index=True)
    y_true = np.array([1,  # SMPP
                        1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # SS7
    smpp_scores = np.array([0.5])
    ss7_scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.15, 0.1, 0.05, 0.25])
    score = np.concatenate([smpp_scores, ss7_scores])

    result = evaluate_precision_at_k(df, "source", y_true, score, percentiles=(50.0,))
    # SS7 top-50% (k=5) by score: idx 0-4, all flagged -> precision 1.0
    assert result["SS7_precision_at_top_50_0pct_precision"] == pytest.approx(1.0)
    assert "overall_precision_at_top_50_0pct_precision" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
