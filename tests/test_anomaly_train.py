"""
pytest suite for models.anomaly.train.evaluate_against_rule_labels() -
the anomaly-specific wrapper. The underlying PR-AUC/per-source logic it
delegates to is tested generically in tests/test_metrics.py; this file
only covers what's unique to this wrapper: deriving y_true from
rule_evaluated/rule_flagged and filtering to the evaluated subset.

Run:
    pytest tests/test_anomaly_train.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.train import evaluate_against_rule_labels


def test_only_rule_evaluated_rows_are_included():
    """Rows with rule_evaluated=False must not leak into the metric,
    regardless of their (meaningless, since untouched) rule_flagged
    value or anomaly_score."""
    df = pd.DataFrame({
        "source": ["SS7"] * 6,
        "rule_evaluated": [True, True, True, True, False, False],
        "rule_flagged": [True, True, False, False, None, None],
    })
    # if the untouched rows leaked in, n would be 6, not 4
    anomaly_score = np.array([0.9, 0.8, 0.1, 0.2, 999.0, -999.0])
    result = evaluate_against_rule_labels(df, anomaly_score)
    assert result["overall_n"] == 4


def test_rule_flagged_true_becomes_the_positive_class():
    df = pd.DataFrame({
        "source": ["SS7"] * 4,
        "rule_evaluated": [True, True, True, True],
        "rule_flagged": [True, True, False, False],
    })
    anomaly_score = np.array([0.9, 0.8, 0.1, 0.2])  # flagged scored higher
    result = evaluate_against_rule_labels(df, anomaly_score)
    assert result["overall_pr_auc"] == pytest.approx(1.0)


def test_smpp_style_all_flagged_source_is_skipped_ss7_style_mixed_source_computes():
    """The real scenario this whole thing was built for."""
    df = pd.DataFrame({
        "source": ["SMPP"] * 3 + ["SS7"] * 4,
        "rule_evaluated": [True] * 7,
        "rule_flagged": [True, True, True, True, True, False, False],
    })
    anomaly_score = np.array([0.5, 0.6, 0.55, 0.9, 0.8, 0.1, 0.2])
    result = evaluate_against_rule_labels(df, anomaly_score)
    assert not any(k.startswith("SMPP_") for k in result)
    assert "SS7_pr_auc" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
