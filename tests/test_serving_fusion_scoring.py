"""
pytest suite for serving/fusion_scoring.py's score_fusion() - populates the
module's champion cache directly with a small real fitted pipeline (same
shape models/decision_fusion/train.py logs), same "no real MLflow registry
needed" convention as tests/test_serving_scoring.py's build_rule_pattern_row
tests. _load_champion() itself (the MLflow registry lookup) is not
exercised here.

Run:
    pytest tests/test_serving_fusion_scoring.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serving.fusion_scoring as fusion_scoring
from serving.fusion_scoring import score_fusion


def _fit_pipeline() -> Pipeline:
    """A tiny, real fitted pipeline - rule_pattern_score and anomaly_score
    both positively correlated with the label, same sign as
    models/decision_fusion/train.py's real measured coefficients (both
    SMPP and SS7 fits had positive coefficients on both features)."""
    X = np.array([
        [0.9, 0.1], [0.8, 0.05], [0.95, 0.12], [0.1, -0.05],
        [0.05, -0.08], [0.02, -0.1], [0.5, 0.09], [0.4, -0.09],
    ])
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(autouse=True)
def _reset_cache():
    fusion_scoring.reset_cache()
    yield
    fusion_scoring.reset_cache()


def test_score_fusion_uses_cached_champion_directly():
    fusion_scoring._cached["SS7"] = fusion_scoring._LoadedFusionModel(
        pipeline=_fit_pipeline(),
        feature_names=["rule_pattern_score", "anomaly_score"],
        version="1",
    )
    fusion_score, version = score_fusion("SS7", rule_pattern_score=0.9, anomaly_score=0.1)
    assert 0.0 <= fusion_score <= 1.0
    assert version == "1"


def test_high_rule_pattern_and_anomaly_score_high_fusion_score():
    """Both raw scores agreeing at the high end should push fusion_score
    high too - sanity check on sign, not a tuned threshold."""
    fusion_scoring._cached["SS7"] = fusion_scoring._LoadedFusionModel(
        pipeline=_fit_pipeline(), feature_names=["rule_pattern_score", "anomaly_score"], version="1",
    )
    high, _ = score_fusion("SS7", rule_pattern_score=0.95, anomaly_score=0.12)
    low, _ = score_fusion("SS7", rule_pattern_score=0.02, anomaly_score=-0.1)
    assert high > low


def test_feature_order_from_champion_is_respected_even_if_reversed():
    """feature_names.json is authoritative (same convention as
    serving/scoring.py) - if a future champion logs [anomaly_score,
    rule_pattern_score] order, this module must not silently swap them."""
    fusion_scoring._cached["SS7"] = fusion_scoring._LoadedFusionModel(
        pipeline=_fit_pipeline(), feature_names=["anomaly_score", "rule_pattern_score"], version="1",
    )
    # pipeline was fit on [rule_pattern_score, anomaly_score] order, but
    # feature_names says the reverse - score_fusion must build X in
    # feature_names' order regardless of the kwargs' own order.
    score_reversed_names, _ = score_fusion("SS7", rule_pattern_score=0.02, anomaly_score=0.95)
    # X sent to the pipeline is [anomaly_score, rule_pattern_score] = [0.95, 0.02]
    # per feature_names above - confirms score_fusion doesn't hardcode order.
    expected = _fit_pipeline().predict_proba([[0.95, 0.02]])[:, 1][0]
    assert score_reversed_names == pytest.approx(expected, rel=1e-6)


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
