"""
pytest suite for serving/fraud_type_scoring.py's score_fraud_type() -
populates the module's champion cache directly with a small real fitted
LGBMClassifier, same "no real MLflow registry needed" convention as
tests/test_serving_fusion_scoring.py.

Run:
    pytest tests/test_serving_fraud_type_scoring.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serving.fraud_type_scoring as fraud_type_scoring
from serving.fraud_type_scoring import score_fraud_type


def _fit_model() -> LGBMClassifier:
    X = np.array([
        [1.0, 0.0], [1.1, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 1.1], [0.0, 0.9],
    ])
    y = np.array(["gambling_promo", "gambling_promo", "gambling_promo",
                  "loan_scam", "loan_scam", "loan_scam"])
    model = LGBMClassifier(n_estimators=10, min_child_samples=1, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(autouse=True)
def _reset_cache():
    fraud_type_scoring.reset_cache()
    yield
    fraud_type_scoring.reset_cache()


def test_score_fraud_type_uses_cached_champion_directly():
    fraud_type_scoring._cached["SS7"] = fraud_type_scoring._LoadedFraudTypeModel(
        model=_fit_model(), feature_names=["has_gambling_keyword", "has_loan_keyword"], version="1",
    )
    label, confidence, version = score_fraud_type(
        "SS7", {"has_gambling_keyword": 1.0, "has_loan_keyword": 0.0},
    )
    assert label == "gambling_promo"
    assert 0.0 <= confidence <= 1.0
    assert version == "1"


def test_feature_order_from_champion_is_respected():
    """feature_names.json is authoritative, same convention as
    serving/fusion_scoring.py - score_fraud_type must build X in
    feature_names' order, not the row dict's own key order."""
    fraud_type_scoring._cached["SS7"] = fraud_type_scoring._LoadedFraudTypeModel(
        model=_fit_model(), feature_names=["has_loan_keyword", "has_gambling_keyword"], version="1",
    )
    row = {"has_gambling_keyword": 0.0, "has_loan_keyword": 1.0}
    label, _, _ = score_fraud_type("SS7", row)
    assert label == "gambling_promo"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
