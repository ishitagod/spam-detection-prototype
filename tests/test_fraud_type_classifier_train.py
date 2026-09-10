"""
pytest suite for models.fraud_type_classifier.train's pure logic
(drop_rare_classes) - MLflow-logging/CLI orchestration in run() is not
unit-tested here, same convention as models/anomaly/train.py's own run().

Run:
    pytest tests/test_fraud_type_classifier_train.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fraud_type_classifier.train import drop_rare_classes


def test_drop_rare_classes_removes_below_threshold():
    df = pd.DataFrame({
        "fraud_type_label": ["a"] * 5 + ["b"] * 2 + ["c"] * 10,
    })
    out = drop_rare_classes(df, min_class_count=3)
    assert set(out["fraud_type_label"]) == {"a", "c"}
    assert "b" not in set(out["fraud_type_label"])


def test_drop_rare_classes_boundary_is_inclusive():
    df = pd.DataFrame({"fraud_type_label": ["a"] * 3 + ["b"] * 2})
    out = drop_rare_classes(df, min_class_count=3)
    assert set(out["fraud_type_label"]) == {"a"}


def test_drop_rare_classes_keeps_everything_when_all_above_threshold():
    df = pd.DataFrame({"fraud_type_label": ["a"] * 5 + ["b"] * 5})
    out = drop_rare_classes(df, min_class_count=3)
    assert len(out) == len(df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
