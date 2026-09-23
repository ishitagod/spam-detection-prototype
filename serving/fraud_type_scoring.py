"""
Loads the fraud_type_classifier champion (LGBMClassifier, multiclass,
models/fraud_type_classifier/train.py) and scores one already-built
rule_pattern row - same per-source champion-cache pattern as
serving/scoring.py, serving/anomaly_scoring.py, serving/fusion_scoring.py.

No champion exists for a source until a real, CONFIRMED-label run
(`--label_source confirmed`) has been trained and promoted - see that
module's docstring. Raises ChampionUnavailableError until then, same
convention as the other scorers; serving/app.py treats this as
best-effort and additive, never gating prediction/recommended_action.
"""
import logging
import time
from dataclasses import dataclass

import mlflow
import mlflow.lightgbm
import numpy as np

from models.registry import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

FRAUD_TYPE_MODEL_NAME = "fraud_type_classifier"  # base name - actual
# registered model is source-suffixed (fraud_type_classifier_SMPP / _SS7).
CHAMPION_ALIAS = "champion"


class ChampionUnavailableError(RuntimeError):
    """No model currently holds CHAMPION_ALIAS for this source's
    registered name - expected until a --label_source confirmed run has
    been trained and promoted for it."""


@dataclass
class _LoadedFraudTypeModel:
    model: object  # LGBMClassifier, loaded via mlflow.lightgbm.load_model
    feature_names: list[str]
    version: str


_cached: dict[str, _LoadedFraudTypeModel] = {}  # keyed by source, load-once


def _load_champion(source: str) -> _LoadedFraudTypeModel:
    if source in _cached:
        return _cached[source]

    load_start = time.perf_counter()
    logger.info("loading fraud_type_classifier champion for source=%s (cold cache)", source)
    registered_name = f"{FRAUD_TYPE_MODEL_NAME}_{source}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException as e:
        raise ChampionUnavailableError(
            f"No {CHAMPION_ALIAS!r} version registered for {registered_name!r} - "
            f"run `python -m models.fraud_type_classifier.train --source {source} "
            f"--label_source confirmed` then `python -m models.compare_versions "
            f"--experiment_name {FRAUD_TYPE_MODEL_NAME} --registered_name {registered_name} "
            f"--metric_key test_macro_f1` to promote one."
        ) from e

    model = mlflow.lightgbm.load_model(f"models:/{registered_name}@{CHAMPION_ALIAS}")
    feature_names = mlflow.artifacts.load_dict(
        f"runs:/{version.run_id}/feature_names.json"
    )["feature_names"]

    _cached[source] = _LoadedFraudTypeModel(
        model=model, feature_names=feature_names, version=str(version.version),
    )
    logger.info(
        "loaded fraud_type_classifier champion v%s for source=%s in %dms",
        version.version, source, int((time.perf_counter() - load_start) * 1000),
    )
    return _cached[source]


KNOWN_SOURCES = ["SMPP", "SS7"]


def preload() -> None:
    """Warms _cached for every known source at process startup - a source
    with no promoted fraud_type_classifier champion yet is expected and
    must not block startup, same reasoning as fusion_scoring.preload()."""
    for source in KNOWN_SOURCES:
        try:
            _load_champion(source)
        except ChampionUnavailableError as e:
            logger.warning("preload: fraud_type_classifier champion unavailable for source=%s: %s", source, e)


def reset_cache() -> None:
    """Test hook - forces the next score_fraud_type() call to reload from
    MLflow instead of reusing whatever this process already cached."""
    global _cached
    _cached = {}


def score_fraud_type(source: str, row: dict) -> tuple[str, float, str]:
    """Returns (fraud_type_label, confidence, model_version) -
    confidence is the winning class's predict_proba, 0-1. Raises
    ChampionUnavailableError if no champion exists for `source` yet."""
    loaded = _load_champion(source)
    X = np.array([[row[f] for f in loaded.feature_names]], dtype=np.float64)
    proba = loaded.model.predict_proba(X)[0]
    classes = loaded.model.classes_
    top_idx = int(np.argmax(proba))
    return str(classes[top_idx]), float(proba[top_idx]), loaded.version
