"""
Loads the decision-fusion champion (StandardScaler -> LogisticRegression,
models/decision_fusion/train.py) and scores one (rule_pattern_score,
anomaly_score) pair - same per-source champion-cache pattern as
serving/scoring.py and serving/anomaly_scoring.py.

An additive third score, not a replacement for the two it combines (see
serving/app.py - falls back to rule_pattern_score alone when no champion
exists). Both raw scores stay in the response unchanged.

No fusion champion exists for a source until
`python -m models.decision_fusion.train --source <SOURCE>` has run and
been promoted. Raises ChampionUnavailableError, same convention as the
other two scorers.
"""
import logging
import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np

from models.registry import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

FUSION_MODEL_NAME = "decision_fusion_model"  # base name - actual registered
# model is source-suffixed (decision_fusion_model_SMPP / _SS7), same
# convention as RULE_PATTERN_MODEL_NAME/ANOMALY_MODEL_NAME.
CHAMPION_ALIAS = "champion"


class ChampionUnavailableError(RuntimeError):
    """No model currently holds CHAMPION_ALIAS for this source's
    registered name - expected before models/compare_versions.py has
    promoted a fusion champion for it."""


@dataclass
class _LoadedFusionModel:
    pipeline: object  # sklearn Pipeline(StandardScaler, LogisticRegression)
    feature_names: list[str]  # authoritative column order, from feature_names.json
    version: str


_cached: dict[str, _LoadedFusionModel] = {}  # keyed by source, load-once


def _load_champion(source: str) -> _LoadedFusionModel:
    if source in _cached:
        return _cached[source]

    load_start = time.perf_counter()
    logger.info("loading decision_fusion champion for source=%s (cold cache)", source)
    registered_name = f"{FUSION_MODEL_NAME}_{source}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException as e:
        raise ChampionUnavailableError(
            f"No {CHAMPION_ALIAS!r} version registered for {registered_name!r} - "
            f"run `python -m models.decision_fusion.train --source {source}` then "
            f"`python -m models.compare_versions --experiment_name decision_fusion_{source} "
            f"--registered_name {registered_name} --metric_key test_overall_pr_auc` to promote one."
        ) from e

    pipeline = mlflow.sklearn.load_model(f"models:/{registered_name}@{CHAMPION_ALIAS}")
    feature_names = mlflow.artifacts.load_dict(
        f"runs:/{version.run_id}/feature_names.json"
    )["feature_names"]

    _cached[source] = _LoadedFusionModel(
        pipeline=pipeline, feature_names=feature_names, version=str(version.version),
    )
    logger.info(
        "loaded decision_fusion champion v%s for source=%s in %dms",
        version.version, source, int((time.perf_counter() - load_start) * 1000),
    )
    return _cached[source]


KNOWN_SOURCES = ["SMPP", "SS7"]


def preload() -> None:
    """Warms _cached for every known source at process startup, same
    reasoning as serving.scoring.preload(). A source with no promoted
    fusion champion yet is expected (app.py falls back to
    rule_pattern_score alone) and must not block startup."""
    for source in KNOWN_SOURCES:
        try:
            _load_champion(source)
        except ChampionUnavailableError as e:
            logger.warning("preload: decision_fusion champion unavailable for source=%s: %s", source, e)


def reset_cache() -> None:
    """Test hook - forces the next score_fusion() call to reload from
    MLflow instead of reusing whatever this process already cached."""
    global _cached
    _cached = {}


def score_fusion(source: str, rule_pattern_score: float, anomaly_score: float) -> tuple[float, str]:
    """Returns (fusion_score, model_version) - same P(rule_flagged==True)
    convention as rule_pattern_score. Raises ChampionUnavailableError if no
    champion exists for `source` yet."""
    loaded = _load_champion(source)
    values = {"rule_pattern_score": rule_pattern_score, "anomaly_score": anomaly_score}
    X = np.array([[values[f] for f in loaded.feature_names]], dtype=np.float64)
    fusion_score = float(loaded.pipeline.predict_proba(X)[:, 1][0])
    return fusion_score, loaded.version
