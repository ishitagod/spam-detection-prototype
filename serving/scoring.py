"""
Loads the rule_pattern_score champion and scores one canonical row.

SCOPE: rule_pattern_score only (see serving/schemas.py's module docstring
for why anomaly_score is a deliberate follow-up, not built here).

FEATURE PARITY WITH TRAINING: rule_pattern_score's champion is trained by
models/rule_pattern/train.py on the DEFAULT path (no --with_embeddings/
--with_tfidf - see that module's docstring: neither flag is the real
baseline yet), whose feature vector is exactly
models/rule_pattern/data.py's _base_feature_frame() - BEHAVIORAL_COLS,
`dcs`, `text_decode_failed`, `text_length`, one-hot `source`. This module
rebuilds that same 9-column frame for one live row rather than importing
_base_feature_frame() directly (that helper takes a whole DataFrame + is
private) - column NAMES and construction below are kept deliberately
identical to it so the two never drift apart silently.

The exact column ORDER actually used at training time is authoritative,
not assumed here: every training run logs `feature_names.json`
(models/rule_pattern/train.py) - this module fetches it from the
champion's own MLflow run and reindexes the built row to match, so a
column-order change in training can never silently produce a wrong score
here. If the champion was trained WITH --with_embeddings/--with_tfidf
(feature_names.json names columns this module can't build - emb_pca_*,
tfidf_*), scoring raises a clear error rather than silently misaligning
the feature vector - see ChampionUnsupportedError.
"""
from dataclasses import dataclass

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
import numpy as np

from models.registry import MLFLOW_TRACKING_URI
from serving.canonical import CanonicalRow

RULE_PATTERN_MODEL_NAME = "rule_pattern_score_model"
CHAMPION_ALIAS = "champion"

BEHAVIORAL_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr",
    "sender_unique_destinations_1hr", "sender_repeat_content_ratio_1hr",
]


class ChampionUnavailableError(RuntimeError):
    """No model currently holds CHAMPION_ALIAS for RULE_PATTERN_MODEL_NAME
    - a real, expected state before the first models/compare_versions.py
    promotion has ever run (see that module's get_champion_metric()
    docstring for the same "no champion yet" case on the training side)."""


class ChampionUnsupportedError(RuntimeError):
    """Champion's feature_names.json names a column this module can't
    build live (embeddings/TF-IDF - see module docstring)."""


@dataclass
class _LoadedRulePatternModel:
    model: object  # LGBMClassifier, loaded via mlflow.lightgbm.load_model
    feature_names: list[str]
    version: str


_cached: _LoadedRulePatternModel | None = None  # process-wide cache, same
# "load once, reuse everywhere" convention as
# features/text_embeddings.py's _model - loading a model off the MLflow
# registry costs real time, not worth repeating per request.


def _load_champion() -> _LoadedRulePatternModel:
    global _cached
    if _cached is not None:
        return _cached

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(RULE_PATTERN_MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException as e:
        raise ChampionUnavailableError(
            f"No {CHAMPION_ALIAS!r} version registered for {RULE_PATTERN_MODEL_NAME!r} - "
            "run models/rule_pattern/train.py then models/compare_versions.py to promote one."
        ) from e

    model = mlflow.lightgbm.load_model(f"models:/{RULE_PATTERN_MODEL_NAME}@{CHAMPION_ALIAS}")
    feature_names = mlflow.artifacts.load_dict(
        f"runs:/{version.run_id}/feature_names.json"
    )["feature_names"]

    unsupported = [f for f in feature_names if f.startswith("emb_pca_") or f.startswith("tfidf_")]
    if unsupported:
        raise ChampionUnsupportedError(
            f"Champion run {version.run_id} was trained with content features this "
            f"endpoint doesn't compute live yet ({unsupported[:3]}{'...' if len(unsupported) > 3 else ''}) "
            "- retrain via models/rule_pattern/train.py without --with_embeddings/--with_tfidf, "
            "or extend serving/scoring.py to build them."
        )

    _cached = _LoadedRulePatternModel(model=model, feature_names=feature_names, version=str(version.version))
    return _cached


def reset_cache() -> None:
    """Test hook - forces the next score_rule_pattern() call to reload
    from MLflow instead of reusing whatever this process already cached."""
    global _cached
    _cached = None


def build_rule_pattern_row(canonical: CanonicalRow, behavioral: dict) -> dict:
    """The exact 9 named columns models/rule_pattern/data.py's
    _base_feature_frame() builds for training, for one live row - see
    module docstring. `behavioral`: serving/feature_lookup.py's
    get_sender_features() output (None per-key on a cold-start sender,
    treated as 0 here - same convention models/anomaly/data.py's
    plausibility_check() and this endpoint's cold_start flag use)."""
    row = {col: (behavioral.get(col) or 0) for col in BEHAVIORAL_COLS}
    row["dcs"] = canonical.dcs if canonical.dcs is not None else np.nan  # LightGBM
    # has native missing-value handling - no imputation, same as training
    # (models/rule_pattern/data.py's _base_feature_frame() docstring).
    row["text_decode_failed"] = int(canonical.text_decode_failed)
    row["text_length"] = len(canonical.text or "")
    row["source_SMPP"] = int(canonical.source == "SMPP")
    row["source_SS7"] = int(canonical.source == "SS7")
    return row


def score_rule_pattern(canonical: CanonicalRow, behavioral: dict) -> tuple[float, str, dict]:
    """Returns (probability, model_version, features_used) - probability
    is model.predict_proba(...)[:, 1], the same column
    models/rule_pattern/train.py evaluates (P(rule_flagged==True))."""
    loaded = _load_champion()
    row = build_rule_pattern_row(canonical, behavioral)

    missing = [f for f in loaded.feature_names if f not in row]
    if missing:
        # Should be unreachable given the unsupported-feature check at
        # load time, but fail loudly rather than silently misaligning
        # columns if feature_names.json ever names something new.
        raise ChampionUnsupportedError(
            f"Champion expects feature(s) this endpoint doesn't compute: {missing}"
        )

    X = np.array([[row[f] for f in loaded.feature_names]], dtype=np.float64)
    probability = float(loaded.model.predict_proba(X)[:, 1][0])
    return probability, loaded.version, row
