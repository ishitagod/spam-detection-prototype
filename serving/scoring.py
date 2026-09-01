"""
Loads the rule_pattern_score champion and scores one canonical row.

SCOPE: rule_pattern_score only (see serving/schemas.py's module docstring
for why anomaly_score is a deliberate follow-up, not built here).

FEATURE PARITY WITH TRAINING: whatever models/rule_pattern/train.py
actually trained the champion with - the base 9-column frame
(models/rule_pattern/data.py's _base_feature_frame(): BEHAVIORAL_COLS,
`dcs`, `text_decode_failed`, `text_length`, one-hot `source`), and
OPTIONALLY TF-IDF (`tfidf_*`) and/or PCA-reduced MiniLM embeddings
(`emb_pca_*`) if the champion was trained with --with_tfidf/
--with_embeddings. This module rebuilds the base frame rather than
importing _base_feature_frame() directly (that helper takes a whole
DataFrame + is private) - column NAMES and construction below are kept
deliberately identical to it so the two never drift apart silently. The
TF-IDF/embedding paths reuse the champion's OWN fitted
tfidf_vectorizer/embedding_pca_pipeline artifacts (logged alongside
`model` in the same MLflow run) and features/text_embeddings.py's
embed_texts() for the raw MiniLM encode - same transformers training
fit, never refit here, same encoder training used, so live scoring can
never silently drift from what the champion actually learned.

The exact column ORDER actually used at training time is authoritative,
not assumed here: every training run logs `feature_names.json`
(models/rule_pattern/train.py) - this module fetches it from the
champion's own MLflow run and reindexes the built row to match, so a
column-order change in training can never silently produce a wrong score
here.
"""
from dataclasses import dataclass

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
import mlflow.sklearn
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
    """Champion's feature_names.json names a column this module still
    can't build live even after accounting for tfidf_*/emb_pca_* -
    should be unreachable in practice (see score_rule_pattern()'s
    belt-and-suspenders check), a real bug if ever raised, not an
    expected "unsupported flag combo" state anymore."""


@dataclass
class _LoadedRulePatternModel:
    model: object  # LGBMClassifier, loaded via mlflow.lightgbm.load_model
    feature_names: list[str]
    version: str
    tfidf_vectorizer: object | None = None  # fitted TfidfVectorizer, only
    # present if the champion was trained --with_tfidf
    embedding_pca_pipeline: object | None = None  # fitted sklearn Pipeline
    # (StandardScaler -> PCA), only present if trained --with_embeddings


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

    # Load the SAME fitted transformers training used, from the champion's
    # own run - never refit here (see module docstring). Only load what
    # feature_names.json actually names, so a plain (no-flags) champion
    # doesn't pay for artifacts that were never logged for it.
    needs_tfidf = any(f.startswith("tfidf_") for f in feature_names)
    needs_embeddings = any(f.startswith("emb_pca_") for f in feature_names)
    tfidf_vectorizer = (
        mlflow.sklearn.load_model(f"runs:/{version.run_id}/tfidf_vectorizer") if needs_tfidf else None
    )
    embedding_pca_pipeline = (
        mlflow.sklearn.load_model(f"runs:/{version.run_id}/embedding_pca_pipeline")
        if needs_embeddings else None
    )

    _cached = _LoadedRulePatternModel(
        model=model, feature_names=feature_names, version=str(version.version),
        tfidf_vectorizer=tfidf_vectorizer, embedding_pca_pipeline=embedding_pca_pipeline,
    )
    return _cached


def reset_cache() -> None:
    """Test hook - forces the next score_rule_pattern() call to reload
    from MLflow instead of reusing whatever this process already cached."""
    global _cached
    _cached = None


def build_rule_pattern_row(
    canonical: CanonicalRow, behavioral: dict,
    tfidf_vectorizer=None, embedding_pca_pipeline=None,
) -> dict:
    """The base 9 named columns models/rule_pattern/data.py's
    _base_feature_frame() builds for training, for one live row - see
    module docstring. `behavioral`: serving/feature_lookup.py's
    get_sender_features() output (None per-key on a cold-start sender,
    treated as 0 here - same convention models/anomaly/data.py's
    plausibility_check() and this endpoint's cold_start flag use).

    tfidf_vectorizer/embedding_pca_pipeline: the champion's OWN fitted
    transformers (see _load_champion()) - when given, adds tfidf_*/
    emb_pca_* columns on top of the base 9, built from `canonical.text`
    exactly the way models/rule_pattern/data.py's build_feature_matrix()
    built them at training time (same vectorizer.transform(), same
    embed-then-PCA order). None (the default) skips that column group
    entirely - correct for a champion trained without that flag, where
    feature_names.json won't ask for those columns anyway."""
    row = {col: (behavioral.get(col) or 0) for col in BEHAVIORAL_COLS}
    row["dcs"] = canonical.dcs if canonical.dcs is not None else np.nan  # LightGBM
    # has native missing-value handling - no imputation, same as training
    # (models/rule_pattern/data.py's _base_feature_frame() docstring).
    row["text_decode_failed"] = int(canonical.text_decode_failed)
    row["text_length"] = len(canonical.text or "")
    row["source_SMPP"] = int(canonical.source == "SMPP")
    row["source_SS7"] = int(canonical.source == "SS7")

    text = canonical.text or ""
    if tfidf_vectorizer is not None:
        tfidf_vector = tfidf_vectorizer.transform([text]).toarray()[0]
        for name, value in zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vector):
            row[f"tfidf_{name}"] = float(value)
    if embedding_pca_pipeline is not None:
        # Lazy import: features/text_embeddings.py pulls in
        # sentence-transformers, real weight-loading cost that a
        # plain/TF-IDF-only champion should never pay for.
        from features.text_embeddings import embed_texts

        raw_embedding = embed_texts([text])  # (1, 384), same MiniLM model/
        # normalization training used (features/text_embeddings.py)
        reduced = embedding_pca_pipeline.transform(raw_embedding)[0]
        for i, value in enumerate(reduced):
            row[f"emb_pca_{i}"] = float(value)
    return row


def score_rule_pattern(canonical: CanonicalRow, behavioral: dict) -> tuple[float, str, dict]:
    """Returns (probability, model_version, features_used) - probability
    is model.predict_proba(...)[:, 1], the same column
    models/rule_pattern/train.py evaluates (P(rule_flagged==True))."""
    loaded = _load_champion()
    row = build_rule_pattern_row(
        canonical, behavioral,
        tfidf_vectorizer=loaded.tfidf_vectorizer,
        embedding_pca_pipeline=loaded.embedding_pca_pipeline,
    )

    missing = [f for f in loaded.feature_names if f not in row]
    if missing:
        # Should be unreachable now that _load_champion() loads whatever
        # transformers feature_names.json actually calls for - fail
        # loudly rather than silently misaligning columns if
        # feature_names.json ever names something genuinely new.
        raise ChampionUnsupportedError(
            f"Champion expects feature(s) this endpoint doesn't compute: {missing}"
        )

    X = np.array([[row[f] for f in loaded.feature_names]], dtype=np.float64)
    probability = float(loaded.model.predict_proba(X)[:, 1][0])
    return probability, loaded.version, row
