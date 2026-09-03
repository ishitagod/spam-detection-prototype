"""
Loads the anomaly_score champion (Isolation Forest + its full
preprocessor, logged as one sklearn Pipeline by models/anomaly/train.py)
and scores one canonical row - the anomaly_score counterpart to
serving/scoring.py's score_rule_pattern(), following serving/schemas.py's
"deliberate follow-up" note now that this design is settled.

MODEL ROUTING: split by source, same as rule_pattern_score
(serving/scoring.py) - scripts/check_source_split_justified.py justified
it, so this loads a per-source champion registered as
f"{ANOMALY_MODEL_NAME}_{source}" (anomaly_SMPP / anomaly_SS7), cached in
a dict keyed by source, same pattern serving/scoring.py already uses -
SMPP and SS7 never share a champion.

THE HARD PART - NEAR-DUP FEATURES AT SERVING TIME: Isolation Forest needs
near_dup_match_count/max_similarity/distinct_senders for both windows
(models/anomaly/data.py's NEAR_DUP_COLS), which the batch pipeline
computes by comparing every message against a corpus of OTHER messages
(features/faiss_index.py). A live request has no such corpus of its own -
this prototype has no streaming/live-appended traffic (CLAUDE.md:
"Batch/synchronous feature computation, not streaming, at this stage").

DELIBERATE DESIGN: reuse each source's EXISTING historical corpus
(data/processed/<source>/embeddings.npy + embeddings_id_map.parquet, the
same files features/faiss_index.py already builds an index from) as the
"recent traffic" a live request is compared against, loaded once into an
in-process FAISS index per source (same load-once-cache convention as
serving/scoring.py's champion cache). Windowing is computed relative to
the REQUEST'S OWN declared canonical.timestamp (not wall-clock receipt
time) - the same point-in-time convention training uses, and the only
sensible reference point given the historical corpus's own timestamps
don't track wall-clock "now" in a demo/prototype setting. STALENESS
CAVEAT: this corpus is exactly as fresh as the last
features/text_embeddings.py + features/faiss_index.py run - a real
production deployment would need a live-appended or periodically
refreshed corpus (see scripts/refresh_feast.py for the equivalent
staleness caveat already accepted for behavioral features); revisit the
refresh cadence, not this module, if that becomes the bottleneck.

FEATURE PARITY WITH TRAINING: models/anomaly/data.py's build_combined_frame()
is reused directly (unlike serving/scoring.py's rule-pattern path, which
reimplements its own base-frame construction because that helper is
private) - build_combined_frame() is a public, pure function, so reusing
it here means the log1p/IMSI-known transforms can never drift from
training. Its `known_sources` parameter exists specifically for this
module: a single live row always has nunique()==1 on `source`, which
would otherwise silently drop whichever source_* dummy column the
already-fitted preprocessor still expects (see that parameter's
docstring).
"""
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from features.faiss_index import build_index, compute_near_dup_features_for_live_query
from models.anomaly.data import (
    BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL, build_combined_frame,
)
from models.registry import MLFLOW_TRACKING_URI
from serving.canonical import CanonicalRow

ANOMALY_MODEL_NAME = "anomaly"  # base name - actual registered model is
# source-suffixed (anomaly_SMPP / anomaly_SS7), see module docstring
CHAMPION_ALIAS = "champion"
KNOWN_SOURCES = ["SMPP", "SS7"]


DEFAULT_DATA_DIR = Path("data/processed")


class ChampionUnavailableError(RuntimeError):
    """No model currently holds CHAMPION_ALIAS for this source's
    registered name (ANOMALY_MODEL_NAME + "_" + source) - a real, expected
    state before models/compare_versions.py has ever promoted a champion
    for THAT source (see serving/scoring.py's identical error for
    rule_pattern_score). Each source is independent: SMPP having a
    champion doesn't mean SS7 does, or vice versa."""


class CorpusUnavailableError(RuntimeError):
    """This source has no embeddings.npy/embeddings_id_map.parquet yet -
    run features/text_embeddings.py then features/faiss_index.py for it
    first (see pipeline steps 1-3 in scripts/run_full_pipeline.ps1)."""


@dataclass
class _LoadedAnomalyModel:
    pipeline: object  # sklearn Pipeline(preprocessor, iforest), loaded via mlflow.sklearn.load_model
    version: str


@dataclass
class _LoadedCorpus:
    index: "object"  # faiss.Index, built once over this source's embeddings
    timestamps: np.ndarray
    originators: np.ndarray


_cached_model: dict[str, _LoadedAnomalyModel] = {}  # keyed by source - one
# process-wide cache entry per source-specific champion, same reasoning as
# serving/scoring.py's per-source champion cache (real cost, load once).
_cached_corpus: dict[str, _LoadedCorpus] = {}  # keyed by source - each
# source's corpus is loaded/indexed independently, same reasoning as
# serving/scoring.py's per-source champion cache (real cost, load once).


def _load_champion(source: str) -> _LoadedAnomalyModel:
    if source in _cached_model:
        return _cached_model[source]

    registered_name = f"{ANOMALY_MODEL_NAME}_{source}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException as e:
        raise ChampionUnavailableError(
            f"No {CHAMPION_ALIAS!r} version registered for {registered_name!r} - "
            f"run `python -m models.anomaly.train --sources {source}` then "
            f"`python -m models.compare_versions --experiment_name isolation_forest_{source} "
            f"--registered_name {registered_name} --metric_key overall_pr_auc` to promote one."
        ) from e

    pipeline = mlflow.sklearn.load_model(f"models:/{registered_name}@{CHAMPION_ALIAS}")
    _cached_model[source] = _LoadedAnomalyModel(pipeline=pipeline, version=str(version.version))
    return _cached_model[source]


def _load_corpus(source: str, data_dir: Path) -> _LoadedCorpus:
    if source in _cached_corpus:
        return _cached_corpus[source]

    source_dir = data_dir / source
    emb_path = source_dir / "embeddings.npy"
    id_map_path = source_dir / "embeddings_id_map.parquet"
    messages_path = source_dir / "messages_with_behavioral.csv"
    if not emb_path.exists() or not id_map_path.exists():
        raise CorpusUnavailableError(
            f"No embeddings found for source {source!r} in {source_dir} - "
            "run features/text_embeddings.py for it first."
        )

    embeddings = np.load(emb_path)
    id_map = pd.read_parquet(id_map_path)

    # Same originator-merge features/faiss_index.py::run_faiss_near_dup()
    # does - text_embeddings.py's id_map has message_key+timestamp but no
    # notion of sender.
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=["source", "record_id", "originator"]
    )
    messages["source"] = messages["source"].astype(str)
    messages["record_id"] = messages["record_id"].astype(str)
    messages["message_key"] = messages["source"] + "|" + messages["record_id"]
    id_map = id_map.merge(messages[["message_key", "originator"]], on="message_key", how="left")

    index = build_index(embeddings)
    timestamps = pd.to_datetime(id_map["timestamp"]).to_numpy()
    originators = id_map["originator"].astype(str).to_numpy()

    loaded = _LoadedCorpus(index=index, timestamps=timestamps, originators=originators)
    _cached_corpus[source] = loaded
    return loaded


def reset_cache() -> None:
    """Test hook - forces the next score_anomaly() call to reload the
    champion and rebuild every source's corpus index instead of reusing
    whatever this process already cached."""
    global _cached_model, _cached_corpus
    _cached_model = {}
    _cached_corpus = {}


def build_anomaly_row(
    canonical: CanonicalRow, behavioral: dict, near_dup_features: dict, embedding: np.ndarray,
) -> dict:
    """
    One raw (pre-preprocessor) row, in the shape build_combined_frame()
    expects as input - RAW counts (not log1p'd, that happens inside
    build_combined_frame()), raw 384-dim MiniLM embedding (PCA happens
    inside the loaded pipeline's preprocessor step, not here).

    `behavioral`: serving/feature_lookup.py's get_sender_features() output
    merged with get_imsi_features()'s - None per-key on a cold-start
    sender, treated as 0 for BEHAVIORAL_COLS (same convention
    serving/scoring.py's build_rule_pattern_row() uses) but left as NaN
    for imsi_distinct_originators_1hr (build_combined_frame() already
    knows how to turn that into a _known indicator + 0, same as training -
    fabricating 0 here directly would lose that distinction).
    `near_dup_features`: compute_near_dup_features_for_live_query()'s
    output - already the right near_dup_* column names.
    `embedding`: raw (1, 384) MiniLM output for canonical.text.
    """
    row = {col: (behavioral.get(col) or 0) for col in BEHAVIORAL_COLS}
    imsi_value = behavioral.get(IMSI_DISTINCT_ORIG_COL)
    row[IMSI_DISTINCT_ORIG_COL] = imsi_value if imsi_value is not None else np.nan
    # sender_velocity_zscore_5min: same RAW-value-or-NaN treatment as IMSI
    # above - kept OUT of BEHAVIORAL_COLS deliberately (models/anomaly/
    # data.py's comment) since build_combined_frame() below needs the real
    # None-vs-real distinction to build this column's _known indicator
    # correctly; a 0-fill here would fabricate "known, exactly average"
    # for a cold-start sender with no real baseline yet.
    velocity_value = behavioral.get(SENDER_VELOCITY_ZSCORE_COL)
    row[SENDER_VELOCITY_ZSCORE_COL] = velocity_value if velocity_value is not None else np.nan
    row.update(near_dup_features)
    row["source"] = canonical.source
    for i, value in enumerate(embedding.reshape(-1)):
        row[f"emb_{i}"] = float(value)
    return row


def score_anomaly(
    canonical: CanonicalRow, behavioral: dict, data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[float, str, dict]:
    """
    Returns (anomaly_score, model_version, features_used) - anomaly_score
    is HIGHER = more anomalous, same sign convention as
    models/anomaly/train.py::score_anomalies() (the loaded pipeline's
    final step is the raw IsolationForest, so its .decision_function()
    needs the same negation applied here that training applies).

    NEVER averaged with rule_pattern_score, and never gates
    recommended_action on its own yet (see serving/app.py) - CLAUDE.md:
    "Keep rule_pattern_score and anomaly_score separate. Do not average
    them. Disagreements between the two scores are valuable and should
    remain visible."
    """
    loaded = _load_champion(canonical.source)
    corpus = _load_corpus(canonical.source, data_dir)

    # Lazy import: features/text_embeddings.py pulls in
    # sentence-transformers - real weight-loading cost, same reasoning as
    # serving/scoring.py's embedding-path import.
    from features.text_embeddings import embed_texts

    embedding = embed_texts([canonical.text or ""])  # (1, 384)
    near_dup_features = compute_near_dup_features_for_live_query(
        corpus.index, embedding[0], np.datetime64(canonical.timestamp),
        corpus.timestamps, corpus.originators,
    )

    row = build_anomaly_row(canonical, behavioral, near_dup_features, embedding)
    combined, _, _ = build_combined_frame(pd.DataFrame([row]), known_sources=KNOWN_SOURCES)

    raw_decision = loaded.pipeline.decision_function(combined)[0]
    anomaly_score = float(-raw_decision)  # higher = more anomalous, see docstring
    return anomaly_score, loaded.version, row
