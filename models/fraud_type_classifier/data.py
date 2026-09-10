"""
Loads + prepares labelled data for the EXPERIMENTAL multiclass fraud-type
classifier - the production plan's Stage C bootstrap
(docs/sms_spam_technical_architecture_plan.md), seeded from
docs/experiments/anomaly_clustering.md's clustering workflow.

LABEL SOURCE - read this before trusting anything trained from this
module. Two sources, same downstream shape:
  "suggested" (default) - models/anomaly/suggest_cluster_labels.py's
    heuristic, UNCONFIRMED guesses (cluster_labels_suggested.csv). As of
    this module's introduction, NO human has hand-confirmed any cluster
    yet (docs/experiments/anomaly_clustering.md's step 4 is still
    outstanding) - "suggested" is the only label source that actually
    exists right now. Training on it answers "does this feature set carry
    ANY signal for fraud-type classification", NOT "is this a real
    model" - see models/fraud_type_classifier/train.py's module docstring
    for the MLflow-naming discipline that keeps this from ever being
    mistaken for a real candidate.
  "confirmed" - labels/cluster_labels.py's real hand-confirmed labels
    (cluster_labels.parquet, written by
    models/anomaly/ingest_cluster_labels.py after a human fills in
    inspect_clusters.py's fraud_type_label column). Use this once real
    labeling work exists - same code path, real data.

FEATURES: canonical + behavioral + source, deliberately mirroring
models/rule_pattern/data.py's default (no-flags) scope rather than
models/anomaly/data.py's (no embeddings, no age-bucketing/diversity-
gating) - this is testing whether CHEAP, already-computed features carry
fraud-type signal before reaching for anything heavier. Behavioral
columns imported from models.anomaly.data (BEHAVIORAL_COLS etc.) rather
than duplicated, same "single source of truth" convention used
throughout this codebase - and passed through RAW/unbucketed, since the
narrow-range/small-sample failure modes that motivated bucketing/gating
for models/anomaly/data.py were established specifically for Isolation
Forest's isolation-path mechanism, not for LightGBM's scale-invariant
tree splits (models/rule_pattern/data.py makes the same choice for the
same reason).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from models.anomaly.data import BEHAVIORAL_COLS, IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL

CANONICAL_COLS = ["dcs", "text_decode_failed"]
LABEL_COL = "fraud_type_label"


def load_cluster_labeled_messages(
    source: str, data_dir: Path, label_source: str = "suggested",
) -> pd.DataFrame:
    """
    One row per labelled message: `LABEL_COL` (the cluster-derived
    fraud-type name, whichever `label_source` it came from) plus
    canonical+behavioral feature columns, ready for build_feature_matrix().
    Raises a clear, specific FileNotFoundError (not a silent empty frame)
    naming the exact upstream command to run if the requested label
    source doesn't exist yet - same convention as
    models/anomaly/cluster_discovery.py's load_features_and_scores() etc.
    """
    source_dir = Path(data_dir) / source

    if label_source == "suggested":
        template_path = source_dir / "cluster_labels_suggested.csv"
        clusters_path = source_dir / "fraud_type_clusters.parquet"
        if not template_path.exists():
            raise FileNotFoundError(
                f"{template_path} not found - run "
                f"`python -m models.anomaly.suggest_cluster_labels --source {source}` first."
            )
        if not clusters_path.exists():
            raise FileNotFoundError(
                f"{clusters_path} not found - run "
                f"`python -m models.anomaly.cluster_discovery --sources {source}` first."
            )
        clusters = pd.read_parquet(clusters_path)[["message_key", "cluster_label"]]
        template = pd.read_csv(template_path)[["cluster_label", "suggested_fraud_type_label"]]
        labels = clusters.merge(template, on="cluster_label", how="inner")
        labels = labels.rename(columns={"suggested_fraud_type_label": LABEL_COL})
    elif label_source == "confirmed":
        labels_path = source_dir / "cluster_labels.parquet"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"{labels_path} not found - no confirmed cluster labels yet. Hand-fill "
                f"data/processed/{source}/cluster_labels_template.csv's fraud_type_label "
                "column (docs/experiments/anomaly_clustering.md's step 4), then run "
                f"`python -m models.anomaly.ingest_cluster_labels --source {source}`."
            )
        labels = pd.read_parquet(labels_path).rename(columns={"cluster_fraud_type_label": LABEL_COL})
    else:
        raise ValueError(f"label_source must be 'suggested' or 'confirmed', got {label_source!r}")

    labels["record_id"] = labels["message_key"].str.split("|", n=1).str[1]

    messages_path = source_dir / "messages_with_behavioral.csv"
    wanted_cols = (
        ["record_id", "source", "text"] + CANONICAL_COLS + BEHAVIORAL_COLS
        + [IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL]
    )
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=lambda c: c in set(wanted_cols),
        dtype={"record_id": str},
    )
    if IMSI_DISTINCT_ORIG_COL not in messages.columns:
        messages[IMSI_DISTINCT_ORIG_COL] = np.nan

    df = labels.merge(messages, on="record_id", how="inner", validate="one_to_one")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y, feature_names) - canonical + behavioral + source
    (one-hot, only added when this pool spans more than one source - same
    "no constant/information-free column" rule build_feature_matrix()
    already follows elsewhere in this codebase), y = the raw string
    fraud_type_label (LGBMClassifier's sklearn API handles string labels
    natively for multiclass, no manual encoding needed here). NOT scaled -
    tree splits are scale-invariant.
    """
    text_length = df["text"].fillna("").str.len().rename("text_length")
    text_decode_failed = df["text_decode_failed"].astype(int).rename("text_decode_failed")
    pieces = [df[BEHAVIORAL_COLS], df[[IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL]],
              df[["dcs"]], text_decode_failed, text_length]
    if df["source"].nunique() > 1:
        pieces.append(pd.get_dummies(df["source"], prefix="source"))
    matrix = pd.concat(pieces, axis=1)
    y = df[LABEL_COL].to_numpy()
    return matrix.to_numpy(dtype=np.float64), y, matrix.columns.tolist()
