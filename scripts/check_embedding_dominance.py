"""
One-off diagnostic, NOT permanent pipeline code: does the 384-dim MiniLM
embedding block dominate the 396-feature joint Isolation Forest, drowning
out the 12 hand-built behavioral+near-dup features? A measurement to
decide whether PCA/reweighting (see the conversation this was built from)
is even worth building - not something to run as part of normal
training.

Method: train three Isolation Forests, same hyperparameters, on
  (a) everything (the real 396-feature matrix train.py actually uses)
  (b) behavioral + near-dup + source only (12ish features, no embeddings)
  (c) embeddings only (384 features, nothing else)
Then compare how closely (a)'s scores correlate with (b) vs (c). If (a)
tracks (c) much more closely than (b), that's real evidence the
embeddings are dominating the joint model. If both correlations are
similar, both feature groups are meaningfully contributing.

Logs the three correlation numbers (and row/feature counts) to MLflow as
metrics/params only - NOT mlflow.sklearn.log_model() for any of the three
throwaway models, since none of them are meant to be reused or promoted,
unlike models/anomaly/train.py's real run. Logged to its own experiment
(MLFLOW_EXPERIMENT_NAME below), kept separate from the real "anomaly_score"
experiment so one-off diagnostic checks never get mistaken for candidate
models in the MLflow UI.

Usage:
    python scripts/check_embedding_dominance.py
"""
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import MLFLOW_TRACKING_URI
from models.anomaly.data import BEHAVIORAL_COLS, COUNT_COLS, NEAR_DUP_COLS, build_feature_matrix, load_source_features

DATA_DIR = REPO_ROOT / "data" / "processed"
SOURCES = ["SMPP", "SS7"]
MLFLOW_EXPERIMENT_NAME = "anomaly_score_diagnostics"


def fit_score(X: np.ndarray, name: str) -> np.ndarray:
    print(f"Training on {name} ({X.shape[1]} features) ...")
    model = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X)
    return -model.decision_function(X)  # higher = more anomalous, matches train.py's convention


def main():
    print("Loading + joining ...")
    frames = []
    for source in SOURCES:
        source_dir = DATA_DIR / source
        frames.append(load_source_features(source_dir, source_dir / "messages_with_behavioral.csv"))
    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df)} rows")

    # (a) everything - exactly what models/anomaly/train.py actually uses
    X_full, _, _ = build_feature_matrix(df)

    # (b) behavioral + near-dup + source only
    transformed = df.copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])
    source_dummies = pd.get_dummies(transformed["source"], prefix="source")
    behavioral_matrix = pd.concat([transformed[BEHAVIORAL_COLS + NEAR_DUP_COLS], source_dummies], axis=1)
    X_behavioral = StandardScaler().fit_transform(behavioral_matrix.to_numpy(dtype=np.float64))

    # (c) embeddings only
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    X_embeddings = StandardScaler().fit_transform(df[embedding_cols].to_numpy(dtype=np.float64))

    print(
        f"  full: {X_full.shape[1]} features | "
        f"behavioral+near_dup: {X_behavioral.shape[1]} | embeddings: {X_embeddings.shape[1]}"
    )

    score_full = fit_score(X_full, "everything")
    score_behavioral = fit_score(X_behavioral, "behavioral+near_dup only")
    score_embeddings = fit_score(X_embeddings, "embeddings only")

    # pandas' own Spearman implementation - no scipy dependency needed
    scores = pd.DataFrame({"full": score_full, "behavioral": score_behavioral, "embeddings": score_embeddings})
    corr = scores.corr(method="spearman")
    corr_full_behavioral = float(corr.loc["full", "behavioral"])
    corr_full_embeddings = float(corr.loc["full", "embeddings"])
    corr_behavioral_embeddings = float(corr.loc["behavioral", "embeddings"])

    print()
    print("Spearman rank correlation between each pair of models' scores:")
    print(f"  full <-> behavioral+near_dup only: {corr_full_behavioral:.3f}")
    print(f"  full <-> embeddings only:          {corr_full_embeddings:.3f}")
    print(f"  behavioral+near_dup <-> embeddings (for reference): {corr_behavioral_embeddings:.3f}")
    print()
    print(
        "If 'full <-> embeddings' is much higher than 'full <-> behavioral', "
        "the joint model is dominated by embeddings. If they're close, both "
        "feature groups are meaningfully contributing."
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({
            "sources": ",".join(SOURCES),
            "n_rows": len(df),
            "n_features_full": X_full.shape[1],
            "n_features_behavioral": X_behavioral.shape[1],
            "n_features_embeddings": X_embeddings.shape[1],
        })
        mlflow.log_metrics({
            "corr_full_behavioral": corr_full_behavioral,
            "corr_full_embeddings": corr_full_embeddings,
            "corr_behavioral_embeddings": corr_behavioral_embeddings,
        })
    print(f"Logged to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={MLFLOW_EXPERIMENT_NAME})")


if __name__ == "__main__":
    main()
