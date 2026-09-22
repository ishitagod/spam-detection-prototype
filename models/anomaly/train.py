"""
Trains the unsupervised anomaly layer (`anomaly_score`) - Isolation
Forest over [MiniLM embedding + behavioral + FAISS near-dup features]
jointly, zero labels used for training (this layer exists to catch spam
the rule engine hasn't encoded).

CURRENT SCALE: full corpus per source (SMPP 5.5M / SS7 2.74M rows) - see
models/anomaly/data.py's ChunkedEmbeddingReducer for the memory-bounded
embedding preprocessing this needs at that scale.

OUTPUT SCORE: `-model.decision_function(X)`, negated so higher = more
anomalous (sklearn's decision_function is lower for anomalies).
`contamination` left at "auto" - it only governs `.predict()`'s binary
cutoff, which this project doesn't use (block/allow happens later, at
serving time).

EVALUATION: rule_evaluated/rule_flagged labels are used as a VALIDATION
set only, never fed into training. `evaluate_against_rule_labels()`
computes real PR-AUC/log loss (overall + per source) plus
precision@top-K% (0.1/0.5/1.0/5.0%, more representative of how this
score actually gets used - only the extreme top, see
cluster_discovery.py).

CAVEAT: this only measures agreement with patterns the rule engine
ALREADY knows - the opposite of this layer's real purpose (catching
spam rules can't see). Treat a good score as a floor-level sanity check,
not proof of novel-spam detection.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from config.settings import MLFLOW_TRACKING_URI
from models.anomaly.data import (
    build_combined_frame,
    build_feature_matrix,
    load_source_features,
)
from models.metrics import evaluate_overall_and_per_source, evaluate_precision_at_k

MLFLOW_EXPERIMENT_NAME = "isolation_forest"


def train_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 100,
    max_samples: str | int | float = "auto",
    contamination: str | float = "auto",
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_anomalies(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous - see module docstring for the sign flip."""
    return -model.decision_function(X)


def plausibility_check(df: pd.DataFrame, anomaly_score: np.ndarray) -> dict:
    """Label-free sanity check, not a training signal or formal metric -
    see module docstring. Returns a plain dict, loggable to MLflow."""
    # `== True` (not fillna+astype) - these cols are nullable-bool
    # (True/False/None); `== True` treats None as not-True directly and
    # avoids a pandas FutureWarning on object-dtype columns.
    evaluated = (df["rule_evaluated"] == True).to_numpy()  # noqa: E712
    flagged = (df["rule_flagged"] == True).to_numpy() & evaluated
    clean = evaluated & ~flagged

    result = {
        "n_rule_evaluated": int(evaluated.sum()),
        "n_rule_flagged": int(flagged.sum()),
        "mean_anomaly_score_overall": float(anomaly_score.mean()),
    }
    if flagged.any():
        result["mean_anomaly_score_flagged"] = float(anomaly_score[flagged].mean())
    if clean.any():
        result["mean_anomaly_score_rule_confirmed_clean"] = float(
            anomaly_score[clean].mean()
        )
    return result


def evaluate_against_rule_labels(df: pd.DataFrame, anomaly_score: np.ndarray) -> dict:
    """Real PR-AUC/log loss (models/metrics.py, shared with
    rule_pattern/train.py) using rule_evaluated==True rows as a
    VALIDATION set only - see module docstring for what this does/
    doesn't prove, and why SMPP-only PR-AUC is skipped.

    Evaluated overall + per source (an aggregate can hide one source
    performing badly). Also computes precision@top-K% on the same
    (y_true, score) pair - more honest than full PR-AUC for how this
    score is actually used (only the extreme top, via
    cluster_discovery.py's --anomaly_percentile); same known-pattern-only
    ceiling still applies.
    """
    evaluated = (df["rule_evaluated"] == True).to_numpy()  # noqa: E712
    flagged = (df["rule_flagged"] == True).to_numpy() & evaluated
    y_true = flagged[evaluated].astype(int)
    score = anomaly_score[evaluated]
    result = evaluate_overall_and_per_source(df[evaluated], "source", y_true, score)
    result.update(evaluate_precision_at_k(df[evaluated], "source", y_true, score))
    return result


def run(
    sources: list[str],
    data_dir: Path,
    n_estimators: int,
    max_samples: str,
    contamination: str,
    random_state: int,
) -> None:
    # Source-restricted run gets its own MLflow experiment, suffixed by
    # source - keeps compare_versions.py from comparing a challenger and
    # champion trained on different populations. Full default (both
    # sources) keeps the plain experiment name.
    experiment_name = MLFLOW_EXPERIMENT_NAME
    if sorted(sources) != sorted(["SMPP", "SS7"]):
        experiment_name += "_" + "_".join(sources)

    print(f"Loading + joining features for sources: {sources} ...")
    frames = []
    for source in sources:
        source_dir = data_dir / source
        messages_path = source_dir / "messages_with_behavioral.csv"
        df = load_source_features(source_dir, messages_path)
        print(
            f"  {source}: {len(df)} message(s) (sampled subset with embeddings+near-dup)"
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    print(f"Building feature matrix ({len(df)} rows) ...")
    X, feature_names, preprocessor = build_feature_matrix(df)
    print(f"  {X.shape[1]} feature(s): {len(feature_names)} columns")

    max_samples_arg = (
        max_samples
        if max_samples == "auto"
        else (int(max_samples) if float(max_samples) > 1 else float(max_samples))
    )
    contamination_arg = (
        contamination if contamination == "auto" else float(contamination)
    )

    print("Training Isolation Forest ...")
    model = train_isolation_forest(
        X,
        n_estimators=n_estimators,
        max_samples=max_samples_arg,
        contamination=contamination_arg,
        random_state=random_state,
    )
    anomaly_score = score_anomalies(model, X)

    checks = plausibility_check(df, anomaly_score)
    print("Plausibility check (weaker than PR-AUC below, just a quick eyeball):")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print(
        "Real evaluation - PR-AUC / log loss against rule_evaluated labels (validation only, not training):"
    )
    pr_auc_metrics = evaluate_against_rule_labels(df, anomaly_score)
    for k, v in pr_auc_metrics.items():
        print(f"  {k}: {v}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "sources": ",".join(sources),
                "n_rows": len(df),
                "n_features": X.shape[1],
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "contamination": contamination,
                "random_state": random_state,
            }
        )
        mlflow.log_metrics(checks)
        mlflow.log_metrics(pr_auc_metrics)
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        pipeline = Pipeline([("preprocessor", preprocessor), ("iforest", model)])
        # Signature/input_example describe the RAW pre-preprocessor frame,
        # since `pipeline` bundles the preprocessor itself. df.head(5)
        # only - rebuilding the combined frame for the full corpus again
        # here OOM'd on SS7's 2.74M rows; 5 rows is all infer_signature needs.
        input_example, _, _ = build_combined_frame(df.head(5))
        signature = infer_signature(input_example, pipeline.predict(input_example))
        # ChunkedEmbeddingReducer is our own class, not stdlib sklearn -
        # mlflow's skops serializer refuses unrecognized types by default.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["models.anomaly.data.ChunkedEmbeddingReducer"],
        )
        print(
            f"Logged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={experiment_name})"
        )

    # min-max normalized column (0-1, easy to eyeball) added alongside
    # the raw decision_function-derived anomaly_score.
    score_min, score_max = anomaly_score.min(), anomaly_score.max()
    normalized = (
        (anomaly_score - score_min) / (score_max - score_min)
        if score_max > score_min
        else np.zeros_like(anomaly_score)
    )
    df_out = pd.DataFrame(
        {
            "message_key": df["source"] + "|" + df["record_id"],
            "anomaly_score": anomaly_score,
            "anomaly_score_normalized": normalized,
        }
    )
    for source in sources:
        out_path = data_dir / source / "anomaly_scores.parquet"
        subset = df_out[df_out["message_key"].str.startswith(f"{source}|")]
        subset.to_parquet(out_path, index=False)
        print(f"Wrote {len(subset)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, nargs="+", default=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_samples", type=str, default="auto")
    parser.add_argument("--contamination", type=str, default="auto")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    run(
        args.sources,
        Path(args.data_dir),
        args.n_estimators,
        args.max_samples,
        args.contamination,
        args.random_state,
    )


if __name__ == "__main__":
    main()
