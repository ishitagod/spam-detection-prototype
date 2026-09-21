"""
Trains the decision-fusion meta-model. Does not violate CLAUDE.md's "keep
rule_pattern_score and anomaly_score separate, don't average them" - it
learns a small, fully-inspectable combination of them (2 coefficients + an
intercept) that serving/fusion_scoring.py exposes as its own additive
`fusion_score`, alongside the two raw scores which stay unchanged in the
response (see serving/app.py's module docstring).

MODEL: StandardScaler -> LogisticRegression on exactly 2 features
(rule_pattern_score, anomaly_score). Deliberately the smallest model that
could plausibly work: the inputs are already two models' opinions, so
fusion's only job is learning how much to trust each - 2 coefficients are
enough, and unlike a boosted-tree meta-model they're directly readable off
the fitted model (model.coef_) without SHAP.

TRAINING POOL IS SMALL AND SOURCE-SKEWED RIGHT NOW: see
models/decision_fusion/data.py's module docstring (SS7's
anomaly_scores.parquet is still a 20k-row sample) - printed at run time,
not smoothed over. Re-run once SS7's Isolation Forest is retrained on its
full corpus.

NOT wired into pipeline.py, same reasoning as the other train.py scripts.
Run by hand, per source:

    python -m models.decision_fusion.train --source SS7
    python -m models.decision_fusion.train --source SMPP

EVALUATION: same shared PR-AUC/log-loss/precision@K helpers
(models/metrics.py) as the two base models.
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import MLFLOW_TRACKING_URI
from models.decision_fusion.data import build_feature_matrix, build_fusion_training_data
from models.metrics import evaluate_overall_and_per_source, evaluate_precision_at_k

MLFLOW_EXPERIMENT_NAME = "decision_fusion"


def train_fusion_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def run(
    source: str, data_dir: Path, test_size: float, random_state: int,
) -> None:
    experiment_name = f"{MLFLOW_EXPERIMENT_NAME}_{source}"

    print(f"Building fusion training data for {source} ...")
    df = build_fusion_training_data(source, data_dir, random_state=random_state)
    if len(df) == 0:
        raise ValueError(
            f"{source}: 0 rows available for fusion training (single-class rule_flagged "
            "pool, or zero anomaly_score join coverage) - see "
            "models/decision_fusion/data.py's module docstring."
        )
    print(f"  {len(df)} row(s) with both scores + a real label")

    X, y, feature_names = build_feature_matrix(df)
    if len(set(y.tolist())) < 2:
        raise ValueError(
            f"{source}: fusion training pool collapsed to a single class after the "
            "anomaly_score join - cannot train/evaluate a classifier on it."
        )

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state,
    )
    X_train, y_train, df_train = X[idx_train], y[idx_train], df.iloc[idx_train]
    X_test, y_test, df_test = X[idx_test], y[idx_test], df.iloc[idx_test]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Training fusion model (StandardScaler -> LogisticRegression) ...")
    model = train_fusion_model(X_train, y_train, random_state=random_state)

    train_score = model.predict_proba(X_train)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    print("Train set evaluation:")
    train_metrics = evaluate_overall_and_per_source(df_train, "source", y_train, train_score, prefix="train_")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    print("Test set evaluation (the real number):")
    test_metrics = evaluate_overall_and_per_source(df_test, "source", y_test, test_score, prefix="test_")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    test_metrics.update(evaluate_precision_at_k(df_test, "source", y_test, test_score, prefix="test_"))

    logreg = model.named_steps["logreg"]
    coefficients = dict(zip(feature_names, logreg.coef_[0].tolist()))
    print(f"Fitted coefficients (post-StandardScaler, so directly comparable): {coefficients}")
    print(f"Intercept: {logreg.intercept_[0]:.4f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "source": source,
            "n_rows": len(df),
            "n_positive": int(y.sum()),
            "test_size": test_size,
            "random_state": random_state,
        })
        mlflow.log_metrics({**train_metrics, **test_metrics})
        mlflow.log_dict(
            {"feature_names": feature_names, "coefficients": coefficients,
             "intercept": logreg.intercept_[0]},
            "feature_names.json",
        )

        input_example = X_train[:5]
        signature = infer_signature(input_example, model.predict_proba(input_example))
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=input_example)
        print(f"Logged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={experiment_name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    run(args.source, Path(args.data_dir), args.test_size, args.random_state)


if __name__ == "__main__":
    main()
