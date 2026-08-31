"""
Trains the unsupervised anomaly layer (`anomaly_score`) - Isolation
Forest over [MiniLM embedding + behavioral features + FAISS near-dup
features] jointly, per README.md's modeling plan. Zero labels used for
training, on purpose - see that section for why this is the layer meant
to catch spam the rule engine has never encoded.

NOT wired into pipeline.py: training is a deliberate, versioned action,
not a deterministic feature-computation step - see pipeline.py's own
docstring and the discussion this was built from. Run this by hand:

    python -m models.anomaly.train
    python -m models.anomaly.train --n_estimators 200 --contamination 0.02

CURRENT SCALE: trains on whatever features/text_embeddings.py's
--sample_n sample covers (40k SMPP / 20k SS7 as of writing) - see
models/anomaly/data.py's module docstring for why the join is
necessarily restricted to that sample right now.

OUTPUT SCORE: `-model.decision_function(X)`, not `.predict()`'s binary
label. sklearn's decision_function is HIGHER for normal points, LOWER
(more negative) for anomalies - negated here so higher = more anomalous,
matching this project's `anomaly_score` convention (see
docs/prototype_plan.md's response contract). `contamination` is left at
scikit-learn's default ("auto") rather than tuned, since it only governs
`.predict()`'s binary cutoff, which this project's architecture doesn't
use - the confidence-gated block/allow decision happens later, at
serving time, not baked into training (see README.md's "problem,
precisely" section).

EVALUATION: trained with zero labels, but NOT evaluated with zero
labels - the real rule_evaluated/rule_flagged labels (the same ones
LightGBM will train on later) are used here purely as a validation set,
never fed into training. `evaluate_against_rule_labels()` computes real
PR-AUC and log loss (per README.md's stated evaluation convention:
PR-AUC/log loss primary, evaluated overall + per source), checking
whether anomaly_score actually ranks real rule-confirmed spam above
rule-confirmed clean.

BE HONEST ABOUT WHAT THIS METRIC DOES AND DOESN'T PROVE: it measures
agreement with patterns the RULE ENGINE ALREADY KNOWS - the exact
opposite of this layer's real purpose (catching spam the rules can't
see, on the unlabelled majority). There is no way to formally evaluate
THAT without labels, which is precisely why this layer exists in the
first place. Treat a good score here as a floor-level sanity check, not
proof of novel-spam detection.

SMPP HAS NO CONFIRMED-CLEAN LABELS (verified: 2,693/2,693 rule-evaluated
SMPP rows are flagged, zero are confirmed-clean - see README.md's data
reality check) - PR-AUC is mathematically undefined with only one class
present, so SMPP-only PR-AUC is skipped, not silently computed wrong.

Also kept: a simpler plausibility_check() (mean anomaly_score by group)
- weaker than PR-AUC (doesn't account for the full score distribution),
but cheap and easy to sanity-eyeball alongside the real metric.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from models.anomaly.data import build_feature_matrix, load_source_features
from models.metrics import evaluate_overall_and_per_source

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
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
    """
    Label-free sanity check, NOT a training signal or a formal metric -
    see module docstring. Returns a plain dict, loggable straight to
    MLflow as metrics.
    """
    # `== True` rather than `.fillna(False).to_numpy(dtype=bool)`: these
    # columns are nullable-bool-shaped (True/False/None, see
    # labels/rule_labels.py) - `== True` treats None/NaN as not-True
    # directly, without pandas' fillna-then-downcast path, which throws
    # a FutureWarning on object-dtype columns as of this pandas version.
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
    """
    Real PR-AUC/log loss (models/metrics.py, shared with
    models/rule_pattern/train.py), using rule_evaluated==True rows as a
    VALIDATION set only - never used in training. See module docstring
    for exactly what this does and doesn't prove, and why SMPP-only is
    skipped (mathematically undefined, not computed wrong).

    Evaluated three ways, per README.md's stated convention: overall,
    then per source - an aggregate number can hide one source
    performing badly, and SMPP/SS7's labelled pools look very different
    (see the data reality check in README.md).
    """
    evaluated = (df["rule_evaluated"] == True).to_numpy()  # noqa: E712
    flagged = (df["rule_flagged"] == True).to_numpy() & evaluated
    y_true = flagged[evaluated].astype(int)
    score = anomaly_score[evaluated]
    return evaluate_overall_and_per_source(df[evaluated], "source", y_true, score)


def run(
    sources: list[str],
    data_dir: Path,
    n_estimators: int,
    max_samples: str,
    contamination: str,
    random_state: int,
) -> None:
    # A source-restricted run (e.g. --sources SMPP alone) gets its own
    # MLflow experiment, suffixed by source - keeps a source-specific
    # champion/challenger lineage separate from the combined-sources
    # experiment, so compare_versions.py never compares a challenger
    # trained on one population against a champion trained on another.
    # The full default (both sources) keeps the plain experiment name -
    # no behavior change for existing combined runs.
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
        mlflow.sklearn.log_model(pipeline, name="model")
        print(
            f"Logged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={experiment_name})"
        )

    # Score-per-message output, one file per source, min-max normalized
    # column added for convenience (0-1, easy to eyeball) alongside the
    # raw decision_function-derived score (the actual anomaly_score).
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
