"""
EXPERIMENTAL multiclass fraud-TYPE classifier - trains on cluster-derived
labels (models/fraud_type_classifier/data.py), the production plan's
Stage C bootstrap (docs/sms_spam_technical_architecture_plan.md).

THIS IS NOT A REAL MODEL YET - read before trusting any number this
prints. Default `--label_source suggested` trains on
models/anomaly/suggest_cluster_labels.py's heuristic, UNCONFIRMED guesses -
no human has hand-confirmed a single cluster as of this module's
introduction (docs/experiments/anomaly_clustering.md's step 4 is still
outstanding). A model trained on unconfirmed labels can only ever tell you
"does this feature set carry signal for the CATEGORIES A HEURISTIC
GUESSED", not "does it predict real fraud types" - those are different
questions, and only the second one matters for anything downstream of
this prototype. Every run below logs to `fraud_type_classifier_suggested_
labels`, a SEPARATE MLflow experiment from `fraud_type_classifier` itself
(reserved for `--label_source confirmed` runs) - same "never let an early/
throwaway run get mistaken for the real baseline" convention as
`anomaly_score_diagnostics`/`rule_pattern_score_experimental` elsewhere in
this codebase. Do not promote anything from the `_suggested_labels`
experiment to a registered model - there is nothing here CLAUDE.md's
champion/challenger convention should ever act on until real confirmed
labels exist.

Once docs/experiments/anomaly_clustering.md's step 4/5 produces real
confirmed labels (models/anomaly/ingest_cluster_labels.py writes
cluster_labels.parquet), re-run this with `--label_source confirmed` -
identical code path, real data, and THAT run is the one worth actually
evaluating as a candidate.

Run by hand (not wired into pipeline.py, same reasoning as
models/anomaly/train.py and models/rule_pattern/train.py - a deliberate,
versioned action, not a feature-computation step):

    python -m models.fraud_type_classifier.train --source SS7
    python -m models.fraud_type_classifier.train --source SMPP --label_source confirmed

MULTICLASS, NOT BINARY - models/metrics.py's PR-AUC/log-loss helpers are
binary-specific (rule_pattern_score/anomaly_score's shared convention),
not reused here. Evaluated instead with accuracy + macro-F1 (unweighted
mean F1 across classes - treats a rare fraud type's performance as equally
important as a common one, matching this project's general "don't let an
aggregate number hide a segment doing badly" principle) plus a full
per-class precision/recall/F1 report, via sklearn's own
classification_report - a real, standard multiclass evaluation, not a
bespoke reimplementation.

CLASS-COUNT FILTERING: a class with fewer than 2*MIN_CLASS_COUNT rows
can't be meaningfully stratified-split (sklearn's train_test_split raises
on a class with <2 members in either fold) - classes below
--min_class_count are DROPPED before splitting, not silently merged into
an "other" bucket (that would fabricate a category nobody actually
labeled) or crashed on. Dropped classes/counts are printed, not hidden.
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from config.settings import MLFLOW_TRACKING_URI
from models.fraud_type_classifier.data import build_feature_matrix, load_cluster_labeled_messages

MLFLOW_EXPERIMENT_BASE_NAME = "fraud_type_classifier"
MIN_CLASS_COUNT = 10  # starting point, not tuned - same "documented, not
# proven" status as this project's other threshold constants
# (FAISS_NEAR_DUP_THRESHOLD, SENDER_DIVERSITY_MIN_MSGS).


def drop_rare_classes(df: pd.DataFrame, min_class_count: int) -> pd.DataFrame:
    """Removes rows whose fraud_type_label has fewer than min_class_count
    total examples - see module docstring's CLASS-COUNT FILTERING note.
    Prints exactly what got dropped, doesn't hide it."""
    counts = df["fraud_type_label"].value_counts()
    keep_labels = counts[counts >= min_class_count].index
    dropped = counts[counts < min_class_count]
    if len(dropped):
        print(
            f"  Dropping {len(dropped)} class(es) with < {min_class_count} example(s): "
            f"{dropped.to_dict()}"
        )
    return df[df["fraud_type_label"].isin(keep_labels)].reset_index(drop=True)


def run(
    source: str,
    data_dir: Path,
    label_source: str,
    min_class_count: int,
    test_size: float,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> None:
    experiment_name = (
        MLFLOW_EXPERIMENT_BASE_NAME if label_source == "confirmed"
        else f"{MLFLOW_EXPERIMENT_BASE_NAME}_suggested_labels"
    )

    print(f"Loading cluster-labeled messages for {source} (label_source={label_source!r}) ...")
    df = load_cluster_labeled_messages(source, data_dir, label_source=label_source)
    print(f"  {len(df)} labelled message(s) across {df['fraud_type_label'].nunique()} distinct label(s)")

    df = drop_rare_classes(df, min_class_count)
    n_classes = df["fraud_type_label"].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Only {n_classes} class(es) survive --min_class_count={min_class_count} - "
            "need at least 2 to train a classifier. Lower --min_class_count, or label "
            "more clusters first."
        )
    print(f"  {len(df)} row(s) / {n_classes} class(es) after filtering")

    X, y, feature_names = build_feature_matrix(df)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=y,
    )

    print(f"Training LGBMClassifier (multiclass, {n_classes} classes) ...")
    model = LGBMClassifier(
        objective="multiclass", n_estimators=n_estimators, learning_rate=learning_rate,
        random_state=random_state, verbosity=-1,
    )
    model.fit(X[idx_train], y[idx_train])

    y_pred_test = model.predict(X[idx_test])
    test_accuracy = accuracy_score(y[idx_test], y_pred_test)
    test_macro_f1 = f1_score(y[idx_test], y_pred_test, average="macro")
    report = classification_report(y[idx_test], y_pred_test, zero_division=0)

    print(f"\nTest accuracy: {test_accuracy:.3f} | Test macro-F1: {test_macro_f1:.3f}")
    print("Per-class report (test set):")
    print(report)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "source": source,
            "label_source": label_source,
            "min_class_count": min_class_count,
            "n_rows": len(df),
            "n_classes": n_classes,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
        })
        mlflow.log_metrics({"test_accuracy": test_accuracy, "test_macro_f1": test_macro_f1})
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_dict({"feature_names": feature_names, "classes": sorted(set(y.tolist()))}, "feature_names.json")

        input_example = pd.DataFrame(X[idx_train][:5], columns=feature_names)
        signature = infer_signature(input_example, model.predict(X[idx_train][:5]))
        mlflow.lightgbm.log_model(model, name="model", signature=signature, input_example=input_example)
        print(f"\nLogged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={experiment_name})")
        if label_source == "suggested":
            print(
                "REMINDER: this run is under the _suggested_labels experiment - trained "
                "on unconfirmed heuristic guesses, not real labels. Do not promote it."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--label_source", type=str, default="suggested", choices=["suggested", "confirmed"],
        help="'suggested' (default) = models/anomaly/suggest_cluster_labels.py's unconfirmed "
        "heuristic guesses - the only label source that exists before any hand-labeling is "
        "done. 'confirmed' = real hand-confirmed labels via ingest_cluster_labels.py.",
    )
    parser.add_argument("--min_class_count", type=int, default=MIN_CLASS_COUNT)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    run(
        args.source, Path(args.data_dir), args.label_source, args.min_class_count,
        args.test_size, args.n_estimators, args.learning_rate, args.random_state,
    )


if __name__ == "__main__":
    main()
