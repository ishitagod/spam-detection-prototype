"""
Trains rule_pattern_score: LightGBM on rule_evaluated==True rows only,
labelled by rule_flagged. Recognizes known rule-engine patterns; does not
catch novel spam.

Not wired into pipeline.py - training is a deliberate, versioned action.
Run manually:

    python -m models.rule_pattern.train
    python -m models.rule_pattern.train --n_estimators 200 --learning_rate 0.05

Label composition: SMPP has 2,693 rule_evaluated rows, all flagged (zero
confirmed-clean, so SMPP-only PR-AUC is undefined and skipped - see
models/metrics.py). SS7 has 349,962 rows, 284,073 flagged / 65,889 clean.
Spam is the majority of this pool overall (~85%) because of which messages
the rule engine evaluates, not the true traffic-wide rate - no class
weighting applied.

Evaluation: stratified train/test split, PR-AUC + log loss via
models/metrics.py, on both train and test sets - a large train/test gap
is the overfitting signal to watch for.

--with_embeddings / --with_tfidf: independent flags (see data.py's module
docstring). SS7's embeddings.npy is a full-dataset run, so
`--with_embeddings --sources SS7` is ready to run (not yet re-logged to
MLflow). SMPP has no embeddings.npy yet. --with_tfidf has measured
standalone signal (PR-AUC 0.934 on a text-grouped split, see data.py).

Either flag splits before featurization, since TF-IDF vocabulary and
embedding PCA are corpus-dependent and must fit on the train fold only
(data.py's train_mask). The plain default path splits first too, for one
consistent code path.

Any experimental flag combination logs to a separate MLflow experiment
(rule_pattern_score_experimental) instead of the baseline one, so a
partial-coverage run is never mistaken for the real candidate.
with_embeddings/with_tfidf are logged as params on every run either way.

--include_content_labels: expands the training pool with confident
positives mined from rule_evaluated==False rows, scored by a
LogisticRegression fit on real rule_flagged labels
(labels/rule_labels.py::fit_content_flag_weights()/
content_flagged_by_weight()) rather than an unweighted flag count. Must
be fit on a `sources` pool with both classes present (SMPP-only raises).
Positive-only additions - see
label_content_flagged_positives()'s docstring. Adds a test-set breakdown
by label_source, since content_static_rules labels are partly derived
from features this model also uses (a good score there is expected, not
evidence of generalization).

Combinable with --with_embeddings: content-labelled rows get embeddings
joined the same way as the base pool (join_embeddings()); rows without
coverage are dropped from the addition, not the whole run. --with_tfidf
needs no such join. In practice this means `--sources SS7` for the
embeddings combination today.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from config.settings import MLFLOW_TRACKING_URI
from labels.rule_labels import content_flag_weights, fit_content_flag_weights
from models.anomaly.data import N_EMBEDDING_COMPONENTS
from models.metrics import evaluate_overall_and_per_source
from models.rule_pattern.data import (
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    build_feature_matrix,
    join_embeddings,
    label_content_flagged_positives,
    load_labelled_messages,
    load_labelled_messages_with_embeddings,
    load_unevaluated_messages,
)

MLFLOW_EXPERIMENT_NAME = "light_gbm"
MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME = "rule_pattern_score_experimental"


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    random_state: int = 42,
    colsample_bytree: float = 1.0,
    min_child_samples: int = 20,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> lgb.LGBMClassifier:
    """
    colsample_bytree/min_child_samples/reg_alpha/reg_lambda default to
    LightGBM's own library defaults - no behavior change unless overridden
    via CLI.

    Added after observing (SS7 --with_tfidf champion, explain.py's SHAP
    output) that a single token (tfidf_https) swung risk_score from 0 to
    96. May be a correct learned pattern, but also a real evasion risk.
    Three independent anti-single-feature-dominance levers:
      - colsample_bytree < 1.0: excludes features per tree
      - min_child_samples > 20: requires more support per leaf
      - reg_alpha/reg_lambda > 0: L1/L2 penalty on leaf weights
    Not tuned/validated yet - starting points only.
    """
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def run(
    sources: list[str],
    data_dir: Path,
    test_size: float,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
    colsample_bytree: float = 1.0,
    min_child_samples: int = 20,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    with_embeddings: bool = False,
    n_embedding_components: int = N_EMBEDDING_COMPONENTS,
    with_tfidf: bool = False,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
    tfidf_ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    tfidf_min_df: int = TFIDF_MIN_DF,
    include_content_labels: bool = False,
    content_flag_weight_threshold: float = 0.5,
) -> None:
    print(
        f"Loading rule_evaluated rows for sources: {sources} "
        f"(with_embeddings={with_embeddings}, with_tfidf={with_tfidf}, "
        f"include_content_labels={include_content_labels}) ..."
    )
    frames = []
    for source in sources:
        source_dir = data_dir / source
        messages_path = source_dir / "messages_with_behavioral.csv"
        if with_embeddings:
            df = load_labelled_messages_with_embeddings(source_dir, messages_path)
        else:
            df = load_labelled_messages(messages_path)
        print(f"  {source}: {len(df)} row(s)")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if include_content_labels:
        # Fit on the labelled pool built above, then score each source's
        # rule_evaluated==False rows with the fitted weights.
        weight_model = fit_content_flag_weights(df)
        print("Content-flag weights (LogisticRegression fit on real rule_flagged labels):")
        for name, weight in sorted(
            content_flag_weights(weight_model).items(), key=lambda kv: -abs(kv[1])
        ):
            print(f"  {name}: {weight:+.3f}")

        content_frames = []
        for source in sources:
            source_dir = data_dir / source
            messages_path = source_dir / "messages_with_behavioral.csv"
            unevaluated = load_unevaluated_messages(messages_path)
            content_df = label_content_flagged_positives(
                unevaluated, weight_model, threshold=content_flag_weight_threshold
            )
            if with_embeddings:
                # Inner join, silently restricted to whatever coverage
                # source_dir's embeddings.npy actually has.
                before = len(content_df)
                content_df = join_embeddings(content_df, source_dir)
                if len(content_df) < before:
                    print(
                        f"  {source}: {before - len(content_df)} content-flagged row(s) "
                        f"dropped - no embedding coverage for them in {source_dir}"
                    )
            print(
                f"  {source}: +{len(content_df)} content-flagged row(s) "
                f"(rule_evaluated==False, weight_threshold={content_flag_weight_threshold})"
            )
            content_frames.append(content_df)
        df = pd.concat([df] + content_frames, ignore_index=True)

    y_full = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712

    # Split before featurization - TF-IDF/embedding PCA must fit on the
    # train fold only (see data.py's train_mask docstring).
    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=y_full,
        random_state=random_state,
    )
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    X, y, feature_names, fitted = build_feature_matrix(
        df,
        train_mask=train_mask,
        use_embeddings=with_embeddings,
        n_embedding_components=n_embedding_components,
        use_tfidf=with_tfidf,
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_min_df=tfidf_min_df,
    )
    print(
        f"Total: {len(df)} rows, {X.shape[1]} feature(s), {int(y.sum())} positive ({y.mean():.1%})"
    )
    assert np.array_equal(
        y, y_full
    ), "build_feature_matrix()'s y must match the pre-split labels"

    X_train, y_train, df_train = X[idx_train], y[idx_train], df.iloc[idx_train]
    X_test, y_test, df_test = X[idx_test], y[idx_test], df.iloc[idx_test]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Training LightGBM ...")
    model = train_lightgbm(
        X_train,
        y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )

    train_score = model.predict_proba(X_train)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    print("Train set evaluation:")
    train_metrics = evaluate_overall_and_per_source(
        df_train, "source", y_train, train_score, prefix="train_"
    )
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    print(
        "Test set evaluation (the real number - train set above is a sanity/overfitting check only):"
    )
    test_metrics = evaluate_overall_and_per_source(
        df_test, "source", y_test, test_score, prefix="test_"
    )
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    if include_content_labels:
        # content_static_rules rows' label is partly derived from
        # CONTENT_FLAG_COLS, which are also features - a good score on
        # that slice alone isn't evidence of generalization.
        print(
            "Test set evaluation by label_source (content_static_rules rows are partly "
            "self-referential):"
        )
        label_source_metrics = evaluate_overall_and_per_source(
            df_test, "label_source", y_test, test_score, prefix="test_by_label_source_"
        )
        for k, v in label_source_metrics.items():
            print(f"  {k}: {v}")
        test_metrics.update(label_source_metrics)

    # Any experimental flag routes to a separate experiment; a plain run
    # stays in MLFLOW_EXPERIMENT_NAME. Flags are logged as params either way.
    experiment_name = (
        MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME
        if (with_embeddings or with_tfidf or include_content_labels)
        else MLFLOW_EXPERIMENT_NAME
    )
    # Source-restricted runs get their own experiment, suffixed by source,
    # keeping that champion/challenger lineage separate from combined-sources.
    if sorted(sources) != sorted(["SMPP", "SS7"]):
        experiment_name += "_" + "_".join(sources)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "sources": ",".join(sources),
                "n_rows": len(df),
                "n_features": X.shape[1],
                "n_positive": int(y.sum()),
                "test_size": test_size,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": random_state,
                "colsample_bytree": colsample_bytree,
                "min_child_samples": min_child_samples,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "with_embeddings": with_embeddings,
                "with_tfidf": with_tfidf,
                "include_content_labels": include_content_labels,
                **(
                    {"content_flag_weight_threshold": content_flag_weight_threshold}
                    if include_content_labels
                    else {}
                ),
                **(
                    {"n_embedding_components": n_embedding_components}
                    if with_embeddings
                    else {}
                ),
                **(
                    {
                        "tfidf_max_features": tfidf_max_features,
                        "tfidf_ngram_range": str(tfidf_ngram_range),
                        "tfidf_min_df": tfidf_min_df,
                    }
                    if with_tfidf
                    else {}
                ),
            }
        )
        mlflow.log_metrics({**train_metrics, **test_metrics})
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        model_signature = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.lightgbm.log_model(
            model, name="model", signature=model_signature, input_example=X_train[:5]
        )
        # Each fitted transformer logged as its own artifact (not folded
        # into one sklearn Pipeline) since each sees only its own slice of
        # columns. Must be loaded and applied in the same order at
        # inference time. Signature/input_example use each transformer's
        # own raw input, matching what serving/scoring.py feeds it live.
        if "embedding_pca_pipeline" in fitted:
            embedding_cols = [c for c in df.columns if c.startswith("emb_")]
            embedding_sample = (
                df.loc[train_mask, embedding_cols].head(5).to_numpy(dtype=np.float64)
            )
            embedding_signature = infer_signature(
                embedding_sample,
                fitted["embedding_pca_pipeline"].transform(embedding_sample),
            )
            mlflow.sklearn.log_model(
                fitted["embedding_pca_pipeline"],
                name="embedding_pca_pipeline",
                signature=embedding_signature,
                input_example=embedding_sample,
            )
        if "tfidf_vectorizer" in fitted:
            text_sample = df.loc[train_mask, "text"].fillna("").head(5).to_numpy()
            tfidf_signature = infer_signature(
                text_sample, fitted["tfidf_vectorizer"].transform(text_sample).toarray()
            )
            mlflow.sklearn.log_model(
                fitted["tfidf_vectorizer"],
                name="tfidf_vectorizer",
                signature=tfidf_signature,
                input_example=text_sample,
            )
        print(
            f"Logged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={experiment_name})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, nargs="+", default=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=1.0,
        help="Fraction of features sampled per tree - lower to reduce single-feature "
        "dominance. See train_lightgbm()'s docstring.",
    )
    parser.add_argument(
        "--min_child_samples",
        type=int,
        default=20,
        help="Minimum rows per leaf - raise to require more support per leaf.",
    )
    parser.add_argument(
        "--reg_alpha", type=float, default=0.0, help="L1 regularization on leaf weights."
    )
    parser.add_argument(
        "--reg_lambda", type=float, default=0.0, help="L2 regularization on leaf weights."
    )
    parser.add_argument(
        "--with_embeddings",
        action="store_true",
        help="Add PCA-reduced text embeddings as features. Combinable with --with_tfidf.",
    )
    parser.add_argument(
        "--n_embedding_components", type=int, default=N_EMBEDDING_COMPONENTS
    )
    parser.add_argument(
        "--with_tfidf",
        action="store_true",
        help="Add TF-IDF n-gram features, fit on the train fold only. Combinable with "
        "--with_embeddings.",
    )
    parser.add_argument("--tfidf_max_features", type=int, default=TFIDF_MAX_FEATURES)
    parser.add_argument(
        "--tfidf_ngram_range",
        type=int,
        nargs=2,
        default=list(TFIDF_NGRAM_RANGE),
        metavar=("MIN_N", "MAX_N"),
    )
    parser.add_argument("--tfidf_min_df", type=int, default=TFIDF_MIN_DF)
    parser.add_argument(
        "--include_content_labels",
        action="store_true",
        help="Expand training with confident positives from rule_evaluated==False rows, "
        "scored by a LogisticRegression fit on real rule_flagged labels. See "
        "label_content_flagged_positives()'s docstring.",
    )
    parser.add_argument(
        "--content_flag_weight_threshold",
        type=float,
        default=0.5,
        help="Minimum fitted-model P(rule_flagged) to count as a confident content-flagged "
        "positive. Only used with --include_content_labels.",
    )
    args = parser.parse_args()
    run(
        args.sources,
        Path(args.data_dir),
        args.test_size,
        args.n_estimators,
        args.learning_rate,
        args.max_depth,
        args.random_state,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        with_embeddings=args.with_embeddings,
        n_embedding_components=args.n_embedding_components,
        with_tfidf=args.with_tfidf,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_range=tuple(args.tfidf_ngram_range),
        tfidf_min_df=args.tfidf_min_df,
        include_content_labels=args.include_content_labels,
        content_flag_weight_threshold=args.content_flag_weight_threshold,
    )


if __name__ == "__main__":
    main()
