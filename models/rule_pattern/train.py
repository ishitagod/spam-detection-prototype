"""
Trains the supervised layer (`rule_pattern_score`) - LightGBM on
rule_evaluated==True rows only, labelled by rule_flagged. This model can only
ever re-recognize patterns the rules already encode; it is NOT the
layer meant to catch novel spam.

NOT wired into pipeline.py, same reasoning as models/anomaly/train.py:
training is a deliberate, versioned action, not a feature-computation
step. Run this by hand:

    python -m models.rule_pattern.train
    python -m models.rule_pattern.train --n_estimators 200 --learning_rate 0.05

REAL LABEL COMPOSITION, worth knowing before reading the metrics below:
SMPP contributes 2,693 rule_evaluated rows, ALL flagged - zero
confirmed-clean (verified across the full raw dataset, not a sample -
see README.md's data reality check and CLAUDE.md's roadmap notes).
SS7 contributes 349,962, split 284,073 flagged / 65,889 confirmed-clean.
Two consequences:
  1. SMPP-only PR-AUC is mathematically undefined (one class only) and
     is skipped, not silently computed wrong - see models/metrics.py.
  2. Spam is the MAJORITY of this rule-evaluated pool overall (~85%),
     not the minority - the usual "spam is rare" imbalance framing is
     backwards here. This is an artifact of WHICH messages the rule
     engine bothers to evaluate (see the module docstring in
     models/anomaly/train.py for the same point made about
     anomaly_score's evaluation), not the true traffic-wide spam rate.
     No explicit class-weighting is applied here for that reason - it's
     not obviously warranted given the real, measured composition,
     rather than assumed from the generic "imbalanced spam" prior.

EVALUATION: train/test split (stratified by label), PR-AUC + log loss
via models/metrics.py (shared with models/anomaly/train.py, same
single-class-skip guard), evaluated on BOTH train and test sets - a
large train/test gap is the actual overfitting signal to watch for,
given this pool's small-for-SMPP / imbalanced-for-SS7 shape.

--with_embeddings / --with_tfidf: independently toggleable, see
models/rule_pattern/data.py's module docstring for why they're separate
flags rather than one combined switch. --with_embeddings coverage is now
source-dependent: SS7's embeddings.npy is a full-dataset GPU run (all
2,742,301 rows), so `--with_embeddings --sources SS7` trains on the full
SS7 rule_evaluated pool - ready to run, just not yet re-trained/logged to
MLflow as of writing (the one logged rule_pattern_score_with_embeddings
run still reflects the old ~1%-sample era). SMPP has no embeddings.npy
yet - still blocked on its own full-dataset run. --with_tfidf IS
useful today - real standalone signal already measured (PR-AUC 0.934 on
a text-grouped split, see data.py docstring), no sample-coverage blocker.

Either flag routes the split BEFORE featurization, not after: TF-IDF
vocabulary and embedding PCA are corpus-dependent, so they must be fit on
the train fold only (see data.py's train_mask docstring) - fitting on the
full pool first, THEN splitting, would leak test-set information into
featurization itself. The plain default path (neither flag set) doesn't
care about split order since nothing in it is corpus-fit, but the split
now happens first unconditionally, for one consistent code path rather
than two.

Any experimental combination of these flags is logged to a SEPARATE
MLflow experiment (rule_pattern_score_experimental) so an early or
partial-coverage run never gets mistaken for the real baseline candidate
in the MLflow UI - same reasoning as
scripts/check_embedding_dominance.py's separate diagnostics experiment.
with_embeddings/with_tfidf are logged as params on every run, so runs
are filterable/comparable within that one experiment rather than
scattered across per-combination experiment names.
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
from models.anomaly.data import N_EMBEDDING_COMPONENTS
from models.metrics import evaluate_overall_and_per_source
from models.rule_pattern.data import (
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    build_feature_matrix,
    load_labelled_messages,
    load_labelled_messages_with_embeddings,
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
    LightGBM's own library defaults - unless explicitly overridden via
    train.py's CLI, behavior is unchanged from before these existed.

    These are the real anti-single-feature-dominance knobs, added after a
    real observation (SS7 --with_tfidf champion, models/rule_pattern/
    explain.py's SHAP output): tfidf_https alone accounted for a large
    swing in a live prediction (risk_score 0 -> 96 from adding one token).
    That may be a genuinely correct learned pattern (this model's scope is
    known rule-engine patterns, not novel spam - CLAUDE.md), not
    necessarily a bug to eliminate - but it's also a real production risk
    (trivially evadable by not using that literal string). Three
    independent levers, not one, since they attack different mechanisms:
      - colsample_bytree < 1.0: randomly excludes some features from each
        tree, so no single feature can be the split at every tree's root.
      - min_child_samples > 20: requires more rows per leaf, so a leaf
        specialized around one rare-but-strong token needs more support.
      - reg_alpha/reg_lambda > 0: L1/L2 penalty on leaf weights directly
        discourages the large leaf-value swing that produces a near-100
        risk_score jump from one token flipping.
    None of these are tuned/validated yet - starting points to experiment
    with via PR-AUC/log loss (does regularizing hurt the real metric) and
    re-running explain.py (does it actually reduce tfidf_https's SHAP
    magnitude), not assumed to be correct as-is.
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
) -> None:
    print(
        f"Loading rule_evaluated rows for sources: {sources} "
        f"(with_embeddings={with_embeddings}, with_tfidf={with_tfidf}) ..."
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
    y_full = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712

    # Split BEFORE featurization, not after: with either flag on, TF-IDF
    # vocabulary / embedding PCA are corpus-dependent transformers that
    # must never see the test fold during fit (see data.py's train_mask
    # docstring). Splitting first and passing a train_mask into
    # build_feature_matrix() is the one code path that's correct for the
    # default case too (train_mask is a no-op there).
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
    assert np.array_equal(y, y_full), "build_feature_matrix()'s y must match the pre-split labels"

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

    # Any experimental flag routes to a SEPARATE experiment - a plain run
    # (neither flag set) stays the real baseline candidate in
    # MLFLOW_EXPERIMENT_NAME; with_embeddings/with_tfidf are logged as
    # params either way so runs stay filterable/comparable in one place
    # rather than proliferating one experiment per combination.
    experiment_name = (
        MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME
        if (with_embeddings or with_tfidf)
        else MLFLOW_EXPERIMENT_NAME
    )
    # A source-restricted run (e.g. --sources SMPP alone) gets its own
    # experiment on top of that, suffixed by source - same reasoning as
    # models/anomaly/train.py: keeps a source-specific champion/challenger
    # lineage separate from the combined-sources one.
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
        # Each fitted transformer logged as its OWN artifact, not folded
        # into one sklearn Pipeline with the LightGBM model: each only
        # sees its own slice of columns (embedding_cols / text), not the
        # full feature matrix, so they don't chain the way
        # models/anomaly/train.py's single combined preprocessor+model
        # pipeline does. Must be loaded and applied in the same order at
        # inference time later (train/serve skew otherwise). Signature/
        # input_example for each are built from its OWN raw input (raw
        # embedding columns / raw text), matching what serving/scoring.py
        # actually feeds these transformers live - not X, which is the
        # already-combined final feature matrix.
        if "embedding_pca_pipeline" in fitted:
            embedding_cols = [c for c in df.columns if c.startswith("emb_")]
            embedding_sample = df.loc[train_mask, embedding_cols].head(5).to_numpy(dtype=np.float64)
            embedding_signature = infer_signature(
                embedding_sample, fitted["embedding_pca_pipeline"].transform(embedding_sample)
            )
            mlflow.sklearn.log_model(
                fitted["embedding_pca_pipeline"], name="embedding_pca_pipeline",
                signature=embedding_signature, input_example=embedding_sample,
            )
        if "tfidf_vectorizer" in fitted:
            text_sample = df.loc[train_mask, "text"].fillna("").head(5).to_numpy()
            tfidf_signature = infer_signature(
                text_sample, fitted["tfidf_vectorizer"].transform(text_sample).toarray()
            )
            mlflow.sklearn.log_model(
                fitted["tfidf_vectorizer"], name="tfidf_vectorizer",
                signature=tfidf_signature, input_example=text_sample,
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
        "--colsample_bytree", type=float, default=1.0,
        help="Fraction of features randomly sampled per tree - lower (e.g. 0.7) so no "
        "single feature (e.g. a highly-discriminative tfidf_* token) can be the split "
        "at every tree's root. See train_lightgbm()'s docstring.",
    )
    parser.add_argument(
        "--min_child_samples", type=int, default=20,
        help="Minimum rows per leaf - raise (e.g. 50) to require more support before a "
        "leaf specializes around one rare-but-strong token.",
    )
    parser.add_argument(
        "--reg_alpha", type=float, default=0.0, help="L1 regularization on leaf weights.",
    )
    parser.add_argument(
        "--reg_lambda", type=float, default=0.0, help="L2 regularization on leaf weights.",
    )
    parser.add_argument(
        "--with_embeddings",
        action="store_true",
        help="Add PCA-reduced text embeddings as features - not useful until "
        "features/text_embeddings.py's full-dataset run is done (see module docstring). "
        "Independently combinable with --with_tfidf.",
    )
    parser.add_argument(
        "--n_embedding_components", type=int, default=N_EMBEDDING_COMPONENTS
    )
    parser.add_argument(
        "--with_tfidf",
        action="store_true",
        help="Add TF-IDF n-gram features, fit on the train fold only - real "
        "standalone signal already measured (see data.py module docstring). "
        "Independently combinable with --with_embeddings.",
    )
    parser.add_argument("--tfidf_max_features", type=int, default=TFIDF_MAX_FEATURES)
    parser.add_argument(
        "--tfidf_ngram_range", type=int, nargs=2, default=list(TFIDF_NGRAM_RANGE),
        metavar=("MIN_N", "MAX_N"),
    )
    parser.add_argument("--tfidf_min_df", type=int, default=TFIDF_MIN_DF)
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
    )


if __name__ == "__main__":
    main()
