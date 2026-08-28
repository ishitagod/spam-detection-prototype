"""
SHAP explainability for LightGBM. Loads an ALREADY
TRAINED model from MLflow by run_id.

WHY THIS LOADS run.data.params, NOT JUST run.data (the actual "how does it
link to the model" answer): the feature matrix this script builds MUST be
IDENTICAL in shape/column-order to whatever that specific run was trained
on, or LightGBM raises a hard shape error (verified: loading SS7-only data
against a model trained on sources=SMPP,SS7 fails with "8 features vs 9
expected" - SMPP+SS7 combined gives a `source_SMPP` AND `source_SS7` dummy
column, SS7-alone collapses to just one). So `sources`, `test_size`, and
`random_state` are read back from the run's own logged params - never
re-guessed - and the resulting feature matrix's column names are asserted
against the run's own logged feature_names.json before anything else runs.

Uses shap.TreeExplainer, not KernelExplainer/LIME: exact (not sampled) for
tree ensembles, and LightGBM's Booster is exactly what it's built for - see
docs/experiments/rule_pattern.md for why this is the right tool for THIS
model specifically (LIME's approximate, model-agnostic approach is reserved
for the anomaly model instead, once it has real full-dataset embeddings).

Run by hand:
    python -m models.rule_pattern.explain
    python -m models.rule_pattern.explain --run_id <specific_run_id>
"""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use(
    "Agg"
)  # headless: this is a script, not a notebook - no display to render to
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from models.rule_pattern.data import build_feature_matrix, load_labelled_messages
from models.rule_pattern.train import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def resolve_run_id(run_id: str | None) -> str:
    """
    No run_id given -> latest real run, checked across BOTH the real
    baseline experiment and the experimental one (with_embeddings/
    with_tfidf runs) - imported directly from models.rule_pattern.train
    rather than redefined here, so this can never silently drift out of
    sync with wherever train.py actually logs to (this already happened
    once: train.py's experiment name changed and this file's own stale
    copy stopped finding new runs).
    """
    if run_id:
        return run_id
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME, MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No runs found in '{MLFLOW_EXPERIMENT_NAME}' or "
            f"'{MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME}' - train one first."
        )
    resolved = runs.iloc[0]["run_id"]
    print(f"No --run_id given, using latest: {resolved}")
    return resolved


def rebuild_test_split(run: mlflow.entities.Run, data_dir: Path):
    """
    Rebuilds the EXACT held-out test set that run trained/evaluated on, by
    reading sources/test_size/random_state/with_embeddings/with_tfidf back
    from the run's own params - not fresh CLI defaults, which could
    silently mismatch what the loaded model actually saw. Also rebuilds
    the same train_mask train.py used, since with_tfidf/with_embeddings
    runs fit their vectorizer/PCA on the train fold only - passing the
    wrong mask here would silently score the test set through a
    differently-fit transformer than the one the model was trained
    against.
    """
    params = run.data.params
    sources = params["sources"].split(",")
    test_size = float(params["test_size"])
    random_state = int(params["random_state"])
    with_embeddings = params.get("with_embeddings") == "True"
    with_tfidf = params.get("with_tfidf") == "True"

    if with_embeddings:
        from models.rule_pattern.data import load_labelled_messages_with_embeddings
        frames = [
            load_labelled_messages_with_embeddings(data_dir / s, data_dir / s / "messages_with_behavioral.csv")
            for s in sources
        ]
    else:
        frames = [
            load_labelled_messages(data_dir / s / "messages_with_behavioral.csv")
            for s in sources
        ]
    df = pd.concat(frames, ignore_index=True)
    y_full = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712

    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=y_full,
        random_state=random_state,
    )
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    tfidf_ngram_range = (
        ast.literal_eval(params["tfidf_ngram_range"]) if "tfidf_ngram_range" in params else (1, 3)
    )
    X, y, feature_names, _fitted = build_feature_matrix(
        df,
        train_mask=train_mask,
        use_embeddings=with_embeddings,
        n_embedding_components=int(params["n_embedding_components"]) if with_embeddings else 30,
        use_tfidf=with_tfidf,
        tfidf_max_features=int(params.get("tfidf_max_features", 500)),
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_min_df=int(params.get("tfidf_min_df", 5)),
    )
    return X[idx_test], y[idx_test], feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run to explain; default = latest rule_pattern_score run.",
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--n_summary_samples",
        type=int,
        default=5000,
        help="Rows plotted in the summary plot - TreeExplainer itself is exact/fast on the full test set, this only caps plot rendering cost.",
    )
    parser.add_argument(
        "--n_local_examples",
        type=int,
        default=3,
        help="How many individual test rows to print a local explanation for.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/explanations/rule_pattern"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = resolve_run_id(args.run_id)
    run = mlflow.get_run(run_id)

    print(f"Loading model + feature_names.json from run {run_id} ...")
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    booster = (
        model.booster_
    )  # TreeExplainer needs the native Booster, not the sklearn wrapper
    logged_feature_names = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/feature_names.json"
    )["feature_names"]

    print(
        "Rebuilding this run's exact held-out test set (same sources/test_size/random_state it trained with) ..."
    )
    X_test, y_test, feature_names = rebuild_test_split(run, Path(args.data_dir))
    assert feature_names == logged_feature_names, (
        f"Feature mismatch: rebuilt {feature_names} vs run's logged {logged_feature_names} - "
        "the run's params and models/rule_pattern/data.py have drifted apart, fix before trusting SHAP output."
    )
    print(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} feature(s)")

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)
    # LightGBM binary classification via TreeExplainer: shape is either a
    # single (n, features) array already for the positive class, a
    # [neg, pos] list, or (n, features, 2) depending on shap/lightgbm
    # version - normalize to "positive class" contributions once here so
    # everything below doesn't have to guess.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- global: which features drive rule_pattern_score overall ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    print("\nGlobal feature importance (mean |SHAP value|):")
    print(importance.to_string(index=False))
    importance_path = output_dir / "shap_global_importance.csv"
    importance.to_csv(importance_path, index=False)

    plot_idx = np.random.RandomState(42).choice(
        len(X_test),
        size=min(args.n_summary_samples, len(X_test)),
        replace=False,
    )
    shap.summary_plot(
        shap_values[plot_idx], X_test[plot_idx], feature_names=feature_names, show=False
    )
    summary_plot_path = output_dir / "shap_summary.png"
    plt.savefig(summary_plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    # --- local: why THIS specific message got the score it did ---
    print(f"\nLocal explanations for {args.n_local_examples} example test row(s):")
    for i in range(min(args.n_local_examples, len(X_test))):
        print(
            f"\n  row {i}  (true label={'spam' if y_test[i] == 1 else 'clean'}, "
            f"base_value={explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]:.4f}):"
        )
        contribs = pd.Series(shap_values[i], index=feature_names).sort_values(
            key=np.abs, ascending=False
        )
        for feat, val in contribs.items():
            print(
                f"    {feat:35s} value={X_test[i][feature_names.index(feat)]:.3f}  shap={val:+.4f}"
            )

    # Log back to the SAME run, not a new one - ties this explanation to
    # the specific model version it was computed from (MLflow convention
    # from docs/ml/modeling.md: champion/challenger tracking is explicit).
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(summary_plot_path))
    print(
        f"\nLogged shap_global_importance.csv + shap_summary.png back to run {run_id}"
    )


if __name__ == "__main__":
    main()
