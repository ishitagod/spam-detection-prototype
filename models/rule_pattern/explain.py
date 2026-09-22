"""
SHAP explainability for LightGBM. Loads an already-trained model from
MLflow by run_id.

The rebuilt feature matrix must match the trained run's shape/column
order exactly, or LightGBM raises a shape error (e.g. SMPP+SS7 combined
adds a `source_SMPP` and `source_SS7` dummy column; SS7-alone collapses
to one). So `sources`, `test_size`, and `random_state` are read back from
the run's own logged params, and the rebuilt feature names are asserted
against the run's logged feature_names.json.

Uses shap.TreeExplainer (exact for tree ensembles), not KernelExplainer/
LIME - LIME is reserved for the anomaly model instead.

Run by hand:
    python -m models.rule_pattern.explain
    python -m models.rule_pattern.explain --run_id <specific_run_id>
"""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
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
    No run_id given -> latest run, checked across both the baseline and
    experimental experiments - names imported directly from
    models.rule_pattern.train so this can't drift out of sync with
    wherever train.py actually logs to.
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
    Rebuilds the exact held-out test set that run trained/evaluated on, by
    reading sources/test_size/random_state/with_embeddings/with_tfidf back
    from the run's own params rather than fresh CLI defaults. Also
    rebuilds the same train_mask train.py used, since with_tfidf/
    with_embeddings runs fit their vectorizer/PCA on the train fold only.
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
        help="Rows plotted in the summary plot - caps rendering cost only.",
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
    booster = model.booster_  # TreeExplainer needs the native Booster
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
    # Shape varies by shap/lightgbm version - normalize to positive-class
    # contributions once here.
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

    # Log back to the same run, not a new one - ties this explanation to
    # the specific model version it was computed from.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(summary_plot_path))
    print(
        f"\nLogged shap_global_importance.csv + shap_summary.png back to run {run_id}"
    )


if __name__ == "__main__":
    main()
