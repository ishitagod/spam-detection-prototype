"""
SHAP + LIME explainability for the anomaly (Isolation Forest) model.
Loads an already-trained model + preprocessing pipeline from MLflow by
run_id (never during train.py itself). Two tools, since the input space
is genuinely two kinds of feature:

  - BEHAVIORAL_COLS + NEAR_DUP_COLS + one-hot source (~10-12 features)
    pass through build_preprocessor() unmixed (log1p, then per-feature
    StandardScaler), keeping their identity into the tree splits - SHAP
    TreeExplainer gives exact, whole-dataset, real per-feature attribution.
  - The 384-dim MiniLM embedding genuinely gets mixed - PCA-reduced to
    N_EMBEDDING_COMPONENTS anonymous combinations (`emb_pca_0..29`)
    before the model sees it, so per-dimension SHAP attribution isn't
    actionable. Summed per-row into one CONTENT_EMBEDDING_BUCKET instead
    (summarize_shap_importance()) - "how much did content matter overall".

SHAP SIGN CONVENTION: shap.TreeExplainer explains IsolationForest's raw
ensemble output, where LOWER = more anomalous - opposite of this
project's anomaly_score (higher = more anomalous). Negated in
compute_shap_contributions() so positive = "pushed anomaly_score up" -
verified empirically (Spearman -1.0 between raw output and anomaly_score
on a synthetic check). A reliable ranking signal, not an exact additive
decomposition of anomaly_score's own scale (IsolationForest's own
offset/normalization isn't visible to SHAP's tree walk).

WHY LIME TOO: SHAP explains content only as one aggregate bucket, with
no story for which structured feature values (in original units) would
flip the verdict - LIME's fixed-embedding local surrogate
(make_predict_fn()) answers that for a handful of instances. LIME
doesn't scale like SHAP - fits a fresh surrogate per instance, so only
explains a small chosen set (highest-anomaly rows + a few random for
contrast); its "global" importance is an average over only those
explained instances, not dataset-wide (hence lime_local_importance_approx.csv).

CURRENT SCALE: like train.py, runs against whatever --sample_n the
explained run was trained on (~1% of full corpus as of writing).

Run by hand:
    python -m models.anomaly.explain
    python -m models.anomaly.explain --run_id <specific_run_id>
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this is a script, not a notebook
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from models.anomaly.data import (
    BEHAVIORAL_COLS,
    COUNT_COLS,
    NEAR_DUP_COLS,
    SENDER_AGE_BUCKET_EDGES_DAYS,
    SENDER_AGE_BUCKET_LABELS,
    SENDER_AGE_DAYS_COL,
    SENDER_DIVERSITY_LONG_COL,
    SENDER_DIVERSITY_LONG_KNOWN_COL,
    SENDER_DIVERSITY_LONG_MSGS_COL,
    SENDER_DIVERSITY_MIN_MSGS,
    SENDER_DIVERSITY_SHORT_COL,
    SENDER_DIVERSITY_SHORT_KNOWN_COL,
    SENDER_DIVERSITY_SHORT_MSGS_COL,
    load_source_features,
)
from models.anomaly.train import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, score_anomalies


def resolve_run_id(run_id: str | None) -> str:
    """No run_id given -> latest real isolation_forest run - mirrors
    models/rule_pattern/explain.py's resolve_run_id()."""
    if run_id:
        return run_id
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME], order_by=["start_time DESC"], max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs found in '{MLFLOW_EXPERIMENT_NAME}' - train one first.")
    resolved = runs.iloc[0]["run_id"]
    print(f"No --run_id given, using latest: {resolved}")
    return resolved


def rebuild_dataset(run: mlflow.entities.Run, data_dir: Path) -> pd.DataFrame:
    """Rebuilds the same per-source join train.py's run() used. Unlike
    rule_pattern/explain.py, no train/test split to reproduce - Isolation
    Forest trains unsupervised on the full joined stream, so this full
    rebuilt frame IS the data the run trained on."""
    sources = run.data.params["sources"].split(",")
    frames = [
        load_source_features(data_dir / s, data_dir / s / "messages_with_behavioral.csv")
        for s in sources
    ]
    return pd.concat(frames, ignore_index=True)


def build_interpretable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The non-embedding columns LIME perturbs - log1p'd COUNT_COLS +
    ratios/similarities as-is + one-hot `source` (only when the rebuilt
    df spans >1 source). Embedding columns excluded - see module docstring.

    SENDER_AGE_DAYS_COL is bucketed (same pd.cut()/get_dummies() as
    build_combined_frame()), not passed raw - the fitted pipeline expects
    sender_age_bucket_* columns. Getting this wrong isn't cosmetic:
    main()'s reindex-to-expected_cols would silently zero-fill every
    bucket column (a state no real row can be in) instead of erroring,
    corrupting X_transformed for both SHAP and LIME.

    SENDER_DIVERSITY_SHORT_COL/LONG_COL: same reasoning, gated on the
    same message-count threshold (SENDER_DIVERSITY_MIN_MSGS) rather than
    passed raw.
    """
    transformed = df.copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])

    # Uses `df` (pre-log1p) - same gate as build_combined_frame().
    below_min_short = df[SENDER_DIVERSITY_SHORT_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_short, SENDER_DIVERSITY_SHORT_COL] = np.nan
    transformed[SENDER_DIVERSITY_SHORT_KNOWN_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_SHORT_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].fillna(0.0)

    below_min_long = df[SENDER_DIVERSITY_LONG_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_long, SENDER_DIVERSITY_LONG_COL] = np.nan
    transformed[SENDER_DIVERSITY_LONG_KNOWN_COL] = transformed[SENDER_DIVERSITY_LONG_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_LONG_COL] = transformed[SENDER_DIVERSITY_LONG_COL].fillna(0.0)

    raw_passthrough_behavioral_cols = [
        c for c in BEHAVIORAL_COLS
        if c not in (SENDER_AGE_DAYS_COL, SENDER_DIVERSITY_SHORT_COL, SENDER_DIVERSITY_LONG_COL)
    ]
    diversity_cols = [
        SENDER_DIVERSITY_SHORT_COL, SENDER_DIVERSITY_SHORT_KNOWN_COL,
        SENDER_DIVERSITY_LONG_COL, SENDER_DIVERSITY_LONG_KNOWN_COL,
    ]
    age_bucket = pd.cut(
        transformed[SENDER_AGE_DAYS_COL],
        bins=SENDER_AGE_BUCKET_EDGES_DAYS, labels=SENDER_AGE_BUCKET_LABELS,
    )
    age_bucket_dummies = pd.get_dummies(age_bucket, prefix="sender_age_bucket")
    pieces = [
        transformed[raw_passthrough_behavioral_cols + NEAR_DUP_COLS + diversity_cols],
        age_bucket_dummies,
    ]
    if transformed["source"].nunique() > 1:
        pieces.append(pd.get_dummies(transformed["source"], prefix="source"))
    return pd.concat(pieces, axis=1)


_CONTENT_EMBEDDING_BUCKET = "content_embedding (sum of emb_pca_* |contribution|)"


def compute_shap_contributions(iforest, X_transformed: np.ndarray) -> np.ndarray:
    """SHAP TreeExplainer over the fitted IsolationForest's raw
    (post-preprocessing) input space - exact, one vectorized pass, scales
    to the whole dataset cheaply (unlike LIME below). Returns an
    (n_rows, n_transformed_features) array, sign-flipped from
    shap_values() so positive = "pushed anomaly_score up" - see module
    docstring's SHAP SIGN CONVENTION."""
    explainer = shap.TreeExplainer(iforest)
    raw_shap_values = explainer.shap_values(X_transformed)
    return -raw_shap_values


def summarize_shap_importance(shap_contributions: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Per-feature mean(|contribution|), with every `emb_pca_*` column
    collapsed into one _CONTENT_EMBEDDING_BUCKET row (individual PCA
    components aren't interpretable - see module docstring). Bucket value
    is mean-over-rows of (sum-over-embedding-dims of |contribution|) -
    total content-driven push per message, not an average of per-dimension
    means (which would understate one dominant dimension vs. many small
    consistent ones)."""
    abs_contrib = np.abs(shap_contributions)
    is_embedding = np.array([f.startswith("emb_pca_") for f in feature_names])

    rows = []
    if is_embedding.any():
        rows.append({
            "feature": _CONTENT_EMBEDDING_BUCKET,
            "mean_abs_shap_contribution": float(abs_contrib[:, is_embedding].sum(axis=1).mean()),
        })
    for i, name in enumerate(feature_names):
        if not is_embedding[i]:
            rows.append({"feature": name, "mean_abs_shap_contribution": float(abs_contrib[:, i].mean())})
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap_contribution", ascending=False)
        .reset_index(drop=True)
    )


def make_predict_fn(pipeline, expected_cols: list[str], embedding_cols: list[str], fixed_embedding: np.ndarray):
    """Returns a LIME-compatible predict_fn(perturbed) -> anomaly_score
    array (LimeTabularExplainer regression mode). `interpretable_cols` is
    bound via closure per explain_instance() call. `fixed_embedding`:
    this one instance's real embedding, reattached unchanged to every
    perturbed row - a batch of instances each needs its own call."""
    def build(interpretable_cols: list[str]):
        def predict_fn(perturbed: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(perturbed, columns=interpretable_cols)
            for name, value in zip(embedding_cols, fixed_embedding):
                frame[name] = value
            # Reindex to the fitted ColumnTransformer's exact expected
            # columns - any column missing here (e.g. a source dummy this
            # single-source rebuild never produced) is filled 0.
            for col in expected_cols:
                if col not in frame.columns:
                    frame[col] = 0
            return -pipeline.decision_function(frame[expected_cols])
        return predict_fn
    return build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="MLflow run to explain; default = latest isolation_forest run.",
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--n_local_examples", type=int, default=5,
        help="Highest-anomaly-score rows to explain individually (see module docstring).",
    )
    parser.add_argument(
        "--n_random_examples", type=int, default=3,
        help="Additional random (non-top-anomaly) rows, for contrast.",
    )
    parser.add_argument(
        "--num_lime_samples", type=int, default=500,
        help="Perturbations LIME draws per explained instance. Library default is "
        "5000; lowered since this runs once per instance - raise for a more "
        "faithful local surrogate at direct runtime cost.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/explanations/anomaly"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = resolve_run_id(args.run_id)
    run = mlflow.get_run(run_id)

    print(f"Loading pipeline (preprocessor + Isolation Forest) from run {run_id} ...")
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    reduce_step = pipeline.named_steps["preprocessor"].named_steps["reduce"]
    expected_cols = list(reduce_step.feature_names_in_)
    embedding_cols = [c for c in expected_cols if c.startswith("emb_")]
    interpretable_cols = [c for c in expected_cols if c not in embedding_cols]

    print(f"Rebuilding run {run_id}'s training data (sources={run.data.params['sources']}) ...")
    df = rebuild_dataset(run, Path(args.data_dir))
    interpretable_frame = build_interpretable_frame(df)
    for col in interpretable_cols:
        if col not in interpretable_frame.columns:
            interpretable_frame[col] = 0  # same reasoning as make_predict_fn()'s reindex
    interpretable_frame = interpretable_frame[interpretable_cols]
    embeddings = df[embedding_cols].to_numpy(dtype=np.float64)
    print(f"  {len(df)} row(s), {len(interpretable_cols)} interpretable feature(s) (+{len(embedding_cols)} embedding dims held fixed per-instance)")

    full_frame = pd.concat(
        [interpretable_frame.reset_index(drop=True), df[embedding_cols].reset_index(drop=True)], axis=1,
    )[expected_cols]
    X_transformed = pipeline.named_steps["preprocessor"].transform(full_frame)
    anomaly_score = score_anomalies(pipeline.named_steps["iforest"], X_transformed)

    # transformed_feature_names is the run's own logged column order
    # (train.py's feature_names.json) - authoritative, not re-derived here.
    transformed_feature_names = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/feature_names.json"
    )["feature_names"]
    print(f"\nComputing SHAP TreeExplainer contributions over all {len(df)} row(s) ...")
    shap_contributions = compute_shap_contributions(pipeline.named_steps["iforest"], X_transformed)
    shap_importance = summarize_shap_importance(shap_contributions, transformed_feature_names)
    print("Global feature importance (mean |SHAP contribution to anomaly_score|):")
    print(shap_importance.to_string(index=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_importance_path = output_dir / "shap_global_importance.csv"
    shap_importance.to_csv(shap_importance_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(shap_importance) + 1))
    ordered = shap_importance.iloc[::-1]
    ax.barh(ordered["feature"], ordered["mean_abs_shap_contribution"])
    ax.set_xlabel("mean |SHAP contribution to anomaly_score| (sign-flipped - see module docstring)")
    shap_plot_path = output_dir / "shap_global_importance.png"
    fig.savefig(shap_plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    rng = np.random.RandomState(args.random_state)
    top_idx = np.argsort(anomaly_score)[::-1][: args.n_local_examples]
    remaining = np.setdiff1d(np.arange(len(df)), top_idx)
    random_idx = rng.choice(remaining, size=min(args.n_random_examples, len(remaining)), replace=False)
    explain_idx = np.concatenate([top_idx, random_idx])

    explainer = LimeTabularExplainer(
        interpretable_frame.to_numpy(dtype=np.float64),
        feature_names=interpretable_cols,
        categorical_features=[i for i, c in enumerate(interpretable_cols) if c.startswith("source_")],
        mode="regression",
        random_state=args.random_state,
    )

    print(f"\nExplaining {len(explain_idx)} instance(s) ({len(top_idx)} top-anomaly + {len(random_idx)} random) ...")
    local_rows = []
    representative_explanation = None
    for rank, i in enumerate(explain_idx):
        frame_predict_fn = make_predict_fn(pipeline, expected_cols, embedding_cols, embeddings[i])(interpretable_cols)
        explanation = explainer.explain_instance(
            interpretable_frame.to_numpy(dtype=np.float64)[i],
            frame_predict_fn,
            num_features=len(interpretable_cols),
            num_samples=args.num_lime_samples,
        )
        kind = "top_anomaly" if rank < len(top_idx) else "random"
        print(f"\n  row {i} ({kind}, anomaly_score={anomaly_score[i]:.4f}):")
        for feature, weight in explanation.as_list():
            print(f"    {feature:40s} weight={weight:+.4f}")
            local_rows.append({"row": int(i), "kind": kind, "feature": feature, "weight": weight})
        if representative_explanation is None and kind == "top_anomaly":
            representative_explanation = (i, explanation)

    local_df = pd.DataFrame(local_rows)
    local_path = output_dir / "lime_local_explanations.csv"
    local_df.to_csv(local_path, index=False)

    # Average over only the explained instances, not the whole dataset -
    # named _approx to keep that honest.
    approx_importance = (
        local_df.assign(feature_base=local_df["feature"].str.extract(r"^([^ <>=]+)")[0])
        .groupby("feature_base")["weight"]
        .apply(lambda w: w.abs().mean())
        .sort_values(ascending=False)
        .rename("mean_abs_lime_weight")
        .reset_index()
    )
    print(f"\nApproximate feature importance (mean |LIME weight| over the {len(explain_idx)} explained instance(s) only):")
    print(approx_importance.to_string(index=False))
    importance_path = output_dir / "lime_local_importance_approx.csv"
    approx_importance.to_csv(importance_path, index=False)

    plot_path = None
    if representative_explanation is not None:
        row_i, explanation = representative_explanation
        fig = explanation.as_pyplot_figure()
        plot_path = output_dir / "lime_local_example.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"\nSaved representative local explanation plot (row {row_i}) to {plot_path}")

    # Log back to the same run, not a new one - ties this to the model version.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(shap_importance_path))
        mlflow.log_artifact(str(shap_plot_path))
        mlflow.log_artifact(str(local_path))
        mlflow.log_artifact(str(importance_path))
        if plot_path is not None:
            mlflow.log_artifact(str(plot_path))
    print(f"\nLogged SHAP + LIME explanation artifacts back to run {run_id}")


if __name__ == "__main__":
    main()
