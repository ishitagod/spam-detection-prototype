"""
LIME explainability for the anomaly (Isolation Forest) model. Loads an
ALREADY TRAINED model + preprocessing pipeline from MLflow by run_id -
same "explain a specific promoted/candidate run, never during train.py
itself" convention as models/rule_pattern/explain.py.

WHY LIME, NOT SHAP (the inverse choice from models/rule_pattern/explain.py
- see that module's docstring): the Isolation Forest never sees the 12
hand-built behavioral/near-dup/source features directly - it sees
models/anomaly/data.py's build_preprocessor() OUTPUT (log1p -> PCA'd
embedding components -> joint StandardScaler). A tree-SHAP explanation
would attribute anomaly_score to e.g. `emb_pca_7`, meaningless to an
analyst. LIME is model-agnostic - it wraps the WHOLE
Pipeline(preprocessor + iforest) as one black box and perturbs the
ORIGINAL interpretable columns instead, so the explanation comes back in
terms an analyst actually recognizes.

SCOPE: explains ONLY the interpretable behavioral/near-dup/source columns
(BEHAVIORAL_COLS + NEAR_DUP_COLS + one-hot source) - the 384-dim MiniLM
embedding is held FIXED at each explained instance's own real value while
LIME perturbs everything else, never itself perturbed/explained. Raw
embedding dimensions carry no human-interpretable meaning even before PCA
(opaque sentence-transformer components) - "how much did dimension 214
contribute" isn't an answer an analyst can act on either way. Holding it
fixed and explaining only the structured features answers the actually
useful question ("was this flagged for BEHAVIORAL reasons, or for its
semantic content") at the cost of never explaining the content signal
itself - a deliberate trade-off, not an oversight. Revisit with a
text-specific tool (attention/token-level) if the content side itself
ever needs explaining - not LIME's job.

LIME DOES NOT SCALE THE WAY SHAP's TreeExplainer DOES: TreeExplainer
computes exact SHAP values for an entire test set in one vectorized pass
(models/rule_pattern/explain.py explains thousands of rows cheaply). LIME
fits a fresh local linear surrogate PER INSTANCE, each requiring
`num_lime_samples` calls through the full pipeline - explaining even a few
hundred rows this way is a real cost. So this script explains a SMALL,
deliberately chosen set of instances (the highest-anomaly-score rows, plus
a few random ones for contrast) rather than a full test set, and the
"global" importance below is an AVERAGE OVER ONLY THOSE EXPLAINED
INSTANCES, not a true dataset-wide statistic the way SHAP's is - see
main()'s --n_local_examples/--n_random_examples and the output CSV's own
naming (lime_local_importance_approx.csv, not shap's _global_).

CURRENT SCALE CAVEAT: like models/anomaly/train.py, this runs against
whatever features/text_embeddings.py's --sample_n sample the explained
run was itself trained on (~1% of the full corpus as of writing) - not a
new limitation, just inherited.

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
import mlflow.sklearn
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from models.anomaly.data import (
    BEHAVIORAL_COLS,
    COUNT_COLS,
    NEAR_DUP_COLS,
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
    """
    Rebuilds the same per-source join models/anomaly/train.py's run()
    used (load_source_features() per source in the run's own logged
    `sources` param, concatenated) - unlike rule_pattern/explain.py there
    is no train/test split to reproduce: Isolation Forest trains
    unsupervised on the FULL joined stream (see models/anomaly/train.py's
    module docstring), so "the data this run trained on" IS the full
    rebuilt frame, not a slice of it.
    """
    sources = run.data.params["sources"].split(",")
    frames = [
        load_source_features(data_dir / s, data_dir / s / "messages_with_behavioral.csv")
        for s in sources
    ]
    return pd.concat(frames, ignore_index=True)


def build_interpretable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    The non-embedding columns LIME actually perturbs - log1p'd count
    columns (mirrors models/anomaly/data.py's build_feature_matrix()
    exactly, same COUNT_COLS) + ratios/similarities as-is + one-hot
    `source` (only added when this rebuilt df spans more than one source
    - same "don't add a constant, information-free column" rule
    build_feature_matrix() itself now follows). Embedding columns are
    deliberately NOT included here - see module docstring.
    """
    transformed = df.copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])
    pieces = [transformed[BEHAVIORAL_COLS + NEAR_DUP_COLS]]
    if transformed["source"].nunique() > 1:
        pieces.append(pd.get_dummies(transformed["source"], prefix="source"))
    return pd.concat(pieces, axis=1)


def make_predict_fn(pipeline, expected_cols: list[str], embedding_cols: list[str], fixed_embedding: np.ndarray):
    """
    Returns a LIME-compatible predict_fn(perturbed: np.ndarray) ->
    np.ndarray of anomaly_score (LimeTabularExplainer's regression mode:
    a plain 1-D array of predicted values, not class probabilities).

    `interpretable_cols` (the columns `perturbed`'s columns correspond to,
    positionally) is bound via closure at call sites below - kept out of
    this signature since it's fixed per explain_instance() call, not per
    predict_fn call. `fixed_embedding`: this ONE instance's real embedding
    values, reattached unchanged to every perturbed row (see module
    docstring's SCOPE note) - a batch of instances would each need their
    own predict_fn/explainer call, never a shared fixed embedding.
    """
    def build(interpretable_cols: list[str]):
        def predict_fn(perturbed: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(perturbed, columns=interpretable_cols)
            for name, value in zip(embedding_cols, fixed_embedding):
                frame[name] = value
            # Reindex to the FITTED ColumnTransformer's exact expected
            # columns/order - not assumed from how this script happens to
            # have built `frame` - any column the fitted preprocessor
            # wants that this rebuild didn't produce (e.g. this explain
            # run's df has only one source, dropping a dummy column a
            # multi-source training run had) is filled 0, matching "that
            # source never appeared in this row" honestly rather than
            # erroring.
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
        help="Highest-anomaly-score rows to explain individually (see module docstring on why LIME can't cheaply cover a whole test set).",
    )
    parser.add_argument(
        "--n_random_examples", type=int, default=3,
        help="Additional random (non-top-anomaly) rows to explain, for contrast against the top-anomaly set.",
    )
    parser.add_argument(
        "--num_lime_samples", type=int, default=500,
        help="Perturbations LIME draws PER explained instance (its own num_samples param) - "
        "LIME's library default is 5000; lowered here since this runs once per instance "
        "(see module docstring) - raise for a more faithful local surrogate at direct runtime cost.",
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
            interpretable_frame[col] = 0  # see make_predict_fn()'s reindex comment - same reasoning
    interpretable_frame = interpretable_frame[interpretable_cols]
    embeddings = df[embedding_cols].to_numpy(dtype=np.float64)
    print(f"  {len(df)} row(s), {len(interpretable_cols)} interpretable feature(s) (+{len(embedding_cols)} embedding dims held fixed per-instance)")

    full_frame = pd.concat(
        [interpretable_frame.reset_index(drop=True), df[embedding_cols].reset_index(drop=True)], axis=1,
    )[expected_cols]
    anomaly_score = score_anomalies(pipeline.named_steps["iforest"], pipeline.named_steps["preprocessor"].transform(full_frame))

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # --- approximate "global" importance: see module docstring, this is
    # an average over ONLY the explained instances above, not the whole
    # dataset - named _approx to keep that honest at a glance.
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

    # Log back to the SAME run, not a new one - ties this explanation to
    # the specific model version it was computed from (same convention as
    # models/rule_pattern/explain.py).
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(local_path))
        mlflow.log_artifact(str(importance_path))
        if plot_path is not None:
            mlflow.log_artifact(str(plot_path))
    print(f"\nLogged LIME explanation artifacts back to run {run_id}")


if __name__ == "__main__":
    main()
