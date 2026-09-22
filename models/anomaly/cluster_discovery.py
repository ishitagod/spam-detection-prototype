"""
Fraud-TYPE discovery via DBSCAN, run on top of (not instead of) the
Isolation Forest anomaly layer - see docs/ml/modeling.md. Isolation
Forest gives one anomaly score per row; this groups similar anomalies
together to answer "anomalous in what way, resembling what other cases".

- Reads train.py's already-written `anomaly_scores.parquet` (doesn't
  re-score) and clusters only the top --anomaly_percentile (default 10%).
- Uses the SAME PCA fit as Isolation Forest (fit on the full candidate
  pool, then sliced), but clusters only on content-similarity columns
  (embedding PCA + near-dup), not the full joint feature space - see
  select_clustering_features(). Clustering on behavioral columns too
  once split one confirmed 109-message spam template across 13 clusters,
  since instances came from senders with different behavioral stats
  despite identical text.
- NOT deployable: DBSCAN has no .predict() for new data, so no model is
  logged to MLflow. Offline discovery only - hand-inspect/name clusters,
  and only hand-confirmed labels become training data (CLAUDE.md's
  Stage A -> Stage B bootstrap).
- Cluster label -1 (DBSCAN's "noise") is kept as-is, not forced into a
  cluster - a genuine one-off anomaly is a different finding from a
  cluster of near-identical bursts.

Usage:
    python -m models.anomaly.cluster_discovery
    python -m models.anomaly.cluster_discovery --anomaly_percentile 95 --min_samples 8
    python -m models.anomaly.cluster_discovery --eps 2.1   # override the auto-suggested eps

SCALE WARNING: at full (non-sampled) dataset scale, default
--anomaly_percentile 90 selects hundreds of thousands of rows - DBSCAN
over that many rows crashed the machine once (see
MAX_RECOMMENDED_CANDIDATES). `run()` refuses above that count without
--force; use a tighter --anomaly_percentile (99-99.9) instead.
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config.settings import MLFLOW_TRACKING_URI
from models.anomaly.data import BEHAVIORAL_COLS, NEAR_DUP_COLS, build_feature_matrix, load_source_features

# Separate from "anomaly_score" - this can't be a promotable model (see
# module docstring), so it gets its own MLflow experiment name.
MLFLOW_EXPERIMENT_NAME = "fraud_type_cluster_discovery"

# Raw (pre-PCA/scale) columns to print per cluster - human-readable
# ("mean sender_msgs_last_1hr=8,200") vs. PCA-component values a human
# can't map back to what the cluster actually is.
SUMMARY_RAW_COLS = BEHAVIORAL_COLS + NEAR_DUP_COLS

# Documented starting point, not tuned. Default --anomaly_percentile 90
# on the full 8.2M-row dataset once selected 824,823 candidates, and
# DBSCAN's neighbor search at that count/dimensionality crashed the
# machine (not just the process) - see docs/experiments/anomaly_clustering.md.
# `run()` refuses above this without --force; use a tighter
# --anomaly_percentile instead.
MAX_RECOMMENDED_CANDIDATES = 50_000


def suggest_eps(X: np.ndarray, min_samples: int) -> float:
    """Standard DBSCAN eps heuristic: 90th percentile of each point's
    distance to its min_samples-th nearest neighbor (approximates the
    k-distance curve's "knee"). Not calibrated against any ground truth -
    override with --eps once real clusters look wrong."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    suggested = float(np.percentile(k_distances, 90))
    print(
        f"  --eps not given - suggesting {suggested:.3f} (90th percentile of "
        f"{min_samples}-NN distances over the {len(X)} candidate rows). "
        "This is a heuristic starting point, not a tuned value - inspect "
        "the clusters it produces and override --eps if they look wrong "
        "(too many tiny clusters -> eps too small; one giant cluster -> "
        "eps too large)."
    )
    return suggested


def load_features_and_scores(sources: list[str], data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Loads the same joined feature set train.py trains on, then merges
    in that run's anomaly_score by message_key. Raises clearly if
    anomaly_scores.parquet is missing - this script consumes it, doesn't
    compute it."""
    frames = []
    for source in sources:
        source_dir = data_dir / source
        messages_path = source_dir / "messages_with_behavioral.csv"
        scores_path = source_dir / "anomaly_scores.parquet"
        if not scores_path.exists():
            raise FileNotFoundError(
                f"{scores_path} not found - run `python -m models.anomaly.train` "
                f"for {source} first (see module docstring: this script consumes "
                "that output, it doesn't compute anomaly scores itself)."
            )
        df = load_source_features(source_dir, messages_path)
        scores = pd.read_parquet(scores_path)[["message_key", "anomaly_score"]]
        df = df.merge(scores, on="message_key", how="inner", validate="one_to_one")
        print(f"  {source}: {len(df)} message(s) with both features and an anomaly_score")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    X, feature_names, _ = build_feature_matrix(combined)
    return combined, X, feature_names


def select_clustering_features(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Restricts DBSCAN's clustering input to content-similarity columns
    only (embedding PCA + NEAR_DUP_COLS) - see module docstring for the
    109-message-template-split-across-13-clusters failure this avoids.
    Doesn't change the PCA fit itself, only which columns the distance
    metric sees; behavioral columns stay available in `df_subset` for
    summarize_clusters()'s reporting."""
    keep = [
        i for i, name in enumerate(feature_names)
        if name.startswith("emb_pca_") or name in NEAR_DUP_COLS
    ]
    return X[:, keep]


def select_anomalous_subset(
    df: pd.DataFrame, X: np.ndarray, percentile: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Threshold computed once across the combined (all-sources) pool,
    not per source - matches Isolation Forest's joint-source training."""
    threshold = float(np.percentile(df["anomaly_score"].to_numpy(), percentile))
    mask = df["anomaly_score"].to_numpy() >= threshold
    print(
        f"  anomaly_score >= {threshold:.4f} (top {100 - percentile:.0f}%): "
        f"{mask.sum()} / {len(df)} rows selected for clustering"
    )
    return df[mask].reset_index(drop=True), X[mask]


def run_dbscan(X: np.ndarray, eps: float, min_samples: int, n_jobs: int) -> np.ndarray:
    # n_jobs defaults to 4, not -1 (all cores) - more workers means more
    # concurrent memory for the neighbor search, not just more speed,
    # once the candidate pool is large. Pass -1 explicitly if safe to.
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    return model.fit_predict(X)


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """One summary block per cluster label (including -1) - size, mean
    anomaly_score, mean raw behavioral/near-dup columns, source
    breakdown. Feeds the human hand-labeling step (Stage A -> Stage B);
    not a formal metric."""
    working = df.copy()
    working["cluster_label"] = labels
    summary = {}
    for cluster_id, group in working.groupby("cluster_label"):
        key = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        summary[key] = {
            "n_rows": int(len(group)),
            "mean_anomaly_score": float(group["anomaly_score"].mean()),
            "source_counts": group["source"].value_counts().to_dict(),
            **{col: float(group[col].mean()) for col in SUMMARY_RAW_COLS},
        }
    return summary


def print_cluster_summary(summary: dict) -> None:
    for name, stats in sorted(summary.items(), key=lambda kv: -kv[1]["n_rows"]):
        print(f"  {name}: {stats['n_rows']} row(s), mean_anomaly_score={stats['mean_anomaly_score']:.3f}, "
              f"sources={stats['source_counts']}")
        for col in SUMMARY_RAW_COLS:
            print(f"      {col}: {stats[col]:.3f}")


def run(
    sources: list[str],
    data_dir: Path,
    anomaly_percentile: float,
    eps: float | None,
    min_samples: int,
    n_jobs: int = 4,
    force: bool = False,
) -> None:
    print(f"Loading features + anomaly scores for sources: {sources} ...")
    df, X, feature_names = load_features_and_scores(sources, data_dir)

    print(f"Selecting top {100 - anomaly_percentile:.0f}% by anomaly_score ...")
    df_subset, X_subset = select_anomalous_subset(df, X, anomaly_percentile)
    X_subset = select_clustering_features(X_subset, feature_names)
    print(f"  clustering on {X_subset.shape[1]} content-similarity column(s) "
          f"(embedding PCA + near-dup) - see select_clustering_features()")
    if len(df_subset) < min_samples:
        raise ValueError(
            f"Only {len(df_subset)} row(s) selected, fewer than --min_samples "
            f"({min_samples}) - DBSCAN can't form any cluster from this few "
            "points. Raise the candidate pool (lower --anomaly_percentile) or "
            "lower --min_samples."
        )
    if len(df_subset) > MAX_RECOMMENDED_CANDIDATES and not force:
        raise ValueError(
            f"{len(df_subset)} candidate rows selected, above "
            f"MAX_RECOMMENDED_CANDIDATES ({MAX_RECOMMENDED_CANDIDATES}) - see "
            "module docstring for the crash this guards against. Raise "
            "--anomaly_percentile to shrink the pool (e.g. 99.5 for ~top "
            "0.5%), or pass --force to proceed anyway."
        )

    resolved_eps = eps if eps is not None else suggest_eps(X_subset, min_samples)

    print(f"Running DBSCAN (eps={resolved_eps:.4f}, min_samples={min_samples}, n_jobs={n_jobs}) on {len(X_subset)} rows ...")
    labels = run_dbscan(X_subset, resolved_eps, min_samples, n_jobs)
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  {n_clusters} cluster(s) found, {n_noise} row(s) unclustered (noise, label -1)")

    summary = summarize_clusters(df_subset, labels)
    print("Cluster summary (see module docstring - this feeds human hand-labeling, not a formal metric):")
    print_cluster_summary(summary)

    dataset_label = "+".join(sorted(sources))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_input(
            mlflow.data.from_pandas(
                df_subset,
                source=",".join(str(data_dir / s / "anomaly_scores.parquet") for s in sources),
                name=dataset_label,
            ),
            context="clustering",
        )
        mlflow.log_params({
            "dataset": dataset_label,
            "anomaly_percentile": anomaly_percentile,
            "n_candidate_rows": len(df_subset),
            "eps": resolved_eps,
            "eps_was_auto_suggested": eps is None,
            "min_samples": min_samples,
            "n_jobs": n_jobs,
        })
        mlflow.log_metrics({"n_clusters": n_clusters, "n_noise": n_noise})
        mlflow.log_dict(summary, "cluster_summary.json")
        # No mlflow.sklearn.log_model() - not a deployable model, see module docstring.
        print(f"Logged run to MLflow (tracking_uri={MLFLOW_TRACKING_URI}, experiment={MLFLOW_EXPERIMENT_NAME})")

    df_out = pd.DataFrame({
        "message_key": df_subset["source"] + "|" + df_subset["record_id"],
        "source": df_subset["source"],
        "anomaly_score": df_subset["anomaly_score"],
        "cluster_label": labels,
    })
    for source in sources:
        out_path = data_dir / source / "fraud_type_clusters.parquet"
        subset = df_out[df_out["source"] == source].drop(columns="source")
        subset.to_parquet(out_path, index=False)
        print(f"Wrote {len(subset)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, nargs="+", default=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--anomaly_percentile", type=float, default=90.0,
        help="Only cluster rows at/above this anomaly_score percentile "
        "(default: top 10%%). Not calibrated - see module docstring.",
    )
    parser.add_argument(
        "--eps", type=float, default=None,
        help="DBSCAN eps. Default: auto-suggested from k-distance (see suggest_eps()).",
    )
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument(
        "--n_jobs", type=int, default=4,
        help="DBSCAN parallel workers. Default 4, not -1 - see run_dbscan().",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed past MAX_RECOMMENDED_CANDIDATES - see that constant's "
        "comment. Prefer a tighter --anomaly_percentile instead.",
    )
    args = parser.parse_args()
    run(
        args.sources, Path(args.data_dir), args.anomaly_percentile, args.eps,
        args.min_samples, n_jobs=args.n_jobs, force=args.force,
    )


if __name__ == "__main__":
    main()
