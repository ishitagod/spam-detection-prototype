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
    python -m models.anomaly.cluster_discovery                       # HDBSCAN, content-only (default)
    python -m models.anomaly.cluster_discovery --cluster_features all  # experiment: blend in behavior
    python -m models.anomaly.cluster_discovery --algorithm dbscan --eps 2.1  # old DBSCAN path, kept for comparison
    python -m models.anomaly.cluster_discovery --algorithm dbscan --eps_auto

SCALE WARNING: at full (non-sampled) dataset scale, --anomaly_percentile 90
selects hundreds of thousands of rows - DBSCAN over that many rows crashed
the machine once (see MAX_RECOMMENDED_CANDIDATES). `run()` refuses above
that count without --force; this is why the default is now 99.5, not 90.

EPS DEFAULT IS A FIXED CONSTANT, NOT AUTO-SUGGESTED, ON PURPOSE: two
different people running this with no flags used to get two different
outcomes depending only on the machine's k-NN distance distribution that
day, and the auto-suggested value at this project's current full-corpus
scale is measurably bad - on SS7's --anomaly_percentile 99.5 pool (13,712
candidates), suggest_eps() returned 4.817, which DBSCAN turned into one
cluster holding 84% of the pool (the exact "eps too large, one giant
cluster" failure docs/experiments/anomaly_clustering.md warns about) -
useless for hand-labeling. DEFAULT_EPS below (0.6) was found by sweeping
eps on that same pool until no single cluster dominated - see that same
doc for the sweep. Pass --eps to override, or --eps_auto to fall back to
the old auto-suggested-from-k-distance behavior (suggest_eps()) if you're
running against a materially different candidate pool size/shape where
0.6 may not apply - eps is genuinely pool-size-dependent, so re-validate
rather than assuming 0.6 travels to every --anomaly_percentile/source
combination.
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

from config.settings import MLFLOW_TRACKING_URI
from models.anomaly.data import BEHAVIORAL_COLS, NEAR_DUP_COLS, build_feature_matrix, load_source_features

# Separate from "anomaly_score" - this can't be a promotable model (see
# module docstring), so it gets its own MLflow experiment name.
MLFLOW_EXPERIMENT_NAME = "fraud_type_cluster_discovery"

# Subset of NEAR_DUP_COLS actually safe for clustering "content" mode -
# near_dup_distinct_senders_1hr/24hr are SENDER-IDENTITY-derived (how many
# DIFFERENT originators sent near-duplicate text), not content similarity,
# despite living in NEAR_DUP_COLS. Measured on this project's real data: a
# multi-sender campaign (WhatsApp-invite lure, sent by many rotating
# senders) has near_dup_distinct_senders_24hr ranging 0-34 (std 4.86)
# WITHIN that one campaign, vs. a fixed-2-sender campaign (RM69 gambling
# spam) sitting flat at 0-2 (std 0.09) - i.e. this column varies a lot
# based purely on HOW MANY senders happened to be running a campaign, the
# same "behavioral variance splits identical content" risk documented for
# BEHAVIORAL_COLS above, just hiding inside NEAR_DUP_COLS instead.
# near_dup_match_count_*/near_dup_max_similarity_* stay in - genuinely
# content-recurrence/embedding-similarity, not sender-identity.
CONTENT_SAFE_NEAR_DUP_COLS = [
    "near_dup_match_count_1hr", "near_dup_max_similarity_1hr",
    "near_dup_match_count_24hr", "near_dup_max_similarity_24hr",
]

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

# Fixed default, not auto-suggested - see module docstring's "EPS DEFAULT
# IS A FIXED CONSTANT" section. Validated against SS7's
# --anomaly_percentile 99.5 pool (13,712 candidates) only; re-validate
# (sweep a few values, watch for one cluster dominating the pool) before
# trusting this on a materially different pool size/source/percentile.
DEFAULT_EPS = 0.6

# Paired with DEFAULT_EPS above - the old default (90) selects hundreds of
# thousands of rows at full-corpus scale and is what crashed the machine
# (see MAX_RECOMMENDED_CANDIDATES); 99.5 is the tighter value this project
# has actually been running with.
DEFAULT_ANOMALY_PERCENTILE = 99.5


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


def select_clustering_features(
    X: np.ndarray, feature_names: list[str], mode: str = "content",
) -> np.ndarray:
    """Selects which columns the clustering distance metric sees.

    mode="content" (default, validated): content-similarity columns only
    (embedding PCA + CONTENT_SAFE_NEAR_DUP_COLS) - see module docstring
    for the 109-message-template-split-across-13-clusters failure this
    avoids. Uses CONTENT_SAFE_NEAR_DUP_COLS, NOT the full NEAR_DUP_COLS -
    near_dup_distinct_senders_1hr/24hr are sender-identity-derived, not
    content similarity, and measurably vary within a single real campaign
    depending on how many senders it happens to use - see that constant's
    comment for the numbers. Doesn't change the PCA fit itself, only which
    columns the distance metric sees; behavioral AND sender-count columns
    stay available in `df_subset` for summarize_clusters()'s reporting
    either way.

    mode="all": everything build_feature_matrix() built, behavioral
    columns included - an explicit, opt-in EXPERIMENT to see what
    combining content+behavior into one distance metric actually does to
    known campaigns (e.g. does the RM69/REG-REQ/WhatsApp-invite clusters
    found under mode="content" fragment when behavioral distance is
    blended in), not a recommended default - see module docstring."""
    if mode == "all":
        return X
    if mode != "content":
        raise ValueError(f"Unknown cluster_features mode {mode!r} - expected 'content' or 'all'.")
    keep = [
        i for i, name in enumerate(feature_names)
        if name.startswith("emb_pca_") or name in CONTENT_SAFE_NEAR_DUP_COLS
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


def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int, n_jobs: int) -> np.ndarray:
    """No eps to pick - HDBSCAN builds a cluster hierarchy across a RANGE
    of density thresholds and extracts whichever groupings are stable
    across the widest range, instead of one global flat density cutoff.
    Handles uneven density (a 492-row near-exact-duplicate burst and a
    looser, more-varied templated campaign both getting correctly found
    at their own natural density) in a way one fixed --eps structurally
    can't - see module docstring. `min_cluster_size` replaces --eps as
    the main knob: the smallest group size worth calling a cluster at
    all (below that, points fall to noise, label -1, same convention as
    DBSCAN). sklearn.cluster.HDBSCAN, not the standalone hdbscan package -
    already available in this project's installed sklearn (>=1.3), no
    new dependency."""
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=n_jobs)
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
    algorithm: str = "hdbscan",
    cluster_features: str = "content",
    min_cluster_size: int = 5,
) -> None:
    print(f"Loading features + anomaly scores for sources: {sources} ...")
    df, X, feature_names = load_features_and_scores(sources, data_dir)

    print(f"Selecting top {100 - anomaly_percentile:.0f}% by anomaly_score ...")
    df_subset, X_subset = select_anomalous_subset(df, X, anomaly_percentile)
    X_subset = select_clustering_features(X_subset, feature_names, mode=cluster_features)
    print(f"  clustering on {X_subset.shape[1]} column(s), cluster_features={cluster_features!r} "
          f"(algorithm={algorithm!r}) - see select_clustering_features()")
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

    resolved_eps = None
    if algorithm == "dbscan":
        resolved_eps = eps if eps is not None else suggest_eps(X_subset, min_samples)
        print(f"Running DBSCAN (eps={resolved_eps:.4f}, min_samples={min_samples}, n_jobs={n_jobs}) on {len(X_subset)} rows ...")
        labels = run_dbscan(X_subset, resolved_eps, min_samples, n_jobs)
    elif algorithm == "hdbscan":
        print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
              f"n_jobs={n_jobs}) on {len(X_subset)} rows ...")
        labels = run_hdbscan(X_subset, min_cluster_size, min_samples, n_jobs)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r} - expected 'dbscan' or 'hdbscan'.")
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
            "algorithm": algorithm,
            "cluster_features": cluster_features,
            "eps": resolved_eps,
            "eps_was_auto_suggested": eps is None if algorithm == "dbscan" else None,
            "min_cluster_size": min_cluster_size if algorithm == "hdbscan" else None,
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
        "--anomaly_percentile", type=float, default=DEFAULT_ANOMALY_PERCENTILE,
        help="Only cluster rows at/above this anomaly_score percentile "
        f"(default: top {100 - DEFAULT_ANOMALY_PERCENTILE:.1f}%%, tighter than the "
        "old 90 default - see module docstring's SCALE WARNING).",
    )
    parser.add_argument(
        "--eps", type=float, default=DEFAULT_EPS,
        help=f"DBSCAN eps. Default: {DEFAULT_EPS} (fixed, validated constant - see "
        "module docstring's EPS DEFAULT section), not auto-suggested, so everyone "
        "running with no flags gets the same, known-good clustering shape. Pass "
        "--eps_auto instead to fall back to suggest_eps()'s k-distance heuristic.",
    )
    parser.add_argument(
        "--eps_auto", action="store_true",
        help="Ignore --eps/DEFAULT_EPS and re-suggest eps from this run's own "
        "k-distance distribution (suggest_eps()) - use when clustering a "
        "materially different candidate pool (different --anomaly_percentile, "
        "source, or corpus size) where DEFAULT_EPS hasn't been validated.",
    )
    parser.add_argument(
        "--algorithm", type=str, default="hdbscan", choices=["hdbscan", "dbscan"],
        help="Clustering algorithm. Default hdbscan (no --eps to tune, handles "
        "uneven density natively - see run_hdbscan()). --dbscan kept for "
        "comparison/rollback, uses --eps/--eps_auto.",
    )
    parser.add_argument(
        "--cluster_features", type=str, default="content", choices=["content", "all"],
        help="Which columns the clustering distance metric sees. 'content' "
        "(default, validated) = embedding PCA + near-dup only - see "
        "select_clustering_features(). 'all' includes behavioral columns too - "
        "an explicit experiment, not a recommended default, see that function's "
        "docstring for what this risks.",
    )
    parser.add_argument(
        "--min_cluster_size", type=int, default=5,
        help="HDBSCAN only: smallest group size worth calling a cluster. Ignored "
        "for --algorithm dbscan (which uses --min_samples/--eps instead).",
    )
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument(
        "--n_jobs", type=int, default=4,
        help="Parallel workers (DBSCAN or HDBSCAN). Default 4, not -1 - see run_dbscan().",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed past MAX_RECOMMENDED_CANDIDATES - see that constant's "
        "comment. Prefer a tighter --anomaly_percentile instead.",
    )
    args = parser.parse_args()
    resolved_eps = None if args.eps_auto else args.eps
    run(
        args.sources, Path(args.data_dir), args.anomaly_percentile, resolved_eps,
        args.min_samples, n_jobs=args.n_jobs, force=args.force,
        algorithm=args.algorithm, cluster_features=args.cluster_features,
        min_cluster_size=args.min_cluster_size,
    )


if __name__ == "__main__":
    main()
