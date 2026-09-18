"""
Fraud-TYPE discovery via DBSCAN, run ON TOP OF (not instead of) the
Isolation Forest anomaly layer - see docs/ml/modeling.md. Isolation
Forest gives exactly ONE number per row (how anomalous); it has no
concept of grouping similar anomalies together, so it structurally
cannot answer "anomalous in what WAY, resembling what other cases" -
that's what this script adds. Two different unsupervised jobs, not two
competing techniques for the same job.

CONNECTION TO ISOLATION FOREST, concretely: this reads
models/anomaly/train.py's ALREADY-WRITTEN `anomaly_scores.parquet` per
source rather than re-scoring - if that file is stale relative to a real
feature-set change, re-run `python -m models.anomaly.train` first, this
script does not do that for you. Rows are then filtered down to the top
`--anomaly_percentile` by that same anomaly_score (default: top 10%)
BEFORE clustering - clustering the full traffic stream would spend all
its effort characterizing normal messages, which isn't this script's
job; the whole point is characterizing what the anomaly layer already
flagged as worth a second look.

SAME PCA FIT AS ISOLATION FOREST, BUT NOT THE SAME CLUSTERING SPACE: the
feature matrix is built via models.anomaly.data.build_feature_matrix()
over the FULL candidate pool first (same PCA/scaling fit Isolation
Forest itself trained on), and only THEN sliced down to the anomalous
subset's rows - not re-fit on the subset alone. Refitting PCA on just
the anomalous rows would silently change the basis, making "the same
messages, described two different ways" instead of one consistent basis
both techniques share. That basis-consistency argument does NOT mean
DBSCAN should cluster on every column of it, though - see
select_clustering_features() below for why the actual clustering
distance metric is restricted to a CONTENT-similarity subset (embedding
PCA + near-dup columns), not the full joint feature space
(behavioral/age/diversity/velocity included) Isolation Forest scores on.
Isolation Forest genuinely needs both content and behavior - "how
anomalous is this row overall" is a real multi-factor question.
Clustering here answers a different question - "which messages are the
same campaign/template" - a content question, where sender-behavioral
differences are a property OF instances within a campaign, not a
legitimate boundary BETWEEN campaigns. Real, measured failure this
guards against: clustering on the full joint space split one confirmed
109-message spam template ("Let's chat on WhatsApp!...") across 13
different DBSCAN clusters, because its instances came from senders with
different message counts/ages/near-dup windows even though the TEXT was
identical every time - the standard industry pattern for spam/phishing-
campaign clustering (and this project's own FAISS near-dup search,
features/faiss_index.py) is to cluster on content similarity alone and
treat behavioral/sender features as a separate enrichment/filter layer,
never blended into the same distance metric that defines cluster
membership.

NOT A DEPLOYABLE MODEL: scikit-learn's DBSCAN has no .predict() for
genuinely new data at all - unlike IsolationForest/LightGBM, nothing
here generalizes to the next incoming message, so no model artifact is
logged to MLflow the way train.py logs one (would be misleading - there
is nothing to reload and reuse). This is a periodic, OFFLINE discovery
tool: run it, hand-inspect+name the resulting clusters using
summarize_clusters()'s output, and ONLY the resulting hand-confirmed
labels later become training data for a real supervised multiclass
classifier (see CLAUDE.md's Stage A -> Stage B bootstrap note - same
principle as rule-derived labels being a bootstrap, not the final
answer). This script itself never runs at live inference.

CLUSTER LABEL -1 IS MEANINGFUL, NOT A FAILURE: DBSCAN's own convention
for "doesn't belong to any dense group" - kept as its own value rather
than forced into the nearest cluster, because a genuinely novel one-off
anomaly (not resembling any other flagged case yet) is a real, different
finding from "here are 40 near-identical flooding bursts".

Usage:
    python -m models.anomaly.cluster_discovery
    python -m models.anomaly.cluster_discovery --anomaly_percentile 95 --min_samples 8
    python -m models.anomaly.cluster_discovery --eps 2.1   # override the auto-suggested eps

SCALE WARNING - read before raising --anomaly_percentile's default pool
size: at FULL (non-sampled) dataset scale, the default --anomaly_percentile
90 selects hundreds of thousands of candidate rows - DBSCAN over that many
rows in this feature space's dimensionality is a real, observed crash (see
MAX_RECOMMENDED_CANDIDATES below), not a theoretical risk. `run()` refuses
above that many candidates without --force. Use a tighter
--anomaly_percentile (99-99.9) at full scale instead of --force.
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

# Own experiment name, separate from "anomaly_score" - see docs/ml/
# modeling.md's MLflow conventions: a variant that isn't a real
# promotable candidate (which this structurally can't be - see module
# docstring's NOT A DEPLOYABLE MODEL note) gets logged under its own
# name so it's never mistaken for one in the MLflow UI.
MLFLOW_EXPERIMENT_NAME = "fraud_type_cluster_discovery"

# Columns worth printing per cluster to actually eyeball what it IS -
# raw (pre-log1p, pre-scale) values, not the PCA/scaled X the model
# actually clustered on, because a human reading "mean=1.3, std=0.4" in
# PCA-component-space can't map that back to "this is a flooding burst"
# the way "mean sender_msgs_last_1hr=8,200" can.
SUMMARY_RAW_COLS = BEHAVIORAL_COLS + NEAR_DUP_COLS

# Documented starting point, not a proven-correct constant (same honesty
# convention as FAISS_NEAR_DUP_THRESHOLD/N_EMBEDDING_COMPONENTS) - real,
# observed failure mode above this: running the default --anomaly_percentile
# 90 against a FULL (non-sampled) 8.2M-row dataset selected 824,823
# candidate rows, and DBSCAN's neighbor search over that many rows at this
# feature-space's dimensionality (~40+) exhausted system memory badly
# enough to crash the whole machine (VS Code included), not just the
# Python process. This tool was sized against the sample-scale run
# (tens of thousands of candidates), not full-scale - see
# docs/experiments/anomaly_clustering.md. `run()` refuses to proceed past
# this many candidate rows without `--force`, on purpose: the fix is
# almost always a tighter --anomaly_percentile (this tool exists to
# characterize the EXTREME tail into a hand-reviewable number of
# clusters, not to cluster 10% of all traffic), not overriding this.
MAX_RECOMMENDED_CANDIDATES = 50_000


def suggest_eps(X: np.ndarray, min_samples: int) -> float:
    """
    Standard DBSCAN eps heuristic: for every point, its distance to its
    own min_samples-th nearest neighbor; eps set at a percentile of that
    distribution is a common starting point (approximating the "knee" of
    the sorted k-distance curve without requiring a human to eyeball a
    plot in what's meant to be a non-interactive CLI script).
    NOT calibrated against any labelled ground truth - there isn't one
    for cluster quality here - same "documented starting point, not a
    proven-correct constant" honesty as config/settings.py's
    FAISS_NEAR_DUP_THRESHOLD. Override with --eps once real clusters
    have been eyeballed and this guess looks wrong.
    """
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
    """
    Loads the SAME joined feature set models/anomaly/train.py trains on
    (models.anomaly.data.load_source_features + build_feature_matrix),
    then merges in that run's already-computed anomaly_score by
    message_key. Raises a clear error (not a silent empty join) if
    anomaly_scores.parquet is missing for a source - this script depends
    on that file existing, it does not compute it.
    """
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
    """
    Restricts DBSCAN's actual clustering input to CONTENT-similarity
    columns only - embedding PCA components (`emb_pca_*`) plus
    NEAR_DUP_COLS (themselves derived from the same MiniLM embeddings via
    FAISS similarity search) - not the full joint feature space
    build_feature_matrix() built for Isolation Forest. See module
    docstring's "SAME PCA FIT... BUT NOT THE SAME CLUSTERING SPACE"
    section for the real, measured failure this fixes and why this is
    standard practice for campaign/template clustering specifically, not
    a general "behavioral features are bad" claim.

    Does NOT change the PCA fit itself - `X`/`feature_names` are still
    build_feature_matrix()'s full-candidate-pool-fit output; this only
    slices which of its columns the clustering DISTANCE METRIC sees.
    Behavioral columns remain fully available in `df_subset` for
    summarize_clusters()'s human-readable per-cluster reporting - this
    changes cluster MEMBERSHIP, not what gets reported once clusters
    exist.
    """
    keep = [
        i for i, name in enumerate(feature_names)
        if name.startswith("emb_pca_") or name in NEAR_DUP_COLS
    ]
    return X[:, keep]


def select_anomalous_subset(
    df: pd.DataFrame, X: np.ndarray, percentile: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Threshold computed ONCE across the combined (all-sources) pool, not
    per source - matches how Isolation Forest itself was trained jointly
    across sources (CLAUDE.md's "one model to start, not two"), so the
    cutoff reflects the same shared score distribution the model actually
    produced, not two separately-recalibrated ones.
    """
    threshold = float(np.percentile(df["anomaly_score"].to_numpy(), percentile))
    mask = df["anomaly_score"].to_numpy() >= threshold
    print(
        f"  anomaly_score >= {threshold:.4f} (top {100 - percentile:.0f}%): "
        f"{mask.sum()} / {len(df)} rows selected for clustering"
    )
    return df[mask].reset_index(drop=True), X[mask]


def run_dbscan(X: np.ndarray, eps: float, min_samples: int, n_jobs: int) -> np.ndarray:
    # n_jobs is a real CLI knob, not hardcoded -1 (all cores) - see
    # MAX_RECOMMENDED_CANDIDATES's docstring: each parallel worker holds
    # its own chunk of the neighbor-search tree, so more workers means
    # more concurrent memory, not just more speed, once a run is anywhere
    # near candidate-count-heavy. -1 is still available (pass explicitly)
    # for a candidate pool small enough that this doesn't matter.
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    return model.fit_predict(X)


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    One summary block per distinct cluster label (including -1, see
    module docstring) - size, mean anomaly_score, mean of the raw
    interpretable behavioral/near-dup columns, and source breakdown.
    This is the actual input to the human hand-labeling step (Stage A ->
    Stage B) - not a formal metric, there's no ground truth to score
    cluster quality against here.
    """
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
            "this module's docstring for the real crash this guards against "
            "(DBSCAN on ~800k rows exhausted system memory badly enough to "
            "take down the whole machine, not just this process). Raise "
            "--anomaly_percentile to shrink the candidate pool (e.g. 99.5 for "
            "~top 0.5%), or pass --force to proceed anyway at your own risk."
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

    # dataset_label names which source(s) this run actually clustered
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
        # Deliberately NO mlflow.sklearn.log_model() here - see module
        # docstring's NOT A DEPLOYABLE MODEL note: there is no reusable
        # artifact to reload, logging one would misrepresent what this is.
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
        help="DBSCAN parallel workers. Default 4, not -1 (all cores) - see "
        "run_dbscan()'s docstring: more workers means more CONCURRENT memory "
        "for the neighbor search, not just more speed, once the candidate "
        "pool is anywhere near MAX_RECOMMENDED_CANDIDATES-sized.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed even if the selected candidate pool exceeds "
        "MAX_RECOMMENDED_CANDIDATES - see that constant's docstring for the "
        "real crash this normally guards against. Use a tighter "
        "--anomaly_percentile instead unless you specifically need this.",
    )
    args = parser.parse_args()
    run(
        args.sources, Path(args.data_dir), args.anomaly_percentile, args.eps,
        args.min_samples, n_jobs=args.n_jobs, force=args.force,
    )


if __name__ == "__main__":
    main()
