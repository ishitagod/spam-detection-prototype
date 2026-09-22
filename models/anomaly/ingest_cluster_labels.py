"""
The label-INGESTION step for the cluster-discovery workflow
(anomaly_clustering.md step 5). Reads a source's fraud_type_clusters.parquet
(cluster_discovery.py's output) and cluster_labels_template.csv
(inspect_clusters.py's output, hand-filled by a human), joins them via
labels.cluster_labels.build_cluster_labels(), and accumulates the result
into a durable, appendable cluster_labels.parquet - the training-label
source for confirmed cluster-derived labels.

Join logic itself lives in labels/cluster_labels.py (pure, no file I/O) -
this script is just the CLI/file-I/O shell around it.

WHY ACCUMULATE, NOT OVERWRITE: cluster_label values are per-run, not
stable, so a human reruns cluster_discovery -> inspect_clusters -> this
script repeatedly over time against freshly-renumbered batches. Each
run's confirmed labels must ADD to the pool, not replace it - see
accumulate_labels() for the dedup rule.

Usage:
    python -m models.anomaly.ingest_cluster_labels --source SS7
    python -m models.anomaly.ingest_cluster_labels --source SMPP --data_dir data/processed
"""
import argparse
from pathlib import Path

import pandas as pd

from labels.cluster_labels import build_cluster_labels

DEFAULT_DATA_DIR = Path("data/processed")


def load_inputs(source: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the two per-source input files; raises a clear error naming
    the upstream command to run if either is missing."""
    source_dir = data_dir / source
    clusters_path = source_dir / "fraud_type_clusters.parquet"
    template_path = source_dir / "cluster_labels_template.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(
            f"{clusters_path} not found - run "
            f"`python -m models.anomaly.cluster_discovery --sources {source}` first."
        )
    if not template_path.exists():
        raise FileNotFoundError(
            f"{template_path} not found - run "
            f"`python -m models.anomaly.inspect_clusters --source {source}` first, "
            "then hand-fill its fraud_type_label column before running this script."
        )
    clusters_df = pd.read_parquet(clusters_path)[["message_key", "cluster_label"]]
    template_df = pd.read_csv(template_path)
    return clusters_df, template_df


def accumulate_labels(new_labels: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Merges this run's confirmed labels into whatever's accumulated at
    out_path, de-duplicated on message_key.

    KEEP-NEWEST: if a message_key is confirmed in both the existing file
    and this run, this run's label wins - a later hand-review is assumed
    more current than an older one. Means a label can change across runs
    if a human changes their mind - intended, this file is a living
    "current best label per message" view, not an append-only log.

    Returns (full accumulated frame, count of message_keys newly added
    this run - distinct from rows reconfirmed/changed).
    """
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        n_new = len(set(new_labels["message_key"]) - set(existing["message_key"]))
        combined = pd.concat([existing, new_labels], ignore_index=True)
        # keep="last": new_labels concatenated after existing, so ties go to this run (KEEP-NEWEST).
        combined = combined.drop_duplicates(subset="message_key", keep="last").reset_index(drop=True)
    else:
        n_new = new_labels["message_key"].nunique()
        combined = new_labels.drop_duplicates(subset="message_key", keep="last").reset_index(drop=True)
    return combined, n_new


def run(source: str, data_dir: Path) -> None:
    clusters_df, template_df = load_inputs(source, data_dir)
    new_labels = build_cluster_labels(clusters_df, template_df)
    print(
        f"{len(new_labels)} message(s) across {new_labels['cluster_label'].nunique()} "
        f"confirmed cluster(s) this run."
    )

    out_path = data_dir / source / "cluster_labels.parquet"
    combined, n_new = accumulate_labels(new_labels, out_path)
    combined.to_parquet(out_path, index=False)
    print(
        f"{n_new} NEW label(s) added this run; {len(combined)} total confirmed "
        f"cluster label(s) now accumulated at {out_path}"
    )
    if len(new_labels) == 0:
        print(
            "0 confirmed clusters found in the template - this is expected if "
            "nobody has filled in fraud_type_label yet (see "
            "docs/experiments/anomaly_clustering.md's step 4)."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    run(args.source, Path(args.data_dir))


if __name__ == "__main__":
    main()
