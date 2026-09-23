"""
Human-facing inspection tool for cluster_discovery.py's output
(`fraud_type_clusters.parquet`) - makes anomaly_clustering.md's step 4
("hand-label each cluster") runnable instead of a manual pandas join
redone by hand. Joins cluster_label/anomaly_score back to
messages_with_behavioral.csv for the text/originator a human needs to
name a cluster.

DON'T LABEL FROM SAMPLE TEXTS ALONE:
1. A small sample can look uniform while the cluster isn't -
   n_unique_texts/n_unique_originators below check the WHOLE cluster
   (n_unique_texts==1 confirms one repeated message; close to n_rows
   means no coherent pattern, whatever the sample suggested).
2. Text alone can't tell "one sender flooding the same content" from
   "many senders running the same template" - the BEHAVIORAL_COLS means
   below distinguish them: high mean sender_msgs_last_1hr with
   n_unique_originators==1 is a single-actor flood; similar text with
   n_unique_originators near n_rows and low per-sender velocity is a
   templated multi-sender campaign.

near-dup columns aren't recomputed here - not in
messages_with_behavioral.csv; check cluster_discovery.py's own printed
summary/cluster_summary.json instead.

NOT the label-ingestion step (anomaly_clustering.md step 5) - only
produces what a human needs to assign fraud-type names. The CSV it
writes (cluster_labels_template.csv, blank fraud_type_label column) is
meant to be hand-filled and become the real label source once that
ingestion path exists (doesn't yet).

Cluster IDs are per-run, not stable - always run against the
fraud_type_clusters.parquet from the SAME cluster_discovery.py run
you're hand-labeling.

Usage:
    python -m models.anomaly.inspect_clusters --source SS7
    python -m models.anomaly.inspect_clusters --source SMPP --n_samples 8
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from models.anomaly.data import BEHAVIORAL_COLS

DEFAULT_DATA_DIR = Path("data/processed")
N_SAMPLES_DEFAULT = 5


def load_cluster_messages(source: str, data_dir: Path) -> pd.DataFrame:
    """fraud_type_clusters.parquet joined back to
    messages_with_behavioral.csv's real content. message_key is
    "{source}|{record_id}", so record_id is recovered by splitting it.

    fraud_type is pulled through too - a cluster where most rows already
    carry a real rule-engine fraud_type is a strong free hint for naming
    it, even though these are mostly from the unresolved/
    never-rule-evaluated pool (anomaly_score's training scope).
    """
    source_dir = data_dir / source
    clusters_path = source_dir / "fraud_type_clusters.parquet"
    messages_path = source_dir / "messages_with_behavioral.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(
            f"{clusters_path} not found - run "
            f"`python -m models.anomaly.cluster_discovery --sources {source}` first "
            "(this tool inspects that output, it doesn't compute clusters itself)."
        )

    clusters = pd.read_parquet(clusters_path).copy()
    clusters["record_id"] = clusters["message_key"].str.split("|", n=1).str[1]

    messages = pd.read_csv(
        messages_path, low_memory=False,
        usecols=["record_id", "text", "originator", "timestamp", "fraud_type"] + BEHAVIORAL_COLS,
        dtype={"record_id": str},
    )
    merged = clusters.merge(messages, on="record_id", how="left", validate="one_to_one")
    return merged


def cluster_diagnostics(group: pd.DataFrame) -> dict:
    """Whole-cluster checks sample texts alone can't give you - see
    module docstring. Computed over every row in the cluster, not just
    the printed sample, so the CSV template alone is enough to tell a
    single-actor flood apart from a templated multi-sender campaign."""
    return {
        "n_unique_texts": int(group["text"].nunique()),
        "n_unique_originators": int(group["originator"].nunique()),
        **{col: float(group[col].mean()) for col in BEHAVIORAL_COLS},
    }


def print_cluster_samples(df: pd.DataFrame, n_samples: int) -> None:
    """Largest cluster first, noise (-1) included. Fixed random_state
    keeps the printed sample and the template file's sample_texts column
    showing the same rows."""
    sizes = df.groupby("cluster_label").size().sort_values(ascending=False)
    for cluster_id, n_rows in sizes.items():
        label = "noise (-1)" if cluster_id == -1 else f"cluster {cluster_id}"
        group = df[df["cluster_label"] == cluster_id]
        mean_score = group["anomaly_score"].mean()
        rule_hits = int(group["fraud_type"].notna().sum())
        diag = cluster_diagnostics(group)
        print(
            f"\n=== {label}: {n_rows} row(s), mean_anomaly_score={mean_score:.3f}, "
            f"{rule_hits} already rule-flagged, "
            f"{diag['n_unique_texts']} unique text(s), {diag['n_unique_originators']} unique originator(s) ==="
        )
        print("  " + ", ".join(f"{col}={diag[col]:.2f}" for col in BEHAVIORAL_COLS))
        sample = group.sample(min(n_samples, len(group)), random_state=0)
        for _, row in sample.iterrows():
            print(f"  [{row['originator']}] {row['text']!r}")


def write_labeling_template(df: pd.DataFrame, out_path: Path, n_samples: int) -> pd.DataFrame:
    """One row per cluster (noise included), enough context to name it
    without reopening the terminal output, plus a blank fraud_type_label
    column to fill in and save - see module docstring."""
    rows = []
    for cluster_id, group in df.groupby("cluster_label"):
        sample = group.sample(min(n_samples, len(group)), random_state=0)
        rows.append({
            "cluster_label": cluster_id,
            "n_rows": len(group),
            "mean_anomaly_score": group["anomaly_score"].mean(),
            "n_already_rule_flagged": int(group["fraud_type"].notna().sum()),
            **cluster_diagnostics(group),
            "sample_texts": " ||| ".join(sample["text"].dropna().astype(str)),
            "fraud_type_label": "",  # <- fill this in by hand, then save
        })
    out_df = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    out_df.to_csv(out_path, index=False)
    return out_df


def run(source: str, data_dir: Path, n_samples: int) -> None:
    df = load_cluster_messages(source, data_dir)
    print_cluster_samples(df, n_samples)

    template_path = data_dir / source / "cluster_labels_template.csv"
    out_df = write_labeling_template(df, template_path, n_samples)
    print(f"\nWrote labeling template ({len(out_df)} cluster(s), including noise) to {template_path}")
    print(
        "Open it, fill in fraud_type_label per row using the samples printed above "
        "(or the sample_texts column itself) and save - see "
        "docs/experiments/anomaly_clustering.md's step 4/5. If a cluster is "
        "confirmed NOT fraud, label it 'not_fraud' (labels/cluster_labels.py's "
        "NOT_FRAUD_LABEL) rather than leaving it blank - blank means "
        "unreviewed, 'not_fraud' means reviewed and ruled out."
    )


def main():
    # Windows' cp1252 console can't encode real multilingual/garbled SMS
    # content and crashes mid-print - replace instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--n_samples", type=int, default=N_SAMPLES_DEFAULT,
        help="Sample messages to print/store per cluster (default 5).",
    )
    args = parser.parse_args()
    run(args.source, Path(args.data_dir), args.n_samples)


if __name__ == "__main__":
    main()
