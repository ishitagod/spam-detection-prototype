"""
Human-facing inspection tool for models/anomaly/cluster_discovery.py's
output (`fraud_type_clusters.parquet`) - makes
docs/experiments/anomaly_clustering.md's step 4 ("hand-label each cluster")
concrete and runnable, instead of a manual pandas join redone by hand every
time. Joins cluster_label/anomaly_score back to messages_with_behavioral.csv
for the actual text/originator a human needs to SEE to name a cluster.

DON'T LABEL FROM SAMPLE TEXTS ALONE - two real failure modes that a
handful of sample rows can't catch:
1. The sample can look uniform while the cluster isn't - N_SAMPLES rows
   out of a cluster that might be thousands is not a promise the rest
   look the same. `n_unique_texts`/`n_unique_originators` below are the
   whole-cluster diversity check: a cluster with n_unique_texts==1 really
   is one exact repeated message; a cluster where n_unique_texts is close
   to n_rows is NOT one coherent pattern, whatever the 5 samples suggested.
2. Text alone can't tell "one sender flooding the same content" apart
   from "many senders running the same template" - same-looking text,
   different fraud shape, and a different real-world response. That's
   what the behavioral column means below are for (imported from
   models.anomaly.data.BEHAVIORAL_COLS - the same columns
   cluster_discovery.py's own summarize_clusters() prints, just not
   previously surfaced here too). A high mean sender_msgs_last_1hr with
   n_unique_originators==1 is a single-actor flood; a similar text
   pattern with n_unique_originators close to n_rows and LOW per-sender
   velocity is a templated multi-sender campaign instead - name these
   differently.

near-dup columns (near_dup_match_count_1hr/24hr etc.) are DELIBERATELY
NOT recomputed here - they're not in messages_with_behavioral.csv, and
cluster_discovery.py already prints/logs them per cluster (its own
printed summary, and cluster_summary.json on the matching MLflow run) -
check there rather than duplicating that join.

NOT the label-INGESTION step - see anomaly_clustering.md's step 5. This
tool only produces what a human needs to assign fraud-type names; it does
not feed anything back into training. The CSV it writes
(cluster_labels_template.csv, one row per cluster with a blank
fraud_type_label column) is meant to be hand-filled-in and become the real
label source once that ingestion path exists - it doesn't exist yet
(no labels/cluster_labels.py counterpart to labels/rule_labels.py).

CLUSTER IDS ARE PER-RUN, NOT STABLE (see cluster_discovery.py's docstring)
- always run this against the fraud_type_clusters.parquet produced by the
SAME cluster_discovery.py run you're currently hand-labeling, never a
stale one from an earlier run with a different --eps/--anomaly_percentile.

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
    """
    cluster_discovery.py's fraud_type_clusters.parquet (message_key,
    anomaly_score, cluster_label) joined back to
    messages_with_behavioral.csv's real content - message_key is
    "{source}|{record_id}" (see cluster_discovery.py's df_out
    construction), so record_id is recovered by splitting it rather than
    needing a second source column here (this file is already
    single-source, unlike the combined pool cluster_discovery.py itself
    may have clustered over).

    fraud_type/rule_flagged are pulled through too, deliberately: a
    cluster where most rows already carry a REAL rule-engine fraud_type is
    a strong, free hint for naming it (e.g. "the rules already call this
    generic fraud, not spam specifically") even though these are, by
    construction, mostly from the unresolved/never-rule-evaluated pool
    (anomaly_score's training scope - see docs/ml/modeling.md).
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
    """The whole-cluster checks sample texts alone can't give you - see
    module docstring. n_unique_texts/n_unique_originators are computed
    over EVERY row in the cluster, not just the printed/template sample -
    that's the point: a low n_unique_texts confirms "one repeated
    message" even when the sample happened to only show 2 of them: a
    n_unique_texts close to n_rows means the sample's apparent uniformity
    doesn't generalize to the whole cluster. Behavioral means are the
    same BEHAVIORAL_COLS cluster_discovery.py's own summarize_clusters()
    already prints - surfaced here too so the CSV template alone (without
    reopening that run's terminal output) is still enough to tell a
    single-actor flood apart from a templated multi-sender campaign."""
    return {
        "n_unique_texts": int(group["text"].nunique()),
        "n_unique_originators": int(group["originator"].nunique()),
        **{col: float(group[col].mean()) for col in BEHAVIORAL_COLS},
    }


def print_cluster_samples(df: pd.DataFrame, n_samples: int) -> None:
    """Largest cluster first (matches cluster_discovery.py's own
    print_cluster_summary() ordering) - noise (-1) included, same "real
    finding, not a failure" treatment as everywhere else this label
    appears. A fixed random_state keeps the printed sample and the
    template file's sample_texts column showing the SAME rows."""
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
    """One row per cluster (noise included) - enough context (size, mean
    anomaly_score, rule-flagged count, a handful of real sample texts) to
    name it WITHOUT reopening the terminal output, plus a blank
    fraud_type_label column for a human to fill in and save. This file
    itself becomes the hand-confirmed label source once step 5's ingestion
    path is built - see module docstring."""
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
        "docs/experiments/anomaly_clustering.md's step 4/5."
    )


def main():
    # Real SMS content includes multilingual text and (on text_decode_failed
    # rows) genuinely garbled bytes - Windows' default console codepage
    # (cp1252) can't encode a lot of that and crashes mid-print rather than
    # just showing it oddly. errors="replace" swaps the unprintable
    # character for a placeholder instead of raising - never crash a
    # hand-labeling session over a display quirk.
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
