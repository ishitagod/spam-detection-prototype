"""
Turns a human-confirmed row of models/anomaly/inspect_clusters.py's
cluster_labels_template.csv into the one canonical label column this
project's cluster-discovery workflow produces: cluster_fraud_type_label.

Named cluster_labels.py to mirror labels/rule_labels.py exactly - same
"derive training labels FROM an external verdict" shape, just a different
verdict source. There are three distinct "cluster label" things in this
codebase - don't conflate them:
  1. models/anomaly/cluster_discovery.py's `cluster_label` column (a raw
     DBSCAN integer, including -1 for noise) - NOT a fraud-type verdict,
     just a grouping id, and NOT STABLE across reruns (see that module's
     docstring). Never treat a bare cluster_label as a label.
  2. models/anomaly/inspect_clusters.py's cluster_labels_template.csv
     `fraud_type_label` column - a per-cluster hand-typed name, BLANK by
     default, meaningful only once a human has actually reviewed the
     cluster's samples and confirmed what it is.
  3. this module - joins #1 (per-message) to #2 (per-cluster, confirmed
     only) to produce cluster_fraud_type_label, a per-message label.

cluster_fraud_type_label is deliberately NOT named `fraud_type` or
`rule_flagged` (see labels/rule_labels.py and common/schemas.py) - those
are the rule engine's labels, sourced from an upstream system with its own
confidence/provenance. A cluster-derived label comes from unsupervised
DBSCAN grouping + one human's read of a handful of sample messages -
different provenance, different confidence, must stay traceable to which
one a downstream consumer is actually using. Never merge this column into
rule_flagged, and never let a classifier trained on this pretend it was
trained on rule-engine ground truth.

Rows whose cluster was never reviewed, or was reviewed and deliberately
left unlabeled (an incoherent DBSCAN merge artifact, or one a human wasn't
confident enough to name), simply do not appear in this module's output -
this module never fabricates a label for a message whose cluster identity
nobody actually confirmed.
"""
import pandas as pd

# The one reserved fraud_type_label value meaning "a human reviewed this
# anomalous cluster and confirmed it's NOT fraud" (e.g. a legitimate
# bulk-OTP sender that just looks unusual). Distinct from a blank label
# (not reviewed / not confident) - see is_cluster_confirmed(). Case/
# whitespace-insensitive on comparison; models/fraud_type_classifier/
# data.py excludes rows carrying this label from classifier training
# (fraud_type_classifier answers "what kind of fraud", not "is this
# fraud" - see that module's docstring).
NOT_FRAUD_LABEL = "not_fraud"


def is_cluster_confirmed(template_df: pd.DataFrame, label_col: str = "fraud_type_label") -> pd.Series:
    """
    True where a human actually typed a real name into `label_col` for
    that cluster row - False means either "not yet reviewed" or
    "reviewed, deliberately left blank" (an incoherent cluster, or one the
    reviewer wasn't confident about) - both are "not a real label" here,
    same as is_rule_evaluated()==False covers both "untouched" and
    "whitelist-only" as one non-trustworthy bucket in labels/rule_labels.py.

    Handles NaN AND empty/whitespace-only strings as "not confirmed" - a
    human leaving a spreadsheet cell empty can produce either depending on
    the tool (pandas reads a truly empty CSV cell as NaN, but a cell with
    just spaces, or a value later cleared in Excel/Sheets, can come back as
    "" or "   " instead) - both must be treated the same, or a
    whitespace-only cell would silently sneak through as a "confirmed"
    empty-string label.
    """
    stripped = template_df[label_col].astype("string").str.strip()
    return stripped.notna() & (stripped != "")


def build_cluster_labels(clusters_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    The actual join: filters template_df down to confirmed rows only (via
    is_cluster_confirmed), then inner-joins clusters_df (message_key,
    cluster_label) onto that confirmed subset on cluster_label. Returns
    message_key, cluster_label, cluster_fraud_type_label - one row per
    message whose cluster was hand-confirmed. A message whose cluster was
    never confirmed (or confirmed-blank) simply doesn't appear here - never
    fabricated.

    cluster_label -1 (DBSCAN noise) is not special-cased - if a human
    reviewed the noise bucket's samples and decided it's actually a real,
    nameable pattern (rare, but not impossible - see
    models/anomaly/cluster_discovery.py's "CLUSTER LABEL -1 IS MEANINGFUL"
    note), it's confirmed and joined exactly like any other cluster_label
    value.

    MISMATCHED-RUN GUARD: cluster_label values are only meaningful within
    the SAME cluster_discovery.py run - they are NOT stable across reruns
    (see that module's docstring). A stale cluster_labels_template.csv
    (hand-labeled against an earlier run) joined against a freshly
    regenerated fraud_type_clusters.parquet would silently produce
    garbage: cluster 3's confirmed name from the old run could land on
    completely different messages that happen to share cluster_label==3 in
    the new run. Rather than let that fail silently (inner join just drops
    non-matches, no error), this checks that every CONFIRMED cluster_label
    in template_df is actually present in clusters_df - if a confirmed
    label refers to a cluster_label clusters_df has never heard of, the
    two inputs almost certainly come from different runs (or a hand-edited
    template with a typo'd cluster_label), and this raises rather than
    quietly dropping those confirmed rows out of the join. This is
    intentionally NOT symmetric: clusters_df is allowed to contain
    cluster_label values with no confirmed template row (that's just an
    unreviewed or deliberately-unlabeled cluster, the normal/expected
    case), it's only "template confirms a cluster_label clusters_df
    doesn't have" that signals a real universe mismatch.
    """
    confirmed = template_df[is_cluster_confirmed(template_df)]
    confirmed_cluster_ids = set(confirmed["cluster_label"])
    known_cluster_ids = set(clusters_df["cluster_label"])
    unknown = confirmed_cluster_ids - known_cluster_ids
    if unknown:
        raise ValueError(
            f"template_df confirms cluster_label(s) {sorted(unknown, key=str)} "
            "that don't exist in clusters_df - template_df and clusters_df "
            "look like they're from DIFFERENT cluster_discovery.py runs "
            "(cluster_label values aren't stable across reruns - see that "
            "module's docstring). Re-run models/anomaly/inspect_clusters.py "
            "against the SAME fraud_type_clusters.parquet you're labeling, "
            "or re-label against the current one, before ingesting."
        )

    joined = clusters_df.merge(
        confirmed[["cluster_label", "fraud_type_label"]],
        on="cluster_label",
        how="inner",
    )
    joined = joined.rename(columns={"fraud_type_label": "cluster_fraud_type_label"})
    return joined[["message_key", "cluster_label", "cluster_fraud_type_label"]]
