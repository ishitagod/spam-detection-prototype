"""
EXPERIMENTAL rule-based cluster-label SUGGESTIONS - a testable heuristic
to speed up hand-labeling (docs/experiments/anomaly_clustering.md's
step 4), NOT a replacement for it.

HARD CONSTRAINT, do not weaken this: nothing here ever writes to
`fraud_type_label` (models/anomaly/inspect_clusters.py's output column) or
feeds `labels/cluster_labels.py::build_cluster_labels()` directly.
`labels/cluster_labels.py`'s own docstring is explicit that a
`cluster_label` alone is never a trustworthy verdict - DBSCAN's job is
grouping, not deciding ground truth - and this module's rules are a much
weaker signal than even that (simple keyword/shape heuristics, no model
training, no validation against real labels since none exist yet). This
writes to its OWN file (`cluster_labels_suggested.csv`, a
`suggested_fraud_type_label` column) specifically so it can never
collide with or silently overwrite a human's in-progress edits to the
real `cluster_labels_template.csv`. A human still has to read the
suggestion, check it against the real sample texts, and type the real
answer into `fraud_type_label` themselves.

BE HONEST ABOUT WHAT THIS DOES AND DOESN'T DO: keyword matching
(WhatsApp/Telegram/gambling/bank/OTP phrasing) is ENGLISH-BIASED. Real
data in this project is heavily multilingual (Malay, Indonesian, German,
Bengali all observed in real top-anomaly clusters this session) - a
non-English cluster will usually fall through to a behavioral-shape-only
suggestion (flooding_burst / templated_multi_sender_campaign /
unclassified_review_needed), not a content-based one. This is a real,
expected limitation, not a bug to silently paper over - print how often
it happens so it's visible, not hidden in the output file.

Reads models/anomaly/inspect_clusters.py's own output
(cluster_labels_template.csv - already has n_rows, n_already_rule_flagged,
n_unique_texts, n_unique_originators, sample_texts per cluster, no need
to recompute any of that) - never reads fraud_type_clusters.parquet or
messages_with_behavioral.csv directly.

Usage:
    python -m models.anomaly.suggest_cluster_labels --source SS7
    python -m models.anomaly.suggest_cluster_labels --source SMPP
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/processed")

# Fraction of a cluster's rows that must be unique text for it to be
# treated as "not a real coherent pattern" rather than mis-set eps
# merging heterogeneous content together - see
# docs/experiments/anomaly_clustering.md's step 3 "eps too large" note.
# A starting point, not tuned/validated against real hand-labels (none
# exist yet) - same "documented, not proven" status as this project's
# other threshold constants (FAISS_NEAR_DUP_THRESHOLD,
# SENDER_DIVERSITY_MIN_MSGS).
#
# BOTH text AND originator diversity must be high before calling a
# cluster incoherent - text uniqueness ALONE is not sufficient evidence.
# Real counter-example found testing this against actual SMPP output:
# a 13,112-row cluster had 99.4% unique texts (real bank/OTP alert
# templates with embedded dynamic fields - card numbers, amounts, OTP
# codes - trivially make every message's exact string unique) but only
# 65 distinct originators (0.5% of rows) - a genuinely coherent,
# sender-concentrated pattern, not incoherent noise. A cluster
# concentrated on either axis (few distinct texts OR few distinct
# senders) is a real pattern; only diverse on BOTH axes at once is
# actually incoherent.
INCOHERENT_UNIQUE_TEXT_RATIO = 0.5
INCOHERENT_UNIQUE_ORIGINATOR_RATIO = 0.5

# Ordered (first match wins) case-insensitive substring rules over
# sample_texts - deliberately simple, deliberately not a real NLP
# classifier. English-biased on purpose acknowledged - see module
# docstring. Each entry: (label, [substrings]).
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("chat_app_invite_spam", ["whatsapp", "telegram", "chat with me", "join me"]),
    ("gambling_promo_spam", ["jackpot", "claim free", "bonus", "prize", " rm ", "reward"]),
    ("otp_verification_bait", ["otp", "verify your", "verification", "reg-req"]),
    ("bank_transaction_alert", ["bank", "debit card", "withdraw", "account", "balance"]),
    ("url_link_spam", ["http://", "https://", "www.", ".com/"]),
]


def _keyword_label(sample_texts: str) -> str | None:
    """First matching KEYWORD_RULES label, or None if nothing matched
    (the expected, common case for non-English content - see module
    docstring)."""
    text = (sample_texts or "").lower()
    for label, substrings in KEYWORD_RULES:
        if any(s in text for s in substrings):
            return label
    return None


def suggest_label(row: pd.Series) -> str:
    """
    One cluster's suggested label, from models/anomaly/inspect_clusters.py's
    own per-cluster template row (n_rows, n_unique_texts,
    n_unique_originators, sample_texts) - see module docstring for the
    full rule ordering/reasoning. Rules checked in this specific order,
    first match wins:
      1. Coherence gate - a cluster diverse on BOTH text AND originator
         at once isn't a real single pattern yet (see
         INCOHERENT_UNIQUE_TEXT_RATIO/INCOHERENT_UNIQUE_ORIGINATOR_RATIO's
         comment - text diversity alone is not sufficient, dynamic
         template fields make that a false signal), no point guessing a
         fraud type for it.
      2. Keyword content match (chat-app invite / gambling / OTP / bank /
         URL) - only reached for clusters that passed the coherence gate,
         since a content label on an incoherent cluster is meaningless.
      3. Behavioral-shape fallback when no keyword matched: single
         unique-text + single originator = one actor repeating itself
         (flooding_burst); single unique-text + multiple originators =
         the same template from different senders
         (templated_multi_sender_campaign).
      4. unclassified_review_needed - nothing above matched; still a
         coherent cluster (passed the gate), just not one this heuristic
         has a rule for. A real, expected fallback, not a failure.
    """
    n_rows = row["n_rows"]
    unique_text_ratio = row["n_unique_texts"] / n_rows if n_rows else 0.0
    unique_originator_ratio = row["n_unique_originators"] / n_rows if n_rows else 0.0
    if (
        unique_text_ratio >= INCOHERENT_UNIQUE_TEXT_RATIO
        and unique_originator_ratio >= INCOHERENT_UNIQUE_ORIGINATOR_RATIO
    ):
        return "incoherent_no_pattern"

    keyword_hit = _keyword_label(row.get("sample_texts", ""))
    if keyword_hit is not None:
        return keyword_hit

    if row["n_unique_texts"] <= 2:
        if row["n_unique_originators"] == 1:
            return "flooding_burst"
        return "templated_multi_sender_campaign"

    return "unclassified_review_needed"


def suggest_labels(template_df: pd.DataFrame) -> pd.DataFrame:
    """template_df (models/anomaly/inspect_clusters.py's
    cluster_labels_template.csv, read as-is) with one new column,
    suggested_fraud_type_label - see module docstring for why this is a
    SEPARATE column/file from fraud_type_label, never that column
    itself."""
    out = template_df.copy()
    out["suggested_fraud_type_label"] = out.apply(suggest_label, axis=1)
    return out


def run(source: str, data_dir: Path) -> None:
    template_path = data_dir / source / "cluster_labels_template.csv"
    if not template_path.exists():
        raise FileNotFoundError(
            f"{template_path} not found - run "
            f"`python -m models.anomaly.inspect_clusters --source {source}` first."
        )

    template_df = pd.read_csv(template_path)
    suggested = suggest_labels(template_df)

    print(f"{len(suggested)} cluster(s) - suggested label breakdown:")
    print(suggested["suggested_fraud_type_label"].value_counts().to_string())

    non_english_fallback = suggested["suggested_fraud_type_label"].isin(
        ["flooding_burst", "templated_multi_sender_campaign", "unclassified_review_needed"]
    ).sum()
    keyword_matched = len(suggested) - non_english_fallback - (
        suggested["suggested_fraud_type_label"] == "incoherent_no_pattern"
    ).sum()
    print(
        f"\n{keyword_matched}/{len(suggested)} cluster(s) got a keyword-based content "
        f"suggestion; the rest fell back to shape-only or incoherent - see module "
        "docstring's honest English-bias limitation before trusting this heuristic "
        "on multilingual clusters."
    )

    out_path = data_dir / source / "cluster_labels_suggested.csv"
    suggested.to_csv(out_path, index=False)
    print(
        f"\nWrote {out_path} - a SEPARATE file from cluster_labels_template.csv, "
        "never overwrites your real fraud_type_label edits. Review "
        "suggested_fraud_type_label against the real sample_texts before typing "
        "anything into the real template's fraud_type_label column."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    run(args.source, Path(args.data_dir))


if __name__ == "__main__":
    main()
