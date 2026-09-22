"""
EXPERIMENTAL rule-based cluster-label SUGGESTIONS - a heuristic to speed
up hand-labeling (anomaly_clustering.md step 4), not a replacement for it.

HARD CONSTRAINT: nothing here ever writes to `fraud_type_label`
(inspect_clusters.py's output column) or feeds
`labels/cluster_labels.py::build_cluster_labels()` directly - a
cluster_label alone is never a trustworthy verdict, and this module's
keyword/shape heuristics are weaker still. Writes to its own file
(`cluster_labels_suggested.csv`, `suggested_fraud_type_label` column) so
it can never collide with a human's in-progress edits to the real
`cluster_labels_template.csv`. A human still has to check the suggestion
against real sample texts and type the real answer themselves.

CAVEAT: keyword matching (WhatsApp/Telegram/gambling/bank/OTP phrasing)
is English-biased. Real data here is heavily multilingual (Malay,
Indonesian, German, Bengali observed) - a non-English cluster usually
falls through to a behavioral-shape-only suggestion, not a content-based
one. Expected, not hidden - the CLI prints how often it happens.

Reads inspect_clusters.py's own cluster_labels_template.csv (already has
n_rows/n_unique_texts/n_unique_originators/sample_texts per cluster) -
never reads fraud_type_clusters.parquet or messages_with_behavioral.csv
directly.

Usage:
    python -m models.anomaly.suggest_cluster_labels --source SS7
    python -m models.anomaly.suggest_cluster_labels --source SMPP
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/processed")

# Fraction of a cluster's rows that must be unique text/originator for it
# to be treated as "not a real coherent pattern" (mis-set eps merging
# heterogeneous content) rather than a real cluster. Not tuned against
# real hand-labels (none exist yet).
#
# BOTH text AND originator diversity must be high before calling a
# cluster incoherent - real counter-example: a 13,112-row SMPP cluster
# had 99.4% unique texts (bank/OTP templates with dynamic fields - card
# numbers, amounts, OTP codes - make every string unique) but only 65
# distinct originators (0.5% of rows) - a genuinely coherent,
# sender-concentrated pattern, not noise. Concentrated on either axis
# alone is a real pattern; only diverse on both at once is incoherent.
INCOHERENT_UNIQUE_TEXT_RATIO = 0.5
INCOHERENT_UNIQUE_ORIGINATOR_RATIO = 0.5

# Ordered (first match wins) case-insensitive substring rules over
# sample_texts - deliberately simple, not a real NLP classifier,
# English-biased (see module docstring). Each entry: (label, [substrings]).
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("chat_app_invite_spam", ["whatsapp", "telegram", "chat with me", "join me"]),
    ("gambling_promo_spam", ["jackpot", "claim free", "bonus", "prize", " rm ", "reward"]),
    ("otp_verification_bait", ["otp", "verify your", "verification", "reg-req"]),
    ("bank_transaction_alert", ["bank", "debit card", "withdraw", "account", "balance"]),
    ("url_link_spam", ["http://", "https://", "www.", ".com/"]),
]


def _keyword_label(sample_texts: str) -> str | None:
    """First matching KEYWORD_RULES label, or None (the expected, common
    case for non-English content)."""
    text = (sample_texts or "").lower()
    for label, substrings in KEYWORD_RULES:
        if any(s in text for s in substrings):
            return label
    return None


def suggest_label(row: pd.Series) -> str:
    """One cluster's suggested label, first match wins:
      1. Coherence gate - diverse on both text AND originator isn't a
         real pattern yet (see INCOHERENT_*_RATIO comment).
      2. Keyword content match, only for clusters that passed the gate.
      3. Behavioral-shape fallback: single unique-text + single
         originator = flooding_burst; + multiple originators =
         templated_multi_sender_campaign.
      4. unclassified_review_needed - coherent, but no rule matched.
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
    """template_df with one new column, suggested_fraud_type_label -
    separate from fraud_type_label, never overwrites it."""
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
