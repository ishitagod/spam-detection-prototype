"""
pytest suite for models.anomaly.suggest_cluster_labels - the EXPERIMENTAL
rule-based suggestion heuristic (see that module's docstring for the hard
constraint this must never violate: it never writes fraud_type_label or
feeds labels/cluster_labels.py directly, only its own separate
suggested_fraud_type_label column/file).

Run:
    pytest tests/test_suggest_cluster_labels.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly.suggest_cluster_labels import suggest_label, suggest_labels


def _row(**overrides) -> pd.Series:
    defaults = dict(
        cluster_label=0, n_rows=10, n_unique_texts=1, n_unique_originators=1,
        sample_texts="hello there",
    )
    defaults.update(overrides)
    return pd.Series(defaults)


def test_incoherent_gate_fires_before_anything_else():
    """High diversity on BOTH text AND originator must win even if the
    row would otherwise match a keyword rule - an incoherent cluster
    shouldn't get a confident-sounding content label."""
    row = _row(n_rows=10, n_unique_texts=9, n_unique_originators=9, sample_texts="whatsapp chat")
    assert suggest_label(row) == "incoherent_no_pattern"


def test_high_text_diversity_alone_is_not_incoherent_when_originators_are_concentrated():
    """Real counter-example this fix was built for: a large cluster with
    near-100% unique texts (dynamic template fields - amounts/card
    numbers/OTP codes) but few distinct originators is a genuinely
    coherent, sender-concentrated pattern, not incoherent noise - see
    INCOHERENT_UNIQUE_ORIGINATOR_RATIO's comment."""
    row = _row(n_rows=1000, n_unique_texts=994, n_unique_originators=12, sample_texts="unmatched content")
    assert suggest_label(row) != "incoherent_no_pattern"


def test_incoherent_ratio_boundary_is_inclusive():
    row = _row(n_rows=10, n_unique_texts=5, n_unique_originators=5)  # both exactly 0.5
    assert suggest_label(row) == "incoherent_no_pattern"


@pytest.mark.parametrize("text,expected", [
    ("Chat with me on WhatsApp!", "chat_app_invite_spam"),
    ("Claim your jackpot bonus now", "gambling_promo_spam"),
    ("Your OTP is 1234, verify your account", "otp_verification_bait"),
    ("RM0 AmBank: your debit card withdraw alert", "bank_transaction_alert"),
    ("Click https://example.com/win now", "url_link_spam"),
])
def test_keyword_rules_match_expected_label(text, expected):
    row = _row(n_rows=10, n_unique_texts=1, n_unique_originators=3, sample_texts=text)
    assert suggest_label(row) == expected


def test_keyword_rules_are_case_insensitive():
    row = _row(sample_texts="WHATSAPP invite here")
    assert suggest_label(row) == "chat_app_invite_spam"


def test_no_keyword_match_single_text_single_originator_is_flooding_burst():
    row = _row(n_unique_texts=1, n_unique_originators=1, sample_texts="random unmatched text")
    assert suggest_label(row) == "flooding_burst"


def test_no_keyword_match_single_text_multi_originator_is_templated_campaign():
    row = _row(n_unique_texts=2, n_unique_originators=5, sample_texts="random unmatched text")
    assert suggest_label(row) == "templated_multi_sender_campaign"


def test_no_match_and_diverse_text_falls_back_to_unclassified():
    """Below the incoherent threshold but above the flooding/campaign
    n_unique_texts<=2 cutoff, with no keyword hit - a real, expected
    fallback, not a bug (see module docstring)."""
    row = _row(n_rows=10, n_unique_texts=3, n_unique_originators=3, sample_texts="assorted text")
    assert suggest_label(row) == "unclassified_review_needed"


def test_zero_rows_does_not_crash_on_division():
    """n_rows=0 is a degenerate/defensive case (shouldn't occur from real
    inspect_clusters.py output, which never writes an empty cluster row) -
    must not raise ZeroDivisionError, and the 0/0 ratio correctly reads as
    0.0 (not incoherent), falling through to the shape-based rules."""
    row = _row(n_rows=0, n_unique_texts=0, n_unique_originators=1, sample_texts="")
    assert suggest_label(row) == "flooding_burst"


def test_suggest_labels_adds_column_without_touching_fraud_type_label():
    """HARD CONSTRAINT check: this must never write/modify a real
    fraud_type_label column, even if the input template already has one
    (e.g. a partially hand-labeled real template)."""
    df = pd.DataFrame([
        {**_row(n_unique_texts=1, n_unique_originators=1, sample_texts="x").to_dict(),
         "fraud_type_label": "human_already_labeled_this"},
        {**_row(n_unique_texts=1, n_unique_originators=1, sample_texts="y").to_dict(),
         "fraud_type_label": ""},
    ])
    out = suggest_labels(df)
    assert "suggested_fraud_type_label" in out.columns
    # The human's existing label must survive completely untouched.
    assert out["fraud_type_label"].tolist() == ["human_already_labeled_this", ""]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
