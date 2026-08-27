"""
pytest suite for labels.rule_labels - is_rule_evaluated() / build_rule_labels().
Operates directly on small label_source-shaped frames (decision/rule/
rule_name/fraud_type columns), independent of any source's ingestion code.

Run:
    pytest tests/test_rule_labels.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labels.rule_labels import build_rule_labels, is_rule_evaluated


@pytest.fixture
def label_source() -> pd.DataFrame:
    return pd.DataFrame([
        # 0: real spam-pattern rule fired and flagged as spam specifically
        {"decision": 1, "rule": "S_regex_url", "rule_name": "Regex 1 - HTTP/HTTPS URL smpp", "fraud_type": "spam"},
        # 1: never touched by any rule at all
        {"decision": None, "rule": None, "rule_name": None, "fraud_type": None},
        # 2: SmartWhitelist pre-check hit - allowlisted, content never scored
        {"decision": 0, "rule": "SW_1154836128318251_20251209005342_master1", "rule_name": "Esms3 A2P Domestic", "fraud_type": None},
        # 3: real spam-pattern rule fired, explicitly allowed (not flagged)
        {"decision": 0, "rule": "S_another_check", "rule_name": "Some other check", "fraud_type": None},
        # 4: real rule fired, decision=1, but flagged for a DIFFERENT fraud
        # type - not spam. Real pattern found on SS7 data: decision==1 rows
        # split spam/generic/abuse_word - only "spam" should count as
        # rule_flagged for this (spam-specific) project.
        {"decision": 1, "rule": "S_generic_check", "rule_name": "Some generic fraud check", "fraud_type": "generic"},
        # 5: decision set (0) but rule/rule_name both blank - the gap real
        # SMPP data has (see ingestion/smpp.py's LABEL_SOURCE_COLS comment):
        # decision is a more complete "touched" signal than rule/rule_name,
        # so this must still count as evaluated even without a rule id.
        {"decision": 0, "rule": None, "rule_name": None, "fraud_type": None},
    ])


def test_is_rule_evaluated_true_for_real_rule_hit(label_source):
    assert is_rule_evaluated(label_source).tolist() == [True, False, False, True, True, True]


def test_is_rule_evaluated_true_for_decision_only_with_blank_rule(label_source):
    """Row 5: decision=0, rule/rule_name both blank - decision is the base
    "touched" signal now, so this counts as evaluated with no rule id at
    all, and since `rule` isn't SW_-prefixed (it's blank, not SW_) it isn't
    excluded as whitelist-only either."""
    assert is_rule_evaluated(label_source).iloc[5] == True


def test_is_rule_evaluated_excludes_whitelist_only_rows(label_source):
    """SW_* is a pre-check allowlist gate (decision=0, content never
    evaluated for spam) - it must NOT count as evaluated, or a supervised
    model would train on whitelist membership instead of spam content."""
    assert is_rule_evaluated(label_source).iloc[2] == False


def test_build_rule_labels_flags_fraud_type_spam_specifically(label_source):
    """Positive label is fraud_type=='spam', not decision==1 - decision==1
    covers other fraud categories too (row 4: generic), which must NOT be
    treated as spam-positive for this project."""
    flagged = build_rule_labels(label_source)
    assert flagged.tolist() == [True, False, False, False, False, False]


def test_decision_equals_one_is_not_sufficient_for_flagged(label_source):
    """Row 4 has decision==1 (the rule engine flagged it for something) but
    fraud_type=='generic', not 'spam' - must be False, not True."""
    flagged = build_rule_labels(label_source)
    assert flagged.iloc[4] == False


def test_rule_flagged_is_na_when_never_evaluated(label_source):
    """rule_flagged must never be fabricated as False for a row no
    spam-pattern rule ever touched (untouched OR whitelist-only) - that
    would claim a verdict the rule engine never gave."""
    evaluated = is_rule_evaluated(label_source)
    flagged = build_rule_labels(label_source).where(evaluated)
    assert pd.isna(flagged.iloc[1])  # never touched
    assert pd.isna(flagged.iloc[2])  # whitelist-only


def test_raises_when_decision_column_missing():
    with pytest.raises(ValueError):
        is_rule_evaluated(pd.DataFrame({"rule": ["S_x", None]}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
