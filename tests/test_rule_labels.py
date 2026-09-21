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

import numpy as np

from labels.rule_labels import (
    build_rule_labels,
    content_flag_count,
    content_flag_weighted_score,
    content_flag_weights,
    content_flagged_by_count,
    content_flagged_by_weight,
    fit_content_flag_weights,
    is_rule_evaluated,
)
from models.anomaly.data import CONTENT_FLAG_COLS


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
        # 6: decision is a real, non-blank value but NOT 0 or 1 - real SS7
        # data has 9 other decision codes (6, 27, 13, 32, 31, 5, 3, 34, 9,
        # 36) that are NOT the confirmed-clean(0)/flagged(1) binary SMPP
        # uses; every one of those rows has fraud_type==NaN in real data,
        # so must NOT count as evaluated (would otherwise be a fabricated
        # "confirmed clean" negative).
        {"decision": 6, "rule": None, "rule_name": None, "fraud_type": None},
    ])


def test_is_rule_evaluated_true_for_real_rule_hit(label_source):
    assert is_rule_evaluated(label_source).tolist() == [True, False, False, True, True, True, False]


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


def test_is_rule_evaluated_excludes_non_binary_decision_codes(label_source):
    """Row 6: decision=6 - a real, non-blank SS7 decision code that isn't
    the confirmed-clean(0)/flagged(1) pair. Must NOT count as evaluated:
    every real row with a decision outside {0, 1} has fraud_type==NaN, so
    including them would fabricate a "confirmed clean" negative out of an
    unconfirmed signal."""
    assert is_rule_evaluated(label_source).iloc[6] == False


def test_build_rule_labels_flags_fraud_type_spam_specifically(label_source):
    """Positive label is fraud_type=='spam', not decision==1 - decision==1
    covers other fraud categories too (row 4: generic), which must NOT be
    treated as spam-positive for this project."""
    flagged = build_rule_labels(label_source)
    assert flagged.tolist() == [True, False, False, False, False, False, False]


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


@pytest.fixture
def content_flags_frame() -> pd.DataFrame:
    """One row per flag count (0, 1, 2, all-10) - CONTENT_FLAG_COLS order
    doesn't matter here, only how many are True."""
    n = len(CONTENT_FLAG_COLS)
    rows = [
        dict.fromkeys(CONTENT_FLAG_COLS, 0),  # 0 flags
        {**dict.fromkeys(CONTENT_FLAG_COLS, 0), CONTENT_FLAG_COLS[0]: 1},  # 1 flag
        {**dict.fromkeys(CONTENT_FLAG_COLS, 0), CONTENT_FLAG_COLS[0]: 1, CONTENT_FLAG_COLS[1]: 1},  # 2 flags
        dict.fromkeys(CONTENT_FLAG_COLS, 1),  # all n flags
    ]
    return pd.DataFrame(rows)


def test_content_flag_count_sums_across_all_flags(content_flags_frame):
    n = len(CONTENT_FLAG_COLS)
    assert content_flag_count(content_flags_frame).tolist() == [0, 1, 2, n]


def test_content_flagged_by_count_thresholds_on_aggregate_not_one_flag(content_flags_frame):
    """min_flags=2: rows with 0 or 1 flag are NOT flagged even though each
    individually fired at least one real pattern - this is the aggregate
    signal, distinct from content_flagged()'s hand-picked combinations."""
    flagged = content_flagged_by_count(content_flags_frame, min_flags=2)
    assert flagged.tolist() == [False, False, True, True]


def test_content_flag_count_raises_on_missing_columns():
    with pytest.raises(ValueError):
        content_flag_count(pd.DataFrame({"has_url": [1, 0]}))


@pytest.fixture
def weighted_labelled_frame() -> pd.DataFrame:
    """Synthetic rule_evaluated==True pool, both classes present:
    has_gambling_keyword is a near-perfect predictor of rule_flagged,
    every other flag is pure noise (uncorrelated coin flips) - a fitted
    LogisticRegression should assign has_gambling_keyword by far the
    largest positive coefficient."""
    rng = np.random.RandomState(0)
    n = 200
    gambling = rng.randint(0, 2, size=n)
    rows = {col: rng.randint(0, 2, size=n) for col in CONTENT_FLAG_COLS}
    rows["has_gambling_keyword"] = gambling
    df = pd.DataFrame(rows)
    df["rule_flagged"] = gambling.astype(bool)
    return df


def test_fit_content_flag_weights_gives_strongest_predictor_largest_weight(weighted_labelled_frame):
    model = fit_content_flag_weights(weighted_labelled_frame)
    weights = content_flag_weights(model)
    strongest = max(weights, key=lambda k: abs(weights[k]))
    assert strongest == "has_gambling_keyword"
    assert weights["has_gambling_keyword"] > 0


def test_content_flagged_by_weight_recovers_the_true_signal(weighted_labelled_frame):
    model = fit_content_flag_weights(weighted_labelled_frame)
    score = content_flag_weighted_score(weighted_labelled_frame, model)
    flagged = content_flagged_by_weight(weighted_labelled_frame, model, threshold=0.5)
    assert (score >= 0).all() and (score <= 1).all()
    # Near-perfect separation on the fit data itself (in-sample) - real
    # held-out generalization isn't what this unit test checks, only that
    # scoring/thresholding wires together correctly.
    assert (flagged == weighted_labelled_frame["rule_flagged"]).mean() > 0.9


def test_fit_content_flag_weights_raises_on_single_class():
    n = 20
    df = pd.DataFrame({col: [0] * n for col in CONTENT_FLAG_COLS})
    df["rule_flagged"] = True  # every row the same class
    with pytest.raises(ValueError):
        fit_content_flag_weights(df)


def test_fit_content_flag_weights_raises_when_rule_flagged_missing():
    df = pd.DataFrame({col: [0, 1] for col in CONTENT_FLAG_COLS})
    with pytest.raises(ValueError):
        fit_content_flag_weights(df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
