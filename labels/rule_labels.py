"""
Turns the upstream rule engine's raw verdict columns (decision/rule/
rule_name) into rule_evaluated/rule_flagged.

Also holds a second, independent label source: content_flagged()/
is_content_evaluated(). The telecom rule engine has zero content/regex
matching of its own - features/content_flags.py's regex flags are this
codebase's own label source, kept under a separate label_source
("content_static_rules") and never merged with rule_flagged
("telecom_rule_engine"), to avoid training on flags as both feature and
label (tautology).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.settings import CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS, WHITELIST_RULE_PREFIX
from models.anomaly.data import CONTENT_FLAG_COLS

# label_source values - never merged into rule_flagged/is_rule_evaluated().
LABEL_SOURCE_TELECOM_RULE_ENGINE = "telecom_rule_engine"
LABEL_SOURCE_CONTENT_STATIC_RULES = "content_static_rules"


def build_rule_labels(
    label_source_df: pd.DataFrame, fraud_type_col="fraud_type"
) -> pd.Series:
    """
    rule_flagged = fraud_type == "spam" (not decision==1, which covers
    multiple fraud categories, not just spam). Checked against real
    decision==1 rows: SMPP is 100% spam (230/230); SS7 splits spam
    (28,559) / generic (2,327) / abuse_word (14) - the old decision==1
    definition was quietly training SS7 on 2,341 non-spam rows as
    spam-positive.

    IMPORTANT: rule_flagged=False is NOT "confirmed legitimate" - it can
    also mean never evaluated. Always gate supervised use on
    is_rule_evaluated()==True.
    """
    return label_source_df[fraud_type_col] == "spam"


def is_rule_evaluated(label_source_df: pd.DataFrame) -> pd.Series:
    """
    True where a spam-pattern rule actually reached an explicit verdict
    (decision 0=allow or 1=flag).

    Excludes `rule` starting with WHITELIST_RULE_PREFIX ("SW_" -
    SmartWhitelist): it's a pre-check gate that auto-allows before any
    spam rule runs, so "SW_ matched" means content was never checked, not
    checked-and-clean. Confirmed on real data: SW_* rows are always
    decision=0 and outnumber real spam-pattern (S_*) rows ~1000:1.

    True -> labelled pool, use for supervised training.
    False -> unlabelled (untouched or whitelist-only); feed to the
    unsupervised layer, never assign rule_flagged=0 here.

    Only decision in {0, 1} counts. Real SS7 data has 11 distinct decision
    codes; the other 9 (634,783 rows) all have fraud_type==NaN, so
    restricting to {0,1} costs zero positives while avoiding treating
    those unconfirmed codes as clean. SMPP's decision is always 0/1/blank.
    """
    if "decision" not in label_source_df.columns:
        raise ValueError("label_source_df has no `decision` column")
    decision = pd.to_numeric(label_source_df["decision"], errors="coerce")
    evaluated = decision.isin([0, 1])
    if "rule" in label_source_df.columns:
        whitelist_only = (
            label_source_df["rule"]
            .astype("string")
            .str.startswith(WHITELIST_RULE_PREFIX, na=False)
        )
        evaluated &= ~whitelist_only
    return evaluated


# --- content-static-rules label source ---
# Different input shape from above: these operate on
# messages_with_behavioral.csv AFTER content_flags.py's Stage 3b (needs
# reassembled `text`, not available at ingestion time).

def is_content_evaluated(messages_df: pd.DataFrame) -> pd.Series:
    """
    True where content_flags.py actually ran on decodable text. Excludes
    text_decode_failed rows - those get all-0 flags via fillna(""), but
    "no flags fired, no content to check" isn't "checked, found nothing".
    """
    missing = [c for c in CONTENT_FLAG_COLS if c not in messages_df.columns]
    if missing:
        raise ValueError(
            f"messages_df is missing content-flag column(s): {missing} - "
            "run features/content_flags.py (pipeline.py Stage 3b) first"
        )
    if "text_decode_failed" in messages_df.columns:
        return ~messages_df["text_decode_failed"].astype(bool)
    return pd.Series(True, index=messages_df.index)


def content_flagged(messages_df: pd.DataFrame) -> pd.Series:
    """True where any CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS combo
    fires (all flags in one combo true). Independent label source from
    rule_flagged, never merged - see module docstring."""
    missing = [c for c in CONTENT_FLAG_COLS if c not in messages_df.columns]
    if missing:
        raise ValueError(
            f"messages_df is missing content-flag column(s): {missing} - "
            "run features/content_flags.py (pipeline.py Stage 3b) first"
        )
    flagged = pd.Series(False, index=messages_df.index)
    for combination in CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS:
        combo_hit = pd.Series(True, index=messages_df.index)
        for col in combination:
            combo_hit &= messages_df[col].astype(bool)
        flagged |= combo_hit
    return flagged


def content_flag_count(messages_df: pd.DataFrame) -> pd.Series:
    """Row-wise count of how many CONTENT_FLAG_COLS fired - lets callers
    threshold on "at least N flags" instead of a fixed combination."""
    missing = [c for c in CONTENT_FLAG_COLS if c not in messages_df.columns]
    if missing:
        raise ValueError(
            f"messages_df is missing content-flag column(s): {missing} - "
            "run features/content_flags.py (pipeline.py Stage 3b) first"
        )
    return messages_df[CONTENT_FLAG_COLS].sum(axis=1)


def content_flagged_by_count(
    messages_df: pd.DataFrame, min_flags: int = 3
) -> pd.Series:
    """True where content_flag_count() >= min_flags (default 3 of 10
    flags) - broader, less hand-picked than content_flagged().

    Superseded for actual training use by fit_content_flag_weights()/
    content_flagged_by_weight() below - a plain count treats every flag
    as equally strong evidence, which is arbitrary the same way a
    fixed combination is. Kept as a simple reference utility.
    """
    return content_flag_count(messages_df) >= min_flags


def fit_content_flag_weights(labelled_df: pd.DataFrame) -> LogisticRegression:
    """
    Fits LogisticRegression(CONTENT_FLAG_COLS -> rule_flagged) on
    rule_evaluated==True rows - coefficients become each flag's
    data-driven weight (magnitude + sign), vs. an unweighted count or
    hand-picked combination. Ground truth is real rule_flagged, so this
    generalizes to the unlabelled pool rather than being tautological.

    Must be fit on a pool with both classes present - SMPP alone has zero
    confirmed-clean rule_evaluated rows, so fit on combined SMPP+SS7.
    """
    missing = [c for c in CONTENT_FLAG_COLS if c not in labelled_df.columns]
    if missing:
        raise ValueError(
            f"labelled_df is missing content-flag column(s): {missing} - "
            "run features/content_flags.py (pipeline.py Stage 3b) first"
        )
    if "rule_flagged" not in labelled_df.columns:
        raise ValueError(
            "labelled_df has no rule_flagged column - pass rule_evaluated==True rows"
        )
    X = labelled_df[CONTENT_FLAG_COLS].astype(float).to_numpy()
    y = (labelled_df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712
    if len(set(y.tolist())) < 2:
        raise ValueError(
            "fit_content_flag_weights() needs both classes present in labelled_df - got only "
            "one. SMPP alone has zero confirmed-clean rule_evaluated rows (see CLAUDE.md); "
            "fit on the combined SMPP+SS7 labelled pool instead."
        )
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def content_flag_weights(model: LogisticRegression) -> dict:
    """Named {flag: coefficient} view of a model fit by
    fit_content_flag_weights() - for printing/logging the measured
    weights, not for scoring (see content_flag_weighted_score())."""
    return dict(zip(CONTENT_FLAG_COLS, model.coef_[0].tolist()))


def content_flag_weighted_score(
    messages_df: pd.DataFrame, model: LogisticRegression
) -> pd.Series:
    """
    P(rule_flagged) per row from a model fit by fit_content_flag_weights(),
    applied to ANY frame with CONTENT_FLAG_COLS present (typically the
    rule_evaluated==False pool - see content_flagged_by_weight()).
    """
    missing = [c for c in CONTENT_FLAG_COLS if c not in messages_df.columns]
    if missing:
        raise ValueError(
            f"messages_df is missing content-flag column(s): {missing} - "
            "run features/content_flags.py (pipeline.py Stage 3b) first"
        )
    X = messages_df[CONTENT_FLAG_COLS].astype(float).to_numpy()
    return pd.Series(model.predict_proba(X)[:, 1], index=messages_df.index)


def content_flagged_by_weight(
    messages_df: pd.DataFrame,
    model: LogisticRegression,
    threshold: float = 0.5,
) -> pd.Series:
    """True where content_flag_weighted_score() >= threshold - data-driven
    counterpart to content_flagged_by_count()/content_flagged().
    threshold=0.5 is unvalidated, not tuned against target precision/recall."""
    return content_flag_weighted_score(messages_df, model) >= threshold
