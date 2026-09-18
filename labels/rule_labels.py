"""
Turns the upstream rule engine's raw verdict columns (decision/rule/
rule_name) into the two canonical label columns every source's mapper
returns: rule_evaluated, rule_flagged.

Named rule_labels.py, not rule_engine.py, on purpose: this module does not
implement a rule engine, it derives training labels FROM one. There are
three distinct "rule" things in this codebase - don't conflate them:
  1. the upstream rule engine (external system, produces decision/rule/
     rule_name - not our code)
  2. this module (label derivation from #1's output)
  3. models/rule_pattern/ (the LightGBM model trained to approximate #1)

A FOURTH, genuinely independent label source lives at the bottom of this
module (content_flagged()/is_content_evaluated()): the upstream telecom
rule engine (#1 above) has ZERO content/regex matching of its own
(confirmed against real op-4 data) - features/content_flags.py's static
regex flags are this codebase's OWN label source, not a restatement of #1.
Kept in its own label_source ("content_static_rules") and NEVER silently
merged with rule_flagged ("telecom_rule_engine") - see those functions'
docstrings for why, and the architecture plan's Section 6 for the
tautology-ablation this separation unblocks (training with content flags
as BOTH feature and label source would just reconstruct the labeling
formula if the two were conflated).

Rows where decision/rule/fraud_type are NULL/empty = the rule engine did
NOT flag them. This is exactly the pool the unsupervised layer
(Isolation Forest + FAISS near-duplicate) needs to work on - it's the only
place genuinely novel (rule-invisible) spam can be found, since anything
the rules already caught is, by definition, a KNOWN pattern.
"""
import pandas as pd

from config.settings import WHITELIST_RULE_PREFIX
from models.anomaly.data import CONTENT_FLAG_COLS

# label_source column values - see content_flagged()/is_content_evaluated()
# below for why these are never silently merged into rule_flagged/
# is_rule_evaluated()'s telecom-derived labels.
LABEL_SOURCE_TELECOM_RULE_ENGINE = "telecom_rule_engine"
LABEL_SOURCE_CONTENT_STATIC_RULES = "content_static_rules"


def build_rule_labels(label_source_df: pd.DataFrame, fraud_type_col="fraud_type") -> pd.Series:
    """
    Turns the rule engine's raw fraud_type column into a binary
    rule_flagged label: True = rule engine caught this as SPAM specifically
    (fraud_type == "spam"), False otherwise.

    Uses fraud_type, NOT decision==1, on purpose. `decision` is a general
    "did any rule fire" signal covering MULTIPLE fraud categories, not just
    spam - this project is a spam detector (see CLAUDE.md), so the
    supervised label needs to be spam-specific, not "flagged for any
    reason". Checked against real decision==1 rows across 6 files/source:
      SMPP: fraud_type is "spam" on 100% of decision==1 rows (230/230) -
            no difference from the old decision==1 definition here.
      SS7:  fraud_type splits spam (28,559) / generic (2,327) /
            abuse_word (14) / NaN (21,518, but these are ALL MT_SRI_response
            rows ingestion.ss7.clean() already drops - never reach here).
            So on real MO/MT_request data, ~7.5% of decision==1 rows are
            "generic"/"abuse_word", not spam - the old decision==1
            definition was quietly training the SS7 supervised model on
            2,341 non-spam-fraud rows as if they were spam-positive.
    No variant spellings found ("spam_burst" etc.) - fraud_type is a clean,
    consistent vocabulary in the real data checked, exact match is safe.

    IMPORTANT: rule_flagged=False on its own is NOT "confirmed legitimate" -
    it only means fraud_type wasn't "spam" (could be evaluated-but-allowed,
    OR evaluated-and-flagged-for-a-different-fraud-type, OR never evaluated
    at all). Some rule_flagged=False rows were never even evaluated by any
    rule (see is_rule_evaluated() below) - those are NOT a trustworthy
    negative for supervised training, they're unknown status. Always gate
    supervised use of this label on is_rule_evaluated()==True.
    """
    return label_source_df[fraud_type_col] == "spam"


def is_rule_evaluated(label_source_df: pd.DataFrame) -> pd.Series:
    """
    True where a spam-PATTERN rule actually scored this message and reached
    an explicit verdict - "allow" (decision=0) or "flag" (decision=1).

    `rule` starting with WHITELIST_RULE_PREFIX ("SW_" - SmartWhitelist) is
    excluded and does NOT count as evaluated: SmartWhitelist is a pre-check
    gate (sender/route ID present in a whitelist DB -> auto-allow),
    decision=0, that runs BEFORE any fraud/spam rule and skips content
    evaluation entirely when it hits - so "SW_ matched" means "content was
    never checked", not "checked and clean". Confirmed on real op-4 data
    across 6 files: `SW_*` rows are always decision=0 and vastly outnumber
    real spam-pattern (`S_*`) rows (~1000:1) - treating them as confident
    negatives would train the supervised model on whitelist membership, not
    spam content.

    is_rule_evaluated == True  -> LABELLED pool (real S_* verdict) - use for
                                   supervised "rule_pattern" training.
    is_rule_evaluated == False -> UNLABELLED pool (untouched OR
                                   whitelist-only) - feed to the
                                   unsupervised layer instead. Do NOT
                                   assign rule_flagged=0 to these rows for
                                   supervised training.

    `decision` is the base "touched by the rule engine" signal, not
    `rule`/`rule_name`: real SMPP data has rows where `decision` is set (0
    or 1) but `rule`/`rule_name` is blank - decision is the more complete
    signal (see ingestion/smpp.py's LABEL_SOURCE_COLS comment). `rule` is
    then used only to subtract whitelist bypasses out of that pool -
    `rule_name` isn't needed for anything here.

    ONLY decision in {0, 1} counts as evaluated - real SS7 data has 11
    distinct `decision` codes, not just 0/1 (0: 13.2M, 1: 824,765, plus 9
    other codes totalling 634,783 rows - 6, 27, 13, 32, 31, 5, 3, 34, 9,
    36). Checked: EVERY one of those 634,783 other-code rows has
    fraud_type == NaN - none are ever labelled spam, so restricting to
    {0, 1} costs zero real positive examples. Without this restriction
    they'd all fall into the "confirmed clean" bucket by default (fraud_type
    != "spam"), which is not a trustworthy negative - decision=0
    specifically means "rule engine confident not spam"; these other codes
    are a different, unconfirmed signal (SMPP never hits this - its
    `decision` is already only ever 0, 1, or blank).
    """
    if "decision" not in label_source_df.columns:
        raise ValueError("label_source_df has no `decision` column")
    decision = pd.to_numeric(label_source_df["decision"], errors="coerce")
    evaluated = decision.isin([0, 1])
    if "rule" in label_source_df.columns:
        whitelist_only = label_source_df["rule"].astype("string").str.startswith(
            WHITELIST_RULE_PREFIX, na=False
        )
        evaluated &= ~whitelist_only
    return evaluated


# --- content-static-rules label source (see module docstring's "FOURTH
# rule thing" note) -------------------------------------------------------
#
# DIFFERENT INPUT SHAPE from build_rule_labels()/is_rule_evaluated() above:
# those operate on label_source_df, the raw per-source ingestion frame
# (decision/rule/fraud_type - present at ingestion time, before
# reassembly). The functions below operate on messages_with_behavioral.csv
# AFTER features/content_flags.py's Stage 3b has run (CONTENT_FLAG_COLS +
# text_decode_failed present) - a genuinely later pipeline stage, since
# content flags need reassembled `text`, which doesn't exist yet at
# ingestion time. Callers must pass the right frame to the right function.

# High-confidence flag COMBINATIONS that count as "content-rule flagged" -
# a single common flag alone (e.g. has_url) is too weak/noisy on its own
# (real messages link things routinely); these combinations were picked as
# a stronger signal, same "starting point, not calibrated" status as
# config/settings.py's CONTENT_FLAG_PATTERNS regex list itself - revisit
# once real precision/recall against confirmed spam is measured.
CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS = [
    ["has_gambling_keyword"],
    ["has_otp_keyword", "has_urgency_keyword"],
    ["has_url", "has_urgency_keyword"],
    ["has_url", "has_prize_keyword"],
    ["has_loan_keyword", "has_urgency_keyword"],
]


def is_content_evaluated(messages_df: pd.DataFrame) -> pd.Series:
    """
    True wherever features/content_flags.py's compute_content_flags()
    actually ran on real, decodable text - always True in practice except
    text_decode_failed rows (undecodable text still gets flag columns, all
    0, via compute_content_flags()'s NaN-safe fillna("") - but "no flags
    fired because there was no real content to check" is not the same
    claim as "checked and found nothing", the same is_rule_evaluated()
    vs rule_flagged distinction this module already draws for the telecom
    label source above.
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
    """
    True where CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS' set of flag
    combinations fires (any one combination, ALL of its flags true) - a
    SECOND, INDEPENDENT label source from rule_flagged (see module
    docstring). Never merged with rule_flagged into one column - a
    training pool that wants to combine them must pass an explicit
    `label_sources` list (see architecture plan Section 6) so it's always
    clear which rows came from which source, and combined use stays
    opt-in rather than the default.
    """
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
