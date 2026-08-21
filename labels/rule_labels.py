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

Rows where decision/rule/fraud_type are NULL/empty = the rule engine did
NOT flag them. This is exactly the pool the unsupervised layer
(Isolation Forest + FAISS near-duplicate) needs to work on - it's the only
place genuinely novel (rule-invisible) spam can be found, since anything
the rules already caught is, by definition, a KNOWN pattern.
"""
import pandas as pd

from config.settings import WHITELIST_RULE_PREFIX


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
    """
    cols = [c for c in ("rule", "rule_name") if c in label_source_df.columns]
    if not cols:
        raise ValueError("label_source_df has neither `rule` nor `rule_name` column")
    evaluated = pd.Series(False, index=label_source_df.index)
    for c in cols:
        evaluated |= label_source_df[c].notna()
    if "rule" in label_source_df.columns:
        whitelist_only = label_source_df["rule"].astype("string").str.startswith(
            WHITELIST_RULE_PREFIX, na=False
        )
        evaluated &= ~whitelist_only
    return evaluated
