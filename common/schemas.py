"""
Canonical schema contract - the single source of truth for column names
that ingestion/smpp.py and ingestion/ss7.py must both map their raw,
source-specific columns into. Before this module existed, the canonical
column set only existed implicitly as the target-side keys of each
source's *_FEATURE_MAP dict - nothing stopped smpp.py and ss7.py from
drifting a name (e.g. "originator" vs "originator_id") independently.
"""

import pandas as pd

# canonical column -> expected pandas dtype (as a string, checked loosely -
# see validate_features). Matches CLAUDE.md's "one shared feature contract
# for both sources": every canonical column below is source-agnostic,
# `source` itself is the only thing that tells SMPP and SS7 rows apart.
CANONICAL_FEATURE_SCHEMA: dict[str, str] = {
    "record_id": "object",  # SMPP join key (message_id is null on op-4 rows)
    "message_id": "object",  # SS7 join key
    "source": "object",  # "SMPP" | "SS7"
    "originator": "object",
    "destination": "object",
    "text": "object",
    "timestamp": "object",  # left as raw string/object at this stage;
    # parsed to datetime downstream in
    # features/behavioral.py, not here
    "dcs": "float64",
    "text_decode_failed": "bool",  # SMPP-only today - real feature per
    # ingestion/smpp.py, not bookkeeping
}

# canonical label columns - output of labels/rule_labels.py, NEVER also
# present in CANONICAL_FEATURE_SCHEMA (see ingestion/base.py docstring on
# why features and labels are always returned as two separate frames).
CANONICAL_LABEL_SCHEMA: dict[str, str] = {
    "record_id": "object",  # SMPP
    "message_id": "object",  # SS7
    "fraud_type": "object",
    "rule_evaluated": "bool",
    "rule_flagged": "object",  # nullable bool (True / False / NA) -
    # NA is a real, distinct value here,
    # see labels/rule_labels.py
}


def validate_features(df: pd.DataFrame, *, required: list[str] | None = None) -> None:
    """
    Loose structural check: every column in `required` (default: every key
    in CANONICAL_FEATURE_SCHEMA that's actually present in df's source -
    callers should pass the subset their source populates) must exist.
    Raises ValueError naming what's missing. Does not check dtypes strictly
    (real data has NaN-laden columns that don't round-trip cleanly through
    a single dtype) - this catches "column renamed/dropped by accident",
    the actual drift risk this module exists to prevent.
    """
    required = required if required is not None else list(CANONICAL_FEATURE_SCHEMA)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical column(s): {missing}")


def validate_labels(df: pd.DataFrame, *, required: list[str] | None = None) -> None:
    """Same check as validate_features(), against CANONICAL_LABEL_SCHEMA."""
    required = required if required is not None else list(CANONICAL_LABEL_SCHEMA)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical label column(s): {missing}")
