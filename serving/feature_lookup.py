"""
Single-lookup manual test script for the Feast online store - the
`predict.py`-style "run it by hand against one row" pattern CLAUDE.md
carries over from the earlier fraud-detection project, applied to feature
lookup rather than model inference (no model exists yet to call this from
- see CLAUDE.md's next-steps list; models/serving/FastAPI will call
get_sender_features() below the same way this script's __main__ block
does).

This is what "fast" inference-time feature retrieval looks like in this
prototype: one SQLite key-value read for the three stored behavioral
counts (sender_msgs_last_5min/1hr, sender_unique_destinations_1hr) plus
one in-process Python function call for the fourth
(sender_repeat_content_ratio_1hr, computed on demand from the incoming
message's own text - see feature_repo/definitions.py for why that one
can't be precomputed). No per-request full-history rescan.

STALENESS CAVEAT: values returned here are only as fresh as the last
`python scripts/refresh_feast.py` run - the online store does not
recompute anything at read time, it serves whatever was last
materialized. A sender who has been sending heavily in the seconds since
the last refresh will under-count until the next refresh. Acceptable for
this prototype's batch/synchronous design (CLAUDE.md: "Batch/synchronous
feature computation, not streaming, at this stage") - revisit the refresh
cadence, not this script, if that lag becomes a problem.

Usage:
    python serving/feature_lookup.py --sender_id "SMPP|66688" --candidate_text "WIN A PRIZE NOW"
"""
import argparse
from pathlib import Path

from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO_DIR = REPO_ROOT / "feature_repo"

FEATURE_REFS = [
    "sender_behavioral_stats:sender_msgs_last_5min",
    "sender_behavioral_stats:sender_msgs_last_1hr",
    "sender_behavioral_stats:sender_unique_destinations_1hr",
    "sender_repeat_content_ratio:sender_repeat_content_ratio_1hr",
    # Tier 0 additions - see feature_repo/definitions.py's schema comment
    # and features/behavioral_snapshot.py's docstring. sender_age_days was
    # already stored/materialized before this - it just was never
    # requested here.
    "sender_behavioral_stats:sender_age_days",
    "sender_behavioral_stats:sender_recipient_diversity_ratio_5min",
    "sender_behavioral_stats:sender_recipient_diversity_ratio_1hr",
    "sender_behavioral_stats:sender_velocity_zscore_5min",
]

# SS7-only, keyed on `imsi` not `sender_id` (feature_repo/definitions.py's
# `imsi` entity - see features/behavioral.py's IMSI-LINKAGE note). A
# separate feature ref list/lookup function, not folded into
# get_sender_features() above: a different entity means a different
# Feast entity_rows key, not just another feature name.
IMSI_FEATURE_REFS = [
    "imsi_behavioral_stats:imsi_distinct_originators_1hr",
]


def get_sender_features(sender_id: str, candidate_text: str, store: FeatureStore | None = None) -> dict:
    """
    Returns the four behavioral features for one (sender, candidate
    message) pair, as a plain dict - what a FastAPI inference endpoint
    would call per incoming message once that service exists.

    An unknown sender_id (never seen in the last snapshot/materialize)
    returns None for every feature, not 0 - Feast is honest about "no
    entity found" rather than silently guessing cold-start zeros; the
    CALLER (eventually the FastAPI service, here just __main__ below)
    decides whether "unknown sender" should be treated as cold-start-zero
    for scoring purposes, same "disclose, don't fake certainty" principle
    as the architecture plan's `confidence` field.
    """
    store = store or FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    result = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"sender_id": sender_id, "candidate_text": candidate_text}],
    ).to_dict()
    return {k: v[0] for k, v in result.items() if k not in ("sender_id", "candidate_text")}


def get_imsi_features(imsi: str | None, store: FeatureStore | None = None) -> dict:
    """
    SS7-only counterpart to get_sender_features() above, keyed on `imsi`
    (see IMSI_FEATURE_REFS). `imsi=None` (SMPP requests, or an SS7 request
    whose own imsi is genuinely unknown) skips the Feast lookup entirely
    and returns {"imsi_distinct_originators_1hr": None} directly - same
    "honest unknown, not a fabricated 0" result an unrecognized imsi would
    get back from Feast anyway, without paying for a lookup Feast can
    never answer (there is no None entity to look up).
    """
    if imsi is None:
        return {"imsi_distinct_originators_1hr": None}
    store = store or FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    result = store.get_online_features(
        features=IMSI_FEATURE_REFS,
        entity_rows=[{"imsi": imsi}],
    ).to_dict()
    return {k: v[0] for k, v in result.items() if k != "imsi"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender_id", type=str, required=True, help='e.g. "SMPP|66688" - source|originator, see features/behavioral.py')
    parser.add_argument("--candidate_text", type=str, required=True)
    args = parser.parse_args()

    features = get_sender_features(args.sender_id, args.candidate_text)
    print(f"sender_id:       {args.sender_id}")
    print(f"candidate_text:  {args.candidate_text!r}")
    for name, value in features.items():
        print(f"  {name}: {value}")
    if all(v is None for v in features.values()):
        print("\n(all None - sender_id not found in the online store; either it's genuinely new, or scripts/refresh_feast.py needs a re-run)")


if __name__ == "__main__":
    main()
