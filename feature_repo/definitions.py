"""
Feast feature repo: what CLAUDE.md's "offline and online ML platform are
the SAME environment for this prototype" simplification looks like in
practice - the online store exists so FastAPI can look up a sender's behavioral state in a
single fast key-value read at inference time

Two-piece design, because one of the four behavioral features can't be
precomputed:
  - `sender_behavioral_stats` (FeatureView, backed by
    features/behavioral_snapshot.py's materialized parquet): the pure
    sender-state features - sender_msgs_last_5min, sender_msgs_last_1hr,
    sender_unique_destinations_1hr, sender_age_days - plus
    recent_text_counts_json, which exists ONLY to feed the on-demand view
    below, not as a feature in its own right.
  - `sender_repeat_content_ratio` (on-demand feature view): the fourth
    feature, sender_repeat_content_ratio_1hr, needs the INCOMING
    message's text - a value that doesn't exist until the actual
    inference request arrives, so it cannot live in the online store like
    the other three. Feast's on-demand feature view is exactly the
    mechanism for "combine a stored feature with a request-time value at
    serving time" - see features/behavioral_snapshot.py's module
    docstring for why recent_text_counts_json is a top-K approximation,
    not the exact set.

Apply / materialize :
    cd feature_repo
    feast apply
    feast materialize-incremental $(python -c "import datetime; print(datetime.datetime.now().isoformat())")

Both driven by scripts/refresh_feast.py in this repo, which also rebuilds
the snapshot parquet first - see that script rather than running the
steps above by hand.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Int64, String, Float64
from feast.value_type import ValueType
from feast import Project

project = Project(
    name="spam_detection",
    description="SMS spam detection prototype - sender-behavioural features for online serving",
)
REPO_ROOT = Path(__file__).resolve().parent.parent
# `feast apply`/`FeatureStore(repo_path=...)` import this file directly by
# path, not as part of the `features` package, so the repo root isn't on
# sys.path by default - add it so DEFAULT_SNAPSHOT_PATH below can come
# from features/behavioral_snapshot.py's single source of truth instead
# of a second hardcoded copy of the same path (see that module's comment
# on why the path lives there).
sys.path.insert(0, str(REPO_ROOT))
from features.behavioral_snapshot import (
    DEFAULT_SNAPSHOT_PATH as SNAPSHOT_PATH,
)  # noqa: E402

sender = Entity(
    name="sender_id",
    join_keys=["sender_id"],
    value_type=ValueType.STRING,
    description=(
        "Compound key 'source|originator' - see features/behavioral.py's "
        "sender-key note: SMPP originators (business sender IDs) and SS7 "
        "originators (real MSISDNs) are different namespaces that could "
        "coincidentally collide on the same string, so source is always "
        "part of the key."
    ),
)

sender_behavioral_snapshot_source = FileSource(
    name="sender_behavioral_snapshot_source",
    path=str(SNAPSHOT_PATH),
    timestamp_field="event_timestamp",
    description="features/behavioral_snapshot.py's output - refreshed by scripts/refresh_feast.py, not written by hand.",
)

sender_behavioral_stats = FeatureView(
    name="sender_behavioral_stats",
    entities=[sender],
    # Point-in-time OFFLINE joins only look back this far for a sender's
    # snapshot row - deliberately looser than the 1hr/5min feature windows
    # themselves, so a sender who's been quiet for a few hours still
    # resolves to their (correctly zeroed-out) last snapshot instead of a
    # missing join. Does NOT bound ONLINE serving freshness - the online
    # store always returns whatever was last materialized regardless of
    # age; see scripts/refresh_feast.py's docstring for how staleness is
    # actually bounded in this prototype (re-run cadence, not TTL).
    ttl=timedelta(hours=6),
    schema=[
        Field(name="sender_msgs_last_5min", dtype=Int64),
        Field(name="sender_msgs_last_1hr", dtype=Int64),
        Field(name="sender_unique_destinations_1hr", dtype=Int64),
        Field(name="recent_text_counts_json", dtype=String),
        # ALL-TIME aggregate (not windowed like the four above) - see
        # features/behavioral_snapshot.py's SENDER_AGE_DAYS docstring note.
        Field(name="sender_age_days", dtype=Float64),
    ],
    online=True,
    source=sender_behavioral_snapshot_source,
)

# Request-time-only input: the message actually being scored right now.
# Not stored anywhere - exists purely so the on-demand view below can
# accept it alongside the entity's stored features.
candidate_message_request = RequestSource(
    name="candidate_message_request",
    schema=[Field(name="candidate_text", dtype=String)],
)


@on_demand_feature_view(
    sources=[sender_behavioral_stats, candidate_message_request],
    schema=[Field(name="sender_repeat_content_ratio_1hr", dtype=Float64)],
)
def sender_repeat_content_ratio(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors features/behavioral.py's exact repeat_content_ratio
    definition (candidate text's share of this sender's trailing-1hr
    message count) EXCEPT it can only see the top-K texts
    features/behavioral_snapshot.py stored, not the sender's full recent
    text set - see that module's docstring for why that's an accepted
    prototype approximation.
    """
    import json

    denom = inputs["sender_msgs_last_1hr"]
    ratios = []
    for counts_json, candidate, n in zip(
        inputs["recent_text_counts_json"], inputs["candidate_text"], denom
    ):
        if not n or n <= 0:
            ratios.append(0.0)
            continue
        # Guarded, not just "if it's a non-empty str": `feast apply` probes
        # this UDF with RANDOM dummy values to infer the output schema
        # (not just real materialized rows at serving time), so a
        # non-empty-but-garbage string here is expected, not a data bug.
        try:
            counts = (
                json.loads(counts_json)
                if isinstance(counts_json, str) and counts_json
                else {}
            )
        except (json.JSONDecodeError, TypeError):
            counts = {}
        ratios.append(counts.get(candidate, 0) / n)
    return pd.DataFrame({"sender_repeat_content_ratio_1hr": ratios})
