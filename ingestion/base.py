"""
Shared interface every ingestion source (smpp.py, ss7.py) implements.

SourceHandlers pairs a source's clean() and map_to_canonical() as ONE
registered unit (see run_ingest.py's SOURCES dict) instead of two separate
dicts that can drift out of sync - which is exactly how SS7 ended up
silently unwired for a while: it had a mapper but no cleaner, and nothing
surfaced that gap structurally, only a comment.

clean() decides which raw ROWS are real signal and repairs bad VALUES;
map_to_canonical() decides which raw COLUMNS become canonical features vs.
label-source, and returns (features_df, label_source_df) - kept as two
separate frames, always, so label-source columns (the rule engine's own
output) can never accidentally be passed to a model as a feature. See
common/schemas.py for the canonical column contract both must satisfy.
"""
from typing import Callable, NamedTuple

import pandas as pd

CleanFn = Callable[[pd.DataFrame], pd.DataFrame]
MapFn = Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


class SourceHandlers(NamedTuple):
    clean: CleanFn
    map_to_canonical: MapFn
