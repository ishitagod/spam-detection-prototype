"""
Content-rule flags: binary, deterministic regex features computed from
`text` alone - no behavioral/history dependency, unlike features/
behavioral.py. Patterns live in config/settings.py::CONTENT_FLAG_PATTERNS
(single source of truth, shared by training and live serving).

These are ML FEATURES, not Rule Engine gates - the upstream telecom Rule
Engine has zero content/regex matching of its own (confirmed against real
op-4 data this session); this module is this codebase's own, independent
content-signal source. Feeds BOTH models/rule_pattern (LightGBM) and
models/anomaly (Isolation Forest) as base features - see the architecture
plan's Section 1 for why these aren't ablation-gated like TF-IDF/embeddings
(cheap, deterministic, always-available, same category as behavioral/
near-dup columns).

Sequenced in pipeline.py as Stage 3b, after Stage 3 (behavioral) purely so
every downstream stage reads one file (messages_with_behavioral.csv) - this
module has no actual dependency on Stage 3's output, it only reads `text`.
"""
import argparse
from pathlib import Path

import pandas as pd

from config.settings import CONTENT_FLAG_PATTERNS

REQUIRED_COLS = ["text"]


def compute_content_flags(text: pd.Series) -> pd.DataFrame:
    """
    Pure function: one int8 (0/1) column per CONTENT_FLAG_PATTERNS entry,
    same index as `text`. Vectorized `str.contains`, case-insensitive.
    NaN text (see ingestion's text_decode_failed rows) matches nothing -
    every flag is 0, not NaN, since "no flag fired" is the correct,
    unambiguous state for undecodable/empty text, not a missing value.
    """
    text = text.fillna("")
    out = {}
    for name, pattern in CONTENT_FLAG_PATTERNS.items():
        out[name] = text.str.contains(pattern, case=False, regex=True, na=False).astype("int8")
    return pd.DataFrame(out, index=text.index)


def enrich_with_content_flags(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Returns `messages` with the content-flag columns added, same row order
    as the input. Does not modify `messages` in place.
    """
    missing = [c for c in REQUIRED_COLS if c not in messages.columns]
    if missing:
        raise ValueError(f"messages is missing required column(s): {missing}")

    flags = compute_content_flags(messages["text"])
    return pd.concat([messages.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)


def run_content_flags(messages_path: Path, out_path: Path) -> pd.DataFrame:
    messages_path = Path(messages_path)
    if not messages_path.exists():
        print(f"No messages file found at {messages_path}")
        return pd.DataFrame()

    print(f"Loading {messages_path} ...")
    messages = pd.read_csv(messages_path, low_memory=False)
    print(f"Computing content-rule flags for {len(messages)} message(s)...")
    enriched = enrich_with_content_flags(messages)

    for name in CONTENT_FLAG_PATTERNS:
        print(f"  {name}: {int(enriched[name].sum())} flagged")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote {len(enriched)} rows to {out_path}")
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages_path", type=str, default="data/processed/SMPP/messages_with_behavioral.csv")
    parser.add_argument("--out_path", type=str, default="data/processed/SMPP/messages_with_behavioral.csv")
    args = parser.parse_args()
    run_content_flags(Path(args.messages_path), Path(args.out_path))


if __name__ == "__main__":
    main()
