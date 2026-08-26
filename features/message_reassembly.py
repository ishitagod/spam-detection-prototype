"""
Reassembles multi-part SMS PDUs (physical SMS "parts") into one row per
logical message, using the concat_ref/concat_total_parts/concat_part_num
grouping key both ingestion.smpp.clean() (UDH/SAR-sourced) and
ingestion.ss7.clean() (native sarref/msg_part/msg_parts-sourced) compute -
same column names, works unchanged across both sources.

PERFORMANCE: concat_ref is null for the vast majority of real rows
(single-part messages - e.g. ~45k/49k in one real SS7 file checked). Those
are handled by a vectorized fast path (_reassemble_single_part) with no
Python-level looping; only rows with a real concat_ref go through the
groupby loop (_reassemble_multi_part). Looping over every row unconditionally
(the original implementation) took 2+ minutes on that one file; splitting
on concat_ref.notna() first is what makes this tractable at real data
volumes - do not merge the two paths back into one unconditional loop.

WHY THIS HAS TO BE A SEPARATE GLOBAL PASS, not done inside ingestion's
per-file clean()/map_to_canonical(): the raw CDR files are split hourly,
and there is no guarantee all parts of one logical message land in the
same file - a message submitted right at an hour boundary could have part
1 in one file and part 2 in the next. Reassembling per-file would silently
treat that as two broken partial messages instead of one complete one.
Same reasoning as features/behavioral.py's rolling windows - and this
stage must run BEFORE behavioral.py, since behavioral.py's per-message
counts (sender_msgs_last_5min etc.) would otherwise double/triple-count
one logical multipart message as N separate messages.

Pipeline (matches the requested steps):
  1. UDH strip + concat/SAR metadata extraction - ingestion.smpp.clean()
  2. Decode per DCS                             - ingestion.smpp.clean()
  3. Group by (originator, destination, concat_ref, concat_total_parts),
     order by concat_part_num ascending          - reassemble_messages() below
  4. Concatenate decoded text of ordered parts   - reassemble_messages() below
  5. Partial/error flagging when incomplete      - reassemble_messages() below
  6. Collapse to one row per logical message     - reassemble_messages() below

GROUPING KEY CAVEAT (read before trusting this at scale): concat_ref from
an 8-bit UDH concat IE only has 256 possible values, so it WILL repeat
across genuinely different messages from the same (originator, destination)
pair over time. This grouping key does not include a time window to guard
against that collision - a known prototype simplification. If reassembly
starts silently merging unrelated messages once the real dataset scales up
(watch message_partial rate and part counts for anomalies), add a
max-time-gap-between-parts bound here.

LABEL RECONCILIATION POLICY (unverified against real data, flag if wrong):
a message is rule_evaluated if ANY of its parts was rule_evaluated, and
rule_flagged if ANY evaluated part was flagged (union, not intersection) -
i.e. one flagged part is enough to flag the whole reassembled message.
This assumes the rule engine's per-part verdicts are meaningful signals
about the whole message, which hasn't been confirmed against real data
(e.g. does the rule engine evaluate every part independently, or only
ever flag on one specific part?) - worth checking before relying on this
for training.
"""
import argparse
from pathlib import Path

import pandas as pd

from common.schemas import verify_csv_roundtrip
from ingestion.run_ingest import load_features_csv

REQUIRED_FEATURE_COLS = [
    "record_id", "originator", "destination", "timestamp",
    "text", "text_decode_failed",
    "concat_ref", "concat_total_parts", "concat_part_num",
]
REQUIRED_LABEL_COLS = ["record_id", "rule_evaluated", "rule_flagged", "fraud_type"]


def _reassemble_single_part(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized fast path for rows with no concat_ref - already single-part
    messages, no grouping/concatenation needed. This is the overwhelming
    majority of real rows (e.g. ~45k/49k in one real SS7 file checked) -
    looping over these one Python group at a time (the only thing the old
    single groupby-over-everything implementation did) is what made this
    take 2+ minutes on that one file; the real multipart logic only ever
    needs to run on the small remainder (see _reassemble_multi_part).
    """
    out = df.drop(columns=["concat_ref", "concat_total_parts", "concat_part_num"])
    out["message_part_count"] = 1
    out["message_expected_parts"] = 1
    out["message_partial"] = False

    evaluated = df["rule_evaluated"].fillna(False).astype(bool)
    flagged = (df["rule_flagged"] == True) & evaluated
    out["rule_evaluated"] = evaluated
    out["rule_flagged"] = flagged.where(evaluated)  # NA where not evaluated
    out["fraud_type"] = df["fraud_type"].where(flagged)
    return out


def _reassemble_multi_part(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized groupby-aggregate over rows that actually have a concat_ref
    (real multipart candidates). Grouped by (originator, destination,
    concat_ref, concat_total_parts) - see the grouping-key caveat in the
    module docstring.

    NOT a per-group Python loop (an earlier version was, building a dict
    per group via g.iloc[0].to_dict() + list-append + pd.DataFrame(rows)) -
    that was fine at the small scale it was tested against (hundreds to a
    few thousand groups) but never finished on real full-dataset SMPP data
    (6M total parts, tens of thousands of real multipart groups) - stuck
    for 10+ minutes before being killed. This version does one sort plus a
    handful of groupby-aggregate calls instead of iterating groups in
    Python; the only per-group Python-level work left is the text join
    (pandas has no vectorized "concatenate strings in order per group"
    primitive), and only over the already-small multipart subset, not
    every row.
    """
    if df.empty:
        return df.drop(columns=["concat_ref", "concat_total_parts", "concat_part_num"])

    group_key = (
        df["originator"].astype(str) + "|" +
        df["destination"].astype(str) + "|" +
        df["concat_ref"].astype("Int64").astype(str) + "|" +
        df["concat_total_parts"].astype("Int64").astype(str)
    )

    # ONE sort for the whole frame (not per-group) so "first row per group"
    # below is the smallest concat_part_num, and the text join below is
    # emitted in ascending part order.
    sort_idx = pd.DataFrame(
        {"group_key": group_key, "part_num": df["concat_part_num"]}
    ).sort_values(["group_key", "part_num"], kind="mergesort").index
    df = df.loc[sort_idx]
    group_key = group_key.loc[sort_idx]

    g = df.groupby(group_key, sort=False)

    total_parts = g["concat_total_parts"].first().astype(int)
    n_unique = g["concat_part_num"].nunique()
    n_seen = g["concat_part_num"].count()
    min_part = g["concat_part_num"].min()
    max_part = g["concat_part_num"].max()
    # `total_parts` DISTINCT part numbers spanning exactly [1, total_parts]
    # can only be {1..total_parts} itself (pigeonhole) - vectorized
    # equivalent of the old per-group
    # `sorted(part_nums) == list(range(1, total_parts+1))` check (missing
    # part, duplicate part, or extra part all fail one of these four).
    complete = (
        (n_unique == total_parts) & (n_seen == total_parts) &
        (min_part == 1) & (max_part == total_parts)
    )

    evaluated_bool = df["rule_evaluated"].fillna(False).astype(bool)
    flagged_mask = (df["rule_flagged"] == True) & evaluated_bool
    evaluated = evaluated_bool.groupby(group_key, sort=False).any()
    flagged_any = flagged_mask.groupby(group_key, sort=False).any()
    fraud_if_flagged = df["fraud_type"].where(flagged_mask)

    is_first_in_group = ~group_key.duplicated()
    first_rows = (
        df.loc[is_first_in_group]
        .drop(columns=["concat_ref", "concat_total_parts", "concat_part_num"])
        .set_index(group_key.loc[is_first_in_group])
    )

    out = first_rows
    out["timestamp"] = g["timestamp"].min()  # earliest part - point-in-time correct
    out["text"] = g["text"].apply(lambda s: "".join(s.fillna("")))
    out["text_decode_failed"] = g["text_decode_failed"].any()
    out["message_part_count"] = g.size()
    out["message_expected_parts"] = total_parts
    out["message_partial"] = ~complete
    out["rule_evaluated"] = evaluated
    out["rule_flagged"] = flagged_any.where(evaluated)
    out["fraud_type"] = fraud_if_flagged.groupby(group_key, sort=False).first()

    return out.reset_index(drop=True)


def reassemble_messages(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in features.columns]
    if missing:
        raise ValueError(f"features is missing required column(s): {missing}")
    missing = [c for c in REQUIRED_LABEL_COLS if c not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required column(s): {missing}")

    df = features.merge(labels, on="record_id", how="left", validate="one_to_one")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    is_multi = df["concat_ref"].notna()
    single = _reassemble_single_part(df[~is_multi])
    if not is_multi.any():
        return single  # skip concat entirely - an empty multi-part frame's
                        # dtypes (esp. timestamp) don't always match single's
                        # cleanly, and there's nothing to add anyway.
    multi = _reassemble_multi_part(df[is_multi])
    if single.empty:
        return multi
    return pd.concat([single, multi], ignore_index=True)


def run_reassembly(features_dir: Path, labels_dir: Path, out_path: Path) -> pd.DataFrame:
    feature_files = sorted(Path(features_dir).glob("*.csv"))
    if not feature_files:
        print(f"No feature CSVs found under {features_dir}")
        return pd.DataFrame()

    print(f"Loading {len(feature_files)} feature file(s)...")
    features = pd.concat([load_features_csv(f) for f in feature_files], ignore_index=True)

    label_files = sorted(Path(labels_dir).glob("*.csv"))
    print(f"Loading {len(label_files)} label file(s)...")
    # low_memory=False - same reason as ingestion/run_ingest.py's ingest_file():
    # columns like fraud_type/rule_flagged are mostly-null with mixed True/
    # False/NaN or string/NaN content, which pandas' chunked type-sniffing
    # can mis-infer differently across chunk boundaries (DtypeWarning) and
    # sometimes actually load inconsistently, not just warn.
    labels = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in label_files], ignore_index=True
    )

    print(f"Reassembling {len(features)} parts into logical messages...")
    messages = reassemble_messages(features, labels)
    multipart = int((messages["message_part_count"] > 1).sum())
    partial = int(messages["message_partial"].sum())
    print(f"-> {len(messages)} messages ({multipart} multipart, {partial} partial/incomplete)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    messages.to_csv(out_path, index=False)
    print(f"Wrote {len(messages)} rows to {out_path}")
    # Catches CSV write/round-trip corruption right here, same run - see
    # common/schemas.py's verify_csv_roundtrip() docstring for the real
    # bug (a 2.7M-row messages.csv that silently corrupted) this exists
    # because of.
    verify_csv_roundtrip(messages, out_path)
    print("Verified: reads back cleanly.")
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/processed/SMPP/features")
    parser.add_argument("--labels_dir", type=str, default="data/processed/SMPP/labels")
    parser.add_argument("--out_path", type=str, default="data/processed/SMPP/messages.csv")
    args = parser.parse_args()
    run_reassembly(Path(args.features_dir), Path(args.labels_dir), Path(args.out_path))


if __name__ == "__main__":
    main()
