"""
SMPP ingestion: row cleaning + raw-column-to-canonical mapping, combined in
one module (unlike the old data_cleaning.py / cdr_mapping.py split) because
for a single source these two steps are tightly coupled - map_to_canonical
requires clean()'s output columns (record_id, text_clean, text_decode_failed)
to exist, and there was never a case where you'd use one without the other.
See ingestion/base.py for why the two are still kept as SEPARATE FUNCTIONS
registered together, rather than merged into one.
"""
from typing import NamedTuple

import pandas as pd

from config.settings import SMPP_SUBMIT_SM_OPERATION
from ingestion.dcs_codecs import decode_by_dcs
from labels.rule_labels import build_rule_labels, is_rule_evaluated

# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------

# Concatenation Information-Element IDs, per GSM 03.40 - the only two UDH
# IEs that carry ref/total-parts/part-number. Other real UDH IEs exist (e.g.
# 0x05, port addressing - seen on real OTA/WAP-push rows in this data) and
# are still detected+stripped for decoding, they just don't yield concat
# fields.
_CONCAT_IEI_8BIT_REF = 0x00   # 3-byte IE: ref(1) total(1) part(1)
_CONCAT_IEI_16BIT_REF = 0x08  # 4-byte IE: ref_hi(1) ref_lo(1) total(1) part(1)


class UdhInfo(NamedTuple):
    present: bool
    header_len: int                 # UDHL; 0 if no UDH
    concat_ref: int | None
    concat_total_parts: int | None
    concat_part_num: int | None


def _parse_udh(raw: bytes) -> UdhInfo:
    """
    Detects a User Data Header from CONTENT BYTES ALONE - no external flag
    column (the old `gsm_features`) needed - and, if present, extracts
    concatenation info when the header contains a concatenation IE.

    Detection is structural: a UDH is a sequence of Information Elements
    (each iei+iel+data) that must exactly fill the declared UDHL byte -
    real message text coincidentally satisfying that for a full chain of
    IEs is effectively impossible. Verified against real op-4 data across
    6 files/155k+ rows using the old gsm_features column as ground truth:
    0 false negatives (every gsm_features==1 row detected), a small false
    positive rate (~0.03%) on rows gsm_features said had no UDH - spot-
    checked and those were rows with unusual/garbled content, a data
    quality quirk unrelated to this detector, not real message text being
    misdetected.
    """
    if len(raw) < 2:
        return UdhInfo(False, 0, None, None, None)
    udhl = raw[0]
    if udhl == 0 or 1 + udhl > len(raw):
        return UdhInfo(False, 0, None, None, None)

    pos, end = 1, 1 + udhl
    concat_ref = concat_total = concat_part = None
    while pos < end:
        if pos + 2 > end:
            return UdhInfo(False, 0, None, None, None)  # malformed, not a real UDH
        iei, iel = raw[pos], raw[pos + 1]
        data_start = pos + 2
        if data_start + iel > end:
            return UdhInfo(False, 0, None, None, None)
        if iei == _CONCAT_IEI_8BIT_REF and iel == 3:
            concat_ref = raw[data_start]
            concat_total, concat_part = raw[data_start + 1], raw[data_start + 2]
        elif iei == _CONCAT_IEI_16BIT_REF and iel == 4:
            concat_ref = (raw[data_start] << 8) | raw[data_start + 1]
            concat_total, concat_part = raw[data_start + 2], raw[data_start + 3]
        pos = data_start + iel

    if pos != end:
        return UdhInfo(False, 0, None, None, None)  # IEs didn't exactly fill udhl
    return UdhInfo(True, udhl, concat_ref, concat_total, concat_part)


def _strip_udh(payload: bytes, udh: UdhInfo) -> bytes:
    """Drop the User Data Header (concatenated-SMS framing bytes) if present."""
    if not udh.present or len(payload) == 0:
        return payload
    return payload[1 + udh.header_len :]


def _decode_row(content_hex, dcs) -> dict:
    """
    Reconstructs SMS text directly from the raw PDU hex (`content`), rather
    than trusting the upstream `decoded_content` column, and returns the
    concatenation metadata alongside it in one pass (both need the same
    parsed UDH).

    Two real issues found by inspecting actual op-4 rows byte-for-byte:

    1. `decoded_content` does NOT strip the User Data Header on concatenated
       messages - the first `1 + UDHL` bytes of `content` are framing
       metadata (ref/total-parts/part-number), not text, and end up as a
       garbled prefix ahead of otherwise-correct text.
    2. `dcs` is stored as a SIGNED byte upstream (e.g. -15 instead of the
       correct unsigned 241 = 0xF1) - the codec table needs `dcs % 256`
       first to recover the real DCS byte.

    IMPORTANT - what this function does NOT fix, on purpose: rows that
    "look" unreadable in a terminal but are actually correctly-decoded
    non-English text (verified against dcs=8/UCS-2 rows containing genuine
    Chinese-language messages) are left alone. Re-decoding those would be
    wrong, not a fix - the point of the multilingual embedding step
    downstream is to handle exactly this content, not filter it out.

    Actual per-DCS-value codec choice lives in ingestion/dcs_codecs.py
    (shared with SS7) - see that module's docstring for why SMPP's "gsm7"
    bucket decodes as plain byte-per-char (not septet-unpacked) and why
    that's a source-specific storage convention, verified against a real
    payload, not a guess.
    """
    empty = {
        "text": None, "concat_ref": None,
        "concat_total_parts": None, "concat_part_num": None,
    }
    if not isinstance(content_hex, str) or not content_hex:
        return empty
    try:
        raw_bytes = bytes.fromhex(content_hex)
    except ValueError:
        return empty

    udh = _parse_udh(raw_bytes)
    payload = _strip_udh(raw_bytes, udh)

    dcs_byte = int(dcs) % 256 if pd.notna(dcs) else None
    text, _codec_used = decode_by_dcs(payload, dcs_byte, source="SMPP")

    return {
        "text": text,
        "concat_ref": udh.concat_ref,
        "concat_total_parts": udh.concat_total_parts,
        "concat_part_num": udh.concat_part_num,
    }


def decode_sms_text(content_hex, dcs) -> str | None:
    """Convenience wrapper around _decode_row() for callers that only need
    the decoded text (e.g. notebook exploration) - see _decode_row() for
    the concatenation-metadata-aware version clean() actually uses."""
    return _decode_row(content_hex, dcs)["text"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY smpp_operation == SMPP_SUBMIT_SM_OPERATION (submit_sm request
    rows) - the actual A2P message submission that passes through the
    network and is what's revenue-relevant for spam here.

    Rows with no decodable text are KEPT, not dropped: inference can't
    just refuse to score a message because its text didn't decode, so
    training data shouldn't get to skip that case either. Instead
    `text_clean` becomes "" and `text_decode_failed` is set True, so the
    model can see and learn from the failure itself (same pattern as the
    architecture plan's cold-start `confidence` disclosure - a structural
    gap gets surfaced as a feature, not silently dropped).

    Also assigns concat_ref/concat_total_parts/concat_part_num - the
    multipart grouping key features.smpp_reassembly (not yet built) needs
    to reconstruct one logical message out of N physical SMS parts. Two
    sources, preferred in this order:
      1. UDH concatenation IE, parsed from `content` bytes directly (see
         _parse_udh) - the mechanism actually populated in this data today
         (24k+/file).
      2. SAR fields (sar_ref/sar_msg_parts/sar_msg_part) - the PDU-level
         alternative mechanism, essentially unpopulated in this data so far
         (31/145k+ op-4 rows checked) but kept as a fallback for future
         traffic that may use it instead of UDH.
    Rows with neither are genuinely single-part messages: concat_total_parts
    = concat_part_num = 1, concat_ref = None (forms its own group of one).
    """
    df = df[df["smpp_operation"] == SMPP_SUBMIT_SM_OPERATION].copy()

    df["dcs"] = pd.to_numeric(df["dcs"], errors="coerce") % 256

    decoded = df.apply(
        lambda r: _decode_row(r.get("content"), r.get("dcs")), axis=1, result_type="expand"
    )
    df["text_clean"] = decoded["text"].fillna(df["decoded_content"])

    text_missing = df["text_clean"].isna() | (
        df["text_clean"].fillna("").str.strip() == ""
    )
    if text_missing.any():
        print(
            f"  clean (SMPP): {text_missing.sum()} op-4 row(s) have no decodable "
            "text - kept (not dropped), flagged via text_decode_failed"
        )
    df["text_decode_failed"] = text_missing
    df["text_clean"] = df["text_clean"].where(~text_missing, "")

    sar_ref = pd.to_numeric(df.get("sar_ref"), errors="coerce")
    sar_total = pd.to_numeric(df.get("sar_msg_parts"), errors="coerce")
    sar_part = pd.to_numeric(df.get("sar_msg_part"), errors="coerce")
    use_sar = decoded["concat_total_parts"].isna() & sar_total.notna()

    df["concat_ref"] = decoded["concat_ref"].where(~use_sar, sar_ref)
    df["concat_total_parts"] = decoded["concat_total_parts"].where(~use_sar, sar_total)
    df["concat_part_num"] = decoded["concat_part_num"].where(~use_sar, sar_part)

    single_part = df["concat_total_parts"].isna()
    df["concat_total_parts"] = df["concat_total_parts"].where(~single_part, 1)
    df["concat_part_num"] = df["concat_part_num"].where(~single_part, 1)
    # concat_ref stays None for single-part rows - reassembly must treat a
    # None ref as "forms its own group", never group two None-ref rows
    # together just because they share the same (missing) key.

    df["record_id"] = df["index"].astype(str)
    return df


# ---------------------------------------------------------------------------
# map_to_canonical()
# ---------------------------------------------------------------------------

FEATURE_MAP = {
    "originator": "oa",              # originating address (sender ID / MSISDN)
    "originator_ton": "oa_ton",      # type of number - business sender IDs
    "originator_npi": "oa_npi",      # vs. real MSISDNs often differ here
    "destination": "da",
    "text": "text_clean",            # reconstructed in clean() above
                                       # (UDH-stripped, DCS-correct) - NOT the raw
                                       # decoded_content column, which was found to
                                       # leak UDH framing bytes into the text
    "timestamp": "time_stamp",
    "system_id": "system_id",        # aggregator/binding identity - useful
                                       # as a secondary reputation key above
                                       # individual sender ID
    "dcs": "dcs",                     # data coding scheme - language signal
                                       # (sign-corrected in clean())
    "messaging_mode": "messaging_mode",
    "source_ip": "source_ip",        # clustering/grouping key - same-gateway
                                       # connection bursts
    "dest_ip": "dest_ip",
    "instance_id": "instance_id",    # connection/session identity - finer-
                                       # grained than system_id, groups
                                       # messages from the same bind session
    "virtual_gt": "virtual_gt",      # proxy/virtual identity behind the
                                       # displayed sender - if two `oa`
                                       # values share a virtual_gt, that's a
                                       # spoofing/sender-rotation signal a
                                       # per-sender-only feature would miss
    "concat_ref": "concat_ref",              # multipart grouping key - see
    "concat_total_parts": "concat_total_parts",  # clean()'s docstring.
    "concat_part_num": "concat_part_num",        # UDH-sourced when present
                                                   # (the mechanism actually
                                                   # populated in this data),
                                                   # SAR-sourced fallback
                                                   # otherwise, else 1/1/None
                                                   # (genuinely single-part).
                                                   # Superseded raw sar_ref/
                                                   # sar_msg_parts/sar_msg_part
                                                   # as features - concat_* is
                                                   # the unified, authoritative
                                                   # signal now.
}

# Checked against real op-4 rows and deliberately NOT included:
#   result, dest_port, inverted,
#   msg_type                       -> constant across all op-4 rows, no signal
#                                      (msg_type checked across two separate
#                                      files - 0.0 in both, 65965 rows total)
#   source_port                    -> ephemeral per-TCP-connection, no
#                                      behavioral meaning on its own
#   app_dest_port,
#   app_src_port, message_state,
#   receipted_message_id           -> entirely NULL on op-4 rows (populated
#                                      on other operation types only, same
#                                      trap as message_id - see clean() above)
#   esme_class                     -> dropped: not required (per-project
#                                      decision, not a data-quality finding)
#   sequence_no                    -> dropped: not required (per-project
#                                      decision, not a data-quality finding)
#   gsm_features                   -> dropped: replaced by content-byte UDH
#                                      detection (_parse_udh) - no longer
#                                      read anywhere, not even internally.
#   sar_ref, sar_msg_parts,
#   sar_msg_part                   -> superseded by concat_ref/
#                                      concat_total_parts/concat_part_num,
#                                      which already fold SAR in as a
#                                      fallback source (see clean()) -
#                                      keeping the raw sar_* columns too
#                                      would just be a redundant duplicate
#                                      of the same signal.

# These are the RULE ENGINE'S OUTPUT. Raw ingredients for labels/rule_labels.py
# only - never returned as-is, never a feature. See map_to_canonical().
LABEL_SOURCE_COLS = ["decision", "rule", "rule_name", "fraud_type"]


def map_to_canonical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features_df, label_source_df) - kept separate on purpose.

    Expects `df` to already be cleaned via clean() (op-4-only, `record_id`
    assigned) - `record_id` is the join key between the two returned
    frames, replacing `message_id` which is null on every row this function
    ever sees (only the ack rows clean() drops ever have it populated).
    """
    missing_required = [
        c for c in ("record_id", "text_clean", "text_decode_failed")
        if c not in df.columns
    ]
    if missing_required:
        raise ValueError(
            f"df is missing {missing_required} - run smpp.clean() on the raw "
            "dataframe before mapping it to canonical."
        )

    features = pd.DataFrame({
        canonical: df[raw] for canonical, raw in FEATURE_MAP.items()
        if raw in df.columns
    })
    features["record_id"] = df["record_id"]
    features["source"] = "SMPP"
    # Real feature, not bookkeeping - a message we couldn't decode text for
    # is itself a signal (and inference has to score it anyway, so training
    # data keeps it rather than dropping it - see clean()).
    features["text_decode_failed"] = df["text_decode_failed"]

    # LABEL_SOURCE_COLS are pulled in only as the RAW INGREDIENTS to compute
    # the two label columns below (is_rule_evaluated needs rule/rule_name,
    # build_rule_labels needs decision) - they are dropped again before
    # returning. The model never needs to know which literal rule fired or
    # what the raw decision code was, only whether a spam-pattern rule
    # evaluated this message and, if so, what it decided.
    raw_label_cols = df[[c for c in LABEL_SOURCE_COLS if c in df.columns]].copy()
    label_source = pd.DataFrame({"record_id": df["record_id"]})
    if "fraud_type" in raw_label_cols.columns:
        label_source["fraud_type"] = raw_label_cols["fraud_type"]
    label_source["rule_evaluated"] = is_rule_evaluated(raw_label_cols)
    label_source["rule_flagged"] = build_rule_labels(raw_label_cols).where(
        label_source["rule_evaluated"]
    )
    return features, label_source
