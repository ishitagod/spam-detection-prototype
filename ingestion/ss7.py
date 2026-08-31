"""
SS7 row cleaning + raw-column-to-canonical mapping.

ROW-FILTERING RULE (the thing an earlier draft of this module flagged as
"not yet confirmed"): SS7's message_type splits into two logical flows -
  MO (mobile-originated, message_type=3): a single row, self-contained,
    real message content already on it, no further phases.
  MT (mobile-terminated): FOUR message_type codes across one delivery -
    SRI_request(5) -> SRI_response(1) -> MT_request(2) -> MT_response(4).
    Checked against real data (message_type value counts + non-null
    `content`/`vlr_address` counts per type, one file):
      MT_SRI_request   82,622 rows - 0 content, 0 vlr_address  (pure trigger)
      MT_SRI_response  82,615 rows - 0 content, 72,434 vlr_address (routing
                                       lookup RESULT - resolved serving VLR,
                                       populated NOWHERE else)
      MT_request        9,074 rows - 9,072 content, 0 vlr_address (the
                                       actual message)
      MT_response       6,693 rows - 0 content, 0 vlr_address  (delivery ack)
    So only MO and MT_request carry real content; MT_SRI_request and
    MT_response carry neither content nor a join key and are DROPPED here
    (same role as SMPP's ack PDUs in ingestion/smpp.py's clean()).
    MT_SRI_response is dropped as its OWN row too, but not discarded - its
    vlr_address is merged into the matching MT_request row first, via
    `virtual_imsi` - checked against real data: 8,399/8,400 MT_request
    virtual_imsi values also appear among MT_SRI_response's (99.99% match),
    a real and reliable join key, not a coincidence. MO rows have no SRI
    phase and no virtual_imsi to join on - their vlr_address is NA, which
    is the correct/expected state for that source flow, not missing data.

MULTIPART: SS7 signals concatenation via its own native fields (`sarref`,
`msg_part`, `msg_parts`) - checked on real data, genuinely populated
(3,899/221,211 rows in the first file checked, msg_parts up to 18, on BOTH
MO and MT_request) unlike SMPP's SAR fields which were almost always
empty. clean() maps these into the same concat_ref/concat_total_parts/
concat_part_num columns ingestion/smpp.py produces (from UDH instead), so
features/message_reassembly.py works unchanged across both sources -
msg_parts of 0 or 1 (or missing) means genuinely single-part, matching
SMPP's convention (concat_total_parts=1, concat_part_num=1, concat_ref=None).
"""
import pandas as pd

from common.schemas import (
    CANONICAL_FEATURE_SCHEMA,
    CANONICAL_LABEL_SCHEMA,
    validate_features,
    validate_labels,
)
from ingestion.dcs_codecs import decode_by_dcs
from labels.rule_labels import build_rule_labels, is_rule_evaluated

# message_type meanings (given directly, not yet independently verified
# against real content/flow patterns the way the DCS codec tables were -
# worth confirming against real data before relying on this for anything
# beyond documentation). Two logical flows - see module docstring:
#   MT (mobile-terminated): SRI query/response, then the actual delivery
#     request/response - four message_type codes across that one flow.
#   MO (mobile-originated): a single message_type code, no separate
#     request/response split.
MESSAGE_TYPE_MEANING: dict[int, str] = {
    5: "MT_SRI_request",
    1: "MT_SRI_response",
    2: "MT_request",
    4: "MT_response",
    3: "MO",
}
_MT_SRI_RESPONSE = 1
_MT_REQUEST = 2
_MO = 3

# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------


def _decode_row(content_hex, dcs) -> dict:
    """
    Decodes SS7 `content` directly via dcs_codecs.decode_by_dcs(source=
    "SS7"), rather than trusting the upstream `decoded_content` column -
    mirrors ingestion/smpp.py's approach, though the underlying reason
    differs: SMPP's decoded_content was found to leak UDH framing bytes
    (a real bug); SS7's decoded_content MATCHED decode_by_dcs exactly on a
    real multipart sample checked (dcs=8/UCS-2, msg_parts=6) - decoding
    independently here is about having one auditable, source-consistent
    codec path (decode_by_dcs's returned codec name), not about fixing a
    known bug the way SMPP's rewrite was.

    NOTE: unlike SMPP, no UDH-stripping is applied - checked a real
    multipart SS7 row byte-for-byte and `content` has no embedded header/
    framing prefix. SS7 signals concatenation via its own sarref/msg_part/
    msg_parts columns instead (see clean()'s concat_ref/concat_total_parts/
    concat_part_num and the module docstring's MULTIPART note), not via
    UDH-in-content the way SMPP does.
    """
    if not isinstance(content_hex, str) or not content_hex:
        return {"text": None}
    try:
        payload = bytes.fromhex(content_hex)
    except ValueError:
        return {"text": None}
    dcs_int = int(dcs) if pd.notna(dcs) else None
    text, _codec_used = decode_by_dcs(payload, dcs_int, source="SS7")
    return {"text": text}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps MO and MT_request rows (the only ones with real content - see
    module docstring), enriching MT_request with vlr_address merged in
    from the matching MT_SRI_response row via virtual_imsi. Drops
    MT_SRI_request and MT_response entirely (no content, no join key).

    Also decodes `text_clean`/`text_decode_failed` here (not in
    map_to_canonical) - mirrors ingestion/smpp.py's clean(), which keeps
    undecodable rows rather than dropping them, same reasoning: inference
    can't skip a message it can't decode, so training data shouldn't
    either.
    """
    # message_type in (1, 4, 7) with decision != 0 are delivery-response
    # rows, not real SRI/MT traffic - drop before anything else. type
    # 4/7 rows never reach `kept` regardless (only MO/MT_request do), but
    # type 1 (MT_SRI_response) rows feed sri_lookup below - a decision != 0
    # row there is a delivery-response record misusing the SRI_response
    # message_type, not a genuine routing lookup, and would otherwise merge
    # a bogus vlr_address into the matching MT_request row.
    # decision blank/NaN means "unspecified/never reached a rule" (see
    # ingestion/smpp.py's decision-semantics comment) - not a delivery
    # response, so only an explicit non-zero decision counts here.
    decision = pd.to_numeric(df.get("decision"), errors="coerce")
    delivery_response = df["message_type"].isin([1, 4, 7]) & decision.notna() & (decision != 0)
    if delivery_response.any():
        print(
            f"  clean (SS7): dropping {delivery_response.sum()} delivery-response "
            "row(s) (message_type in (1,4,7), decision != 0)"
        )
    df = df[~delivery_response]

    sri_response = df[df["message_type"] == _MT_SRI_RESPONSE]
    # Sort by time before dedup so "keep last" means "most recent lookup",
    # not input-order-dependent. A handful of virtual_imsi values repeat
    # across multiple SRI_response rows (72,431 unique / 72,434 rows in the
    # file checked) - re-queries, presumably; the latest one is the routing
    # decision that was actually still in effect.
    sri_lookup = (
        sri_response[["virtual_imsi", "vlr_address", "time_stamp"]]
        .dropna(subset=["virtual_imsi"])
        .sort_values("time_stamp", kind="mergesort")
        .drop_duplicates(subset="virtual_imsi", keep="last")
        [["virtual_imsi", "vlr_address"]]
    )

    # MT_request rows already HAVE a `vlr_address` column of their own
    # (always null - confirmed against real data, see module docstring) -
    # drop it before merging sri_lookup's vlr_address in, or pandas would
    # silently suffix both to vlr_address_x/vlr_address_y instead of
    # raising, leaving the column this code actually reads on empty.
    mt_request = df[df["message_type"] == _MT_REQUEST].drop(
        columns="vlr_address", errors="ignore"
    ).merge(sri_lookup, on="virtual_imsi", how="left", validate="many_to_one")
    mo = df[df["message_type"] == _MO].copy()
    # no SRI phase for MO - NA is the correct state here, not a join miss.
    # Matches mt_request's (post-merge) vlr_address dtype explicitly rather
    # than assigning pd.NA directly - an all-NA column's dtype is otherwise
    # ambiguous at concat time (pandas FutureWarning: dtype inference for
    # all-NA columns during concat is changing).
    mo["vlr_address"] = pd.Series(index=mo.index, dtype=mt_request["vlr_address"].dtype)

    kept = pd.concat([mo, mt_request], ignore_index=True)

    decoded = kept.apply(
        lambda r: _decode_row(r.get("content"), r.get("dcs")), axis=1, result_type="expand"
    )
    kept["text_clean"] = decoded["text"].fillna(kept.get("decoded_content"))
    text_missing = kept["text_clean"].isna() | (
        kept["text_clean"].fillna("").str.strip() == ""
    )
    if text_missing.any():
        print(
            f"  clean (SS7): {text_missing.sum()} MO/MT_request row(s) have no "
            "decodable text - kept (not dropped), flagged via text_decode_failed"
        )
    kept["text_decode_failed"] = text_missing
    kept["text_clean"] = kept["text_clean"].where(~text_missing, "")

    # Multipart grouping key, in the same concat_ref/concat_total_parts/
    # concat_part_num shape ingestion/smpp.py produces (from UDH instead of
    # SS7's native sarref/msg_part/msg_parts) - see module docstring's
    # MULTIPART note. msg_parts of 0/1/missing means genuinely single-part.
    msg_parts = pd.to_numeric(kept.get("msg_parts"), errors="coerce")
    single_part = msg_parts.isna() | (msg_parts <= 1)
    kept["concat_ref"] = pd.to_numeric(kept.get("sarref"), errors="coerce").where(~single_part)
    kept["concat_total_parts"] = msg_parts.where(~single_part, 1)
    kept["concat_part_num"] = pd.to_numeric(kept.get("msg_part"), errors="coerce").where(~single_part, 1)

    kept["record_id"] = kept["index"].astype(str)
    return kept


# ---------------------------------------------------------------------------
# map_to_canonical()
# ---------------------------------------------------------------------------

FEATURE_MAP = {
    "originator": "calling_gt",
    "destination": "called_gt",        # NOTE: was `b_number` in an earlier
                                         # draft of this mapping - switched
                                         # to called_gt per the current field
                                         # list. `b_number` and `msisdn` both
                                         # still exist as raw columns, unused
                                         # here now.
    "text": "text_clean",               # reconstructed in clean() above
    "timestamp": "time_stamp",
    "create_date": "create_date",       # kept as its OWN feature, distinct
                                         # from timestamp/time_stamp - exact
                                         # semantic difference between the
                                         # two not yet confirmed.
    "message_id": "reference",
    "smsc": "smsc",
    "imsi": "imsi",
    "virtual_imsi": "virtual_imsi",
    "vlr_address": "vlr_address",       # roaming/location signal - merged
                                         # in from MT_SRI_response in
                                         # clean(); NA for MO rows (no SRI
                                         # phase, expected, not missing).
    "dcs": "dcs",                       # SS7 stores this unsigned already
                                         # (unlike SMPP) - see
                                         # ingestion/dcs_codecs.py's docstring.
    "message_type": "message_type",     # see MESSAGE_TYPE_MEANING above
    "ton": "ton",
    "npi": "npi",
    "pid": "pid",
    "tpdu_length": "tpdu_length",
    "concat_ref": "concat_ref",              # multipart grouping key - see
    "concat_total_parts": "concat_total_parts",  # clean()'s docstring /
    "concat_part_num": "concat_part_num",        # module docstring's
                                                   # MULTIPART note. Sourced
                                                   # from sarref/msg_part/
                                                   # msg_parts, same shape as
                                                   # SMPP's UDH-sourced
                                                   # columns, so
                                                   # features/message_reassembly.py
                                                   # works unchanged.
}

# Checked against real data and deliberately NOT included:
#   virtual_vlr_outbound_smsc_gt   -> entirely NULL across every message_type
#                                      in the one file checked so far -
#                                      dropped (per-project decision;
#                                      unconfirmed whether it's populated in
#                                      other files, revisit if so).
#   b_number, msisdn                -> superseded by called_gt/calling_gt
#                                      per the current field list.
#   sarref, msg_part, msg_parts     -> superseded by concat_ref/
#                                      concat_total_parts/concat_part_num,
#                                      which already fold these in (see
#                                      clean()) - keeping the raw columns
#                                      too would just be a redundant
#                                      duplicate of the same signal (same
#                                      reasoning as SMPP's sar_ref/
#                                      sar_msg_parts/sar_msg_part).

# Schema validation, wired into map_to_canonical() below - see
# ingestion/smpp.py's REQUIRED_FEATURE_COLS comment for why this exists.
# Unlike SMPP, SS7 DOES populate message_id (FEATURE_MAP["message_id"] =
# "reference") - so features require the full schema, minus `esm_class`
# (SMPP-only - ESME class is an SMPP PDU field, no SS7 equivalent - see
# common/schemas.py's CANONICAL_FEATURE_SCHEMA comment; mirrors how
# ingestion/smpp.py excludes message_id from ITS required list). Labels
# still exclude message_id: label_source only ever sets record_id, for
# both sources (see map_to_canonical() below), never message_id.
REQUIRED_FEATURE_COLS = [c for c in CANONICAL_FEATURE_SCHEMA if c != "esm_class"]
REQUIRED_LABEL_COLS = [c for c in CANONICAL_LABEL_SCHEMA if c != "message_id"]

# See ingestion/smpp.py's LABEL_SOURCE_COLS docstring - same purpose here:
# raw ingredients for labels/rule_labels.py only, never returned as-is, and
# NEVER read by the unsupervised layer - fraud_type/decision/rule/status
# are the rule engine's OWN output. The unsupervised anomaly layer has to
# find spam from behavioral/content signal alone, not from what the rule
# engine already decided - handing it fraud_type would just make it a
# worse copy of the rule engine instead of a layer that catches what the
# rules miss (see CLAUDE.md's rule_pattern vs anomaly_score split).
LABEL_SOURCE_COLS = ["decision", "rule", "fraud_type", "status"]


def map_to_canonical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features_df, label_source_df) - kept separate on purpose.

    Expects `df` to already be cleaned via clean() (MO/MT_request only,
    vlr_address merged in, `record_id` assigned).
    """
    missing_required = [
        c for c in ("record_id", "text_clean", "text_decode_failed")
        if c not in df.columns
    ]
    if missing_required:
        raise ValueError(
            f"df is missing {missing_required} - run ss7.clean() on the raw "
            "dataframe before mapping it to canonical."
        )

    features = pd.DataFrame({
        canonical: df[raw] for canonical, raw in FEATURE_MAP.items()
        if raw in df.columns
    })
    features["record_id"] = df["record_id"]
    features["source"] = "SS7"
    features["text_decode_failed"] = df["text_decode_failed"]

    raw_label_cols = df[[c for c in LABEL_SOURCE_COLS if c in df.columns]].copy()
    label_source = pd.DataFrame({"record_id": df["record_id"]})
    if "fraud_type" in raw_label_cols.columns:
        label_source["fraud_type"] = raw_label_cols["fraud_type"]
    label_source["rule_evaluated"] = is_rule_evaluated(raw_label_cols)
    label_source["rule_flagged"] = build_rule_labels(raw_label_cols).where(
        label_source["rule_evaluated"]
    )

    validate_features(features, required=REQUIRED_FEATURE_COLS)
    validate_labels(label_source, required=REQUIRED_LABEL_COLS)
    return features, label_source
