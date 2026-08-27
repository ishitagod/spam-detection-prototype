"""
pytest suite for ingestion.ss7.clean() + ingestion.ss7.map_to_canonical().
Synthetic rows shaped like the real raw SS7 columns - doesn't touch
data/raw/, runs regardless of whether the real CDR files are present.

Run:
    pytest tests/test_ss7_ingestion.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.ss7 import clean, map_to_canonical

# Real UTF-16BE payload for "Hi" (dcs=8/UCS-2), used across rows that need
# decodable content.
HI_HEX = "00480069"

BASE_ROW = {
    "index": 0, "time_stamp": "2026-08-19T10:00:00", "message_type": 3,
    "reference": 1000, "calling_gt": "60120000001", "smsc": "6012000000",
    "msisdn": "60120000002", "b_number": "60120000002", "dcs": 8,
    "sarref": None, "msg_part": None, "msg_parts": None,
    "content": HI_HEX, "raw_user_data": HI_HEX,
    "decision": None, "rule": None, "fraud_type": None, "status": None,
    "create_date": "2026-08-19T10:00:01", "update_date": "2026-08-19T10:00:01",
    "file_name": "x.csv", "decoded_content": "Hi",
    "called_gt": "60120000002", "imsi": 502122000000001,
    "virtual_imsi": None, "vlr_address": None,
    "virtual_vlr_outbound_smsc_gt": None,
    "ton": 1, "npi": 1, "pid": 0, "tpdu_length": 2,
    "error1": None, "error2": None, "delivery_code": None, "action_code": None,
}


def row(**overrides) -> dict:
    return {**BASE_ROW, **overrides}


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    return pd.DataFrame([
        row(index=1, message_type=3, reference=1, calling_gt="MO_SENDER"),  # MO - standalone
        row(index=2, message_type=5, reference=2, content=None, decoded_content=None),  # SRI_request - dropped
        row(  # SRI_response for virtual_imsi=555 - carries vlr_address, no content
            index=3, message_type=1, reference=3, content=None, decoded_content=None,
            virtual_imsi=555, vlr_address="VLR_A", time_stamp="2026-08-19T10:00:00",
        ),
        row(  # MT_request matching virtual_imsi=555 - should get vlr_address="VLR_A" merged in
            index=4, message_type=2, reference=4, virtual_imsi=555, vlr_address=None,
        ),
        row(index=5, message_type=4, reference=5, content=None, decoded_content=None),  # MT_response - dropped
        row(  # MT_request with NO matching SRI_response - vlr_address stays NA, not dropped
            index=6, message_type=2, reference=6, virtual_imsi=999, vlr_address=None,
        ),
        row(  # a SECOND, later SRI_response for the SAME virtual_imsi=555 (re-query) -
              # its vlr_address should win over the earlier one
            index=7, message_type=1, reference=7, content=None, decoded_content=None,
            virtual_imsi=555, vlr_address="VLR_B", time_stamp="2026-08-19T10:05:00",
        ),
    ])


@pytest.fixture
def cleaned(raw_rows) -> pd.DataFrame:
    return clean(raw_rows)


@pytest.fixture
def mapped(cleaned):
    return map_to_canonical(cleaned)


@pytest.fixture
def features(mapped):
    return mapped[0]


@pytest.fixture
def labels(mapped):
    return mapped[1]


def test_mo_and_mt_request_kept_sri_and_response_dropped(cleaned):
    """Only MO (index 1) and the two MT_request rows (index 4, 6) survive -
    SRI_request (2), SRI_response (3, 7 - merged in not kept as own row),
    MT_response (5) are all dropped."""
    assert sorted(cleaned["index"].tolist()) == [1, 4, 6]


def test_mt_request_gets_vlr_address_merged_from_matching_sri_response(cleaned):
    mt_req = cleaned.loc[cleaned["index"] == 4].iloc[0]
    assert mt_req["vlr_address"] == "VLR_B"  # the LATER of the two SRI_response rows


def test_mt_request_with_no_matching_sri_response_keeps_na_vlr_address(cleaned):
    """virtual_imsi=999 has no SRI_response row at all - must not crash,
    must not fabricate a value."""
    mt_req = cleaned.loc[cleaned["index"] == 6].iloc[0]
    assert pd.isna(mt_req["vlr_address"])


def test_mo_row_has_na_vlr_address_not_a_join_miss(cleaned):
    mo = cleaned.loc[cleaned["index"] == 1].iloc[0]
    assert pd.isna(mo["vlr_address"])


def test_content_decoded_via_dcs_not_trusted_decoded_content_column(cleaned):
    mt_req = cleaned.loc[cleaned["index"] == 4].iloc[0]
    assert mt_req["text_clean"] == "Hi"


def test_label_source_columns_never_leak_into_features(features):
    assert "decision" not in features.columns
    assert "rule" not in features.columns
    assert "fraud_type" not in features.columns
    assert "status" not in features.columns


def test_record_id_joins_features_to_labels(features, labels):
    assert (features["record_id"] == labels["record_id"]).all()
    assert features["record_id"].is_unique


def test_raises_when_clean_not_run_first():
    with pytest.raises(ValueError):
        map_to_canonical(pd.DataFrame([{"index": 1}]))


# ---------------------------------------------------------------------------
# concat_ref/concat_total_parts/concat_part_num (native SS7 sarref/msg_part/
# msg_parts, not UDH - see module docstring's MULTIPART note)
# ---------------------------------------------------------------------------

def test_single_part_row_gets_normalized_concat_fields():
    """msg_parts null/0/1 all mean genuinely single-part - same convention
    as SMPP's UDH-sourced concat fields."""
    raw = pd.DataFrame([row(index=1, message_type=3, msg_parts=0, msg_part=None, sarref=None)])
    cleaned = clean(raw)
    r = cleaned.iloc[0]
    assert r["concat_total_parts"] == 1
    assert r["concat_part_num"] == 1
    assert pd.isna(r["concat_ref"])


def test_multipart_row_uses_sarref_msg_part_msg_parts():
    raw = pd.DataFrame([
        row(index=1, message_type=3, sarref=231.0, msg_part=4, msg_parts=6),
    ])
    cleaned = clean(raw)
    r = cleaned.iloc[0]
    assert r["concat_ref"] == 231.0
    assert r["concat_total_parts"] == 6
    assert r["concat_part_num"] == 4


# ---------------------------------------------------------------------------
# delivery-response rows (message_type in (1, 4, 7), decision != 0) - not
# real SRI/MT traffic, must not survive clean() or feed the vlr_address
# lookup.
# ---------------------------------------------------------------------------

def test_delivery_response_sri_response_row_dropped_and_not_merged():
    """A message_type=1 row with decision != 0 is a delivery-response
    record misusing the SRI_response type, not a real routing lookup - it
    must not be used to fill vlr_address on the matching MT_request."""
    raw = pd.DataFrame([
        row(  # looks like an SRI_response for virtual_imsi=555, but decision != 0
            index=1, message_type=1, reference=1, content=None, decoded_content=None,
            virtual_imsi=555, vlr_address="BOGUS_VLR", decision=1,
        ),
        row(index=2, message_type=2, reference=2, virtual_imsi=555, vlr_address=None),
    ])
    cleaned = clean(raw)
    assert cleaned["index"].tolist() == [2]
    assert pd.isna(cleaned.iloc[0]["vlr_address"])


def test_delivery_response_message_type_4_and_7_dropped():
    raw = pd.DataFrame([
        row(index=1, message_type=3, reference=1),  # MO, kept
        row(index=2, message_type=4, reference=2, decision=1, content=None, decoded_content=None),
        row(index=3, message_type=7, reference=3, decision=1, content=None, decoded_content=None),
    ])
    cleaned = clean(raw)
    assert cleaned["index"].tolist() == [1]


def test_message_type_1_4_7_with_decision_zero_not_treated_as_delivery_response():
    """decision == 0 on these types is NOT a delivery response - a
    genuine SRI_response with decision=0 must still feed the vlr_address
    merge as before."""
    raw = pd.DataFrame([
        row(
            index=1, message_type=1, reference=1, content=None, decoded_content=None,
            virtual_imsi=555, vlr_address="VLR_A", decision=0,
        ),
        row(index=2, message_type=2, reference=2, virtual_imsi=555, vlr_address=None),
    ])
    cleaned = clean(raw)
    assert cleaned.iloc[0]["vlr_address"] == "VLR_A"


# ---------------------------------------------------------------------------
# Schema validation tripwire - see ingestion/ss7.py's REQUIRED_FEATURE_COLS
# comment and tests/test_smpp_ingestion.py's matching test.
# ---------------------------------------------------------------------------

def test_map_to_canonical_raises_if_a_feature_mapping_breaks(cleaned, monkeypatch):
    import ingestion.ss7 as ss7_module

    broken_map = dict(ss7_module.FEATURE_MAP)
    del broken_map["message_id"]
    monkeypatch.setattr(ss7_module, "FEATURE_MAP", broken_map)

    with pytest.raises(ValueError, match="message_id"):
        ss7_module.map_to_canonical(cleaned)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
