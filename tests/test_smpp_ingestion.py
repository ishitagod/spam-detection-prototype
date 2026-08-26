"""
pytest suite for ingestion.smpp.clean() + ingestion.smpp.map_to_canonical().
One behavior per test, using small synthetic rows shaped like the REAL raw
SMPP columns - doesn't touch anything under data/raw/, so it runs in
milliseconds regardless of whether the real CDR files are present.

Run:
    pytest tests/test_smpp_ingestion.py -v
    pytest tests/ -v                        # whole test suite
    pytest tests/test_smpp_ingestion.py::test_keeps_only_op4_rows   # one test
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.smpp import clean, map_to_canonical

# This exact hex/dcs/gsm_features combo is a real row pulled from
# data/raw/SMPP/0208/stg_smpp_20260802_0500.csv - used to verify UDH-stripping
# and the dcs sign-fix against real bytes, not an invented example.
UDH_PREFIXED_HEX = (
    "050003390201524d302041434f4d2053444e204248443a204869204d5548414d4d41"
    "442041464649512041494d414e2042494e2048414d44414e2c20796f75722072657061"
    "796d656e742044554520697320544f4441592e"
)

# One row with every column the real CSVs have, sane defaults. Individual
# tests override only the fields the scenario actually cares about, via
# row(**overrides) - keeps each test's intent visible without repeating the
# full 30+ column dict every time.
BASE_ROW = {
    "index": 0, "time_stamp": "2026-08-19T10:00:00",
    "result": "ok", "source_ip": "1.2.3.4", "source_port": 2775,
    "dest_ip": "5.6.7.8", "dest_port": 2775, "system_id": "AGG01",
    "instance_id": "i1", "virtual_gt": "vgt1", "sequence_no": 100,
    "message_id": None, "oa_ton": 5, "oa_npi": 0, "oa": "SPX001",
    "da_ton": 1, "da_npi": 1, "da": "9198765001", "dcs": 0,
    "sar_ref": "", "sar_msg_parts": None, "sar_msg_part": None,
    "app_dest_port": "", "app_src_port": "", "decision": None,
    "content": None, "decoded_content": None,
    "rule": None, "fraud_type": None,
    "esme_class": 0, "msg_type": "text", "gsm_features": 0,
    "messaging_mode": "store_forward", "inverted": False,
    "create_date": "2026-08-19", "message_state": "delivered",
    "receipted_message_id": "", "rule_name": None,
}


def row(**overrides) -> dict:
    return {**BASE_ROW, **overrides}


# ---------------------------------------------------------------------------
# Fixtures - pytest builds these once per test that asks for them (by
# argument name) and tears them back down after. `cleaned`/`mapped` depend
# on `raw_rows`, so every test gets a fresh, independent pipeline run - no
# state leaks between tests even though they share the same input data.
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_rows() -> pd.DataFrame:
    return pd.DataFrame([
        row(  # real op-4 submit_sm, UDH-prefixed concatenated message,
              # rule fired and flagged it (decision=1) - the labelled,
              # positive case. Note the rule prefix is deliberately NOT
              # "SW_" here - a whitelist-prefixed rule would (correctly)
              # NOT count as evaluated, see test_rule_labels.py.
            index=1, smpp_operation=4, dcs=-15, gsm_features=1,
            sar_ref="57", sar_msg_parts=2, sar_msg_part=1,
            decision=1, content=UDH_PREFIXED_HEX,
            decoded_content="�@�9$�RM0 ACOM SDN BHD: ...",  # upstream's garbled version
            rule="R017_burst_duplicate", fraud_type="spam",  # real data only
                                       # ever shows "spam" here, never a
                                       # variant like "spam_burst" - see
                                       # labels/rule_labels.py
            rule_name="burst_duplicate_v3",
        ),
        row(  # ack PDU (submit_sm_resp) - no content, no decision, only
              # message_id - not signal, must be dropped entirely
            index=2, smpp_operation=80000004, oa_ton=None, oa_npi=None,
            oa=None, da_ton=None, da_npi=None, da=None, dcs=None,
            message_id="m1", gsm_features="", esme_class="", msg_type=None,
            messaging_mode=None,
        ),
        row(  # op-4, but no rule ever touched it - the unlabelled pool
            index=3, smpp_operation=4, virtual_gt="vgt2", sequence_no=101,
            oa="SPX002", da="9198765002",
            content="48656c6c6f20576f726c64",  # hex for "Hello World"
            decoded_content="Hello World",
        ),
        row(  # op-4, genuinely undecodable text (bad hex, no decoded_content
              # fallback either) - must be kept, not dropped
            index=4, smpp_operation=4, virtual_gt="vgt3", sequence_no=102,
            oa="SPX003", da="9198765003",
            content="not-valid-hex", decoded_content=None,
        ),
    ])


@pytest.fixture
def cleaned(raw_rows) -> pd.DataFrame:
    return clean(raw_rows)


@pytest.fixture
def mapped(cleaned) -> tuple[pd.DataFrame, pd.DataFrame]:
    return map_to_canonical(cleaned)


@pytest.fixture
def features(mapped) -> pd.DataFrame:
    return mapped[0]


@pytest.fixture
def labels(mapped) -> pd.DataFrame:
    return mapped[1]


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------

def test_keeps_only_op4_rows(cleaned):
    """Ack PDUs (op 80000004/5) carry no content or decision - not signal, dropped."""
    assert len(cleaned) == 3


def test_strips_udh_header_from_text(cleaned):
    text = cleaned.loc[cleaned["index"] == 1, "text_clean"].iloc[0]
    assert text.startswith("RM0 ACOM SDN BHD"), f"UDH not stripped: {text!r}"


def test_corrects_dcs_sign(cleaned):
    dcs = cleaned.loc[cleaned["index"] == 1, "dcs"].iloc[0]
    assert dcs == 241, f"expected unsigned 241 (0xF1), got {dcs}"


def test_keeps_undecodable_text_rows_instead_of_dropping(cleaned):
    """Inference can't skip a message it can't decode, so training data
    shouldn't get to skip it either - row is kept, flagged, text is ""
    (not dropped, not garbage, not NaN)."""
    undecodable = cleaned.loc[cleaned["index"] == 4].iloc[0]
    assert undecodable["text_decode_failed"]
    assert undecodable["text_clean"] == ""


# ---------------------------------------------------------------------------
# map_to_canonical()
# ---------------------------------------------------------------------------

def test_label_source_columns_never_leak_into_features(features):
    """decision/rule/rule_name/fraud_type are the rule engine's OUTPUT - if
    these ever end up in `features`, a supervised model would be handed the
    answer directly as an input."""
    assert "decision" not in features.columns
    assert "rule" not in features.columns
    assert "rule_name" not in features.columns
    assert "fraud_type" not in features.columns


def test_record_id_joins_features_to_labels(features, labels):
    """message_id is null on every op-4 row, so record_id is the only valid
    join key back from labels to features."""
    assert (features["record_id"] == labels["record_id"]).all()


def test_text_decode_failed_is_a_feature_not_just_bookkeeping(features):
    assert "text_decode_failed" in features.columns


def test_labels_output_has_no_raw_rule_engine_columns(labels):
    """rule/rule_name/decision are ingredients for computing
    rule_evaluated/rule_flagged inside map_to_canonical(), not part of the
    output - the model only ever needs the two computed columns, never the
    literal rule that fired or the raw decision code."""
    assert "rule" not in labels.columns
    assert "rule_name" not in labels.columns
    assert "decision" not in labels.columns
    assert "rule_evaluated" in labels.columns
    assert "rule_flagged" in labels.columns


def test_rule_evaluated_and_flagged_computed_end_to_end(labels):
    """index 1 (rule fired, decision=1) is the only evaluated+flagged row;
    index 3/4 (no rule ever touched them) are unevaluated with rule_flagged
    left NA, not fabricated as False."""
    row1 = labels.iloc[0]
    assert row1["rule_evaluated"] == True
    assert row1["rule_flagged"] == True

    unevaluated = labels.iloc[1:]
    assert (unevaluated["rule_evaluated"] == False).all()
    assert unevaluated["rule_flagged"].isna().all()


# ---------------------------------------------------------------------------
# Schema validation tripwire - proves map_to_canonical() actually catches
# column drift, not just that it doesn't false-positive on the real shape
# above. See ingestion/smpp.py's REQUIRED_FEATURE_COLS comment.
# ---------------------------------------------------------------------------

def test_map_to_canonical_raises_if_a_feature_mapping_breaks(cleaned, monkeypatch):
    """Simulates a FEATURE_MAP edit that silently drops a required raw
    column (e.g. a typo'd source column name) - map_to_canonical() must
    fail loudly here, not hand back a features frame that's silently
    missing a canonical column."""
    import ingestion.smpp as smpp_module

    broken_map = dict(smpp_module.FEATURE_MAP)
    del broken_map["originator"]
    monkeypatch.setattr(smpp_module, "FEATURE_MAP", broken_map)

    with pytest.raises(ValueError, match="originator"):
        smpp_module.map_to_canonical(cleaned)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
