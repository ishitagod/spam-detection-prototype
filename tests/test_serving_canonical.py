"""
pytest suite for serving/canonical.py's SS7/SMPP -> canonical-row mapping.
Uses the two sample payloads this module was built against (see the
conversation/PR this came from) so the DCS-decode reuse (ingestion/smpp.py,
ingestion/ss7.py) is exercised against real-shaped request bodies, not just
unit-level dcs/content pairs.

Run:
    pytest tests/test_serving_canonical.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.canonical import map_smpp_transaction, map_ss7_transaction
from serving.schemas import SMPPTransaction, SS7Transaction


def test_ss7_maps_originator_destination_and_source():
    txn = SS7Transaction(
        reference="34993520154",
        time_stamp=1683038843499,
        message_type=3,
        calling_gt="60944123210002",
        called_gt="911234500002",
        a_number="99645634751",
        b_number="222",
        content="d9  77  5d  0e  7a  52  a1  a0  f4  1c  54  a3  e1  64  b1  19",
        dcs="0",
        pid="0",
        imsi="1040023547528",
        service_centre_address="944123210003",
        service_centre_time_stamp="23-05-02 20:17:23",
        tpdu_length=43,
        sarref=115,
        msg_part=1,
        msg_parts=2,
    )
    row = map_ss7_transaction(txn)

    assert row.source == "SS7"
    assert row.record_id == "34993520154"
    assert row.originator == "60944123210002"  # calling_gt, not a_number
    assert row.destination == "911234500002"  # called_gt, not b_number
    assert row.dcs == 0.0
    assert row.sender_id == "SS7|60944123210002"
    assert isinstance(row.text, str)
    # 16 content bytes decode to *something* under dcs=0's gsm7 codec -
    # real assertion is that decoding didn't silently fail, exact text
    # isn't asserted (packed-septet GSM-7 on arbitrary bytes, not a
    # constructed legible message like the SMPP case below).
    assert row.text_decode_failed == (row.text.strip() == "")


def test_ss7_epoch_millis_timestamp_converts_to_iso():
    txn = SS7Transaction(
        reference="r1", time_stamp=1683038843499, message_type=3,
        calling_gt="1", called_gt="2", content="00", dcs="0",
    )
    row = map_ss7_transaction(txn)
    assert row.timestamp.startswith("2023-05-02")


def test_smpp_maps_a_number_b_number_and_decodes_ascii_content():
    txn = SMPPTransaction(
        reference="34993520433",
        time_stamp="2026-08-20T10:47:38.000Z",
        smpp_operation="4",
        system_id="ESME_SYS_01",
        source_ip="192.168.1.10",
        a_number="60123456789",
        oa_ton="1", oa_npi="1",
        b_number="60198765432",
        da_ton="1", da_npi="1",
        content="48 65 6c 6c 6f 20 57 6f 72 6c 64",  # "Hello World"
        dcs="0",
        esm_class="0", msg_type="0", gsm_features="0", messaging_mode="0",
    )
    row = map_smpp_transaction(txn)

    assert row.source == "SMPP"
    assert row.record_id == "34993520433"
    assert row.originator == "60123456789"  # a_number
    assert row.destination == "60198765432"  # b_number
    assert row.timestamp == "2026-08-20T10:47:38.000Z"
    assert row.dcs == 0.0
    assert row.sender_id == "SMPP|60123456789"
    # dcs=0 -> SMPP's "gsm7" bucket decodes as plain latin-1 (SMPP stores
    # GSM-7-tagged content pre-unpacked - ingestion/dcs_codecs.py) - these
    # bytes are exactly "Hello World" in ASCII/latin-1.
    assert row.text == "Hello World"
    assert row.text_decode_failed is False


def test_smpp_content_with_spaces_is_dehexed_correctly():
    txn = SMPPTransaction(
        reference="r1", time_stamp="2026-08-20T10:47:38.000Z", smpp_operation="4",
        a_number="1", b_number="2", content="41  42   43", dcs="0",
    )
    row = map_smpp_transaction(txn)
    assert row.text == "ABC"


def test_empty_content_is_flagged_not_decode_failed_silently():
    txn = SMPPTransaction(
        reference="r1", time_stamp="2026-08-20T10:47:38.000Z", smpp_operation="4",
        a_number="1", b_number="2", content="", dcs="0",
    )
    row = map_smpp_transaction(txn)
    assert row.text == ""
    assert row.text_decode_failed is True
