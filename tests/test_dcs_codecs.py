"""
pytest suite for ingestion.dcs_codecs.

Both real-data samples below are pulled from the actual investigation (not
invented): the SMPP one from a real op-4 row (data/raw/SMPP/0208/
stg_smpp_20260802_0500.csv, dcs=-15 / unsigned 241), the SS7 one supplied
directly against a real SS7 `content` value to resolve the DCS 1/3
ambiguity. Both were decoded multiple candidate ways by hand first to
establish which one is actually correct before writing these as
regression tests - see ingestion/dcs_codecs.py's module docstring.

Run:
    pytest tests/test_dcs_codecs.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.dcs_codecs import (
    decode_ascii,
    decode_by_dcs,
    decode_gsm7_packed,
    decode_latin1,
    decode_utf16be,
)

# Real op-4 SMPP payload (UDH already stripped), dcs unsigned 241 -> "gsm7"
# per SMPP_DCS_TABLE. SMPP stores GSM-7 content pre-unpacked (1 byte/char),
# NOT packed septets - this is the fixture that proves it.
SMPP_GSM7_PAYLOAD = bytes.fromhex(
    "524d302041434f4d2053444e204248443a204869204d5548414d4d41442041464649"
    "512041494d414e2042494e2048414d44414e2c20796f75722072657061796d656e74"
    "2044554520697320544f4441592e"
)

# Real SS7 `content` value, no UDH involved - true GSM-7 packed septets.
SS7_GSM7_PACKED_PAYLOAD = bytes.fromhex(
    "c7f79b0c6abee5eeb4fb0c1aa2cbf2743b0c02"
)


# ---------------------------------------------------------------------------
# Individual codecs
# ---------------------------------------------------------------------------

def test_decode_latin1_is_byte_per_char_no_unpack():
    assert decode_latin1(SMPP_GSM7_PAYLOAD) == (
        "RM0 ACOM SDN BHD: Hi MUHAMMAD AFFIQ AIMAN BIN HAMDAN, "
        "your repayment DUE is TODAY."
    )


def test_decode_gsm7_packed_unpacks_septets_and_maps_default_alphabet():
    assert decode_gsm7_packed(SS7_GSM7_PACKED_PAYLOAD) == "Good Morning Cherima "


def test_decode_gsm7_packed_on_already_unpacked_bytes_is_garbage():
    """The whole reason SMPP and SS7 need DIFFERENT 'gsm7' codec functions:
    septet-unpacking SMPP's already-unpacked content does NOT recover
    legible text - it produces a different, wrong string (GSM 03.38's
    alphabet remaps several codes to accented/Greek characters, so a naive
    ASCII-printable-ratio check isn't a reliable "is this garbage" signal
    here - asserting it diverges from the known-correct decode is)."""
    text = decode_gsm7_packed(SMPP_GSM7_PAYLOAD)
    assert not text.startswith("RM0 ACOM SDN BHD")


def test_decode_ascii_rejects_high_bit_bytes():
    assert decode_ascii(b"hello") == "hello"
    assert decode_ascii(bytes([0x68, 0x80, 0x69])) is None


def test_decode_utf16be_rejects_odd_length():
    assert decode_utf16be("hi".encode("utf-16-be")) == "hi"
    assert decode_utf16be(b"\x00h\x00") is None


def test_empty_payload_returns_none_for_every_codec():
    assert decode_latin1(b"") is None
    assert decode_gsm7_packed(b"") is None
    assert decode_ascii(b"") is None
    assert decode_utf16be(b"") is None


# ---------------------------------------------------------------------------
# decode_by_dcs() dispatch
# ---------------------------------------------------------------------------

def test_smpp_gsm7_dcs_decodes_via_latin1_not_septet_unpack():
    text, codec = decode_by_dcs(SMPP_GSM7_PAYLOAD, 241, source="SMPP")
    assert codec == "gsm7"
    assert text.startswith("RM0 ACOM SDN BHD")


def test_smpp_ascii_dcs_is_strict():
    text, codec = decode_by_dcs(bytes([0x68, 0x69, 0x80]), 1, source="SMPP")
    assert codec == "ascii"
    assert text is None  # 0x80 is not valid ASCII - surfaced, not absorbed


def test_ss7_unambiguous_gsm7_dcs_uses_packed_septet_decode():
    text, codec = decode_by_dcs(SS7_GSM7_PACKED_PAYLOAD, 192, source="SS7")
    assert codec == "gsm7"
    assert text == "Good Morning Cherima "


def test_ss7_ambiguous_dcs1_autodetects_gsm7_on_the_real_sample():
    """This is the concrete resolution of the DCS 1/3 conflict: for THIS
    real content, scored auto-detect picks gsm7 over ascii, and produces
    legible text - not a global hardcoded assumption."""
    text, codec = decode_by_dcs(SS7_GSM7_PACKED_PAYLOAD, 1, source="SS7")
    assert codec == "gsm7(auto)"
    assert text == "Good Morning Cherima "


def test_ss7_ambiguous_dcs3_autodetects_gsm7_on_the_real_sample():
    text, codec = decode_by_dcs(SS7_GSM7_PACKED_PAYLOAD, 3, source="SS7")
    assert codec == "gsm7(auto)"
    assert text == "Good Morning Cherima "


def test_ss7_ambiguous_dcs_picks_ascii_when_that_is_actually_more_legible():
    """Auto-detect must not always favor gsm7 - prove it picks the OTHER
    candidate when the bytes are genuinely plain ASCII instead."""
    text, codec = decode_by_dcs(b"Hello World", 1, source="SS7")
    assert codec == "ascii(auto)"
    assert text == "Hello World"


def test_unmapped_dcs_falls_back_to_scored_autodetect():
    text, codec = decode_by_dcs(SS7_GSM7_PACKED_PAYLOAD, 99, source="SS7")
    assert codec == "gsm7(auto)"
    assert text == "Good Morning Cherima "


def test_empty_payload_returns_none_none_regardless_of_dcs():
    assert decode_by_dcs(b"", 0, source="SMPP") == (None, None)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        decode_by_dcs(b"hi", 0, source="XYZ")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
