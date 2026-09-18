"""
User Data Header (UDH) parsing, shared by ingestion/smpp.py and
ingestion/ss7.py - GSM 03.40 Information-Element structure detection,
independent of source.

Moved out of ingestion/smpp.py (where this originated, SMPP-only) once SS7
was found to carry the same structure on real data: `docs/smishing_detection_
plan.md` flagged "whether SS7's content bytes carry the same UDH IE structure
SMPP's do" as an open question; `notebooks/decode_verification.ipynb`
answered it - checking SS7 dcs=4 ("8-bit binary data" class per GSM 03.38,
not a text alphabet) rows byte-for-byte found 128/364 carrying a genuine
UDH with an Application Port Addressing IE (0x04/0x05, real destination
port 1234 observed), and once that header is stripped the remainder decodes
as legible ASCII - real recoverable content that ingestion/ss7.py was
previously auto-detecting a text codec against WITHOUT stripping first,
producing confident-looking gibberish instead.

Two IE families are extracted here:
  - Concatenation (0x00 8-bit ref / 0x08 16-bit ref) - part of a multipart
    message, used by ingestion/smpp.py (SS7 does NOT need this - it signals
    concatenation via its own sarref/msg_part/msg_parts columns instead,
    see ingestion/ss7.py's module docstring).
  - Application Port Addressing (0x04 8-bit port / 0x05 16-bit port) - marks
    the payload as addressed to a specific application (WAP Push, OTA,
    binary telemetry, etc.), used by both sources.
"""
from typing import NamedTuple

_CONCAT_IEI_8BIT_REF = 0x00   # 3-byte IE: ref(1) total(1) part(1)
_CONCAT_IEI_16BIT_REF = 0x08  # 4-byte IE: ref_hi(1) ref_lo(1) total(1) part(1)
_PORT_IEI_8BIT = 0x04         # 2-byte IE: dest_port(1) orig_port(1)
_PORT_IEI_16BIT = 0x05        # 4-byte IE: dest_hi(1) dest_lo(1) orig_hi(1) orig_lo(1)


class UdhInfo(NamedTuple):
    present: bool
    header_len: int                 # UDHL; 0 if no UDH
    concat_ref: int | None
    concat_total_parts: int | None
    concat_part_num: int | None
    dest_port: int | None           # Application Port Addressing IE's
                                     # destination port, if present


def parse_udh(raw: bytes) -> UdhInfo:
    """
    Detects a User Data Header from CONTENT BYTES ALONE - no external flag
    column needed - and, if present, extracts concatenation info and/or the
    application destination port when those IEs are present.

    Detection is structural: a UDH is a sequence of Information Elements
    (each iei+iel+data) that must exactly fill the declared UDHL byte -
    real message text coincidentally satisfying that for a full chain of
    IEs is effectively impossible. Verified against real SMPP op-4 data
    across 6 files/155k+ rows using the old gsm_features column as ground
    truth: 0 false negatives, ~0.03% false positive rate (spot-checked as
    unrelated data-quality quirks, not real text misdetected as a UDH).

    That 0.03% baseline was measured on GSM-7/ASCII/Latin1 content. UTF-16BE
    content (dcs=8) has a real, much higher false-positive rate against the
    "purely structural" version of this check: every other byte is 0x00 (or
    close to it) for BMP characters, so a chain of coincidental (iei, iel)
    pairs lines up far more easily than in single-byte text -
    `notebooks/decode_verification.ipynb`'s full-SS7-corpus run found real
    examples (a message starting with U+200E LRM, byte 0x20 0x0e; a Chinese
    message with an unrelated iei/iel coincidence) where a "header" was
    detected with NEITHER a recognized concat IE nor a recognized port IE -
    i.e. we accepted the structure but extracted nothing from it - and
    stripping that fake header left an ODD-length remainder, which
    correctly-implemented decode_utf16be then refused to decode at all
    (text=None, a spurious decode failure on perfectly legible real text).
    Fix: only accept a header that actually yields a recognized IE (concat
    or port) - the only two kinds this module (or anything downstream) ever
    reads anyway, so a "structurally valid but recognizes nothing" match is
    never useful and, per the above, is exactly the shape a false positive
    takes. Real concat-only and port-only headers (the SMPP multipart test
    fixture; the SS7 dcs=4 Application Port Addressing case) still pass,
    since those DO populate one of the two.
    """
    if len(raw) < 2:
        return UdhInfo(False, 0, None, None, None, None)
    udhl = raw[0]
    if udhl == 0 or 1 + udhl > len(raw):
        return UdhInfo(False, 0, None, None, None, None)

    pos, end = 1, 1 + udhl
    concat_ref = concat_total = concat_part = dest_port = None
    while pos < end:
        if pos + 2 > end:
            return UdhInfo(False, 0, None, None, None, None)  # malformed
        iei, iel = raw[pos], raw[pos + 1]
        data_start = pos + 2
        if data_start + iel > end:
            return UdhInfo(False, 0, None, None, None, None)
        if iei == _CONCAT_IEI_8BIT_REF and iel == 3:
            concat_ref = raw[data_start]
            concat_total, concat_part = raw[data_start + 1], raw[data_start + 2]
        elif iei == _CONCAT_IEI_16BIT_REF and iel == 4:
            concat_ref = (raw[data_start] << 8) | raw[data_start + 1]
            concat_total, concat_part = raw[data_start + 2], raw[data_start + 3]
        elif iei == _PORT_IEI_8BIT and iel == 2:
            dest_port = raw[data_start]
        elif iei == _PORT_IEI_16BIT and iel == 4:
            dest_port = (raw[data_start] << 8) | raw[data_start + 1]
        pos = data_start + iel

    if pos != end:
        return UdhInfo(False, 0, None, None, None, None)  # IEs didn't exactly fill udhl
    if concat_ref is None and dest_port is None:
        # Structurally valid but recognized nothing - see the false-positive
        # note above. Not a real UDH match.
        return UdhInfo(False, 0, None, None, None, None)
    return UdhInfo(True, udhl, concat_ref, concat_total, concat_part, dest_port)


def strip_udh(payload: bytes, udh: UdhInfo) -> bytes:
    """Drop the User Data Header (framing bytes) if present."""
    if not udh.present or len(payload) == 0:
        return payload
    return payload[1 + udh.header_len:]
