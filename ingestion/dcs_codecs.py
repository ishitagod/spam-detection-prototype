"""
DCS (Data Coding Scheme) -> text codec dispatch, shared by ingestion/smpp.py
and ingestion/ss7.py.

Two real findings drove this design (verified against real bytes, not
assumed - see the git history / conversation for the actual samples):

1. The same DCS byte does NOT imply the same on-the-wire storage convention
   across sources. GSM-7-tagged content:
     - SMPP stores it already unpacked (1 byte/char) - septet-unpacking a
       real op-4 payload (dcs=241, UDH-stripped) produces garbage; plain
       latin-1 decode produces perfect legible text.
     - SS7 stores true packed 7-bit septets - latin-1/no-unpack on a real
       sample produces garbage (15.8% printable); proper septet-unpack
       produces clean legible English ("Good Morning Cherima ").
   So "gsm7" is NOT one function shared across sources - SMPP_CODEC_FUNCS
   and SS7_CODEC_FUNCS point "gsm7" at different decoders on purpose.

2. The source table this module implements has a genuine conflict for SS7:
   DCS 1 and DCS 3 are listed under BOTH "US-ASCII/Latin-1 (both sources)"
   and "GSM-7 (SS7 only)". Rather than pick one globally, ambiguous codes
   are resolved per-row by trying every candidate codec and keeping
   whichever produces the most legible (highest printable-ratio) text -
   see decode_by_dcs(). This is also the fallback for any DCS value not in
   the table at all.

DCS values below are UNSIGNED - callers normalize sign (SMPP stores dcs as
a signed byte, SS7 appears to already store it unsigned) before calling in
here. This module has no opinion on that, it's a source-specific quirk.

3. Decoded text is sanitized (embedded \r/\n collapsed to a space) before
   being returned - see _sanitize_for_storage(). Real messages legitimately
   decode WITH raw newlines (GSM 03.38 maps 0x0A/0x0D to '\n'/'\r' - see
   _GSM7_BASIC), that's correct decoding, not a bug. But it's real-world-
   confirmed fragile once that text goes through a CSV round-trip at scale:
   a genuine SS7 multi-line message decoded fine and round-tripped fine
   through pandas in its OWN per-file features CSV (~90K rows), then broke
   ("ParserError: EOF inside string") reading it back from the ~2.7M-row
   concatenated messages.csv features/message_reassembly.py produces -
   same string content, only the file scale differed. Rather than chase
   the exact pandas/CSV large-file quirk, decoded text never contains a
   raw newline in the first place - sidesteps the whole bug class.
"""
import re

# ---------------------------------------------------------------------------
# GSM 03.38 default alphabet + septet unpacking
# ---------------------------------------------------------------------------

# GSM 03.38 default alphabet, basic (single-septet) table. Codes with no
# defined character (reserved) are simply absent - _decode_gsm7_codes()
# renders those as a placeholder so they get penalized by _printable_score
# instead of silently faking a character.
_GSM7_BASIC = {
    0x00: '@', 0x01: '£', 0x02: '$', 0x03: '¥', 0x04: 'è', 0x05: 'é',
    0x06: 'ù', 0x07: 'ì', 0x08: 'ò', 0x09: 'Ç', 0x0A: '\n', 0x0B: 'Ø',
    0x0C: 'ø', 0x0D: '\r', 0x0E: 'Å', 0x0F: 'å',
    0x10: 'Δ', 0x11: '_', 0x12: 'Φ', 0x13: 'Γ', 0x14: 'Λ', 0x15: 'Ω',
    0x16: 'Π', 0x17: 'Ψ', 0x18: 'Σ', 0x19: 'Θ', 0x1A: 'Ξ',
    # 0x1B is the escape-to-extension-table code, handled separately.
    0x1C: 'Æ', 0x1D: 'æ', 0x1E: 'ß', 0x1F: 'É',
    0x20: ' ', 0x21: '!', 0x22: '"', 0x23: '#', 0x24: '¤', 0x25: '%',
    0x26: '&', 0x27: "'", 0x28: '(', 0x29: ')', 0x2A: '*', 0x2B: '+',
    0x2C: ',', 0x2D: '-', 0x2E: '.', 0x2F: '/',
    0x3A: ':', 0x3B: ';', 0x3C: '<', 0x3D: '=', 0x3E: '>', 0x3F: '?',
    0x40: '¡',
    0x5B: 'Ä', 0x5C: 'Ö', 0x5D: 'Ñ', 0x5E: 'Ü', 0x5F: '§',
    0x60: '¿',
    0x7B: 'ä', 0x7C: 'ö', 0x7D: 'ñ', 0x7E: 'ü', 0x7F: 'à',
}
for _c in range(0x30, 0x3A):  # digits 0-9
    _GSM7_BASIC[_c] = chr(_c)
for _c in range(0x41, 0x5B):  # A-Z
    _GSM7_BASIC[_c] = chr(_c)
for _c in range(0x61, 0x7B):  # a-z
    _GSM7_BASIC[_c] = chr(_c)

# GSM 03.38 extension table (reachable via escape code 0x1B) - only the
# common subset (page break, currency, brackets, common punctuation). An
# escape byte not followed by one of these is rendered as a placeholder,
# same as any other undefined code.
_GSM7_EXT = {
    0x0A: '\x0c', 0x14: '^', 0x28: '{', 0x29: '}', 0x2F: '\\', 0x3C: '[',
    0x3D: '~', 0x3E: ']', 0x40: '|', 0x65: '€',
}

_UNMAPPED = '�'  # placeholder for any code with no defined character -
                       # deliberately excluded from the "printable" count in
                       # _printable_score so a bad decode can't score well
                       # just because U+FFFD itself is technically printable.


def _unpack_septets(payload: bytes) -> list[int]:
    """LSB-first bit unpacking of GSM 7-bit-packed octets into septet
    values. Trailing bits that don't fill a full septet are padding,
    discarded (not a character)."""
    bits = ''.join(f'{b:08b}'[::-1] for b in payload)
    n = len(bits) - len(bits) % 7
    return [int(bits[i:i + 7][::-1], 2) for i in range(0, n, 7)]


def _decode_gsm7_codes(codes: list[int]) -> str:
    chars = []
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == 0x1B and i + 1 < len(codes):  # extension escape
            chars.append(_GSM7_EXT.get(codes[i + 1], _UNMAPPED))
            i += 2
        else:
            chars.append(_GSM7_BASIC.get(c, _UNMAPPED))
            i += 1
    return ''.join(chars)


# ---------------------------------------------------------------------------
# Per-codec decoders - each takes raw payload bytes (UDH already stripped by
# the caller) and returns decoded text, or None if that codec flatly cannot
# apply (wrong byte length, invalid bytes) - never raises.
# ---------------------------------------------------------------------------

def decode_gsm7_packed(payload: bytes) -> str | None:
    """True GSM 7-bit packed-septet decode - SS7's storage convention for
    GSM-7 content (see module docstring). NOT what SMPP's 'gsm7' DCS bucket
    needs - SMPP_CODEC_FUNCS points 'gsm7' at decode_latin1 instead."""
    if not payload:
        return None
    return _decode_gsm7_codes(_unpack_septets(payload))


def decode_ascii(payload: bytes) -> str | None:
    """Strict - any byte outside 7-bit ASCII fails the whole decode, rather
    than silently accepting it the way latin-1 would. Content tagged as a
    US-ASCII DCS containing high-bit bytes is a real anomaly worth
    surfacing (text_decode_failed), not something to paper over."""
    if not payload:
        return None
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError:
        return None


def decode_latin1(payload: bytes) -> str | None:
    """1 byte = 1 char, never raises. This is also SMPP's actual decode for
    GSM-7-tagged content - see module docstring."""
    if not payload:
        return None
    return payload.decode("latin-1")


def decode_utf16be(payload: bytes) -> str | None:
    if not payload or len(payload) % 2 != 0:
        return None
    try:
        return payload.decode("utf-16-be")
    except UnicodeDecodeError:
        return None


def _sanitize_for_storage(text: str | None) -> str | None:
    """Collapse embedded \\r\\n / \\r / \\n into a single space - see the
    module docstring for why this exists (real, correctly-decoded text can
    contain these; storing them raw in a CSV field is what's fragile, not
    the decode). Runs of whitespace this creates are left as-is - this is
    a minimal fix for the newline-in-CSV problem specifically, not a
    general text-normalization pass."""
    if text is None:
        return None
    return re.sub(r"\r\n|\r|\n", " ", text)


def _printable_score(text: str | None) -> float:
    """Fraction of characters that are printable, excluding the unmapped-
    code placeholder so a bad GSM-7 decode full of undefined codes can't
    score well just because U+FFFD is technically a printable character."""
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() and ch != _UNMAPPED)
    return good / len(text)


# ---------------------------------------------------------------------------
# Per-source DCS -> codec-name tables, built directly from the source table
# (see module docstring for the SS7 1/3 conflict, resolved via auto-detect).
# ---------------------------------------------------------------------------

SMPP_DCS_TABLE: dict[int, str] = {
    0: "gsm7", 4: "gsm7", 241: "gsm7", 245: "gsm7",
    1: "ascii", 26: "ascii", 244: "ascii",
    8: "utf16", 24: "utf16", 25: "utf16",
    3: "latin1",
}

SS7_DCS_TABLE: dict[int, str] = {
    0: "gsm7", 12: "gsm7", 16: "gsm7", 17: "gsm7", 18: "gsm7",
    192: "gsm7", 241: "gsm7", 242: "gsm7", 245: "gsm7",
    8: "utf16", 24: "utf16", 25: "utf16",
    26: "ascii", 244: "ascii",
    # 1 and 3 deliberately absent - see SS7_AMBIGUOUS_DCS below.
}

# DCS values the source table lists under more than one codec for SS7.
# Candidates are tried in this order and the best-scoring result wins (see
# decode_by_dcs) - order only matters for tie-breaks.
SS7_AMBIGUOUS_DCS: dict[int, tuple[str, ...]] = {
    1: ("gsm7", "ascii"),
    3: ("gsm7", "latin1"),
}

SMPP_CODEC_FUNCS = {
    "gsm7": decode_latin1,   # see module docstring - SMPP stores GSM-7
                              # content pre-unpacked, not packed septets
    "ascii": decode_ascii,
    "utf16": decode_utf16be,
    "latin1": decode_latin1,
}

SS7_CODEC_FUNCS = {
    "gsm7": decode_gsm7_packed,  # true packed septets - see module docstring
    "ascii": decode_ascii,
    "utf16": decode_utf16be,
    "latin1": decode_latin1,
}

_TABLES = {"SMPP": SMPP_DCS_TABLE, "SS7": SS7_DCS_TABLE}
_AMBIGUOUS = {"SMPP": {}, "SS7": SS7_AMBIGUOUS_DCS}
_FUNCS = {"SMPP": SMPP_CODEC_FUNCS, "SS7": SS7_CODEC_FUNCS}


def _best_of(payload: bytes, codec_names, funcs: dict) -> tuple[str | None, str | None]:
    best_text, best_codec, best_score = None, None, -1.0
    for name in codec_names:
        text = funcs[name](payload)
        score = _printable_score(text)
        if score > best_score:
            best_text, best_codec, best_score = text, name, score
    return best_text, best_codec


def decode_by_dcs(payload: bytes, dcs: int | None, *, source: str) -> tuple[str | None, str | None]:
    """
    Decode `payload` (UDH already stripped) for the given unsigned `dcs`
    byte and `source` ("SMPP" | "SS7"). Returns (text, codec_used):
      - text is None if nothing decodable.
      - codec_used names which codec actually produced `text` - "gsm7" /
        "ascii" / "utf16" / "latin1" for a table hit, or one of those
        suffixed "(auto)" when resolved by legibility-scoring an ambiguous
        or unmapped DCS value, so callers/analysts can audit how often
        auto-detect fires and which way it resolves.
    """
    if source not in _TABLES:
        raise ValueError(f"unknown source {source!r} - expected 'SMPP' or 'SS7'")
    if not payload:
        return None, None

    table, ambiguous, funcs = _TABLES[source], _AMBIGUOUS[source], _FUNCS[source]
    dcs_int = int(dcs) if dcs is not None else None

    if dcs_int in ambiguous:
        text, codec = _best_of(payload, ambiguous[dcs_int], funcs)
        codec_used = f"{codec}(auto)" if codec else None
    elif dcs_int in table:
        codec = table[dcs_int]
        text, codec_used = funcs[codec](payload), codec
    else:
        # Unmapped DCS - don't guess a single default, score every codec
        # this source supports and keep the most legible result.
        text, codec = _best_of(payload, funcs.keys(), funcs)
        codec_used = f"{codec}(auto)" if codec else None

    # Scoring above (_best_of/_printable_score) runs on the RAW decoded
    # text on purpose - sanitizing first would penalize a legitimately
    # multi-line message's score for no reason. Only the final returned
    # text is sanitized - see module docstring / _sanitize_for_storage.
    return _sanitize_for_storage(text), codec_used
