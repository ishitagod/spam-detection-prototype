"""
Maps one live SMPP/SS7 request (serving/schemas.py) to a single canonical
row - the same originator/destination/text/timestamp/dcs/text_decode_failed
shape common/schemas.py's CANONICAL_FEATURE_SCHEMA defines for batch
training, just for one message instead of a whole file.

Reuses ingestion/smpp.py's and ingestion/ss7.py's own DCS-decode + UDH-strip
helpers (`_decode_row`) rather than reimplementing that logic here - this
is exactly the kind of thing that quietly drifts from the batch pipeline if
duplicated (real historical bug: the old `decoded_content` column leaking
UDH framing bytes - see ingestion/smpp.py's _decode_row docstring). Two
scoring runs must decode a byte-identical payload the same way regardless
of whether it came through pipeline.py or this live endpoint.

SCOPE: only the columns rule_pattern_score actually needs (see
serving/scoring.py) - `source`, `originator`, `text`, `dcs`,
`text_decode_failed`, plus `record_id`/`timestamp` for bookkeeping/Feast
lookups. Does not attempt every canonical column (e.g. SS7's `vlr_address`
needs the SRI-response join batch ingestion does, not meaningful for a
single live message and not consumed by rule_pattern_score anyway).
`imsi` is the one exception carried through despite not being a
CANONICAL_FEATURE_SCHEMA column itself - it's the Feast lookup KEY for
imsi_distinct_originators_1hr (serving/feature_lookup.py's
get_imsi_features()), SS7-only, always None for SMPP (no IMSI concept).
"""
from dataclasses import dataclass

import pandas as pd

from ingestion.smpp import _decode_row as _smpp_decode_row
from ingestion.ss7 import _decode_row as _ss7_decode_row
from serving.schemas import SMPPTransaction, SS7Transaction


@dataclass
class CanonicalRow:
    source: str  # "SMPP" | "SS7"
    record_id: str
    originator: str
    destination: str | None
    text: str
    timestamp: str
    dcs: float | None
    text_decode_failed: bool
    imsi: str | None = None  # SS7-only, see module docstring

    @property
    def sender_id(self) -> str:
        """Compound key matching feature_repo/definitions.py's `sender_id`
        entity ('source|originator') - SMPP business sender IDs and SS7
        MSISDNs are different namespaces that could coincidentally
        collide on the same string, so `source` is always part of the key."""
        return f"{self.source}|{self.originator}"


def _hex_to_bytes(content: str) -> str:
    """Request payloads carry space-separated hex byte pairs
    ("d9 77 5d ..."), unlike the batch CDR's unspaced hex string
    ingestion/*.py's `_decode_row` expects - strip whitespace, nothing
    else (the decoders below already tolerate a malformed/odd-length
    result by returning no text rather than raising)."""
    return "".join((content or "").split())


def map_ss7_transaction(txn: SS7Transaction) -> CanonicalRow:
    """SS7 stores `dcs` unsigned already (no sign-correction needed,
    unlike SMPP - see ingestion/dcs_codecs.py's module docstring). SS7
    signals MULTIPART concatenation via sarref/msg_part/msg_parts, not UDH
    - but content bytes can still carry a UDH for other reasons (e.g.
    Application Port Addressing on binary-data-class DCS values - see
    ingestion/ss7.py's module docstring), so ss7._decode_row still detects
    and strips a UDH when present, same as SMPP."""
    dcs = int(txn.dcs) if txn.dcs not in (None, "") else None
    decoded = _ss7_decode_row(_hex_to_bytes(txn.content), dcs)
    text = decoded["text"] or ""
    timestamp = pd.Timestamp(txn.time_stamp, unit="ms").isoformat()

    return CanonicalRow(
        source="SS7",
        record_id=str(txn.reference),
        originator=str(txn.calling_gt),
        destination=str(txn.called_gt) if txn.called_gt is not None else None,
        text=text,
        timestamp=timestamp,
        dcs=float(dcs) if dcs is not None else None,
        text_decode_failed=not text.strip(),
        imsi=str(txn.imsi) if txn.imsi not in (None, "") else None,
    )


def map_smpp_transaction(txn: SMPPTransaction) -> CanonicalRow:
    """originator/destination come from a_number/b_number (this live wire
    shape's actual sender/recipient fields), not ingestion/smpp.py's
    batch-CDR `oa`/`da` column names - same canonical columns either way.
    dcs is sign-corrected (`% 256`) before decoding, mirroring
    ingestion/smpp.py's clean() - smpp._decode_row mods again internally,
    which is idempotent, not a double-correction bug."""
    dcs_raw = int(txn.dcs) if txn.dcs not in (None, "") else None
    dcs = (dcs_raw % 256) if dcs_raw is not None else None
    decoded = _smpp_decode_row(_hex_to_bytes(txn.content), dcs)
    text = decoded["text"] or ""

    return CanonicalRow(
        source="SMPP",
        record_id=str(txn.reference),
        originator=str(txn.a_number),
        destination=str(txn.b_number) if txn.b_number is not None else None,
        text=text,
        timestamp=txn.time_stamp,  # already ISO 8601 - canonical timestamp
        # is left as a raw string at this stage, same as batch ingestion
        # (see common/schemas.py's CANONICAL_FEATURE_SCHEMA comment)
        dcs=float(dcs) if dcs is not None else None,
        text_decode_failed=not text.strip(),
    )
