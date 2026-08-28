"""
pytest suite for common.schemas - validate_features(), validate_labels(),
and verify_csv_roundtrip(). Previously untested directly (only exercised
indirectly through ingestion/smpp.py and ingestion/ss7.py's own tests) -
this covers the module in isolation, including the failure paths those
indirect tests don't reach (e.g. a genuinely corrupted CSV on disk).

Run:
    pytest tests/test_schemas.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.schemas import validate_features, validate_labels, verify_csv_roundtrip


# ---------------------------------------------------------------------------
# validate_features() / validate_labels()
# ---------------------------------------------------------------------------

def test_validate_features_passes_when_required_columns_present():
    df = pd.DataFrame({"record_id": [1], "source": ["SMPP"], "text": ["hi"]})
    validate_features(df, required=["record_id", "source", "text"])  # no raise


def test_validate_features_raises_and_names_missing_columns():
    df = pd.DataFrame({"record_id": [1]})
    with pytest.raises(ValueError, match="source"):
        validate_features(df, required=["record_id", "source", "text"])


def test_validate_labels_raises_and_names_missing_columns():
    df = pd.DataFrame({"record_id": [1]})
    with pytest.raises(ValueError, match="rule_flagged"):
        validate_labels(df, required=["record_id", "rule_flagged"])


def test_validate_features_default_required_is_full_canonical_schema():
    """No `required` passed -> every canonical key is required. This is
    the strict default the docstring documents - callers are expected to
    pass their own subset (see ingestion/smpp.py's REQUIRED_FEATURE_COLS),
    not rely on this catching everything correctly on its own."""
    df = pd.DataFrame({"record_id": [1]})
    with pytest.raises(ValueError):
        validate_features(df)


# ---------------------------------------------------------------------------
# verify_csv_roundtrip()
# ---------------------------------------------------------------------------

def test_verify_csv_roundtrip_passes_for_a_clean_write(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "text": ["hi", "bye", "multi\nline"]})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    verify_csv_roundtrip(df, path)  # no raise


def test_verify_csv_roundtrip_raises_on_row_count_mismatch(tmp_path):
    """Simulates a truncated/corrupted write - the exact failure mode a
    real bug hit (features/message_reassembly.py's messages.csv silently
    lost rows to a ParserError downstream, days after being written)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "truncated.csv"
    df.to_csv(path, index=False)
    # Truncate the file mid-write, simulating corruption.
    with open(path, "r+b") as f:
        f.truncate(f.tell() + 4)

    with pytest.raises(ValueError, match="round-trip"):
        verify_csv_roundtrip(df, path)


def test_verify_csv_roundtrip_raises_with_path_in_message_on_unparseable_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "unparseable.csv"
    # A genuinely broken CSV: an opened quote that's never closed - the
    # same "EOF inside string" failure mode the real bug hit.
    path.write_text('a\n"unterminated', encoding="utf-8")

    with pytest.raises(ValueError, match=r"unparseable\.csv"):
        verify_csv_roundtrip(df, path)


def test_verify_csv_roundtrip_only_materializes_one_column(tmp_path):
    """Real bug: reading every column of a multi-column file just to count
    rows OOM'd on a real 5.5M-row, ~20-column SMPP write. usecols=[0] must
    still catch row-count corruption on a WIDE frame, not just narrow
    single-column test fixtures - corruption confined to a later column
    (not column 0) must still surface, since the C parser tokenizes every
    field of every row regardless of which columns get materialized."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})
    path = tmp_path / "wide.csv"
    df.to_csv(path, index=False)
    verify_csv_roundtrip(df, path)  # no raise - clean wide write

    # Corrupt it: an unterminated quote in column "b" (not column "a"/index 0).
    path.write_text('a,b,c\n1,"unterminated,1.0\n2,y,2.0\n3,z,3.0\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"wide\.csv"):
        verify_csv_roundtrip(df, path)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
