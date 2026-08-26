"""
pytest suite for features.faiss_index.compute_near_dup_features() and
compute_near_dup_features_chunked().

2D unit vectors are used throughout (not real 384-dim embeddings) - FAISS
inner-product search doesn't care about dimension count, and 2D lets
every test construct an EXACT, readable cosine similarity via
unit_vec(cos_sim) below, rather than approximating with real text.

Run:
    pytest tests/test_faiss_index.py -v
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.faiss_index import compute_near_dup_features, compute_near_dup_features_chunked

THRESHOLD = 0.92
WINDOWS = {"w": np.timedelta64(1, "h")}  # single named window - keeps most assertions simple


def unit_vec(cos_sim: float) -> np.ndarray:
    """A unit vector whose dot product with [1.0, 0.0] is EXACTLY cos_sim."""
    return np.array([cos_sim, np.sqrt(1 - cos_sim ** 2)], dtype=np.float32)


BASE = unit_vec(1.0)  # dot with itself/identical vectors = 1.0


def run(embeddings, rows, threshold=THRESHOLD, windows=WINDOWS):
    id_map = pd.DataFrame(rows)
    return compute_near_dup_features(
        np.array(embeddings, dtype=np.float32), id_map, threshold=threshold, windows=windows,
    )


def row(message_key="m1", originator="S1", timestamp="2026-08-19T10:00:00"):
    return {"message_key": message_key, "originator": originator, "timestamp": timestamp}


def test_no_match_within_threshold_gives_zero():
    embeddings = [BASE, unit_vec(0.0)]  # orthogonal - similarity 0.0
    rows = [row("m1", timestamp="2026-08-19T10:00:00"), row("m2", timestamp="2026-08-19T10:05:00")]
    result = run(embeddings, rows)
    assert result.iloc[1]["near_dup_match_count_w"] == 0
    assert result.iloc[1]["near_dup_max_similarity_w"] == 0.0
    assert result.iloc[1]["near_dup_distinct_senders_w"] == 0


def test_single_message_never_matches_itself():
    result = run([BASE], [row("m1")])
    assert result.iloc[0]["near_dup_match_count_w"] == 0


def test_earlier_near_dup_is_counted_for_the_later_message():
    embeddings = [BASE, unit_vec(1.0)]  # identical - similarity 1.0
    rows = [
        row("m1", timestamp="2026-08-19T10:00:00"),
        row("m2", timestamp="2026-08-19T10:05:00"),
    ]
    result = run(embeddings, rows)
    assert result.iloc[1]["near_dup_match_count_w"] == 1  # m2 sees earlier m1
    assert result.iloc[1]["near_dup_max_similarity_w"] == pytest.approx(1.0)


def test_later_near_dup_does_not_count_for_the_earlier_message():
    """Point-in-time: m1 must not see m2, which hasn't happened yet."""
    embeddings = [BASE, unit_vec(1.0)]
    rows = [
        row("m1", timestamp="2026-08-19T10:00:00"),
        row("m2", timestamp="2026-08-19T10:05:00"),
    ]
    result = run(embeddings, rows)
    assert result.iloc[0]["near_dup_match_count_w"] == 0


def test_window_boundary_is_inclusive():
    """A match exactly `window` old still counts - same convention as
    features/behavioral.py's closed-left window."""
    embeddings = [BASE, unit_vec(1.0)]
    rows = [
        row("m1", timestamp="2026-08-19T09:00:00"),
        row("m2", timestamp="2026-08-19T10:00:00"),  # exactly 1hr later
    ]
    result = run(embeddings, rows, windows={"w": np.timedelta64(1, "h")})
    assert result.iloc[1]["near_dup_match_count_w"] == 1


def test_match_older_than_window_is_excluded():
    embeddings = [BASE, unit_vec(1.0)]
    rows = [
        row("m1", timestamp="2026-08-19T08:59:59"),
        row("m2", timestamp="2026-08-19T10:00:00"),  # just over 1hr later
    ]
    result = run(embeddings, rows, windows={"w": np.timedelta64(1, "h")})
    assert result.iloc[1]["near_dup_match_count_w"] == 0


def test_below_threshold_similarity_is_excluded():
    embeddings = [BASE, unit_vec(0.5)]  # below the 0.92 threshold
    rows = [row("m1", timestamp="2026-08-19T10:00:00"), row("m2", timestamp="2026-08-19T10:05:00")]
    result = run(embeddings, rows)
    assert result.iloc[1]["near_dup_match_count_w"] == 0


def test_distinct_senders_counts_unique_originators_not_raw_matches():
    embeddings = [BASE, unit_vec(1.0), unit_vec(1.0), unit_vec(1.0)]
    rows = [
        row("m1", originator="A", timestamp="2026-08-19T10:00:00"),
        row("m2", originator="A", timestamp="2026-08-19T10:01:00"),  # same sender as m1
        row("m3", originator="B", timestamp="2026-08-19T10:02:00"),  # different sender
        row("m4", originator="C", timestamp="2026-08-19T10:03:00"),  # the query
    ]
    result = run(embeddings, rows)
    last = result.iloc[3]
    assert last["near_dup_match_count_w"] == 3  # m1, m2, m3
    assert last["near_dup_distinct_senders_w"] == 2  # only A and B


def test_max_similarity_picks_the_highest_qualifying_match():
    embeddings = [unit_vec(0.95), unit_vec(0.99), BASE]  # query is BASE (index 2)
    rows = [
        row("m1", timestamp="2026-08-19T10:00:00"),
        row("m2", timestamp="2026-08-19T10:01:00"),
        row("m3", timestamp="2026-08-19T10:02:00"),  # query
    ]
    result = run(embeddings, rows)
    assert result.iloc[2]["near_dup_max_similarity_w"] == pytest.approx(0.99)


def test_output_preserves_row_order_and_message_keys():
    embeddings = [unit_vec(0.0), unit_vec(1.0)]
    rows = [row("keyB", timestamp="2026-08-19T10:05:00"), row("keyA", timestamp="2026-08-19T10:00:00")]
    result = run(embeddings, rows)
    assert list(result["message_key"]) == ["keyB", "keyA"]


def test_raises_on_missing_id_map_column():
    bad_id_map = pd.DataFrame([{"message_key": "m1"}])
    with pytest.raises(ValueError):
        compute_near_dup_features(np.array([BASE], dtype=np.float32), bad_id_map)


# --- dual-window behavior ---------------------------------------------

def test_dual_windows_short_misses_what_long_catches():
    """The actual reason for two windows: a match 90min old (older than a
    1hr short window, within a 24hr long window) shows up in the long
    window's count but not the short one's."""
    embeddings = [BASE, unit_vec(1.0)]
    rows = [
        row("m1", timestamp="2026-08-19T08:30:00"),
        row("m2", timestamp="2026-08-19T10:00:00"),  # 90min after m1
    ]
    windows = {"short": np.timedelta64(1, "h"), "long": np.timedelta64(24, "h")}
    result = run(embeddings, rows, windows=windows)
    later = result.iloc[1]
    assert later["near_dup_match_count_short"] == 0
    assert later["near_dup_match_count_long"] == 1


def test_dual_windows_share_the_same_underlying_matches_beyond_long_window():
    """A match older than BOTH windows counts for neither."""
    embeddings = [BASE, unit_vec(1.0)]
    rows = [
        row("m1", timestamp="2026-08-18T09:00:00"),
        row("m2", timestamp="2026-08-19T10:00:00"),  # 25hr after m1
    ]
    windows = {"short": np.timedelta64(1, "h"), "long": np.timedelta64(24, "h")}
    result = run(embeddings, rows, windows=windows)
    later = result.iloc[1]
    assert later["near_dup_match_count_short"] == 0
    assert later["near_dup_match_count_long"] == 0


# --- chunked processing -------------------------------------------------

def _spread_rows(n, minutes_apart=20, same_text_every=3):
    """n rows spread `minutes_apart` apart, with every `same_text_every`th
    row sharing an identical embedding (so there's real near-dup signal
    to detect across chunk boundaries, not just noise)."""
    base_time = pd.Timestamp("2026-08-19T00:00:00")
    rows = []
    embeddings = []
    for i in range(n):
        rows.append(row(
            message_key=f"m{i}",
            originator=f"S{i % 5}",
            timestamp=(base_time + timedelta(minutes=minutes_apart * i)).isoformat(),
        ))
        embeddings.append(unit_vec(1.0) if i % same_text_every == 0 else unit_vec(0.0))
    return embeddings, rows


def test_chunked_matches_unchunked_exactly():
    embeddings, rows = _spread_rows(30)
    id_map = pd.DataFrame(rows)
    emb = np.array(embeddings, dtype=np.float32)
    windows = {"short": np.timedelta64(1, "h"), "long": np.timedelta64(6, "h")}

    unchunked = compute_near_dup_features(emb, id_map, windows=windows)
    chunked = compute_near_dup_features_chunked(emb, id_map, chunk_size=7, windows=windows)

    unchunked_sorted = unchunked.sort_values("message_key").reset_index(drop=True)
    chunked_sorted = chunked.sort_values("message_key").reset_index(drop=True)
    pd.testing.assert_frame_equal(unchunked_sorted, chunked_sorted)


def test_chunked_preserves_caller_row_order_not_sorted_order():
    """id_map here is deliberately NOT in timestamp order - chunked must
    still return rows matching the caller's original order."""
    embeddings = [unit_vec(1.0), unit_vec(1.0)]
    rows = [
        row("later", timestamp="2026-08-19T10:05:00"),
        row("earlier", timestamp="2026-08-19T10:00:00"),
    ]
    id_map = pd.DataFrame(rows)
    result = compute_near_dup_features_chunked(
        np.array(embeddings, dtype=np.float32), id_map, chunk_size=1, windows=WINDOWS,
    )
    assert list(result["message_key"]) == ["later", "earlier"]


def test_chunked_catches_a_match_straddling_a_chunk_boundary():
    """A near-dup pair split across two small chunks must still be found
    via the buffer - this is the whole point of the buffer logic."""
    embeddings = [BASE] + [unit_vec(0.0)] * 3 + [unit_vec(1.0)]  # match at position 0 and 4
    rows = [
        row("m0", timestamp="2026-08-19T10:00:00"),
        row("m1", timestamp="2026-08-19T10:10:00"),
        row("m2", timestamp="2026-08-19T10:20:00"),
        row("m3", timestamp="2026-08-19T10:30:00"),
        row("m4", timestamp="2026-08-19T10:40:00"),  # near-dup of m0, in a later chunk
    ]
    id_map = pd.DataFrame(rows)
    # chunk_size=2 puts m0 in the first chunk and m4 in the third -
    # only the buffer (covering the 1hr window back from each chunk's
    # start) makes this detectable at all.
    result = compute_near_dup_features_chunked(
        np.array(embeddings, dtype=np.float32), id_map, chunk_size=2, windows=WINDOWS,
    )
    m4 = result[result["message_key"] == "m4"].iloc[0]
    assert m4["near_dup_match_count_w"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
