"""
pytest suite for features/identity_baseline.py.

Run:
    pytest tests/test_identity_baseline.py -v
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import identity_baseline as ib
from features.behavioral_snapshot import compute_sender_snapshots


def test_shannon_entropy_known_values():
    assert ib.shannon_entropy("aaaa") == 0.0
    assert ib.shannon_entropy("abab") == pytest.approx(1.0)
    assert ib.shannon_entropy("abcd") == pytest.approx(2.0)


@pytest.mark.parametrize("text", [None, "", float("nan")])
def test_shannon_entropy_null_is_nan_not_zero(text):
    assert math.isnan(ib.shannon_entropy(text))


def test_ema_cold_start_then_update():
    s = ib.update_baseline("entropy", 4.0, None)
    assert s == {"level": 4.0, "residual_spread": 0.0}
    s = ib.update_baseline("entropy", 5.0, s)  # alpha=0.1
    assert s["level"] == pytest.approx(4.1)
    assert s["residual_spread"] == pytest.approx(0.1)


def test_holt_winters_learns_a_seasonal_pattern(monkeypatch):
    cfg = ib.BaselineConfig(alpha=0.3, use_seasonality=True, season_length=4, gamma=0.5)
    monkeypatch.setitem(ib.BASELINE_CONFIGS, "volume", cfg)
    cycle = [10.0, 20.0, 30.0, 20.0]
    state = None
    for t in range(80):
        state = ib.update_baseline("volume", cycle[t % 4], state, bucket_index=t)
    # Steady-state forecast error on a clean cycle is ~0.
    assert state["residual_spread"] < 0.5
    assert state["season"][2] > state["season"][0]


def test_entropy_zscore_undefined_cases_are_none():
    assert ib.entropy_zscore("hello", None, None) is None
    assert ib.entropy_zscore("hello", 3.0, 0.0) is None  # no measured variation yet
    assert ib.entropy_zscore("", 3.0, 0.5) is None
    assert ib.entropy_zscore("hello", float("nan"), float("nan")) is None


def test_entropy_zscore_arithmetic_and_floor():
    h = ib.shannon_entropy("abcd")  # 2.0
    assert ib.entropy_zscore("abcd", 1.0, 0.5) == pytest.approx((h - 1.0) / 0.5)
    assert ib.entropy_zscore("abcd", 1.0, 0.001) == pytest.approx((h - 1.0) / ib.MIN_SPREAD)


NOW = pd.Timestamp("2026-08-19T10:00:00")


def _msg(ts, text, originator="S1"):
    return {"source": "SMPP", "originator": originator, "destination": "9198765001",
            "timestamp": ts, "text": text}


def test_snapshot_carries_entropy_baseline():
    rows = [
        _msg("2026-08-19T06:10:00", "abab"),
        _msg("2026-08-19T06:20:00", "abab"),  # same bucket -> one mean
        _msg("2026-08-19T07:10:00", "abcd"),
        _msg("2026-08-19T23:00:00", "zzzz"),  # after NOW - not history
        _msg("2026-08-19T08:00:00", "", originator="S2"),  # undecodable only
    ]
    snap = compute_sender_snapshots(pd.DataFrame(rows), now=NOW).set_index("sender_id")
    s1 = snap.loc["SMPP|S1"]
    assert s1["entropy_level"] == pytest.approx(0.1 * 2.0 + 0.9 * 1.0)
    assert s1["entropy_residual_spread"] == pytest.approx(0.1 * 1.0)
    assert math.isnan(snap.loc["SMPP|S2", "entropy_level"])
