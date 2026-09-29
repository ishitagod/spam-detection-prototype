"""
Per-sender baselines ("is this sender off from ITS OWN history"), unlike
features/behavioral.py's peer comparison ("off from other senders").

Metric: text entropy (bits/char) - not a model feature, only self-baselined
and exposed as entropy_zscore in serving/app.py.

update_baseline() is generic (EMA or Holt-Winters per BASELINE_CONFIGS);
entropy uses plain EMA (no known time-of-day cycle yet).
"""
from collections import Counter
from dataclasses import dataclass
from math import log2

import numpy as np
import pandas as pd

BUCKET_FREQ = "1h"
MIN_SPREAD = 0.05  # z-score floor, avoids inflating near-zero spread


@dataclass(frozen=True)
class BaselineConfig:
    alpha: float
    use_seasonality: bool = False
    beta: float = 0.05  # Holt-Winters trend smoothing
    gamma: float = 0.1  # Holt-Winters season smoothing
    season_length: int = 24


BASELINE_CONFIGS: dict[str, BaselineConfig] = {
    "entropy": BaselineConfig(alpha=0.1, use_seasonality=False),  # unvalidated starting point
}


def shannon_entropy(text) -> float:
    """Bits/char. NaN for null/empty text - "no signal", not 0.0."""
    if not isinstance(text, str) or not text:
        return float("nan")
    n = len(text)
    return -sum((c / n) * log2(c / n) for c in Counter(text).values())


def ema_update(cfg: BaselineConfig, new_value: float, prev_state: dict | None) -> dict:
    if prev_state is None:
        return {"level": new_value, "residual_spread": 0.0}
    residual = new_value - prev_state["level"]
    return {
        "level": cfg.alpha * new_value + (1 - cfg.alpha) * prev_state["level"],
        "residual_spread": (
            cfg.alpha * abs(residual) + (1 - cfg.alpha) * prev_state["residual_spread"]
        ),
    }


def holt_winters_update(
    cfg: BaselineConfig, new_value: float, prev_state: dict | None, bucket_index: int
) -> dict:
    """Additive Holt-Winters; `bucket_index` = position in the cycle."""
    if prev_state is None:
        return {
            "level": new_value, "trend": 0.0, "residual_spread": 0.0,
            "season": [0.0] * cfg.season_length,
        }
    i = bucket_index % cfg.season_length
    level, trend, season = prev_state["level"], prev_state["trend"], prev_state["season"]
    residual = new_value - (level + trend + season[i])
    new_level = cfg.alpha * (new_value - season[i]) + (1 - cfg.alpha) * (level + trend)
    new_season = list(season)
    new_season[i] = cfg.gamma * (new_value - new_level) + (1 - cfg.gamma) * season[i]
    return {
        "level": new_level,
        "trend": cfg.beta * (new_level - level) + (1 - cfg.beta) * trend,
        "season": new_season,
        "residual_spread": (
            cfg.alpha * abs(residual) + (1 - cfg.alpha) * prev_state["residual_spread"]
        ),
    }


def update_baseline(
    metric_name: str, new_value: float, prev_state: dict | None, bucket_index: int = 0
) -> dict:
    """One step for one entity. `prev_state=None` = cold start."""
    cfg = BASELINE_CONFIGS[metric_name]
    if cfg.use_seasonality:
        return holt_winters_update(cfg, new_value, prev_state, bucket_index)
    return ema_update(cfg, new_value, prev_state)


def compute_entropy_baselines(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Entropy baseline per sender as of `now`. Needs sender_id/timestamp/text."""
    msgs = df.loc[df["timestamp"] <= now, ["sender_id", "timestamp", "text"]].copy()
    msgs["entropy"] = msgs["text"].map(shannon_entropy)
    msgs = msgs.dropna(subset=["entropy"])
    msgs["bucket"] = msgs["timestamp"].dt.floor(BUCKET_FREQ)
    buckets = (
        msgs.groupby(["sender_id", "bucket"], sort=True)["entropy"].mean().reset_index()
    )

    levels, spreads = {}, {}
    state, current = None, None
    for sender, bucket, value in zip(
        buckets["sender_id"], buckets["bucket"], buckets["entropy"]
    ):
        if sender != current:
            if current is not None:
                levels[current], spreads[current] = state["level"], state["residual_spread"]
            current, state = sender, None
        state = update_baseline("entropy", value, state, bucket_index=bucket.hour)
    if current is not None:
        levels[current], spreads[current] = state["level"], state["residual_spread"]

    return pd.DataFrame(
        {"entropy_level": pd.Series(levels), "entropy_residual_spread": pd.Series(spreads)},
        dtype="float64",
    )


def entropy_zscore(text, level, residual_spread) -> float | None:
    """Residual-spreads from sender's own level. None (not 0) when undefined."""
    h = shannon_entropy(text)
    if np.isnan(h) or level is None or residual_spread is None:
        return None
    if np.isnan(level) or np.isnan(residual_spread) or residual_spread <= 0:
        return None
    return (h - level) / max(residual_spread, MIN_SPREAD)
