"""
Training data for the decision-fusion meta-model - one row per
rule_evaluated message with BOTH base scores (rule_pattern_score,
anomaly_score - kept separate per CLAUDE.md, never averaged) plus the real
label, so a small third model can learn how to weigh their disagreement.

GROUND TRUTH: rule_flagged (labels/rule_labels.py) - the same label
rule_pattern_score trains on. content_flagged is deliberately not used
(never silently merge label sources, per that module).

ANOMALY_SCORE COVERAGE IS UNEVEN, measured not assumed: joined by
message_key against data/processed/<source>/anomaly_scores.parquet (inner
join, same partial-coverage convention as
models/rule_pattern/data.py::load_labelled_messages_with_embeddings()).
SMPP: anomaly_scores.parquet covers the full 5,505,921-row corpus (100%
join coverage) - and, correcting a stale claim elsewhere
(models/rule_pattern/train.py's "SMPP: all flagged" note), SMPP's real
rule_evaluated pool is 139,546 rows / 2,692 flagged, a real two-class pool.
SS7: anomaly_scores.parquet is still a 20,000-row sample (pre-dates the
full embeddings.npy build), so only ~0.7% (19,369/2,654,369) of SS7's
rule_evaluated pool joins - re-run `models.anomaly.train --sources SS7` to
refresh it, then re-run this. Printed every run, not hidden.

RULE_PATTERN_SCORE MUST BE OUT-OF-FOLD: the champion LightGBM was fit on
this exact pool, so scoring it with that same model would teach the fusion
model to trust memorized rows, not real disagreement (stacking leakage).
Uses cross_val_predict (StratifiedKFold) with a fresh LightGBM over
models/rule_pattern/data.py::_base_feature_frame() (base features only, no
TF-IDF/embeddings - matches the production champion's default path).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from models.rule_pattern.data import _base_feature_frame, load_labelled_messages

N_FOLDS = 5  # starting point, not tuned


def build_fusion_training_data(
    source: str, data_dir: Path, random_state: int = 42,
) -> pd.DataFrame:
    """Returns [message_key, source, rule_pattern_score, anomaly_score,
    rule_flagged_label]. Empty (0 rows) if the source's rule_evaluated pool
    is single-class - callers must handle that."""
    source_dir = Path(data_dir) / source
    messages_path = source_dir / "messages_with_behavioral.csv"

    df = load_labelled_messages(messages_path)
    df["message_key"] = df["source"] + "|" + df["record_id"]
    y = (df["rule_flagged"] == True).to_numpy(dtype=int)  # noqa: E712
    print(f"  {source}: {len(df)} rule_evaluated row(s)")

    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        print(
            f"  {source}: only one class present in rule_flagged ({y.sum()} positive / "
            f"{len(y) - y.sum()} negative) - skipping, cannot cross-validate a single class."
        )
        return pd.DataFrame(
            columns=["message_key", "source", "rule_pattern_score", "anomaly_score", "rule_flagged_label"]
        )

    print(f"  {source}: computing out-of-fold rule_pattern_score ({N_FOLDS}-fold, base features only) ...")
    X = _base_feature_frame(df).to_numpy(dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    oof = cross_val_predict(
        LGBMClassifier(random_state=random_state, verbosity=-1),
        X, y, cv=skf, method="predict_proba",
    )[:, 1]
    df["rule_pattern_score"] = oof
    df["rule_flagged_label"] = y

    anomaly_path = source_dir / "anomaly_scores.parquet"
    anomaly = pd.read_parquet(anomaly_path, columns=["message_key", "anomaly_score"])
    before = len(df)
    df = df.merge(anomaly, on="message_key", how="inner")
    coverage = len(df) / before if before else 0.0
    print(f"  {source}: {len(df)}/{before} row(s) have an anomaly_score ({coverage:.1%} coverage)")

    return df[
        ["message_key", "source", "rule_pattern_score", "anomaly_score", "rule_flagged_label"]
    ].reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """X = [rule_pattern_score, anomaly_score] raw (unscaled - the fusion
    Pipeline scales internally), y = rule_flagged_label. Deliberately just
    these two columns - fusion's job is weighing the two scores that
    already exist, not re-deriving a third opinion from raw features."""
    feature_names = ["rule_pattern_score", "anomaly_score"]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["rule_flagged_label"].to_numpy(dtype=int)
    return X, y, feature_names
