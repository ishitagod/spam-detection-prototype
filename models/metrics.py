"""
Evaluation helpers shared across models/anomaly/train.py and
models/rule_pattern/train.py - both need the same real PR-AUC/log-loss
computation against rule_evaluated labels (validation-only for the
unsupervised model, the actual training labels for the supervised one),
with the same single-class guard: SMPP has zero confirmed-clean labels
(verified - see README.md's data reality check), so PR-AUC is
mathematically undefined for SMPP-only slices. Both models hit this
exact scenario, so the guard lives here once, not duplicated per model.
"""
import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def pr_auc_and_log_loss(y_true: np.ndarray, score: np.ndarray) -> dict | None:
    """
    `score` can be any real-valued ranking score (probabilities, raw
    decision-function output, etc.) - average_precision_score only cares
    about relative order. Returns None (not a fabricated 0.5, not a
    crash) if only one class is present in `y_true`, so callers can skip
    that slice explicitly rather than silently logging a meaningless
    number.
    """
    if len(set(y_true.tolist())) < 2:
        return None
    score_min, score_max = score.min(), score.max()
    y_prob = (
        (score - score_min) / (score_max - score_min)
        if score_max > score_min else np.full_like(score, 0.5, dtype=np.float64)
    )
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "n": int(len(y_true)),
        "n_positive": int(np.asarray(y_true).sum()),
    }


def evaluate_overall_and_per_source(df, source_col: str, y_true: np.ndarray, score: np.ndarray, prefix: str = "") -> dict:
    """
    Runs pr_auc_and_log_loss() overall, then once per distinct value in
    `df[source_col]` (aligned positionally with y_true/score - same
    length, same row order) - printing a clear skip message instead of
    a metric for any slice with only one class. `prefix` namespaces the
    returned metric keys (e.g. "train_", "test_") when a caller needs
    both without collisions.
    """
    result = {}
    overall = pr_auc_and_log_loss(y_true, score)
    if overall:
        result.update({f"{prefix}overall_{k}": v for k, v in overall.items()})
    else:
        print(f"  {prefix}overall: skipped (only one class present)")

    sources = df[source_col].to_numpy()
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        metrics = pr_auc_and_log_loss(y_true[mask], score[mask])
        if metrics:
            result.update({f"{prefix}{source}_{k}": v for k, v in metrics.items()})
        else:
            print(f"  {prefix}{source}: skipped (only one class present)")

    return result
