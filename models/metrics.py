"""
Evaluation helpers shared across models/anomaly/train.py and
models/rule_pattern/train.py - both need the same real PR-AUC/log-loss
computation against rule_evaluated labels (validation-only for the
unsupervised model, the actual training labels for the supervised one),
with the same single-class guard: SMPP has zero confirmed-clean labels
(verified - see README.md's data reality check), so PR-AUC is
mathematically undefined for SMPP-only slices. Both models hit this
exact scenario, so the guard lives here once, not duplicated per model.

PRECISION@K, alongside PR-AUC: PR-AUC integrates over EVERY possible
threshold, most of which this project never operates at in practice -
only the extreme top of a ranking score ever gets acted on (see
models/anomaly/cluster_discovery.py's --anomaly_percentile design, or
docs/sms_spam_technical_architecture_plan.md's "precision at fixed
recall - the number that actually maps to a business decision" note).
precision_at_k()/precision_at_k_percentiles() answer the more honest
operational question: of the top N% ranked by score, what fraction are
really positive? Same single-class-style guard as pr_auc_and_log_loss()
below - a K outside [1, n] returns None (skip), not a fabricated number.
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


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> dict | None:
    """
    Precision within the top `k` rows by `score` (descending) - see
    module docstring for why this, not just PR-AUC. Returns None (not a
    fabricated number) if k < 1 or k > len(y_true) - same "don't compute
    a meaningless number silently" guard as pr_auc_and_log_loss(), for
    the same reason: a K outside the valid range isn't a real cutoff.
    """
    n = len(y_true)
    if k < 1 or k > n:
        return None
    top_idx = np.argsort(score)[::-1][:k]
    return {
        "precision": float(np.asarray(y_true)[top_idx].mean()),
        "k": int(k),
        "n_positive_in_k": int(np.asarray(y_true)[top_idx].sum()),
    }


def precision_at_k_percentiles(
    y_true: np.ndarray, score: np.ndarray,
    percentiles: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0),
    label: str = "",
) -> dict:
    """
    precision_at_k() at several top-N% cutoffs - percentile -> k via
    plain rounding, NOT forced to at least 1: a percentile that rounds to
    0 on a small pool is genuinely out of range (precision_at_k() itself
    skips it with a clear message), not silently promoted to "top 1 row"
    and passed off as a real 0.1%/0.5%/etc. reading. `label` prefixes the
    printed skip message only (e.g. "overall_", "SS7_") - metric dict
    keys are always plain precision_at_top_<N>pct_* regardless of label,
    matching pr_auc_and_log_loss()'s own bare keys before
    evaluate_overall_and_per_source() namespaces them.
    """
    result = {}
    n = len(y_true)
    for p in percentiles:
        k = round(n * p / 100)
        metrics = precision_at_k(y_true, score, k)
        key_p = str(p).replace(".", "_")
        if metrics:
            result.update({f"precision_at_top_{key_p}pct_{mk}": mv for mk, mv in metrics.items()})
        else:
            print(f"  {label}precision_at_top_{key_p}pct: skipped (k={k} out of range for n={n})")
    return result


def evaluate_precision_at_k(
    df, source_col: str, y_true: np.ndarray, score: np.ndarray,
    percentiles: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0), prefix: str = "",
) -> dict:
    """
    precision_at_k_percentiles(), overall then per source - same
    overall-AND-per-source convention as evaluate_overall_and_per_source()
    above (a shared model can look fine in aggregate while quietly
    underperforming on one segment) and the same `prefix` namespacing.
    """
    result = {}
    result.update({
        f"{prefix}overall_{k}": v
        for k, v in precision_at_k_percentiles(y_true, score, percentiles, label=f"{prefix}overall_").items()
    })

    sources = df[source_col].to_numpy()
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        per_source = precision_at_k_percentiles(
            y_true[mask], score[mask], percentiles, label=f"{prefix}{source}_",
        )
        result.update({f"{prefix}{source}_{k}": v for k, v in per_source.items()})

    return result
