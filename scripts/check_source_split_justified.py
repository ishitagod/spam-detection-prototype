"""
Answers the actual question CLAUDE.md's architecture rule poses: "start
with one shared model, split by source only if segment evaluation
justifies it" - has it?

Does NOT train anything new. Both models/rule_pattern/train.py and
models/anomaly/train.py already log per-source PR-AUC/log-loss on every
run via models/metrics.py's evaluate_overall_and_per_source() - this
script just reads the latest logged run of each and reports the gap
between the shared model's overall metric and its per-source metrics.

Usage:
    python -m scripts.check_source_split_justified
    python -m scripts.check_source_split_justified --min_gap 0.05

Interpreting the output: a persistent, large PR-AUC gap between sources
is the actual justification threshold per CLAUDE.md - not a vibe call.
A small gap means the shared model (with `source` as a one-hot feature)
is already capturing what's source-specific; splitting would just add
2x models/2x MLflow tracking/2x promotion logic for no measured benefit.

CAVEAT you should know before trusting this: SMPP's rule_evaluated pool
is 2,693 rows, ALL flagged (see models/rule_pattern/train.py's module
docstring) - PR-AUC is mathematically undefined for SMPP-only slices in
that model, so it's skipped upstream (not computed, not silently wrong)
and will show as "not available" below, not zero. The rule_pattern
comparison is therefore SS7-only in practice today. anomaly_score's
comparison is real for both sources (its validation pool includes
SMPP's flagged rows even though PR-AUC there is also skipped for the
same reason - so anomaly_score's per-source numbers below face the same
SMPP gap).
"""
import argparse
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# (experiment_name, metric_prefix, label) - metric_prefix matches what
# each train.py actually passes to evaluate_overall_and_per_source().
CHECKS = [
    ("light_gbm", "test_", "rule_pattern_score (LightGBM, plain)"),
    ("rule_pattern_score_experimental", "test_", "rule_pattern_score (--with_embeddings/--with_tfidf)"),
    ("isolation_forest", "", "anomaly_score (Isolation Forest)"),
]

SOURCES = ["SMPP", "SS7"]


def latest_run(experiment_name: str):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs.iloc[0] if len(runs) else None


def report(experiment_name: str, prefix: str, label: str, min_gap: float) -> None:
    print(f"\n=== {label} (experiment: {experiment_name}) ===")
    run = latest_run(experiment_name)
    if run is None:
        print("  No runs found - train this model first.")
        return

    overall_key = f"metrics.{prefix}overall_pr_auc"
    overall = run.get(overall_key)
    if overall is None or pd_isna(overall):
        print(f"  {overall_key} not present in latest run - skipping.")
        return
    print(f"  overall pr_auc: {overall:.4f}")

    per_source = {}
    for source in SOURCES:
        key = f"metrics.{prefix}{source}_pr_auc"
        val = run.get(key)
        if val is not None and not pd_isna(val):
            per_source[source] = val
            print(f"  {source} pr_auc:  {val:.4f}  (gap vs overall: {val - overall:+.4f})")
        else:
            print(f"  {source} pr_auc:  not available (likely single-class slice - see module docstring)")

    if len(per_source) < 2:
        print("  Fewer than 2 sources have a defined PR-AUC this run - no real cross-source comparison possible yet.")
        return

    gap = max(per_source.values()) - min(per_source.values())
    print(f"  max cross-source gap: {gap:.4f} (threshold: {min_gap})")
    if gap >= min_gap:
        print(f"  -> gap >= threshold: consider evaluating a source-specific model for {label}.")
    else:
        print(f"  -> gap < threshold: shared model looks fine for {label}, no split justified yet.")


def pd_isna(x) -> bool:
    import math
    return x is None or (isinstance(x, float) and math.isnan(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min_gap", type=float, default=0.05,
        help="PR-AUC gap between sources at/above which a split is worth evaluating (default: 0.05).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for experiment_name, prefix, label in CHECKS:
        report(experiment_name, prefix, label, args.min_gap)


if __name__ == "__main__":
    main()
