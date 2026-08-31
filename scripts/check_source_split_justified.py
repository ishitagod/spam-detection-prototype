"""
Answers the actual question CLAUDE.md's architecture rule poses: "start
with one shared model, split by source only if segment evaluation
justifies it" - has it? And once split (see models/anomaly/train.py and
models/rule_pattern/train.py's --sources filtering, and
scripts/run_full_pipeline.ps1 -SplitBySource), did splitting actually
help?

Does NOT train anything new - reads what's already logged to MLflow.

TWO checks, run for each layer:

1. SHARED-MODEL GAP (pre-split question): does the shared model's own
   per-source PR-AUC slice (models/metrics.py's
   evaluate_overall_and_per_source(), logged on every combined
   --sources SMPP SS7 run) vary a lot by source? A big, persistent gap
   is the signal that a split might help - see module docstring further
   down for caveats.

2. SPLIT-VS-SHARED (post-split question, only shown once a split
   experiment has runs): for each source, is the SOURCE-SPECIFIC
   model's own PR-AUC actually better than that source's slice of the
   shared model? This is the number that actually justifies KEEPING the
   split - a source-specific model trained on a smaller pool can easily
   score worse than the shared model's slice of it (see the small-SMPP-
   pool caveat below), so don't assume splitting always wins just
   because the pre-split gap existed.

Usage:
    python -m scripts.check_source_split_justified
    python -m scripts.check_source_split_justified --min_gap 0.05

CAVEAT you should know before trusting either check: SMPP's
rule_evaluated pool is small (2,693 rows as of writing) - both the
per-source PR-AUC slice and any SMPP-only split model are trained/
evaluated on a much smaller sample than SS7's, so a gap here is more
likely to be sampling noise than a real source difference. Rerun with a
different --random_state on the SMPP-only training command before
trusting a one-run gap.
"""
import argparse
import math
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# (combined_experiment_name, metric_prefix, label) - metric_prefix
# matches what each train.py actually passes to
# evaluate_overall_and_per_source(). The split experiment name for a
# given source is derived as f"{combined_experiment_name}_{source}" -
# see models/anomaly/train.py / models/rule_pattern/train.py's
# experiment_name suffixing.
CHECKS = [
    ("light_gbm", "test_", "rule_pattern_score (LightGBM, plain)"),
    ("rule_pattern_score_experimental", "test_", "rule_pattern_score (--with_embeddings/--with_tfidf)"),
    ("isolation_forest", "", "anomaly_score (Isolation Forest)"),
]

SOURCES = ["SMPP", "SS7"]


def pd_isna(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


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


def shared_model_gap(experiment_name: str, prefix: str, label: str, min_gap: float):
    """Check 1: does the shared model's per-source slice vary a lot?
    Returns {source: pr_auc} for reuse by split_vs_shared() below."""
    print(f"\n=== {label} - shared-model gap (experiment: {experiment_name}) ===")
    run = latest_run(experiment_name)
    if run is None:
        print("  No runs found - train the combined (both-sources) model first.")
        return {}

    overall_key = f"metrics.{prefix}overall_pr_auc"
    overall = run.get(overall_key)
    if overall is None or pd_isna(overall):
        print(f"  {overall_key} not present in latest run - skipping.")
        return {}
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
        return per_source

    gap = max(per_source.values()) - min(per_source.values())
    print(f"  max cross-source gap: {gap:.4f} (threshold: {min_gap})")
    if gap >= min_gap:
        print(f"  -> gap >= threshold: worth evaluating a source-specific model for {label}.")
    else:
        print(f"  -> gap < threshold: shared model looks fine for {label}, no split justified yet.")
    return per_source


def split_vs_shared(experiment_name: str, prefix: str, label: str, min_gap: float, shared_per_source: dict):
    """Check 2: for each source with a split experiment logged, does the
    source-specific model actually beat that source's slice of the
    shared model? Only prints anything for sources that HAVE a split
    experiment - silent (not a failure) if you haven't split yet."""
    any_split = False
    for source in SOURCES:
        split_experiment = f"{experiment_name}_{source}"
        run = latest_run(split_experiment)
        if run is None:
            continue  # not split for this source - nothing to compare
        any_split = True

        split_metric = run.get(f"metrics.{prefix}overall_pr_auc")
        if split_metric is None or pd_isna(split_metric):
            print(f"\n=== {label} - split vs shared ({source}) ===")
            print(f"  {split_experiment}: latest run has no {prefix}overall_pr_auc - skipping.")
            continue

        shared_metric = shared_per_source.get(source)
        print(f"\n=== {label} - split vs shared ({source}) ===")
        print(f"  split model  ({split_experiment}): pr_auc = {split_metric:.4f}")
        if shared_metric is None:
            print(f"  shared model's {source} slice: not available this run - can't compare directly.")
            continue
        print(f"  shared model's {source} slice:      pr_auc = {shared_metric:.4f}")
        delta = split_metric - shared_metric
        print(f"  delta (split - shared): {delta:+.4f} (threshold: {min_gap})")
        if delta >= min_gap:
            print(f"  -> split model clearly beats the shared model's {source} slice - keep the split for {source}.")
        elif delta <= -min_gap:
            print(f"  -> split model is WORSE than the shared model's {source} slice - "
                  f"revert {source} to the shared model (likely too little data to split, see module docstring).")
        else:
            print(f"  -> within noise of the shared model's {source} slice - split isn't earning its complexity yet.")

    if not any_split:
        print(f"\n=== {label} - split vs shared: no split runs found (see scripts/run_full_pipeline.ps1 -SplitBySource) ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min_gap", type=float, default=0.05,
        help="PR-AUC gap/delta at/above which a split is worth evaluating or keeping (default: 0.05).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for experiment_name, prefix, label in CHECKS:
        shared_per_source = shared_model_gap(experiment_name, prefix, label, args.min_gap)
        split_vs_shared(experiment_name, prefix, label, args.min_gap, shared_per_source)


if __name__ == "__main__":
    main()
