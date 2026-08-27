"""
Champion/challenger promotion - explicit rule in code, per CLAUDE.md's
"Champion/challenger promotion is explicit in code" convention. Not
automatic, not silent: compares the latest run in an MLflow experiment
(the "challenger") against whichever version currently holds the
`champion` alias for a registered model name, and only promotes (moves
the alias) if the challenger's chosen metric beats the champion's by at
least --min_improvement. First-ever run for a registered name always
promotes (bootstrap case - there's no champion to lose to yet).

Usage:
    python -m models.compare_versions \
        --experiment_name anomaly_score --registered_name anomaly_score_model \
        --metric_key overall_pr_auc

    python -m models.compare_versions \
        --experiment_name rule_pattern_score --registered_name rule_pattern_score_model \
        --metric_key test_overall_pr_auc

Run this by hand after a training run, same as train.py itself - not
wired into anything automatic. See models/registry.py for the loading
side (how serving code picks up whatever this promotes).
"""
import argparse

import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def get_latest_run(experiment_name: str) -> mlflow.entities.Run:
    """The most recent run in this experiment - the challenger."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise ValueError(f"No runs found in experiment {experiment_name!r}")
    return runs[0]


def get_champion_metric(registered_name: str, alias: str, metric_key: str) -> float | None:
    """None if no champion exists yet - a real, expected first-run state, not an error."""
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, alias)
    except mlflow.exceptions.MlflowException:
        return None
    champion_run = client.get_run(version.run_id)
    return champion_run.data.metrics.get(metric_key)


def run(
    experiment_name: str, registered_name: str, metric_key: str,
    artifact_path: str = "model", alias: str = "champion", min_improvement: float = 0.0,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    challenger_run = get_latest_run(experiment_name)
    challenger_metric = challenger_run.data.metrics.get(metric_key)
    if challenger_metric is None:
        raise ValueError(f"Latest run {challenger_run.info.run_id} has no metric {metric_key!r}")
    print(f"Challenger run {challenger_run.info.run_id}: {metric_key}={challenger_metric}")

    champion_metric = get_champion_metric(registered_name, alias, metric_key)
    if champion_metric is None:
        print(f"No current {alias!r} for {registered_name!r} - bootstrapping.")
        promote = True
    else:
        print(f"Current {alias!r}: {metric_key}={champion_metric}")
        promote = challenger_metric >= champion_metric + min_improvement

    if not promote:
        print(f"NOT promoting - challenger does not beat champion by >= {min_improvement}.")
        return

    model_uri = _find_model_uri(challenger_run, artifact_path)
    version = mlflow.register_model(model_uri, registered_name)
    mlflow.MlflowClient().set_registered_model_alias(registered_name, alias, version.version)
    print(f"PROMOTED: {registered_name} v{version.version} is now {alias!r}.")


def _find_model_uri(run: mlflow.entities.Run, artifact_path: str) -> str:
    """
    MLflow 3.x logs each model as its own first-class "LoggedModel"
    entity (run.outputs.model_outputs), not just a bare artifact path
    under the run - the older `runs:/{run_id}/{artifact_path}` URI
    convention still resolves for SIMPLE runs, but mlflow.register_model
    has to fall back/guess to make it work (verified: it prints a real
    warning doing so), and guessing breaks outright for a run that logs
    MORE THAN ONE model - e.g. models/rule_pattern/train.py's
    --with_embeddings path logs both the LightGBM model AND a separate
    embedding_pca_pipeline in the same run. This looks up the exact
    LoggedModel by `name` (the artifact_path used at logging time, e.g.
    "model") instead of relying on the old convention resolving right.
    """
    client = mlflow.MlflowClient()
    if not run.outputs or not run.outputs.model_outputs:
        raise ValueError(f"Run {run.info.run_id} has no logged models at all")
    for output in run.outputs.model_outputs:
        logged_model = client.get_logged_model(output.model_id)
        if logged_model.name == artifact_path:
            return logged_model.model_uri
    names = [client.get_logged_model(o.model_id).name for o in run.outputs.model_outputs]
    raise ValueError(f"Run {run.info.run_id} has no logged model named {artifact_path!r} - found: {names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--registered_name", required=True)
    parser.add_argument("--metric_key", required=True)
    parser.add_argument("--artifact_path", default="model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--min_improvement", type=float, default=0.0)
    args = parser.parse_args()
    run(
        args.experiment_name, args.registered_name, args.metric_key,
        args.artifact_path, args.alias, args.min_improvement,
    )


if __name__ == "__main__":
    main()
