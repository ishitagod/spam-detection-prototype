"""
pytest suite for models.compare_versions and models.registry.

Uses a real, isolated MLflow store per test (tmp_path-scoped sqlite
file) - genuine integration coverage of the registry/alias mechanics,
not mocked, same preference this project uses everywhere else.

Run:
    pytest tests/test_compare_versions.py -v
"""
import sys
from pathlib import Path

import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.compare_versions import _find_model_uri, get_champion_metric, get_latest_run, run
from models.registry import load_champion


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Isolated tracking URI for this test only - prevents cross-test
    state leakage and never touches the real project mlflow.db."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr("models.compare_versions.MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("models.registry.MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri


def _log_run(experiment_name: str, metric_key: str, metric_value: float, artifact_path: str = "model"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as active_run:
        mlflow.log_metric(metric_key, metric_value)
        model = LogisticRegression().fit([[0], [1]], [0, 1])  # trivial, real, cheap
        mlflow.sklearn.log_model(model, name=artifact_path)
        return active_run.info.run_id


def test_bootstrap_promotes_when_no_champion_exists(mlflow_store):
    _log_run("exp1", "pr_auc", 0.7)
    run("exp1", "model1", "pr_auc")
    champion = get_champion_metric("model1", "champion", "pr_auc")
    assert champion == 0.7


def test_better_challenger_gets_promoted(mlflow_store):
    _log_run("exp1", "pr_auc", 0.7)
    run("exp1", "model1", "pr_auc")  # bootstrap champion at 0.7

    _log_run("exp1", "pr_auc", 0.9)  # a better challenger
    run("exp1", "model1", "pr_auc")

    assert get_champion_metric("model1", "champion", "pr_auc") == 0.9


def test_worse_challenger_does_not_get_promoted(mlflow_store):
    _log_run("exp1", "pr_auc", 0.9)
    run("exp1", "model1", "pr_auc")  # bootstrap champion at 0.9

    _log_run("exp1", "pr_auc", 0.7)  # a worse challenger
    run("exp1", "model1", "pr_auc")

    assert get_champion_metric("model1", "champion", "pr_auc") == 0.9  # unchanged


def test_min_improvement_threshold_blocks_marginal_gains(mlflow_store):
    _log_run("exp1", "pr_auc", 0.900)
    run("exp1", "model1", "pr_auc")  # bootstrap at 0.900

    _log_run("exp1", "pr_auc", 0.901)  # technically better, but tiny
    run("exp1", "model1", "pr_auc", min_improvement=0.01)

    assert get_champion_metric("model1", "champion", "pr_auc") == 0.900  # not promoted


def test_get_latest_run_raises_on_missing_experiment(mlflow_store):
    with pytest.raises(ValueError):
        get_latest_run("does_not_exist")


def test_get_champion_metric_returns_none_when_no_champion_exists(mlflow_store):
    assert get_champion_metric("brand_new_model_name", "champion", "pr_auc") is None


def test_find_model_uri_picks_the_right_model_when_a_run_logs_more_than_one(mlflow_store):
    """The real scenario this was built for: models/rule_pattern/train.py's
    --with_embeddings path logs BOTH the LightGBM model AND a separate
    embedding_pca_pipeline in the same run - must pick "model" by name,
    not just whichever LoggedModel happens to be listed first."""
    mlflow.set_experiment("multi_model_exp")
    with mlflow.start_run() as active_run:
        # log the "other" artifact FIRST, so a naive [0]-index pick would get it wrong
        other = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(other, name="embedding_pca_pipeline")
        real_model = LogisticRegression().fit([[0], [1]], [1, 0])
        mlflow.sklearn.log_model(real_model, name="model")
        run_id = active_run.info.run_id

    client = mlflow.MlflowClient()
    challenger_run = client.get_run(run_id)
    model_uri = _find_model_uri(challenger_run, "model")
    loaded = mlflow.sklearn.load_model(model_uri)
    assert loaded.predict([[0]])[0] == 1  # confirms it's `real_model`, not `other`


def test_find_model_uri_raises_clearly_when_name_not_found(mlflow_store):
    mlflow.set_experiment("exp1")
    with mlflow.start_run() as active_run:
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, name="model")
        run_id = active_run.info.run_id

    client = mlflow.MlflowClient()
    challenger_run = client.get_run(run_id)
    with pytest.raises(ValueError, match="no logged model named"):
        _find_model_uri(challenger_run, "does_not_exist")


def test_load_champion_loads_the_promoted_model(mlflow_store):
    """End-to-end: promote via compare_versions, then actually load it
    back via registry.load_champion() - the real serving-side path."""
    _log_run("exp1", "pr_auc", 0.8)
    run("exp1", "model1", "pr_auc")

    loaded = load_champion("model1")
    prediction = loaded.predict([[0]])
    assert prediction is not None  # a real, usable, loaded model


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
