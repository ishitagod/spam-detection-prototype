"""
Serving-side model loading - the only way inference code should load a
trained model. Never load by run_id directly in serving code - loading by
registered-name + alias means promoting a new model
(models/compare_versions.py) is just moving the alias, never a code change.
"""
import mlflow

from config.settings import MLFLOW_TRACKING_URI


def load_champion(registered_name: str, alias: str = "champion"):
    """
    Loads whatever model version currently holds `alias` for
    `registered_name`. Uses the generic pyfunc loader (works across
    sklearn/LightGBM/etc. uniformly) - not for non-model artifacts like
    the standalone embedding PCA pipeline (needs its native
    mlflow.sklearn.load_model() instead, no .predict()/pyfunc wrapper).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{registered_name}@{alias}")
