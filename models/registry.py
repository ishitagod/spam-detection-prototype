"""
Serving-side model loading - the ONLY way inference code (eventually
FastAPI) should load a trained model. Never load by run_id directly in
serving code - that hardcodes one specific training run into the
service. Loading by registered-name + alias means promoting a new
model (models/compare_versions.py) is just moving the alias, never a
serving code change.
"""
import mlflow

from config.settings import MLFLOW_TRACKING_URI


def load_champion(registered_name: str, alias: str = "champion"):
    """
    Loads whatever model version currently holds `alias` for
    `registered_name`. Uses the generic pyfunc loader (works across
    sklearn/LightGBM/etc. flavors uniformly) - NOT for artifacts that
    aren't themselves predictive models (e.g. the standalone embedding
    PCA pipeline models/rule_pattern/train.py --with_embeddings logs
    separately - that one has no .predict(), no pyfunc wrapper exists
    for it, and needs its native mlflow.sklearn.load_model() instead).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{registered_name}@{alias}")
