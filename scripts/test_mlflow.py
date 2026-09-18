import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.lightgbm.load_model("models:/rule_pattern_score_model_SS7@champion")
feature_names = mlflow.artifacts.load_dict(
    f"runs:/{mlflow.MlflowClient().get_model_version_by_alias('rule_pattern_score_model_SS7', 'champion').run_id}/feature_names.json"
)["feature_names"]

for name, imp in sorted(
    zip(feature_names, model.feature_importances_), key=lambda x: -x[1]
)[:20]:
    print(f"{imp:6d}  {name}")
