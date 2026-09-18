import ast
import sys
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MLFLOW_TRACKING_URI
from models.rule_pattern.data import build_feature_matrix, load_labelled_messages, load_labelled_messages_with_embeddings

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
run_id = "f7da9adb39e249c580cb4c7408741319"
run = mlflow.get_run(run_id)
params = run.data.params
logged_feature_names = mlflow.artifacts.load_dict(f"runs:/{run_id}/feature_names.json")["feature_names"]

sources = params["sources"].split(",")
data_dir = Path("data/processed")
with_embeddings = params.get("with_embeddings") == "True"
with_tfidf = params.get("with_tfidf") == "True"

if with_embeddings:
    frames = [load_labelled_messages_with_embeddings(data_dir / s, data_dir / s / "messages_with_behavioral.csv") for s in sources]
else:
    frames = [load_labelled_messages(data_dir / s / "messages_with_behavioral.csv") for s in sources]
df = pd.concat(frames, ignore_index=True)
y_full = (df["rule_flagged"] == True).astype(int).to_numpy()

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=float(params["test_size"]), stratify=y_full, random_state=int(params["random_state"]))
train_mask = np.zeros(len(df), dtype=bool)
train_mask[idx_train] = True

tfidf_ngram_range = ast.literal_eval(params["tfidf_ngram_range"]) if "tfidf_ngram_range" in params else (1, 3)
_, _, feature_names, _ = build_feature_matrix(
    df, train_mask=train_mask, use_embeddings=with_embeddings,
    n_embedding_components=int(params.get("n_embedding_components", 30)),
    use_tfidf=with_tfidf, tfidf_max_features=int(params.get("tfidf_max_features", 500)),
    tfidf_ngram_range=tfidf_ngram_range, tfidf_min_df=int(params.get("tfidf_min_df", 5)),
)

print("rebuilt:", len(feature_names), "logged:", len(logged_feature_names))
print("df rows now:", len(df))
print("in rebuilt not logged:", set(feature_names) - set(logged_feature_names))
print("in logged not rebuilt:", set(logged_feature_names) - set(feature_names))
print("same set, different order:", set(feature_names) == set(logged_feature_names) and feature_names != logged_feature_names)