"""
Joins the three independently-computed feature sources into one aligned
matrix for Isolation Forest, by message_key:
  - messages_with_behavioral.csv  (behavioral features + source)
  - embeddings.npy + embeddings_id_map.parquet (MiniLM embeddings)
  - faiss_output.parquet          (near-dup features, both windows)

INNER join, not left: embeddings/faiss currently only cover the sampled
subset built by features/text_embeddings.py's --sample_n (see that
module's docstring for why - the full 8.2M-row encode is a ~21hr job).
Training data is therefore restricted to that same sample, consistently
- not a new limitation introduced here, just carried through explicitly
rather than silently.

`source` IS included as a feature (one-hot), not used to route to
separate models - per CLAUDE.md's "one model to start, not two"
decision. `rule_evaluated`/`rule_flagged` are carried through for
train.py's label-free plausibility check ONLY - never as a training
input, since Isolation Forest is trained on the FULL traffic stream with
zero labels by design (see README.md's modeling plan).

PREPROCESSING, two things, not one:
  1. log1p on heavy-tailed count features - real observed ranges make
     this necessary, not optional: sender_msgs_last_1hr up to 16,971,
     near_dup_match_count_24hr up to 381 vs. embedding dimensions
     confined to roughly [-1, 1].
  2. PCA on the 384 embedding dimensions down to N_EMBEDDING_COMPONENTS,
     BEFORE combining with the 12 hand-built behavioral+near-dup
     features - measured, not assumed, to be necessary: a real ablation
     (scripts/check_embedding_dominance.py) showed the joint model's
     anomaly-score rankings correlated 0.808 with an embeddings-only
     model but only 0.326 with a behavioral-only model - the 384-vs-12
     dimension imbalance was genuinely drowning out the hand-built
     features, not just a theoretical risk. PCA is applied via a
     ColumnTransformer (sklearn.compose) so it only touches the
     embedding columns - the 12 hand-built features pass through
     unchanged into the same final joint StandardScaler.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BEHAVIORAL_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr",
    "sender_unique_destinations_1hr", "sender_repeat_content_ratio_1hr",
]
NEAR_DUP_COLS = [
    "near_dup_match_count_1hr", "near_dup_max_similarity_1hr", "near_dup_distinct_senders_1hr",
    "near_dup_match_count_24hr", "near_dup_max_similarity_24hr", "near_dup_distinct_senders_24hr",
]
# Heavy-tailed count columns that get log1p'd before scaling - ratios
# (already 0-1) and similarity scores (already ~0-1) are left alone.
COUNT_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr", "sender_unique_destinations_1hr",
    "near_dup_match_count_1hr", "near_dup_distinct_senders_1hr",
    "near_dup_match_count_24hr", "near_dup_distinct_senders_24hr",
]

# Not tuned against a real target explained-variance threshold - a
# starting point that brings 384 down to something closer in order of
# magnitude to the 12 hand-built features, per the ablation finding
# above. build_feature_matrix() prints the actual retained variance at
# this component count every run, so this number stays honest rather
# than a one-time guess nobody checks again.
N_EMBEDDING_COMPONENTS = 30


def load_source_features(source_dir: Path, messages_path: Path) -> pd.DataFrame:
    """
    One row per SAMPLED message (see module docstring on the inner-join
    restriction), with behavioral + near_dup + embedding columns
    (`emb_0`..`emb_{d-1}`) plus `source`/`rule_evaluated`/`rule_flagged`
    carried through unscaled, for the caller to split off before/after
    building the model's actual input matrix.
    """
    source_dir = Path(source_dir)
    messages = pd.read_csv(
        messages_path, low_memory=False,
        usecols=["source", "record_id"] + BEHAVIORAL_COLS + ["rule_evaluated", "rule_flagged"],
    )
    messages["source"] = messages["source"].astype(str)
    messages["record_id"] = messages["record_id"].astype(str)
    messages["message_key"] = messages["source"] + "|" + messages["record_id"]

    embeddings = np.load(source_dir / "embeddings.npy")
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df["message_key"] = id_map["message_key"].to_numpy()

    near_dup = pd.read_parquet(source_dir / "faiss_output.parquet")

    df = messages.merge(emb_df, on="message_key", how="inner")
    df = df.merge(near_dup, on="message_key", how="inner")
    return df


def embedding_pca_pipeline(n_embedding_components: int) -> Pipeline:
    """
    StandardScaler -> PCA(n_embedding_components), for embedding columns
    ONLY - shared by this module's build_preprocessor() (which wraps it
    in a final joint scaler too, since Isolation Forest needs everything
    on a comparable scale) and models/rule_pattern/data.py's embeddings-
    aware feature matrix (which does NOT scale its other features - tree
    splits don't need it - so it uses this piece alone, not the full
    build_preprocessor()). One fit, reused everywhere the embedding-PCA
    step itself is needed, rather than two models independently
    re-deriving the same reduction.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_embedding_components, random_state=42)),
    ])


def build_preprocessor(embedding_cols: list[str], other_cols: list[str], n_embedding_components: int) -> Pipeline:
    """
    embedding_cols -> embedding_pca_pipeline(); other_cols -> passthrough
    (already log1p'd/one-hot by the caller); both concatenated, THEN a
    final StandardScaler over the combined result - the PCA components
    themselves have very unequal variance (the first component always
    varies far more than the last), so re-scaling after PCA matters for
    the same reason scaling mattered before it.

    Returned as a single fitted-once, reused-everywhere Pipeline - this
    IS the artifact that must travel to inference unchanged, not
    something to refit on new data (train/serve skew otherwise).
    """
    reduce = ColumnTransformer([
        ("embeddings", embedding_pca_pipeline(n_embedding_components), embedding_cols),
        ("other", "passthrough", other_cols),
    ])
    return Pipeline([
        ("reduce", reduce),
        ("final_scale", StandardScaler()),
    ])


def build_feature_matrix(
    df: pd.DataFrame, n_embedding_components: int = N_EMBEDDING_COMPONENTS,
) -> tuple[np.ndarray, list[str], Pipeline]:
    """
    Returns (X, feature_names, fitted_preprocessor) - see
    build_preprocessor()'s docstring for why the preprocessor is
    returned rather than just applied.
    """
    transformed = df.copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])

    source_dummies = pd.get_dummies(transformed["source"], prefix="source")
    embedding_cols = [c for c in transformed.columns if c.startswith("emb_")]
    other_cols = BEHAVIORAL_COLS + NEAR_DUP_COLS + list(source_dummies.columns)

    combined = pd.concat([transformed[BEHAVIORAL_COLS + NEAR_DUP_COLS], source_dummies, transformed[embedding_cols]], axis=1)

    preprocessor = build_preprocessor(embedding_cols, other_cols, n_embedding_components)
    X = preprocessor.fit_transform(combined)

    explained = preprocessor.named_steps["reduce"].named_transformers_["embeddings"].named_steps["pca"].explained_variance_ratio_
    print(f"  PCA: {n_embedding_components} components retain {explained.sum():.1%} of embedding variance")

    feature_names = [f"emb_pca_{i}" for i in range(n_embedding_components)] + other_cols
    return X, feature_names, preprocessor
