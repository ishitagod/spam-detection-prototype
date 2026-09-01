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
     unchanged into the same final joint StandardScaler. (The "12" above
     is the ablation's real measured count at the time it ran - two more
     hand-built columns, IMSI_DISTINCT_ORIG_COL and its _known indicator,
     were added after; the imbalance direction the ablation found doesn't
     change from +2 features, so it wasn't worth re-running for this.)
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
# SS7-only SIM-farming signal (features/behavioral.py's IMSI-LINKAGE note) -
# kept OUT of BEHAVIORAL_COLS deliberately: unlike those 4, this column is
# entirely ABSENT (not NaN-filled) from SMPP's messages_with_behavioral.csv,
# so callers that read it need the presence check load_source_features()
# does below, not a plain usecols=[...] that would raise on SMPP. Real SS7
# data also has null imsi on ~31.7% of rows even where the column exists -
# both cases collapse to the same "genuinely unknown" NaN, handled uniformly
# in build_feature_matrix() (fillna(0) + a separate _known indicator, since
# sklearn's Pipeline can't take NaN the way LightGBM natively can).
IMSI_DISTINCT_ORIG_COL = "imsi_distinct_originators_1hr"
IMSI_DISTINCT_ORIG_KNOWN_COL = f"{IMSI_DISTINCT_ORIG_COL}_known"
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
    # Lambda usecols (not a plain list) so a source file missing
    # IMSI_DISTINCT_ORIG_COL entirely (SMPP - see that constant's comment
    # above) is silently skipped rather than raising - a plain list would
    # error on any name not present in the file's header.
    wanted_cols = (
        ["source", "record_id"] + BEHAVIORAL_COLS
        + [IMSI_DISTINCT_ORIG_COL] + ["rule_evaluated", "rule_flagged"]
    )
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=lambda c: c in set(wanted_cols),
    )
    if IMSI_DISTINCT_ORIG_COL not in messages.columns:
        messages[IMSI_DISTINCT_ORIG_COL] = np.nan
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


def build_combined_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    The pre-preprocessor frame build_feature_matrix() fits/transforms -
    factored out so callers that need the RAW input shape the returned
    Pipeline actually expects (e.g. models/anomaly/train.py logging an
    MLflow model signature for the full preprocessor+model pipeline) can
    get it without duplicating this construction. Returns (combined,
    embedding_cols, other_cols) - same three pieces build_feature_matrix()
    passes to build_preprocessor().
    """
    transformed = df.copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])

    # IMSI_DISTINCT_ORIG_COL may be entirely absent (test fixtures, or any
    # caller other than load_source_features() that didn't add it) - treat
    # that the same as "present but NaN", not a required column, so the
    # rest of this function has one code path either way.
    if IMSI_DISTINCT_ORIG_COL not in transformed.columns:
        transformed[IMSI_DISTINCT_ORIG_COL] = np.nan
    # NaN means "genuinely unknown" here (no imsi at all for this source,
    # or a null imsi on this SS7 row) - not zero. fillna(0) alone would
    # fabricate "zero distinct originators" for rows where the thing this
    # feature measures was never observed; the _known indicator lets the
    # model tell the two apart instead of silently conflating them.
    transformed[IMSI_DISTINCT_ORIG_KNOWN_COL] = transformed[IMSI_DISTINCT_ORIG_COL].notna().astype(float)
    transformed[IMSI_DISTINCT_ORIG_COL] = np.log1p(transformed[IMSI_DISTINCT_ORIG_COL].fillna(0))

    # `source` is only a real feature when more than one source is present
    # in this training run - a single-source run (e.g. --sources SMPP for
    # a split model, see CLAUDE.md's "Split by source" note) would produce
    # a constant one-hot column carrying zero information, just dead
    # weight through StandardScaler. Combined-sources runs keep the dummy
    # unchanged - same behavior as before.
    imsi_cols = [IMSI_DISTINCT_ORIG_COL, IMSI_DISTINCT_ORIG_KNOWN_COL]
    embedding_cols = [c for c in transformed.columns if c.startswith("emb_")]
    other_cols = list(BEHAVIORAL_COLS) + list(NEAR_DUP_COLS) + imsi_cols
    pieces = [transformed[BEHAVIORAL_COLS + NEAR_DUP_COLS + imsi_cols]]
    if transformed["source"].nunique() > 1:
        source_dummies = pd.get_dummies(transformed["source"], prefix="source")
        other_cols += list(source_dummies.columns)
        pieces.append(source_dummies)
    pieces.append(transformed[embedding_cols])

    combined = pd.concat(pieces, axis=1)
    return combined, embedding_cols, other_cols


def build_feature_matrix(
    df: pd.DataFrame, n_embedding_components: int = N_EMBEDDING_COMPONENTS,
) -> tuple[np.ndarray, list[str], Pipeline]:
    """
    Returns (X, feature_names, fitted_preprocessor) - see
    build_preprocessor()'s docstring for why the preprocessor is
    returned rather than just applied.
    """
    combined, embedding_cols, other_cols = build_combined_frame(df)

    preprocessor = build_preprocessor(embedding_cols, other_cols, n_embedding_components)
    X = preprocessor.fit_transform(combined)

    explained = preprocessor.named_steps["reduce"].named_transformers_["embeddings"].named_steps["pca"].explained_variance_ratio_
    print(f"  PCA: {n_embedding_components} components retain {explained.sum():.1%} of embedding variance")

    feature_names = [f"emb_pca_{i}" for i in range(n_embedding_components)] + other_cols
    return X, feature_names, preprocessor
