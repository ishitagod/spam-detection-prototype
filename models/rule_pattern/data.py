"""
Loads + prepares the supervised (`rule_pattern_score`) training data:
messages_with_behavioral.csv, filtered to rule_evaluated==True,
labelled by rule_flagged (True=spam, False=confirmed-clean).

GENUINELY DIFFERENT SCOPE from models/anomaly/data.py, not just a
different model: per README.md's modeling plan, LightGBM's features are
canonical schema + behavioral + source - NOT embeddings, NOT FAISS
near-dup. That means this module reads straight from
messages_with_behavioral.csv (the FULL dataset, every row, every
source), unrestricted by features/text_embeddings.py's sampled subset -
Isolation Forest needs that sample because it needs embeddings;
LightGBM doesn't touch embeddings at all, so it isn't bottlenecked by
it. Real pool: 2,693 SMPP + 349,962 SS7 rule_evaluated rows - much
bigger than Isolation Forest's ~60k-row sample-bounded input.

FEATURES, deliberately narrower than Isolation Forest's:
  - behavioral: the same 4 columns as models/anomaly/data.py, imported
    from there rather than duplicated
  - canonical: dcs, text_decode_failed, plus text_length (a cheap
    derived signal, zero extra cost to add)
  - source (one-hot)
EXCLUDED ON PURPOSE for this first build: originator/destination - too
high-cardinality to one-hot without real overfitting risk on a pool
this size, and behavioral features already capture originator-level
BEHAVIOR (velocity, repeat content) without needing the raw identity.
CatBoost's native categorical handling - already named in README.md as
the reason to benchmark it later - is the right place to revisit this,
not a one-hot hack here.

NOT SCALED: unlike Isolation Forest, tree-based LightGBM splits are
scale-invariant - no StandardScaler needed here.

LABEL: rule_flagged, restricted to rule_evaluated==True - NEVER
decision==1 directly (labels/rule_labels.py found ~7.5% of SS7's
decision==1 rows are non-spam fraud types; rule_flagged already encodes
fraud_type=="spam" specifically, decision alone doesn't).

EMBEDDINGS-AWARE VARIANT (load_labelled_messages_with_embeddings() +
build_feature_matrix_with_embeddings()): "not embeddings" above is a
scoping decision for the DEFAULT path, not a permanent architectural
stance - real evidence points the other way. A trained baseline model's
own feature importances show `text_length` (the only content-adjacent
signal it has) as the single most important feature by a wide margin -
meaning even a crude proxy for content carries real separating power,
so genuine content (real embeddings, not just character count) would
plausibly help more, not be redundant. This variant exists to test that
once features/text_embeddings.py's full-dataset run is done (not just
the current CPU-scale sample - see that module's docstring): as of
writing, the sample only overlaps ~18/2,693 SMPP and ~2,635/349,962 SS7
rule_evaluated rows (under 1% either way), so this path is not yet
useful as the main training path, only ready for when it is. Reuses
models/anomaly/data.py's embedding_pca_pipeline() (StandardScaler + PCA)
for the SAME reason it's needed there: 384 raw dims would swamp this
model's other ~10 features the same way it swamped Isolation Forest's,
even though tree splits themselves don't require scaling.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models.anomaly.data import BEHAVIORAL_COLS, N_EMBEDDING_COMPONENTS, embedding_pca_pipeline

CANONICAL_COLS = ["dcs", "text_decode_failed"]
REQUIRED_COLS = ["source", "record_id", "rule_evaluated", "rule_flagged", "text"] + CANONICAL_COLS + BEHAVIORAL_COLS


def load_labelled_messages(messages_path: Path) -> pd.DataFrame:
    """
    Rows with rule_evaluated==True only, from ONE source's full
    messages_with_behavioral.csv. Caller concatenates across sources.

    NO low_memory=False here, unlike most other CSV reads in this
    codebase (features/behavioral.py, features/text_embeddings.py,
    etc.) - those all operate on a single hourly file or the much
    smaller embedding-sample scale. This is the first reader in the
    project to load a source's ENTIRE messages_with_behavioral.csv in
    one call (up to 5.5M rows, unrestricted by the embedding sample -
    see module docstring for why) - low_memory=False forces pandas to
    buffer the whole file for one-shot dtype inference, which is what
    actually ran out of memory here. Explicit dtypes below make that
    inference unnecessary instead, which is both the fix and the
    faster/lower-memory path for a file this size.
    """
    messages_path = Path(messages_path)
    dtypes = {
        "source": str, "record_id": str, "rule_evaluated": bool,
        "dcs": "float64", "text_decode_failed": bool,
        "sender_msgs_last_5min": "int64", "sender_msgs_last_1hr": "int64",
        "sender_unique_destinations_1hr": "int64", "sender_repeat_content_ratio_1hr": "float64",
        # pandas' nullable extension dtype, not plain bool: rule_flagged
        # is genuinely True/False/NA (labels/rule_labels.py - NA is a
        # real, distinct value, not missing data to impute), and without
        # an explicit dtype here pandas sees different apparent types
        # across chunks (bool-only vs bool+None) and throws a
        # DtypeWarning - this is the fix, not a suppression of it.
        "rule_flagged": "boolean",
    }
    df = pd.read_csv(
        messages_path, usecols=lambda c: c in set(REQUIRED_COLS),
        dtype=dtypes,  # `text` deliberately left out - free text doesn't fit a fixed dtype
    )
    # source/record_id already forced to str via the dtype= dict above -
    # same convention as every other module in this codebase (a real bug
    # hit before: pandas can infer one source file's record_id/
    # originator as int64 and another's as str, which silently breaks
    # downstream joins/dtype consistency once concatenated).
    return df[df["rule_evaluated"] == True].copy()  # noqa: E712


def _base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    The canonical + behavioral + source columns shared by both
    build_feature_matrix() and build_feature_matrix_with_embeddings().
    `dcs` can be NaN in real data - left as-is deliberately, LightGBM
    has native missing-value handling built in, no imputation needed.
    """
    text_length = df["text"].fillna("").str.len().rename("text_length")
    text_decode_failed = df["text_decode_failed"].astype(int).rename("text_decode_failed")
    source_dummies = pd.get_dummies(df["source"], prefix="source")
    return pd.concat(
        [df[BEHAVIORAL_COLS], df[["dcs"]], text_decode_failed, text_length, source_dummies], axis=1,
    )


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, feature_names). y = 1 for rule_flagged (spam), 0 for confirmed-clean."""
    matrix = _base_feature_frame(df)
    y = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712
    return matrix.to_numpy(dtype=np.float64), y, matrix.columns.tolist()


def load_labelled_messages_with_embeddings(source_dir: Path, messages_path: Path) -> pd.DataFrame:
    """
    Same rule_evaluated==True filter as load_labelled_messages(), INNER
    JOINED with features/text_embeddings.py's output (emb_0..emb_{d-1}).
    See module docstring - this is restricted to whatever the embedding
    sample currently covers, which as of writing is under 1% of either
    source's rule_evaluated pool. Ready for the full run, not useful as
    the main training path today.
    """
    source_dir = Path(source_dir)
    df = load_labelled_messages(messages_path)
    df = df.copy()
    df["message_key"] = df["source"] + "|" + df["record_id"]

    embeddings = np.load(source_dir / "embeddings.npy")
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df["message_key"] = id_map["message_key"].to_numpy()

    return df.merge(emb_df, on="message_key", how="inner").reset_index(drop=True)


def build_feature_matrix_with_embeddings(
    df: pd.DataFrame, n_embedding_components: int = N_EMBEDDING_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, list[str], Pipeline]:
    """
    Same base features as build_feature_matrix(), PLUS PCA-reduced
    embeddings. `df` must come from load_labelled_messages_with_embeddings()
    (needs emb_* columns present). Embeddings get scaled+PCA'd (needs
    it - see models/anomaly/data.py's embedding_pca_pipeline()); the
    base features stay unscaled, same as build_feature_matrix() - tree
    splits don't need it. Returns the fitted PCA pipeline too - same
    reuse-at-inference requirement as models/anomaly/data.py's
    preprocessor (train/serve skew otherwise).
    """
    base = _base_feature_frame(df)
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    pca_pipeline = embedding_pca_pipeline(n_embedding_components)
    embeddings_reduced = pca_pipeline.fit_transform(df[embedding_cols].to_numpy(dtype=np.float64))
    embedding_names = [f"emb_pca_{i}" for i in range(n_embedding_components)]

    matrix = pd.concat([base, pd.DataFrame(embeddings_reduced, columns=embedding_names)], axis=1)
    y = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712
    return matrix.to_numpy(dtype=np.float64), y, matrix.columns.tolist(), pca_pipeline
