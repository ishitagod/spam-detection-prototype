"""
Loads + prepares the supervised (`rule_pattern_score`) training data:
messages_with_behavioral.csv, filtered to rule_evaluated==True, labelled
by rule_flagged (True=spam, False=confirmed-clean).

Different scope from models/anomaly/data.py: LightGBM's features are
canonical + behavioral + source, not embeddings/FAISS. Reads straight
from messages_with_behavioral.csv (full dataset, every source), not
Isolation Forest's sampled subset. Real pool: 2,693 SMPP + 349,962 SS7
rule_evaluated rows.

Features (narrower than Isolation Forest's):
  - behavioral: same 4 columns as models/anomaly/data.py
  - canonical: dcs, text_decode_failed, text_length
  - content-rule flags (features/content_flags.py, CONTENT_FLAG_COLS)
  - source (one-hot)
Excluded: originator/destination - too high-cardinality to one-hot
without overfitting risk; behavioral features already capture
originator-level behavior. CatBoost's native categorical handling is the
right place to revisit this later.

Not scaled - tree-based LightGBM splits are scale-invariant.

Label: rule_flagged, restricted to rule_evaluated==True - never
decision==1 directly (~7.5% of SS7's decision==1 rows are non-spam fraud
types; rule_flagged encodes fraud_type=="spam" specifically).

Opt-in second label pool: load_unevaluated_messages() +
label_content_flagged_positives() below add confident positives (never
negatives) from rule_evaluated==False rows, scored by a
LogisticRegression fit on real rule_flagged labels
(labels/rule_labels.py::fit_content_flag_weights()/
content_flagged_by_weight()), gated behind train.py's
--include_content_labels. Kept in its own label_source column - never
silently merged with the base pool.

Embeddings/TF-IDF are optional and independently toggleable
(use_embeddings=, use_tfidf= on build_feature_matrix()). A baseline
model's own feature importances show `text_length` as its top feature by
a wide margin, suggesting real content would help. Confirmed for TF-IDF:
a standalone TfidfVectorizer+LogisticRegression test, split by unique
text, scored PR-AUC 0.934 vs a 0.669 naive baseline.

use_embeddings is usable: features/text_embeddings.py's full-dataset run
has completed, covering 100% of both sources' rule_evaluated pool
(139,546/139,546 SMPP, 2,654,369/2,654,369 SS7). See
docs/experiments/rule_pattern.md.

Embeddings and TF-IDF are independent flags, not combined: TF-IDF
recognizes literal repeated templates (this dataset's spam is heavily
templated); embeddings could generalize to semantically-similar but
differently-worded spam. LightGBM handles both together without one
dominating (unlike Isolation Forest's distance-based scoring, see
scripts/check_embedding_dominance.py) since tree splits evaluate each
feature's information gain independently.

Fit-on-train-only: embedding PCA and TF-IDF vocabulary are
corpus-dependent transformers - fitting on the full pool would leak
test-set information into featurization. build_feature_matrix() takes a
`train_mask` for this: fit on df[train_mask] only, transform every row.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from labels.rule_labels import (
    LABEL_SOURCE_CONTENT_STATIC_RULES,
    LABEL_SOURCE_TELECOM_RULE_ENGINE,
    content_flagged_by_weight,
)
from models.anomaly.data import (
    BEHAVIORAL_COLS,
    CONTENT_FLAG_COLS,
    CONTENT_FLAG_META_COLS,
    IMSI_DISTINCT_ORIG_COL,
    N_EMBEDDING_COMPONENTS,
    SENDER_VELOCITY_ZSCORE_COL,
    compute_content_flag_meta_features,
    embedding_pca_pipeline,
)

CANONICAL_COLS = ["dcs", "text_decode_failed"]
REQUIRED_COLS = (
    ["source", "record_id", "rule_evaluated", "rule_flagged", "text"]
    + CANONICAL_COLS
    + BEHAVIORAL_COLS
    + [IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL]
    + CONTENT_FLAG_COLS
)

# Validated empirically (see module docstring) on the full SS7 corpus,
# not tuned against a target metric - a reasonable starting point.
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 5


def _load_messages_csv(messages_path: Path) -> pd.DataFrame:
    """
    Shared dtype-explicit CSV read behind load_labelled_messages() and
    load_content_labelled_messages() below - every row of one source's
    full messages_with_behavioral.csv, unfiltered.

    No low_memory=False here: this reads up to 5.5M rows in one call, and
    low_memory=False forces pandas to buffer the whole file for one-shot
    dtype inference, which ran out of memory. Explicit dtypes below make
    inference unnecessary - faster and lower-memory for a file this size.
    """
    messages_path = Path(messages_path)
    dtypes = {
        "source": str,
        "record_id": str,
        "rule_evaluated": bool,
        "dcs": "float64",
        "text_decode_failed": bool,
        "sender_msgs_last_5min": "int64",
        "sender_msgs_last_1hr": "int64",
        "sender_unique_destinations_1hr": "int64",
        "sender_repeat_content_ratio_1hr": "float64",
        "sender_age_days": "float64",
        "sender_recipient_diversity_ratio_5min": "float64",
        "sender_recipient_diversity_ratio_1hr": "float64",
        "sender_velocity_zscore_5min": "float64",  # can be NaN
        # Nullable extension dtype, not plain bool: rule_flagged is
        # genuinely True/False/NA (NA is a real, distinct value) - without
        # this pandas sees inconsistent types across chunks and warns.
        "rule_flagged": "boolean",
    }
    df = pd.read_csv(
        messages_path,
        usecols=lambda c: c in set(REQUIRED_COLS),
        dtype=dtypes,  # `text` left out - free text doesn't fit a fixed dtype
    )
    # IMSI_DISTINCT_ORIG_COL is SS7-only, absent from SMPP's file - add
    # back as all-NaN so every caller sees the same column regardless of
    # source (LightGBM treats NaN as a genuine missing split).
    if IMSI_DISTINCT_ORIG_COL not in df.columns:
        df[IMSI_DISTINCT_ORIG_COL] = np.nan
    # Same treatment for content flags, but 0 (no flags known) not NaN.
    for col in CONTENT_FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


def load_labelled_messages(messages_path: Path) -> pd.DataFrame:
    """
    Rows with rule_evaluated==True only, from one source's full
    messages_with_behavioral.csv - the telecom-rule-engine-derived label
    pool. Caller concatenates across sources. label_source is always
    tagged LABEL_SOURCE_TELECOM_RULE_ENGINE so a caller mixing in
    load_content_labelled_messages() below can tell the pools apart.
    """
    df = _load_messages_csv(messages_path)
    df = df[df["rule_evaluated"] == True].copy()  # noqa: E712
    df["label_source"] = LABEL_SOURCE_TELECOM_RULE_ENGINE
    return df


def load_unevaluated_messages(messages_path: Path) -> pd.DataFrame:
    """
    Rows with rule_evaluated==False only, from one source's full
    messages_with_behavioral.csv - the pool label_content_flagged_positives()
    below draws candidate positives from. rule_flagged is NA for every
    row here, unlike load_labelled_messages().
    """
    df = _load_messages_csv(messages_path)
    return df[df["rule_evaluated"] == False].copy()  # noqa: E712


def label_content_flagged_positives(
    unevaluated_df: pd.DataFrame, weight_model, threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Confident positives only, from a rule_evaluated==False pool
    (load_unevaluated_messages()) - expands the training pool using
    labels/rule_labels.py::content_flagged_by_weight(): `weight_model` is
    a LogisticRegression fit by fit_content_flag_weights() on the real
    labelled pool, so each content flag counts toward the label in
    proportion to how well it actually predicted rule_flagged.

    No negatives come from this pool: a row scoring below `threshold` is
    not labelled clean - a low content-flag score doesn't mean "not spam",
    only a confident positive hit is a strong enough signal to use.

    label_source is tagged LABEL_SOURCE_CONTENT_STATIC_RULES - never
    blended with load_labelled_messages()'s telecom-derived rows, since
    these rows' label is derived from the same CONTENT_FLAG_COLS this
    model also uses as features (real risk of an inflated-looking metric
    on this slice, disclosed via the label_source breakdown in
    --include_content_labels).
    """
    positives = unevaluated_df[
        content_flagged_by_weight(unevaluated_df, weight_model, threshold=threshold)
    ].copy()
    positives["rule_flagged"] = True
    positives["label_source"] = LABEL_SOURCE_CONTENT_STATIC_RULES
    return positives


def _base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    The canonical + behavioral + source columns build_feature_matrix()
    always includes, regardless of use_embeddings/use_tfidf. `dcs` can be
    NaN - left as-is, LightGBM has native missing-value handling.
    """
    text_length = df["text"].fillna("").str.len().rename("text_length")
    text_decode_failed = (
        df["text_decode_failed"].astype(int).rename("text_decode_failed")
    )
    # Absent for a caller that skipped load_labelled_messages() (e.g. a
    # test fixture) - same NaN-means-unknown treatment as `dcs`.
    if IMSI_DISTINCT_ORIG_COL in df.columns:
        imsi_col = df[[IMSI_DISTINCT_ORIG_COL]]
    else:
        imsi_col = pd.DataFrame({IMSI_DISTINCT_ORIG_COL: np.nan}, index=df.index)
    if SENDER_VELOCITY_ZSCORE_COL in df.columns:
        velocity_col = df[[SENDER_VELOCITY_ZSCORE_COL]]
    else:
        velocity_col = pd.DataFrame(
            {SENDER_VELOCITY_ZSCORE_COL: np.nan}, index=df.index
        )
    pieces = [
        df[BEHAVIORAL_COLS],
        imsi_col,
        velocity_col,
        df[["dcs"]],
        text_decode_failed,
        text_length,
        df[CONTENT_FLAG_COLS],
        compute_content_flag_meta_features(df),
    ]
    # `source` is dead weight once a run is restricted to one source -
    # only add it when df actually spans more than one.
    if df["source"].nunique() > 1:
        pieces.append(pd.get_dummies(df["source"], prefix="source"))
    return pd.concat(pieces, axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    train_mask: np.ndarray | None = None,
    use_embeddings: bool = False,
    use_tfidf: bool = False,
    n_embedding_components: int = N_EMBEDDING_COMPONENTS,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
    tfidf_ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    tfidf_min_df: int = TFIDF_MIN_DF,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Returns (X, y, feature_names, fitted). X/y cover all rows of df in its
    original order (caller slices by idx_train/idx_test). `fitted` is a
    dict of whichever corpus-dependent transformers were actually used
    ({"embedding_pca_pipeline": ..., "tfidf_vectorizer": ...}), empty if
    neither flag is set. Both must be reused unchanged at inference time.

    `train_mask`: boolean array, same length as df, True = training fold.
    Embeddings PCA and TF-IDF vocabulary are fit on df[train_mask] only,
    then applied to every row (see module docstring). Defaults to
    all-True when omitted; models/rule_pattern/train.py must always pass
    a real mask whenever use_embeddings or use_tfidf is True.
    """
    if train_mask is None:
        train_mask = np.ones(len(df), dtype=bool)

    base = _base_feature_frame(df)
    pieces = [base]
    fitted: dict = {}

    if use_embeddings:
        embedding_cols = [c for c in df.columns if c.startswith("emb_")]
        pca_pipeline = embedding_pca_pipeline(n_embedding_components)
        pca_pipeline.fit(df.loc[train_mask, embedding_cols].to_numpy(dtype=np.float64))
        embeddings_reduced = pca_pipeline.transform(
            df[embedding_cols].to_numpy(dtype=np.float64)
        )
        embedding_names = [f"emb_pca_{i}" for i in range(n_embedding_components)]
        pieces.append(
            pd.DataFrame(embeddings_reduced, columns=embedding_names, index=df.index)
        )
        fitted["embedding_pca_pipeline"] = pca_pipeline

    if use_tfidf:
        text = df["text"].fillna("")
        vectorizer = TfidfVectorizer(
            ngram_range=tfidf_ngram_range,
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
        )
        vectorizer.fit(text[train_mask])
        tfidf_matrix = vectorizer.transform(text).toarray()
        tfidf_names = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
        pieces.append(pd.DataFrame(tfidf_matrix, columns=tfidf_names, index=df.index))
        fitted["tfidf_vectorizer"] = vectorizer

    matrix = pd.concat(pieces, axis=1)
    y = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712
    return matrix.to_numpy(dtype=np.float64), y, matrix.columns.tolist(), fitted


def join_embeddings(df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """
    Inner join of any frame with source/record_id columns against
    features/text_embeddings.py's output (emb_0..emb_{d-1}) for
    `source_dir` - shared by load_labelled_messages_with_embeddings()
    below and train.py's --include_content_labels --with_embeddings path.

    Coverage depends on `source_dir`'s own embeddings.npy - full-dataset
    for SS7, still absent for SMPP. Silently restricts to whatever's
    present - pass a single source_dir/df pair per source.
    """
    source_dir = Path(source_dir)
    df = df.copy()
    df["message_key"] = df["source"] + "|" + df["record_id"]

    embeddings = np.load(source_dir / "embeddings.npy")
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    emb_df = pd.DataFrame(
        embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )
    emb_df["message_key"] = id_map["message_key"].to_numpy()

    return df.merge(emb_df, on="message_key", how="inner").reset_index(drop=True)


def load_labelled_messages_with_embeddings(
    source_dir: Path, messages_path: Path
) -> pd.DataFrame:
    """
    Same rule_evaluated==True filter as load_labelled_messages(), inner
    joined with embeddings via join_embeddings() above. See that
    function's docstring for coverage caveats.
    """
    return join_embeddings(load_labelled_messages(messages_path), Path(source_dir))
