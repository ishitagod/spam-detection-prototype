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

EMBEDDINGS/TF-IDF, both OPTIONAL and INDEPENDENTLY toggleable
(use_embeddings=, use_tfidf= on build_feature_matrix() below): "not
embeddings" in the module summary above is a scoping decision for the
DEFAULT path, not a permanent architectural stance - real evidence points
the other way. A trained baseline model's own feature importances show
`text_length` (the only content-adjacent signal it has) as the single
most important feature by a wide margin - meaning even a crude proxy for
content carries real separating power, so genuine content would plausibly
help more, not be redundant. Confirmed empirically for TF-IDF specifically:
a standalone TfidfVectorizer+LogisticRegression test on the full real SS7
rule_evaluated pool, split by UNIQUE TEXT (no template leaking across
train/test), scored PR-AUC 0.934 vs a 0.669 naive baseline - real
generalizing signal, not just template memorization.

use_embeddings is NOT yet useful as the main training path: the MiniLM
embedding sample only overlaps ~1% of the rule_evaluated pool (under 1%
for both sources as of writing - see load_labelled_messages_with_embeddings()),
because features/text_embeddings.py's full-dataset run (~21hr CPU job) is
still sample-scale. This flag exists so the comparison is one command
away once that full run lands.

Embeddings and TF-IDF are deliberately independent flags, not one combined
"with_content" toggle: they catch different things in real spam here - TF-IDF
is good at recognizing literal repeated TEMPLATES (this dataset's real spam
is heavily templated), embeddings are the ones that could plausibly
generalize to spam that's semantically similar but not worded the same.
LightGBM can use both together without one dominating the other the way
raw embeddings dominated Isolation Forest's joint distance-based scoring
(scripts/check_embedding_dominance.py) - tree splits evaluate each
feature's information gain independently, they don't blend into one
distance metric, so the 384-vs-500-vs-~10 dimension imbalance that
mattered for Isolation Forest isn't the same risk here.

FIT-ON-TRAIN-ONLY: both the embedding PCA and the TF-IDF vocabulary are
corpus-dependent transformers - fitting them on the full pool (train+test
together) would leak test-set distribution/vocabulary into featurization
itself, before the model ever sees a train/test split. build_feature_matrix()
takes a `train_mask` for exactly this reason: fit happens on
df[train_mask] only, transform applies to every row. (This also FIXES a
pre-existing minor leak: the previous embeddings-only path fit PCA on the
full df before train.py's split - harmless in practice for PCA, but worth
closing now that TF-IDF's vocabulary fit makes the same mistake far more
consequential.)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from models.anomaly.data import BEHAVIORAL_COLS, N_EMBEDDING_COMPONENTS, embedding_pca_pipeline

CANONICAL_COLS = ["dcs", "text_decode_failed"]
REQUIRED_COLS = ["source", "record_id", "rule_evaluated", "rule_flagged", "text"] + CANONICAL_COLS + BEHAVIORAL_COLS

# Validated empirically (see module docstring) on the full real SS7 corpus,
# not tuned against a target metric - a reasonable starting point, same
# spirit as N_EMBEDDING_COMPONENTS above.
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 5


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
    The canonical + behavioral + source columns build_feature_matrix()
    always includes, regardless of use_embeddings/use_tfidf.
    `dcs` can be NaN in real data - left as-is deliberately, LightGBM
    has native missing-value handling built in, no imputation needed.
    """
    text_length = df["text"].fillna("").str.len().rename("text_length")
    text_decode_failed = df["text_decode_failed"].astype(int).rename("text_decode_failed")
    source_dummies = pd.get_dummies(df["source"], prefix="source")
    return pd.concat(
        [df[BEHAVIORAL_COLS], df[["dcs"]], text_decode_failed, text_length, source_dummies], axis=1,
    )


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
    Returns (X, y, feature_names, fitted). X/y cover ALL rows of df in its
    original order (caller slices by idx_train/idx_test) - `fitted` is a
    dict of whichever corpus-dependent transformers were actually used
    ({"embedding_pca_pipeline": ..., "tfidf_vectorizer": ...}, only the
    keys for flags that were True), empty if neither use_embeddings nor
    use_tfidf is set. Both must be reused unchanged at inference time,
    same reason as models/anomaly/data.py's preprocessor.

    `train_mask`: boolean array, same length as df, True = this row is in
    the training fold. Embeddings PCA and TF-IDF vocabulary are FIT on
    df[train_mask] ONLY, then applied (.transform()) to every row - see
    module docstring for why this matters (test-set leakage into
    featurization itself, not just into the model). Defaults to "every
    row is train" (all-True) when omitted - correct for the base-features-
    only path (nothing here is corpus-fit) and for tests that don't care
    about train/test leakage, but the real training entrypoint
    (models/rule_pattern/train.py) must always pass a real mask whenever
    use_embeddings or use_tfidf is True.
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
        embeddings_reduced = pca_pipeline.transform(df[embedding_cols].to_numpy(dtype=np.float64))
        embedding_names = [f"emb_pca_{i}" for i in range(n_embedding_components)]
        pieces.append(pd.DataFrame(embeddings_reduced, columns=embedding_names, index=df.index))
        fitted["embedding_pca_pipeline"] = pca_pipeline

    if use_tfidf:
        text = df["text"].fillna("")
        vectorizer = TfidfVectorizer(
            ngram_range=tfidf_ngram_range, max_features=tfidf_max_features, min_df=tfidf_min_df,
        )
        vectorizer.fit(text[train_mask])
        tfidf_matrix = vectorizer.transform(text).toarray()
        tfidf_names = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
        pieces.append(pd.DataFrame(tfidf_matrix, columns=tfidf_names, index=df.index))
        fitted["tfidf_vectorizer"] = vectorizer

    matrix = pd.concat(pieces, axis=1)
    y = (df["rule_flagged"] == True).astype(int).to_numpy()  # noqa: E712
    return matrix.to_numpy(dtype=np.float64), y, matrix.columns.tolist(), fitted


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
