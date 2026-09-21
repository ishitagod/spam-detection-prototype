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
  - content-rule flags (features/content_flags.py, CONTENT_FLAG_COLS):
    base features here too, same as Isolation Forest - see that module's
    CONTENT_FLAG_COLS comment and the architecture plan's Section 3 for
    why these aren't gated behind use_embeddings/use_tfidf the way
    corpus-fit TF-IDF/embeddings are
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

OPT-IN SECOND LABEL POOL: load_unevaluated_messages() +
label_content_flagged_positives() below add confident POSITIVES (never
negatives) from rule_evaluated==False rows, scored by a LogisticRegression
fit on the REAL rule_flagged labels (labels/rule_labels.py::
fit_content_flag_weights()/content_flagged_by_weight()) instead of an
unweighted flag count - gated behind models/rule_pattern/train.py's
--include_content_labels, never in the default load_labelled_messages()
path. Kept in its own label_source column, same "never silently merge
label sources" rule as labels/rule_labels.py's content_flagged() -
CONTENT_FLAG_COLS are also features here, so this pool's label is still
derived from them; see label_content_flagged_positives()'s docstring for
the full reasoning.

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

use_embeddings is NOW actually usable, not yet tried: features/
text_embeddings.py's full-dataset run (previously a ~21hr blocker) has
completed - embeddings now cover 100% of both sources' rule_evaluated
pool (139,546/139,546 SMPP, 2,654,369/2,654,369 SS7 - verified against
embeddings_id_map.parquet), not the ~1% sample-era overlap this docstring
used to describe. The `--with_embeddings` comparison in
models/rule_pattern/train.py is a real, runnable experiment now - see
docs/experiments/rule_pattern.md for the case for running it (baseline
`text_length` feature importance) and its result once run.

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

from labels.rule_labels import (
    LABEL_SOURCE_CONTENT_STATIC_RULES,
    LABEL_SOURCE_TELECOM_RULE_ENGINE,
    content_flagged_by_weight,
)
from models.anomaly.data import (
    BEHAVIORAL_COLS,
    CONTENT_FLAG_COLS,
    IMSI_DISTINCT_ORIG_COL,
    N_EMBEDDING_COMPONENTS,
    SENDER_VELOCITY_ZSCORE_COL,
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

# Validated empirically (see module docstring) on the full real SS7 corpus,
# not tuned against a target metric - a reasonable starting point, same
# spirit as N_EMBEDDING_COMPONENTS above.
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 5


def _load_messages_csv(messages_path: Path) -> pd.DataFrame:
    """
    Shared dtype-explicit CSV read behind load_labelled_messages() and
    load_content_labelled_messages() below - every row of ONE source's
    full messages_with_behavioral.csv, unfiltered (callers pick their own
    rule_evaluated slice).

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
        "sender_velocity_zscore_5min": "float64",  # can be NaN - a plain
        # float64 column already handles that fine, unlike rule_flagged's
        # nullable "boolean" below (True/False/NA trichotomy needs the
        # extension dtype; a NaN float doesn't need one).
        # pandas' nullable extension dtype, not plain bool: rule_flagged
        # is genuinely True/False/NA (labels/rule_labels.py - NA is a
        # real, distinct value, not missing data to impute), and without
        # an explicit dtype here pandas sees different apparent types
        # across chunks (bool-only vs bool+None) and throws a
        # DtypeWarning - this is the fix, not a suppression of it.
        "rule_flagged": "boolean",
    }
    df = pd.read_csv(
        messages_path,
        usecols=lambda c: c in set(REQUIRED_COLS),
        dtype=dtypes,  # `text` deliberately left out - free text doesn't fit a fixed dtype
    )
    # No dtype entry above for IMSI_DISTINCT_ORIG_COL on purpose - it's
    # SS7-only (see models/anomaly/data.py's comment) and entirely absent
    # from SMPP's file, so the lambda usecols() above silently drops it
    # for SMPP rather than erroring. Add it back as all-NaN so every
    # caller sees the same column regardless of source - LightGBM treats
    # NaN as a genuine "missing" split, no imputation needed (same
    # handling as `dcs` above).
    if IMSI_DISTINCT_ORIG_COL not in df.columns:
        df[IMSI_DISTINCT_ORIG_COL] = np.nan
    # Same "absent -> default, not a required column" treatment as IMSI
    # above, but 0 (no flags known) rather than NaN - see
    # models/anomaly/data.py::load_source_features()'s matching comment.
    for col in CONTENT_FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
    # source/record_id already forced to str via the dtype= dict above -
    # same convention as every other module in this codebase (a real bug
    # hit before: pandas can infer one source file's record_id/
    # originator as int64 and another's as str, which silently breaks
    # downstream joins/dtype consistency once concatenated).
    return df


def load_labelled_messages(messages_path: Path) -> pd.DataFrame:
    """
    Rows with rule_evaluated==True only, from ONE source's full
    messages_with_behavioral.csv - the real, telecom-rule-engine-derived
    label pool. Caller concatenates across sources. label_source is always
    tagged LABEL_SOURCE_TELECOM_RULE_ENGINE (labels/rule_labels.py) so a
    caller that also mixes in load_content_labelled_messages() below can
    tell the two pools apart in evaluation, never by silently assuming
    "everything in df is telecom-labelled".
    """
    df = _load_messages_csv(messages_path)
    df = df[df["rule_evaluated"] == True].copy()  # noqa: E712
    df["label_source"] = LABEL_SOURCE_TELECOM_RULE_ENGINE
    return df


def load_unevaluated_messages(messages_path: Path) -> pd.DataFrame:
    """
    Rows with rule_evaluated==False only, from ONE source's full
    messages_with_behavioral.csv - the pool label_content_flagged_positives()
    below draws candidate positives from. Unlike load_labelled_messages(),
    rule_flagged is NOT a usable label here (NA for every row in this
    pool, by definition of rule_evaluated==False).
    """
    df = _load_messages_csv(messages_path)
    return df[df["rule_evaluated"] == False].copy()  # noqa: E712


def label_content_flagged_positives(
    unevaluated_df: pd.DataFrame, weight_model, threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Confident POSITIVES only, from a rule_evaluated==False pool
    (load_unevaluated_messages()) - expands the rule_pattern_score
    training pool beyond the rule-evaluated pool using
    labels/rule_labels.py::content_flagged_by_weight(): `weight_model` is
    a LogisticRegression fit by fit_content_flag_weights() on the REAL
    labelled pool (rule_evaluated==True, both sources - see that
    function's docstring for why SMPP alone can't fit one), so each
    content flag counts toward the label in proportion to how well it
    ACTUALLY predicted rule_flagged, not an unweighted count or a
    hand-picked combination.

    NO negatives come from this pool: a row scoring below `threshold` is
    not labelled clean here - a low content-flag score doesn't mean
    "not spam" (it may be spam that doesn't use these literal patterns at
    all, or ordinary untouched traffic) - only a confident positive hit is
    a strong enough signal to use as a label at all.

    label_source is tagged LABEL_SOURCE_CONTENT_STATIC_RULES - NEVER
    silently blended with load_labelled_messages()'s telecom-derived rows;
    a caller that combines both must keep this column so evaluation can be
    broken down per label_source (see models/rule_pattern/train.py's
    --include_content_labels), since these rows' rule_flagged label is
    still derived from the same CONTENT_FLAG_COLS this model also uses as
    features - real risk of an inflated-looking metric on this slice
    specifically, disclosed rather than hidden inside one combined number.
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
    always includes, regardless of use_embeddings/use_tfidf.
    `dcs` can be NaN in real data - left as-is deliberately, LightGBM
    has native missing-value handling built in, no imputation needed.
    """
    text_length = df["text"].fillna("").str.len().rename("text_length")
    text_decode_failed = (
        df["text_decode_failed"].astype(int).rename("text_decode_failed")
    )
    # Absent entirely for a caller that didn't run it through
    # load_labelled_messages() (e.g. a test fixture) - same "NaN means
    # unknown, not missing" treatment as `dcs`, not a required column.
    if IMSI_DISTINCT_ORIG_COL in df.columns:
        imsi_col = df[[IMSI_DISTINCT_ORIG_COL]]
    else:
        imsi_col = pd.DataFrame({IMSI_DISTINCT_ORIG_COL: np.nan}, index=df.index)
    # SENDER_VELOCITY_ZSCORE_COL: same "absent -> NaN, not a required
    # column" treatment - unlike IMSI this IS present for every row of
    # every source in the real file (see models/anomaly/data.py's
    # comment), so absence here only happens for a test fixture that
    # didn't include it. Raw NaN passthrough, no _known indicator needed
    # (unlike models/anomaly/data.py's sklearn Pipeline, LightGBM handles
    # NaN natively - same reasoning as `dcs` above).
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
    ]
    # Same reasoning as models/anomaly/data.py's build_feature_matrix():
    # `source` is dead weight (a constant column) once a run is restricted
    # to one source (--sources SMPP/SS7 for a split model) - only add it
    # when this run's df actually spans more than one source.
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


def load_labelled_messages_with_embeddings(
    source_dir: Path, messages_path: Path
) -> pd.DataFrame:
    """
    Same rule_evaluated==True filter as load_labelled_messages(), INNER
    JOINED with features/text_embeddings.py's output (emb_0..emb_{d-1}).
    See module docstring - coverage depends on `source_dir`'s own
    embeddings.npy: full-dataset for SS7 as of writing, still
    sample-scale (or absent) for SMPP. The inner join silently restricts
    to whatever's actually present - pass a single source_dir/messages_path
    pair per source rather than assuming combined coverage.
    """
    source_dir = Path(source_dir)
    df = load_labelled_messages(messages_path)
    df = df.copy()
    df["message_key"] = df["source"] + "|" + df["record_id"]

    embeddings = np.load(source_dir / "embeddings.npy")
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    emb_df = pd.DataFrame(
        embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )
    emb_df["message_key"] = id_map["message_key"].to_numpy()

    return df.merge(emb_df, on="message_key", how="inner").reset_index(drop=True)
