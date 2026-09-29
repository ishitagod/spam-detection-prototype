"""
Joins the three independently-computed feature sources into one aligned
matrix for Isolation Forest, by message_key:
  - messages_with_behavioral.csv  (behavioral features + source)
  - embeddings.npy + embeddings_id_map.parquet (MiniLM embeddings)
  - faiss_output.parquet          (near-dup features, both windows)

INNER join (not left) - a row missing embeddings/faiss output should
drop out explicitly, not silently train on a partially-NaN row.

`source` is a one-hot feature, not a routing key (CLAUDE.md's "one
model to start, not two"). `rule_evaluated`/`rule_flagged` are carried
through only for train.py's label-free plausibility check - Isolation
Forest itself trains on zero labels.

PREPROCESSING:
  1. log1p on heavy-tailed count features (e.g. sender_msgs_last_1hr up
     to 16,971) vs. embedding dims confined to ~[-1, 1].
  2. PCA on the 384 embedding dims down to N_EMBEDDING_COMPONENTS before
     combining with the hand-built features - an ablation
     (scripts/check_embedding_dominance.py) found the joint model's
     score rankings correlated 0.808 with embeddings-only but only 0.326
     with behavioral-only, i.e. embeddings were drowning out the
     hand-built features without this. Applied via ColumnTransformer so
     only embedding columns get PCA'd; hand-built features pass through
     into the same final joint StandardScaler.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS, CONTENT_FLAG_PATTERNS

# features/content_flags.py's output columns - already binary 0/1,
# passed through unchanged (same as NEAR_DUP_COLS's ~0-1 similarity scores).
CONTENT_FLAG_COLS = list(CONTENT_FLAG_PATTERNS.keys())

# Engineered on top of CONTENT_FLAG_COLS, not read from CSV - same
# treatment as any other passthrough feature (0/1 or small int, no
# scaling needed). content_flag_high_conf reuses the exact combination
# list labels/rule_labels.py's content_flagged() uses to build the
# SEPARATE content_flagged label - safe as a FEATURE against
# rule_flagged (the telecom-engine label these models actually train
# against, computed with zero content/regex matching - see
# labels/rule_labels.py's module docstring), but would be tautological
# if content_flagged were ever used as a training target instead.
CONTENT_FLAG_META_COLS = [
    "content_flag_hit_count",
    "content_flag_any",
    "content_flag_high_conf",
]


def compute_content_flag_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """CONTENT_FLAG_META_COLS from a frame already carrying CONTENT_FLAG_COLS."""
    flags = df[CONTENT_FLAG_COLS].astype(int)
    hit_count = flags.sum(axis=1)
    any_flag = (hit_count > 0).astype(int)
    high_conf = pd.Series(False, index=df.index)
    for combination in CONTENT_FLAG_HIGH_CONFIDENCE_COMBINATIONS:
        high_conf |= flags[combination].all(axis=1)
    return pd.DataFrame(
        {
            "content_flag_hit_count": hit_count,
            "content_flag_any": any_flag,
            "content_flag_high_conf": high_conf.astype(int),
        },
        index=df.index,
    )

BEHAVIORAL_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr",
    "sender_unique_destinations_1hr", "sender_repeat_content_ratio_1hr",
    "sender_age_days",
    "sender_recipient_diversity_ratio_5min", "sender_recipient_diversity_ratio_1hr",
]
# sender_velocity_zscore_5min: kept OUT of BEHAVIORAL_COLS since it can be
# NaN (<0.1% of rows, both sources - fewer than 2 prior same-sender
# readings, or zero variance) and needs the fillna(0)+_known-indicator
# treatment below (sklearn's Pipeline can't take NaN). Sender-relative
# burst signal: a bank's normal marketing blast scores near its OWN
# typical burst size (z-score ~0); a spam burst against a quiet/new
# baseline doesn't.
SENDER_VELOCITY_ZSCORE_COL = "sender_velocity_zscore_5min"
SENDER_VELOCITY_ZSCORE_KNOWN_COL = f"{SENDER_VELOCITY_ZSCORE_COL}_known"
# SS7-only SIM-farming signal - entirely absent (not NaN) from SMPP's
# messages_with_behavioral.csv, so load_source_features() needs a presence
# check. Also null on ~31.7% of SS7 rows even where present - both cases
# collapse to the same fillna(0)+_known-indicator treatment.
IMSI_DISTINCT_ORIG_COL = "imsi_distinct_originators_1hr"
IMSI_DISTINCT_ORIG_KNOWN_COL = f"{IMSI_DISTINCT_ORIG_COL}_known"

# SS7-only (same absence pattern as IMSI_DISTINCT_ORIG_COL above) - opaque
# app/session verification tokens and Apple FaceTime "REG-REQ" activation
# pings (ingestion/ss7.py's SS7_PLATFORM_TOKEN_PATTERN/
# SS7_FACETIME_ACTIVATION_PATTERN). Rows are dropped entirely wherever this
# is True (load_source_features() below, models/rule_pattern/data.py's
# _load_messages_csv()) - not a feature, a training-pool exclusion, since
# neither model should spend capacity on content that can never be spam.
PLATFORM_TOKEN_COL = "is_platform_token_message"

# sender_age_days: kept in BEHAVIORAL_COLS (still passed raw to
# rule_pattern_score) but NOT passed raw into Isolation Forest - this
# prototype's ~2-day CDR sample only has sender_age_days in [0.0, 2.0],
# so isolation splits trivially separated "brand new sender" regardless
# of content (measured: top-0.1%-by-anomaly_score median age 0.625 vs
# overall 1.247, precision@top-0.1% below baseline). Bucketed into
# coarse, real-world-meaningful edges instead, which collapses that
# exploitable fine-grained ordering (most rows fall in the two lowest
# buckets today - expected on this sample; higher buckets matter once
# production data spans weeks/months).
SENDER_AGE_DAYS_COL = "sender_age_days"
SENDER_AGE_BUCKET_EDGES_DAYS = [-np.inf, 1 / 24, 1, 7, 30, np.inf]  # hour, day, week, month
SENDER_AGE_BUCKET_LABELS = ["lt_1hr", "1hr_to_1day", "1day_to_7day", "7day_to_30day", "gte_30day"]
SENDER_AGE_BUCKET_COLS = [f"sender_age_bucket_{label}" for label in SENDER_AGE_BUCKET_LABELS]

# sender_recipient_diversity_ratio_5min/1hr: also kept raw in
# BEHAVIORAL_COLS but not passed raw into Isolation Forest - ratio is
# unique_destinations/message_count, so 1 message in the window always
# gives a trivially extreme ratio of 1.0 regardless of real diversity.
# Bucketing wouldn't fix it (1.0-from-1-msg and 1.0-from-20-msgs would
# still land in the same bucket). Measured: top-0.1%-by-anomaly_score
# MEAN ratio 0.718 vs overall MEDIAN 0.002 - ~48x enrichment. Fixed by
# gating on message count instead - below SENDER_DIVERSITY_MIN_MSGS, the
# ratio is unreliable, treated as unknown via the same
# fillna(0)+_known-indicator pattern as velocity/IMSI above.
SENDER_DIVERSITY_MIN_MSGS = 3  # starting point, not tuned
SENDER_DIVERSITY_SHORT_COL = "sender_recipient_diversity_ratio_5min"
SENDER_DIVERSITY_SHORT_KNOWN_COL = f"{SENDER_DIVERSITY_SHORT_COL}_known"
SENDER_DIVERSITY_SHORT_MSGS_COL = "sender_msgs_last_5min"  # gate column - same window
SENDER_DIVERSITY_LONG_COL = "sender_recipient_diversity_ratio_1hr"
SENDER_DIVERSITY_LONG_KNOWN_COL = f"{SENDER_DIVERSITY_LONG_COL}_known"
SENDER_DIVERSITY_LONG_MSGS_COL = "sender_msgs_last_1hr"  # gate column - same window

NEAR_DUP_COLS = [
    "near_dup_match_count_1hr", "near_dup_max_similarity_1hr", "near_dup_distinct_senders_1hr",
    "near_dup_match_count_24hr", "near_dup_max_similarity_24hr", "near_dup_distinct_senders_24hr",
]
# Heavy-tailed count columns that get log1p'd before scaling - ratios
# and similarity scores (already ~0-1) are left alone.
COUNT_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr", "sender_unique_destinations_1hr",
    "near_dup_match_count_1hr", "near_dup_distinct_senders_1hr",
    "near_dup_match_count_24hr", "near_dup_distinct_senders_24hr",
]

# Not tuned against a target explained-variance threshold - a starting
# point to bring 384 dims closer in magnitude to the hand-built
# features (see module docstring's ablation). build_feature_matrix()
# prints actual retained variance every run.
N_EMBEDDING_COMPONENTS = 30


def normalize_messages_frame(df: pd.DataFrame, log_prefix: str) -> pd.DataFrame:
    """Post-read cleanup shared by every messages_with_behavioral.csv loader
    (load_source_features() below, models/rule_pattern/data.py's
    _load_messages_csv()) so both models see the same pool:
      - IMSI_DISTINCT_ORIG_COL: SS7-only, absent from SMPP's file -> all-NaN
        (NaN = genuinely unknown, not zero).
      - CONTENT_FLAG_COLS: absent from a CSV that predates content_flags.py
        (or a later CONTENT_FLAG_PATTERNS addition) -> 0, not NaN; unlike
        IMSI/velocity, "unknown" isn't meaningful for a deterministic regex.
      - PLATFORM_TOKEN_COL: SS7-only, absent from SMPP or a pre-flag CSV ->
        False. Rows where it's True are DROPPED (opaque app/session tokens
        and Apple FaceTime "REG-REQ" pings can never be spam, so neither
        model should spend capacity on them), and the column is removed.
    Adds the missing columns to `df` IN PLACE (no up-front copy - this runs
    on multi-million-row frames, see ChunkedEmbeddingReducer's OOM note) and
    returns the filtered copy; callers must use the return value."""
    if IMSI_DISTINCT_ORIG_COL not in df.columns:
        df[IMSI_DISTINCT_ORIG_COL] = np.nan
    for col in CONTENT_FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
    if PLATFORM_TOKEN_COL not in df.columns:
        df[PLATFORM_TOKEN_COL] = False
    dropped = df[PLATFORM_TOKEN_COL].sum()
    if dropped:
        print(f"  {log_prefix}: dropping {dropped} platform/protocol signaling row(s) (is_platform_token_message)")
    return df[~df[PLATFORM_TOKEN_COL]].drop(columns=PLATFORM_TOKEN_COL).copy()


def build_source_dummies(source: pd.Series, known_sources: list[str] | None = None) -> pd.DataFrame | None:
    """One-hot `source`, shared by both models' feature builders.

    `known_sources` forces a column for every listed name regardless of how
    many distinct values `source` holds - needed by live single-row serving
    (nunique() is always 1), which would otherwise silently drop a source_*
    column the fitted model still expects. None (all training callers) only
    emits dummies when >1 source is present: a single-source run (e.g.
    --sources SMPP) would produce a constant, information-free column.
    Returns None when no dummies should be added."""
    if known_sources is not None:
        return pd.get_dummies(
            source.astype(pd.CategoricalDtype(categories=sorted(known_sources))), prefix="source",
        )
    if source.nunique() > 1:
        return pd.get_dummies(source, prefix="source")
    return None


class MissingEmbeddingsError(RuntimeError):
    """This source has no embeddings.npy/embeddings_id_map.parquet/
    faiss_output.parquet yet - same missing-corpus state as serving's
    CorpusUnavailableError (serving/anomaly_scoring.py), raised here too so
    training fails with an actionable message instead of a bare
    FileNotFoundError deep inside np.load()/pd.read_parquet()."""


def load_source_features(source_dir: Path, messages_path: Path) -> pd.DataFrame:
    """One row per message with behavioral + near_dup + embedding columns
    (`emb_0`..`emb_{d-1}`) plus `source`/`rule_evaluated`/`rule_flagged`
    carried through unscaled."""
    source_dir = Path(source_dir)
    missing = [
        name for name in ("embeddings.npy", "embeddings_id_map.parquet", "faiss_output.parquet")
        if not (source_dir / name).exists()
    ]
    if missing:
        raise MissingEmbeddingsError(
            f"{source_dir} is missing {missing} - run features/text_embeddings.py "
            "then features/faiss_index.py for this source first."
        )
    # Lambda usecols so a file missing IMSI_DISTINCT_ORIG_COL (SMPP) is
    # skipped rather than raising, unlike a plain list.
    wanted_cols = (
        ["source", "record_id"] + BEHAVIORAL_COLS
        + [IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL]
        + CONTENT_FLAG_COLS
        + ["rule_evaluated", "rule_flagged", PLATFORM_TOKEN_COL]
    )
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=lambda c: c in set(wanted_cols),
    )
    # Isolation Forest trains on the full traffic stream (CLAUDE.md) MINUS
    # the platform/protocol-signaling carve-out normalize_messages_frame()
    # drops (shared with rule_pattern's loader).
    messages = normalize_messages_frame(messages, "load_source_features")
    messages["source"] = messages["source"].astype(str)
    messages["record_id"] = messages["record_id"].astype(str)
    messages["message_key"] = messages["source"] + "|" + messages["record_id"]

    # If features/text_embeddings.py's last run for this source was
    # --sample_n (not the full corpus), it leaves this file as a
    # persistent, deliberate disclosure (see its docstring) - surface it
    # HERE too, not just in that script's own console output, because the
    # consequence is worse than "fewer training rows": every message NOT
    # in the sample is also invisible to serving/anomaly_scoring.py's
    # near-dup comparison corpus (same embeddings.npy/id_map, see that
    # module's docstring) - a live message matching an unsampled pattern
    # will show near_dup_match_count=0 ("never seen this before") even if
    # that pattern is common in real traffic, purely from this gap, not
    # genuine novelty.
    sample_info_path = source_dir / "embeddings_sample_info.txt"
    if sample_info_path.exists():
        print(
            f"  load_source_features: WARNING - {source_dir}'s embeddings are a SAMPLE, "
            "not full corpus coverage (see embeddings_sample_info.txt below). Both this "
            "training run AND live serving's near-dup corpus (serving/anomaly_scoring.py) "
            "are missing every unsampled row - re-run features/text_embeddings.py without "
            "--sample_n before trusting anomaly_score for this source.\n"
            f"    {sample_info_path.read_text().strip().replace(chr(10), chr(10) + '    ')}"
        )

    embeddings = np.load(source_dir / "embeddings.npy")
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df["message_key"] = id_map["message_key"].to_numpy()

    near_dup = pd.read_parquet(source_dir / "faiss_output.parquet")

    before = len(messages)
    df = messages.merge(emb_df, on="message_key", how="inner")
    df = df.merge(near_dup, on="message_key", how="inner")
    after = len(df)
    if after < before:
        dropped_n = before - after
        print(
            f"  load_source_features: dropping {dropped_n}/{before} ({dropped_n / before:.1%}) "
            f"row(s) from {source_dir} with no matching embedding/near-dup output (INNER join) "
            "- see this function's docstring; a large fraction here usually means the "
            "embeddings/FAISS corpus is stale or sampled relative to messages_with_behavioral.csv, "
            "not that this many rows are individually anomalous."
        )
    return df


def embedding_pca_pipeline(n_embedding_components: int) -> Pipeline:
    """StandardScaler -> PCA(n_embedding_components), embedding columns
    only. Shared by build_preprocessor() below (wrapped in a final joint
    scaler) and models/rule_pattern/data.py's embeddings-aware matrix
    (which skips scaling other features - tree splits don't need it)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_embedding_components, random_state=42)),
    ])


EMBEDDING_CHUNK_SIZE = 200_000  # rows per partial_fit/transform batch


class ChunkedEmbeddingReducer(BaseEstimator, TransformerMixin):
    """Same as embedding_pca_pipeline() (StandardScaler -> PCA), but
    fit/transformed in EMBEDDING_CHUNK_SIZE-row batches via partial_fit -
    only used in build_preprocessor() below, for full-corpus training. A
    plain StandardScaler.fit() over the whole embedding matrix briefly
    upcasts float32 X to float64 internally (~2x embeddings.npy's size),
    which measured to OOM a 16GB machine on SS7's 2.74M rows. Chunking
    bounds that, still fitting the full corpus, not a sample. Unused by
    rule_pattern's embeddings path or live single-row serving - neither
    operates at this scale."""

    def __init__(self, n_components: int, chunk_size: int = EMBEDDING_CHUNK_SIZE):
        self.n_components = n_components
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.scaler_ = StandardScaler()
        for start in range(0, len(X), self.chunk_size):
            self.scaler_.partial_fit(X[start:start + self.chunk_size])

        self.pca_ = IncrementalPCA(n_components=self.n_components)
        for start in range(0, len(X), self.chunk_size):
            chunk = self.scaler_.transform(X[start:start + self.chunk_size])
            self.pca_.partial_fit(chunk)

        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), self.n_components), dtype=np.float32)
        for start in range(0, len(X), self.chunk_size):
            end = start + self.chunk_size
            out[start:end] = self.pca_.transform(self.scaler_.transform(X[start:end]))
        return out


def build_preprocessor(embedding_cols: list[str], other_cols: list[str], n_embedding_components: int) -> Pipeline:
    """embedding_cols -> ChunkedEmbeddingReducer(); other_cols ->
    passthrough (already log1p'd/one-hot); both concatenated, then a
    final (non-chunked - already PCA-reduced, small) StandardScaler over
    the combined result, since PCA components themselves have unequal
    variance. Returned Pipeline is the artifact that must travel to
    inference unchanged - never refit on new data."""
    reduce = ColumnTransformer([
        ("embeddings", ChunkedEmbeddingReducer(n_embedding_components), embedding_cols),
        ("other", "passthrough", other_cols),
    ])
    return Pipeline([
        ("reduce", reduce),
        ("final_scale", StandardScaler()),
    ])


def build_combined_frame(
    df: pd.DataFrame, known_sources: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The pre-preprocessor frame build_feature_matrix() fits/transforms -
    factored out for callers (e.g. train.py logging an MLflow model
    signature) that need the raw input shape without duplicating this
    construction. Returns (combined, embedding_cols, other_cols).

    `known_sources`: forces one-hot columns for every name in this list
    to exist in the output, regardless of how many distinct `source`
    values `df` contains - needed by serving/anomaly_scoring.py's single
    live row (`nunique()` always 1), which would otherwise silently drop
    a source_* column the fitted preprocessor still expects. None
    (default, all training callers) preserves the old nunique()>1 behavior.
    """
    # Excludes emb_* from the copy - nothing below mutates them, and
    # copying a (n_rows, 384) float32 block untouched doubles memory for
    # no reason (measured to OOM SS7's full 2.74M-row corpus on 16GB).
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    transformed = df.drop(columns=embedding_cols).copy()
    for col in COUNT_COLS:
        transformed[col] = np.log1p(transformed[col])

    # May be entirely absent (test fixtures) - treat as "present but NaN".
    if IMSI_DISTINCT_ORIG_COL not in transformed.columns:
        transformed[IMSI_DISTINCT_ORIG_COL] = np.nan
    # NaN = genuinely unknown, not zero - fillna(0) alone would fabricate
    # "zero distinct originators"; _known lets the model tell them apart.
    transformed[IMSI_DISTINCT_ORIG_KNOWN_COL] = transformed[IMSI_DISTINCT_ORIG_COL].notna().astype(float)
    transformed[IMSI_DISTINCT_ORIG_COL] = np.log1p(transformed[IMSI_DISTINCT_ORIG_COL].fillna(0))

    # Same _known treatment as IMSI, but NOT log1p'd (already z-score
    # shaped, can be negative). fillna(0) is a neutral default here (0 =
    # "at this sender's own typical burst size"), unlike IMSI's fabricated-
    # count concern - still paired with _known in case the distinction matters.
    if SENDER_VELOCITY_ZSCORE_COL not in transformed.columns:
        transformed[SENDER_VELOCITY_ZSCORE_COL] = np.nan
    transformed[SENDER_VELOCITY_ZSCORE_KNOWN_COL] = transformed[SENDER_VELOCITY_ZSCORE_COL].notna().astype(float)
    transformed[SENDER_VELOCITY_ZSCORE_COL] = transformed[SENDER_VELOCITY_ZSCORE_COL].fillna(0.0)

    # Bucketed, not raw - see SENDER_AGE_BUCKET_EDGES_DAYS's comment.
    # pd.cut with explicit `labels=` always yields all
    # SENDER_AGE_BUCKET_LABELS as categories (verified even for a
    # single-row df), so get_dummies() below always produces all
    # SENDER_AGE_BUCKET_COLS - same column-stability guarantee
    # `known_sources` gives `source`, without needing a parameter here.
    age_bucket = pd.cut(
        transformed[SENDER_AGE_DAYS_COL],
        bins=SENDER_AGE_BUCKET_EDGES_DAYS, labels=SENDER_AGE_BUCKET_LABELS,
    )
    age_bucket_dummies = pd.get_dummies(age_bucket, prefix="sender_age_bucket")

    # Gated on message count, not passed raw - see SENDER_DIVERSITY_MIN_MSGS.
    # Uses `df` (pre-log1p), not `transformed` - the gate needs the real
    # message count, not the log1p'd value COUNT_COLS already applied.
    below_min_short = df[SENDER_DIVERSITY_SHORT_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_short, SENDER_DIVERSITY_SHORT_COL] = np.nan
    transformed[SENDER_DIVERSITY_SHORT_KNOWN_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_SHORT_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].fillna(0.0)

    below_min_long = df[SENDER_DIVERSITY_LONG_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_long, SENDER_DIVERSITY_LONG_COL] = np.nan
    transformed[SENDER_DIVERSITY_LONG_KNOWN_COL] = transformed[SENDER_DIVERSITY_LONG_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_LONG_COL] = transformed[SENDER_DIVERSITY_LONG_COL].fillna(0.0)

    # `source` is only a real feature with >1 source present - a
    # single-source run (e.g. --sources SMPP) would produce a constant
    # one-hot column carrying zero information.
    raw_passthrough_behavioral_cols = [
        c for c in BEHAVIORAL_COLS
        if c not in (SENDER_AGE_DAYS_COL, SENDER_DIVERSITY_SHORT_COL, SENDER_DIVERSITY_LONG_COL)
    ]
    imsi_cols = [IMSI_DISTINCT_ORIG_COL, IMSI_DISTINCT_ORIG_KNOWN_COL]
    velocity_cols = [SENDER_VELOCITY_ZSCORE_COL, SENDER_VELOCITY_ZSCORE_KNOWN_COL]
    diversity_cols = [
        SENDER_DIVERSITY_SHORT_COL, SENDER_DIVERSITY_SHORT_KNOWN_COL,
        SENDER_DIVERSITY_LONG_COL, SENDER_DIVERSITY_LONG_KNOWN_COL,
    ]
    content_flag_meta = compute_content_flag_meta_features(transformed)
    other_cols = (
        raw_passthrough_behavioral_cols + list(NEAR_DUP_COLS) + imsi_cols + velocity_cols
        + diversity_cols + SENDER_AGE_BUCKET_COLS + CONTENT_FLAG_COLS + CONTENT_FLAG_META_COLS
    )
    pieces = [
        transformed[raw_passthrough_behavioral_cols + NEAR_DUP_COLS + imsi_cols + velocity_cols + diversity_cols],
        age_bucket_dummies,
        transformed[CONTENT_FLAG_COLS],
        content_flag_meta,
    ]
    source_dummies = build_source_dummies(transformed["source"], known_sources)
    if source_dummies is not None:
        other_cols += list(source_dummies.columns)
        pieces.append(source_dummies)
    pieces.append(df[embedding_cols])

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

    explained = preprocessor.named_steps["reduce"].named_transformers_["embeddings"].explained_variance_ratio_
    print(f"  PCA: {n_embedding_components} components retain {explained.sum():.1%} of embedding variance")

    feature_names = [f"emb_pca_{i}" for i in range(n_embedding_components)] + other_cols
    return X, feature_names, preprocessor
