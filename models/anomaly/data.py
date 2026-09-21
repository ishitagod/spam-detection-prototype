"""
Joins the three independently-computed feature sources into one aligned
matrix for Isolation Forest, by message_key:
  - messages_with_behavioral.csv  (behavioral features + source)
  - embeddings.npy + embeddings_id_map.parquet (MiniLM embeddings)
  - faiss_output.parquet          (near-dup features, both windows)

INNER join, not left: embeddings/faiss now cover the FULL corpus (the
former --sample_n-only restriction is gone - features/text_embeddings.py's
full 8.2M-row encode, previously a ~21hr job, has since completed; see
docs/experiments/anomaly.md's "Current scale"). INNER (not LEFT) is kept
regardless, on principle - a row missing embeddings/faiss output should
drop out explicitly, not silently train on a partially-NaN feature row.

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
     is the ablation's real measured count at the time it ran - six more
     hand-built columns were added after: IMSI_DISTINCT_ORIG_COL + its
     _known indicator, SENDER_VELOCITY_ZSCORE_COL + its _known indicator,
     and two new BEHAVIORAL_COLS entries (sender_age_days,
     sender_recipient_diversity_ratio_5min/1hr count as +3, not +2 -
     see that list's own comment) added for the "bank marketing vs spam"
     gap. The imbalance direction the ablation found doesn't change from
     a few more hand-built columns against 384 embedding dimensions, so
     it wasn't worth re-running check_embedding_dominance.py for this -
     revisit if a future ablation ever suggests otherwise.)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CONTENT_FLAG_PATTERNS

# features/content_flags.py's output columns - already binary 0/1 (int8),
# passed through unchanged into `other_cols` below, same treatment as
# NEAR_DUP_COLS's similarity scores (already ~0-1, no log1p/bucketing
# needed) - see the architecture plan's Section 3 for why these are base
# features here too, not gated like embeddings/TF-IDF.
CONTENT_FLAG_COLS = list(CONTENT_FLAG_PATTERNS.keys())

BEHAVIORAL_COLS = [
    "sender_msgs_last_5min", "sender_msgs_last_1hr",
    "sender_unique_destinations_1hr", "sender_repeat_content_ratio_1hr",
    "sender_age_days",
    "sender_recipient_diversity_ratio_5min", "sender_recipient_diversity_ratio_1hr",
]
# sender_velocity_zscore_5min is DELIBERATELY kept OUT of BEHAVIORAL_COLS,
# same reason as IMSI_DISTINCT_ORIG_COL below: it CAN be NaN (fewer than 2
# prior same-sender readings, or zero variance - see
# features/behavioral.py's VELOCITY note) so it needs the same
# fillna(0)-plus-_known-indicator treatment build_combined_frame() applies
# to IMSI, which a plain BEHAVIORAL_COLS passthrough doesn't give it.
# UNLIKE IMSI, this column IS present for every row of every source (not
# SS7-only) - real measured NaN rate is negligible (<0.1% of rows, both
# sources) but still real, and sklearn's Pipeline can't take ANY NaN.
# This is the sender-relative burst signal the "any bulk sender looks
# anomalous" problem needs: a bank's normal marketing blast scores near
# its OWN typical burst size (z-score near 0), where a spam sender's burst
# (against its own usually-quiet or brand-new baseline) doesn't.
SENDER_VELOCITY_ZSCORE_COL = "sender_velocity_zscore_5min"
SENDER_VELOCITY_ZSCORE_KNOWN_COL = f"{SENDER_VELOCITY_ZSCORE_COL}_known"
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

# sender_age_days is kept in BEHAVIORAL_COLS above (still read from CSV/
# Feast/serving unchanged, still passed RAW to rule_pattern_score - see
# models/rule_pattern/data.py's _base_feature_frame(), which never calls
# build_combined_frame() below) but is DELIBERATELY NOT passed raw into
# Isolation Forest - real measured failure (docs/experiments/anomaly.md /
# the conversation this was built from): in this prototype's fixed ~2-day
# CDR sample, sender_age_days only ever ranges 0.0-2.0, so "brand new
# sender" is trivially separable by isolation splits regardless of
# content - real measured result, top-0.1%-by-anomaly_score median
# sender_age_days (0.625) vs overall median (1.247), with a clean
# monotonic gradient and precision@top-0.1% BELOW the naive baseline.
# Bucketed into coarse, REAL-WORLD-meaningful edges instead (not fit to
# this dataset's own narrow range, which would just re-encode the same
# problem) - collapses the fine-grained ordering isolation splits were
# exploiting, while staying genuinely useful once production data spans
# weeks/months and a truly-old sender becomes rare again (right now,
# nearly everything falls in the two lowest buckets - that's expected and
# fine, it's what actually fixes the problem on this sample; the buckets
# above that are for when they eventually start populating).
SENDER_AGE_DAYS_COL = "sender_age_days"
SENDER_AGE_BUCKET_EDGES_DAYS = [-np.inf, 1 / 24, 1, 7, 30, np.inf]  # hour, day, week, month
SENDER_AGE_BUCKET_LABELS = ["lt_1hr", "1hr_to_1day", "1day_to_7day", "7day_to_30day", "gte_30day"]
SENDER_AGE_BUCKET_COLS = [f"sender_age_bucket_{label}" for label in SENDER_AGE_BUCKET_LABELS]

# sender_recipient_diversity_ratio_5min/1hr are kept in BEHAVIORAL_COLS
# above (still read raw, still passed RAW to rule_pattern_score - see
# models/rule_pattern/data.py, which never calls build_combined_frame()
# below) but NOT passed raw into Isolation Forest either - a DIFFERENT
# failure mode from sender_age_days above, even more severe by real
# measurement: this ratio is unique_destinations / message_count in the
# window, so a sender with only 1 message in that window gets a
# TRIVIALLY extreme ratio (1.0 - one destination out of one message,
# guaranteed) regardless of real behavioral diversity. Bucketing the
# VALUE (like age above) wouldn't fix this - a ratio of 1.0 from 1
# message and a ratio of 1.0 from 20 messages would still land in the
# same bucket, treated identically, even though only the second one
# means anything. Real measured result: top-0.1%-by-anomaly_score MEAN
# sender_recipient_diversity_ratio_5min (0.718) vs overall MEDIAN
# (0.002) - ~48x enrichment by mean, effectively unbounded by median.
# Fixed by gating on the underlying message count instead - below
# SENDER_DIVERSITY_MIN_MSGS messages in that window, the ratio is
# genuinely unreliable (not just an extreme value), treated as unknown
# via the SAME NaN -> fillna(0) + _known-indicator pattern as
# SENDER_VELOCITY_ZSCORE_COL/IMSI_DISTINCT_ORIG_COL above, rather than a
# sharp, spurious 0/1 the model can trivially isolate on.
SENDER_DIVERSITY_MIN_MSGS = 3  # starting point, not tuned against a real
# target - same "documented, not proven" status as
# FAISS_NEAR_DUP_THRESHOLD/N_EMBEDDING_COMPONENTS.
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
        + [IMSI_DISTINCT_ORIG_COL, SENDER_VELOCITY_ZSCORE_COL]
        + CONTENT_FLAG_COLS
        + ["rule_evaluated", "rule_flagged"]
    )
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=lambda c: c in set(wanted_cols),
    )
    if IMSI_DISTINCT_ORIG_COL not in messages.columns:
        messages[IMSI_DISTINCT_ORIG_COL] = np.nan
    # Absent entirely for a messages_with_behavioral.csv produced before
    # features/content_flags.py existed - default to 0 (no flags known),
    # not NaN, since sklearn's Pipeline can't take NaN and "unknown" isn't
    # a meaningful state for a deterministic regex feature the way it is
    # for IMSI/velocity (those measure something that may genuinely be
    # unobserved; a missing content-flag column just means this run
    # predates the feature).
    for col in CONTENT_FLAG_COLS:
        if col not in messages.columns:
            messages[col] = 0
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


EMBEDDING_CHUNK_SIZE = 200_000  # rows per partial_fit/transform batch


class ChunkedEmbeddingReducer(BaseEstimator, TransformerMixin):
    """
    Same operation as embedding_pca_pipeline() (StandardScaler -> PCA),
    fit/transformed in EMBEDDING_CHUNK_SIZE-row batches via partial_fit -
    ONLY used inside build_preprocessor() below, for training at the full
    multi-million-row corpus scale where a plain StandardScaler.fit() on
    the whole embedding matrix at once is unsafe: numpy upcasts float32 X
    to float64 when subtracting sklearn's float64 mean internally
    (X - T in sklearn.utils.extmath._incremental_mean_and_var), so a
    single (n_rows, 384) fit briefly needs float64-sized memory (~2x the
    raw embeddings.npy) regardless of X's own dtype - measured to OOM a
    16GB machine on SS7's 2.74M-row corpus. Chunking bounds that
    intermediate to EMBEDDING_CHUNK_SIZE rows at a time, fitting the exact
    same full corpus, not a sample - see docs/experiments/anomaly.md.
    embedding_pca_pipeline() itself is untouched (still a plain
    Pipeline(StandardScaler, PCA)) - rule_pattern's embeddings path and
    live single-row serving (serving/scoring.py) don't operate at this
    scale and don't need chunking.
    """

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
    """
    embedding_cols -> ChunkedEmbeddingReducer(); other_cols -> passthrough
    (already log1p'd/one-hot by the caller); both concatenated, THEN a
    final StandardScaler over the combined result - the PCA components
    themselves have very unequal variance (the first component always
    varies far more than the last), so re-scaling after PCA matters for
    the same reason scaling mattered before it. The final StandardScaler
    is NOT chunked - its input is already PCA-reduced (n_embedding_
    components wide, not 384), small enough that a full-corpus fit stays
    well within memory even with the same float64-upcast behavior.

    Returned as a single fitted-once, reused-everywhere Pipeline - this
    IS the artifact that must travel to inference unchanged, not
    something to refit on new data (train/serve skew otherwise).
    """
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
    """
    The pre-preprocessor frame build_feature_matrix() fits/transforms -
    factored out so callers that need the RAW input shape the returned
    Pipeline actually expects (e.g. models/anomaly/train.py logging an
    MLflow model signature for the full preprocessor+model pipeline) can
    get it without duplicating this construction. Returns (combined,
    embedding_cols, other_cols) - same three pieces build_feature_matrix()
    passes to build_preprocessor().

    `known_sources`: forces one-hot columns for EVERY name in this list to
    exist in the output, regardless of how many distinct `source` values
    `df` itself contains - for serving/anomaly_scoring.py's single live
    row, where `df["source"].nunique()` is always 1 and would otherwise
    silently drop whichever source_* column the fitted preprocessor still
    expects (it was fit on a combined-sources df where both existed).
    None (default, every training caller) preserves the old nunique()>1
    behavior exactly - unaffected by this parameter.
    """
    # Excludes emb_* columns from the copy - nothing below mutates them,
    # and copying a (n_rows, 384) float32 block just to leave it untouched
    # doubles memory for no reason (measured: this alone OOM'd SS7's full
    # 2.74M-row corpus on a 16GB machine). embedding_cols is read straight
    # from df at the bottom of this function instead.
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    transformed = df.drop(columns=embedding_cols).copy()
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

    # SENDER_VELOCITY_ZSCORE_COL: same _known-indicator treatment as IMSI
    # above, NOT log1p'd (already roughly z-score-shaped, can be negative -
    # see BEHAVIORAL_COLS' comment on why this column is handled here
    # rather than living in that list directly). fillna(0) is a genuinely
    # neutral default for a z-score specifically (0 = "exactly at this
    # sender's own typical burst size"), not a fabricated count like 0
    # would be for IMSI - still paired with a _known indicator so the
    # model can tell "genuinely average" from "no baseline existed yet"
    # if that distinction carries real signal.
    if SENDER_VELOCITY_ZSCORE_COL not in transformed.columns:
        transformed[SENDER_VELOCITY_ZSCORE_COL] = np.nan
    transformed[SENDER_VELOCITY_ZSCORE_KNOWN_COL] = transformed[SENDER_VELOCITY_ZSCORE_COL].notna().astype(float)
    transformed[SENDER_VELOCITY_ZSCORE_COL] = transformed[SENDER_VELOCITY_ZSCORE_COL].fillna(0.0)

    # SENDER_AGE_DAYS_COL: bucketed, NOT passed raw - see
    # SENDER_AGE_BUCKET_EDGES_DAYS's comment above for why (real measured
    # failure on this dataset's narrow 0.0-2.0 range). pd.cut with an
    # explicit `labels=` always yields ALL SENDER_AGE_BUCKET_LABELS as
    # categories regardless of which bins this particular df actually
    # populates - verified this holds even for a single-row df - so
    # get_dummies() below always produces all SENDER_AGE_BUCKET_COLS, the
    # same "single live row must still match the fitted preprocessor's
    # expected columns" guarantee `known_sources` gives `source` below,
    # without needing an equivalent parameter here.
    age_bucket = pd.cut(
        transformed[SENDER_AGE_DAYS_COL],
        bins=SENDER_AGE_BUCKET_EDGES_DAYS, labels=SENDER_AGE_BUCKET_LABELS,
    )
    age_bucket_dummies = pd.get_dummies(age_bucket, prefix="sender_age_bucket")

    # SENDER_DIVERSITY_SHORT_COL/LONG_COL: gated on the underlying message
    # count, NOT passed raw - see SENDER_DIVERSITY_MIN_MSGS's comment above
    # for why (real measured failure, more severe than age's). Uses `df`
    # (the original, pre-log1p frame), not `transformed` - the COUNT_COLS
    # loop above already log1p'd transformed[*_MSGS_COL] in place, and the
    # gate needs the REAL message count, not its log1p'd value.
    below_min_short = df[SENDER_DIVERSITY_SHORT_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_short, SENDER_DIVERSITY_SHORT_COL] = np.nan
    transformed[SENDER_DIVERSITY_SHORT_KNOWN_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_SHORT_COL] = transformed[SENDER_DIVERSITY_SHORT_COL].fillna(0.0)

    below_min_long = df[SENDER_DIVERSITY_LONG_MSGS_COL] < SENDER_DIVERSITY_MIN_MSGS
    transformed.loc[below_min_long, SENDER_DIVERSITY_LONG_COL] = np.nan
    transformed[SENDER_DIVERSITY_LONG_KNOWN_COL] = transformed[SENDER_DIVERSITY_LONG_COL].notna().astype(float)
    transformed[SENDER_DIVERSITY_LONG_COL] = transformed[SENDER_DIVERSITY_LONG_COL].fillna(0.0)

    # `source` is only a real feature when more than one source is present
    # in this training run - a single-source run (e.g. --sources SMPP for
    # a split model, see CLAUDE.md's "Split by source" note) would produce
    # a constant one-hot column carrying zero information, just dead
    # weight through StandardScaler. Combined-sources runs keep the dummy
    # unchanged - same behavior as before.
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
    other_cols = (
        raw_passthrough_behavioral_cols + list(NEAR_DUP_COLS) + imsi_cols + velocity_cols
        + diversity_cols + SENDER_AGE_BUCKET_COLS + CONTENT_FLAG_COLS
    )
    pieces = [
        transformed[raw_passthrough_behavioral_cols + NEAR_DUP_COLS + imsi_cols + velocity_cols + diversity_cols],
        age_bucket_dummies,
        transformed[CONTENT_FLAG_COLS],
    ]
    if known_sources is not None:
        source_dummies = pd.get_dummies(
            transformed["source"].astype(pd.CategoricalDtype(categories=sorted(known_sources))),
            prefix="source",
        )
        other_cols += list(source_dummies.columns)
        pieces.append(source_dummies)
    elif transformed["source"].nunique() > 1:
        source_dummies = pd.get_dummies(transformed["source"], prefix="source")
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
