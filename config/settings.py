"""
Tunable operational parameters - values someone might reasonably want to
change without touching pipeline logic. Deliberately does NOT include
protocol constants (e.g. GSM 03.38's DCS alphabet-bit masks in
ingestion/smpp.py) - those are spec, not config, and externalizing them
would just add indirection around a value that will never change.
"""

import os

# --- MLflow tracking store -----------------------------------------------
# Single source of truth - every train.py/registry/serving module that
# calls mlflow.set_tracking_uri() imports this instead of hardcoding
# "sqlite:///mlflow.db" separately (that was duplicated across ~8 files).
# Points at docker-compose.yml's postgres service by default (the "mlflow"
# database created by scripts/postgres_init/01_create_mlflow_db.sql,
# alongside Feast's own feast_registry database in the same instance) -
# override via env var for a non-docker-compose Postgres or CI.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "postgresql+psycopg2://spam_detection:spam_detection@localhost:5433/mlflow",
)

# --- SMPP ingestion (ingestion/smpp.py) ---------------------------------
SMPP_SUBMIT_SM_OPERATION = 4  # keep only op-4 (submit_sm) rows - see
# ingestion/smpp.py clean_smpp_raw()

# --- Rule-engine label derivation (labels/rule_labels.py) --------------
# SmartWhitelist rule-name prefix: a `rule` value starting with this is a
# pre-check allowlist hit (sender/route ID in a whitelist DB, decision=0,
# content never evaluated for spam) - NOT a genuine "evaluated, clean"
# verdict. Confirmed against real op-4 data across 6 files: SW_* rows are
# always decision=0 and vastly outnumber real spam-pattern (S_*) rows.
WHITELIST_RULE_PREFIX = "SW_"
BLACKLIST_RULE_PREFIX = "SR_"

# --- Behavioral features (features/behavioral.py) -----------------------
BEHAVIORAL_VERY_SHORT_WINDOW = "1min"  # sender_msgs_last_1min
BEHAVIORAL_SHORT_WINDOW = (
    "5min"  # sender_msgs_last_5min, sender_unique_destinations_5min,
)
# sender_recipient_diversity_ratio_5min, sender_velocity_zscore_5min
BEHAVIORAL_LONG_WINDOW = "1h"  # sender_msgs_last_1hr, sender_unique_destinations_1hr,
# sender_repeat_content_ratio_1hr, sender_recipient_diversity_ratio_1hr

# --- Content-rule flags (features/content_flags.py) ---------------------
# Named regex registry - one binary column per entry, computed from `text`
# alone (no behavioral/history dependency, unlike features/behavioral.py).
# These are ML FEATURES, not Rule Engine gates - the upstream telecom Rule
# Engine (labels/rule_labels.py's SW_/SR_ prefixes) has zero content/regex
# matching of its own, confirmed against real data this session; these
# flags are this codebase's own, independent, second signal source, fed to
# BOTH LightGBM and Isolation Forest as base features (see the architecture
# plan's Section 1 - cheap, deterministic, always-available, same category
# as behavioral/near-dup, not corpus-fit like TF-IDF/embeddings).
#
# English-language keyword lists are a starting point, not a calibrated or
# complete set - this system scores multilingual text (paraphrase-
# multilingual-MiniLM-L12-v2 is the embedding model precisely because
# content isn't all English) - real per-language corpus mining is future
# work, flagged here rather than silently assumed complete.
CONTENT_FLAG_PATTERNS = {
    "has_url": r"https?://|www\.",
    "has_shortlink": r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|ow\.ly|rebrand\.ly|cutt\.ly|tiny\.cc|shorturl\.at)\b",
    "has_phone_number": r"\b(?:\+?\d[\d\-\s]{8,}\d)\b",
    "has_currency_symbol": r"[$€£₹¥₩₦]|\b(?:usd|inr|eur|gbp|jpy|krw|ngn)\b",
    # English + Spanish/Portuguese/French/Hindi(Latin-script)/Indonesian
    # urgency phrasing - same MiniLM multilingual corpus this project
    # already embeds text with (paraphrase-multilingual-MiniLM-L12-v2), so
    # these regex flags cover the same language mix rather than English-only.
    "has_urgency_keyword": (
        r"\b(?:urgent|immediately|act now|expires?|last chance|final notice|limited time"
        r"|urgente|inmediatamente|ahora mismo|expira|ultima chance|urgent(?:e)?|immediatement"
        r"|jaldi|abhi|turant|segera|sekarang)\b"
    ),
    "has_prize_keyword": (
        r"\b(?:won|winner|prize|reward|claim now|congratulations|selected"
        r"|ganador|premio|felicidades|reclamar ahora|ganhador|premio|parabens"
        r"|felicitations|gagnant|inaam|jeeta|badhai ho|pemenang|hadiah|selamat)\b"
    ),
    "has_gambling_keyword": (
        r"\b(?:casino|betting|lottery|jackpot|bet now|poker"
        r"|loteria|apuesta|apostar ahora|loteria|aposta|sorteio"
        r"|satta|matka|jua|judi|togel)\b"
    ),
    "has_loan_keyword": (
        r"\b(?:loan|credit approved|pre-?approved|cash advance|instant loan"
        r"|prestamo|credito aprobado|prestamo instantaneo|emprestimo|credito aprovado"
        r"|pret|credit approuve|karza|rin|udhaar|pinjaman|kredit disetujui)\b"
    ),
    "has_otp_keyword": r"\b(?:otp|one[- ]time password|verification code|security code|code de verification|codigo de verificacion|codigo de seguranca)\b",
    "has_excessive_punctuation": r"[!?]{2,}",
    # Impersonation of a bank/government/delivery entity plus a call-to-
    # action verb - distinct from generic urgency, catches phishing/smishing
    # framed as an official notice rather than a marketing-style offer.
    "has_account_verification_keyword": (
        r"\b(?:verify your account|account suspended|account locked|unusual activity"
        r"|update your (?:kyc|details|payment)|confirm your identity|reactivate your account"
        r"|verifica tu cuenta|cuenta suspendida|confirme sua conta)\b"
    ),
    "has_delivery_scam_keyword": (
        r"\b(?:package (?:is )?held|delivery failed|redeliver(?:y)?|customs fee|pay a? ?small fee"
        r"|shipment on hold|track your (?:package|parcel|order)|paquete retenido|entrega fallida)\b"
    ),
    "has_crypto_investment_keyword": (
        r"\b(?:crypto|bitcoin|invest(?:ment)? opportunity|guaranteed returns?|double your money"
        r"|forex signals?|trading bot|inversion garantizada|criptomoneda|retornos garantidos)\b"
    ),
    "has_tax_authority_impersonation_keyword": (
        r"\b(?:irs|tax refund|income tax department|tax rebate|customs duty|penalty notice"
        r"|reembolso de impuestos|receita federal|imposto de renda)\b"
    ),
}

# --- Text embeddings (features/text_embeddings.py) -----------------------
# paraphrase-multilingual-MiniLM-L12-v2 was picked
# over the originally-named Distil-mBERT/XLM-R production candidates
# specifically because it's ALREADY trained for sentence-similarity
TEXT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_EMBEDDING_BATCH_SIZE = 256  # sentence-transformers' internal encode() batch size -
# tune down if running on constrained memory, up if a GPU is available

# --- FAISS near-duplicate index (features/faiss_index.py) ----------------
# Cosine similarity cutoff for "counts as near-duplicate" - not
# calibrated, a starting point from real scores seen while building this:
# true paraphrase 0.93, same-category-different-template ~0.81-0.84,
# unrelated ~0.11-0.43. 0.92 sits between the top two.
FAISS_NEAR_DUP_THRESHOLD = 0.92

# Short catches bursts; long catches paced-out campaigns that never cluster
# within an hour (a scammer sending one near-dup every few hours across
# many sender IDs shows zero 1hr matches, but real signal over 24hr).
# Only matches STRICTLY BEFORE the message being scored count, within
# each window (point-in-time, same convention as behavioral.py).
FAISS_NEAR_DUP_WINDOW_SHORT = BEHAVIORAL_LONG_WINDOW  # "1h" - reuses
# behavioral.py's long-window value, no evidence yet they should differ
FAISS_NEAR_DUP_WINDOW_LONG = "24h"

# Target rows per chunk for compute_near_dup_features_chunked() - bounds
# memory to roughly this many rows regardless of total corpus size (see
# that function's docstring). ~500k rows x 384 floats x 4 bytes =~
# 770MB per chunk, comfortable headroom on typical hardware.
FAISS_CHUNK_SIZE = 500_000

# How many query rows compute_near_dup_features() sends to FAISS per call -
# does NOT bound the index itself (that's FAISS_CHUNK_SIZE + the 24hr
# buffer, which for a dense real corpus can be a large fraction of the
# whole thing - see that function's docstring). Batching the QUERY side
# spreads a call's total match allocation across many smaller calls -
# changes nothing about which matches get found, just bounds peak memory
# per call.
FAISS_QUERY_BATCH_SIZE = 10_000

# --- Serving-time FAISS index (serving/anomaly_scoring.py) --------------
# The batch/training path above (compute_near_dup_features_chunked) stays
# exact IndexFlatIP - each chunk is already bounded to FAISS_CHUNK_SIZE +
# a window buffer, so brute-force is fine there (see
# features/faiss_index.py::build_index()'s docstring). Serving is
# different: it holds the ENTIRE per-source historical corpus in ONE
# index, queried once per live request - SS7's full corpus is now
# 2,742,301 vectors, a real memory constraint on the serving host that
# ruled out both exact IndexFlatIP (raw 384*4=1536 bytes/vector) and
# HNSW (~1.5-2x that for graph overhead). IVF-PQ trades some recall for
# real compression (m bytes/vector) - accepted specifically because the
# constraint here is memory, not primarily latency or accuracy, but the
# params below are chosen to keep latency and recall reasonable too, not
# to chase maximum compression alone.
FAISS_SERVING_INDEX_TYPE = "ivfpq"  # vs "flat" (exact, batch/training default)

# nlist: number of k-means coarse-quantizer cells. Sized off FAISS's own
# published guidance (roughly 4*sqrt(N) to 16*sqrt(N) for N in the low
# millions) - sqrt(2,742,301) =~ 1655, so 4096 sits inside that range,
# toward the lower end (favors faster/cheaper training and search over
# maximum partition granularity). NOT tuned/validated against this
# corpus's real recall yet - a starting point, same spirit as
# FAISS_NEAR_DUP_THRESHOLD's own "not calibrated" starting-point comment.
FAISS_IVFPQ_NLIST = 4096

# nprobe: how many of those nlist cells get searched per query - the real
# latency/recall dial (higher = slower + closer to exact, lower = faster
# + more approximate). 32 of 4096 cells (~0.8%) is a deliberately
# middle-of-the-road starting point, not tuned - raise this first (before
# touching nlist/m/nbits) if a real recall measurement comes back too low,
# since it's the cheapest lever to turn without rebuilding the index.
FAISS_IVFPQ_NPROBE = 32

# m: number of sub-vector splits for product quantization - must evenly
# divide the embedding dimension (384). CHECKED empirically (synthetic
# near-dup benchmark: clustered vectors calibrated to this project's own
# real "near-dup" similarity range, ~0.88-0.95 within-cluster - see the
# session that added this comment for the exact test), not just a rule-
# of-thumb guess: m=48 (8 dims/sub-vector) LOST a true near-dup match
# exact IndexFlatIP search found (quantized self-similarity dropped to
# 0.915, below FAISS_NEAR_DUP_THRESHOLD=0.92 - a real recall regression,
# not a rounding artifact). m=64 (6 dims/sub-vector) recovered EXACT
# recall parity with flat search in that same test, at 64 bytes/vector
# vs 384*4=1536 bytes raw = 24x compression (vs m=48's 32x) - a small
# compression cost for closing a real accuracy gap. Still only a
# synthetic-benchmark validation, not measured against this project's
# actual corpus - re-verify against real embeddings once that's
# practical, don't assume this transfers unchanged.
FAISS_IVFPQ_M = 64

# nbits: bits per sub-quantizer codebook - 8 is FAISS's standard default
# (256 centroids/sub-vector); needs roughly >= 256 real training points
# per nlist cell to fit well, comfortably true at this corpus size.
FAISS_IVFPQ_NBITS = 8

# Coarse-quantizer + PQ codebook training is a one-time cost per process
# (the index is built once and cached - see
# serving/anomaly_scoring.py::_load_corpus()), but training on the FULL
# 2.74M-vector corpus is unnecessary - FAISS's own guidance is that
# ~30-256x nlist training points is enough to shape cell/codebook
# boundaries well. Capped here to bound index-build time; add() still
# indexes every real vector regardless of how many were used to train,
# so this does NOT reduce how much of the corpus is actually searchable.
FAISS_IVFPQ_TRAIN_SAMPLE_SIZE = 500_000

# Hard cap on near-dup candidates returned PER QUERY MESSAGE -
# compute_near_dup_features() uses a fixed-K faiss.Index.search() (top-K
# by similarity), then threshold-filters the K results, instead of
# faiss.Index.range_search() (find EVERY match above threshold, however
# many). range_search's uncapped result size was the real cause of a
# native crash (STATUS_STACK_BUFFER_OVERRUN, exit -1073740791) on a dense
# real corpus - one bursty sender's messages within a chunk+24hr-buffer
# can mutually match in the hundreds of thousands to millions of pairs.
#
# This is a genuine tradeoff, not a free optimization - same one common
# production near-dup/dedup systems make: bounded, predictable resource
# usage over guaranteed-exact counts for pathological cases. If a
# message's true near-dup count exceeds this cap, it's undercounted (see
# that function's truncation warning) - accepted because a message that
# hits a 100k-candidate cap is already unambiguously an extreme,
# rule-worthy outlier; the exact count past that point adds little.
#
# Sized off real data: SMPP's real sender_msgs_last_1hr distribution
# (see models/rule_pattern/train.py's REAL LABEL COMPOSITION-style
# checks) has median 6,726, 99.99th percentile 48,554, max 48,606 -
# 100k gives real headroom above the single worst observed hour, while
# still bounding compute/memory to a small, fixed fraction of what a
# true unbounded blowup could reach.
FAISS_MAX_MATCHES_PER_QUERY = 100_000

# --- Retraining automation (scripts/check_retrain_trigger.py) -----------
# Not tuned against a real target - starting points, same "documented, not
# proven" status as FAISS_NEAR_DUP_THRESHOLD/N_EMBEDDING_COMPONENTS above.
#
# RETRAIN_MIN_NEW_ROWS: enough new rule_evaluated volume since the last
# full LightGBM/Isolation Forest retrain (compare_versions.py's promotion
# gate) to be worth the cost of a full pipeline run + retrain + promotion
# check - deliberately large, this is the expensive trigger.
RETRAIN_MIN_NEW_ROWS = 50_000

# DBSCAN_MIN_NEW_ANOMALY_ROWS: DBSCAN re-runs are cheap and decoupled from
# model retraining (clusters on frozen pretrained MiniLM embeddings, needs
# only a re-run, not a retrain) - can fire far more often than
# RETRAIN_MIN_NEW_ROWS, so this threshold is much smaller.
DBSCAN_MIN_NEW_ANOMALY_ROWS = 5_000
