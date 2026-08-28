"""
Tunable operational parameters - values someone might reasonably want to
change without touching pipeline logic. Deliberately does NOT include
protocol constants (e.g. GSM 03.38's DCS alphabet-bit masks in
ingestion/smpp.py) - those are spec, not config, and externalizing them
would just add indirection around a value that will never change.
"""

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

# How many query rows compute_near_dup_features() sends to index.range_search()
# per call - does NOT bound the index itself (that's FAISS_CHUNK_SIZE +
# the 24hr buffer, which for a dense real corpus can be a large fraction
# of the whole thing - see that function's docstring). range_search's
# match-array size scales with query-count x candidate-density, so one
# call over the FULL chunk+buffer as queries can allocate an enormous
# array in one shot (a real crash hit in practice, exit -1073740791,
# on a corpus this dense). Batching the QUERY side only spreads that same
# total allocation across many smaller calls - it changes nothing about
# which matches get found, just bounds peak memory per call.
FAISS_QUERY_BATCH_SIZE = 10_000
