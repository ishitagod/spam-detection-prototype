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
BEHAVIORAL_SHORT_WINDOW = "5min"  # sender_msgs_last_5min, sender_unique_destinations_5min,
# sender_recipient_diversity_ratio_5min, sender_velocity_zscore_5min
BEHAVIORAL_LONG_WINDOW = "1h"  # sender_msgs_last_1hr, sender_unique_destinations_1hr,
# sender_repeat_content_ratio_1hr, sender_recipient_diversity_ratio_1hr

# --- Text embeddings (features/text_embeddings.py) -----------------------
# Switched from all-MiniLM-L6-v2 (English-only, 384-dim/6-layer) to this
# multilingual variant after checking the real data: a Malay-marker
# heuristic over a 5,000-message sample found ~10-11% of BOTH sources'
# traffic is genuine Bahasa Malaysia/mixed content (this is Malaysian-
# market SMS - CIMB/OCBC/UOB/AirAsia, etc.), which an English-only model
# would embed poorly. paraphrase-multilingual-MiniLM-L12-v2 was picked
# over the originally-named Distil-mBERT/XLM-R production candidates
# specifically because it's ALREADY trained for sentence-similarity
# (sentence-transformers-native, same family/objective as the model it's
# replacing) - Distil-mBERT's base checkpoint is a masked-language model,
# not a sentence-embedder, and using it directly (not fine-tuned first)
# tends to perform worse at this project's actual job (similarity/near-
# dup matching) than a purpose-built model, even a smaller one. Verified
# working: same 384-dim output as the model it replaces (12 layers vs 6,
# ~2x the CPU cost, not 4x like the Distil-mBERT-based alternatives),
# real EN/Malay-paraphrase similarity check confirmed cross-lingual
# matching actually works before committing to this swap.
# Same model name FastAPI will load later for live single-message
# encoding at inference time - one model, two callers (this batch script,
# and the future serving path), never two separately-tuned encoders.
TEXT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_EMBEDDING_BATCH_SIZE = 256  # sentence-transformers' internal encode() batch size -
# tune down if running on constrained memory, up if a GPU is available

# --- FAISS near-duplicate index (features/faiss_index.py) ----------------
# Cosine similarity cutoff for "counts as near-duplicate" - not
# calibrated, a starting point from real scores seen while building this:
# true paraphrase 0.93, same-category-different-template ~0.81-0.84,
# unrelated ~0.11-0.43. 0.92 sits between the top two.
FAISS_NEAR_DUP_THRESHOLD = 0.92

# Two windows, not one - mirrors behavioral.py's short/long split. Short
# catches bursts; long catches paced-out campaigns that never cluster
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
