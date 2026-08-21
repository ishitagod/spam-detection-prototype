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
BEHAVIORAL_SHORT_WINDOW = "5min"  # sender_msgs_last_5min
BEHAVIORAL_LONG_WINDOW = "1h"  # sender_msgs_last_1hr, sender_unique_destinations_1hr,
# sender_repeat_content_ratio_1hr
