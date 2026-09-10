# Feature Reference — SS7 vs SMPP

The **full** feature inventory across both sources: canonical schema,
source-specific raw fields, behavioral features (training-time and
Feast-servable), text embeddings, FAISS near-dup, labels, and which of
these each model actually consumes.

Scope note: `docs/feature_catalog.md` already documents the narrower
"what's servable from Feast" slice in detail (join keys, on-demand vs
stored, known limitations) — this doc is the wider inventory everything
else lives in; where they overlap, `feature_catalog.md` is the more
detailed source on Feast specifics.

## 1. Canonical schema (shared, `common/schemas.py`)

Every feature below this point is computed FROM these columns, after
`ingestion/{smpp,ss7}.py::map_to_canonical()` maps each source's raw
fields into them. This is the contract that lets one shared ML pipeline
run on both sources.

| Column | Type | SMPP | SS7 |
|---|---|---|---|
| `record_id` | object | join key | join key |
| `message_id` | object | not populated (NA) | populated (`reference`) |
| `source` | object | `"SMPP"` | `"SS7"` |
| `originator` | object | business sender ID (`oa`) | real MSISDN (`calling_gt`) |
| `destination` | object | `da` | `called_gt` |
| `text` | object | decoded, UDH-stripped | decoded |
| `timestamp` | object | `time_stamp` | `time_stamp` |
| `dcs` | float64 | sign-corrected in `clean()` | already unsigned |
| `text_decode_failed` | bool | real feature, not bookkeeping (both sources — a message inference must still score even if undecodable, so training data keeps it rather than dropping it) | same |
| `esm_class` | float64 | populated (`esme_class`) | NA (no SS7 equivalent) |

`originator`/`destination` are **different namespaces per source** —
SMPP originators are business sender-ID strings, SS7 originators are
real MSISDNs. Every sender-keyed feature below uses the compound key
`source|originator` specifically to avoid a cross-source collision.

## 2. Source-specific raw fields carried through unchanged

These aren't canonical (no shared-schema entry), but survive
`map_to_canonical()` into `messages_with_behavioral.csv` as extra,
source-only columns — real signal, not dropped just because the other
source has no equivalent (per `CLAUDE.md`: *"SS7-specific fields may
remain SS7-only when they carry real signal"* — same principle applied
symmetrically to SMPP here).

**SMPP-only** (`ingestion/smpp.py::FEATURE_MAP`):
`originator_ton`/`originator_npi` (business sender ID vs. real MSISDN
often differ here), `system_id` (aggregator/binding identity — a
reputation key above individual sender ID), `messaging_mode`,
`source_ip`/`dest_ip`/`instance_id` (connection/session-level grouping,
finer than `system_id`), `virtual_gt` (proxy identity behind the
displayed sender — two different `originator` values sharing a
`virtual_gt` is a spoofing/rotation signal), `concat_ref`/
`concat_total_parts`/`concat_part_num` (multipart grouping key, sourced
from UDH concatenation bytes in `content`, SAR fields as fallback).

**SS7-only** (`ingestion/ss7.py::FEATURE_MAP`):
`create_date` (distinct from `timestamp` — exact semantic difference not
yet confirmed), `smsc`, `imsi` / `virtual_imsi` (physical SIM identity —
see IMSI-LINKAGE below), `vlr_address` (roaming/serving-location signal,
merged in from the matching `MT_SRI_response` row; NA for MO rows —
expected, no SRI phase exists for those, not missing data), `message_type`
(MO=3 vs MT_request=2 — the only two kept after row-filtering; drives the
row-filtering rule itself, not folded into a generic cross-source
`event_type` since SMPP has no equivalent signal there), `ton`/`npi`,
`pid`, `tpdu_length`, `concat_ref`/`concat_total_parts`/`concat_part_num`
(same shape as SMPP's, sourced from native `sarref`/`msg_part`/
`msg_parts` instead of UDH).

**Deliberately excluded on both sides** (checked against real data, not
guessed): SMPP's `msg_type` (constant, zero signal), `gsm_features`
(superseded by content-byte UDH detection), `sequence_no`,
`receipted_message_id` (always NULL on op-4 rows); SS7's
`virtual_vlr_outbound_smsc_gt` (entirely NULL in the data checked),
`b_number`/`msisdn` (superseded by `called_gt`/`calling_gt`).

## 3. Behavioral features — point-in-time, training (`features/behavioral.py`)

Computed per **message**, using only strictly-earlier same-sender history
(never the current message itself — a sender's first-ever message
correctly gets all-zero/cold-start values, not excluded). Sender key is
`source|originator` for all of these except the one IMSI-keyed feature.
Windows: `1min` (very short) / `5min` (short) / `1h` (long) —
`config/settings.py`.

| Feature | Window | SMPP | SS7 |
|---|---|---|---|
| `sender_msgs_last_1min` | 1min | ✓ | ✓ |
| `sender_msgs_last_5min` | 5min | ✓ | ✓ |
| `sender_msgs_last_1hr` | 1h | ✓ | ✓ |
| `sender_unique_destinations_5min` | 5min | ✓ | ✓ |
| `sender_unique_destinations_1hr` | 1h | ✓ | ✓ |
| `sender_recipient_diversity_ratio_5min` | 5min | ✓ | ✓ |
| `sender_recipient_diversity_ratio_1hr` | 1h | ✓ | ✓ |
| `sender_repeat_content_ratio_1hr` | 1h | ✓ | ✓ |
| `sender_velocity_zscore_5min` | 5min (baseline: sender's own full history) | ✓ | ✓ |
| `sender_age_days` | all-time (no window) | ✓ | ✓ |
| `imsi_distinct_originators_1hr` | 1h | **absent** (no IMSI concept) | ✓ SS7-only |

`imsi_distinct_originators_1hr` is the SIM-farming signal: keyed on
`imsi` (the physical SIM), not `source|originator` (the apparent sender
identity) — a genuinely different entity, the whole point being to catch
cases where the two diverge. Column is left **off** SMPP's output
entirely (not NaN-filled) since the underlying concept doesn't exist
there.

## 4. Feast-servable behavioral snapshot (`features/behavioral_snapshot.py`, current-state, both sources)

A **different, simpler** computation than section 3 — "as of right now,
what's each sender's current state" (refreshed by
`scripts/refresh_feast.py`) vs. section 3's "as of just before this
specific row." Same feature names/meanings, subset of section 3's list
(`docs/feature_catalog.md` has the full stored/on-demand/entity detail):

- **Stored** (`sender_behavioral_stats` FeatureView): `sender_msgs_last_5min`,
  `sender_msgs_last_1hr`, `sender_unique_destinations_1hr`,
  `sender_age_days`, `sender_recipient_diversity_ratio_5min/1hr`,
  `sender_velocity_zscore_5min`, plus `recent_text_counts_json`
  (internal — exists only to feed the on-demand feature below).
- **On-demand** (`sender_repeat_content_ratio` feature view — needs the
  live request's own candidate text, can't be precomputed):
  `sender_repeat_content_ratio_1hr`.
- **Separate entity, SS7-only** (`imsi_behavioral_stats` FeatureView, keyed
  on `imsi`): `imsi_distinct_originators_1hr`.

## 5. Text embeddings (`features/text_embeddings.py`)

One 384-dim vector per message, model
`paraphrase-multilingual-MiniLM-L12-v2` (`config/settings.py`'s
`TEXT_EMBEDDING_MODEL` — chosen for multilingual coverage across this
real, multi-language SMS traffic; note `features/text_embeddings.py`'s own
module docstring still says `all-MiniLM-L6-v2`, which is stale relative to
the actual constant it imports and uses — worth a one-line fix). Computed
identically for both sources; deduplicated by distinct text, then
broadcast back to every row sharing that text (the expensive step: ~14ms
per distinct text on CPU). Currently a **full, non-sampled** run for both
sources (SMPP 5,505,921 rows, SS7 2,742,301 rows — the earlier
`--sample_n`-restricted 40k/20k run has been superseded).

## 6. FAISS near-duplicate features (`features/faiss_index.py`)

Computed from the embeddings above, both windows, identical for both
sources: `near_dup_match_count_1hr`, `near_dup_max_similarity_1hr`,
`near_dup_distinct_senders_1hr`, `near_dup_match_count_24hr`,
`near_dup_max_similarity_24hr`, `near_dup_distinct_senders_24hr`.
FlatIP index. SIM-farming (`imsi_distinct_originators_1hr`, section 3) is
a **separate** SS7 behavioral feature, deliberately not folded in here.

## 7. Labels

| Column | Source | Meaning |
|---|---|---|
| `rule_evaluated` | `labels/rule_labels.py` | a REAL (non-`SW_*`) content rule reached decision 0 or 1 — both sources |
| `rule_flagged` | `labels/rule_labels.py` | `fraud_type=="spam"` specifically, restricted to `rule_evaluated==True` rows; NA (not False) when never evaluated |
| `cluster_fraud_type_label` | `labels/cluster_labels.py` | hand-confirmed DBSCAN cluster identity (`models/anomaly/cluster_discovery.py` → `inspect_clusters.py` → `ingest_cluster_labels.py`) — its own label source, never merged into `rule_flagged` |

Real current composition (`docs/experiments/rule_pattern.md` has the full
detail/history): SMPP 139,546 `rule_evaluated` rows (1.9% flagged), SS7
2,654,369 rows (10.7% flagged) — both sources now have real confirmed-clean
populations, spam is a minority on both.

## 8. What each model actually consumes

Neither model uses every feature above — each has a deliberately scoped
subset (`docs/ml/modeling.md` has the full rationale).

**`rule_pattern_score`** (LightGBM, `models/rule_pattern/data.py`) —
canonical (`dcs`, `text_decode_failed`, `text_length`) + section 3's
behavioral columns **raw, unbucketed** (including `sender_age_days`,
diversity ratios, `sender_velocity_zscore_5min` — NaN passed through
natively, no `_known` indicators needed) + `imsi_distinct_originators_1hr`
+ one-hot `source`. No embeddings by default (`--with_embeddings`/
`--with_tfidf` are opt-in, independently toggleable).

**`anomaly_score`** (Isolation Forest, `models/anomaly/data.py`) —
embeddings (PCA to 30 dims) + near-dup (section 6) + behavioral, but
**not raw** for two columns found to break Isolation Forest specifically
(narrow-range/small-sample statistical artifacts, not real signal — see
`docs/experiments/anomaly.md`):
- `sender_age_days` → bucketed (`<1hr`, `1hr–1day`, `1day–7day`,
  `7day–30day`, `≥30day`), not the raw day-count.
- `sender_recipient_diversity_ratio_5min`/`1hr` → gated on message count
  (`< 3` messages in the window ⇒ treated as unknown, not a trivially
  extreme 0/1 ratio), each paired with a `_known` indicator.
- `sender_velocity_zscore_5min`/`imsi_distinct_originators_1hr` → raw
  value + `_known` indicator (can be genuinely NaN).

**`cluster_discovery.py`'s DBSCAN step** — embeddings (PCA) + near-dup
**only**. Deliberately excludes ALL behavioral columns from the
clustering distance metric (unlike Isolation Forest, which needs them) —
mixing sender-behavioral differences into a "which messages are the same
campaign" question let identical spam templates fragment across many
clusters; standard practice for this task is content-similarity
clustering with behavioral features kept as a separate reporting/filter
layer, not blended into cluster membership.

## 9. Quick reference — feature availability by source

| Feature group | SMPP | SS7 |
|---|---|---|
| Canonical schema | ✓ (no `message_id`) | ✓ (no `esm_class`) |
| Behavioral (sender-keyed) | ✓ full set | ✓ full set |
| IMSI SIM-farming | ✗ | ✓ |
| Text embeddings | ✓ full-scale | ✓ full-scale |
| FAISS near-dup | ✓ | ✓ |
| `rule_pattern_score` training pool | 139,546 rows, 1.9% flagged | 2,654,369 rows, 10.7% flagged |
| `anomaly_score` — split models | `anomaly_SMPP` | `anomaly_SS7` |
