# Domain Knowledge & Architecture Decisions Reference

Consolidated reference covering protocol fundamentals, CDR field meanings,
fraud-type feature engineering, and model architecture principles worked
out across design discussions. Complements (does not replace)
`docs/architecture.md` and `docs/ml/modeling.md` - those are the
authoritative current-state docs; this is background/reference knowledge
those decisions were built on.

---

## 1. Protocol Fundamentals

### SMPP (A2P - Application to Person)

External systems (aggregators, enterprises) inject messages via a
client-server protocol over TCP/IP, since they aren't part of the mobile
core and have no other way in.

- `submit_sm` (command ID `0x00000004`) - client submits a message
- `submit_sm_resp` (`0x80000004`) - SMSC acknowledges (response = request
  ID with high bit set - `0x80000000` OR'd in, universal SMPP pattern)
- `deliver_sm` (`0x00000005`) - used for TWO purposes under one op code:
  genuine MT push to a client, OR a delivery receipt (DLR). Only DLR rows
  populate `receipted_message_id`, linking back to the original
  `message_id`. **Filtering to "op 4 and 5 = all submit_sm" is wrong** -
  op 5 is a structurally different event (receipt/status), not a plain
  submission.

### SS7 (P2P - Person to Person delivery, and A2P's delivery leg)

Phones are already part of the SS7 signalling core - no separate
"login"/injection protocol needed. SMPP only gets an A2P message as far as
the SMSC; final delivery to a handset is ALWAYS SS7 (SRI + MT), regardless
of how the message entered.

**MO / SRI / MT relationship** (observed message_type mapping in this
project's real data - values are pipeline-specific, not a universal SS7
standard, confirm against your own ingestion):
```
MO   (subscriber originates) -> no SRI needed for the sender's OWN location
                                  (they're already connected to a serving MSC)
MT_SRI_request  -> SMSC asks HLR: "where is the RECIPIENT?"
MT_SRI_response -> HLR answers with routing info (VLR/MSC) - THIS IS THE
                    ONLY ROW TYPE THAT CARRIES vlr_address
MT_request      -> SMSC pushes the message to that MSC
MT_response     -> MSC confirms delivery (or reports failure)
```
- SRI is always about the **recipient's** location, never the sender's -
  "MT_SRI" naming makes this explicit. A subscriber's OWN originating
  location is not captured by SRI/VLR at all in the fields available to
  this project - `smsc` and `file_name` are the only weak proxies
  identified so far; ask the ingestion owner whether the underlying
  system logs an originating-MSC/serving-node identifier that isn't yet
  exposed in this export.
- One logical message can produce **up to 5 rows** across these types.
  Velocity/volume features MUST count distinct `reference` values, not
  raw row counts, or real traffic volume gets ~5x inflated.
- An incomplete sequence (e.g., SRI request with no response, or SRI
  resolved but no MT delivery follows) is itself a signal - can indicate
  stalled/rejected delivery or reconnaissance-style HLR querying (GT
  scanning) with no real delivery intent.

### SIP (VoIP/IMS voice - not yet built, noted for future Voice work)

Structurally closer to SMPP than SS7 (text-based, request/response over
IP) despite carrying voice. `INVITE` = call setup (SS7's `IAM`
equivalent), `From`/`To` headers = origin/destination (SS7's GT/MSISDN
equivalent) but far more spoofable, since SIP trunk providers often don't
enforce strict `From`-header validation the way carrier SS7 interconnects
do. A SIP trunk is voice's version of an SMPP bind - an external system's
bolt-on doorway into the network.

---

## 2. Raw CDR Field Reference

### Fields confirmed/decoded through direct investigation

| Field | Meaning | Caveat |
|---|---|---|
| `oa_ton`/`da_ton` | Type of Number: 0=unknown, 1=international (has country code), 2=national (no country code), 3=network specific, 5=alphanumeric | NOT evidence of actual international routing - just address FORMAT. In this project's real data, `da_ton` was found CONSTANT at 1 (gateway normalizes all destinations to E.164) - zero discriminative signal. `oa_ton` DOES vary (short codes = 2, brand IDs = 5) - useful. Always check variance before trusting a TON-based feature. |
| `app_dest_port`/`app_src_port` | Populated = WAP Push / binary-addressed SMS (Service Indication with embedded URL, renders as clickable notification not plain text) | Strong, cheap smishing signal - legitimate transactional SMS essentially never uses this path |
| `sar_ref`/`sar_msg_parts`/`sar_msg_part` | Multi-part/concatenated SMS reassembly fields | Must reassemble multipart messages into one logical message BEFORE embedding/near-dup - otherwise each fragment gets scored independently, corrupting content features |
| `receipted_message_id` | Only populated on delivery-receipt rows (deliver_sm), links back to original `message_id` | Enables delivery-failure-rate feature via join; do NOT treat `deliver_sm` rows as new messages for velocity counting |
| `calling_gt` vs `msisdn` (SS7) | GT = signalling-layer routing address (may identify a NETWORK ELEMENT like the SMSC, not the subscriber); msisdn = actual subscriber number | These are NOT interchangeable - confirm empirically which is reliably populated for which message_type before using either as "the originator" |
| `imsi` vs `msisdn` | IMSI = SIM's permanent hardware identity; MSISDN = the dialable number (self-declared in signalling, comparatively easy to spoof) | SIM-farm/spoofing detection should key behavioral features on `imsi` where possible - hardware identity is expensive to fake, phone number claims are not |
| `virtual_imsi`, `virtual_gt`, `virtual_vlr_outbound_smsc_gt` | Likely roaming/visited-network view vs. home-network view of the same identity | Unconfirmed - a mismatch between virtual and non-virtual versions of the same field is a hypothesis worth testing, not yet validated |
| Numeric ID columns (`oa`, `da`, `msisdn`, `imsi`, etc.) | Real-world example seen: `oa=61000.00`, `da=60176522205.00` | **MUST be loaded as strings, not floats.** Float storage silently drops leading zeros and can lose precision on IMSI-length (15-digit) values beyond float64's exact-integer range (~2^53). Breaks any prefix-matching or exact-lookup feature if left numeric. |

### Fields flagged as needing empirical confirmation before building features on them

Same discipline applied throughout: check actual value distributions in
real data before trusting a spec definition or a plausible-sounding guess.

- `esme_class` - NOT the same as standard `esm_class` (UDHI/reply-path
  bitmask). Likely a vendor-specific ESME (client connection) category
  (aggregator/direct/reseller) - confirm via value_counts before use.
- `gsm_features` - likely a human-readable unpacking of `esm_class` bits
  (UDHI, reply-path flags) - unconfirmed.
- `messaging_mode` - possibly store-and-forward vs. datagram delivery -
  unconfirmed.
- `dcs` values outside the common 0/4/8 set (e.g., 15 landed on a
  spec-"reserved" combination in one check) - cross-check against
  `decoded_content` readability before trusting.
- `inverted` - meaning genuinely unclear from column name alone.

**General rule established repeatedly in this project**: a column name or
spec definition is a *hypothesis*, not a fact, until checked against real
value distributions in this project's actual data. Several early guesses
in this doc's source discussions were later corrected once real data was
checked (e.g., `oa_ton=3` guessed as "short code" was corrected to
`oa_ton=2`/National once real values were seen).

---

## 3. Fraud-Type -> Feature Mapping

Consolidated reference table. Many features are shared across fraud
types - that overlap is itself useful information (see Section 4).

| Fraud Type | Key CDR Columns | Derived Features | Note |
|---|---|---|---|
| **Spam (general)** | `decoded_content`, `oa`, `sar_ref`/`sar_msg_parts` | `text_embedding`, `near_dup_match_count_1hr`, `repeat_content_ratio`, `sender_age_days` | Multipart reassembly is a data-processing prerequisite, not a feature itself |
| **Flooding** | `oa`/`msisdn`, `da`/`b_number`, `reference` | `msgs_last_1min/5min/1hr`, `unique_recipients_1hr`, `recipient_diversity_ratio`, `velocity_zscore` (vs. own baseline, not raw volume) | Raw volume alone isn't the signal - a legitimate high-volume sender can be fine; deviation from THAT sender's own baseline is what matters. Cold-start (no baseline yet) needs a separate age-gated rule. |
| **Phishing/Smishing** | `decoded_content` (URLs), `app_dest_port` | URL structural features (having-IP, prefix-suffix, subdomain, shortener - all zero-external-call), domain reputation (needs async enrichment - WHOIS/SSL/DNS, NOT inline), `is_wap_push` | Domain reputation lookups are too slow for the inline path - separate async enrichment + cache, inference only ever does a fast cache read |
| **Spoofing (sender ID / CLI)** | `oa`, `system_id`, `source_ip`; SS7: `calling_gt`, `msisdn`, `vlr_address` | `neighbor_spoof_score` (oa/da digit-prefix similarity, TON-aware), `impersonation_score` (fuzzy match against known-brand reference list), `system_id` baseline-mismatch, NCLI flag (missing calling_gt/msisdn) | Neighbor spoofing = sender ID engineered to resemble the RECIPIENT's own number - near-zero legitimate baseline rate, strong signal |
| **SIM Box/Farming** | `imsi`, `vlr_address` (SRI-response rows only), `calling_gt` | `msgs_last_1hr` keyed by `imsi` not `msisdn`, `imsi_age_days`, `distinct_msisdns_per_imsi_last_1hr`, `vlr_changes_last_1hr` (needs new rolling-change computation) | Originator's OWN location is NOT directly observable (see Section 2) - this is a real, acknowledged data gap, not yet solved |
| **eSIM provisioning abuse** | N/A in current schema | N/A | **Data gap** - requires a separate SM-DP+ provisioning log source; not derivable from SMPP/SS7 CDRs at all |
| **SRI Spoofing / SMS interception** | `vlr_address` (SRI-response only), `imsi`/`msisdn`, `time_stamp` | `vlr_flip_rate`, `impossible_travel_flag` (distance/time plausibility check) | Higher severity than traffic-abuse fraud types - this is account-takeover/OTP-interception adjacent, not just revenue fraud |
| **SMS Pumping / AIT** | `da`, `system_id`, `decoded_content` | `destination_concentration_score`, `otp_request_velocity_per_destination`, `otp_pattern_match` | Confirming actual fraud (vs. just flagging the pattern) needs app-layer signal not present in CDR data - CDR can only flag traffic PATTERN |
| **DLR Faking** | `message_state`, `receipted_message_id`, `system_id`, `time_stamp` | `delivery_confirmation_rate_per_aggregator`, `dlr_latency_anomaly` | Suspiciously perfect delivery rates are themselves the signal - real delivery always has some failure rate |
| **Neighbor Spoofing / Impersonation** (from smishing research) | `oa` vs `da` prefix; `oa` vs curated brand list | `neighbor_spoof_score`, `impersonation_score` | See Spoofing row - same features |
| **International/roaming gap exploitation** | `system_id` (aggregator-type lookup if available), destination country code, `vlr_address` | `content_vs_routing_mismatch` (claims domestic trust, routes internationally), roaming+anomaly combination flag | Roaming alone is NOT a signal (legitimate) - only roaming COMBINED with other anomaly signals |

### Data-hygiene prerequisites before any of the above is trustworthy

1. Load identifier columns as strings, never floats (Section 2)
2. Count distinct `reference` values for velocity, never raw rows
3. Only use features knowable **at actual inference time** - never
   post-hoc/completed-transaction fields (point-in-time discipline,
   Section 5)
4. Keep rule-engine output columns (`decision`/`rule`/`rule_name`/
   `fraud_type`/`status`) strictly as a LABEL SOURCE, never a model input
   feature - this is a direct leak, more severe than the general
   rule-derived-label circularity concern, since it hands the model the
   answer directly

---

## 4. Model Architecture: The Evidence-Based Splitting Principle

This is the single most-applied decision framework across this project.
Stated once, applied repeatedly - worth understanding as a general
principle, not a one-off choice.

### The principle

**Start with one shared model. Any candidate split dimension (data
source, fraud type, operator) goes in as a FEATURE first, not a routing
key.** Only split into separate models/pipelines if per-segment evaluation
(PR-AUC/log loss computed separately for each candidate segment)
empirically shows the shared model underperforming on a specific segment
in a way that added features can't close.

### Why this matters - the cost side of the tradeoff

Every additional split axis:
- Fragments training data (each model sees a smaller slice - a real
  problem when already volume-constrained)
- Duplicates maintenance (rules, features, retraining, versioning, drift
  monitoring - all multiplied by the number of models)
- Risks silent drift between parallel pipelines that should stay
  comparable but aren't enforced to
- Compounds badly if multiple split dimensions are combined without each
  being individually evidence-tested (e.g., do NOT combine an untested
  fraud-type split with an already-tested-negative source split into an
  assumed model matrix)

### Applied examples in this project

| Candidate split | Status | Reasoning |
|---|---|---|
| SMPP vs SS7 (SMS) | **Tested, evidence says NO split** | Per-segment PR-AUC/log-loss evaluated separately, neither model showed a reason to split. `source` stays a feature. |
| Fraud-type / sub-cluster (SMS spam) | **Untested** | DBSCAN clustering on the existing feature space (not yet run) is the prerequisite evidence-gathering step - clusters reveal whether spam sub-types are actually separable in feature space before any split decision is made |
| Wangiri vs Flash Call (Voice/IGW) | **Untested** | A plausible-sounding argument for splitting exists (different definitions of "anomalous," independent rule-update lifecycles) but has not been empirically tested via per-fraud-type evaluation. Its own "when one model would make sense" section describes exactly the DBSCAN-cluster-then-label workflow already planned for SMS - the fix for its stated "no labels" problem is getting labels via clustering, not defaulting to separate models. |
| Multi-operator customization | **Not yet applicable (single operator currently)** | When relevant: content-layer components (embeddings, n-gram vocab, curated keyword lexicon, prototype-similarity sets) are inherently operator-transferable and should be built once, shared. Behavioral/velocity features are inherently operator-specific by construction (computed from that operator's own Feast store). The classifier itself: test `operator_id` as an added FEATURE on pooled data first (cheapest test, matches the general principle) before building separate base+fine-tune model artifacts per operator. |

### When splitting genuinely is justified (real reasons, not assumptions)

- Per-segment evaluation shows a persistent, feature-engineering-resistant
  performance gap
- Different downstream business action per segment requiring different
  decision thresholds (doesn't necessarily require separate MODELS,
  though - can be handled at the decision/rule-engine layer on one
  shared score)
- Independent retrain/redeploy lifecycle needs (real, but addressable via
  MLflow registry versioning of one shared model in many cases, before
  reaching for fully separate model artifacts)
- Structurally different raw schemas requiring different FEATURE
  EXTRACTION (already true for SMPP vs SS7 - but note this project's own
  resolution was source-specific extraction feeding ONE shared model, not
  source-specific models)

---

## 5. Real-Time Inference Design

### Point-in-time feature discipline

Training data (especially from IGW/complete-CDR sources) can represent a
FINISHED transaction. Inference happens BEFORE that transaction is
complete. Any feature that only exists once a transaction finishes
(final duration, final delivery status, complete campaign size) CANNOT be
used to train the real-time scoring model - it will never be available at
the actual moment of inference, and training on it silently teaches the
model to rely on information it will never have in production
(train/serve skew via leakage, not just a data-quality nuisance).

**Resolution pattern**: rich/complete data still has value, just not as a
direct model input for the real-time model:
1. Use it to build **precomputed baseline features** (offline batch job)
   that ARE available at inference as a fast lookup (e.g., "this
   subscriber's historical average duration" computed offline, looked up
   online - not "this call's actual duration," which doesn't exist yet)
2. Use it to train a separate, deliberately offline/batch model (the
   CDR-Analyzer-style pattern) whose OUTPUT (a confirmed label, a
   correlation/agreement signal) feeds back into the real-time model's
   retraining data - not its live feature vector
3. Techniques like knowledge distillation do NOT solve missing-feature
   problems - distillation is model compression at a FIXED feature set;
   it cannot hand a smaller/faster model information that isn't in its
   own inputs. Learning Using Privileged Information (LUPI) is the
   correctly-matched technique for train-rich/serve-thin scenarios, but
   even LUPI only provides modest gains, not full recovery of missing
   information - this is an information-theoretic limit, not something
   clever training bypasses.

### Two-model correlation as a legitimate cross-check (Voice-side pattern)

Rather than trying to inject IGW-only features into the real-time SIP/SS7
model, keep them as two independently-trained models with different data
scope and correlate their outputs after the fact:
- Model A: offline, post-call, full IGW+Voice CDR data
- Model B: real-time, inline, only inference-available fields
- Track `agreement_rate` between them as a production health metric
- **Disagreement cases become genuine (non-circular) retraining labels**
  for Model B - Model A's richer verdict surfaces patterns Model B's
  reduced feature set couldn't see live. This is meaningfully different
  from rule-derived labels, which just teach a model to replicate
  existing rules.

### Cold-start is structural, not a tuning problem

Applies to flash-call detection, new spam campaigns, new SIM-box sources
alike: the FIRST few events from a new source cannot be caught by
behavioral/velocity features, because there's no history yet to compute
them from. Two speeds of resolution:
- **Fast path**: once enough events accumulate for a source, the
  EXISTING model (no retrain) starts catching its subsequent events, as
  precomputed features in the feature store catch up
- **Slow path**: genuinely novel patterns get caught via offline
  clustering/CDR-analysis, confirmed, and only then trigger an actual
  model retrain

This should be stated as a known, disclosed limitation - not something to
promise away or quietly paper over.

### API interaction shape: synchronous REST, not webhook

The Rule Engine BLOCKS waiting for a score (confirmed requirement) - this
is exactly what a synchronous request/response HTTP call is for. A
webhook is for the opposite pattern (caller doesn't wait, gets called
back later once a long-running job completes) and would require building
a correlation-ID + callback-endpoint + timeout/retry system to simulate
what a plain synchronous call already provides for free. Given measured
component latencies (embedding ~14ms, Feast lookup fast, FAISS
sub-millisecond at prototype scale), the whole chain fits comfortably in
a normal synchronous request - nowhere near "long-running job" territory.

**Where async/webhook-style DOES fit**: the slow-path retraining feedback
loop (labels flowing back, MLflow retrain, promotion) - genuinely
long-running and disconnected from any single live request. Already
matches the existing batch-refresh pattern (`scripts/refresh_feast.py`) -
don't let this legitimately-async piece pull the live scoring API toward
an async pattern it doesn't need.

**Speculative parallel execution (Rule Engine + ML API racing, cancel on
early rule-engine resolution)**: a real technique, but weigh carefully
before building it:
- Real cost: every whitelist/blacklist-resolved message would still pay
  full ML inference compute (embedding, Feast lookup, both models) even
  though its result gets discarded - undermines the entire two-stage
  design's point (per `docs/ml/modeling.md`: ML should only ever run on
  traffic the rule engine couldn't resolve)
- Real complexity: race conditions on near-simultaneous resolution,
  partial side-effect cleanup (e.g., the FAISS near-dup index update
  needs to NOT fire for a cancelled/discarded speculative call, or the
  index gets polluted)
- Measure first: compare actual rule-engine-only latency vs. actual
  ML-inference latency, and check how often rule-engine resolves without
  needing ML at all (that resolution rate = your wasted-compute rate
  under speculative execution) before deciding this is worth the
  complexity
- **Lower-risk alternative with real latency benefit and no
  correctness risk**: parallelize independent steps WITHIN the ML
  service itself (Feast lookup and embedding computation don't depend on
  each other - both are needed once you're in the ML path at all, so
  running them concurrently via `asyncio.gather` has no downside)

---

## 6. Explainability

- **Real-time/inline path**: cheap, fast methods only - native LightGBM
  feature contributions (`pred_contrib`), or lightweight LIME. These
  become the `reason_codes` returned alongside the score.
- **Offline/analyst tooling**: full SHAP or deeper LIME analysis - not on
  the latency-critical path, used for post-hoc case review and building
  confidence in the model's real-world behavior.
- Curated keyword-lexicon features (urgency/authority/reward/threat
  scores) double as free, directly human-readable reason codes - often
  more useful to an analyst than a raw SHAP value on an embedding
  dimension, since the lexicon score has an obvious name and meaning.

## 7. NLP Feature Engineering (content-side, beyond raw embeddings)

Raw dense embedding dimensions are individually uninterpretable and don't
suit tree-based models (LightGBM) well - meaning lives in the
relationship BETWEEN dimensions (what cosine similarity captures), not in
any single dimension crossing a threshold, which is how a tree splits.
This is consistent with an empirical finding in this project: LightGBM
reached for `text_length` (a tree-friendly signal) in the absence of
better content features, not because content doesn't matter.

**Better-fitting content features for a tree-based classifier**:
- TF-IDF n-grams (uni/bi/trigrams) - sparse, each feature independently
  thresholdable, captures exact phrasing/templated patterns embeddings
  can smooth over
- Curated keyword lexicon scores (urgency/authority/reward/threat) -
  interpretable, double as reason codes; build the term list EMPIRICALLY
  from differential word-frequency between confirmed-spam and
  confirmed-legit messages, not from intuition alone
- Structural URL features (having-IP, prefix-suffix, shortener,
  subdomain) - zero external calls needed, cheap
- Derived embedding SUMMARIES rather than raw dimensions, if embedding
  signal is wanted at all: `similarity_to_nearest_known_spam_prototype`,
  or `near_dup_match_count` (already tree-friendly, arguably already
  embedding signal reaching LightGBM indirectly via FAISS)

Raw embeddings remain the right fit for FAISS (pure distance/similarity
search - exactly what they're built for) and reasonably suited to
Isolation Forest (doesn't need individual dimensions to be meaningful,
just needs rare combinations to isolate faster than common ones).

## 8. Cold-Start Bootstrap: Prototype/Zero-Shot Similarity

A legitimate production technique for the gap between "pure unsupervised,
zero domain knowledge" and "enough real labels for supervised training":
hand-pick a small set of known-bad example messages, embed them, score
new messages by cosine similarity to those prototypes. Requires far less
human effort than a training set, and can inject targeted domain
knowledge directly.

**Known limitation, not a minor caveat**: brittle to paraphrase evasion -
an adversary only needs to reword until similarity drops below threshold.
A model fine-tuned on real accumulated paraphrase variation generalizes
past this in a way a static prototype set cannot. Treat this as a
COLD-START BOOTSTRAP stage, explicitly positioned before rule-derived/
human-reviewed label accumulation, not as the end-state detection method.

---

## Cross-references

- `docs/architecture.md` - current pipeline/system design, file layout,
  environment notes
- `docs/ml/modeling.md` - the two-score design, evaluation conventions,
  MLflow conventions, current status
- `docs/experiments/faiss.md`, `anomaly.md`, `rule_pattern.md` - per-model
  tuning evidence
- `mlflow_demo/` (earlier fraud-detection project) - origin of the
  `train.py`/`compare_versions.py`/`predict.py`/`plot_probabilities.py`
  conventions reused throughout this project