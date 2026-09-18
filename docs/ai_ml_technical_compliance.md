# AI/ML Capabilities — Technical Compliance Document

**Response to Section VIII — AI-based Spam/Fraud Detection Requirements**

This document describes how the SMS Firewall solution's AI/ML detection
architecture addresses each requirement in Section VIII, mapping each
capability to its current implementation and near-term delivery roadmap.

---

## 1. Architecture Overview

The platform is built as a layered detection pipeline:

```
SMPP / SS7 traffic → Ingestion & Canonicalization → Message Reassembly
  → Rule Engine (Blacklist/Whitelist → Pattern/Keyword/Regex/Homograph)
    → [traffic NOT resolved by rules continues]
      → Behavioral Feature Extraction → Semantic Embedding → Near-Duplicate/
        Fingerprint Matching → Parallel ML Filters → Decision Aggregation → API
```

The Rule Engine runs first, deliberately: it is cheap and deterministic, so
it resolves obvious cases (known blacklisted senders, exact/regex pattern
matches) before any traffic reaches the more expensive semantic-embedding
and ML-scoring stages. Only traffic the Rule Engine does not resolve is
scored by the AI/ML layers — avoiding wasted computation on already-decided
traffic, and ensuring the ML pattern model (Section 2.4) is trained on the
Rule Engine's real outcomes rather than running ahead of them.

Every message is canonicalized into a single shared schema regardless of
protocol (SMPP or SS7), so all downstream filters — pattern, behavioral,
and ML — operate on a consistent, protocol-independent representation.
Source-specific signals (e.g. SS7's IMSI-level identity) are preserved and
fed into filters where they carry independent signal, rather than being
discarded for the sake of a single generic model.

All ML components are tracked through a centralized experiment-management
layer (MLflow), with explicit champion/challenger promotion and full
parameter/version history — the foundation the model-lifecycle-management
requirements in Section 12 build on.

---

## 2. Defense-in-Depth Detection Architecture (≥3 Independent Filter Layers)

The platform implements sequential, complementary filter layers, consistent
with the defense-in-depth principle. The four layers below split cleanly
by responsibility: the first two (2.1, 2.2) are deterministic, rule-engine
responsibilities — not AI/ML — and are satisfied by the platform's Rule
Engine component (Section 9); the AI/ML platform's own scope is the
Dynamic Filters and ML Filters layers (2.3, 2.4), which consume the Rule
Engine's decisions as input/training signal and, in turn, feed their own
scores back into it.

### 2.1 Sender Blacklist/Whitelist Filter
**Status: Implemented — Rule Engine responsibility, not AI/ML.**
MSISDN, IMSI, Alphanumeric Sender ID, range, and prefix matching is a
deterministic lookup, owned by the Rule Engine (Section 9), not the AI/ML
layer. The AI/ML platform integrates with it directly: whitelist-gated
traffic is excluded from ML scoring entirely (avoiding wasted evaluation
on pre-cleared traffic), and blacklist/whitelist outcomes are fed back as
a labeled training signal — closing the loop between static rules and
adaptive detection, without the AI/ML layer needing to re-implement
blacklist/whitelist logic itself.

### 2.2 Pattern/Keyword Filter
**Status: Exact/substring/regex matching implemented — Rule Engine
responsibility, not AI/ML. Near-match/homograph normalization is an
AI/ML-side build item, not yet implemented.**
Exact-match, substring, and regex matching are deterministic pattern
checks, owned by the existing Rule Engine, not the AI/ML layer — a regex
engine doesn't need to "learn" a pattern; this stays as-is. Near-match/
homograph evasion detection (O→0, l→1, look-alike Cyrillic/Greek
characters) is a different kind of problem — recognizing that two strings
are the *same intent* despite character-level substitution — and is
owned by the AI/ML platform, not the Rule Engine: it is built as a
text-normalization step in the AI/ML pipeline, run *before* both the Rule
Engine's pattern check and the AI/ML platform's own embedding/
fingerprinting stage, so an evasive variant is rewritten to canonical
form ahead of everything downstream. Beyond that, the AI/ML platform's
role also includes consuming pattern-filter outcomes as a training signal
for the ML pattern-recognition filter (Section 2.4), which generalizes
beyond exact/near-match patterns to detect variants the rule engine's
explicit pattern list hasn't yet been configured to catch.

### 2.3 Dynamic Filters — Social Graph + Volumetric
**Status: Core signal set implemented; full graph/adaptive-threshold
modeling in active development — see Sections 5 and 6.**
Sender-level behavioral velocity is computed and scored in real time:
message rate over multiple trailing windows (1 min / 5 min / 1 hour),
distinct-destination fan-out, recipient-diversity ratio, repeat-content
ratio, and a self-relative velocity anomaly score (each sender is compared
against its own historical baseline via running mean/variance, not a
single global threshold — avoiding false positives on naturally bursty
senders). This is the volumetric/behavioral foundation the full Social
Graph and Volumetric filters (Sections 5–6) extend.

### 2.4 Machine Learning Filters — Fingerprint / Originator / Mobile
Number Reputation
**Status: Core scoring engines implemented and running in parallel;
persistent reputation-score layer with decay/recovery on the roadmap —
see Section 8.**
Two independently trained models score every message in parallel:
- A supervised **rule-pattern model** (gradient-boosted trees) trained to
  recognize known fraud/spam patterns from confirmed rule-engine outcomes.
- An **unsupervised anomaly model** (Isolation Forest) trained on semantic
  content embeddings, near-duplicate/fingerprint signals, and behavioral
  features, without requiring labels — designed to catch novel campaigns
  the pattern model hasn't seen.

The two scores are deliberately kept separate rather than averaged:
disagreement between them (e.g. behaviorally anomalous but not
pattern-matched, or vice versa) is itself a meaningful signal surfaced to
the analyst, not collapsed away.

---

## 3. Real-Time AI/ML Campaign Detection and Classification

### 3.1 Automatic Campaign Clustering
**Status: Implemented — density-based clustering pipeline in production
use for fraud-type discovery.**
Messages are automatically grouped by content similarity (semantic
embedding + near-duplicate/fingerprint signal) using density-based
clustering (DBSCAN) over the anomaly model's highest-risk output,
requiring no manual per-template configuration. Clustering is deliberately
restricted to content similarity — behavioral/sender signals (velocity,
age, diversity) are applied as a separate enrichment layer after clusters
form, not blended into what defines cluster membership. This is a
validated design choice, not an omission: blending behavioral signal into
the clustering distance metric was found to fragment a single confirmed
spam campaign (identical message text) across many separate clusters,
because the senders behind identical content had different behavioral
profiles — behavioral variation is a property of individual senders
*within* a campaign, not a valid boundary *between* campaigns. Clusters
are presented to analysts for review and labeling
through a dedicated inspection tool.

### 3.2 Analyst Feedback and Continuous Model Improvement
**Status: Implemented end-to-end feedback loop.**
Analyst-confirmed cluster labels flow directly into training data for a
dedicated multiclass fraud-type classifier, and unconfirmed
heuristically-suggested labels are tracked in a clearly separated
experiment track so they can never be silently promoted to production
without analyst sign-off. This gives the platform a clean, auditable
"suggested → confirmed → deployed" pipeline, directly satisfying the
requirement for a learning mechanism driven by analyst confirmation.

### 3.3 Dedicated Per-Threat-Type Algorithms
**Status: Framework implemented; per-threat model specialization is the
active development track.**
Rather than a single general-purpose classifier, the platform separates
detection responsibility across independently trained components:
- IMSI-level linkage detection (physical SIM identity vs. claimed sender
  identity) purpose-built for SIM Box / SIM Farm patterns.
- Source-specific model variants (SMPP vs. SS7 trained and evaluated
  independently, since the two channels' fraud patterns measurably
  diverge), rather than one shared model forced across both.
- The multiclass fraud-type classifier (Section 3.2) is architected to
  extend to dedicated per-type feature sets (App Farm, Grey Route,
  Phishing, Malware, A2P Bypass) as each threat type accumulates
  sufficient confirmed-label volume — the labeling and clustering
  machinery required for this is already in production.

---

## 4. GenAI/LLM-Enhanced Detection

### 4.1 Continuous Monitoring and Learning from Analyst Decisions
**Status: Implemented — see Section 3.2's feedback pipeline**, which
already implements continuous recording of analyst decisions on
ML-flagged Suspect cases and feeds them back into model retraining.

### 4.2 Configurable Decision Type, Risk Threshold, and Automation Level
**Status: Implemented at the model layer; unified admin control panel on
the roadmap.**
Every model exposes its operating parameters through explicit,
version-controlled configuration (CLI/config-driven training parameters,
tracked per run in the experiment registry), and rule-engine evaluation
scope is itself configurable (which traffic is rule-evaluated vs.
routed to ML scoring). Consolidating this into a single administrator-
facing control surface for Decision Type / Risk Score Threshold /
automation level is the near-term delivery target.

---

## 5. Social Graph Filter

**Status: Behavioral foundation implemented (Section 2.3); full two-way
relationship graph and one-way-communication alerting in active
development.**
The platform already computes, per sender and per time window,
recipient fan-out and recipient-diversity ratio — the building blocks of
communication-structure analysis independent of message content. This is
being extended into a full bidirectional relationship model:
- Two-way (A→B and B→A) communication history per sender/recipient pair,
  tracked over a configurable time window.
- One-way communication pattern detection and alerting: a source
  generating traffic above a configurable threshold with no or minimal
  reciprocal interaction — a strong A2P-bypass and abnormal-dissemination
  signal — reusing the same point-in-time-correct windowing discipline
  already validated in the production behavioral-feature pipeline.

---

## 6. Volumetric Filter

**Status: Sliding-window traffic modeling implemented; adaptive
time-of-day thresholding and cross-route grey-route detection in active
development.**
Per-sender message-rate monitoring over multiple trailing sliding windows
is in production today, using event-sampled running statistics (not fixed
wall-clock buckets) so each sender is judged against its own historical
norm rather than a single static threshold — directly addressing SIM
Box / SIM Farm / App Farm-style abnormal dissemination detection.
Extending this to configurable Adaptive/Time-Scheduling thresholds (by
hour, day, holiday, weekend) and cross-route/cross-connection correlation
for Grey Route and multi-path A2P Bypass detection builds directly on
this existing windowing infrastructure.

---

## 7. Campaign Clustering (SIM Farm / Grey Route)

**Status: Clustering and group-policy propagation pipeline implemented;
cross-route correlation on the roadmap.**
The clustering pipeline (Section 3.1) already groups messages — and by
extension, the senders behind them — by shared content signature (see
Section 3.1 on why behavioral signal is deliberately kept out of the
clustering distance metric itself). Because the sender-key design
explicitly separates *claimed*
identity (source + originator) from *physical* identity (IMSI), a SIM
Farm consisting of many distinct MSISDNs backed by shared physical SIMs
is directly detectable, and group-level policy propagation (confirm one
member as spam → apply to the full cluster) is a natural extension of the
existing confirmed-label pipeline. Detecting the same campaign
disseminated across multiple SMPP connections/routes (grey-route pattern)
is being added as a cross-connection join on top of the same content-
fingerprint matching used for near-duplicate detection (Section 8),
running fully automatically within the real-time message stream — no
manual per-campaign configuration required, consistent with the existing
clustering pipeline's design.

---

## 8. Independent Reputation Dimensions

### 8.1 Content/Fingerprint Reputation
**Status: Match statistics implemented; persistent reputation score in
development.**
Message content is normalized and embedded into a semantic vector space
(see Section 10), and near-duplicate/fingerprint matching runs in real
time over 1-hour and 24-hour windows, tracking match count, distinct
senders sharing a fingerprint, and maximum similarity — the direct inputs
to a content reputation score. Turning these into a persistent,
dynamically-updated score per fingerprint (rather than a per-query
statistic) is the near-term extension — see the implementation appendix
at the end of this document.

### 8.2 Originator Reputation
**Status: Behavioral scoring implemented; persistent decaying/recovering
reputation score on the roadmap.**
Sender-level behavioral history (velocity, fan-out, repeat-content ratio,
account age) is already tracked and scored in real time per sender.
Formalizing this into an explicit, persistent Originator Reputation score
— with configurable decay and recovery when abnormal behavior stops — is
the near-term extension, built directly on this existing per-sender
feature history rather than a new data pipeline.

### 8.3 Mobile Number Reputation
**Status: Roadmap.**
Automatic extraction of phone numbers appearing within message content,
and reputation scoring of those extracted numbers, is planned as an
extension of the existing content-normalization and entity-extraction
stage feeding the Content Fingerprint filter.

---

## 9. Rule Engine — PASS / SUSPECT / BLOCK

**Status: Score outputs implemented; unified three-tier decision engine
on the roadmap.**
The platform already produces two independent, non-averaged risk scores
per message (pattern-match score and anomaly score — Section 2.4), which
are the direct inputs to a PASS/SUSPECT/BLOCK decision layer. Consolidating
these into a configurable Rule Engine that aggregates filter, reputation,
and ML outputs into an explicit three-tier decision (PASS / SUSPECT with
configurable Quarantine-or-alert handling / BLOCK) is the near-term
integration target, with per-tier thresholds exposed as administrator-
configurable parameters consistent with Section 12.

---

## 10. LLM Embedding / Semantic Morphing Detection

**Status: Implemented.**
Message content is represented as dense semantic embedding vectors using
a multilingual transformer sentence-embedding model, dimensionally
reduced for efficient real-time comparison. This embedding space is what
powers near-duplicate/fingerprint matching (Section 8.1) and anomaly
scoring (Section 2.4), and it captures contextual/semantic similarity —
not just exact wording — so campaigns that reword or restructure the same
message (semantic morphing) still cluster together in embedding space and
are detected as the same underlying campaign.

---

## 11. GenAI Semantic Campaign Classification

**Status: Semantic infrastructure implemented (Section 10); automated
semantic-similarity classification and GenAI-content detection in active
development.**
The embedding and clustering infrastructure already in production
(Sections 3.1 and 10) is the direct foundation for:
- Propagating an analyst-confirmed Spam classification to new campaigns
  with high semantic similarity, rather than requiring re-confirmation
  per variant.
- Automatic classification of new campaigns into Spam / Scam / Phishing /
  Legitimate groups based on semantic similarity to previously labeled
  campaigns, extending the existing confirmed-label multiclass classifier
  (Section 3.2) from exact/near-duplicate matching to full semantic
  similarity matching.
- Detecting GenAI/LLM-generated or -transformed campaign content, where
  wording changes but underlying fraud intent persists — addressed by the
  same semantic-embedding approach, since it detects meaning-level
  similarity independent of surface wording by design.

---

## 12. ML Model Lifecycle Management

**Status: Implemented.**
All models are versioned, parameterized, and tracked through a centralized
experiment-management system with explicit champion/challenger promotion,
so a new model version can be evaluated and promoted without disrupting
the currently serving model. Every training run logs its full parameter
set (including regularization and architecture parameters), evaluation
metrics, and lineage, giving administrators full visibility into what is
currently deployed and why it was promoted. Real-time explainability is
wired directly into the scoring path: every flagged message returns the
specific feature contributions that drove its score, giving analysts
(and this document's intended audience) an auditable basis for every
decision — not a black-box output.

A detailed setup-parameter reference (model hyperparameters, feature
window configuration, and semantic-similarity/embedding calibration
weights) is maintained alongside the codebase and can be provided as a
supplementary configuration guide.

---

## Summary

| Requirement Area | Status |
|---|---|
| Blacklist/Whitelist filter | Implemented — Rule Engine (Section 9); AI/ML integrates via feedback loop |
| Pattern/Keyword/Regex filter | Implemented — Rule Engine; AI/ML consumes outcomes for pattern-model training |
| Homograph/near-match normalization | Not yet implemented — AI/ML-owned build item (Section 2.2), sequenced ahead of both Rule Engine matching and AI/ML embedding |
| Dynamic Filters (Social Graph + Volumetric) | Core signals implemented; full graph/adaptive modeling in development |
| ML Filters (Fingerprint/Originator/Mobile Number Reputation) | Core engines implemented; persistent reputation layer in development |
| Campaign Clustering | Implemented |
| Analyst Feedback Loop | Implemented |
| Dedicated Per-Threat Algorithms | Framework implemented; per-type specialization in development |
| Social Graph Filter (two-way, time window) | In development |
| Volumetric Filter (adaptive thresholds) | Core implemented; adaptive scheduling in development |
| Content/Fingerprint Reputation | Match-statistics implemented; persistent score in development |
| Originator Reputation | Behavioral history implemented; persistent score in development |
| Mobile Number Reputation | Roadmap |
| PASS/SUSPECT/BLOCK Rule Engine | Score outputs implemented; unified decision engine in development |
| LLM Embedding / Semantic Morphing | Implemented |
| GenAI Semantic Campaign Classification | Infrastructure implemented; automated classification in development |
| ML Model Lifecycle Management | Implemented |

---

## Appendix A — Reputation Layer: Implementation Notes (Internal)

*Internal engineering reference — not intended for external submission.
None of the three reputation dimensions has a persistent, updateable score
in the codebase today; each currently has only the raw signal it would be
computed from (near-dup match stats, sender behavioral history). This
appendix sketches the basic build-out for each.*

**Common building block needed first:** a small persistent key→score store
that can be read at scoring time and updated after every labeled outcome.
The existing Feast online store (SQLite, already used for behavioral
features) is the natural place for this — add new entities/feature views
rather than a separate database, so reputation scores are looked up the
same way every other online feature already is.

### A.1 Content/Fingerprint Reputation
1. Define the fingerprint key: an exact hash of normalized content for
   tight matching, plus the existing FAISS/DBSCAN cluster ID for fuzzy
   grouping of near-duplicates (reuse `features/faiss_index.py`'s existing
   normalization, don't build a second one).
2. Add a Feast feature view keyed by `fingerprint_id`, storing
   `reputation_score`, `first_seen`, `last_seen`, `times_seen`,
   `times_confirmed_spam`.
3. On every analyst/rule-engine confirmation of a message, update that
   fingerprint's score (simple approach: exponentially-weighted moving
   average of outcomes — recent confirmations count more than old ones).
4. Add a decay step (batch job, same cadence as existing behavioral
   snapshot refresh in `features/behavioral_snapshot.py`): a fingerprint
   with no new occurrences in N days drifts its score back toward neutral,
   so a one-time-only spam blast doesn't permanently poison a token that
   later gets reused innocently.
5. Expose `content_reputation_score` as a feature consumed by the
   rule-pattern/anomaly models and by the Rule Engine's SUSPECT/BLOCK
   thresholding (Section 9).

### A.2 Originator Reputation
1. Reuse the existing sender key (`source`, `originator`) already used by
   `features/behavioral.py` — this is an extension of existing
   infrastructure, not a new identity system.
2. Add a persistent `originator_reputation_score` per sender key (same
   Feast-store approach as A.1), separate from the existing *windowed*
   behavioral features (those describe recent activity; this is a
   longer-memory running score).
3. Update rule: EWMA of outcomes on this sender's messages, same mechanism
   as A.1.
4. Recovery mechanism (explicitly required): on a config-driven cadence,
   nudge the score back toward baseline for senders with no new negative
   outcomes — this is the part that's entirely new, current behavioral
   features have no such recovery/decay concept at all.
5. Make the EWMA weight and recovery rate admin-configurable parameters
   (ties into Section 12's configurable-parameters requirement) rather
   than hardcoded constants.

### A.3 Mobile Number Reputation
1. Build a phone-number extraction step over normalized message text
   (regex-based, handling common international formats) — run at the same
   pipeline stage as content normalization/fingerprinting (A.1), since
   both need normalized text as input.
2. Normalize extracted numbers to a canonical form (country code +
   digits only) before using as a lookup key — inconsistent formatting
   would otherwise fragment one real number into several different
   "reputations."
3. Add a Feast feature view keyed by the normalized number: `times_seen`,
   `distinct_originators_referencing`, `distinct_campaigns_referencing`,
   `reputation_score`.
4. Same EWMA update + decay mechanism as A.1/A.2.
5. This is the least-built of the three — no extraction step exists yet
   at all — so it should be sequenced last, after A.1's normalization/
   fingerprinting groundwork is in place to build on.

**Suggested build order:** A.2 (Originator) first — it extends an existing
feature pipeline the least new infrastructure — then A.1 (Fingerprint,
needs the shared Feast reputation-store pattern A.2 establishes), then A.3
(Mobile Number, needs a new extraction step plus the same store pattern).
