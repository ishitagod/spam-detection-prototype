# SMS Spam Detection — Technical Architecture & Execution Plan

**Author framing:** ML/Data Science architecture plan for a production-grade,
real-time SMS spam detection system, starting from unlabelled data.

---

## 1. System Requirements (define before building anything)

Before any modeling work, pin these down — vague requirements were the
biggest recurring gap in the reference fraud-detection proposal reviewed
earlier in this project:

| Requirement | Question to answer | Why it matters |
|---|---|---|
| Latency budget | Max ms per message, at p95/p99 under peak load | Determines whether embedding inference can run inline or needs a lighter fallback |
| Throughput | Target: 3000 TPS at scale. **Prototype phase explicitly does NOT need to hit this** — build correct first, optimize (batching, async I/O, index sharding) once the pipeline is proven. Document this as a deliberate two-stage target, not a missed requirement. | Prevents premature optimization; keeps prototype timeline realistic |
| Deployment target | On-prem/CPU-only, or cloud/GPU available | Determines model size choices (MiniLM vs. larger transformers) |
| Languages in scope | Which languages/scripts the SMS traffic actually uses | Determines embedding model coverage requirements |
| Decision authority | Does ML decide directly, or does a rule engine consume an advisory score (recommended) | Determines API contract and governance model |

**Recommendation:** ML never decides directly — it returns probability +
confidence + reason codes; a rule engine (deterministic, auditable) makes
the final allow/flag/block call. Same separation used in the reference
Voice/SMS firewall architecture.

---

## 2. Dual-Protocol Data (SMPP for A2P, SS7 for P2P)

**Correction to an earlier draft of this plan:** SS7 and SMPP are genuinely
different traffic/data types, not just two protocols carrying identical
content. The right design is NOT two parallel model pipelines from day one
— it's one shared ML contract, source-specific feature extraction upstream
of it, and a single model to start, split only if evidence requires it.

### 2.1 Existing ingestion paths (already built, ML sits downstream)

```
SS7:   C program -> Kafka -> Rule Engine -> ML API
SMPP:  HA/Application layer -> Rule Engine -> ML API
```

The Rule Engine already knows the source and event type at the point it
calls the ML API — **the ML layer does not need to detect or infer
SS7-vs-SMPP itself**. That information is passed explicitly as part of the
request. This removes an entire class of ambiguity a naive design would
otherwise have to solve downstream.

### 2.2 Common ML API contract

```json
{
  "source": "SS7 | SMPP",
  "event_type": "MO | SRI | MT | Submit_SM",
  "schema_version": "1.0",
  "features": {
    "...": "point-in-time feature vector - see 2.4"
  }
}
```

`source` and `event_type` are passed through, not inferred. `schema_version`
exists so feature-set changes over time don't silently break older logged
predictions in the audit repository — new fields get added under a new
version rather than mutating the meaning of an existing one.

### 2.3 Source-specific preprocessing, shared model (to start)

```
SS7 raw fields   --> SS7-specific feature extraction   --+
                                                            +--> common feature contract --> ONE model
SMPP raw fields  --> SMPP-specific feature extraction  --+
```

Feature *extraction* is allowed to differ by source (SS7 and SMPP simply
don't expose the same raw fields — sender-ID reputation only makes sense
for SMPP; per-subscriber MSISDN baselining only makes sense for SS7 P2P).
What must stay shared is the **contract those extractors output into** —
same feature names, same types, same schema version — so one model can
consume either source without caring which one it came from.

### 2.4 Point-in-time features only — avoiding train/serve leakage

CDR training data can represent the **complete, finished transaction**.
Inference happens **at a specific point in time**, before the transaction
is complete (same issue discussed earlier for call duration - a field that
only exists once a call has ended can't be used to score that call before
it connects). This applies identically to SS7/SMPP CDR fields:

- Do **not** blindly use every field present in a completed CDR for
  training. Audit each feature: "would this value have existed and been
  knowable at the exact moment the ML API was actually called?"
- Fields that only resolve after the transaction (final delivery status,
  total campaign size once complete, response codes) belong to
  **offline/CDR-based analysis only** (the slow retraining path in Section
  6), never to the point-in-time inference feature set.
- This is the single most common source of an inflated training-time
  metric that quietly fails in production — worth a deliberate schema
  review before Stage C training begins, not caught by accident later.

### 2.5 Model strategy: one model first, split only with evidence

1. **Start with a single common model**, trained on the shared feature
   contract, `source` included as a feature (not a routing key) so the
   model itself can learn source-dependent patterns if they exist.
2. **Evaluate three ways, not one:** overall performance, SS7-only slice,
   SMPP-only slice (same principle as evaluating onnet vs. offnet
   separately in the Voice project - an aggregate metric can hide a
   segment that's performing badly).
3. **Only split into separate per-source models if the evidence shows
   one segment underperforming** in a way the shared model can't resolve
   (e.g., adding more `source`-interaction features doesn't close the gap).
   Don't pre-emptively build two models/pipelines on the assumption they'll
   be needed — that's added complexity paid for before it's proven
   necessary.

**Core principle, stated plainly:** different raw data → source-specific
preprocessing → consistent point-in-time feature contract → start with one
model → evaluate per traffic type → split models only if evidence requires
it.

---

## 3. High-Level Architecture

```
                    ┌─────────────────────────────┐
                    │   Streaming ingestion layer   │
                    │ (every SMS event, async)      │
                    └───────────────┬───────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                                       ▼
  ┌─────────────────────────┐          ┌─────────────────────────────┐
  │ Feature Store (Feast)    │          │ Near-duplicate index          │
  │ - sender velocity        │          │ (FAISS / MinHash-LSH)         │
  │ - unique recipients       │          │ - rolling window of recent    │
  │ - repeat-content ratio   │          │   message embeddings          │
  │ - sender history/age     │          │ - updated incrementally       │
  └───────────┬───────────────┘          └───────────────┬───────────────┘
              │                                           │
              │            ┌──────────────────────────────┘
              │            │
              ▼            ▼
      ┌─────────────────────────────┐
      │  SMS Inference Service        │   <- single message in, single
      │  (per-message, real-time)     │      verdict out - see Section 4
      │  1. embed message text        │
      │  2. lookup behavioral features │
      │  3. lookup near-dup matches    │
      │  4. score (Phase-appropriate   │
      │     model - see Section 3)     │
      └───────────────┬───────────────┘
                       │  probability + confidence + reason codes
                       ▼
              ┌─────────────────┐
              │   Rule Engine     │  <- deterministic decision authority
              └────────┬──────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
      Allow      Flag for review     Block
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Prediction & Audit Repository │  <- every decision logged,
         │ feeds labelling + retraining  │     feeds Section 5 loop
         └─────────────────────────────┘
```

---

## 4. Modeling Roadmap (unsupervised → semi-supervised → supervised)

### Stage A — Unsupervised bootstrap (Week 0-4)
No labels required. Two independent signals, combined:

1. **Near-duplicate/burst detection** — embed messages (multilingual
   sentence-transformer, e.g. MiniLM for CPU efficiency), maintain a rolling
   similarity index. A message matching many recent near-duplicates from the
   same/related senders = strong signal, zero labels needed.
2. **Isolation Forest** on [embedding + behavioral features] jointly, for
   anomalies that aren't exact duplicates (e.g., varied but structurally
   similar phishing templates).

**Output at this stage:** an anomaly score + duplicate-match count. Not yet
a calibrated probability — treat as a ranking signal to prioritize review.

### Stage B — Weak/human-in-the-loop labelling (Week 3-8, overlaps Stage A)
- Auto-confirm the clearest cases (extreme duplicate burst + new/unverified
  sender) as weak-positive labels — but tag these explicitly as
  **rule-derived**, and don't let them dominate the eventual training set.
- Route the **ambiguous middle** (Section on confidence below) to human
  reviewers — this is your highest-value labelling source, since it's where
  a model most needs real signal.
- Route a **random sample of "confident allow" traffic** to review too, to
  catch false negatives the system isn't even flagging.

### Stage C — Supervised classifier (once ~hundreds-low thousands of
confirmed labels exist — volume-gated, not calendar-gated)
- **One common model to start**, trained on the shared point-in-time
  feature contract from Section 2.2/2.4, with `source` (SS7/SMPP) included
  as a model feature rather than used to route to different models.
- LightGBM/XGBoost on [text embedding + behavioral + near-dup features].
- Primary metrics: **PR-AUC, log loss** (not accuracy/ROC-AUC alone —
  spam is a minority class; see reference project's metric discussion).
- Track precision at fixed recall (e.g., "at 90% recall, what's precision")
  as the number that actually maps to a business decision.
- **Evaluate three ways: overall, SS7-only slice, SMPP-only slice.** Do not
  report only an aggregate number — a shared model can look fine overall
  while quietly underperforming on one traffic type (same principle as the
  Voice project's onnet/offnet segment evaluation).
- **Split into separate per-source models only if this segment evaluation
  shows a persistent gap** that adding more `source`-interaction features
  can't close. Don't build two models pre-emptively.

### Stage D — Continuous loop
- MLflow experiment tracking + model registry, champion/challenger
  promotion with an explicit, numeric promotion rule defined up front
  (not left vague — this was a known gap in the reference project).
- Scheduled retrain trigger tied to **volume of new confirmed labels**,
  not a fixed calendar cadence.

---

## 5. Real-Time Inference Contract

**Inference is always single-message.** Bulk/burst context is never
recomputed at inference time — it's precomputed asynchronously and looked
up (Feast for behavioral aggregates, similarity index for near-duplicates).
This mirrors exactly how call-velocity features work in the Voice pipeline.

### Request
```json
{
  "message_id": "msg_88213",
  "sender_id": "AX-BANKALERT",
  "text": "Your OTP is 4821. Do not share.",
  "recipient_count": 1,
  "timestamp": "2026-08-19T10:22:31Z"
}
```

### Response
```json
{
  "spam_probability": 0.94,
  "confidence": 0.88,
  "reason_codes": ["near_duplicate_burst", "new_sender_low_history", "url_shortener_present"],
  "model_version": "spam_classifier_v7",
  "features_used": {
    "sender_msgs_last_5min": 340,
    "near_dup_match_count": 312,
    "sender_history_days": 0.4
  }
}
```

**Probability vs. confidence — kept as separate fields, deliberately:**
- `spam_probability`: model's estimate of spam likelihood.
- `confidence`: independent measure of how much to trust that number —
  derived from feature completeness (cold-start senders = lower
  confidence), proximity to a known cluster, and calibration distance from
  0.5. Low confidence should route to human review regardless of the raw
  probability value.

**Cold-start disclosure (structural, not a bug):** a sender's first few
messages in a new campaign may have thin/no behavioral history. The
`confidence` field should explicitly reflect this rather than reporting
false certainty — same limitation as the Voice project's flash-call
detection.

---

## 6. Feedback & Retraining Loop

```
Human review decisions (Stage B/D)
        │
        ├──> Labelled dataset (balanced: flagged + random-sampled allowed)
        │
        ├──> Fast path: sender's future messages benefit immediately once
        │     enough behavioral history accumulates in the feature store —
        │     NO retrain needed for repeat offenders from known senders
        │
        └──> Slow path: genuinely novel spam patterns feed MLflow retrain
              → validation → champion/challenger → promotion
```

---

## 7. Explainability

- **Real-time:** native feature importance / cheap contribution scores
  (fast enough for the inline path) — surfaced as `reason_codes`.
- **Offline/analyst tooling:** LIME or SHAP for deeper per-case review,
  not on the latency-critical path (same split used in the reference
  Voice project).

---

## 8. Tech Stack Summary

| Layer | Choice | Why |
|---|---|---|
| Text embedding | MiniLM or Distil-mBERT (multilingual) | CPU-friendly, broad language coverage |
| Near-duplicate index | FAISS or MinHash/LSH | Fast approximate similarity at scale, incrementally updatable |
| Anomaly detection | Isolation Forest | Real-time-viable, no labels needed |
| Feature store | Feast + Redis | Same tool as reference project; prevents train/serve skew |
| Classifier (Stage C) | LightGBM | Fast, CPU-friendly, strong on tabular+embedding features |
| Experiment tracking | MLflow | Registry, champion/challenger, audit trail |
| Explainability | Native importances (real-time) + LIME/SHAP (offline) | Matches latency constraints |

---

## 9. First 30 Days — Concrete Starting Steps

1. Pin down latency/language requirements in writing (Section 1).
   Throughput target is 3000 TPS **at scale** — prototype does not need to
   hit this; design the canonical schema/feature-store keying to support it
   later, but don't spend prototype time on load optimization.
2. Define the common ML API contract (`source`, `event_type`,
   `schema_version`, `features`) per Section 2.2 — this is the boundary the
   Rule Engine already calls across for both SS7 and SMPP, so get it right
   before building feature extraction on either side.
3. Build source-specific feature extraction for SS7 and SMPP separately
   (their raw fields genuinely differ), both emitting into the **same**
   point-in-time feature contract. Audit every candidate feature against
   Section 2.4 (was this knowable at inference time, not just present in
   the completed CDR) before it's added.
4. Build the near-duplicate index as a standalone service — this alone,
   with zero ML, likely catches a large share of obvious burst spam and
   gives an early, defensible win.
5. Stand up Isolation Forest on embedding + behavioral features in shadow
   mode (scoring live traffic, not yet blocking anything).
6. Build the human review queue + random-sampling-of-allowed-traffic
   mechanism from day one — this is the labelling engine everything else
   depends on; don't bolt it on later.
7. Only after 3-4 weeks of Stage A/B running and generating confirmed
   labels, begin Stage C supervised model development.

---

## 10. Key Risks to Track From Day One

1. Rule-derived labels risk circularity — model re-learns the rules, not
   beyond them, unless independent human-reviewed labels are prioritized.
2. Cold-start gap is structural — first messages of a new campaign are not
   catchable in real time; disclose this rather than promising otherwise.
3. Class imbalance will make accuracy/ROC-AUC misleading — PR-AUC and log
   loss are the primary metrics, from day one.
4. Multilingual coverage must be verified against actual target languages,
   not assumed from a model's general "multilingual" label.
5. Feature store consistency between training and serving is non-negotiable
   — the single most common source of silent production failures in ML
   systems like this.
6. **Point-in-time leakage from CDR fields** — training data drawn from
   completed CDRs can contain fields that weren't actually knowable at the
   moment inference would have happened. Every feature needs to pass "would
   this have existed when the ML API was called" before use, or reported
   training metrics will look better than production performance ever will.
7. **Premature model splitting** — building separate SS7 and SMPP models
   before evidence shows the shared model actually underperforms on one
   segment adds maintenance cost (two models to retrain, validate, promote)
   without a proven accuracy benefit. Default to one model; split only when
   per-segment evaluation demonstrates it's necessary.