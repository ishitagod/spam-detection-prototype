# SMS Spam Detection — Prototype/Experiment Plan

Scope: a working experiment to showcase, not production-grade. Full
production plan (dual-protocol contract, Kafka, 3000 TPS scaling,
governance) is tracked separately — this document is deliberately smaller.

## Model choices for THIS prototype (not the eventual production picks)

- **Text embedding: Multilingual MiniLM** — smallest footprint, lowest
  latency of the three candidates (Distil-mBERT, MiniLM, XLM-R). Distil-
  mBERT is the reasonable Phase-1 production candidate; XLM-R needs more
  infra than a prototype needs. Pick MiniLM to keep the loop fast.
- **Classifier: LightGBM only** — already proven working end-to-end in the
  fraud-detection build this plan reuses tooling from. XGBoost/CatBoost are
  champion/challenger candidates to benchmark later, not now.
- **Unsupervised layer: Isolation Forest** — real-time-viable, no labels
  needed, matches the "catch what the rule engine can't" requirement.

## Components

### 1. Data ingestion
- SMPP: 23 CSVs, ~2 days to ingest/normalize into the canonical schema.
- SS7: 23 CSVs, ~2 days to ingest/normalize into the canonical schema.
- Both map into the same point-in-time feature contract (`source`,
  `event_type`, `originator`, `originator_type`, `text`, behavioral fields)
  agreed on in the architecture plan — no protocol-specific downstream
  logic past this point.

### 2. Feast feature store
- Real Feast (already stood up and verified working), local SQLite online
  store — no Redis/Kafka needed at prototype scale.
- Behavioral features (velocity, repeat-content ratio, sender age) computed
  in batch from the ingested CSVs, materialized into the online store.
- Same lookup path the real inference API will use later — this is not a
  mocked stand-in, it's the real mechanism at smaller scale.

### 3. ML models — two separate outputs, not one blended score
Per the labelling-source discussion: rule-labelled data can only teach a
model to replicate known patterns; it can't teach it to catch what the
rules miss. Keep both paths explicit and separately evaluated:

- **Supervised (LightGBM)** — trained on rule-engine-labelled data.
  Reported honestly as a "known-pattern detector" (faster/cheaper
  re-implementation of existing rules), not claimed as novel-fraud
  detection.
- **Unsupervised (Isolation Forest + near-duplicate/FAISS matching)** —
  trained with no labels at all, on [MiniLM embedding + behavioral
  features]. This is the layer actually positioned to catch spam the rules
  don't already know about.

### 4. Score outputs
Return both scores from inference, not a single collapsed number:
```json
{
  "rule_pattern_score": 0.91,
  "anomaly_score": 0.34,
  "agreement": "match | disagree",
  "confidence": 0.8,
  "reason_codes": ["near_duplicate_burst", "matches_known_pattern"]
}
```
Cases where `rule_pattern_score` is low but `anomaly_score` is high are the
most valuable output of the whole prototype — they're candidate genuinely-
novel spam patterns, worth surfacing distinctly rather than averaging away.

### 5. MLflow
- Track both models' runs (params, metrics, PR-AUC, log loss).
- Champion/challenger comparison reused directly from the existing
  `train.py` / `compare_versions.py` pattern — same tooling, new features.
- Registry entries tagged clearly by type (`supervised_rule_pattern` vs.
  `unsupervised_anomaly`) so the two are never accidentally compared as if
  they were interchangeable.

### 6. Explainability — LIME for both
- Supervised model: LIME (or native LightGBM feature contributions) on
  individual predictions — already built and tested in the fraud project,
  reused directly.
- Unsupervised model: LIME can still explain Isolation Forest's anomaly
  score the same way (treat the anomaly score as the "prediction" being
  explained) — same tool, same technique, applied to a different model
  type.

### 7. Offline vs. online ML platform — SAME environment for this prototype
In production these would genuinely split (heavy offline
training/clustering platform vs. a lightweight, latency-tuned online
inference service — mirrors the ArmourX reference architecture). For this
prototype, splitting them adds infrastructure overhead with no experiment
value — the thing being tested is whether the modeling approach works, not
whether the deployment topology scales. Use **one environment**: train/
cluster via scripts, serve via one local FastAPI process loading the same
model artifacts. Flag this explicitly as a prototype simplification, not
the target production design, so it isn't mistaken for the final
architecture later.

## What's already built and verified working
- Dual-source (SMPP/SS7) dataset
- Real Feast feature repository — entity, feature view, online lookups
  confirmed working end-to-end
- Real sentence-transformers/FAISS/LightGBM/MLflow libraries installed and
  importable

## What's still needed
- MiniLM embedding integration (code path ready; needs to run somewhere
  with access to huggingface.co to download weights — blocked in this
  sandbox specifically, not a design issue)
- FAISS near-duplicate index
- Isolation Forest training script (unsupervised layer)
- LightGBM training script on rule-labelled data (supervised layer)
- FastAPI inference service combining both, returning the dual-score
  response shape above
- LIME wiring for both model types