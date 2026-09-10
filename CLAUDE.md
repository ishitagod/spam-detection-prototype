# CLAUDE.md

## Response style

- Be concise and direct.
- For technical tasks, prioritize actionable commands/code.
- Don't repeat information already given.

## Project

Prototype SMS fraud detector using real SMPP and SS7 CDRs.

Pipeline:

SMPP/SS7 → ingestion → canonical schema → reassembly → behavioral features
→ text embeddings → FAISS near-dup → ML models → FastAPI

Goal: production-grade working demo first, then iterate.

## Architecture rules

- SMPP and SS7 must map to the same canonical schema before shared ML.
- `source` is a shared feature.
- SS7-specific fields may remain SS7-only when they carry real signal.
- Split by source (SMPP vs SS7) is in progress, not just theoretical: SMPP and SS7 inference
  is expected to diverge, so both `models/anomaly/train.py` and `models/rule_pattern/train.py`
  support `--sources SMPP` / `--sources SS7` to train independent models, and
  `scripts/run_full_pipeline.ps1 -SplitBySource` runs both source-specific pipelines end to end.
  Split models get their own MLflow experiment/registered names, suffixed by source (e.g.
  `isolation_forest_SMPP` / `anomaly_score_model_SMPP`) — never mixed with the combined-sources
  experiment, so champion/challenger comparisons stay apples-to-apples.
  `scripts/check_source_split_justified.py` reports both the shared model's per-source PR-AUC gap
  and, once split runs exist, whether each source-specific model actually beats the shared model's
  slice of it.
- Features must be point-in-time valid.
- Rule-resolved traffic (`rule_evaluated == True`) is not scored at real-time inference.
- Anomaly detection still trains on the full traffic stream.
- Rule-pattern model trains only on `rule_evaluated == True`.
- Keep `rule_pattern_score` and `anomaly_score` separate. Do not average them.
- Disagreements between the two scores are valuable and should remain visible.

## ML

### Rule-pattern model
- LightGBM.
- Models known rule-engine patterns, not novel spam.
- Features: canonical + behavioral + source.
- No embeddings.
- Primary metrics: PR-AUC and log loss.

### Anomaly model
- Isolation Forest.
- Unsupervised; no labels during training.
- Features: MiniLM embedding + behavioral + FAISS features.
- MiniLM: `paraphrase-multilingual-MiniLM-L12-v2`.
- PCA embeddings to 30 dimensions.
- Validation against rule labels is allowed but labels must not influence training.

### FAISS
- Near-duplicate detection using 1h and 24h windows.
- Features: match count, distinct senders, max similarity.
- FlatIP for now.
- SIM-farming is a separate SS7 behavioral feature, not a FAISS feature.

## Environment

- Requires `dill==0.4.1` in requirements.txt, not Feast's own `dill~=0.3.0`
  — this environment's Python version breaks Feast's on-demand feature
  view serialization without the pin. If a fresh install/upgrade breaks
  Feast with a pickling error, check this pin first.

## Feature store

- Feast with local SQLite registry/online store.
- Behavioral sender features are batch-refreshed.
- No Kafka or Redis for this prototype.
- Online freshness depends on refresh cadence.
- See `feature_repo/` and `features/behavioral_snapshot.py`.

## Stack

- Python
- FastAPI
- Feast
- LightGBM
- scikit-learn
- sentence-transformers / MiniLM
- FAISS
- MLflow
- LIME + SHAP
- pandas / numpy

## Project structure

- `common/` — canonical schema contract (`schemas.py`), shared by both sources
- `ingestion/` — SMPP/SS7 ingestion and canonical mapping
- `features/` — feature computation
- `feature_repo/` — Feast definitions
- `models/` — model training
- `serving/` — inference/feature lookup
- `scripts/` — operational scripts
- `pipeline.py` — top-level orchestrator (ingestion → reassembly → behavioral → embeddings → FAISS)
- `tests/` — tests
- `docs/` — detailed architecture, experiments and feature documentation

## Conventions

- Training scripts use CLI/argparse for tunable parameters.
- Use MLflow for experiment tracking.
- Champion/challenger promotion is explicit in code.
- Follow existing `train.py`, `predict.py`, and `compare_versions.py` patterns.
- Add/update tests when changing behavior.
- Do not introduce dependencies or architectural components without a reason.

## Current status

Completed:
- Real SMPP + SS7 ingestion
- Canonical schema
- Message reassembly
- Behavioral features
- Feast integration
- FAISS features
- Isolation Forest training
- LightGBM training
- FastAPI service combining both scores (`serving/app.py`)
- SHAP explainability for both models, LIME for the anomaly model
  (`models/rule_pattern/explain.py`, `models/anomaly/explain.py`) - offline,
  run-id-driven scripts against an already-trained MLflow run
- Real-time SHAP wired into `/v1/score` (`serving/scoring.py::
  explain_rule_pattern`, a cached `shap.TreeExplainer` run inline per
  request) - `_reason_codes()` now derives its codes from real per-request
  contributions instead of fixed thresholds, and `FraudPredictionResult.
  feature_contributions` surfaces the raw top-K SHAP values. FRAUD
  predictions only (cost control); LIME is deliberately NOT wired in here
  (too expensive per-request, stays offline-only - see
  `explain_rule_pattern()`'s docstring)
- Full (non-sampled) `text_embeddings.py` run - DONE. Both FAISS and
  Isolation Forest now train on the full corpus (SMPP 5.5M / SS7 2.7M
  rows), not the old `--sample_n` subset - see
  `docs/experiments/anomaly.md`'s "Current scale". `--sample_n` still
  exists as an opt-in flag for fast local iteration, it's just no longer
  the default path any model trains on.
- DBSCAN cluster-discovery workflow (`models/anomaly/cluster_discovery.py`
  -> `inspect_clusters.py` -> `ingest_cluster_labels.py`) - turns
  top-anomaly `anomaly_score` output into hand-labelable fraud-type
  clusters. See `docs/experiments/anomaly_clustering.md` for the full
  step-by-step.

Next:
1. Hand-confirm DBSCAN clusters (`docs/experiments/anomaly_clustering.md`
   step 4: `models/anomaly/inspect_clusters.py` +
   `models/anomaly/suggest_cluster_labels.py`, then
   `models/anomaly/ingest_cluster_labels.py`), then train
   `models/fraud_type_classifier/train.py --label_source confirmed` - the
   real multiclass fraud-type candidate. `--label_source suggested`
   (unconfirmed heuristic guesses) already runs today but logs to a
   separate `_suggested_labels` MLflow experiment and must never be
   promoted.
2. `models/rule_pattern/train.py --with_embeddings` - now unblocked by
   the full embeddings run above (100% `rule_evaluated` coverage both
   sources), not yet run. Baseline LightGBM's own SHAP importances show
   `text_length` as its top feature, suggesting real content embeddings
   would help - see `docs/experiments/rule_pattern.md`.

## Documentation

For detailed project knowledge, consult these docs when relevant:

- `docs/architecture.md` — architecture and design decisions
- `docs/feature_catalog.md` — feature definitions and contracts
- `docs/experiments/` — experiment results and model evaluations
- `docs/ml/modeling.md` — modeling decisions and methodology

Read the relevant doc before modifying that area. Don't load unrelated docs.