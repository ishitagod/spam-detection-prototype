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

Next:
1. FastAPI service combining both scores
2. LIME integration
3. SHAP integration/evaluation

## Documentation

For detailed project knowledge, consult these docs when relevant:

- `docs/architecture.md` — architecture and design decisions
- `docs/feature_catalog.md` — feature definitions and contracts
- `docs/experiments/` — experiment results and model evaluations
- `docs/ml/modeling.md` — modeling decisions and methodology

Read the relevant doc before modifying that area. Don't load unrelated docs.