# CLAUDE.md

## Response style

- Be concise and direct.
- For technical tasks, prioritize actionable commands/code.
- Don't repeat information already given.

## Project

Production SMS fraud detection system using real SMPP and SS7 CDRs.

Pipeline:

SMPP/SS7 → Kafka ingestion → canonical schema → reassembly → behavioral features
→ text embeddings → FAISS near-dup → ML models → FastAPI

Goal: production-grade real-time fraud detection — streaming ingestion,
low-latency feature serving, and continuous (not scheduled-batch) campaign
discovery, hardened incrementally from the current working baseline.

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
- Real-time campaign discovery is a production target, not yet complete: DBSCAN clustering
  (`models/anomaly/cluster_discovery.py`) currently runs as a scheduled batch job. Production
  iteration moves discovery onto the Kafka-fed stream (see Feature store) so new campaign shapes
  surface continuously instead of on a fixed cadence. Matching a message against an
  *already-fingerprinted* campaign is already real-time per message today — don't conflate that
  with brand-new-campaign discovery, which is the piece still migrating off batch.
- Features must be point-in-time valid.
- Rule-resolved traffic (`rule_evaluated == True`) is not scored at real-time inference.
- Anomaly detection still trains on the full traffic stream.
- Rule-pattern model trains only on `rule_evaluated == True`.
- Keep `rule_pattern_score` and `anomaly_score` separate. Do not average them.
- Disagreements between the two scores are valuable and should remain visible.
- Decision fusion (`models/decision_fusion/`, `serving/fusion_scoring.py`) is a
  small trained LogisticRegression over `[rule_pattern_score, anomaly_score]`,
  logged as its own additive `fusion_score` - not an average. It drives
  `prediction`/`recommended_action` when a fusion champion exists for the
  request's source, falling back to `rule_pattern_score` alone otherwise. Both
  raw scores stay on the response unchanged either way; `ANOMALY_SIGNAL_
  ESCALATION` reason-codes the case where fusion is why a row is FRAUD.

## ML

### Rule-pattern model
- LightGBM.
- Models known rule-engine patterns, not novel spam.
- Features: canonical + behavioral + content-rule flags + source.
- No embeddings by default (`--with_embeddings`/`--with_tfidf` opt in).
- Primary metrics: PR-AUC and log loss.

### Anomaly model
- Isolation Forest.
- Unsupervised; no labels during training.
- Features: MiniLM embedding + behavioral + content-rule flags + FAISS features.
- MiniLM: `paraphrase-multilingual-MiniLM-L12-v2`.
- PCA embeddings to 30 dimensions.
- Validation against rule labels is allowed but labels must not influence training.

### Content-rule flags
- Deterministic regex features (`has_url`, `has_phone_number`,
  `has_gambling_keyword`, etc.) computed from `text` alone -
  `config/settings.py::CONTENT_FLAG_PATTERNS`, `features/content_flags.py`.
- Base features for BOTH models above, same treatment as behavioral - not
  ablation-gated like TF-IDF/embeddings.
- A second, independent label source from the upstream rule engine (which
  has no content/regex matching of its own) - see `labels/rule_labels.py`'s
  `content_flagged`/`is_content_evaluated`, kept in a separate
  `label_source` from the telecom-derived `rule_flagged`, never merged.
- `content_flagged_by_count()` (`labels/rule_labels.py`) is an aggregate
  alternative to `content_flagged()`'s fixed combination list - "at least
  N of the 10 flags fired" instead of one hand-picked combination. Kept as
  a simple utility/reference point, but superseded for actual training use
  by the weighted approach below (a plain count treats every flag as
  equally strong evidence, which is just as arbitrary as a hand-picked
  combination).
- `fit_content_flag_weights()` / `content_flagged_by_weight()`
  (`labels/rule_labels.py`) fit a `LogisticRegression(CONTENT_FLAG_COLS ->
  rule_flagged)` on the REAL labelled pool (must have both classes - SMPP
  alone has zero confirmed-clean `rule_evaluated` rows, so fit on the
  combined SMPP+SS7 pool) - each flag's weight is its own measured
  coefficient, not a guess. `models/rule_pattern/train.py
  --include_content_labels` uses the fitted model to add confident
  POSITIVES ONLY from `rule_evaluated == False` rows into
  `rule_pattern_score`'s training pool, tagged `label_source ==
  "content_static_rules"` and never merged into the telecom-derived rows -
  evaluated as its own breakdown (`test_by_label_source_*` metrics)
  because that slice's label is still derived from `CONTENT_FLAG_COLS`,
  which are also features. Routes to the experimental MLflow experiment,
  same as `--with_embeddings`/`--with_tfidf`. Combinable with both: TF-IDF
  needs no extra work (fit straight from `text`); embeddings are joined
  onto the content-labelled rows too via `models/rule_pattern/data.py::
  join_embeddings()`, since `features/text_embeddings.py` runs over the
  whole file, not just `rule_evaluated` rows - still `--sources SS7` only
  in practice today, since SMPP has no `embeddings.npy` yet.

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

- Feast with a **Postgres** SQL registry and a **Redis** online store
  (`feature_repo/feature_store.yaml`, `docker-compose.yml`) — local SQLite
  registry/online store is retired, not the current setup.
- Behavioral sender features are batch-refreshed today. Production target:
  Kafka-fed streaming refresh, so volumetric/social-graph features and
  campaign discovery (see Architecture rules) move off a fixed cadence —
  not yet wired in.
- Online freshness depends on refresh cadence until streaming lands.
- MLflow's tracking store also runs on the same Postgres instance
  (separate `mlflow` database, see `config/settings.py::
  MLFLOW_TRACKING_URI`, `scripts/postgres_init/`) — not SQLite.
- See `feature_repo/` and `features/behavioral_snapshot.py`.

## Stack

- Python
- FastAPI
- Feast
- Postgres (Feast registry + MLflow tracking store)
- Redis (Feast online store)
- Kafka — production target for streaming ingestion and real-time
  campaign discovery, not yet integrated
- LightGBM
- scikit-learn
- sentence-transformers / MiniLM
- FAISS (exact `IndexFlatIP` for batch/training; IVF-PQ for serving-time
  memory efficiency at multi-million-vector scale — see
  `config/settings.py::FAISS_SERVING_INDEX_TYPE`)
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
- Production infra migration off local SQLite: Feast registry moved to
  **Postgres**, Feast online store moved to **Redis**, MLflow tracking
  store moved onto the same Postgres instance (`docker-compose.yml`,
  `feature_repo/feature_store.yaml`, `config/settings.py::
  MLFLOW_TRACKING_URI`) - DONE, uncommitted.
- Serving-time FAISS index switched from exact `IndexFlatIP` to IVF-PQ
  (`config/settings.py::FAISS_SERVING_INDEX_TYPE` and the
  `FAISS_IVFPQ_*` params) to bound memory at the full-corpus scale
  (SS7: 2.74M vectors) - DONE, uncommitted; `m=64` empirically validated
  against a synthetic near-dup benchmark, not yet re-verified against
  the real corpus.
- Decision fusion (`models/decision_fusion/`, `serving/fusion_scoring.py`) -
  DONE, champions promoted for BOTH sources (`decision_fusion_model_SMPP`/
  `_SS7`, alias `champion`). SMPP: 139,546-row pool, test PR-AUC 0.9954.
  SS7: full 2,654,369-row pool (100% anomaly_score coverage, up from an
  earlier ~19k/2.65M partial sample), test PR-AUC 0.9735.
- Postgres/Redis infra (`docker-compose.yml`) - running locally via Docker
  Desktop, host Postgres port remapped to 5433 (5432 was already taken by
  a native Postgres install on this machine - see `config/settings.py`/
  `feature_repo/feature_store.yaml`). `psycopg2-binary` bumped to 2.9.13
  in requirements.txt (2.9.10 has no Python 3.14 wheel).
- SS7's Isolation Forest retrained on its FULL 2,742,301-row corpus (was
  a 20k-row sample) - champion promoted as `anomaly_SS7`. Needed a real
  fix, not just more RAM: `models/anomaly/data.py`'s `build_preprocessor()`
  now uses a new `ChunkedEmbeddingReducer` (StandardScaler + IncrementalPCA
  via `partial_fit`, batched) for the embedding branch specifically, since
  sklearn's plain `StandardScaler.fit()` upcasts float32 input to float64
  internally (`X - mean`, a numpy promotion rule) - unavoidable ~2x memory
  spike that OOM's a 16GB machine on a (2.74M, 384) array otherwise.
  `embedding_pca_pipeline()` itself is UNCHANGED (still a plain
  `Pipeline(StandardScaler, PCA)`) - rule_pattern's embeddings path and
  live single-row serving don't operate at this scale and don't need
  chunking. Also fixed a wasteful `df.copy()` in `build_combined_frame()`
  that was duplicating the entire (2.74M, 384) embedding block for no
  reason (nothing in that function mutates embeddings).

Next:
0. Kafka-fed streaming ingestion, replacing scheduled-batch behavioral
   feature refresh and DBSCAN cluster discovery with continuous,
   real-time paths (see Architecture rules and Feature store) - the
   last piece of the production infra migration, not started.
1. Full (non-sampled) `text_embeddings.py` run for SMPP - SS7's is done
   (GPU, full corpus, `data/processed/SS7/embeddings.npy`). SS7's full
   embeddings also unblock retraining `rule_pattern_score --with_embeddings
   --sources SS7` (corpus ready, retrain not yet run/logged) - see
   `docs/experiments/rule_pattern.md`.
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