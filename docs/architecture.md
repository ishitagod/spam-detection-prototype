# Architecture

System/pipeline design decisions for the SMS spam-detection prototype -
moved out of `CLAUDE.md` to keep that file a short pointer, not a full
narrative. See `docs/ml/modeling.md` for the ML strategy (two-score
design) and `docs/experiments/` for per-model tuning evidence.

## Pipeline

```
SMPP/SS7 -> ingestion -> canonical schema -> reassembly -> behavioral features
-> text embeddings -> FAISS near-dup -> ML models -> FastAPI
```

Goal: production-grade working demo first, then iterate.

## Key architectural decisions (don't relitigate without reason)

- **One shared feature contract for both sources.** SMPP and SS7 expose
  different raw fields, but both get mapped into the same canonical schema
  (`source`, `originator`, `text` + behavioral features) before touching
  any model. `event_type` and `originator_type` were in an earlier draft
  of this schema and have been dropped: on the SMPP side both were
  constant - SMPP is filtered to op-4/submit_sm only, and originator_type
  was a hardcoded "business_sender_id" - so neither carried any signal
  beyond what `source` already encodes. This was a deliberate
  simplification, not a data-driven finding - and SS7 (`ingestion/ss7.py`)
  confirms the prediction: SS7's `message_type` DOES vary and DOES carry
  real signal. It drives SS7's entire row-filtering rule (MO/MT_request
  kept - the only types with real content; SRI_request/MT_response
  dropped; MT_SRI_response merged into MT_request via `virtual_imsi` for
  its `vlr_address`) and is kept as its own SS7-only feature column for
  exactly that reason. It is NOT folded into a generic cross-source
  `event_type` field - SMPP still has no equivalent signal there, so the
  shared-contract decision above stands. Revisit only if per-segment
  evaluation shows this needs to become a first-class shared concept.
- **Point-in-time features only.** Never use a feature in training that
  wouldn't have been knowable at actual inference time (e.g., final
  campaign size once complete). This caused real bugs in the earlier
  fraud-detection project this one reuses tooling from - don't repeat it.
- **Offline and online ML platform are the SAME environment for this
  prototype** (one FastAPI service, no separate training/serving infra
  split). This is a deliberate simplification, not the target production
  design - don't build it assuming it's final.
- **No Kafka, no Redis.** Feast uses local SQLite online store. Batch/
  synchronous feature computation, not streaming, at this stage.
- **Feast is wired in for the behavioral features, batch-refreshed, not
  event-driven.** `feature_repo/definitions.py` defines one entity
  (`sender_id` = `source|originator`), one FeatureView
  (`sender_behavioral_stats`: the three pure sender-state counts) and one
  on-demand feature view (`sender_repeat_content_ratio`: the one
  behavioral feature that needs the INCOMING message's text, which
  doesn't exist until request time - see that file's docstring).
  `features/behavioral_snapshot.py` builds the per-sender snapshot Feast
  materializes from (a genuinely different, simpler computation than
  `features/behavioral.py`'s per-row training features - see that
  module's docstring for why). `scripts/refresh_feast.py` runs the whole
  snapshot -> apply -> materialize sequence; `serving/feature_lookup.py`
  is the manual single-lookup test script (what a future FastAPI endpoint
  will call per request). Online freshness is bounded by refresh cadence,
  not by any TTL Feast enforces at read time - see
  `serving/feature_lookup.py`'s staleness caveat.

## Tech stack

- Feast (feature store, local SQLite registry + online store) -
  `feature_repo/` + `features/behavioral_snapshot.py` + `scripts/
  refresh_feast.py` - wired in for the sender-behavioral features (see
  above); requires `dill==0.4.1` pinned in requirements.txt (see "Known
  blockers" below), not Feast's own `dill~=0.3.0`
- sentence-transformers, model = `all-MiniLM-L6-v2` (prototype choice -
  smallest footprint of the candidates; Distil-mBERT/XLM-R are production
  options, not used here)
- FAISS (near-duplicate detection) - see `docs/experiments/faiss.md`
- scikit-learn Isolation Forest (unsupervised anomaly scoring) - see
  `docs/experiments/anomaly.md`
- LightGBM (supervised classifier) - see `docs/experiments/rule_pattern.md`
- MLflow (experiment tracking + model registry, SQLite backend:
  `sqlite:///mlflow.db`)
- LIME (explainability, both model types) - not yet wired
- SHAP (explainability, added alongside LIME - not yet decided which is
  primary vs. supplementary; revisit once both are wired in)
- FastAPI (inference service) - not yet built

## Known blockers / environment notes

- ~~`huggingface.co` is not reachable from this sandbox~~ - **resolved
  as of 2026-08-24**: verified reachable, `sentence-transformers` +
  `all-MiniLM-L6-v2` installed and downloaded successfully in this same
  sandbox, real encode call confirmed working (`(N, 384)` float32 output).
  Environments change over time - if this blocker resurfaces, re-check
  reachability before assuming the old note still holds; don't take a
  stale blocker note as still-true without verifying live.
- Disk space has been tight in this sandbox before (`pip install
  --no-cache-dir`, avoid re-triggering large caches).
- This environment runs Python 3.14, newer than most libraries have been
  tested against. Feast's on-demand feature view UDF serialization (dill)
  broke on it - see the `dill==0.4.1` pin above. If another dependency
  fails in a way that looks like a pickling/serialization internals
  mismatch rather than a logic bug, check for the same cause before
  assuming it's this project's code.
- Text embedding is the one genuinely expensive step in the whole
  pipeline: ~14ms per DISTINCT text on CPU (measured, not estimated) -
  the full ~8.2M-row dataset would be a ~21hr job. Prototype-scale
  default is to sample (`features/text_embeddings.py --sample_n`); a
  full run is a deliberate later step once downstream FAISS/Isolation
  Forest results validate the approach at sample scale. Every embeddings
  output directory carries an `embeddings_sample_info.txt` disclosure
  file when it's a sample, removed automatically on a real full run, so
  nobody finds `embeddings.npy` later and assumes full coverage without
  checking.

## File layout

```
spam-detection-prototype/
├── common/
│   └── schemas.py              # canonical column/dtype contract, shared by both sources
├── config/
│   └── settings.py             # tunable params (SW_ whitelist prefix, op-4 filter, window sizes, embedding model/batch size)
├── ingestion/
│   ├── base.py                 # SourceHandlers(clean, map_to_canonical) - shared per-source contract
│   ├── dcs_codecs.py           # DCS -> text codec dispatch, shared by smpp.py and ss7.py
│   ├── smpp.py                 # SMPP: op-4 filter, UDH strip, DCS decode, raw->canonical mapping
│   ├── ss7.py                  # SS7: MO/MT_request filter, SRI->MT_request vlr_address merge, mapping
│   └── run_ingest.py           # file-by-file driver; SOURCES registry; load_features_csv()
├── labels/
│   └── rule_labels.py          # rule_evaluated / rule_flagged (fraud_type=="spam") label derivation
├── features/
│   ├── message_reassembly.py   # per-part rows -> one row per logical (multipart-aware) message
│   ├── behavioral.py           # per-MESSAGE point-in-time velocity/repeat-content features (training)
│   ├── behavioral_snapshot.py  # per-SENDER current-state snapshot (Feast online-store source)
│   ├── text_embeddings.py      # MiniLM sentence embeddings, dedup-then-broadcast by distinct text
│   └── faiss_index.py          # near-duplicate features (1hr/24hr windows) - see docs/experiments/faiss.md
├── feature_repo/
│   ├── feature_store.yaml      # Feast config - local SQLite registry + online store
│   └── definitions.py          # entity, FeatureView, on-demand feature view
├── scripts/
│   ├── refresh_feast.py               # snapshot -> feast apply -> feast materialize, one command
│   └── check_embedding_dominance.py   # one-off diagnostic behind the anomaly model's PCA decision - see docs/experiments/anomaly.md
├── models/
│   ├── metrics.py               # shared PR-AUC/log-loss eval, single-class-skip guard - used by both models below
│   ├── anomaly/
│   │   ├── data.py               # feature join (embeddings+behavioral+near-dup) + preprocessing pipeline
│   │   └── train.py              # Isolation Forest training - see docs/experiments/anomaly.md
│   └── rule_pattern/
│       ├── data.py               # rule_evaluated==True feature/label prep
│       └── train.py              # LightGBM training - see docs/experiments/rule_pattern.md
├── serving/
│   └── feature_lookup.py       # manual online-lookup test script (predict.py-style) - no FastAPI service yet
├── pipeline.py                 # top-level orchestrator: ingestion -> reassembly -> behavioral -> text_embeddings -> faiss, per source
├── tests/                      # pytest suite, one file per module above
├── notebooks/                  # ad-hoc real-data exploration (see individual notebook docstrings/markdown cells)
├── docs/
│   ├── architecture.md                    # this file - pipeline/system design decisions
│   ├── ml/
│   │   └── modeling.md                    # the two-score modeling strategy tying the experiments below together
│   ├── experiments/
│   │   ├── faiss.md                       # near-dup index design + tuning evidence
│   │   ├── anomaly.md                     # Isolation Forest design + tuning evidence
│   │   └── rule_pattern.md                # LightGBM design + tuning evidence
│   ├── feature_catalog.md                 # what's servable from Feast - name/type/meaning per feature
│   ├── prototype_plan.md
│   └── sms_spam_technical_architecture_plan.md
└── data/                        # gitignored - raw/ and processed/ real CDR data, feast_sources/ snapshot parquet
```
NOTE: this is the real-CDR ingestion pipeline (`data/raw/{SMPP,SS7}` ->
`data/processed/`), which has fully superseded the earlier synthetic-data
approach this section used to describe (`generate_data.py`,
`sms_events.parquet` - neither exists in the repo any more). Feast itself
is NOT stale, unlike those two - it's been rebuilt against this
pipeline's real output (see above), just under a from-scratch
`feature_repo/definitions.py` rather than whatever the synthetic-data
era's version looked like.

## Reused patterns from the earlier fraud-detection project (`mlflow_demo/`)

Same conventions apply here unless stated otherwise above:
- `train.py`-style CLI scripts (argparse flags for hyperparams, not
  hardcoded values)
- `compare_versions.py`-style champion/challenger comparison via MLflow
  registry, explicit promotion rule in code (MLflow doesn't decide this)
- `predict.py`-style single-row inference script for manual testing
- `plot_probabilities.py`-style calibration/distribution plots
- PR-AUC and log loss as primary metrics, not accuracy/ROC-AUC alone
  (imbalanced classes here too - spam is the minority class overall,
  though notably NOT within the rule_evaluated training pool itself -
  see `docs/experiments/rule_pattern.md`)
