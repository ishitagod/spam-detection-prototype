# CLAUDE.md

## What this project is
Prototype SMS spam detector. Two SMS sources — SMPP (A2P, business sender
IDs) and SS7 (P2P, real MSISDNs) — feed one shared pipeline. Not
production-grade; goal is a working demo, then iterate.

## Key architectural decisions (don't relitigate without reason)
- **One shared feature contract for both sources.** SMPP and SS7 expose
  different raw fields, but both get mapped into the same canonical schema
  (`source`, `originator`, `text` + behavioral features) before touching
  any model. `event_type` and `originator_type` were in an earlier draft
  of this schema and have been dropped: on the SMPP side both were
  constant - SMPP is filtered to op-4/submit_sm only, and originator_type
  was a hardcoded "business_sender_id" - so neither carried any signal
  beyond what `source` already encodes. This was a deliberate
  simplification, not a data-driven finding - and SS7 (now wired in,
  `ingestion/ss7.py`) confirms the prediction: SS7's `message_type` DOES
  vary and DOES carry real signal. It drives SS7's entire row-filtering
  rule (MO/MT_request kept - the only types with real content; SRI_request/
  MT_response dropped; MT_SRI_response merged into MT_request via
  `virtual_imsi` for its `vlr_address`) and is kept as its own SS7-only
  feature column for exactly that reason. It is NOT folded into a generic
  cross-source `event_type` field - SMPP still has no equivalent signal
  there, so the shared-contract decision above stands. Revisit only if
  per-segment evaluation shows this needs to become a first-class shared
  concept.
- **One model to start, not two.** `source` is passed as a feature, not
  used to route to separate models. Only split into per-source models if
  per-segment evaluation (SS7-only vs SMPP-only PR-AUC/log loss) proves the
  shared model underperforms on one side.
- **Two separate scores, not one blended score:**
  - `rule_pattern_score` (supervised LightGBM, trained on rule-engine-
    labelled data) — a faster/cheaper re-implementation of known rules.
    Do NOT claim this generalizes to novel spam.
  - `anomaly_score` (unsupervised Isolation Forest + FAISS near-duplicate
    matching) — trained with zero labels. This is the layer meant to catch
    spam the rules don't already know about.
  - Cases where these two disagree are the most valuable output — surface
    them, don't average them away.
- **Point-in-time features only.** Never use a feature in training that
  wouldn't have been knowable at actual inference time (e.g., final
  campaign size once complete). This caused real bugs in the earlier
  fraud-detection project this one reuses tooling from — don't repeat it.
- **Offline and online ML platform are the SAME environment for this
  prototype** (one FastAPI service, no separate training/serving infra
  split). This is a deliberate simplification, not the target production
  design — don't build it assuming it's final.
- **No Kafka, no Redis.** Feast uses local SQLite online store. Batch/
  synchronous feature computation, not streaming, at this stage.

## Tech stack
- Feast (feature store, local SQLite online store) — `feature_repo/`
- sentence-transformers, model = `all-MiniLM-L6-v2` (prototype choice —
  smallest footprint of the candidates; Distil-mBERT/XLM-R are production
  options, not used here)
- FAISS (near-duplicate detection)
- scikit-learn Isolation Forest (unsupervised anomaly scoring)
- LightGBM (supervised classifier)
- MLflow (experiment tracking + model registry, SQLite backend:
  `sqlite:///mlflow.db`)
- LIME (explainability, both model types)
- SHAP (explainability, added alongside LIME — not yet decided which is
  primary vs. supplementary; revisit once both are wired in)
- FastAPI (inference service)

## Known blockers / environment notes
- `huggingface.co` is not reachable from this sandbox — MiniLM weights
  can't be downloaded here. Code path is correct; test on a machine with
  normal internet access.
- Disk space has been tight in this sandbox before (`pip install
  --no-cache-dir`, avoid re-triggering large caches).

## File layout
```
spam-detection-prototype/
├── common/
│   └── schemas.py              # canonical column/dtype contract, shared by both sources
├── config/
│   └── settings.py             # tunable params (SW_ whitelist prefix, op-4 filter, window sizes)
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
│   └── (behavioral.py pending rebuild - see "Next steps")
├── models/
│   ├── anomaly/                # not yet built - Isolation Forest + FAISS
│   └── rule_pattern/           # not yet built - LightGBM
├── serving/                    # not yet built - FastAPI inference service
├── pipeline.py                 # top-level orchestrator: ingestion -> reassembly, per source
├── tests/                      # pytest suite, one file per module above
├── notebooks/
│   └── explore_raw_cdrs.ipynb  # ad-hoc real-data exploration
├── docs/
│   ├── prototype_plan.md
│   └── sms_spam_technical_architecture_plan.md
└── data/                        # gitignored - raw/ and processed/ real CDR data
```
NOTE: this is the real-CDR ingestion pipeline (data/raw/{SMPP,SS7} ->
data/processed/), which has fully superseded the earlier synthetic-data
approach this section used to describe (`generate_data.py`, a Feast
`feature_repo/`, `sms_events.parquet` - none of these exist in the repo
any more). The "Tech stack" section above still lists Feast - that's now
equally stale and hasn't been revisited; flag/fix it if Feast is still
part of the plan, or drop it if this pipeline has superseded it too.

## Reused patterns from the earlier fraud-detection project (mlflow_demo/)
Same conventions apply here unless stated otherwise above:
- `train.py`-style CLI scripts (argparse flags for hyperparams, not
  hardcoded values)
- `compare_versions.py`-style champion/challenger comparison via MLflow
  registry, explicit promotion rule in code (MLflow doesn't decide this)
- `predict.py`-style single-row inference script for manual testing
- `plot_probabilities.py`-style calibration/distribution plots
- PR-AUC and log loss as primary metrics, not accuracy/ROC-AUC alone
  (imbalanced classes here too — spam is the minority class)

## Next steps (in order)
1. FAISS near-duplicate index
2. Isolation Forest training script (unsupervised layer)
3. LightGBM training script (supervised layer, rule-labelled data)
4. FastAPI service combining both, dual-score response shape
5. LIME wiring for both model types