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
  - **Inference-time scope is narrower than training-time scope, and this
    matters for FastAPI later.** A message the rule engine already
    confidently resolved (`rule_evaluated == True` - genuinely flagged,
    or genuinely evaluated-clean) does NOT get sent to either model at
    real-time inference - its verdict is already final, nothing to add.
    Both models only run on live traffic *because* the rule engine didn't
    confidently resolve it (`SW_*` whitelist bypass, or genuinely
    untouched) - that's the whole reason this system exists. This does
    NOT change training scope: `anomaly_score` still trains on the full
    traffic stream (needs to see confidently-normal traffic to learn
    what "normal" looks like), `rule_pattern_score` still trains only on
    `rule_evaluated == True` rows. Don't build the future FastAPI
    endpoint to blanket-score every request without checking
    `rule_evaluated` first - see README.md's "problem, precisely"
    section for the corrected inference-flow diagram.
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
  snapshot → apply → materialize sequence; `serving/feature_lookup.py` is
  the manual single-lookup test script (what a future FastAPI endpoint
  will call per request). Online freshness is bounded by refresh cadence,
  not by any TTL Feast enforces at read time - see
  `serving/feature_lookup.py`'s staleness caveat.

## Tech stack
- Feast (feature store, local SQLite registry + online store) —
  `feature_repo/` + `features/behavioral_snapshot.py` + `scripts/
  refresh_feast.py` — wired in for the sender-behavioral features (see
  architectural-decisions bullet above); requires `dill==0.4.1` pinned in
  requirements.txt (see "Known blockers" below), not Feast's own
  `dill~=0.3.0`
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
- ~~`huggingface.co` is not reachable from this sandbox~~ — **resolved
  as of 2026-08-24**: verified reachable, `sentence-transformers` +
  `all-MiniLM-L6-v2` installed and downloaded successfully in this same
  sandbox, real encode call confirmed working (`(N, 384)` float32 output).
  Environments change over time — if this blocker resurfaces, re-check
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
│   ├── behavioral.py           # per-MESSAGE point-in-time velocity/repeat-content features (training)
│   └── behavioral_snapshot.py  # per-SENDER current-state snapshot (Feast online-store source)
├── feature_repo/
│   ├── feature_store.yaml      # Feast config - local SQLite registry + online store
│   └── definitions.py          # entity, FeatureView, on-demand feature view
├── scripts/
│   └── refresh_feast.py        # snapshot -> feast apply -> feast materialize, one command
├── models/
│   ├── anomaly/                # not yet built - Isolation Forest + FAISS
│   └── rule_pattern/           # not yet built - LightGBM
├── serving/
│   └── feature_lookup.py       # manual online-lookup test script (predict.py-style) - no FastAPI service yet
├── pipeline.py                 # top-level orchestrator: ingestion -> reassembly -> behavioral, per source
├── tests/                      # pytest suite, one file per module above
├── notebooks/
│   └── explore_raw_cdrs.ipynb  # ad-hoc real-data exploration
├── docs/
│   ├── prototype_plan.md
│   ├── sms_spam_technical_architecture_plan.md
│   └── feature_catalog.md      # what's servable from Feast - name/type/meaning per feature
└── data/                        # gitignored - raw/ and processed/ real CDR data, feast_sources/ snapshot parquet
```
NOTE: this is the real-CDR ingestion pipeline (data/raw/{SMPP,SS7} ->
data/processed/), which has fully superseded the earlier synthetic-data
approach this section used to describe (`generate_data.py`,
`sms_events.parquet` - neither exists in the repo any more). Feast itself
is NOT stale, unlike those two - it's been rebuilt against this
pipeline's real output (see the architectural-decisions bullet above),
just under a from-scratch `feature_repo/definitions.py` rather than
whatever the synthetic-data era's version looked like.

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
1. MiniLM text embeddings — shared upstream dependency for both (2) and
   (3) below, not part of either one specifically. FAISS has no language
   understanding of its own (it's nearest-neighbor search over vectors,
   nothing more); Isolation Forest scores `[MiniLM embedding + behavioral
   features]` jointly, per the modeling plan. Build this once, consume it
   twice.
2. ~~FAISS near-duplicate index~~ — DONE, `features/faiss_index.py`.
   Three features per window, TWO windows (1hr + 24hr), not one:
   `near_dup_match_count` (raw count), `near_dup_distinct_senders` (how
   many different sender IDs behind those matches - disambiguates a
   coordinated spam blast from a bank sending one OTP template to
   thousands of real customers, since both produce a high match count
   but only the former spreads across many sender IDs), and
   `near_dup_max_similarity`. Two windows because a scammer pacing
   sends a few hours apart evades a 1hr-only window entirely - verified
   on real data: 6,907/40,000 SMPP sample messages (17.3%) have zero
   1hr matches but real 24hr matches. Scales independently of total
   corpus size via `compute_near_dup_features_chunked()` - processes
   bounded, sequential time-chunks (with a correctly-sized lookback
   buffer per chunk) rather than holding a growing corpus in one index;
   verified identical to the unchunked result in tests. FlatIP kept as
   the index type deliberately - chunking already keeps N small per
   index, which shrinks IVF's benefit below its recall cost at this
   scale; revisit only with real benchmark evidence.
   Deferred, not forgotten: SIM-farming detection (one physical IMSI
   cycling through many apparent MSISDNs, confirmed present in real SS7
   data - 1,541 IMSIs already show >1 distinct originator) is a
   DIFFERENT signal, not a FAISS window - it's an identity-linkage
   problem, not a content-similarity one. Belongs as a new SS7-only
   behavioral feature keyed by `imsi` (not `virtual_imsi`, which is
   routing-join plumbing, not a subscriber-identity field), mirroring
   `sender_unique_destinations_1hr`'s pattern. Not yet built.
3. ~~Isolation Forest training script~~ — DONE, `models/anomaly/train.py`
   (+ `models/anomaly/data.py` for the join/preprocessing). Trained on
   [embeddings + behavioral + FAISS near-dup] jointly, zero labels, on
   the full traffic stream. NOT wired into `pipeline.py` - training is
   a deliberate, versioned action, not a feature-computation step (see
   that script's own docstring). Two real fixes applied after
   measuring, not guessing: (a) PCA on the 384 embedding dims down to
   30 - an ablation (`scripts/check_embedding_dominance.py`) showed the
   raw dims were dominating the joint model's ranking (0.808 corr with
   embeddings-only vs 0.326 with behavioral-only); after PCA, 0.654 vs
   0.596 - much more balanced. (b) Real PR-AUC/log-loss evaluation
   against `rule_evaluated` labels as a VALIDATION-only proxy (never
   used in training) - PR-AUC 0.857 (SS7; SMPP skipped, mathematically
   undefined with zero confirmed-clean labels), modestly above the
   0.804 naive baseline (spam is the majority of the rule-evaluated
   subset - an artifact of which messages rules bother evaluating, not
   the true traffic-wide spam rate). Log loss not trustworthy yet -
   `anomaly_score` isn't a calibrated probability, would need Platt/
   isotonic calibration first.
4. ~~LightGBM training script~~ — DONE, `models/rule_pattern/train.py`
   (+ `data.py`). Features: canonical + behavioral + `source` -
   deliberately NOT embeddings. Two reasons: tree splits don't use
   dense 384-dim vectors well (a single split on one dimension in
   isolation captures almost nothing - that's what Isolation Forest is
   for); and skipping embeddings means this isn't bottlenecked by
   their sample, so it trains on the full rule_evaluated pool (352,655
   rows) instead. Test PR-AUC 0.999 (SS7/overall; SMPP skipped, no
   confirmed-clean labels) - expected, not remarkable: it's
   reconstructing the rule engine's own boundary from the same signal
   rules use, not evidence it generalizes to novel spam.
5. FastAPI service combining both, dual-score response shape
6. LIME wiring for both model types