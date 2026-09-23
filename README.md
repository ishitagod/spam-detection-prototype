# SMS Spam Detection

A dual-source (SMPP + SS7) SMS fraud detection system that sits
**downstream of an existing rule engine**, not in place of it. It scores
the traffic the rule engine couldn't already confidently decide on, using
layered ML: a supervised pattern model, an unsupervised anomaly model, a
trained fusion model that combines them into the live decision, and a
downstream fraud-type classifier that labels what kind of fraud a
FRAUD-flagged message is.

See [`CLAUDE.md`](CLAUDE.md) for the full architectural rules this
codebase follows, and
[`docs/sms_spam_technical_architecture_plan.md`](docs/sms_spam_technical_architecture_plan.md)
for the target production design.

---

## Architecture

```
                              Incoming message (SMPP or SS7)
                                        │
                                 Rule engine (external)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                                ▼
      rule_evaluated == True                              NOT rule_evaluated
   (real content rule fired -                          (SW_* whitelist bypass,
    decision already final)                              or genuinely untouched)
                │                                                │
        No ML scoring at                              Shared feature computation
        inference - verdict                          behavioral + content-rule flags
        already final. (Still                              + MiniLM embedding
        used for TRAINING both                                    │
        models below.)                          ┌──────────────────┴───────────────────┐
                                                  ▼                                       ▼
                                       FAISS near-dup search                    (features feed both
                                       (1hr/24hr windows)                        models below directly)
                                                  │                                       │
                                                  └───────────────────┬───────────────────┘
                                                                      ▼
                                          ┌───────────────────────────────────────────┐
                                          ▼                                             ▼
                                ┌──────────────────┐                        ┌───────────────────────┐
                                │     LightGBM       │                        │    Isolation Forest     │
                                │  rule_pattern_score │                        │     anomaly_score        │
                                │  supervised, trains  │                        │  unsupervised, trains    │
                                │  on rule_evaluated    │                        │  on the FULL traffic      │
                                │  == True only          │                        │  stream, zero labels       │
                                └──────────┬────────────┘                        └────────────┬────────────┘
                                           │                                                    │
                                           └───────────────────────┬────────────────────────────┘
                                                                    ▼
                                                    Decision Fusion (trained meta-model)
                                                       LogisticRegression([rule_pattern_score,
                                                                            anomaly_score])
                                                                fusion_score
                                                                    │
                                                                    ▼
                                                    prediction: FRAUD / NOT_FRAUD
                                                    recommended_action: BLOCK / PASS
                                                                    │
                                                          (only when FRAUD)
                                                                    ▼
                                                        Fraud Type Classifier
                                                     multiclass LightGBM, trained on
                                                    hand-confirmed DBSCAN cluster labels
                                                          fraud_subtype (label only,
                                                        never influences the decision)
```

**Both raw scores are always returned, never averaged** — disagreement
between them is itself a signal (low `rule_pattern_score` + high
`anomaly_score` is the highest-value output of the whole system: a
candidate novel fraud pattern the rule engine has never seen).
`fusion_score` is what actually drives `prediction`/`recommended_action`
when a fusion champion exists for the request's source; it falls back to
`rule_pattern_score` alone otherwise. `fraud_subtype` never gates the
decision — it's a downstream label computed only after fusion has already
said FRAUD, using a classifier trained exclusively on confirmed-fraud
cluster labels (not-fraud-confirmed clusters are explicitly excluded from
its training pool — see [`labels/cluster_labels.py`](labels/cluster_labels.py)).

### Why four models, not one

- **LightGBM** can only ever re-recognize patterns the rule engine already
  encodes — it's a faster, cheaper reconstruction of known rules, not a
  novel-spam detector.
- **Isolation Forest** trains with zero labels on the full traffic stream
  — it's the layer meant to catch fraud the rules have never seen, at the
  cost of being noisier and less explainable per-request.
- **Decision Fusion** exists because "which of the two scores wins" is
  itself a learnable question, not a fixed threshold — a small, fully
  inspectable (2 coefficients + intercept) LogisticRegression beats a
  hand-picked rule for combining them.
- **Fraud Type Classifier** exists because "is this fraud" and "what kind
  of fraud" are different questions with different label sources (rule
  engine vs. hand-confirmed DBSCAN clusters) and different failure modes
  if conflated — see [`labels/rule_labels.py`](labels/rule_labels.py) and
  [`labels/cluster_labels.py`](labels/cluster_labels.py).

---

## Data: two protocols, one contract

- **SMPP** — Application-to-Person (A2P), business sender IDs, bulk/
  campaign traffic. Ingested from `op-4` (`submit_sm`) PDUs only.
- **SS7** — Person-to-Person (P2P), real MSISDNs on both ends, MO/
  MT_request rows only — see [`ingestion/ss7.py`](ingestion/ss7.py).

Both map into one canonical schema ([`common/schemas.py`](common/schemas.py))
before anything touches a model. `source` ("SMPP"/"SS7") is a shared
feature, not a routing key — models are trained per-source
(`--sources SMPP` / `--sources SS7`, see `CLAUDE.md`'s source-split rule)
because inference is expected to diverge, evaluated against a shared
baseline via [`scripts/check_source_split_justified.py`](scripts/check_source_split_justified.py).

**Point-in-time discipline**: every feature must be knowable at the
moment inference actually happens. Fields that only resolve once a
transaction/campaign is complete are offline-analysis-only, never
training features.

### Real label composition (SMPP vs SS7, full ingestion)

| Source | `rule_evaluated` rows | Flagged (spam) | Confirmed-clean |
|---|---:|---:|---:|
| SMPP | 139,546 | 2,692 (1.9%) | 136,854 (98.1%) |
| SS7  | 2,654,369 | 284,073 (10.7%) | 2,370,296 (89.3%) |

Both sources are genuinely imbalanced toward confirmed-clean — the
supervised model has to be read with that in mind (PR-AUC/precision@K
over raw accuracy, see [Evaluation](#evaluation) below).

---

## Content-rule flags

20 deterministic regex features computed from `text` alone
([`config/settings.py::CONTENT_FLAG_PATTERNS`](config/settings.py),
[`features/content_flags.py`](features/content_flags.py)) — `has_url`,
`has_gambling_keyword`, `has_otp_keyword`, `has_brand_impersonation_keyword`,
`has_known_malicious_domain`, etc., multilingual (English + Malay/
Manglish + regional scam vocabulary). Base features for **both** LightGBM
and Isolation Forest, same treatment as behavioral features — not
ablation-gated like TF-IDF/embeddings.

Three ways to turn these into a label, kept strictly separate from the
rule engine's own `rule_flagged` (which has zero content/regex matching
of its own — see [`labels/rule_labels.py`](labels/rule_labels.py)):
- `content_flagged()` — a fixed high-confidence combination list (8
  combinations, e.g. `has_gambling_keyword` alone, or
  `has_url + has_urgency_keyword` together).
- `content_flagged_by_count()` — "at least N of 20 flags fired," a
  simpler but more arbitrary alternative.
- `fit_content_flag_weights()` / `content_flagged_by_weight()` — fits a
  `LogisticRegression(flags -> rule_flagged)` on the real labelled pool,
  so each flag's weight is a measured coefficient, not a guess. Used by
  `models/rule_pattern/train.py --include_content_labels` to add
  confident positives from unlabelled traffic into LightGBM's training
  pool, tagged with a distinct `label_source` and evaluated as its own
  breakdown — never silently merged into the telecom-derived rows.

---

## Metrics (current, real runs)

### LightGBM — `rule_pattern_score`
- **SS7/overall**: test PR-AUC **0.999** — expected, not remarkable: it's
  reconstructing the rule engine's own decision boundary from the same
  signal the rules use, not evidence of generalizing to novel fraud.
- **SMPP-only**: real confirmed-clean population now exists (98.1% of
  139,546 rows) so a real SMPP-only PR-AUC is computable, but the retrain
  to produce that number hasn't been run yet — see
  [`docs/experiments/rule_pattern.md`](docs/experiments/rule_pattern.md).
- `--with_embeddings --sources SS7`: full SS7 embedding corpus
  (2,742,301 rows) is ready and covers 100% of SS7's `rule_evaluated`
  pool, but the retrain comparing it against the no-embeddings baseline
  (test PR-AUC 0.866) hasn't been logged yet.

### Isolation Forest — `anomaly_score`
- **SS7** (full 2,742,301-row corpus): PR-AUC **0.164** against a naive
  baseline of ~0.107 (SS7's real positive rate) — a real but modest
  ~1.5x lift; PR-AUC alone understates this layer's value (see
  [`docs/experiments/anomaly.md`](docs/experiments/anomaly.md)).
- **Precision@top-0.1%** (the operationally relevant cutoff): **0.204**,
  a ~1.9x lift over the naive baseline, after two measured
  feature-encoding fixes (`sender_age_days` bucketing,
  `sender_recipient_diversity_ratio` small-sample gating).
- **SMPP**: PR-AUC **0.074** against a ~0.019 baseline (~3.8x lift);
  precision@top-0.1% **0.057**.
- Validated against real `rule_flagged` labels purely as a floor-level
  sanity check — this layer's real purpose (catching fraud the rules
  can't see) has no labels to formally evaluate against by definition.

### Decision Fusion — `fusion_score`
Trained `LogisticRegression([rule_pattern_score, anomaly_score])`, both
sources promoted (`decision_fusion_model_SMPP` / `_SS7`, alias
`champion`):
- **SMPP**: 139,546-row pool, test PR-AUC **0.9954**.
- **SS7**: full 2,654,369-row pool (100% `anomaly_score` coverage), test
  PR-AUC **0.9735**.

### Fraud Type Classifier — `fraud_subtype`
**Experimental, not yet a real candidate.** Zero DBSCAN clusters have
been hand-confirmed yet (`docs/experiments/anomaly_clustering.md` step 4
is still outstanding), so no `--label_source confirmed` model has ever
been trained. `--label_source suggested` (unconfirmed heuristic guesses)
runs today but logs to a separate `_suggested_labels` MLflow experiment
and must never be promoted. At serving time
([`serving/fraud_type_scoring.py`](serving/fraud_type_scoring.py)), this
means `fraud_subtype` legitimately comes back `None` for every request
right now — expected, not a bug, handled the same best-effort way as a
missing `fusion_score` champion.

### FAISS near-duplicate matching
Two windows (1hr + 24hr) because a paced-out campaign (a few near-dup
sends every couple hours) evades a single short window — verified on
real SMPP sample data, 17.3% of messages show zero 1hr matches but real
24hr matches. `near_dup_distinct_senders` is the key disambiguator
between a coordinated blast and one legitimate sender's bulk template.

**Evaluation convention**: PR-AUC and log loss are primary, not accuracy
— fraud is a minority class on both sources. Track precision at a fixed
recall/rank cutoff (`models/metrics.py`'s `precision_at_k`) as the number
that actually maps to a block/allow decision. Every model is evaluated
three ways: overall, SMPP-only, SS7-only.

---

## Serving

[`serving/app.py`](serving/app.py) — FastAPI service, one `/v1/score`
call per message:
1. Rule-resolved traffic (`rule_evaluated == True`) is never scored —
   its verdict is already final.
2. `score_rule_pattern()` computes `rule_pattern_score` (always).
3. `score_anomaly()` computes `anomaly_score` — best-effort, degrades to
   `None` if no champion/corpus exists for that source, never fails the
   request.
4. `score_fusion()` computes `fusion_score` from both — best-effort, same
   degradation; `decision_score` falls back to `rule_pattern_score` alone
   if fusion is unavailable.
5. `explain_rule_pattern()` (real-time SHAP, `shap.TreeExplainer`, cached
   per source) runs only on FRAUD predictions (cost control) and drives
   `reason_codes` + `feature_contributions` from real per-request
   contributions, not fixed thresholds.
6. `score_fraud_type()` runs only on FRAUD predictions, alongside SHAP —
   best-effort, populates `fraud_subtype`/`fraud_subtype_confidence` or
   degrades to `None`.

Every per-source model cache (`scoring`, `anomaly_scoring`,
`fusion_scoring`, `fraud_type_scoring`) is warmed at process startup
(`lifespan()`), independently best-effort — a source with no promoted
champion for one model never blocks startup for the others.

---

## Repo layout

```
spam-detection-prototype/
├── common/schemas.py               # canonical column/dtype contract
├── config/settings.py              # tunables, CONTENT_FLAG_PATTERNS registry
├── ingestion/                      # SMPP/SS7 ingestion + canonical mapping
├── labels/
│   ├── rule_labels.py              # rule_evaluated/rule_flagged/content_flagged derivation
│   └── cluster_labels.py           # DBSCAN-confirmed cluster label derivation
├── features/
│   ├── content_flags.py            # deterministic regex features
│   ├── message_reassembly.py       # multipart parts → one row/message
│   ├── behavioral.py               # per-message point-in-time features
│   ├── behavioral_snapshot.py      # per-sender snapshot, feeds Feast
│   ├── text_embeddings.py          # MiniLM embeddings
│   └── faiss_index.py              # near-duplicate matching
├── feature_repo/                   # Feast (Postgres registry, Redis online store)
├── models/
│   ├── anomaly/                    # Isolation Forest + DBSCAN cluster discovery
│   ├── rule_pattern/                # LightGBM supervised model
│   ├── decision_fusion/            # fusion meta-model
│   ├── fraud_type_classifier/      # experimental multiclass fraud-type model
│   └── compare_versions.py         # champion/challenger promotion
├── serving/
│   ├── app.py                      # FastAPI /v1/score
│   ├── scoring.py                  # rule_pattern_score + real-time SHAP
│   ├── anomaly_scoring.py          # anomaly_score
│   ├── fusion_scoring.py           # fusion_score
│   └── fraud_type_scoring.py       # fraud_subtype
├── pipeline.py                     # ingestion → reassembly → behavioral → embeddings → FAISS
├── scripts/                        # operational scripts (Feast refresh, split-justification check, etc.)
├── tests/                          # pytest, one file per module
└── docs/                           # architecture, feature catalog, experiment results
```

---

## Tech stack

- Python, FastAPI
- Feast — Postgres SQL registry, Redis online store (`docker-compose.yml`)
- Postgres — also hosts the MLflow tracking store (separate `mlflow`
  database, `scripts/postgres_init/`)
- Kafka — production target for streaming ingestion and real-time
  campaign discovery, not yet integrated
- LightGBM, scikit-learn, sentence-transformers (MiniLM), FAISS (exact
  `IndexFlatIP` batch/training, IVF-PQ serving-time)
- MLflow (experiment tracking + model registry, champion/challenger
  promotion via `models/compare_versions.py`)
- LIME + SHAP (SHAP wired into real-time `/v1/score`; LIME stays
  offline-only, too expensive per-request)

Local infra: Postgres + Redis run via Docker Desktop
(`docker-compose.yml`), host Postgres remapped to port 5433.

---

## Setup

```bash
pip install --no-cache-dir -r requirements.txt
docker compose up -d                                 # Postgres + Redis
pytest                                                # run the test suite
python pipeline.py                                    # ingest + reassemble real CDRs
uvicorn serving.app:app --reload                       # start the scoring API
```

Real raw/processed CDR data lives under `data/` and is gitignored.

---

## Roadmap

In order (per `CLAUDE.md`'s "Next"):

1. **Kafka-fed streaming ingestion** — replaces scheduled-batch
   behavioral feature refresh and DBSCAN cluster discovery with
   continuous, real-time paths. Last piece of the production infra
   migration, not started.
2. **Full SMPP text-embeddings run** — SS7's is done (GPU, full corpus);
   SMPP's unblocks `rule_pattern_score --with_embeddings --sources SMPP`.
3. **Hand-confirm DBSCAN clusters** → train
   `fraud_type_classifier --label_source confirmed` → promote a real
   champion. Currently zero clusters confirmed, so `fraud_subtype` is
   `None` on every live request.
4. **`rule_pattern_score --with_embeddings --sources SS7`** — corpus is
   ready (100% `rule_evaluated` coverage), retrain not yet run/logged.
5. **Live confirmed-campaign lookup** — join a live message's FAISS
   near-dup match against `cluster_labels.parquet` directly, not just via
   the trained classifier's generalization. Not yet built.
