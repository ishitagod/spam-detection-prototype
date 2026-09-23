# Handover: system overview

Written for someone picking up this repo cold. This is a guided tour, not
the full spec — it explains *what exists, why it's shaped this way, and
where to dig deeper*. For exhaustive detail, follow the pointers to
`docs/architecture.md`, `docs/ml/modeling.md`, `docs/experiments/*.md`,
and `CLAUDE.md` (the file Claude Code itself reads for project
conventions — also a good first read, it's kept short on purpose).

**Read this doc's own dates/claims skeptically.** It's a snapshot. Verify
against the actual code before making a decision on top of it — several
sections below exist specifically because an *earlier* doc went stale in
exactly this way.

## What this system does

Real-time SMS fraud detection over two telecom traffic sources — **SMPP**
(A2P message submission) and **SS7** (network-level signaling that also
carries message content). A message comes in, gets scored by two
independent ML models plus a rule engine's own verdict, and the system
returns a fraud/not-fraud decision with an explanation.

The one thing to internalize before reading anything else: **this project
deliberately keeps two different fraud signals separate and never averages
them.** One catches patterns the rules already know about; the other
catches statistically weird things the rules have never seen. Where they
disagree is the most valuable signal in the whole system — see "The
models" below.

## Pipeline: batch/offline (prepares data, nothing here runs per-request)

```
raw CDRs (data/raw/{SMPP,SS7})
  -> ingestion (ingestion/{smpp,ss7}.py)          canonical schema, per-source cleaning rules
  -> reassembly (features/message_reassembly.py)   multi-part SMS -> one row per logical message
  -> behavioral (features/behavioral.py)           sender velocity / repeat-content features
  -> content flags (features/content_flags.py)     deterministic regex features from text alone
  -> embeddings (features/text_embeddings.py)      MiniLM multilingual sentence embeddings
  -> FAISS near-dup (features/faiss_index.py)       near-duplicate match features, 1h/24h windows
```

Orchestrated by `pipeline.py` (`python pipeline.py`, or `--skip_embeddings`
to stop after content flags when you've only changed labeling/behavioral
logic and don't need to redo the expensive embeddings pass). Every stage
writes to `data/processed/<SOURCE>/...` — all gitignored except
`cluster_labels_template.csv` (see "Hand-labeling" below), since
everything else here is mechanically regenerable from `data/raw/`.

**SMPP and SS7 map to the same canonical schema before either model ever
sees the data** — `source` (`SMPP`/`SS7`) is itself a shared feature, not
a routing key to separate models. Source-specific signal (e.g. SS7's IMSI-
based SIM-farming indicator) stays as its own column rather than being
faked into a generic cross-source field it doesn't really apply to. See
`docs/architecture.md`'s "Key architectural decisions" for the reasoning
behind this and several other calls like it.

**Embeddings get computed once, consumed twice** — by FAISS (near-dup
search over the raw 384-dim vectors) and by Isolation Forest (as PCA-
reduced input features). LightGBM's default path does *not* consume
embeddings (`--with_embeddings` opts in) — see `docs/experiments/
rule_pattern.md` for why, and evidence it may be worth revisiting.

## The models — what each does, and why there are this many

**`rule_pattern_score`** (`models/rule_pattern/`) — supervised LightGBM.
Trains only on `rule_evaluated == True` rows, labeled by the upstream
telecom rule engine's own `rule_flagged` verdict. Answers *"does this
match a fraud pattern the rules already recognize?"* — a faster, more
gradable re-implementation of known patterns, not a generalizer to novel
fraud. Live, scored on every request.

**`anomaly_score`** (`models/anomaly/`) — unsupervised Isolation Forest.
Trains on the *full* traffic stream with zero labels — it has literally
never seen "this was fraud." Answers a different question: *"how
statistically rare is this message's feature combination?"* Catches
patterns rules haven't been written for yet, at the cost of also flagging
weird-but-harmless traffic — real transactional OTP bursts, device-
tracking protocol messages, and legitimate high-frequency system alerts
have all been observed scoring anomalous purely on volume, with nothing
fraudulent about them (see "Hand-labeling" above for the `not_fraud`
outcome that exists specifically for this).
Live, scored on every request via a real `.decision_function()` call
against the loaded champion — see `serving/anomaly_scoring.py`'s
docstring for how it gets near-dup features at serving time (queries a
FAISS index built over the source's *historical* embedding corpus, not a
live stream — a real, disclosed staleness/approximation tradeoff).

**Why two models, never blended into one score:** `rule_pattern_score`
generalizes to nothing new by design; `anomaly_score` can't tell "novel
fraud" from "novel but harmless." A message that's low on one and high on
the other is exactly the case worth surfacing — a possible new campaign
the rules haven't caught up to. Averaging them would bury that signal.
Both raw scores always stay on the API response, whatever else happens
downstream.

**`fusion_score`** (`models/decision_fusion/`, `serving/fusion_scoring.py`)
— a small trained LogisticRegression over `[rule_pattern_score,
anomaly_score]`, one per source. This is what actually decides
FRAUD/NOT_FRAUD at serving time when a champion exists for the request's
source (falls back to `rule_pattern_score` alone otherwise — see
`serving/app.py`'s `decision_score` logic). An *additive* combination, not
a replacement — it's still just two numbers in, one calibrated decision
out; the two raw scores are unaffected and still both returned.

**DBSCAN/HDBSCAN cluster discovery** (`models/anomaly/cluster_discovery.py`)
— **offline only, never runs at serving time, logs no deployable model.**
Takes the top ~0.5% by `anomaly_score` and groups similar anomalies
together in content-embedding space, answering *"which of these already-
flagged-as-weird messages resemble each other"* — Isolation Forest
structurally cannot answer that, it only ranks. A human then names each
real cluster (`spam`, `smishing`, `flooding`, etc. — see "Hand-labeling"
below) via `models/anomaly/inspect_clusters.py`. Default algorithm is now
**HDBSCAN** (no `eps` to hand-tune, handles wildly different campaign
densities in one run) — `--algorithm dbscan` kept for comparison. See
`docs/experiments/anomaly_clustering.md` for the full workflow and the
measured reasoning behind every default.

**`fraud_type_classifier`** (`models/fraud_type_classifier/`,
`serving/fraud_type_scoring.py`) — supervised LightGBM **multiclass**,
trained on the hand-confirmed cluster labels above. Answers *"what kind of
fraud is this"*, layered strictly on top of an already-FRAUD verdict —
only evaluated for requests the fusion/rule-pattern decision already
called FRAUD (cost control, same pattern as real-time SHAP below), never
gates `prediction`/`recommended_action` itself. `--label_source suggested`
(unconfirmed heuristic guesses) trains today but logs to a quarantined
`_suggested_labels`-suffixed MLflow experiment and must never be promoted
— only `--label_source confirmed` runs are real candidates. **A reserved
label value matters here:** `"not_fraud"` (`labels/cluster_labels.py::
NOT_FRAUD_LABEL`) marks a cluster a human confirmed is genuinely clean —
those rows are excluded from this classifier's training pool entirely,
since it answers "what kind of fraud", not "is this fraud." A cluster
that's anomalous/strange but doesn't fit a specific fraud type should be
labeled `"others"` instead — that string *is* included in training (it's
a real class the model should learn to recognize as "unclassified but
worth a second look"), it just isn't `"not_fraud"`. Don't conflate the two
when hand-labeling.

## Hand-labeling — the human-in-the-loop step that feeds the classifier above

`models/anomaly/inspect_clusters.py` writes
`data/processed/<SOURCE>/cluster_labels_template.csv` — one row per
cluster with real sample messages and behavioral stats, blank
`fraud_type_label` column. **This file is the one thing under
`data/processed/` that's tracked in git** (see `.gitignore`'s explicit
exception) — everything else there is regenerable from `data/raw/`, this
isn't; it's real human review work, and losing it means redoing that
work. `models/anomaly/ingest_cluster_labels.py` turns a filled-in template
into `cluster_labels.parquet`, which `fraud_type_classifier --label_source
confirmed` trains on.

Category taxonomy in use for `fraud_type_label`: `spam`, `smishing`,
`spoofing`, `flooding`, `not_fraud`, `others` — see the previous section
for the `not_fraud` vs `others` distinction specifically, it's not
obvious and easy to get backwards.

Until a `--label_source confirmed` `fraud_type_classifier` champion is
promoted for a source, `serving/fraud_type_scoring.py` simply has nothing
to load — that's expected, not a bug, and `serving/app.py` treats it as
best-effort (the request still succeeds, `fraud_subtype` just comes back
empty). This is the normal state for a source that hasn't been through a
full hand-labeling pass yet.

## Live prediction / serving flow

One endpoint, `POST /v1/score` (`serving/app.py`). Per request, in this
exact order (verified against the running code, not assumed):

1. **Feast lookup** (`serving/feature_lookup.py`) — sender-behavioral
   features (Postgres registry + Redis online store — see "Feature store"
   below, this is *not* what `docs/architecture.md` currently says).
   Cold-start (never-seen sender) is detected and handled, not an error.
2. **`rule_pattern_score`** (`serving/scoring.py`) — loads the per-source
   LightGBM champion (cached after first load), real inference. A missing
   champion here is a hard `FAILURE` — this score is load-bearing.
3. **`anomaly_score`** (`serving/anomaly_scoring.py`) — real Isolation
   Forest inference plus a live FAISS near-dup query. Best-effort: missing
   champion/corpus degrades to `None`, never fails the whole request.
4. **`fusion_score`** (`serving/fusion_scoring.py`) — only attempted if
   step 3 succeeded *and* a fusion champion exists for the source.
   `decision_score` = fusion score if available, else `rule_pattern_score`
   alone. This is what the FRAUD/NOT_FRAUD threshold actually applies to.
5. **If FRAUD:** real-time SHAP (`serving/scoring.py::
   explain_rule_pattern` — a cached `shap.TreeExplainer`, cheap/exact, safe
   inline) produces `reason_codes`/`feature_contributions`; separately,
   `fraud_subtype` is attempted via `fraud_type_scoring.py` (best-effort,
   see above — no champion yet means this is silently empty today). LIME
   is deliberately **not** wired in here — too expensive per-request,
   offline-only (`models/anomaly/explain.py`).
6. **Response:** `recommended_action` = `BLOCK` if any fraud_type
   predicted FRAUD, else `PASS`. Both raw scores (`rule_pattern_score` via
   `risk_score`, `anomaly_score`) and `fusion_score` are always present
   regardless of which one drove the decision.

Every stage logs its own timing (`logging`, not `print` — see
`docs/architecture.md`'s stage-level logging note) — a slow or failing
request is diagnosable from the logs alone, stage by stage.

## Feature store — Postgres + Redis, not SQLite

`docs/architecture.md` still says "local SQLite registry + online store,
no Kafka, no Redis" — **that's stale, don't trust it.** Per `CLAUDE.md`
(kept more current) and `docker-compose.yml`/`feature_repo/
feature_store.yaml`: Feast now runs on a **Postgres** SQL registry and a
**Redis** online store; MLflow's tracking store lives on the *same*
Postgres instance (separate `mlflow` database). Both run locally via
Docker Desktop — host Postgres is remapped to port 5433 (5432 was already
taken by a native Postgres install on the dev machine this was built on;
may not apply to yours). `requirements.txt` pins `dill==0.4.1` — Feast's
own `dill~=0.3.0` breaks on-demand feature view serialization under this
project's Python version; check this pin first if you see a pickling
error after a fresh install.

Behavioral features are **batch-refreshed**, not streaming — online
freshness is bounded by refresh cadence (`scripts/refresh_feast.py`), not
any TTL Feast enforces at read time. Real-time streaming ingestion (Kafka)
and continuous (non-batch) campaign discovery are the explicitly-named
biggest unfinished pieces of the production target — see `CLAUDE.md`'s
"Next" list.

**Before trusting any model as "live" or "promoted," check for yourself**
rather than assume this doc is current: a champion only exists once
`python -m models.compare_versions` has actually promoted one for that
source (see each `serving/*_scoring.py` module's `ChampionUnavailableError`
message for the exact commands to check/fix it). With a fresh pipeline run
on new data, expect to start from zero champions and work up through
`rule_pattern_score` → `anomaly_score` → `decision_fusion` →, once enough
hand-confirmed cluster labels exist, `fraud_type_classifier` — that
dependency order (fusion needs both scores trained first; the classifier
needs a full hand-labeling pass) is the real constraint, not calendar
time.

Also worth checking early: whether `messages_with_behavioral.csv` actually
has real content-flag columns (`has_url`, `has_gambling_keyword`, etc. —
see `config/settings.py::CONTENT_FLAG_PATTERNS`) populated, or defaults to
0 everywhere. That happens silently when a source's data predates
`features/content_flags.py` in the pipeline history and nobody's rerun
`pipeline.py` (or at least its content-flags stage) since — both models
train and serve "successfully" either way, just with that whole feature
family silently contributing nothing.

## Operational gotchas worth knowing before you burn an afternoon on them

- **If this repo ends up under OneDrive-synced `Documents`** (it was
  originally built there), watch for two distinct failure modes: (1) a
  *stuck* lock on a specific pre-existing file (doesn't
  clear with retries — delete the file, let it regenerate, or recreate
  small known-content files like MLflow's `registered_model_meta`), and
  (2) *transient* contention from a rapid burst of many small file writes
  in one loop (ingestion writes ~2 files per raw CDR file, back-to-back —
  this can outrun OneDrive's sync filter entirely and fail 100% of writes
  in the burst, not just a few). Pausing OneDrive sync during heavy
  pipeline runs sidesteps both. If you're on a different machine without
  this OneDrive setup, none of this applies — don't pre-emptively add
  retry logic for a problem you may not have.
- **`python` on this dev machine resolves to a project-local `venv`**
  missing some packages (e.g. `openpyxl`) that a system/Anaconda Python
  nearby has — check both before assuming a package needs installing.
- Windows console output can crash mid-print on multilingual/garbled SMS
  content (`UnicodeEncodeError` under the default `cp1252` codepage) —
  scripts that print real message text should reconfigure stdout with
  `errors="replace"` (see `models/anomaly/inspect_clusters.py::main()`)
  or write to a file with explicit `encoding="utf-8"`.

## Where to actually look next, by task

- **Understand a specific model's tuning history/evidence:**
  `docs/experiments/{anomaly,rule_pattern,faiss,anomaly_clustering}.md`.
- **Understand the two-score design philosophy in more depth:**
  `docs/ml/modeling.md`.
- **System/pipeline decisions, file layout, tech stack:**
  `docs/architecture.md` (cross-check its Feast/Kafka/Redis claims against
  this doc's "Feature store" section above — it's known stale there).
- **What's done vs. explicitly next, and why each item is ordered where it
  is:** `CLAUDE.md`'s "Current status" — this is the file kept most
  up-to-date turn to turn, treat it as the primary source of truth over
  the `docs/` files when they conflict.
- **Feature definitions:** `docs/feature_catalog.md` (what's servable from
  Feast) and `docs/feature_reference.md` (the full inventory, source by
  source).
