# SMS Spam Detection — Prototype

A dual-source (SMPP + SS7) SMS spam detector that sits **downstream of an
existing rule engine**, not in place of it. Its job is narrow and specific:
score the traffic the rule engine couldn't already confidently decide on,
using two independent ML layers that answer two different questions —
*"does this look like a pattern we already know?"* and *"does this look
wrong even though we've never seen it before?"*

This is a prototype/demo build, not production-grade — see
[`CLAUDE.md`](CLAUDE.md) for the architectural decisions this codebase
deliberately does and doesn't make, and
[`docs/sms_spam_technical_architecture_plan.md`](docs/sms_spam_technical_architecture_plan.md)
for the full production-scale design this prototype is a smaller slice of.

---

## The problem, precisely

Every inbound SMS (from either source) has already been through an
upstream rule engine before this project ever sees it. That rule engine
leaves each message in one of **three** states, and the three states need
three different treatments — conflating them is the single easiest way to
build a broken training set:

| Rule engine outcome | What actually happened | What it's worth for ML |
|---|---|---|
| **Flagged** (`decision=1`, `rule_flagged=True`) | A real spam-pattern rule matched the content and fired | A trustworthy **positive** label |
| **Evaluated, not flagged** (`decision=0` from a *real* content rule, `rule_evaluated=True`) | A spam-pattern rule actually scored the content and explicitly decided "allow" | A trustworthy **negative** label — `decision=0` on its own genuinely does mean confirmed-clean, *when it comes from a rule that looked at the content* |
| **Whitelisted** (`decision=0` via `SW_*`) | Sender/route ID hit a pre-check allowlist **before** content was ever evaluated — a business exception (client-requested), not a content verdict. The message could still be spam; the rule engine just never checked | **Not a label** — unlabelled, same as blank |
| **Never touched** (`decision` blank/no rule at all) | Nothing about this message matched any rule, allow or deny | **Not a label** — unlabelled, same as blank |

The middle two rows both surface as `decision=0`, and the distinction
between them is the entire point: `decision=0` from a rule that actually
evaluated the content **is** a confirmed negative — but `decision=0` via
`SW_*` is a bypass, not a verdict, and has to be treated as unlabelled
right alongside the truly-untouched rows, not folded into the
confirmed-negative bucket just because both happen to show `decision=0`.
This is exactly what
[`labels/rule_labels.py`](labels/rule_labels.py)'s `is_rule_evaluated()`
already does: it requires a real (non-`SW_*`) rule to have fired — decision
0 or 1, either way — before a row counts as labelled at all; `SW_*` hits
and genuinely untouched rows are both excluded, never silently assigned
`rule_flagged=False`.

**What the real data shows:** as of the currently ingested SMPP files,
this distinction exists in the code but not yet in practice — every real
`decision=0` row checked so far is a `SW_*` bypass or fully untouched;
none come from a content rule that actually ran and said "allow" (see the
[data reality check](#data-reality-check-what-the-real-data-actually-looks-like)
below for the exact counts). So SMPP's confirmed-negative pool is
currently empty not because the code miscategorizes real negatives, but
because the rule engine hasn't produced any yet in this dataset.

**This is why the project needs two different kinds of ML, not one:**

- The **labelled pool** (`rule_evaluated == True`) is small, known-pattern,
  and — see [Data reality check](#data-reality-check-what-the-real-data-actually-looks-like) below —
  can be *extremely* imbalanced. It's enough to train a **supervised**
  model, but that model can only ever learn to re-recognize patterns the
  rule engine already encodes. It cannot, by definition, generalize to
  spam the rules have never seen.
- The **unlabelled pool** (`rule_evaluated == False`) is nearly all of the
  traffic, and it's exactly where genuinely novel spam — new campaigns,
  new templates, spam from a sender that just cleared the whitelist gate —
  would be hiding undetected. There are no labels to supervise on here, so
  this pool needs an **unsupervised** approach: does this message's
  content and behavior look anomalous relative to normal traffic, on its
  own terms, without ever being told what "spam" means?

**Training breadth and inference-time scope are two different questions,
and it's worth being precise about which is which:**

- **Training**: the supervised model trains only on `rule_evaluated ==
  True` rows; the unsupervised model trains on the **full** traffic
  stream, `rule_evaluated` status included — it needs to see what
  confidently-normal traffic looks like too, or its idea of "normal"
  ends up skewed toward only the ambiguous cases.
- **Inference (real-time, per live message)**: a message that the rule
  engine already confidently resolved — genuinely flagged, or genuinely
  evaluated-clean — does **not** get sent to either ML model. Its verdict
  is already final; there's no reason to spend a real-time scoring call
  on a decision that's already made. Both models only run at inference
  time on the traffic that reaches them **because** the rule engine
  didn't confidently resolve it: `SW_*` whitelist bypasses and genuinely
  untouched messages. That's the entire point of this system — it exists
  to score what the rule engine left unresolved, not to re-score what it
  already decided.

### The two scores, kept separate on purpose

```
                    incoming SMS (SMPP or SS7)
                              │
                       rule engine decision
                              │
        ┌─────────────────────┴─────────────────────────┐
        ▼                                                ▼
  rule_evaluated == True                        NOT rule_evaluated
  (real content rule fired -                    (SW_* whitelist bypass,
   decision=1 flagged, OR                        or genuinely untouched -
   decision=0 confirmed-clean)                   the unresolved majority)
        │                                                │
        ▼                                                ▼
  Verdict is already final.                   canonical feature contract
  NO ML SCORING at inference -                        (see Data below)
  nothing left to add. (This                            │
  traffic still feeds TRAINING -              ┌──────────┴──────────┐
  see the note above this diagram.)           ▼                     ▼
                                    rule_pattern_score          anomaly_score
                                    supervised (LightGBM /       unsupervised (Isolation
                                    CatBoost / XGBoost),         Forest + FAISS near-dup
                                    trained on rule_evaluated    match on MiniLM/DistilBERT
                                    ==True rows. A faster,       embeddings), trained with
                                    cheaper re-implementation    ZERO labels on the FULL
                                    of rules the rule engine     traffic stream (training
                                    already knows. Do NOT        breadth, not inference
                                    claim this generalizes       scope - see note above).
                                    to novel spam.               Catches spam the rule
                                                                  engine has never encoded.
                                              │                          │
                                              └────────────┬────────────┘
                                                            ▼
                                    BOTH scores returned, never averaged.
                              agreement="match" | "disagree" is itself a signal:
                           low rule_pattern_score + high anomaly_score = the highest-
                           value output of the whole system — a candidate NOVEL spam
                                    pattern the rule engine has never seen.
                                                            │
                                                            ▼
                                    confidence-gated decision: block only above a high-
                                   confidence threshold; anything else routes to human/
                                              rule-engine-team review
```

This mirrors the response contract in
[`docs/prototype_plan.md`](docs/prototype_plan.md#4-score-outputs):

```json
{
  "rule_pattern_score": 0.91,
  "anomaly_score": 0.34,
  "agreement": "match | disagree",
  "confidence": 0.8,
  "reason_codes": ["near_duplicate_burst", "matches_known_pattern"]
}
```

`confidence` is a distinct field from either score — it reflects how much
behavioral history exists for this sender (a brand-new sender's first
message is structurally low-confidence, not falsely certain either way)
and how close the message sits to a known cluster. **Blocking only happens
above a high-confidence threshold**; low-confidence outputs — regardless
of how high the raw score is — route to human/rule-engine review instead
of an automatic block. See
[`docs/sms_spam_technical_architecture_plan.md`, §5](docs/sms_spam_technical_architecture_plan.md#5-real-time-inference-contract)
for the full reasoning (cold-start disclosure, probability-vs-confidence
split).

---

## Data: two protocols, one contract

SMS reaches this pipeline from two structurally different sources that
happen to carry the same kind of abuse:

- **SMPP** — Application-to-Person (A2P). Business sender IDs, bulk/
  campaign traffic, submitted via an SMPP bind. Ingested from `op-4`
  (`submit_sm`) PDUs only — other operation types (delivery acks, etc.)
  carry no message content and are dropped.
- **SS7** — Person-to-Person (P2P). Real MSISDNs on both ends, signalled
  through the SS7 MAP protocol's MT (mobile-terminated, 4-message-type
  handshake) and MO (mobile-originated, single message) flows. Only `MO`
  and `MT_request` rows carry real content; the SRI query/response and
  delivery-ack rows are dropped or merged in as auxiliary signal (routing/
  VLR address) — see [`ingestion/ss7.py`](ingestion/ss7.py)'s docstring for
  the full row-filtering rationale, verified against real data.

SMPP and SS7 expose **different raw fields** (business sender-ID metadata
vs. real subscriber/roaming metadata) — that's expected and fine. What's
not allowed to differ is what a model actually consumes: both sources are
mapped into the same canonical feature contract
([`common/schemas.py`](common/schemas.py)) before anything touches a
model — `source`, `originator`, `destination`, `text`, `timestamp`, `dcs`,
`text_decode_failed`, plus each source's own extra columns carried through
unchanged. `source` ("SMPP"/"SS7") is passed to the model **as a feature**,
not used to route to two separate models — see
[`docs/sms_spam_technical_architecture_plan.md`, §2.5](docs/sms_spam_technical_architecture_plan.md#25-model-strategy-one-model-first-split-only-with-evidence)
for why: start with one shared model, evaluate SMPP-only / SS7-only
segments separately, and only split into per-source models if that
evaluation proves the shared model is underperforming on one side.

**Point-in-time discipline:** every canonical/behavioral feature has to
pass "would this have been knowable at the moment inference actually
happens" before it's allowed in — see `CLAUDE.md`. Fields that only
resolve once a transaction/campaign is complete (final delivery status,
final campaign size) are offline-analysis-only, never training features.
This is a real, previously-hit bug class in the fraud-detection project
this pipeline's tooling was carried over from — worth restating every time
a new feature is added.

### Data reality check: what the real data actually looks like

The ingested dataset (48 hourly CDR files per source, `2026-08-02` –
`2026-08-03`) is real production-shaped traffic, not synthetic. Numbers
below are from the current `data/processed/` output
(`ingestion_manifest.csv`, post row-cleaning, pre-reassembly):

| Source | Raw rows | Kept (real content) | Rule-evaluated (labelled) | Rule-flagged (spam) |
|---|---:|---:|---:|---:|
| SMPP | 14,101,168 | 6,045,250 | 3,177 (0.05%) | 3,177 (**100%** of evaluated) |
| SS7  | 14,685,436 | 3,417,425 | 435,233 (12.7%) | 363,205 (83.5% of evaluated) |

After multipart message reassembly (SMPP: 5,505,921 logical messages; SS7:
2,742,301 — both stages done, see [Repo layout & current build status](#repo-layout--current-build-status)),
the SMPP numbers hold: **2,693 rule-evaluated messages, all 2,693 flagged
spam.** This is a genuine, load-bearing finding, not a rounding artifact —
confirmed by breaking `decision==0` itself down (sampled across 6 SMPP
files, 178,868 `decision==0` rows):

| `decision==0` rows | count | share | labelled? |
|---|---:|---:|---|
| `SW_*` whitelist bypass | 168,021 | 94.0% | No — business exception, content never scored |
| No rule at all (`rule`/`rule_name` both null) | 10,847 | 6.0% | No — untouched |
| Real content rule, explicitly decided "allow" | **0** | 0.0% | Would be **yes**, if any existed |

`decision==0` genuinely is a confirmed-negative label **when it comes from
a rule that actually evaluated the content** — but in the SMPP data ingested
so far, that case doesn't occur at all: every `decision==0` row is either a
`SW_*` bypass or was never touched by any rule. So **on the data ingested
so far, SMPP has zero confirmed-clean labels from real rule-content
evaluation** — not because the label logic is discarding real negatives,
but because the rule engine hasn't produced any in this dataset yet. Every
SMPP message that a spam-pattern rule actually scored was, in fact, spam.

Consequences this shapes for the modeling plan:

- The supervised `rule_pattern_score` model's labelled training set is
  **almost entirely SS7-sourced** — SS7 supplies both a much larger
  labelled pool and, critically, real confirmed-negative examples that
  SMPP currently doesn't have at all. Per-segment (SMPP-only vs SS7-only)
  PR-AUC has to be evaluated separately regardless, but for SMPP
  specifically, watch for the supervised model effectively learning "spam
  vs. everything-else" from an SS7-shaped decision boundary and needing an
  explicit precision check once real SMPP negatives (if any exist) surface.
- The unsupervised `anomaly_score` layer is doing **almost all of the real
  detection work on SMPP** — with no confirmed-clean supervised signal on
  that side, catching spam the rule engine hasn't already flagged is
  squarely the anomaly/near-duplicate layer's job, not the supervised
  layer's. This directly reinforces the architectural split above, not
  just as a design preference but as what the actual data forces.
- `SW_*` whitelist hits still vastly outnumber real rule verdicts on SMPP
  (~1000:1, per `CLAUDE.md`) — confirming that the unlabelled pool isn't a
  small edge case to clean up later, it's the bulk of the traffic and the
  primary product of this whole prototype.

---

## Repo layout & current build status

```
spam-detection-prototype/
├── common/schemas.py            # canonical column/dtype contract — DONE
├── config/settings.py           # tunables (whitelist prefix, op-4 filter, window sizes)
├── ingestion/
│   ├── base.py                  # SourceHandlers(clean, map_to_canonical) contract
│   ├── dcs_codecs.py            # DCS→text codec dispatch, shared SMPP/SS7
│   ├── smpp.py                  # SMPP: op-4 filter, UDH strip, DCS decode  — DONE
│   ├── ss7.py                   # SS7: MO/MT_request filter, SRI merge     — DONE
│   └── run_ingest.py            # file-by-file driver, SOURCES registry
├── labels/rule_labels.py        # rule_evaluated / rule_flagged derivation — DONE
├── features/
│   ├── message_reassembly.py    # multipart parts → one row/message           — DONE
│   ├── behavioral.py            # per-MESSAGE point-in-time velocity/repeat   — DONE
│   │                             # features, for supervised/unsupervised training
│   └── behavioral_snapshot.py   # per-SENDER current-state snapshot,          — DONE
│                                 # feeds the Feast online store (see below)
├── feature_repo/
│   ├── feature_store.yaml       # Feast config — local SQLite registry + online store
│   └── definitions.py           # entity, FeatureView, on-demand feature view — DONE
├── scripts/refresh_feast.py     # snapshot → feast apply → feast materialize — DONE
├── models/
│   ├── anomaly/                 # Isolation Forest + FAISS near-dup        — NOT YET BUILT
│   └── rule_pattern/            # LightGBM/CatBoost/XGBoost supervised     — NOT YET BUILT
├── serving/
│   └── feature_lookup.py        # manual online-lookup test script          — DONE
│                                 # (predict.py-style; no FastAPI service yet)
├── pipeline.py                  # orchestrator: ingestion → reassembly → behavioral (per source)
├── tests/                       # pytest, one file per module above
├── notebooks/                   # ad-hoc real-data exploration
├── docs/
│   ├── prototype_plan.md                        # this project's scope
│   ├── sms_spam_technical_architecture_plan.md  # full production design
│   └── feature_catalog.md                       # what's servable from Feast, feature by feature
└── data/                        # gitignored — raw/ (real CDRs) + processed/
                                  # (including processed/feast_sources/, the snapshot parquet)
```

Run the pipeline so far:

```bash
python pipeline.py                                  # default: data/raw -> data/processed
python pipeline.py --raw_dir data/raw --out_dir data/processed
python -m ingestion.run_ingest --source SMPP         # one source only
python -m features.message_reassembly \
    --features_dir data/processed/SS7/features \
    --labels_dir   data/processed/SS7/labels \
    --out_path     data/processed/SS7/messages.csv
python scripts/refresh_feast.py                      # rebuild sender snapshot + Feast online store
python serving/feature_lookup.py \
    --sender_id "SMPP|66688" --candidate_text "WIN A PRIZE NOW"
```

`pipeline.py` runs three stages per source: **ingestion** (raw CDR →
canonical features + labels, two separate CSVs per input file — see
[`ingestion/run_ingest.py`](ingestion/run_ingest.py)'s docstring for why
features and label-source columns are physically kept in separate files),
**message reassembly** (multipart SMS parts → one row per logical message
— has to run as a global pass across all hourly files, not per-file,
since a message's parts can straddle an hour boundary), and **behavioral
features** (sender velocity/repeat-content, point-in-time correct — see
[`features/behavioral.py`](features/behavioral.py)). `scripts/
refresh_feast.py` is a separate, explicit step (not part of `pipeline.py`)
since it serves a different purpose: `pipeline.py` produces *training*
data, `refresh_feast.py` refreshes the *online-serving* snapshot — see
[Feature serving](#feature-serving-feast) below. Both models and the
FastAPI service are next — see [Roadmap](#roadmap) below.

### Feature serving (Feast)

See [`docs/feature_catalog.md`](docs/feature_catalog.md) for the full
name/type/meaning table of every feature currently servable from the
store — this section is the how, that doc is the what.

The behavioral features above answer "what does this sender's history
look like as of this training row" — useful for training, useless for a
live inference request, which needs an answer in milliseconds, not a
full-history rescan. Feast closes that gap:

- [`features/behavioral_snapshot.py`](features/behavioral_snapshot.py)
  computes a **per-sender current-state snapshot** (as of "now", not
  per-message) — a genuinely different, simpler computation than
  `behavioral.py`'s per-row training features, not a repackaging of the
  same output.
- [`feature_repo/definitions.py`](feature_repo/definitions.py) declares
  one Feast entity (`sender_id` = `source|originator`), one `FeatureView`
  for the three features that are pure sender-state, and one **on-demand
  feature view** for the fourth (`sender_repeat_content_ratio_1hr`) —
  that one needs the *incoming* message's own text, which doesn't exist
  until the actual request, so it can't be precomputed like the other
  three; Feast combines a stored top-K recent-text-frequency feature with
  the request's `candidate_text` at lookup time.
- [`scripts/refresh_feast.py`](scripts/refresh_feast.py) runs
  snapshot → `feast apply` → `feast materialize` as one command.
- [`serving/feature_lookup.py`](serving/feature_lookup.py) is the manual
  test entry point — the same call a future FastAPI endpoint will make
  per request.

**Freshness is bounded by refresh cadence, not by any Feast TTL enforced
at read time** — the online store always serves whatever was last
materialized; re-run `refresh_feast.py` on whatever cadence the target
staleness tolerance requires (matches `CLAUDE.md`'s "batch/synchronous,
not streaming" design for this prototype).

---

## Modeling plan

### Text embeddings
| Model | Role |
|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | **Prototype default**, switched from `all-MiniLM-L6-v2` after checking the real data: a Malay-marker heuristic over 5,000 sampled messages per source found ~10-11% genuine Bahasa Malaysia/mixed content (this is Malaysian-market SMS — CIMB/OCBC/UOB/AirAsia, etc.) that an English-only model would embed poorly. Purpose-built for sentence similarity (same training objective as the model it replaces, just multilingual), ~2x the CPU cost of `all-MiniLM-L6-v2` — verified working with a real EN/Malay-paraphrase similarity check (0.93 similarity for a true paraphrase pair vs. 0.11 for an unrelated message) before committing to the swap. |
| `all-MiniLM-L6-v2` | Original prototype pick — smallest footprint/lowest latency of the candidates, but English-only; superseded once the real language mix was checked. |
| Distil-mBERT (raw `distilbert-base-multilingual-cased`) | **Not recommended without extra work.** Trained for masked-word prediction, not sentence similarity — using it directly for embeddings tends to underperform a purpose-built model like the one above; would need fine-tuning on a similarity objective first. `distiluse-base-multilingual-cased-v2` (same base, already sentence-transformers-tuned) is the fairer version of this candidate if multilingual DistilBERT coverage is wanted later. |
| XLM-R | Best multilingual coverage of the four, but needs more infra than a prototype warrants — later-phase candidate only. |

**Built:** [`features/text_embeddings.py`](features/text_embeddings.py)
— dedup-before-encode (real spam is repetitive; one busy sender alone
repeats a single text hundreds of times per hour, per
[`docs/feature_catalog.md`](docs/feature_catalog.md) — encoding once per
*distinct* text rather than once per row is a large, real speedup, not a
micro-optimization) plus a `--sample_n` flag for prototype-scale runs —
see below for why the full dataset isn't embedded outright. The earlier
`huggingface.co`-unreachable blocker noted in `CLAUDE.md` no longer
applies as of 2026-08-24 — verified live, weights download and encode
correctly in this environment now.

**A real cost, not a theoretical one — paid off, not just theorized:**
encoding is a transformer forward pass per distinct text (~14ms/text on
this CPU for the original English-only model, measured), not the
microseconds-per-row cost of every earlier pipeline stage. At the real
dedup ratio (74.2% of SMPP's 5.5M rows and 51.3% of SS7's 2.7M rows are
actually distinct — spam templates vary by embedded OTP/amount/
reference-number even when otherwise identical, so exact-match dedup
alone doesn't collapse the corpus much), a full-dataset run was estimated
at ~21hr. That run has since completed — both FAISS and Isolation Forest
now train on the full corpus, not a sample (see
`docs/experiments/anomaly.md`'s "Current scale"). `--sample_n` (seeded
for reproducibility) remains available for fast local iteration, but is
opt-in now, not the default path.

### Supervised — `rule_pattern_score`
| Model | Role |
|---|---|
| **LightGBM** | **Prototype default.** Already proven end-to-end in the reference fraud-detection build this project's tooling is carried over from (`train.py` CLI pattern, MLflow tracking, `compare_versions.py` champion/challenger). |
| XGBoost, CatBoost | Champion/challenger benchmarks once LightGBM is working — not a day-one requirement. CatBoost is worth prioritizing over XGBoost here specifically because of native categorical handling for high-cardinality fields like `originator`/`system_id`/`virtual_gt` without manual encoding. |

Trained **only** on `rule_evaluated == True` rows, labelled via
`rule_flagged` (`fraud_type == "spam"` specifically, not generic
`decision == 1` — see [`labels/rule_labels.py`](labels/rule_labels.py) for
why that distinction matters on real SS7 data, where ~7.5% of
`decision==1` rows are non-spam fraud types). Features: canonical schema
fields + behavioral features + `source` — deliberately **not**
embeddings (tree splits don't use dense 384-dim vectors well; that's
Isolation Forest's job), which also means this model isn't bottlenecked
by the embedding sample the way Isolation Forest is.

**Built:** [`models/rule_pattern/train.py`](models/rule_pattern/train.py)
(+ [`data.py`](models/rule_pattern/data.py)). Trains on the full real
`rule_evaluated` pool (352,655 rows — 2,693 SMPP + 349,962 SS7), not
sample-restricted. Test PR-AUC 0.999 (SS7/overall; SMPP skipped, zero
confirmed-clean labels) — expected, not a strong claim: it's
reconstructing the rule engine's own decision boundary from the same
signal the rules use, reported honestly as a known-pattern detector,
not novel-spam detection.

### Unsupervised — `anomaly_score`
| Component | Role |
|---|---|
| **Isolation Forest** (scikit-learn) | Anomaly scoring over [MiniLM embedding + behavioral features] jointly — catches structurally similar-but-varied spam (e.g. templated phishing with randomized tokens), not just exact repeats. Real-time-viable, zero labels needed. |
| **FAISS** near-duplicate index | Rolling, point-in-time-correct similarity search over recent message embeddings, with zero ML required to compute the match itself. |

Trained with **zero labels**, on the **full** traffic stream (not just
`rule_evaluated == False` rows — the model needs to learn what normal
traffic looks like broadly, then score everything, including the labelled
pool, for the disagreement signal described above to mean anything).

**Built:** [`features/faiss_index.py`](features/faiss_index.py) — consumes
`text_embeddings.py`'s output directly, no new embedding work. Produces
**three features × two windows**, not one flat count, because a raw
match count alone can't tell a coordinated spam blast from a bank
sending one OTP template to thousands of real customers in an hour —
both produce a high count: `near_dup_match_count` (raw count, point-in-time
correct — only matches strictly *before* the message being scored count)
and `near_dup_distinct_senders` (how many *different* sender IDs are
behind those matches — the actual disambiguator: one sender repeating
itself vs. the same template spread across many sender IDs), plus
`near_dup_max_similarity`. **Two windows** (1hr + 24hr), because a
paced-out campaign — a few near-dup sends every couple of hours, never
clustering within one hour — evades a single short window entirely:
verified on real SMPP sample data, 17.3% of messages (6,907/40,000) show
zero 1hr matches but real 24hr matches. Neither window is a complete
defense against an arbitrarily patient adversary; this feature also
never has to work alone — Isolation Forest sees it jointly with
embeddings and behavioral history.

Also verified: the highest-match-count message (362 matches in one hour)
came from a single sender ID (a legit bank promo blast, correctly not
flagged as coordinated); the highest-distinct-senders message (4
different sender IDs, same OTP template) is the kind of pattern worth a
human look, not an automatic verdict — this feature surfaces signal, it
doesn't decide.

**Scales independently of total corpus size** via
`compute_near_dup_features_chunked()` — processes bounded, sequential
time-chunks (each with a correctly-sized lookback buffer, so no message
loses accuracy near a chunk boundary) rather than holding a growing
historical corpus in one FAISS index at once; verified identical to the
unchunked result in tests. Kept the index type as exact `IndexFlatIP`
deliberately, not IVF — chunking already keeps each index small, which
shrinks IVF's speed benefit below its recall cost at this scale.

**Deferred, not forgotten:** SIM-farming (one physical SIM/IMSI cycling
through many apparent phone numbers) is a *different* signal from
near-duplicate content matching — an identity-linkage problem, not a
text-similarity one — and doesn't belong in this module. Confirmed
present in real SS7 data (1,541 IMSIs already show more than one
distinct originator). Belongs as a new SS7-only behavioral feature keyed
by `imsi`, not yet built.

### Explainability
- **LIME** — both model types. For Isolation Forest, treat the anomaly
  score itself as the "prediction" being explained — same technique,
  applied to a different model type, no separate tooling needed.
- **SHAP** — added alongside LIME; not yet decided which is primary vs.
  supplementary for this project, revisit once both are wired in.
- Native LightGBM feature contributions are the **real-time** explanation
  path (`reason_codes`, cheap enough for the inline serving path); LIME/
  SHAP are for offline/analyst deep-dives, not the latency-critical path.

### Evaluation
**PR-AUC and log loss are primary, not accuracy/ROC-AUC** — spam is a
minority class on both sources (see the imbalance numbers above), and even
more so once behavioral negatives are added. Track **precision at a fixed
recall** (e.g. "at 90% recall, what's precision") as the number that
actually maps to a block/allow business decision. **Evaluate three ways
every time: overall, SMPP-only, SS7-only** — an aggregate metric can hide
one source-segment performing badly, especially given how differently
SMPP and SS7's labelled pools look today (see
[Data reality check](#data-reality-check-what-the-real-data-actually-looks-like)).

---

## Tech stack

| Layer | Choice |
|---|---|
| Data manipulation | pandas, numpy |
| Text embedding | sentence-transformers, `all-MiniLM-L6-v2` |
| Near-duplicate index | FAISS |
| Anomaly detection | scikit-learn Isolation Forest |
| Supervised classifier | LightGBM (XGBoost/CatBoost as challengers) |
| Experiment tracking / registry | MLflow, SQLite backend (`sqlite:///mlflow.db`) |
| Explainability | LIME, SHAP |
| Inference service | FastAPI (not yet built) |
| Feature store | Feast, local SQLite registry + online store (`feature_repo/`) — wired in for the sender-behavioral features (see [Feature serving](#feature-serving-feast) above). Deliberately a separate refresh path (`scripts/refresh_feast.py`) from `pipeline.py`, not merged into it — training data and the online-serving snapshot are different computations, not the same output reused. |

Offline and online ML platform are **the same environment** in this
prototype (one process, no separate training/serving infra split) — a
deliberate simplification for iteration speed, not the target production
design. No Kafka, no Redis — batch/synchronous feature computation, local
SQLite, matches prototype scale (not the 3,000 TPS production target,
which this phase explicitly doesn't need to hit).

---

## Setup

```bash
pip install --no-cache-dir -r requirements.txt   # disk space has been tight in
                                                   # this sandbox before — avoid
                                                   # re-triggering large caches
pytest                                            # run the test suite
python pipeline.py                                # ingest + reassemble real CDRs
```

Real raw/processed CDR data lives under `data/` and is gitignored (large,
real traffic — never committed).

---

## Roadmap

In order (per `CLAUDE.md`):

1. **MiniLM text embeddings** — a shared upstream dependency for steps 2
   and 3, not part of either one specifically. FAISS has no language
   understanding on its own — it's nearest-neighbor search over vectors,
   nothing more; the vectors have to come from somewhere, and that's this
   step. Isolation Forest also needs it, per the
   [modeling plan](#unsupervised--anomaly_score) above (`[MiniLM
   embedding + behavioral features]`, jointly). Build once, consume twice.
2. FAISS near-duplicate index (consumes step 1's embeddings)
3. Isolation Forest training script (unsupervised layer; consumes step
   1's embeddings + the behavioral features already built)
4. LightGBM training script (supervised layer, rule-labelled data)
5. FastAPI service combining both, dual-score response shape
6. LIME wiring for both model types

Behavioral features (training) and the Feast online store (serving) are
both done — see [Feature serving](#feature-serving-feast) above. FastAPI
(step 5) is what will actually call `serving/feature_lookup.py`'s
`get_sender_features()` per request instead of it being a manual test
script.

## Known blockers

- ~~`huggingface.co` unreachable from this sandbox~~ — resolved as of
  2026-08-24, verified live (see `CLAUDE.md`'s "Known blockers" for
  detail).
- Disk space has been tight in this sandbox before — use
  `pip install --no-cache-dir`, avoid re-triggering large caches.
