# ML modeling strategy

The overall two-score design and why it's shaped this way. For the
empirical tuning history / evidence behind each individual piece, see
`docs/experiments/faiss.md`, `docs/experiments/anomaly.md`, and
`docs/experiments/rule_pattern.md`. For system/pipeline-level decisions
(feature contract, Feast, environment), see `docs/architecture.md`.

## One model to start, not two

`source` (`SMPP` | `SS7`) is passed into both models as a feature, not
used to route to separate per-source models. Only split into per-source
models if per-segment evaluation (SS7-only vs SMPP-only PR-AUC/log loss)
proves the shared model underperforms on one side. No such evidence
exists yet - both current models evaluate SS7 and SMPP separately (see
the experiments docs) and neither has shown a reason to split.

## Two separate scores, not one blended score

- **`rule_pattern_score`** (supervised LightGBM, trained on rule-engine-
  labelled data) - a faster/cheaper re-implementation of known rules.
  Do NOT claim this generalizes to novel spam.
- **`anomaly_score`** (unsupervised Isolation Forest + FAISS near-duplicate
  matching) - trained with zero labels. This is the layer meant to catch
  spam the rules don't already know about.
- Cases where these two disagree are the most valuable output - surface
  them, don't average them away.

### Inference-time scope is narrower than training-time scope

This matters for FastAPI later. A message the rule engine already
confidently resolved (`rule_evaluated == True` - genuinely flagged, or
genuinely evaluated-clean) does NOT get sent to either model at real-time
inference - its verdict is already final, nothing to add. Both models
only run on live traffic *because* the rule engine didn't confidently
resolve it (`SW_*` whitelist bypass, or genuinely untouched) - that's the
whole reason this system exists.

This does NOT change training scope:
- `anomaly_score` still trains on the FULL traffic stream (needs to see
  confidently-normal traffic to learn what "normal" looks like).
- `rule_pattern_score` still trains only on `rule_evaluated == True` rows.

Don't build the future FastAPI endpoint to blanket-score every request
without checking `rule_evaluated` first - see README.md's "problem,
precisely" section for the corrected inference-flow diagram.

## Shared upstream dependency: text embeddings

`features/text_embeddings.py` (MiniLM `all-MiniLM-L6-v2`) is built once
and consumed twice - by FAISS (near-dup search over the raw vectors) and
by Isolation Forest (scores `[embedding + behavioral features]` jointly).
FAISS has no language understanding of its own (it's nearest-neighbor
search over vectors, nothing more); building the embedding step once and
having both consumers read the same output avoids two independent
encoding passes over the same text.

LightGBM (`rule_pattern_score`) deliberately does NOT consume embeddings
in its default path - see `docs/experiments/rule_pattern.md` for why, and
for the real evidence (`text_length` feature importance) that this may
be worth revisiting once a full-dataset embeddings run exists.

## Shared evaluation convention

Both models are evaluated with the same helper, `models/metrics.py`:
PR-AUC + log loss (not accuracy/ROC-AUC alone - spam is the minority
class overall), computed overall AND per-source, with an explicit
single-class-skip guard rather than a silently-wrong number - SMPP has
zero confirmed-clean rule-evaluated rows (verified against the full raw
dataset), so SMPP-only PR-AUC is mathematically undefined and is skipped
outright in both models' output.

## MLflow conventions

- Tracking URI: `sqlite:///mlflow.db` for every experiment below.
- Real candidate models get their own experiment name: `isolation_forest`
  (models/anomaly/train.py::MLFLOW_EXPERIMENT_NAME), `light_gbm`
  (models/rule_pattern/train.py::MLFLOW_EXPERIMENT_NAME).
- A variant that isn't ready to be treated as a real candidate gets a
  SEPARATE experiment name, on purpose - any `--with_embeddings`/
  `--with_tfidf` LightGBM run routes to `rule_pattern_score_experimental`
  (not yet a meaningful comparison, see `docs/experiments/rule_pattern.md`),
  and a `--sources SMPP`/`--sources SS7`-restricted run of either model
  gets its own source-suffixed experiment (e.g. `isolation_forest_SMPP`).
  This keeps early/throwaway/split-model runs from ever being mistaken
  for the real combined baseline in the MLflow UI.
- Champion/challenger promotion is explicit in code, not left to MLflow
  to decide (same convention as the earlier `mlflow_demo/` project's
  `compare_versions.py`) - not yet needed here since there's only one
  trained version of each model so far.

## Status

Completed: MiniLM embeddings, FAISS near-dup features, Isolation Forest
training, LightGBM training, FastAPI service (`serving/app.py`) computing
BOTH scores per request - see the experiments docs for each model, and
`docs/architecture.md`'s file-layout section for the serving/ modules.

anomaly_score at serving time reuses each source's existing HISTORICAL
corpus (embeddings.npy) as the near-dup comparison set, live-queried
per request and windowed against the request's own declared timestamp -
see `serving/anomaly_scoring.py`'s module docstring for why (no
streaming corpus exists in this prototype) and the same staleness caveat
already accepted for Feast's batch-refreshed behavioral features.
anomaly_score is surfaced on the response for visibility only; it does
NOT gate the FRAUD/NOT_FRAUD decision yet (see "Two separate scores, not
one blended score" above).

SHAP explainability is built for both models (`models/rule_pattern/
explain.py`, `models/anomaly/explain.py`) - offline scripts run against an
already-trained MLflow run_id (default: latest), not something
`serving/app.py` calls per request. Both rebuild the exact data that run
trained/evaluated on from its own logged params, then run
`shap.TreeExplainer` (exact for tree ensembles) to get per-row local
contributions and dataset-wide global importance. The anomaly model also
gets LIME, on a small deliberately-chosen set of instances (top-anomaly +
random) - complementary to SHAP, not redundant: the 384-dim embedding is
PCA'd to ~30 anonymous components before the model sees it, so SHAP can
only report one aggregate "content_embedding" bucket, not
interpretable per-dimension attributions; LIME's local surrogate explains
the structured (behavioral/near-dup) features in their original units for
those chosen rows instead. See each script's module docstring for the
full reasoning.

Real-time SHAP is now wired into `serving/app.py`'s `/v1/score`
(`serving/scoring.py::explain_rule_pattern`): a `shap.TreeExplainer`
cached once per champion load, run inline per request - safe to do
inline only because TreeExplainer is exact/polynomial-time with no
background dataset needed, unlike LIME (which stays offline-only, see
`models/anomaly/explain.py`'s per-instance cost). `_reason_codes()`
derives its codes from these real contributions instead of the old fixed
thresholds, computed only for a FRAUD prediction (cost control); the raw
top-K contributions are also surfaced on `FraudPredictionResult.
feature_contributions`, an additive field alongside `anomaly_score`. A
failure to explain degrades reason_codes/feature_contributions, it never
turns a successful rule_pattern_score into a FAILURE response (same
best-effort convention as anomaly_score).

Not yet built:
1. A full (non-sampled) `text_embeddings.py` run - both FAISS and
   Isolation Forest currently train on the sampled subset
   (`--sample_n`); see `docs/architecture.md`'s "Known blockers" section
   for the real cost (~21hr full run) driving that choice.
