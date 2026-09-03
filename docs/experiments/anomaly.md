# Experiment: Isolation Forest (`anomaly_score`)

Code: `models/anomaly/train.py` (+ `models/anomaly/data.py` for the
join/preprocessing). See `docs/ml/modeling.md` for why this exists
alongside `rule_pattern_score`, and `docs/experiments/faiss.md` for the
near-dup features it consumes.

## What it trains on

Isolation Forest over `[MiniLM embedding + behavioral features + FAISS
near-dup features]` jointly. Zero labels used for training, on purpose -
this is the layer meant to catch spam the rule engine has never encoded.

NOT wired into `pipeline.py`: training is a deliberate, versioned action,
not a deterministic feature-computation step. Run by hand:

```
python -m models.anomaly.train
python -m models.anomaly.train --n_estimators 200 --contamination 0.02
```

**Current scale**: trains on the FULL (non-sampled) dataset now - the
full-dataset `features/text_embeddings.py` encode (previously a ~21hr CPU
job blocking full-scale training, see `docs/architecture.md`) has since
completed. The join in `models/anomaly/data.py` is an INNER join against
the embeddings/FAISS output, which now covers the full row counts: SMPP
5,505,921 rows, SS7 2,742,301 rows (verified this session -
`embeddings_sample_info.txt`, which `features/text_embeddings.py` writes
only for a `--sample_n` run, is no longer present in
`data/processed/{SMPP,SS7}/`, and `embeddings.npy` file sizes match those
full row counts exactly). An earlier version of this doc described a
sampled 40k SMPP / 20k SS7 subset - that framing no longer applies, not
just the specific numbers.

## Score convention

Output is `-model.decision_function(X)`, not `.predict()`'s binary label.
sklearn's `decision_function` is HIGHER for normal points, LOWER (more
negative) for anomalies - negated here so higher = more anomalous,
matching this project's `anomaly_score` convention. `contamination` is
left at scikit-learn's default (`"auto"`) rather than tuned, since it
only governs `.predict()`'s binary cutoff, which this project's
architecture doesn't use - the confidence-gated block/allow decision
happens later, at serving time, not baked into training.

## Preprocessing: two real fixes, both measured

1. **log1p on heavy-tailed count features** - necessary, not optional:
   real observed ranges include `sender_msgs_last_1hr` up to 16,971 and
   `near_dup_match_count_24hr` up to 381, against embedding dimensions
   confined to roughly `[-1, 1]`. Ratios (already 0-1) and similarity
   scores (already ~0-1) are left alone.
2. **PCA on the 384 embedding dimensions down to 30
   (`N_EMBEDDING_COMPONENTS`)**, applied BEFORE combining with the 12
   hand-built behavioral+near-dup features, via a `ColumnTransformer` so
   it only touches the embedding columns - the hand-built features pass
   through unchanged into the same final joint `StandardScaler`.

### The ablation that justified PCA

`scripts/check_embedding_dominance.py` - one-off diagnostic, not
permanent pipeline code, logged to its own MLflow experiment
(`anomaly_score_diagnostics`) so it's never mistaken for a real candidate
model. Method: train three Isolation Forests, same hyperparameters, on
(a) everything (the real feature matrix `train.py` actually uses),
(b) behavioral + near-dup + source only (no embeddings), (c) embeddings
only. Compare (a)'s scores against (b) and (c) via Spearman rank
correlation - if (a) tracks (c) much more closely than (b), the
embeddings are dominating the joint model.

**Real measured result**: before PCA, the joint model's ranking
correlated **0.808 with embeddings-only** but only **0.326 with
behavioral-only** - the 384-vs-12 dimension imbalance was genuinely
drowning out the hand-built features, not just a theoretical risk. After
PCA(30): **0.654 vs 0.596** - much more balanced. `N_EMBEDDING_COMPONENTS
= 30` is a starting point (not tuned against a real target
explained-variance threshold); `build_feature_matrix()` prints the actual
retained variance at this component count every run, so the number stays
honest rather than a one-time guess nobody checks again.

## Two more preprocessing fixes: feature-encoding bugs, anomaly-model-only

Found and fixed this session, both in two "Tier 0" behavioral features
added this session (meant to help distinguish e.g. legitimate bulk
senders from spam campaigns). Full technical detail and the exact
real-data reasoning live as comments in `models/anomaly/data.py` itself -
this section is a summary; read there for the precise mechanism.

**a. `sender_age_days` narrow-range domination.** In this prototype's
fixed ~2-day CDR sample, `sender_age_days` only ever ranges 0.0-2.0 -
Isolation Forest could trivially isolate on "brand new sender" regardless
of actual content/behavior, since the raw feature lived in a tiny range
that dominated the isolation-path split choices. Measured:
precision@top-0.1%-by-`anomaly_score` (see precision@K below) was
**0.081** - WORSE than the ~0.107 naive baseline - before the fix. Fixed
by bucketing into coarse, real-world-meaningful edges
(`SENDER_AGE_BUCKET_EDGES_DAYS` / `SENDER_AGE_BUCKET_LABELS` in
`models/anomaly/data.py`: <1hr, 1hr-1day, 1day-7day, 7day-30day, >=30day)
instead of passing the raw day-count. After fixing: precision@top-0.1%
rose to **0.190**.

**b. `sender_recipient_diversity_ratio_5min`/`_1hr` small-sample noise.**
Also a new Tier 0 feature this session. This ratio (unique destinations /
message count in a window) is trivially 1.0 for a sender with just 1
message in the window, regardless of any real behavioral diversity -
measured top-0.1%-by-`anomaly_score` mean of 0.718 vs the overall MEDIAN
of 0.002 (~48x enrichment - the model was substantially keying off this
noise). Fixed by gating on the underlying message count
(`SENDER_DIVERSITY_MIN_MSGS = 3` in `models/anomaly/data.py`) - below
that many messages in the window, the ratio is treated as unknown
(NaN -> `fillna(0)` + a paired `_known` indicator, the same pattern
already used elsewhere in that file for `sender_velocity_zscore_5min`/
IMSI), not passed raw. After fixing: precision@top-0.1% rose further to
**0.204**.

**Full real SS7 precision@top-K% progression** across both fixes (all
from real `isolation_forest_SS7` MLflow runs this session, at the same
top-K cutoffs `models/anomaly/cluster_discovery.py`'s
`--anomaly_percentile` operates at):

```
               raw (before)   age bucketed   + diversity gated   naive baseline
top 0.1%:      0.081          0.190          0.204               ~0.107
top 0.5%:      0.117          0.145          0.149
top 1.0%:      0.130          0.156          0.166
top 5.0%:      0.299*         0.136          0.149
```
*the 0.299 top-5% figure before fixing is itself inflated/misleading -
spurious noise from the narrow `sender_age_days` range spread across a
wider band, not a real high-water mark. Don't read this as "was better
before the fix."

**Why precision@K, not just PR-AUC, is what actually caught this**:
PR-AUC alone only showed a DECLINE across this fix work (0.183 -> 0.166
-> 0.164, full SS7 run - see "Real result" below) - taken alone, it looks
like the fixes made things worse. Precision@top-0.1% (the operationally
relevant cutoff - see `models/anomaly/cluster_discovery.py`'s own
top-0.5%-to-0.1% real operating range) tells the opposite, correct
story: it rose from WORSE-than-random (0.081, below the ~0.107 naive
baseline) to a real ~1.9x baseline lift (0.204). The PR-AUC decrease is
an expected side effect of removing inflated-noise-driven ranking area at
coarser thresholds, not a real regression. This divergence is exactly why
`precision_at_k`/`precision_at_k_percentiles`/`evaluate_precision_at_k`
(`models/metrics.py`) were added this session, and why precision@K - not
PR-AUC alone - is what to trust for this specific model going forward.

**Scope: anomaly-model-only, not project-wide.** These two fixes live in
`models/anomaly/data.py`'s `build_combined_frame()` /
`build_feature_matrix()` path. `rule_pattern_score` (LightGBM) still
receives `sender_age_days` and the diversity ratios RAW/unbucketed - see
`models/rule_pattern/data.py`'s own separate `_base_feature_frame()`,
which never calls `models/anomaly/data.py`'s `build_combined_frame()`.
This is a deliberate scoping decision, not an oversight: tree-based
LightGBM splits handle a narrow-range/small-sample feature differently
than Isolation Forest's isolation-path mechanism, and this failure mode
was only established empirically for the anomaly model - it hasn't been
tested for, and isn't assumed to apply to, `rule_pattern_score`.

## Evaluation

Trained with zero labels, but NOT evaluated with zero labels - the real
`rule_evaluated`/`rule_flagged` labels (the same ones LightGBM trains on)
are used purely as a validation set, never fed into training.
`evaluate_against_rule_labels()` computes real PR-AUC and log loss
(`models/metrics.py`, shared with `models/rule_pattern/train.py`),
checking whether `anomaly_score` actually ranks real rule-confirmed spam
above rule-confirmed clean. Evaluated overall AND per source (SMPP/SS7
look very different - see below).

**Be honest about what this metric does and doesn't prove**: it measures
agreement with patterns the RULE ENGINE ALREADY KNOWS - the exact
opposite of this layer's real purpose (catching spam the rules can't see,
on the unlabelled majority). There is no way to formally evaluate THAT
without labels, which is precisely why this layer exists in the first
place. Treat a good score here as a floor-level sanity check, not proof
of novel-spam detection.

### PR-AUC alone hid a real problem - precision@K is why it exists too

`models/metrics.py` also has `precision_at_k()` /
`precision_at_k_percentiles()` / `evaluate_precision_at_k()`, added this
session specifically because PR-AUC alone hid the feature-encoding bugs
described above. PR-AUC integrates over EVERY possible score threshold,
most of which this project never actually operates at - only the extreme
top of a ranking ever gets acted on in practice (see
`models/anomaly/cluster_discovery.py`'s `--anomaly_percentile`, the real
operating range this project uses - default top 10%, with the tool's own
guidance pointing toward 99-99.9, i.e. top 0.1-1%, at full scale).
`precision_at_k_percentiles()` answers the more operationally honest
question instead: of the top N% ranked by `anomaly_score`
(0.1/0.5/1.0/5.0% by default), what fraction are really `rule_flagged`?
Same single-class-style guard as `pr_auc_and_log_loss()` - a K outside
`[1, n]` returns `None` (skip), not a fabricated number.
`evaluate_precision_at_k()` runs it overall and per-source, the same
convention as `evaluate_overall_and_per_source()`.

**Real result (full-scale, current)**: SS7 PR-AUC **0.164** (overall and
SS7-only - single-source run) against a naive baseline of **~0.107**
(SS7's real rule-evaluated positive rate, 284,073/2,654,369 flagged - see
`docs/experiments/rule_pattern.md`'s real label composition). That's a
real but modest lift (~1.5x baseline), and it is LOWER than the 0.857
figure this doc previously reported. That old number is stale on two
counts, not one: it was measured on the sample-scale run (see "Current
scale" above - training is full-scale now), and it was compared against
a stale 0.804 naive baseline computed from the old, wrong-direction
"spam is the majority" composition. With the real composition (spam is a
MINORITY of the rule-evaluated pool, ~10.7% for SS7 - see
`docs/experiments/rule_pattern.md`), a naive baseline near the true
positive rate is what "no signal" actually looks like here, so 0.164 vs
~0.107 is the honest comparison, not 0.857 vs 0.804. This is NOT a
regression from the feature-encoding fixes above - PR-AUC actually
decreased slightly across that fix work too (0.183 -> 0.166 -> 0.164) for
the reasons explained above (removing inflated-noise-driven ranking area,
not losing real signal) - precision@K, not this number, is the metric
that shows the fixes actually helped.

SMPP PR-AUC is no longer skipped, unlike this doc's earlier claim ("SMPP
has zero confirmed-clean rule-evaluated rows... mathematically
undefined"). Per `docs/experiments/rule_pattern.md`'s real label
composition, SMPP now has a real confirmed-clean population (98.1% of
139,546 `rule_evaluated` rows), so `models/metrics.py`'s
single-class-skip guard no longer triggers for SMPP. Real measured
result, from the most recent `isolation_forest_SMPP` MLflow run
(`sqlite:///mlflow.db`, run `a52a25c0`, with both feature-encoding fixes
applied): SMPP PR-AUC **0.074** against SMPP's own naive baseline of
**~0.019** (2,692/139,546 flagged) - roughly a 3.8x lift, lower in
absolute terms than SS7's 0.164 but a comparably real signal relative to
SMPP's much sparser positive rate. Same run's SMPP precision@top-K%:
0.1% -> 0.057, 0.5% -> 0.136, 1.0% -> 0.138, 5.0% -> 0.084 (all above the
~0.019 SMPP baseline). Whether/how this SMPP-specific result should
change how the model is read or tuned is an open question, not resolved
here.

Log loss is not trustworthy yet - `anomaly_score` isn't a calibrated
probability; would need Platt/isotonic calibration first.

A simpler `plausibility_check()` (mean anomaly_score by group) is also
kept - weaker than PR-AUC (doesn't account for the full score
distribution), but cheap and easy to sanity-eyeball alongside the real
metric.

## Output

Score-per-message, one parquet file per source
(`data/processed/<SOURCE>/anomaly_scores.parquet`), with both the raw
`anomaly_score` (the actual, sign-flipped `decision_function` output) and
a min-max normalized `anomaly_score_normalized` (0-1, easy to eyeball)
alongside it.

## What to do with the output: clustering into candidate fraud types

`anomaly_score` alone ranks anomalies, it doesn't group or type them.
See `docs/experiments/anomaly_clustering.md` for the DBSCAN-based
clustering workflow (`models/anomaly/cluster_discovery.py`) that turns
this ranked list into hand-labelable candidate fraud-type clusters — the
concrete next step for this model's output, not a hypothetical one.
