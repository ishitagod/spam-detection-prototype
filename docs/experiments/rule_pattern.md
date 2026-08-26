# Experiment: LightGBM (`rule_pattern_score`)

Code: `models/rule_pattern/train.py` (+ `data.py`). See
`docs/ml/modeling.md` for why this exists alongside `anomaly_score`, and
`docs/experiments/anomaly.md` for the sibling model sharing
`models/metrics.py`'s evaluation convention.

## What it trains on

LightGBM on `rule_evaluated == True` rows only, labelled by
`rule_flagged` (True = spam, False = confirmed-clean). A faster/cheaper
re-implementation of rules the rule engine already knows. This model can
only ever re-recognize patterns the rules already encode; it is NOT the
layer meant to catch novel spam (`anomaly_score` is - see
`docs/experiments/anomaly.md`).

NOT wired into `pipeline.py`, same reasoning as the anomaly model:
training is a deliberate, versioned action, not a feature-computation
step. Run by hand:

```
python -m models.rule_pattern.train
python -m models.rule_pattern.train --n_estimators 200 --learning_rate 0.05
```

## Real label composition (worth knowing before reading metrics)

Verified across the FULL raw dataset, not a sample:
- **SMPP**: 2,693 `rule_evaluated` rows, ALL flagged - zero
  confirmed-clean.
- **SS7**: 349,962 `rule_evaluated` rows, split 284,073 flagged / 65,889
  confirmed-clean.

Two consequences:
1. SMPP-only PR-AUC is mathematically undefined (one class only) and is
   skipped, not silently computed wrong (`models/metrics.py`).
2. Spam is the MAJORITY of this rule-evaluated pool overall (~85%), not
   the minority - the usual "spam is rare" imbalance framing is backwards
   here. This is an artifact of WHICH messages the rule engine bothers to
   evaluate (the same point applies to `anomaly_score`'s validation
   metric - see `docs/experiments/anomaly.md`), not the true
   traffic-wide spam rate. No explicit class-weighting is applied here
   for that reason - not obviously warranted given the real, measured
   composition, rather than assumed from the generic "imbalanced spam"
   prior.

This is also genuinely a much bigger training pool than the anomaly
model's: LightGBM reads straight from the FULL
`messages_with_behavioral.csv` (every row, every source), unrestricted by
the embedding sample Isolation Forest needs - LightGBM doesn't touch
embeddings by default, so it isn't bottlenecked by it.

## Features - deliberately narrower than Isolation Forest's

- **Behavioral**: the same 4 columns as `models/anomaly/data.py`,
  imported from there rather than duplicated.
- **Canonical**: `dcs`, `text_decode_failed`, plus `text_length` (a cheap
  derived signal, zero extra cost to add). `dcs` can be NaN in real
  data - left as-is deliberately, LightGBM has native missing-value
  handling built in.
- **`source`** (one-hot).
- **NOT scaled**: unlike Isolation Forest, tree-based LightGBM splits are
  scale-invariant - no `StandardScaler` needed.

Excluded on purpose for this first build: `originator`/`destination` -
too high-cardinality to one-hot without real overfitting risk on a pool
this size, and behavioral features already capture originator-level
BEHAVIOR (velocity, repeat content) without needing the raw identity.
CatBoost's native categorical handling is the right place to revisit
this, not a one-hot hack here.

**Label**: `rule_flagged`, restricted to `rule_evaluated == True` - NEVER
`decision == 1` directly (`labels/rule_labels.py` found ~7.5% of SS7's
`decision == 1` rows are non-spam fraud types; `rule_flagged` already
encodes `fraud_type == "spam"` specifically, `decision` alone doesn't).

## Evaluation

Stratified train/test split, PR-AUC + log loss via `models/metrics.py`
(shared with the anomaly model, same single-class-skip guard), evaluated
on BOTH train and test sets - a large train/test gap is the actual
overfitting signal to watch for, given this pool's small-for-SMPP /
imbalanced-for-SS7 shape.

**Real result**: test PR-AUC 0.999 (SS7/overall; SMPP skipped, no
confirmed-clean labels) - expected, not remarkable: it's reconstructing
the rule engine's own boundary from the same signal rules use, not
evidence it generalizes to novel spam.

## `--with_embeddings`: built, not yet useful

Uses `models/rule_pattern/data.py`'s embeddings-aware loader/feature
builder instead of the default ones, reusing
`models/anomaly/data.py`'s `embedding_pca_pipeline()` (StandardScaler +
PCA) for the same dimension-imbalance reason it's needed there - even
though tree splits themselves don't require scaling, 384 raw dims would
still swamp this model's other ~10 features. Logged to a SEPARATE MLflow
experiment (`rule_pattern_score_with_embeddings`) so an early, tiny-sample
run never gets mistaken for a real baseline candidate.

**Not useful today**: the embedding sample only overlaps ~1% of the
`rule_evaluated` pool - as of writing, ~18/2,693 SMPP and ~2,635/349,962
SS7 `rule_evaluated` rows (under 1% either way). This flag exists so the
comparison is one command away once `features/text_embeddings.py`'s
full-dataset run is done, not something to run expecting a meaningful
result right now.

**Why it's worth doing eventually, not just a nice-to-have**: the
baseline (no-embeddings) model's own feature importances show
`text_length` (the only content-adjacent signal it has) as the single
most important feature by a wide margin - meaning even a crude proxy for
content carries real separating power, so genuine content (real
embeddings, not just character count) would plausibly help more, not be
redundant.
