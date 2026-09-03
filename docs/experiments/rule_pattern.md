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

Verified across the FULL raw dataset, not a sample - re-measured this
session against the current `messages_with_behavioral.csv` files:
- **SMPP**: 139,546 `rule_evaluated` rows, 2,692 flagged (1.9%) / 136,854
  confirmed-clean (98.1%).
- **SS7**: 2,654,369 `rule_evaluated` rows, 284,073 flagged (10.7%) /
  2,370,296 confirmed-clean (89.3%).

Both counts grew a lot since an earlier snapshot of this doc (SMPP
2,693 -> 139,546, ~52x; SS7 349,962 -> 2,654,369, ~7.6x) as real full
ingestion completed - and the composition flipped along with the scale,
not just the row counts:

1. **SMPP-only PR-AUC is no longer mathematically undefined.** This doc
   used to say SMPP was 2,693/2,693 flagged (zero confirmed-clean), so
   `models/metrics.py`'s single-class-skip guard always triggered for
   SMPP-only slices. That was true when measured; it isn't anymore - SMPP
   now has a real confirmed-clean population (98.1%), so a real,
   computable SMPP-only PR-AUC exists. (The "Evaluation" section below
   still shows an old "SMPP skipped" result - flagged there too, not left
   standing as if still current.)
2. **Spam is now a MINORITY of this rule-evaluated pool for both
   sources** (SS7 10.7%, SMPP 1.9%) - the OPPOSITE of what this doc used
   to say (spam as the ~85% majority). That "~85%" figure was computed
   from the earlier, much smaller ingestion snapshot - the mechanism it
   was describing wasn't wrong, just the magnitude and direction. The
   underlying point still holds: this composition is an artifact of
   WHICH messages the rule engine bothers to evaluate (the same point
   applies to `anomaly_score`'s validation metric - see
   `docs/experiments/anomaly.md`), not the true traffic-wide spam rate -
   it's just now the usual "spam is rare" imbalance direction rather than
   the inverted one this doc previously described.

**Open question, not resolved here**: this doc previously concluded "no
explicit class-weighting is applied... not obviously warranted" -
reasoned from the OLD (wrong-direction) spam-majority composition. With
the real numbers, both sources are now genuinely imbalanced toward
confirmed-clean (SS7 89.3%, SMPP 98.1%) - the opposite imbalance
direction from what justified that original conclusion. Whether
class-weighting is actually warranted now is a real, open modeling
question that needs someone to actually retrain/evaluate with it - flag
it here rather than silently keeping the stale conclusion or silently
flipping it to the opposite claim without evidence.

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

**Real result**: test PR-AUC 0.999 (SS7/overall) - expected, not
remarkable: it's reconstructing the rule engine's own boundary from the
same signal rules use, not evidence it generalizes to novel spam. SMPP
was skipped when this was last measured (zero confirmed-clean rows at
the time, per the old composition numbers this doc used to show above).
Per the real label composition now, SMPP has a real confirmed-clean
population (98.1% of 139,546 rows), so a real SMPP-only PR-AUC is
computable - this doc doesn't have that number yet, since it needs an
actual re-run of `models.rule_pattern.train --sources SMPP`, not a
fabricated figure here. Flagging the stale "SMPP skipped" claim rather
than leaving it standing as if still accurate.

## `--with_embeddings`: built, not yet useful

Uses `models/rule_pattern/data.py`'s embeddings-aware loader/feature
builder instead of the default ones, reusing
`models/anomaly/data.py`'s `embedding_pca_pipeline()` (StandardScaler +
PCA) for the same dimension-imbalance reason it's needed there - even
though tree splits themselves don't require scaling, 384 raw dims would
still swamp this model's other ~10 features. Logged to a SEPARATE MLflow
experiment (`rule_pattern_score_with_embeddings`) so an early, tiny-sample
run never gets mistaken for a real baseline candidate.

**Now actually usable, not yet tried**: `features/text_embeddings.py`'s
full-dataset (non-sampled) run is done (see `docs/experiments/anomaly.md`'s
"Current scale") - the ~1% overlap problem this section used to describe
(an embedding SAMPLE covering only ~18/2,693 SMPP and ~2,635/349,962 SS7
`rule_evaluated` rows) no longer applies. Re-checked directly against the
current `embeddings_id_map.parquet` files this session: embeddings now
cover 139,546/139,546 (100%) of SMPP's and 2,654,369/2,654,369 (100%) of
SS7's current `rule_evaluated` pool. The comparison this flag exists for
is genuinely runnable with a meaningful result now, not just "one command
away" - it hasn't actually been run/evaluated yet as of this doc edit, so
no result is claimed here.

**Why it's worth doing eventually, not just a nice-to-have**: the
baseline (no-embeddings) model's own feature importances show
`text_length` (the only content-adjacent signal it has) as the single
most important feature by a wide margin - meaning even a crude proxy for
content carries real separating power, so genuine content (real
embeddings, not just character count) would plausibly help more, not be
redundant.
