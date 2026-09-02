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

**Current scale**: trains on whatever `features/text_embeddings.py`'s
`--sample_n` sample covers (40k SMPP / 20k SS7 as of writing) - the join
in `models/anomaly/data.py` is an INNER join against the embeddings/FAISS
output, which currently only covers that sampled subset (the full
8.2M-row encode is a ~21hr CPU job, see `docs/architecture.md`).

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

**Real result**: PR-AUC 0.857 (SS7) - modestly above the 0.804 naive
baseline (spam is the majority of the rule-evaluated subset, an artifact
of which messages rules bother to evaluate, not the true traffic-wide
spam rate - see `docs/experiments/rule_pattern.md`). SMPP PR-AUC is
skipped, not silently computed wrong: SMPP has zero confirmed-clean
rule-evaluated rows (2,693/2,693 flagged), so PR-AUC is mathematically
undefined with only one class present. Log loss is not trustworthy yet -
`anomaly_score` isn't a calibrated probability; would need Platt/
isotonic calibration first.

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
