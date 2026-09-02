# From `anomaly_score` to fraud-type labels — clustering workflow

What to actually DO with the Isolation Forest output. See
`docs/experiments/anomaly.md` for how `anomaly_score` itself is trained/
evaluated; this doc picks up from there. See `docs/ml/modeling.md` for
where this sits in the two-score design, and
`docs/sms_spam_technical_architecture_plan.md`'s §4 (Stage A → D) for the
production-scale version of this same loop.

## Why clustering, at all

Isolation Forest gives exactly **one number per message**: how anomalous
it is. It has no concept of grouping similar anomalies together, so it
structurally cannot answer *"anomalous in what way, resembling what other
flagged cases?"* A ranked list of 5,000 anomalous messages is not
actionable on its own — nobody can hand-review 5,000 individual rows. What
you want is groups: "these 40 rows are one flooding-burst pattern," "these
12 are a phishing template with rotating URLs," "this one is a genuine
one-off, unlike anything else flagged."

**Isolation Forest ranks anomalies. Clustering characterizes them into
candidate fraud types.** Two different unsupervised jobs, not competing
techniques for the same job — don't try to make Isolation Forest do both.

## The tool already exists: `models/anomaly/cluster_discovery.py`

This isn't something to build — it's built. Read its module docstring for
the full reasoning; the summary:

- Takes the **top N% by `anomaly_score`** (default 10%) from
  `models/anomaly/train.py`'s already-written `anomaly_scores.parquet` —
  it does not re-score, and it does not cluster the full traffic stream
  (that would spend all its effort characterizing *normal* messages).
- Runs **DBSCAN** in the **same PCA/scaled feature space** Isolation
  Forest itself trained in (fit on the full candidate pool, then sliced
  to the anomalous subset — never refit on the subset alone, which would
  silently change the basis and make the two techniques describe
  different spaces).
- Label `-1` ("noise") is a real, meaningful output — a message that
  doesn't resemble any other flagged case densely enough to group. Don't
  treat it as a failure; a genuine one-off is a different finding from a
  40-message flooding burst.
- Prints a human-readable **cluster summary**: size, mean `anomaly_score`,
  source breakdown, and the *raw* (pre-scaling) behavioral/near-dup column
  means — deliberately raw, so "mean sender_msgs_last_1hr=8,200" reads as
  a flooding burst, not an opaque PCA-component number.
- Logs params/metrics/the cluster summary to its own MLflow experiment
  (`fraud_type_cluster_discovery`) — separate from `anomaly_score`, so a
  discovery run is never mistaken for a real candidate model.
- **Deliberately logs no model artifact.** DBSCAN has no `.predict()` for
  new data — nothing here generalizes to the next incoming message. This
  is a periodic, offline, human-in-the-loop tool, not a deployable stage
  of the pipeline (it's not in `pipeline.py`, on purpose, same as
  `train.py`).

## Step by step

**1. Make sure `anomaly_scores.parquet` is fresh for each source.**
```
python -m models.anomaly.train --sources SMPP
python -m models.anomaly.train --sources SS7
```
`cluster_discovery.py` reads this file, it does not compute anomaly
scores itself — a stale file after a feature-set change gives you
clusters built on outdated scores.

**2. Run cluster discovery.**
```
python -m models.anomaly.cluster_discovery
```
At sample scale, the defaults (`--anomaly_percentile 90`, auto-suggested
`eps`, `--min_samples 5`) are fine. **At full dataset scale, start tighter
— `--anomaly_percentile 99` or `99.5`** — see the scale caveat below before
running the plain default; the tool will refuse (not silently hang or
crash) past `MAX_RECOMMENDED_CANDIDATES` candidate rows, but getting there
still means throwing away a run. Read the printed `eps` suggestion and
cluster count.

**3. Eyeball the cluster summary and react to shape, not just numbers:**
- **One giant cluster + tiny noise** → `eps` too large, everything is
  getting lumped together. Lower it (`--eps <smaller>`), or lower
  `--min_samples`.
- **Dozens of tiny 1-2 row clusters** → `eps` too small, real groups are
  being split apart. Raise it.
- **Mostly noise (`-1`), few real clusters** → either genuinely correct
  (a lot of the top-anomaly pool really is heterogeneous one-offs — a
  legitimate finding), or `--anomaly_percentile` is pulling in too wide a
  pool. Try tightening it (`--anomaly_percentile 95`) to see if that
  concentrates the signal.
- There's no ground truth to optimize a cluster-quality metric against
  here (`suggest_eps()`'s heuristic is a starting point, not a proof) —
  this step is inherently iterative and human-judgment-driven. Budget for
  a few `--eps` passes, not one run-and-done.

**4. Hand-label each real (non-noise) cluster.** Use
`models/anomaly/inspect_clusters.py` rather than joining
`fraud_type_clusters.parquet` back to `messages_with_behavioral.csv` by
hand each time:
```
python -m models.anomaly.inspect_clusters --source SS7
python -m models.anomaly.inspect_clusters --source SMPP --n_samples 8
```
This prints every cluster (largest first, noise included) with its size,
mean `anomaly_score`, how many rows the rule engine already flagged
(`fraud_type` is non-null — a free hint, since a cluster that's mostly
already-known fraud types is a different finding from a genuinely novel
one), and a handful of real sample `text`/`originator` values. It also
writes `data/processed/<source>/cluster_labels_template.csv` — one row
per cluster with the same context plus a blank `fraud_type_label` column.
Fill that column in by hand (`flooding_burst`,
`phishing_template_rotating_url`, `smishing_otp_bait`, etc.) using the
printed samples or the CSV's own `sample_texts` column, then save. This
step is manual, on purpose — nothing downstream should treat a DBSCAN
cluster ID as a trustworthy label until a human has actually looked at
representative rows and confirmed what it is; the tool only gets you the
information to do that quickly, it doesn't decide for you.

Cluster IDs are per-run, not stable across reruns (see the caveat below)
— always inspect the `fraud_type_clusters.parquet` from the SAME
`cluster_discovery.py` run you're currently labeling.

**5. Feed confirmed labels forward — this is the part not yet built.**
There is currently no ingestion path from a hand-confirmed cluster label
back into training data (no `labels/cluster_labels.py` counterpart to
`labels/rule_labels.py` yet). When you build it:
- Store it as its own label source, **never merged into `rule_flagged`**
  — a cluster-derived label and a rule-engine label have different
  confidence/provenance and must stay traceable to which one they came
  from (mirrors this project's existing discipline: `rule_flagged` is
  never conflated with `SW_*` bypasses either — see
  `labels/rule_labels.py`).
- Only promote a cluster's rows to training labels once a human has
  actually confirmed the cluster's identity (step 4) — never auto-label
  from `cluster_label` alone. DBSCAN's job was grouping, not deciding
  ground truth.
- Log any classifier trained on these labels to its **own** MLflow
  experiment (e.g. `fraud_type_classifier`, or
  `rule_pattern_score_with_cluster_labels` if extending the existing
  LightGBM model) — same "never mistaken for the existing baseline"
  convention as `anomaly_score_diagnostics` and
  `rule_pattern_score_experimental`. Do not silently fold these rows into
  `rule_pattern_score`'s existing training pool.
- Once confirmed labels exist across multiple fraud types (not just
  binary spam/not-spam), this becomes the seed for a genuine multiclass
  classifier — the production-scale design's Stage C
  (`docs/sms_spam_technical_architecture_plan.md`), gated on label
  *volume*, not a calendar date.

## Caveats to keep in view

- **At full dataset scale, the default `--anomaly_percentile 90` will
  crash your machine — this is observed, not theoretical.** Once a full
  (non-sampled) `text_embeddings.py` run exists, the default top-10% pool
  is hundreds of thousands of rows (824,823 rows, observed, against an
  8.2M-row combined dataset). DBSCAN's neighbor search over that many rows
  at this feature space's dimensionality (~40+, post-PCA) exhausted system
  memory badly enough to take down the whole machine, not just the Python
  process. `run()` now refuses to proceed past `MAX_RECOMMENDED_CANDIDATES`
  (50,000 rows) without `--force` — the fix is a tighter
  `--anomaly_percentile` (try `99` or `99.5`), not `--force`. Also prefer a
  plain terminal over an IDE's integrated one for a large run: if it does
  exhaust memory anyway, you'd rather lose just the terminal than the
  editor. `run_dbscan()`'s `--n_jobs` also now defaults to 4, not `-1` (all
  cores) — more parallel workers means more CONCURRENT memory for the
  neighbor search, not just more speed, once the candidate pool is
  anywhere near that limit.
- **`eps` is specific to the candidate pool size it was suggested for.**
  `suggest_eps()`'s output depends on the k-NN distance distribution of
  whatever rows were actually selected — an `eps` printed for a 824,823-row
  pool is not a reusable constant for a 41,000-row pool from a tighter
  `--anomaly_percentile`. Re-suggest (omit `--eps`) whenever the pool size
  changes materially, rather than carrying an old value forward.
- **DBSCAN here is diagnostic, not production infrastructure.** Every
  run is independent — cluster `3` in one run has no guaranteed
  relationship to cluster `3` in the next (DBSCAN doesn't track cluster
  identity across runs the way a registered classifier tracks versions).
  Don't build anything downstream that assumes cluster IDs are stable
  across reruns; key off the hand-assigned fraud-type NAME instead, once
  step 4 has been done.
- **This never runs at live inference.** It's a periodic offline job you
  run by hand against a batch of already-scored traffic, same status as
  `models/anomaly/train.py` itself (not wired into `pipeline.py`).
