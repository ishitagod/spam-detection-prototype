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

- Takes the **top N% by `anomaly_score`** (default top 0.5%,
  `--anomaly_percentile 99.5`) from `models/anomaly/train.py`'s
  already-written `anomaly_scores.parquet` — it does not re-score, and it
  does not cluster the full traffic stream (that would spend all its
  effort characterizing *normal* messages).
- Runs **HDBSCAN by default** (`--algorithm dbscan` also available) on a
  **content-similarity-only slice** of the same PCA/scaled feature space
  Isolation Forest itself trained in (fit on the full candidate pool, then
  sliced to the anomalous subset — never refit on the subset alone, which
  would silently change the basis and make the two techniques describe
  different spaces) — see `select_clustering_features()` for exactly which
  columns and why.
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
- **Deliberately logs no model artifact.** Neither DBSCAN nor HDBSCAN has
  a `.predict()` for new data — nothing here generalizes to the next
  incoming message. This
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
The default algorithm is now **HDBSCAN**, not DBSCAN — `--algorithm
hdbscan` (default) vs. `--algorithm dbscan` (kept for comparison/
rollback). HDBSCAN has **no single global `eps` to pick at all**: it
builds a cluster hierarchy across a range of density thresholds and
extracts whichever groupings are stable across the widest range, so a
tight near-duplicate burst and a much looser, more-varied campaign can
both be found correctly at their own natural density in the same run —
see `run_hdbscan()`'s docstring. The one knob that matters is
`--min_cluster_size` (default 5, same meaning as DBSCAN's `--min_samples`:
the smallest group worth calling a cluster).

**Why not DBSCAN + a fixed `eps`, which this doc used to recommend:**
measured, not assumed — on SS7's 99.5-percentile pool (13,712
candidates), DBSCAN's auto-suggested `eps` (`4.817`) put 84% of the pool
in one cluster; a hand-swept fixed value (`eps=0.6`) got that down to a
12%-largest-cluster, 296-cluster result, but HDBSCAN beat that with zero
manual tuning — 786 clusters, largest only 1.5% of the pool, lower noise.
`eps=0.6` remains available as `--algorithm dbscan --eps 0.6` (validated
for SS7 at `--anomaly_percentile 99.5` only, re-sweep before trusting it
elsewhere — `--eps_auto` re-suggests from k-distance), but there's no
reason to reach for it unless HDBSCAN itself misbehaves on a given
source/pool.

**Also fixed since the DBSCAN era:** `select_clustering_features()`'s
"content" mode used to include `near_dup_distinct_senders_1hr/24hr` from
`NEAR_DUP_COLS`, which is sender-*identity*-derived, not content
similarity — measured to vary 0–34 within a single real multi-sender
campaign vs. flat 0–2 within a single-sender one, i.e. exactly the kind
of behavioral leakage that risks splitting one campaign apart. Clustering
now uses `CONTENT_SAFE_NEAR_DUP_COLS` instead (match-count/similarity
only, sender-count columns dropped). Separately, an explicit experiment
(`--cluster_features all`, blending behavioral columns back in) confirmed
this cuts both ways depending on the campaign's own consistency — it
fragmented a tight 2-sender gambling-spam cluster (204→38 rows in its
largest sub-cluster) but *improved* cohesion on a many-sender WhatsApp-
invite campaign (noise dropped 40%→21%) — there's no universal answer,
which is why `content` (not `all`) stays the default; use `all` as a
diagnostic on a specific cluster you suspect is under-grouped, not as a
blanket setting.

**3. Eyeball the cluster summary and react to shape, not just numbers:**
- **One giant cluster + tiny noise** → under HDBSCAN, try a larger
  `--min_cluster_size` first (fewer, larger stable groups get pulled out);
  under DBSCAN, `eps` is too large — lower it, or lower `--min_samples`.
- **Dozens of tiny 1-2 row clusters** → under HDBSCAN, try a smaller
  `--min_cluster_size`; under DBSCAN, `eps` is too small — raise it.
- **Mostly noise (`-1`), few real clusters** → either genuinely correct
  (a lot of the top-anomaly pool really is heterogeneous one-offs — a
  legitimate finding), or `--anomaly_percentile` is pulling in too wide a
  pool. Try tightening it (`--anomaly_percentile 95`) to see if that
  concentrates the signal.
- There's no ground truth to optimize a cluster-quality metric against
  here — this step is inherently iterative and human-judgment-driven.
  Budget for a few passes, not one run-and-done, whichever algorithm
  you're using.

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

**5. Feed confirmed labels forward.** `labels/cluster_labels.py`
(mirroring `labels/rule_labels.py`'s "derive labels from an external
verdict" shape) + `models/anomaly/ingest_cluster_labels.py` (the CLI/
file-I/O shell around it, same division as `inspect_clusters.py`) close
this loop:
```
python -m models.anomaly.ingest_cluster_labels --source SS7
python -m models.anomaly.ingest_cluster_labels --source SMPP
```
Run this after hand-filling a `cluster_labels_template.csv` (step 4). It
joins `fraud_type_clusters.parquet`'s per-message `cluster_label` to the
template's confirmed (non-blank `fraud_type_label`) rows only, and
accumulates the result into `data/processed/<source>/cluster_labels.parquet`
(`message_key`, `cluster_label`, `cluster_fraud_type_label`) —
de-duplicated on `message_key`, keep-newest, so repeated labeling
sessions over time (each against a freshly, differently-numbered
clustered batch — cluster IDs are per-run, not stable) accumulate into
one growing, current view instead of overwriting each other or
colliding. `build_cluster_labels()` also refuses to silently join a
template against the wrong run: if a confirmed `cluster_label` doesn't
exist in the `fraud_type_clusters.parquet` it's paired with, it raises
rather than dropping those rows quietly.

The design constraints that shaped it, still true and worth restating:
- `cluster_fraud_type_label` is its own column, **never merged into
  `rule_flagged`** — a cluster-derived label and a rule-engine label
  have different confidence/provenance and must stay traceable to which
  one they came from (mirrors this project's existing discipline:
  `rule_flagged` is never conflated with `SW_*` bypasses either — see
  `labels/rule_labels.py`).
- Only a hand-confirmed cluster (step 4) ever produces a label — never
  auto-labeled from `cluster_label` alone. DBSCAN's job was grouping,
  not deciding ground truth.
- Log any classifier trained on `cluster_labels.parquet` to its **own**
  MLflow experiment (e.g. `fraud_type_classifier`, or
  `rule_pattern_score_with_cluster_labels` if extending the existing
  LightGBM model) — same "never mistaken for the existing baseline"
  convention as `anomaly_score_diagnostics` and
  `rule_pattern_score_experimental`. Do not silently fold these rows
  into `rule_pattern_score`'s existing training pool.
- Once confirmed labels exist across multiple fraud types (not just
  binary spam/not-spam), `cluster_labels.parquet` becomes the seed for a
  genuine multiclass classifier — the production-scale design's Stage C
  (`docs/sms_spam_technical_architecture_plan.md`), gated on label
  *volume*, not a calendar date. That classifier itself is still not
  built — this step only gets the labels into a durable, accumulated
  file; training against them is a separate, later action.

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
