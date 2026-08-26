# Feature Catalog

What's actually servable from the Feast feature store today — the
answer to "what features exist and what do they mean," without having
to cross-reference `feature_repo/definitions.py`,
`features/behavioral_snapshot.py`, and `features/behavioral.py` every
time. Those three files remain the source of truth for *how* each
feature is computed; this doc is the source of truth for *what's
available and what it means*. Keep it in sync as features get added —
see "Adding a new feature" below.

Scope: **only features actually servable from Feast** (the entity/
feature-view table below). Text embeddings (`features/text_embeddings.py`)
and FAISS near-duplicate features (`features/faiss_index.py`) exist now
too, but are training/batch-only — they don't go through Feast, so they
don't belong in this specific catalog. See README.md's modeling plan for
those, or each module's own docstring for exactly what it computes. Model
scores (`rule_pattern_score`, `anomaly_score`) don't exist yet, per
`CLAUDE.md`'s next-steps list.

## Entity

| Entity | Join key | Meaning |
|---|---|---|
| `sender_id` | `source\|originator` (e.g. `SMPP\|66688`) | Compound key, not `originator` alone — SMPP originators (business sender IDs) and SS7 originators (real MSISDNs) are different namespaces that could coincidentally collide on the same string. |

## Features

| Feature | Kind | Feature view | Type | Meaning |
|---|---|---|---|---|
| `sender_msgs_last_5min` | stored | `sender_behavioral_stats` | Int64 | Messages this sender sent in the trailing 5 minutes |
| `sender_msgs_last_1hr` | stored | `sender_behavioral_stats` | Int64 | Messages this sender sent in the trailing 1 hour |
| `sender_unique_destinations_1hr` | stored | `sender_behavioral_stats` | Int64 | Distinct recipients this sender messaged in the trailing 1 hour |
| `recent_text_counts_json` | stored, internal | `sender_behavioral_stats` | String | Top-20 most frequent texts this sender sent in the trailing 1 hour, JSON-encoded `{text: count}`. Not meant to be consumed directly — exists only to feed `sender_repeat_content_ratio_1hr` below. |
| `sender_repeat_content_ratio_1hr` | **on-demand** (computed per request) | `sender_repeat_content_ratio` | Float64 | Of this sender's last-hour messages, the fraction matching the *exact* text of the message being scored right now |

**Stored vs. on-demand isn't a technicality — it's forced by data availability.** The first four are known ahead of time and get precomputed in batch (`features/behavioral_snapshot.py` → `feast materialize`). The fifth needs the *incoming* message's own text, which doesn't exist until the actual inference request — so Feast computes it live, combining the stored `recent_text_counts_json` + `sender_msgs_last_1hr` with the request's `candidate_text`. See `feature_repo/definitions.py`'s `sender_repeat_content_ratio()` for the exact logic.

## Known limitations / approximations

- **Top-20 cap on `recent_text_counts_json`.** A candidate text that matches a prior message outside the sender's top 20 most-frequent texts undercounts to `0` instead of its true (small) frequency. Accepted because this feature's job is catching bulk-repeated blasts — exactly the high-frequency case top-20 always retains.
- **Online freshness = last refresh, not live.** The online store never recomputes on read — every value is only as fresh as the last `scripts/refresh_feast.py` run. A sender who's been active in the seconds since that run will under-count until the next refresh.
- **This prototype's CDR sample is fixed-date (2026-08-02 to 2026-08-03).** The currently-materialized online store was built with `--now` overridden to the data's own end-of-range timestamp for demo/testing purposes, not real wall-clock time — see `features/behavioral_snapshot.py`'s `--now` flag. Real wall-clock `now` is the correct default once this runs against live data.

## Adding a new feature

When a new feature is added to the Feast-servable set (a model score,
or an existing training-only feature like FAISS near-dup gets promoted
to online-servable):

1. Decide stored vs. on-demand — does the value depend on anything only known at request time (like `candidate_text` does), or is it fully computable in batch ahead of time?
2. Stored: add a column to the relevant snapshot builder (extend `features/behavioral_snapshot.py`, or add a sibling snapshot module) and a `Field` to the matching `FeatureView`'s `schema=` in `feature_repo/definitions.py`.
   On-demand: add a new `@on_demand_feature_view` function, following `sender_repeat_content_ratio`'s pattern — remember the local-import gotcha noted in that function's docstring (module-level imports don't survive Feast's dill serialization for on-demand UDFs).
3. Run `scripts/refresh_feast.py` (or `feast apply` alone, for a stored-schema-only change with no new snapshot column yet).
4. Add a row to the table above.
5. Add/extend tests in `tests/test_behavioral_snapshot.py` (or the new module's own test file).
