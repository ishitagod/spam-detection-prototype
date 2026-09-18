# Multi-Client Extensibility Plan

Internal plan for making the fraud detection system easy to swap/extend
across clients who bring different raw data formats, fraud categories,
thresholds, and fields — without forking the codebase or scattering
per-client edits across it.

## Goal

Different clients bring different raw formats, different fraud
categories to prioritize, different thresholds, and sometimes different
fields entirely. "Swappable" here means bounded, mostly-additive
onboarding: a new client should mean new adapter code and new config,
not a fork of the system or edits scattered across unrelated files.

## Already client-agnostic

Two patterns in the codebase already generalize cleanly and can be
reused as templates for further client-specific work:

- **Ingestion adapters** — the `clean` + `map_to_canonical` interface
  (`ingestion/base.py`'s `SourceHandlers`), registered in a lookup table
  in `run_ingest.py`. Adding a source means implementing this interface
  and registering it, not touching shared pipeline code.
- **Source-split training** — `--sources SMPP` / `--sources SS7` CLI
  filters in `models/anomaly/train.py` and `models/rule_pattern/train.py`,
  with MLflow experiment and registered-model names suffixed by source
  (e.g. `isolation_forest_SMPP`), so split models never mix with the
  combined-sources experiment.

## Where the real rework is

| Area | Current state | Risk if left as-is |
|---|---|---|
| Canonical schema | Flat, hardcoded field set in `common/schemas.py`, shared by both sources | New client with genuinely different fields has nowhere to put them without touching the shared schema shared ML code depends on |
| Feature definitions (Feast) | One flat, hardcoded file of feature views | Per-client feature differences mean editing the same shared file for every client, risking cross-client regressions |
| Thresholds | Single hardcoded `_FRAUD_THRESHOLD = 0.5` in `serving/app.py` | Not config-driven, not per-client, not per-fraud-category — every client gets the same cutoff today |

These three are the places genuine rework is needed; everything else
extends the two patterns above.

## Canonical schema: keep the core flat, add a client-extension slot

Don't make the core schema dynamic. Instead, add one reserved typed
extension field (a key-value map) for client-specific attributes that
shared ML code never reads. Promoting a field out of the extension slot
into the shared schema is a deliberate, reviewed change — mirroring how
SS7-only fields already remain SS7-only when they carry real signal,
without polluting the shared schema by default.

## Feature definitions: split the universal set from the per-client set

Split feature definitions into two tiers: a universal feature set
(shared defaults, always computed) and a per-client feature set layered
on top. Each client gets a config file listing which universal features
apply to them plus any client-specific additions. This follows the same
adapter-and-registry shape as ingestion — a per-client config, not a
per-client fork of the feature file.

## Thresholds: move out of code, into a per-client config store

Move the single hardcoded `_FRAUD_THRESHOLD` value out of
`serving/app.py` into a config store keyed by client and fraud category,
read at request time. The two model scores (`rule_pattern_score`,
`anomaly_score`) must keep independent thresholds — never blended,
consistent with the rest of the system. This config store is also the
prerequisite for any threshold-configuration or KPI admin UI to control
real behavior rather than a hardcoded default.

## Extend the source-split pattern to a client dimension

The `--sources SMPP`/`--sources SS7` pattern already does the hard part
— filtering training data and namespacing the resulting experiment and
registered model. Adding `--client` alongside it is the same mechanism,
not a new one:

- Training scripts gain a client filter argument, same shape as the
  existing source filter.
- MLflow experiment and registered-model names get a client suffix the
  same way they already get a source suffix, so champion/challenger
  comparisons stay apples-to-apples within a client and never mix across
  clients.
- Feature-store data gets a client identifier as a partition key (or a
  separate Feast project per client, for clients needing full data
  isolation), so one client's behavioral history never leaks into
  another's feature lookups.
- Rule definitions, already externalized rather than hardcoded, get the
  same per-client layering as feature definitions.

No tenant/client_id concept exists in the codebase today — this is new
ground beyond the naming/suffixing shape the source split already
proves out.

## New-client onboarding checklist

1. Write the ingestion adapter — `clean` + `map_to_canonical` for the
   client's raw format, registered in the source lookup table.
2. Populate the client-extension slot on the canonical schema for any
   client-specific attributes shared ML code doesn't need.
3. Define the client's feature set — which universal features apply,
   plus any client-specific additions, in its per-client feature config.
4. Set initial thresholds per fraud category in the per-client config
   store (start from defaults, tune after live volume arrives).
5. Run source-split (and now client-split) training, producing
   client-suffixed MLflow experiments and registered models.
6. Validate against rule labels for that client before promoting a
   champion.
7. Confirm data isolation — client identifier partitioning or a
   dedicated Feast project — before any shared batch job runs.

## Rollout order

1. **Canonical schema extension slot** — smallest change, unblocks
   everything else, no behavior change for existing clients.
2. **Thresholds out of code** — independent of the client dimension,
   fixes a real gap (single hardcoded value today) regardless of how
   many clients exist.
3. **Feature definitions split** — universal vs. per-client tiers, built
   once the extension slot exists to hold client-specific inputs.
4. **Client-split training** — extend `--sources` to `--client`, once
   there's a second client's data to actually split against.
5. **Onboard the second client end-to-end** using the checklist above —
   the real test of whether the first four steps hold up outside the
   original single-client design.
