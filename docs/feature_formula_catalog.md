# Feature / Formula Catalog — by Fraud Use Case

Cross-reference of every feature (built + proposed) against the fraud
pattern it targets. Columns:

- **Formula Derived** — exact computation, in the same point-in-time
  convention as everything else in this repo (strictly-earlier-than-now
  unless noted).
- **Dataset Type** — `SMPP` / `SS7` / `Both` (canonical, so computed
  identically once source-specific raw fields are mapped in).
- **Data Validity** — `stored` (batch-precomputed, Feast-materialized),
  `on-demand` (computed at request time from stored inputs + the
  incoming message), or `offline-async` (network-call-dependent, cached
  with a staleness TTL, never on the request path — see
  `docs/smishing_detection_plan.md`'s offline/online split).
- **Thresholds / Extra fields needed** — either the real tunable from
  `config/settings.py`, or, for proposed features, what new raw/config
  field would have to exist first.

Existing features are sourced from `features/behavioral.py`,
`features/faiss_index.py`, `feature_repo/definitions.py`,
`config/settings.py`. Proposed features are marked **NEW** and are not
implemented — they follow the same architecture rules (point-in-time,
`source`-agnostic contract, stored-vs-on-demand split) as everything
already built.

## Use Case → Section Mapping

| Use Case (as specified) | What it targets | Where in this doc |
|---|---|---|
| **Use Case 1 — Spam Content Detection**: promotional offers, fake rewards, financial scams, urgency-based messages; NLP-based analysis assigns a spam probability/risk score | single-message content, no history needed | §1 below. The "NLP-based spam probability/risk score" itself is `rule_pattern_score` (LightGBM, optionally `--with_tfidf`/`--with_embeddings` — see `docs/experiments/rule_pattern.md`) and `anomaly_score` (Isolation Forest + MiniLM) — the keyword/URL features in §1 are cheap, interpretable *inputs* to that score, not a replacement for it |
| **Use Case 3 — Sender Behaviour Analysis**: unusually high volume of similar/repeated messages; abnormal sender behaviour risk score | one sender's own pattern over time | §2 (Flooding/Bulk Send) + §3 (SIM-Farming/Identity Cycling) + §5's `sender_repeat_content_ratio_*`/`sender_text_entropy_1hr` |
| **Use Case 4 — Similar/Repeated Spam Content, different senders**: same/slightly-modified content across different sender IDs or sources, evading rule-based detection | cross-sender content matching | §5's FAISS near-dup rows (semantic, catches *modified* content) + §5b's `network_repeat_content_ratio_1hr`/`network_distinct_senders_same_text_1hr` (exact, cheap, catches *unmodified* copy-paste even without embeddings) |

---

## 1. Spam Content Detection — Category Keyword & URL Signals (Use Case 1)

Single-message, per-request, no history/lookup needed — the cheapest
tier of spam signal, computable the moment a message arrives regardless
of sender history. Directly targets the four content categories named in
Use Case 1: promotional offers, fake rewards, financial scams,
urgency-based messages, plus the URL tell that's usually bundled with
all four in practice.

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `has_url` **(NEW)** | regex URL match found in `text` | `text` | Both | on-demand/per-request, no lookup | gates every URL-derived feature (here and in §6's deeper tier) — absent (not `0`) when false, so "no URL" and "URL present but not phishy" never collapse to the same value |
| `promo_offer_keyword_hit_count` **(NEW)** | count of static promotional/offer keyword matches in `text` (e.g. "free", "discount", "% off", "limited offer", "buy now", "sale") | `text` | Both | on-demand, per-request | needs an editable config keyword list, same convention as §6's brand/urgency list — a config file, not hardcoded inline |
| `fake_reward_keyword_hit_count` **(NEW)** | count of reward/prize keyword matches ("congratulations", "winner", "claim your prize", "cashback", "lucky draw", "lottery") | `text` | Both | on-demand | own config keyword list, independently editable from the promo list above — these evolve on different timelines as campaigns shift |
| `financial_scam_keyword_hit_count` **(NEW)** | count of financial-scam-specific keyword matches ("loan approved", "KYC", "account blocked", "verify your bank", "refund pending", "payment failed", "update your details") | `text` | Both | on-demand | own config keyword list; overlaps somewhat with §6's brand-impersonation urgency list by nature — kept separate because this category is bank/finance-specific, not generic urgency |
| `urgency_keyword_hit_count` | count of urgency keyword matches ("verify", "suspended", "OTP", "account locked", "act now", "expires today", ...) | `text` | Both | on-demand | already proposed in the smishing plan (§6 below) — same feature, cross-referenced here since Use Case 1 explicitly names "urgency-based messages" as its own category |
| `spam_category_keyword_total_hit_count` **(NEW)** | `promo_offer + fake_reward + financial_scam + urgency` hit counts, summed | the four features above | Both | on-demand, purely derived | a single composite lexical score — cheap pre-model signal, not a replacement for `rule_pattern_score`/`anomaly_score` |

**Why keyword lists, not just one blended list:** each category evolves independently (a bank-impersonation wave and a lottery-scam wave don't update on the same schedule), and keeping them separate lets `rule_pattern_score`'s SHAP output attribute a prediction to a *specific* category rather than one opaque bucket — same interpretability reasoning already applied to `--with_tfidf`'s per-token SHAP output (see `docs/experiments/rule_pattern.md`).

**Same single-feature-dominance risk as `tfidf_https` applies here too.** The real SHAP finding on the SS7 `--with_tfidf` champion — one token (`tfidf_https`) alone swinging a live prediction's risk_score 0→96 — is a direct warning for these keyword-hit counts as well: a high-weight single keyword match is trivially evadable (misspell it, insert a space). `train_lightgbm()`'s `colsample_bytree`/`min_child_samples`/`reg_alpha`/`reg_lambda` anti-dominance knobs (not yet tuned) apply equally once these features are added, not just to TF-IDF tokens.

---

## 2. Flooding / Bulk Send (Use Case 3)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `sender_msgs_last_1min` / `_5min` / `_1hr` | count of this sender's messages with `ts < now`, `now − ts ≤ W` | `source`, `originator`, `timestamp` | Both | stored | Windows fixed in `config/settings.py` (`BEHAVIORAL_VERY_SHORT/SHORT/LONG_WINDOW` = 1min/5min/1h) — no anomaly cutoff yet, model-learned |
| `sender_unique_destinations_5min` / `_1hr` | count of distinct `destination` in same window | `source`, `originator`, `destination`, `timestamp` | Both | stored | same windows as above |
| `sender_recipient_diversity_ratio_5min` / `_1hr` | `unique_destinations_W / msgs_last_W`, else `0.0` if `msgs_last_W == 0` | above two | Both | stored (derived from the two, same materialize step) | none — ratio, self-normalizing |
| `sender_velocity_zscore_5min` | `(msgs_last_5min − running_mean) / running_stdev`, Welford's algorithm over this sender's *own* prior 5-min readings | `sender_msgs_last_5min` history | Both | stored | `NaN` if <2 prior readings or stdev < 1e-9 (`_VELOCITY_ZSCORE_STDEV_EPSILON`); no fixed alert cutoff — feeds model directly |
| `sender_age_days` | `(now − sender's first-ever message ts) / 1 day` | `source`, `originator`, `timestamp` | Both | stored | none; 0.0 on a sender's first message is correct cold-start, not missing |

---

## 3. SIM-Farming / Identity Cycling — Sender Behaviour (Use Case 3)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `imsi_distinct_originators_1hr` | count of distinct `originator` seen behind this `imsi`, strictly before now, within 1hr | `imsi`, `originator`, `timestamp` | **SS7 only** | training-time only — not yet Feast-served (needs its own `imsi` entity/FeatureView) | `NA` (not 0) when this row's own `imsi` is null — 31.7% of real SS7 traffic is null-imsi, confirmed against real data |
| `imsi_age_days` **(NEW)** | `(now − first_seen_ts(imsi)) / 1 day` — age of the physical SIM, not the apparent number. Catches identity rotation `sender_age_days` alone can't: a SIM-farmer's newest MSISDN always scores `sender_age_days ≈ 0`, but the IMSI behind it may be old. Read together with `sender_age_days` (small + large pair = rotation signal), not as a replacement for it | `imsi`, `timestamp` | **SS7 only** | would be stored, same groupby-min mechanism as `sender_age_days`, keyed by `imsi` instead | `NA` (not 0) when this row's own `imsi` is null, same convention as `imsi_distinct_originators_1hr`; needs an `imsi` entity/FeatureView if Feast-served |

### NEW — Sender-ID Spoofing / Identity Cycling (SMPP-only, proposed)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `virtual_gt_distinct_originators_1hr` **(NEW)** | count of distinct `originator` values sharing this message's `virtual_gt`, strictly before now, within 1hr — direct SMPP mirror of `imsi_distinct_originators_1hr`'s pattern (proxy identity ↔ physical SIM identity) | `virtual_gt`, `originator`, `timestamp` | **SMPP only** | would be stored, same two-pointer mechanism as `imsi_distinct_originators_1hr` | `NA` when `virtual_gt` is null, same non-pooling convention; needs a `virtual_gt` entity/FeatureView if promoted to serving |
| `virtual_gt_age_days` **(NEW)** | `(now − first_seen_ts(virtual_gt)) / 1 day` — SMPP mirror of `imsi_age_days`: age of the proxy identity behind the displayed sender, not the sender-ID itself. Same rotation-detection pairing with `sender_age_days` | `virtual_gt`, `timestamp` | **SMPP only** | would be stored, same groupby-min mechanism as `sender_age_days`, keyed by `virtual_gt` | `NA` when `virtual_gt` is null; needs a `virtual_gt` entity/FeatureView if Feast-served |
| `instance_id_msgs_last_5min` **(NEW)** | count of messages on this `instance_id` (bind session), strictly before now, within 5min — catches a single bulk-injection session regardless of how many distinct `originator` values it spoofs | `instance_id`, `timestamp` | **SMPP only** | stored, same windowed-count pattern as `sender_msgs_last_5min` | needs an `instance_id` entity if Feast-served; window reuses `BEHAVIORAL_SHORT_WINDOW` |

---

## 4. Multipart / Reassembly Abuse

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `concat_total_parts` / `concat_part_num` | direct from UDH IE 0x00/0x08 (SMPP) or native `sarref`/`msg_parts`/`msg_part` (SS7), unified into one shared column pair | raw UDH bytes (SMPP) or `sarref`/`msg_part`/`msg_parts` (SS7) | Both (different raw source, same output shape) | stored, per-message | `1/1/None` = genuinely single-part on both sources |

---

## 5. Content Spam — Repeat / Near-Duplicate (Use Case 3 sender-side + Use Case 4 cross-sender)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `sender_repeat_content_ratio_1hr` | `count(prior msgs, same sender, EXACT text match) / msgs_last_1hr`, else `0.0` | `source`, `originator`, `text`, `timestamp` (+ candidate message's own `text` at request time) | Both | **on-demand** (needs the incoming message's text — Feast `on_demand_feature_view`, `feature_repo/definitions.py::sender_repeat_content_ratio`) | none; exact-match, feeds both models (see note below) |
| `near_dup_match_count_1h` / `_24h` | count of prior messages (any sender) with `cos_sim(embed(msg), embed(m)) ≥ τ`, within window | MiniLM embedding of `text` | Both | **stored at training** (`features/faiss_index.py`, batch corpus) **+ live-recomputed at serving** (`serving/anomaly_scoring.py::score_anomaly()` embeds the request's own text and FAISS-queries a cached per-source corpus, windowed against the request's own `timestamp`) — bypasses Feast entirely, not a Feast on-demand view | `τ = FAISS_NEAR_DUP_THRESHOLD = 0.92`; windows `FAISS_NEAR_DUP_WINDOW_SHORT="1h"`, `_LONG="24h"` (`config/settings.py`); serving corpus staleness = last `text_embeddings.py`+`faiss_index.py` batch run, no live-appended stream in this prototype |
| `near_dup_distinct_senders_1h` / `_24h` | count of distinct senders among the near-dup match set above — **this row is Use Case 4's core signal**: "similar messages sent from different sender IDs" | same + `source`\|`originator` of matches | Both | same as above — stored at training, live-recomputed at serving | same τ/window/staleness as above |
| `near_dup_max_similarity_1h` / `_24h` | `max(cos_sim(msg, m))` over the match set | same | Both | same as above — stored at training, live-recomputed at serving | same τ/window/staleness; capped at `FAISS_MAX_MATCHES_PER_QUERY = 100,000` candidates per query message |

**Why FAISS near-dup is what actually solves Use Case 4's "slightly modified to bypass rule-based detection" clause:** exact-match (`sender_repeat_content_ratio_1hr`, §5b below) only catches byte-identical text. A campaign that swaps one word, randomizes a token, or changes punctuation per victim defeats exact match entirely but still lands at `cos_sim ≥ 0.92` — semantic similarity is specifically the mechanism that survives paraphrasing/templating, which is exactly the evasion Use Case 4 describes.

**On embeddings/TF-IDF and `rule_pattern_score` — not permanently barred, a default-path scoping decision** (`models/rule_pattern/data.py`'s module docstring says so explicitly), with `--with_embeddings`/`--with_tfidf` already built as independent, toggleable flags:
- **`--with_tfidf` is already proven useful**: standalone TF-IDF+LogisticRegression on the full real SS7 `rule_evaluated` pool, split by unique text (no template leakage), scored **PR-AUC 0.934 vs. a 0.669 naive baseline**. Held in the `rule_pattern_score_experimental` MLflow experiment pending formal champion/challenger promotion.
- **`--with_embeddings` coverage is now source-dependent**: SS7's `embeddings.npy` is a full-dataset GPU run (`2,742,301 × 384`, matching SS7's full row count) as of this session — ready to retrain against, not yet done (see `docs/experiments/rule_pattern.md`). SMPP has no embeddings.npy yet.
- Both are logged to a separate MLflow experiment on purpose, so an early/partial run is never mistaken for the real baseline.
- A real production risk surfaced by the SS7 `--with_tfidf` champion's SHAP output: one token (`tfidf_https`) alone swung a live prediction's risk_score 0→96 — plausible correct rule-pattern, but trivially evadable. `train_lightgbm()` now has `colsample_bytree`/`min_child_samples`/`reg_alpha`/`reg_lambda` anti-dominance knobs, not yet tuned.

---

## 5b. Content Spam — Repeat, NETWORK-level and cross-window (proposed, extends §5)

The existing repeat-content features are either purely sender-local
(`sender_repeat_content_ratio_1hr`, exact match) or purely
semantic/cross-sender but embedding-dependent (FAISS near-dup). The gap:
a **cheap, exact-match, cross-sender** signal — catches copy-paste
campaigns fanned out across many spoofed identities without needing
GPU/embeddings at all, so it's available to `rule_pattern_score` too
(no embedding-barrier issue, same exact-string mechanism
`sender_repeat_content_ratio_1hr` already uses, just not sender-scoped).
This is Use Case 4's *unmodified*-content case — the cheap complement to
FAISS near-dup's *modified*-content case above.

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `network_repeat_content_ratio_1hr` **(NEW)** | `count(ANY sender's prior msgs, EXACT text match, within 1hr) / count(ALL msgs, any sender, within 1hr)` — network-wide version of the existing sender-scoped ratio | `text`, `timestamp` (global, not sender-keyed) | Both | would be stored/batch — same two-pointer sliding-window mechanism as `features/behavioral.py`, keyed globally by `text` instead of by sender | needs a global (non-entity-keyed, or a synthetic "text-hash" entity) rolling counter — a genuinely new computation shape, not a small extension |
| `network_distinct_senders_same_text_1hr` **(NEW)** | count of distinct senders among the exact-text matches above, within 1hr — cheap non-embedding analog of `near_dup_distinct_senders_1h` | same as above + `source`\|`originator` of matches | Both | same as above | same as above; catches the "many spoofed IDs, one literal template" case even before any embedding pipeline exists for a source (e.g. covers SMPP today, ahead of its pending full embeddings run) |
| `sender_repeat_content_ratio_24hr` **(NEW)** | same formula as the existing `_1hr` version, window extended to 24hr | same fields, longer window | Both | on-demand, same mechanism, reuses `recent_text_counts_json`-style storage over a longer window | mirrors the exact reasoning FAISS already uses for having both `_1h` and `_24h` windows (short catches bursts, long catches paced-out campaigns that never cluster within an hour) |
| `sender_text_entropy_1hr` **(NEW)** | Shannon entropy of this sender's text distribution in the trailing 1hr: `-Σ p_i·log2(p_i)`, `p_i` = share of messages with each distinct text seen | `source`, `originator`, `text`, `timestamp` | Both | stored, same `recent_text_counts_json`-style counter `sender_repeat_content_ratio_1hr` already maintains, just reduced differently | entropy → 0 for a sender blasting one/few templates repeatedly; high entropy for genuinely varied content. Complements (doesn't replace) `repeat_content_ratio`, which only measures similarity to the *current* message, not overall distribution shape |
| `destination_repeat_content_ratio_1hr` **(NEW)** | `count(prior msgs TO this destination, EXACT text match, within 1hr) / count(msgs TO this destination, within 1hr)` — same-content-to-one-victim angle (OTP bombing / harassment), destination-keyed instead of sender-keyed | `destination`, `text`, `timestamp` | Both | would be stored, needs a new `destination`-keyed entity/FeatureView (parallel to the `domain` entity proposed in §6, just keyed by MSISDN instead of domain) | catches a sender rotating identities but hammering the same victim with the same content — a case none of the sender-keyed features above can see, since they all reset per apparent identity |

---

## 6. Smishing — URL / Domain Phishing (proposed, `docs/smishing_detection_plan.md`)

### NEW — Deeper lexical tier (cheap, per-message, no lookup) — extends §1's `has_url`

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `url_shortener_flag` **(NEW)** | hostname ∈ static shortener list (`bit.ly`, `tinyurl.com`, `t.co`, ...) | extracted host from `text` | Both | on-demand | needs `config/`-style shortener list (new config file) |
| `url_has_userinfo` **(NEW)** | `urllib.parse`-extracted host has an `@`-prefixed userinfo segment | extracted URL | Both | on-demand | none; must use a real URL parser, not regex |
| `url_is_ip_literal` **(NEW)** | host is a raw IP (regex/`ipaddress` stdlib check) | extracted host | Both | on-demand | none |
| `url_is_punycode` **(NEW)** | host starts with `xn--` | extracted host | Both | on-demand | none |
| `url_subdomain_depth` **(NEW)** | count of subdomain labels before the registrable domain (`tldextract`) | extracted host | Both | on-demand | needs `tldextract` dependency |
| `url_suspicious_tld` **(NEW)** | registrable-domain TLD ∈ static list (`.tk`, `.ml`, `.ga`, `.top`, `.xyz`, ...) | registrable domain | Both | on-demand | needs a config-list TLD file |

### NEW — Domain-reputation tier (async, cached, network-dependent)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `url_min_domain_age_days` **(NEW)** | `min` over this message's URLs' registered-domain WHOIS age | registered domain → WHOIS lookup | Both | **offline-async**, cached by `domain`, worst-case-folded at request time | new `domain_reputation` Feast entity/FeatureView; needs `python-whois` |
| `url_any_no_ssl` **(NEW)** | `OR` over this message's URLs: does the domain currently present no valid SSL cert | registered domain → SSL check | Both | offline-async, cached by domain | `httpx`/stdlib `ssl`, no new dependency |
| `url_max_redirect_count` **(NEW)** | `max` over this message's URLs' redirect-chain length | registered domain → redirect-follow | Both | offline-async, cached by domain | same as above |
| `url_any_not_google_indexed` **(NEW)** | `OR`: any URL's domain not indexed | registered domain → index-status check | Both | offline-async, cached by domain | source TBD (open question in smishing plan) |
| `domain_distinct_senders_1hr` **(NEW)** | count of distinct senders (`source`\|`originator`) that linked this domain, strictly before now, within 1hr | registered domain, `source`, `originator`, `timestamp` | Both | **stored**, keyed by `domain` entity (not per-sender) | needs new `domain` Feast entity |

---

## 7. Smishing — WAP Push Malware Delivery (proposed)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `has_wap_port_udh` **(NEW)** | UDH contains an Application Port Addressing IE (0x04 8-bit / 0x05 16-bit) | raw UDH bytes | **SMPP confirmed usable; SS7 unconfirmed** — SS7 has no UDH-stripping today, needs checking first | on-demand, per-request, no lookup | extend `ingestion/smpp.py::_parse_udh` to recognize IE 0x04/0x05 |
| `udh_dest_port` **(NEW)** | addressed port extracted from the same IE, nullable int | raw UDH bytes | same as above | on-demand | same |

---

## 8. Smishing — Sender-ID / Brand Impersonation (proposed)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `sender_id_brand_mismatch` **(NEW)** | `1` if `originator` is alphanumeric and not in a known-brand allow-list, or a long-code MSISDN appears where a short-code is expected; else `0` — soft signal, not a hard rule | `originator`, `originator_ton`/`originator_npi` | **SMPP only** (`originator_ton`/`npi` don't exist on SS7) | on-demand | needs a "known brand allow-list" — doesn't exist in the repo yet (open question in smishing plan) |

`urgency_keyword_hit_count` moved to §1 (Use Case 1 explicitly names urgency-based messages as a spam content category, not smishing-specific) — cross-referenced here since brand impersonation and urgency framing usually co-occur in real messages.

---

## 9. Destination-Sequence Clustering (proposed, SS7-oriented but source-agnostic in shape)

| Feature Name | Formula Derived | Fields Needed | Dataset Type | Data Validity | Thresholds / Extra fields needed |
|---|---|---|---|---|---|
| `sender_dest_sequential_ratio_1hr` **(NEW)** | sort trailing-1hr destinations ascending → `d_1..d_k`; `ratio = count(gap(d_i, d_{i+1}) ≤ gap_threshold) / (k−1)` | `source`, `originator`, `destination`, `timestamp` | Both (numeric destinations needed — SS7's MSISDN destinations are the primary use case; SMPP viable if destinations are numeric too) | stored, same window/mechanism as `sender_unique_destinations_1hr` | `gap_threshold` — not yet chosen, needs checking against real destination-numbering sparsity first (open question in smishing plan) |

---

## Notes

- **No fixed alert thresholds exist for most model-facing features** (`velocity_zscore`, `diversity_ratio`, `repeat_content_ratio`) — they feed LightGBM/Isolation Forest directly and let the model learn cutoffs, consistent with `_reason_codes()` now being SHAP-derived rather than fixed-threshold (see `docs/ml/modeling.md`). The only genuinely fixed, hand-picked threshold in the whole feature set today is `FAISS_NEAR_DUP_THRESHOLD = 0.92`.
- **Every NEW feature above follows the existing architecture rules**: point-in-time (strictly-earlier-than-now), `source`-agnostic canonical contract where the raw data allows it, stored-vs-on-demand split decided by whether it needs the incoming message's own content, and no network call ever on the inference request path (offline-async + cache instead).
- Adding any of these for real: follow `docs/feature_catalog.md`'s "Adding a new feature" checklist (snapshot column or on-demand view → `feature_repo/definitions.py` → `refresh_feast.py` → catalog entry → tests).
