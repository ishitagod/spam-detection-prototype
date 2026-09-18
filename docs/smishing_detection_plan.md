# Smishing Detection Plan

Plan for adding smishing-specific signal (URL phishing, WAP Push malware
delivery, sequential-DA bulk campaigns) on top of the existing pipeline.
Three signal families, from real CDR columns already investigated:

1. URL/domain reputation, from `content_self_decoded` (→ canonical `text`)
2. WAP Push indication, from `app_dest_port`/`app_src_port` + UDH
3. Destination-address (`da`) sequence clustering
4. Text keyword / brand & sender-ID heuristics

None of these need a new pipeline stage — they extend canonical mapping,
`features/behavioral.py`/`behavioral_snapshot.py`, and Feast, following
the same point-in-time and shared-schema rules as everything else in
`CLAUDE.md`.

## Offline vs online architecture

The hard constraint driving every design choice below: **inference stays
a fast cache read plus simple per-message rules, never a live lookup and
never a fresh scan over history.** Everything that needs historical
data (a window of past messages) or a third-party network call
(WHOIS/DNS/SSL/etc.) is computed **offline, on its own schedule**, and
lands in the Feast online store as a plain cached value before any
request touches it — exactly the existing `sender_behavioral_stats`
pattern, just extended to a second (`domain`) entity.

```
OFFLINE / ASYNC  (batch, historical, network calls allowed)
│
│  ingested messages (SMPP + SS7, already flowing through the
│  existing pipeline)
│         │
│         ├─► features/behavioral_snapshot.py (EXISTING, extended)
│         │     - sender_msgs_last_5min/1hr, unique_destinations_1hr
│         │     - sender_dest_sequential_ratio_1hr   (sec. 3, NEW)
│         │
│         ├─► extract URLs -> registered domains  (NEW)
│         │         │
│         │         ▼
│         │     queue table (SQLite - no new broker)
│         │         │
│         │         ▼
│         │     enrichment worker, own schedule/rate limit:
│         │       WHOIS domain age, DNS record, SSL cert state,
│         │       redirect-follow, web-traffic rank, CT-log
│         │       first-seen, google-indexed
│         │         │
│         │         ▼
│         │     domain_reputation cache, keyed by domain (NEW)
│         │       - also: domain_distinct_senders_1hr/24hr (sec. 1)
│         │
│         └─► scripts/refresh_feast.py (EXISTING pattern)
│                 snapshot -> feast apply -> feast materialize
│                 (both sender_behavioral_stats AND domain_reputation)
▼
Feast online store (SQLite)  ◄── the ONLY boundary online code reads
═══════════════════════════════════════════════════════════════════
ONLINE / INFERENCE  (per-request, no network calls, no history scan)
▼
serving/feature_lookup.py (EXISTING pattern, extended):
  1. cheap SIMPLE RULES computed directly from the request, no lookup:
     has_url, url_shortener_flag, url_has_userinfo, url_is_ip_literal,
     url_is_punycode, url_subdomain_depth, url_suspicious_tld,
     urgency/brand keyword hits, sender-ID type mismatch  (secs. 1 & 4)
  2. cache READS only, keyed by sender_id and by this request's
     extracted domain(s): sender_behavioral_stats,
     sender_dest_sequential_ratio_1hr, domain_reputation
     (worst-case-folded across a message's URLs, see sec. 1)
  3. has_wap_port_udh / udh_dest_port from this message's own UDH bytes
     (sec. 2) - also a simple per-request parse, no lookup
        │
        ▼
  rule_pattern_score (LightGBM) / anomaly_score (Isolation Forest)
```

**Why the line is drawn exactly here:** a cache miss or stale entry
(domain never enriched yet, sender snapshot not yet refreshed) is cheap
to handle - fall back to a neutral/unknown value - while a live WHOIS or
redirect-follow call on the request path would violate both the "fast
inference" requirement and point-in-time validity (the enrichment result
for a same-second-registered phishing domain literally doesn't exist yet
at message time no matter how fast the call runs). This is the same
logic already governing `sender_repeat_content_ratio_1hr`'s "why
on-demand, not stored" choice, generalized: **online is a lookup + rule
engine, never a fetch.**

| Feature (from below) | Where computed | Needs history? | Needs network call? |
|---|---|---|---|
| `has_url`, `url_shortener_flag`, `url_has_userinfo`, `url_is_ip_literal`, `url_is_punycode`, `url_subdomain_depth`, `url_suspicious_tld` | online, per-request | no | no |
| urgency/brand keyword hits, sender-ID type mismatch | online, per-request | no | no |
| `has_wap_port_udh` / `udh_dest_port` | online, per-request | no | no |
| domain age/WHOIS, SSL state, DNS record, redirect count, traffic rank, CT-log first-seen, google-indexed | **offline**, cached by domain | no (per-domain, not per-message) | **yes** |
| `domain_distinct_senders_1hr` | **offline**, cached by domain | **yes** (1hr window) | no |
| `sender_dest_sequential_ratio_1hr` | **offline**, cached by sender | **yes** (1hr window) | no |
| existing `sender_msgs_last_*`, `sender_unique_destinations_1hr` | **offline**, cached by sender (unchanged) | **yes** | no |

## 1. URL / domain reputation

**Basic lexical features first — computed synchronously, no external
calls, available at inference immediately.** Before any enrichment
lookup, extract a cheap tier of features directly from the URL string
itself:
- `has_url` (bool) — any URL pattern found in `text` at all. Gate for
  every other URL-derived feature below (including the enrichment-tier
  ones in this section) — they're only meaningful when this is true, and
  should default to a neutral/absent value rather than 0 when it's false
  ("no URL" and "URL present but not phishy" must not collapse to the
  same value).
- `url_shortener_flag` (bool) — hostname matches a known
  shortener/redirector list (`bit.ly`, `tinyurl.com`, `t.co`, etc.).
  Shorteners are exactly the redirect-obscuring mechanism smishing uses
  to hide the real destination, so this is a strong standalone signal
  even before the redirect gets followed.
- `url_has_userinfo` (bool) — an `@` before the host in the URL
  (`http://real-bank.com@evil.tk/...`). Classic disguise: naive readers
  (and naive regexes) see `real-bank.com` at a glance, but the actual
  host resolved is whatever follows `@`. Extract host via a real URL
  parser (`urllib.parse`), not a regex, so this parses out the true host
  correctly rather than just flagging the `@` character in isolation.
- `url_is_ip_literal` (bool) — host is a raw IP address
  (`http://185.23.4.9/...`) instead of a domain name. Legitimate SMS
  links essentially never do this; near-guaranteed phishing-kit tell.
- `url_is_punycode` (bool) — host starts with `xn--` (IDN/homoglyph
  attack — e.g. a Cyrillic lookalike of `microsoft.com` that renders
  visually identical but isn't the real domain).
- `url_subdomain_depth` (int) — count of subdomain labels before the
  registrable domain. Phishing kits often chain a real-looking prefix in
  front of the actual host (`secure.login.verify.appleid.com.evil.tk` —
  the registrable domain is `evil.tk`, everything before it is
  decoration meant to be read at a glance and mistaken for the real
  brand's subdomain).
- `url_suspicious_tld` (bool) — registrable domain's TLD is on a small
  static list heavily overrepresented in phishing relative to legitimate
  SMS traffic (`.tk`, `.ml`, `.ga`, `.top`, `.xyz`, etc. — cheap/free
  registration and abuse are directly correlated). List-membership only,
  no lookup.

These are cheap enough to compute per-message at feature time (same
place `features/behavioral.py` already does per-message computation) and
don't need the domain-keyed cache below — they need no state beyond the
message's own text. **Both models can use these directly** (LightGBM
rule-pattern model per its "no embeddings, canonical + behavioral"
scope, and the anomaly model's behavioral inputs) without waiting on the
async enrichment pipeline to exist at all — a smaller first slice worth
shipping before the domain-reputation tier below.

**Domain-reputation tier (deeper, needs external lookups).**
Regex-extract URLs out of canonical `text` (this is
already the cleaned, UDH-stripped, DCS-correct column both sources
produce — no new raw field needed, despite the raw upstream column being
named `content_self_decoded` on the SS7 side / `decoded_content` on the
SMPP side; `ingestion/*.py` already normalize both into `text_clean`).
Reduce each URL to its registered domain (`tldextract`, not a new
architectural component — just a URL-parsing lib) before any lookup;
lookups and caching are keyed by domain, not by full URL, since spam
campaigns route many message-level URLs (redirect shorteners, per-victim
tokens) through the same handful of domains.

**Why this must be async, not inline at inference:** every lookup listed
(domain age/WHOIS, SSL cert state, DNS record, redirect count, web
traffic rank, PageRank-equivalent, Google index status, Certificate
Transparency first-seen date — see below) is a live network call to a
third party, seconds-scale latency, sometimes rate-limited.
Inference must stay fast and point-in-time — so, same as the "no Kafka,
no Redis, batch not streaming" rule already governing behavioral
features, this becomes a **separate batch/queue job, not a request-path
call**:

```
extract domains (batch, from ingested text)
  -> queue table (SQLite, same pattern as everything else here - no new
     broker component)
  -> enrichment worker: WHOIS / DNS / SSL / redirect-follow / web-traffic
     + index-status lookups, one domain at a time, on its own schedule
  -> cache result keyed by domain (age, SSL state, DNS record present,
     redirect count, traffic rank, page rank, google-indexed, last
     checked timestamp)
  -> inference only ever does a fast cache read, never a live external
     call
```

A domain with no cache entry yet (never enriched) or a stale one past
some TTL scores as a neutral/unknown value, not as "safe" — never
implicitly treat missing enrichment as a clean signal.

**Domain-side behavioral feature (cross-sender fan-out).** Same shape as
`sender_dest_sequential_ratio_1hr` in section 3, but keyed by domain
instead of sender: `domain_distinct_senders_1hr` (or 24hr) — a phishing
domain typically gets reused across many spoofed/rotating sender IDs in
a short window, where a legitimate domain linked in one business's SMS
traffic sees one sender (or a small stable set). Fits the same
`domain_reputation` FeatureView below rather than a third entity, since
it's still batch-refreshed and keyed by domain.

**Feast shape.** This needs a *second* entity alongside the existing
`sender_id` (`source|originator`): a `domain` entity, join key = the
registered domain string. New `FeatureView` `domain_reputation`,
batch-refreshed by the enrichment worker exactly like
`sender_behavioral_stats` is refreshed by `behavioral_snapshot.py` — same
`scripts/refresh_feast.py`-style materialize step, just a second
snapshot source.

A single message can contain zero, one, or several URLs across
potentially different domains, and Feast's online lookup is a
per-entity-row API — it can fetch each extracted domain's cached row,
but it can't itself reduce "several domains on one message" to one
scalar. That reduction has to happen in serving code (same place
`serving/feature_lookup.py` already sits), the same way
`sender_repeat_content_ratio` already needs the incoming message's own
text at request time and gets it via an on-demand feature view: extract
this request's domains -> look each up -> fold into worst-case scalars
the model actually consumes, e.g. `url_count`, `url_min_domain_age_days`,
`url_any_no_ssl`, `url_max_redirect_count`, `url_any_not_google_indexed`.
"Worst-case across URLs in this message" (not average) is the right
reduction — one malicious link should not get diluted by other benign
ones in the same message.

**New dependencies needed**, each single-purpose and justified (per
CLAUDE.md's "don't introduce dependencies without a reason"):
`tldextract` (domain extraction), `dnspython` (DNS record lookup),
`python-whois` (domain age), plus whatever surfaces SSL state and
redirect-following (`httpx`/`ssl` stdlib is likely enough for both — no
new lib needed there). Web-traffic-rank and PageRank-equivalent sources
need picking once we know what's actually reachable from this
environment (Alexa/PageRank as originally conceived are defunct;
Tranco or a similar open ranking list is the modern substitute) — treat
that as an open question, not a blocker on the rest of this plan.
A Certificate Transparency log query (`crt.sh` or similar) for a
domain's earliest-seen cert is worth adding alongside WHOIS domain age —
it's often faster/more reliable for freshly-registered phishing domains,
and doesn't break the way WHOIS increasingly does under
privacy-redacted registrations.

## 2. WAP Push indication

**Current gap, found while checking this:** `app_dest_port` /
`app_src_port` were already investigated for the SMPP op-4 rows this
pipeline ingests and are **entirely NULL there** — see
`ingestion/smpp.py`'s "Checked against real op-4 rows and deliberately
NOT included" block. So the raw port columns are not usable signal *for
the traffic this pipeline currently keeps*. Two real possibilities,
not yet resolved:
- WAP Push / binary-addressed messages arrive under a different SMPP
  operation type than op-4 `submit_sm` and are being filtered out
  upstream before this pipeline ever sees them — in which case detecting
  them means widening the op-4 filter, a real scope decision, not a
  features-only change.
- Or this specific CDR sample genuinely carries no WAP Push traffic.

Don't build the port-based feature against data confirmed to be null —
confirm which of the two is true first (check operation types present
in the raw SMPP source files beyond op-4) before touching the filter.

**UDH-based path (usable today).** Prasad's point about the UDH header
is the more promising angle, and it reuses existing machinery:
`ingestion/smpp.py`'s `_parse_udh` already walks the UDH's
Information-Element structure, but today only extracts the two
concatenation IEs (0x00/0x08). The GSM 03.40 **Application Port
Addressing** IEs (0x04 = 8-bit port, 0x05 = 16-bit port) are a different
IE type in the same header and would tell us a message is
binary/port-addressed (the WAP Push delivery mechanism) directly from
content bytes already being parsed — no dependency on the null raw port
columns at all. Concretely: extend `_parse_udh` to also recognize those
IEs and return the addressed port(s) when present; add a canonical-ish
feature `has_wap_port_udh` (bool) / `udh_dest_port` (nullable int)
alongside the existing `concat_*` columns, same place, same pattern.
SS7 has no UDH-stripping today (`ingestion/ss7.py`) — check whether SS7's
raw content carries the same IE structure before assuming this feature
is SMPP-only.

## 3. Destination-address sequence clustering

This is a **sender-side** behavioral feature, not a new entity — it fits
directly into the existing `sender_id`-keyed feature views. For a given
sender in its trailing window (same 1hr window `sender_behavioral_stats`
already uses), look at the *numeric* destinations messaged and measure
how sequential/adjacent they are (e.g. sort the window's destination
numbers, compute the fraction of consecutive gaps within some small
threshold — mostly-adjacent numbers is the mass/bulk non-targeted
signal; scattered numbers is normal targeted traffic). Add as
`sender_dest_sequential_ratio_1hr` to `features/behavioral_snapshot.py`'s
snapshot builder and `feature_repo/definitions.py`'s
`sender_behavioral_stats` schema — stored, not on-demand, since it only
needs prior messages from that sender, all knowable ahead of the
incoming request (same category as `sender_msgs_last_1hr`, unlike
`sender_repeat_content_ratio_1hr` which needs the *current* message).

If destination numbering in this CDR sample turns out too sparse/random
to show real sequential clustering (worth checking early, same spirit as
the op-4/WAP check above) — say so plainly rather than shipping a
feature with no signal, and note that here.

## 4. Text keyword / brand & sender-ID heuristics

Two more cheap, no-external-call signals, same computation tier as
section 1's lexical URL features (per-message, from columns already in
the canonical schema):

- **Urgency/brand keyword hits in `text`** — static keyword/regex list
  ("verify", "suspended", "OTP", "KYC", "account locked", a specific
  brand name, etc.). `text` is already canonical, so this is a pure
  string-matching feature, no new column. Keep the keyword list itself
  externally editable (a config list, not hardcoded inline), since it's
  the kind of thing that needs updating as campaigns evolve — same spirit
  as `config/settings.py`'s existing tunables.
- **Sender-ID type mismatch** — `originator`/`originator_ton`/
  `originator_npi` already exist (SMPP) and distinguish business
  sender-IDs from real MSISDNs; add a feature flagging when an
  alphanumeric sender ID doesn't match any known brand allow-list, or
  when a long-code/international MSISDN appears where a short-code would
  be expected. This is a brand-impersonation proxy, not a hard rule —
  legitimate small businesses and aggregators use long codes too, so it
  contributes a soft signal, not a block.

Both fit as additional **behavioral/canonical** features per-message,
same as section 1's lexical tier — no new entity, no async component.

## Where these land in the two-score model

Per `CLAUDE.md`: no embeddings in the rule-pattern model, and the
anomaly model already takes "MiniLM embedding + behavioral + FAISS".
All four feature families above are canonical/behavioral in kind (not
embeddings), so they're additions to the shared **behavioral** feature
set both models already draw from — not a third model, not a change to
either model's architecture. `rule_pattern_score` and `anomaly_score`
stay separate as always; these features can move either score and a
disagreement between them stays meaningful exactly as it does today.

## Open questions to resolve before implementation

- Web-traffic-rank / PageRank-equivalent source — Alexa/original PageRank
  are defunct; pick a reachable substitute (e.g. Tranco) or drop those
  two sub-signals if nothing reachable substitutes cleanly.
- Whether WAP Push / binary-addressed traffic exists at all in the
  op-4-filtered SMPP data this pipeline currently ingests, or requires
  widening the `ingestion/smpp.py` operation-type filter.
- Whether SS7's content bytes carry the same UDH IE structure SMPP's do,
  before assuming the port-addressing feature generalizes across both
  sources.
- Whether real destination numbering in this sample shows exploitable
  sequential clustering at all.
- Where the shortener/suspicious-TLD/urgency-keyword static lists live
  and who maintains them (a `config/`-style tunable list, per existing
  convention, not hardcoded inline) — and how often they need updating
  as campaigns evolve.
- What a "known brand allow-list" for sender-ID mismatch actually draws
  from in this environment — no such list exists yet in the repo.
