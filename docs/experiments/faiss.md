# Experiment: FAISS near-duplicate features

Code: `features/faiss_index.py`. Consumed by the anomaly model - see
`docs/experiments/anomaly.md` and `docs/ml/modeling.md`.

## What it computes

For each message, how many near-identical messages were sent in the
trailing window(s) before it, and by how many distinct senders.
Point-in-time correct - only matches STRICTLY BEFORE the message being
scored count (same convention as `features/behavioral.py`).

Three output features per window:
- `near_dup_match_count_<window>` - raw count of near-dup matches. Alone,
  this can't tell a coordinated spam blast from a bank sending the same
  OTP template to thousands of real customers in one hour - both produce
  a high count.
- `near_dup_distinct_senders_<window>` - how many DIFFERENT sender IDs
  are behind those matches. A bank's OTP traffic comes from ONE sender
  ID -> low distinct-sender count despite the high match count. A
  coordinated campaign spreads the same template across MANY sender IDs
  -> high distinct-sender count. That divergence is the actual signal,
  and it's something no single-sender behavioral feature
  (`sender_repeat_content_ratio_1hr` etc.) can see, since those only look
  within one sender's own history.
- `near_dup_max_similarity_<window>` - highest similarity among
  qualifying matches (0.0 if none) - secondary signal, mainly useful to
  distinguish "one very close match" from "many moderately close ones."

## Why two windows, not one

TWO windows (1hr short / 24hr long), computed off one shared raw match
set: a short window catches bursts; a long one catches paced-out
campaigns that deliberately never cluster within an hour to evade the
short window. Verified on real data: **6,907/40,000 SMPP sample messages
(17.3%) have zero 1hr matches but real 24hr matches** - a scammer pacing
sends a few hours apart would evade a 1hr-only window entirely. Neither
window alone is a complete defense against an arbitrarily patient
adversary - this raises the cost of evasion, it doesn't eliminate it.
This feature also never has to work alone: Isolation Forest sees it
jointly with embeddings and behavioral history, so a message that evades
every near-dup window can still surface as anomalous on other axes.

## Search strategy: range_search, not fixed top-K

Uses FAISS `range_search`, not a fixed top-K search: a fixed K would
undercount matches for very prolific senders (real data has senders with
hundreds of thousands of messages). `range_search` returns EVERY match
above the similarity threshold, however many there are. ONE
`range_search` call covers every configured window - windows only change
which of the SAME raw matches get counted per message, so scoring N
windows costs one index build + one search, not N of them.

### Why search everything, then filter - not filter first

FAISS's index has no concept of "before this timestamp" - `IndexFlatIP`
is a static bag of vectors with no per-query time awareness built in.
Filtering *before* searching would require one of:
1. A separate index per message (each has its own cutoff) - building an
   index is itself work; doing it per-row is far more expensive than one
   search.
2. An incremental/streaming index, added to over time - correct for a
   live serving system, but for a *batch* job over historical data this
   means N separate index-updates + N separate searches instead of one
   index build + one batched search.
3. A per-query FAISS `IDSelector` - needs a different selector per query
   row (different cutoff per message), which the batched `range_search`
   call doesn't support; falling back to one un-batched call per message
   loses the whole benefit of doing the similarity math as one BLAS
   matrix multiply.

So the actual design: do the expensive part (the similarity math) exactly
once, batched across every message simultaneously. The time filter
(`age > 0 & age <= window`) is then just cheap numpy boolean masking over
the *results* - essentially free next to the similarity computation
itself.

The "filter before search" idea isn't skipped, it happens at a coarser
level: `compute_near_dup_features_chunked()` prunes down to
`[buffer_start:chunk_end]` via `np.searchsorted` (binary search on sorted
timestamps) before ever building an index for that chunk - a real
filter-before-search step, done once per *chunk* (shared cutoff) rather
than per *row* (which FAISS can't do cheaply in one batched call).

## Why FlatIP, not IVF/HNSW

`build_index()` uses `faiss.IndexFlatIP` - flat, exact inner-product
search, deliberately not an approximate index. IVF/HNSW trade exactness
for speed by only searching a subset of clusters/graph neighbors, which
is a real win once you're searching millions of vectors per call - but
`compute_near_dup_features_chunked()` already bounds every individual
`range_search` call to `chunk_size + buffer_size` (a config-controlled
constant, not the whole growing corpus), so each call is already cheap
and exact. Approximate indexing would add real complexity (tuning
`nlist`/`nprobe`, recall/speed tradeoffs, index training) to solve a
scaling problem chunking already solved. Revisit only if benchmarking at
that bounded scale shows Flat is actually too slow - hasn't happened yet.

## Chunked processing

`compute_near_dup_features_chunked()` scales independently of total
corpus size: rather than holding the entire (growing) historical corpus
in one FAISS index, it sorts by timestamp and processes bounded,
sequential time-chunks, each scored via `compute_near_dup_features()`
UNCHANGED. A message anywhere in a chunk has a lookback that starts no
earlier than `chunk_start - max(window)`, so prepending exactly that much
buffer before each chunk is always sufficient for every row in it. This
keeps memory bounded by `chunk_size + buffer_size` regardless of how much
history has accumulated - the same way a growing production corpus
wouldn't be held in memory all at once. Verified identical to the
unchunked result in `tests/test_faiss_index.py`.

Row order is restored afterward via a real join on `message_key` (not
positional-permutation math), so this is trivially correct rather than
trivially easy to get subtly wrong.

## Deferred: SIM-farming detection

One physical IMSI cycling through many apparent MSISDNs, confirmed
present in real SS7 data - 1,541 IMSIs already show >1 distinct
originator. This is a DIFFERENT signal, not a FAISS window - it's an
identity-linkage problem, not a content-similarity one. Belongs as a new
SS7-only behavioral feature keyed by `imsi` (not `virtual_imsi`, which is
routing-join plumbing, not a subscriber-identity field), mirroring
`sender_unique_destinations_1hr`'s pattern. Not yet built.
