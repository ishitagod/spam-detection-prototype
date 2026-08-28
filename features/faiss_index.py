"""
FAISS near-duplicate features: for each message, how many near-identical
messages were sent in the trailing window(s) before it, and by how many
distinct senders. Point-in-time correct (only matches STRICTLY BEFORE
the message being scored count - same convention as behavioral.py).

Consumes features/text_embeddings.py's output (embeddings.npy +
embeddings_id_map.parquet) - no new embedding work here, just search.

THREE OUTPUT FEATURES PER WINDOW, AND BOTH WINDOWS MATTER:
  - near_dup_match_count_<window>: raw count of near-dup matches. Alone,
    this can't tell a coordinated spam blast from a bank sending the
    same OTP template to thousands of real customers in one hour - both
    produce a high count.
  - near_dup_distinct_senders_<window>: how many DIFFERENT sender IDs
    are behind those matches. A bank's OTP traffic comes from ONE
    sender ID -> low distinct-sender count despite the high match
    count. A coordinated campaign spreads the same template across MANY
    sender IDs -> high distinct-sender count. That divergence is the
    actual signal, and it's something no single-sender behavioral
    feature (sender_repeat_content_ratio_1hr etc.) can see, since those
    only look within one sender's own history.
  - near_dup_max_similarity_<window>: highest similarity among
    qualifying matches (0.0 if none) - secondary signal, mainly useful
    to distinguish "one very close match" from "many moderately close
    ones."
  - TWO windows (short/long, config/settings.py), not one: a short
    window (1hr) catches bursts; a long one (24hr) catches paced-out
    campaigns that deliberately never cluster within an hour to evade
    the short window. Neither window alone is a complete defense
    against an arbitrarily patient adversary - this raises the cost of
    evasion, it doesn't eliminate it. This feature also never has to
    work alone: Isolation Forest sees it jointly with embeddings and
    behavioral history, so a message that evades every near-dup window
    can still surface as anomalous on other axes.

Uses FAISS range_search, not a fixed top-K search: a fixed K would
undercount matches for very prolific senders (real data has senders with
hundreds of thousands of messages - see behavioral.py's docstring).
range_search returns EVERY match above the similarity threshold, however
many there are. ONE range_search call covers every configured window -
windows only change which of the SAME raw matches get counted per
message, so scoring N windows costs one index build + one search, not N
of them.

SCALES INDEPENDENTLY OF TOTAL CORPUS SIZE via
compute_near_dup_features_chunked(): rather than holding the entire
(growing) historical corpus in one FAISS index, it sorts by timestamp
and processes bounded, sequential time-chunks, each scored via
compute_near_dup_features() UNCHANGED. A message anywhere in a chunk has
a lookback that starts no earlier than chunk_start - max(window), so
prepending exactly that much "buffer" before each chunk is always
sufficient for every row in it - see that function's docstring. This
keeps memory bounded by chunk_size + buffer_size regardless of how much
history has accumulated, the same way a growing production corpus
wouldn't be held in memory all at once.
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config.settings import (
    FAISS_CHUNK_SIZE,
    FAISS_NEAR_DUP_THRESHOLD,
    FAISS_NEAR_DUP_WINDOW_LONG,
    FAISS_NEAR_DUP_WINDOW_SHORT,
)
from features.behavioral import _window_timedelta64

REQUIRED_ID_MAP_COLS = ["message_key", "originator", "timestamp"]

COL_MATCH_COUNT = "near_dup_match_count"
COL_MAX_SIM = "near_dup_max_similarity"
COL_DISTINCT_SENDERS = "near_dup_distinct_senders"

DEFAULT_WINDOWS = {
    "1hr": _window_timedelta64(FAISS_NEAR_DUP_WINDOW_SHORT),
    "24hr": _window_timedelta64(FAISS_NEAR_DUP_WINDOW_LONG),
}


_gpu_resources = None  # lazily created, process-wide - one GPU resource
# pool reused across every chunk's index build, not re-allocated per call.


def _to_gpu(index: "faiss.Index") -> "faiss.Index":
    """
    Moves a CPU FAISS index onto GPU 0 - only if a GPU-enabled faiss
    build is actually installed. requirements.txt pins faiss-cpu by
    default (works everywhere, incl. this sandbox); requirements-gpu.txt
    swaps in a GPU build on a machine that has one - see that file's
    comments, package naming has moved around across faiss releases.
    faiss-cpu has no `StandardGpuResources` attribute at all, so this
    degrades to the CPU index with a clear print rather than crashing -
    same "disclose, don't fake certainty" principle as
    text_embeddings.py's sample-info file.
    """
    global _gpu_resources
    if not hasattr(faiss, "StandardGpuResources"):
        print("  --gpu requested but faiss-gpu isn't installed (faiss.StandardGpuResources missing) - using CPU index.")
        return index
    if _gpu_resources is None:
        _gpu_resources = faiss.StandardGpuResources()
    try:
        return faiss.index_cpu_to_gpu(_gpu_resources, 0, index)
    except Exception as e:
        print(f"  Could not move FAISS index to GPU ({e}) - using CPU index.")
        return index


def build_index(embeddings: np.ndarray, use_gpu: bool = False) -> "faiss.Index":
    """Flat inner-product index - exact search, no approximation. Fine at
    the scale compute_near_dup_features_chunked() bounds each call to
    (chunk_size + buffer, not the full corpus); revisit (e.g. IVF) only
    if benchmarking at that bounded scale shows it's actually needed -
    IVF's benefit shrinks once chunking already keeps N small.
    `use_gpu`: see _to_gpu() - falls back to CPU cleanly if unavailable."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    if use_gpu:
        index = _to_gpu(index)
    index.add(embeddings)
    return index


def compute_near_dup_features(
    embeddings: np.ndarray,
    id_map: pd.DataFrame,
    threshold: float = FAISS_NEAR_DUP_THRESHOLD,
    windows: dict[str, np.timedelta64] | None = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    `id_map` must have `message_key`, `originator`, `timestamp` - same
    row order as `embeddings` (row i in both = the same message).
    `windows`: name -> trailing-window duration (defaults to
    DEFAULT_WINDOWS, the configured 1hr/24hr pair). Returns one row per
    input message, same order, with 3 near_dup_* columns per window.
    `use_gpu`: see build_index()/_to_gpu().
    """
    windows = windows if windows is not None else DEFAULT_WINDOWS
    missing = [c for c in REQUIRED_ID_MAP_COLS if c not in id_map.columns]
    if missing:
        raise ValueError(f"id_map is missing required column(s): {missing}")

    n = len(id_map)
    timestamps = pd.to_datetime(id_map["timestamp"]).to_numpy()
    originators = id_map["originator"].astype(str).to_numpy()

    index = build_index(embeddings, use_gpu=use_gpu)
    try:
        lims, sims, matches = index.range_search(embeddings, threshold)
    except RuntimeError as e:
        # Not every GPU faiss build/index type supports range_search
        # (unlike CPU FlatIP, which always does) - this whole feature
        # depends on range_search specifically (see module docstring on
        # why, vs. a fixed top-K), so retry on CPU rather than crash the
        # whole pipeline over a GPU-search gap.
        if not use_gpu:
            raise
        print(f"  GPU index.range_search failed ({e}) - retrying this chunk on CPU.")
        index = build_index(embeddings, use_gpu=False)
        lims, sims, matches = index.range_search(embeddings, threshold)

    columns = {"message_key": id_map["message_key"].to_numpy()}
    for name in windows:
        columns[f"{COL_MATCH_COUNT}_{name}"] = np.zeros(n, dtype=np.int64)
        columns[f"{COL_MAX_SIM}_{name}"] = np.zeros(n, dtype=np.float64)
        columns[f"{COL_DISTINCT_SENDERS}_{name}"] = np.zeros(n, dtype=np.int64)

    for i in range(n):
        row_matches = matches[lims[i] : lims[i + 1]]
        row_sims = sims[lims[i] : lims[i + 1]]

        # Point-in-time, shared across every window: strictly before
        # message i. Excludes i itself automatically (age is never > 0
        # against its own timestamp).
        age = timestamps[i] - timestamps[row_matches]
        earlier = age > np.timedelta64(0, "ns")

        for name, window in windows.items():
            keep = earlier & (age <= window)
            if keep.any():
                columns[f"{COL_MATCH_COUNT}_{name}"][i] = keep.sum()
                columns[f"{COL_MAX_SIM}_{name}"][i] = row_sims[keep].max()
                columns[f"{COL_DISTINCT_SENDERS}_{name}"][i] = len(
                    set(originators[row_matches[keep]])
                )

    return pd.DataFrame(columns)


def compute_near_dup_features_chunked(
    embeddings: np.ndarray,
    id_map: pd.DataFrame,
    chunk_size: int = FAISS_CHUNK_SIZE,
    threshold: float = FAISS_NEAR_DUP_THRESHOLD,
    windows: dict[str, np.timedelta64] | None = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Same output as compute_near_dup_features() (verified equal in
    tests/test_faiss_index.py), computed without ever holding more than
    one bounded time-slice in memory - see module docstring.

    Sorts by timestamp once, then walks forward in `chunk_size`-row
    chunks. Each chunk is scored by calling compute_near_dup_features()
    UNCHANGED on [buffer + chunk], where buffer = every row within
    max(windows) before the chunk's first timestamp. Only the chunk's
    own rows are kept from each call - a buffer row's result here may be
    incomplete (its own true lookback could reach further back than
    THIS chunk's buffer covers), which is fine, because that row gets
    scored correctly on its own turn, when it becomes the chunk.
    """
    windows = windows if windows is not None else DEFAULT_WINDOWS
    n = len(id_map)
    timestamps = pd.to_datetime(id_map["timestamp"]).to_numpy()

    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_embeddings = embeddings[order]
    sorted_id_map = id_map.iloc[order].reset_index(drop=True)

    max_window = max(windows.values())

    n_chunks = -(-n // chunk_size)  # ceil division, for the progress line below
    print(f"    {n_chunks} chunk(s) of up to {chunk_size} rows each")

    chunks = []
    pos = 0
    chunk_num = 0
    while pos < n:
        chunk_num += 1
        chunk_end = min(pos + chunk_size, n)
        buffer_start = np.searchsorted(
            sorted_ts, sorted_ts[pos] - max_window, side="left"
        )
        print(
            f"    chunk {chunk_num}/{n_chunks}: rows {pos}-{chunk_end} "
            f"(+{pos - buffer_start} buffer row(s)) ..."
        )

        slice_result = compute_near_dup_features(
            sorted_embeddings[buffer_start:chunk_end],
            sorted_id_map.iloc[buffer_start:chunk_end].reset_index(drop=True),
            threshold=threshold,
            windows=windows,
            use_gpu=use_gpu,
        )
        chunk_offset = (
            pos - buffer_start
        )  # where the real chunk starts within this slice's output
        chunks.append(slice_result.iloc[chunk_offset:])
        pos = chunk_end

    combined = pd.concat(chunks, ignore_index=True)
    # Chunks were processed in timestamp-sorted order, not the caller's
    # original row order - restore it by message_key (a real join, not
    # positional permutation math) so this is trivially correct rather
    # than trivially easy to get subtly wrong.
    return (
        combined.set_index("message_key")
        .loc[id_map["message_key"].to_numpy()]
        .reset_index()
    )


def run_faiss_near_dup(
    source_dir: Path,
    messages_path: Path,
    out_path: Path,
    chunk_size: int = FAISS_CHUNK_SIZE,
    use_gpu: bool = False,
) -> None:
    """
    `source_dir` must already contain embeddings.npy + embeddings_id_map.parquet
    (features/text_embeddings.py's output). `messages_path` supplies
    `originator`, which text_embeddings.py's id_map doesn't carry (that
    module has no notion of "sender" - kept separate on purpose).
    """
    # Pinned to 1 thread as a diagnostic: a real crash
    # (STATUS_STACK_BUFFER_OVERRUN, exit -1073740791) was hit running this
    # against SMPP's full 5.5M-row embeddings for the first time - real
    # sender velocity here reaches 48,606 msgs/hr for one sender (vs a
    # 6,726 median), which at FAISS_NEAR_DUP_THRESHOLD=0.92 can make
    # range_search's uncapped match array blow up combinatorially for a
    # single bursty sender's window. Native OpenMP parallel range_search
    # crashing on Windows under that kind of load is a known failure
    # class for faiss-cpu; this rules it out (or doesn't) before touching
    # chunk_size/threshold, which trade against the deliberate
    # "don't undercount prolific senders" choice documented above.
    faiss.omp_set_num_threads(1)

    source_dir = Path(source_dir)
    emb_path = source_dir / "embeddings.npy"
    id_map_path = source_dir / "embeddings_id_map.parquet"
    if not emb_path.exists() or not id_map_path.exists():
        print(
            f"No embeddings found in {source_dir} - run features/text_embeddings.py first."
        )
        return

    print(f"Loading {emb_path} ...")
    embeddings = np.load(emb_path)
    id_map = pd.read_parquet(id_map_path)

    print(f"Loading {messages_path} for originator ...")
    messages = pd.read_csv(
        messages_path, low_memory=False, usecols=["source", "record_id", "originator"]
    )
    messages["source"] = messages["source"].astype(str)
    messages["record_id"] = messages["record_id"].astype(str)
    messages["message_key"] = messages["source"] + "|" + messages["record_id"]
    id_map = id_map.merge(
        messages[["message_key", "originator"]], on="message_key", how="left"
    )

    print(
        f"Computing near-dup features for {len(id_map)} message(s) "
        f"(chunk_size={chunk_size}, use_gpu={use_gpu}) ..."
    )
    result = compute_near_dup_features_chunked(
        embeddings, id_map, chunk_size=chunk_size, use_gpu=use_gpu
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    for name in DEFAULT_WINDOWS:
        col = f"{COL_MATCH_COUNT}_{name}"
        print(
            f"  {int((result[col] > 0).sum())} / {len(result)} message(s) have at least one near-dup match ({name})"
        )
    print(f"Wrote {len(result)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="data/processed/SMPP")
    parser.add_argument(
        "--messages_path",
        type=str,
        default="data/processed/SMPP/messages_with_behavioral.csv",
    )
    parser.add_argument(
        "--out_path", type=str, default="data/processed/SMPP/faiss_output.parquet"
    )
    parser.add_argument("--chunk_size", type=int, default=FAISS_CHUNK_SIZE)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU FAISS if installed (see requirements-gpu.txt) - falls "
        "back to CPU automatically, with a printed message, if a GPU build "
        "isn't actually available. Default: CPU (faiss-cpu, requirements.txt).",
    )
    args = parser.parse_args()
    run_faiss_near_dup(
        Path(args.source_dir),
        Path(args.messages_path),
        Path(args.out_path),
        chunk_size=args.chunk_size,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
