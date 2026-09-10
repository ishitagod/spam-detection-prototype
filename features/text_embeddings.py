"""
Text embeddings: turns message's `text` into a fixed-length
vector that captures MEANING, not literal characters.
Both consume the SAME embeddings computed here rather than each
re-encoding text independently - one model, one pass, two downstream
readers. This is exactly why it's its own step rather than being folded
into either consumer (see CLAUDE.md's roadmap note).

MODEL: `paraphrase-multilingual-MiniLM-L12-v2` (config.settings.
TEXT_EMBEDDING_MODEL) via sentence-transformers - switched from the
original `all-MiniLM-L6-v2` pick once real data showed ~10-11% genuine
Malay/mixed content an English-only model would embed poorly (see
README.md's "Modeling plan" section for the measurement). Embeddings are
L2-normalized at encode time, so cosine similarity downstream reduces to
a plain dot product - what FAISS's inner-product index types expect.

DEDUPLICATION: real spam is repetitive by construction - one busy SMPP
sender's own last-hour text-frequency snapshot (docs/feature_catalog.md)
shows a single message repeated 211 times by that sender alone. Encoding
is the expensive step here (a transformer forward pass per text, unlike
behavioral.py's cheap numpy ops), so this module encodes each DISTINCT
text value exactly once and broadcasts the result back to every row
sharing that text, via `pd.factorize`. Not a micro-optimization - at
real data's duplication rate this is the difference between encoding a
few hundred thousand distinct strings and encoding all 8mm+ rows.

WHAT THIS DOES NOT DO: build the FAISS index itself, or train anything -
purely produces the embeddings matrix + a row-aligned identity map for
downstream steps to consume.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import TEXT_EMBEDDING_BATCH_SIZE, TEXT_EMBEDDING_MODEL

REQUIRED_COLS = ["source", "record_id", "timestamp", "text"]

_model = None  # lazily loaded, process-wide cache - loading weights off
# disk/HF cache costs real time, do it once regardless of how many times
# embed_texts() gets called. Both this module's CLI and the future
# FastAPI service (encoding one live message per request) share this.


def _select_device(requested: str | None) -> str:
    """
    Explicit device choice, not left to sentence-transformers' silent
    auto-detection.
    """
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(device: str | None = None):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        resolved_device = _select_device(device)
        print(
            f"  text_embeddings: loading {TEXT_EMBEDDING_MODEL} on device={resolved_device}"
        )
        _model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=resolved_device)
    return _model


def embed_texts(
    texts,
    batch_size: int = TEXT_EMBEDDING_BATCH_SIZE,
    model=None,
    device: str | None = None,
) -> np.ndarray:
    """
    Low-level, reusable encoder: any list/array/Series of strings in,
    float32 (N, D) L2-normalized embeddings out. No dedup here - that's
    compute_message_embeddings()'s job below, since dedup only makes
    sense when the caller can hand back per-row results; a single live
    inference request has nothing to dedup against.

    `model` is injectable (any object exposing sentence-transformers'
    `.encode()`/`.get_embedding_dimension()` interface) specifically so
    tests never need to load real weights just to exercise the
    surrounding pipeline logic - see tests/test_text_embeddings.py's fake
    model. `device` is ignored when `model` is injected (the caller
    already controls that object).
    """
    model = model or _get_model(device)
    texts = list(texts)
    if not texts:
        return np.zeros((0, model.get_embedding_dimension()), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 1000,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def compute_message_embeddings(
    messages: pd.DataFrame, model=None, device: str | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns (embeddings, id_map):
      - embeddings: float32 (N, D) array, row i = message i's embedding.
      - id_map: an N-row DataFrame, SAME row order, with `message_key`
        (source|record_id - same compound-key pattern as
        features/behavioral.py's sender_id, for the same reason:
        record_id alone is only unique WITHIN a source, not across both),
        plus source/record_id/timestamp - enough to trace any embedding
        back to its real message without carrying the full row (and its
        text) around downstream.
    """
    missing = [c for c in REQUIRED_COLS if c not in messages.columns]
    if missing:
        raise ValueError(f"messages is missing required column(s): {missing}")

    df = messages.copy()
    df["source"] = df["source"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    text = df["text"].fillna("")

    # Encode each DISTINCT text exactly once (see module docstring).
    # pd.factorize: codes[i] = which distinct value row i has, uniques =
    # the distinct values themselves - both vectorized, no Python loop.
    codes, uniques = pd.factorize(text, sort=False)
    n = len(df)
    print(
        f"  {n} message(s), {len(uniques)} distinct text(s) "
        f"({len(uniques) / max(n, 1):.1%} unique)"
    )
    unique_embeddings = embed_texts(uniques, model=model, device=device)
    embeddings = unique_embeddings[codes]

    id_map = pd.DataFrame(
        {
            "message_key": df["source"] + "|" + df["record_id"],
            "source": df["source"],
            "record_id": df["record_id"],
            "timestamp": df["timestamp"],
        }
    )
    return embeddings, id_map


def run_text_embeddings(
    messages_path: Path,
    out_dir: Path,
    model=None,
    sample_n: int | None = None,
    seed: int = 42,
    device: str | None = None,
) -> None:
    """
    `sample_n`: encode a random sample of this many rows instead of the
    full file. Prototype-scale default is to sample, and only commit to a full run
    once the downstream FAISS/Isolation Forest steps have validated the
    approach - see CLAUDE.md's roadmap. `seed` makes the sample
    reproducible run to run (same rows every time, not a fresh random
    draw each call).
    """
    messages_path = Path(messages_path)
    if not messages_path.exists():
        print(f"No messages file found at {messages_path}")
        return

    print(f"Loading {messages_path} ...")
    messages = pd.read_csv(
        messages_path,
        low_memory=False,
        usecols=lambda c: c in set(REQUIRED_COLS),
    )
    total_rows = len(messages)
    sampled = sample_n is not None and sample_n < total_rows
    if sampled:
        messages = messages.sample(n=sample_n, random_state=seed)
        print(
            f"SAMPLED: {sample_n} of {total_rows} total rows "
            f"(seed={seed}) - not the full dataset, see module docstring"
        )

    print(f"Computing embeddings for {len(messages)} message(s) ...")
    embeddings, id_map = compute_message_embeddings(
        messages, model=model, device=device
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    id_map_path = out_dir / "embeddings_id_map.parquet"
    np.save(emb_path, embeddings)
    id_map.to_parquet(id_map_path, index=False)
    print(f"Wrote {embeddings.shape} embeddings to {emb_path}")
    print(f"Wrote {len(id_map)} id-map rows to {id_map_path}")

    # Small, persistent disclosure that this was a sample, not a full
    # run - so nobody finds embeddings.npy later and assumes full
    # coverage without checking (same "disclose, don't fake certainty"
    # principle as the architecture plan's `confidence` field).
    info_path = out_dir / "embeddings_sample_info.txt"
    if sampled:
        info_path.write_text(
            f"SAMPLED: {sample_n} of {total_rows} total rows, seed={seed}\n"
            f"source file: {messages_path}\n"
            "This is NOT full coverage - re-run without --sample_n for "
            "the complete dataset.\n"
        )
    elif info_path.exists():
        # A prior run of this same out_dir was a sample; this run isn't -
        # remove the stale disclosure rather than leave a lie on disk.
        info_path.unlink()


def main():
    # SOURCES imported lazily (not at module top) so importing
    # text_embeddings.py for its functions doesn't also pull in
    # ingestion.run_ingest's pandas/ingestion-handler machinery unless
    # main() actually runs.
    from ingestion.run_ingest import SOURCES

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--messages_path",
        type=str,
        default=None,
        help="Explicit path to one messages_with_behavioral.csv, run exactly "
        "against that one file (requires --out_dir too). Default: unset - "
        "runs every source in ingestion.run_ingest.SOURCES (SMPP and SS7 "
        "today) under --processed_dir instead, same layout pipeline.py uses, "
        "so no explicit path is needed to cover both.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for the --messages_path run above. Required "
        "if --messages_path is given, ignored otherwise.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=list(SOURCES.keys()),
        default=None,
        help="Only run one source under --processed_dir. Default: every "
        f"source ({list(SOURCES.keys())}). Ignored if --messages_path is given.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Root of data/processed/<SOURCE>/... - used when --messages_path "
        "isn't given (see that flag).",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Encode a random sample of this many rows instead of the full file (see module docstring).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force a device instead of auto-detecting (see _select_device). "
        "Default: use CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    if args.messages_path:
        if not args.out_dir:
            parser.error("--out_dir is required when --messages_path is given")
        run_text_embeddings(
            Path(args.messages_path),
            Path(args.out_dir),
            sample_n=args.sample_n,
            seed=args.seed,
            device=args.device,
        )
        return

    # No explicit path - run every source (or just --source, if given)
    # under --processed_dir, same data/processed/<SOURCE>/... layout
    # pipeline.py's own per-source loop uses.
    processed_dir = Path(args.processed_dir)
    sources = [args.source] if args.source else list(SOURCES.keys())
    for source in sources:
        print(f"\n-- {source} --")
        run_text_embeddings(
            processed_dir / source / "messages_with_behavioral.csv",
            processed_dir / source,
            sample_n=args.sample_n,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
