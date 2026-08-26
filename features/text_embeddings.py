"""
Text embeddings: turns each logical message's `text` into a fixed-length
vector that captures MEANING, not literal characters - "WIN A PRIZE NOW"
and "You've won a prize!!" end up close together in this vector space
even though the strings barely share any characters. Downstream
consumers, per CLAUDE.md's roadmap:
  - a FAISS near-duplicate index (not yet built) - nearest-neighbor
    search over these vectors, for burst/campaign detection.
  - Isolation Forest (not yet built) - scores [embedding + behavioral
    features] jointly, per docs/feature_catalog.md and README.md's
    modeling plan.
Both consume the SAME embeddings computed here rather than each
re-encoding text independently - one model, one pass, two downstream
readers. This is exactly why it's its own step rather than being folded
into either consumer (see CLAUDE.md's roadmap note).

MODEL: `all-MiniLM-L6-v2` via sentence-transformers
(config/settings.py's TEXT_EMBEDDING_MODEL) - prototype choice, smallest
footprint of the candidates evaluated (Distil-mBERT/XLM-R are production
options, not used here). Embeddings are L2-normalized at encode time, so
cosine similarity downstream reduces to a plain dot product - what
FAISS's inner-product index types expect.

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
downstream steps to consume. It also does not run through Feast/get
served online, unlike sender-behavioral features - a message's embedding
is a pure function of ITS OWN text (no sender history needed), so at
real inference time the future FastAPI service just calls embed_texts()
directly on the one incoming message. Nothing to precompute per sender
ahead of time.
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


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    return _model


def embed_texts(texts, batch_size: int = TEXT_EMBEDDING_BATCH_SIZE, model=None) -> np.ndarray:
    """
    Low-level, reusable encoder: any list/array/Series of strings in,
    float32 (N, D) L2-normalized embeddings out. No dedup here - that's
    compute_message_embeddings()'s job below, since dedup only makes
    sense when the caller can hand back per-row results; a single live
    inference request has nothing to dedup against.

    `model` is injectable (any object exposing sentence-transformers'
    `.encode()` interface) specifically so tests never need to load real
    weights just to exercise the surrounding pipeline logic - see
    tests/test_text_embeddings.py's fake model.
    """
    model = model or _get_model()
    texts = list(texts)
    if not texts:
        # Try the current sentence-transformers 6.x name first, fall back
        # to the deprecated one - kept for compatibility with any
        # injected test double (tests/test_text_embeddings.py's
        # FakeModel) that only implements the older interface.
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return np.zeros((0, get_dim()), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 1000,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def compute_message_embeddings(
    messages: pd.DataFrame, model=None
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
    unique_embeddings = embed_texts(uniques, model=model)
    embeddings = unique_embeddings[codes]

    id_map = pd.DataFrame({
        "message_key": df["source"] + "|" + df["record_id"],
        "source": df["source"],
        "record_id": df["record_id"],
        "timestamp": df["timestamp"],
    })
    return embeddings, id_map


def run_text_embeddings(
    messages_path: Path,
    out_dir: Path,
    model=None,
    sample_n: int | None = None,
    seed: int = 42,
) -> None:
    """
    `sample_n`: encode a random sample of this many rows instead of the
    full file - real wall-clock cost here is ~14ms PER DISTINCT TEXT on
    CPU (measured, not estimated), so the full ~8.2M-row dataset is a
    ~21hr job. Prototype-scale default is to sample, and only commit to
    a full run once the downstream FAISS/Isolation Forest steps have
    validated the approach - see CLAUDE.md's roadmap. `seed` makes the
    sample reproducible run to run (same rows every time, not a fresh
    random draw each call).
    """
    messages_path = Path(messages_path)
    if not messages_path.exists():
        print(f"No messages file found at {messages_path}")
        return

    print(f"Loading {messages_path} ...")
    messages = pd.read_csv(
        messages_path, low_memory=False,
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
    embeddings, id_map = compute_message_embeddings(messages, model=model)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--messages_path", type=str,
        default="data/processed/SMPP/messages_with_behavioral.csv",
    )
    parser.add_argument("--out_dir", type=str, default="data/processed/SMPP")
    parser.add_argument(
        "--sample_n", type=int, default=None,
        help="Encode a random sample of this many rows instead of the full file (see module docstring).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_text_embeddings(
        Path(args.messages_path), Path(args.out_dir),
        sample_n=args.sample_n, seed=args.seed,
    )


if __name__ == "__main__":
    main()
