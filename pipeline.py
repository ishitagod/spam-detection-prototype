"""
Top-level orchestrator, chaining every stage that actually exists today:

    ingestion.run_ingest.run_ingestion()          raw CDRs -> canonical features + labels
    features.message_reassembly.run_reassembly()  per-part rows -> one row per logical message
    features.behavioral.run_behavioral()          reassembled messages -> + sender-behavioral columns
    features.text_embeddings.run_text_embeddings()  reassembled messages -> MiniLM embeddings + id map
    features.faiss_index.run_faiss_near_dup()      embeddings -> near-dup match features (1hr/24hr)

Deliberately does NOT stub out stages that don't exist yet (models/,
serving/) - CLAUDE.md's next-steps list orders those (MiniLM embeddings,
then FAISS, then Isolation Forest, then LightGBM, then FastAPI, then
LIME); add each stage here when its script lands, not before. A "full"
pipeline with placeholder steps for unbuilt stages tends to guess their
shape wrong.

Usage:
    python pipeline.py
    python pipeline.py --raw_dir data/raw --out_dir data/processed
"""
import argparse
from pathlib import Path

from features.behavioral import run_behavioral
from features.faiss_index import run_faiss_near_dup
from features.message_reassembly import run_reassembly
from features.text_embeddings import run_text_embeddings
from ingestion.run_ingest import run_ingestion


def run(raw_dir: Path, out_dir: Path) -> None:
    print("=== Stage 1: ingestion ===")
    manifest = run_ingestion(raw_dir, out_dir)
    if manifest.empty:
        print("Ingestion produced nothing.")
        return

    print("\n=== Stage 2: message reassembly ===")
    # Per source (manifest["source"] is whatever run_ingestion actually
    # produced output for - SMPP today, SS7 once wired in, both once both
    # are) - never merged across sources here, matches run_ingestion's own
    # "two outputs per input file, never joined" convention.
    for source in sorted(manifest["source"].unique()):
        print(f"\n-- {source} --")
        run_reassembly(
            features_dir=out_dir / source / "features",
            labels_dir=out_dir / source / "labels",
            out_path=out_dir / source / "messages.csv",
        )

    print("\n=== Stage 3: behavioral features ===")
    for source in sorted(manifest["source"].unique()):
        print(f"\n-- {source} --")
        run_behavioral(
            messages_path=out_dir / source / "messages.csv",
            out_path=out_dir / source / "messages_with_behavioral.csv",
        )

    print("\n=== Stage 4: text embeddings ===")
    # Consumed by two not-yet-built steps (FAISS near-dup index,
    # Isolation Forest) - see features/text_embeddings.py's module
    # docstring for why this is computed once here rather than by each
    # consumer separately. Loads a real MiniLM model on first call in
    # this process - the slowest stage per-message by far, unlike
    # Stages 1-3's pure pandas/numpy work.
    for source in sorted(manifest["source"].unique()):
        print(f"\n-- {source} --")
        run_text_embeddings(
            messages_path=out_dir / source / "messages_with_behavioral.csv",
            out_dir=out_dir / source,
        )

    print("\n=== Stage 5: FAISS near-duplicate features ===")
    # Consumes Stage 4's embeddings directly - see features/faiss_index.py's
    # module docstring for why this is a separate stage (search, not
    # embedding work) and why it needs `messages_path` too (originator,
    # which text_embeddings.py's id_map deliberately doesn't carry).
    for source in sorted(manifest["source"].unique()):
        print(f"\n-- {source} --")
        run_faiss_near_dup(
            source_dir=out_dir / source,
            messages_path=out_dir / source / "messages_with_behavioral.csv",
            out_path=out_dir / source / "faiss_output.parquet",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
