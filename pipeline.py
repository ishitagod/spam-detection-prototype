"""
Top-level orchestrator, chaining every stage that actually exists today:

    ingestion.run_ingest.run_ingestion()          raw CDRs -> canonical features + labels
    features.message_reassembly.run_reassembly()  per-part rows -> one row per logical message

Deliberately does NOT stub out stages that don't exist yet (a behavioral-
features stage, models/, serving/) - CLAUDE.md's next-steps list orders
those (FAISS, then Isolation Forest, then LightGBM, then FastAPI, then
LIME); add each stage here when its script lands, not before. A "full"
pipeline with placeholder steps for unbuilt stages tends to guess their
shape wrong. (features/behavioral.py previously ran as Stage 2 here -
removed while it's being redone; add its replacement back once it exists,
reading FROM message_reassembly's output, not raw per-part features - see
message_reassembly.py's module docstring for why that ordering matters.)

Usage:
    python pipeline.py
    python pipeline.py --raw_dir data/raw --out_dir data/processed
"""
import argparse
from pathlib import Path

from features.message_reassembly import run_reassembly
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
