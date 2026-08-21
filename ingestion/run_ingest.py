"""
File-by-file ingestion driver: walks data/raw/SMPP and data/raw/SS7,
applies each registered source's clean() + map_to_canonical() to each raw
CSV independently, and writes two OUTPUTS PER INPUT FILE - never merged,
never joined here:

  data/processed/<SOURCE>/features/<same_basename>.csv
      canonical feature columns only - safe to hand to any model, no
      label-source columns present at all (not just excluded by
      convention - they are physically absent from this file).

  data/processed/<SOURCE>/labels/<same_basename>.csv
      join key + rule_evaluated/rule_flagged/fraud_type. Used ONLY to
      build supervised "rule_pattern" training labels (join back to
      features on record_id/message_id when training that model). Never
      read by the unsupervised layer.

File-by-file (not one concatenated pass) because the raw files are large
(tens to ~300MB each) and this keeps memory bounded, makes a bad file fail
loudly without losing prior progress, and lets ingestion resume/parallelize
per file later if needed.

CSV round-trip caveat: rows with text_decode_failed=True get `text` set to
"" (not dropped - see ingestion/smpp.py's clean()), but CSV can't tell an
empty string apart from a missing value - both are written as nothing
between commas. Plain `pd.read_csv` on the features output will read those
rows back as NaN, silently re-introducing the exact problem the "" was
meant to fix. Use load_features_csv() below instead of pd.read_csv directly
when loading this output for training/embedding.

Usage:
    python -m ingestion.run_ingest
    python -m ingestion.run_ingest --raw_dir data/raw --out_dir data/processed
    python -m ingestion.run_ingest --source SMPP   # only one source
"""

import argparse
from pathlib import Path

import pandas as pd

from ingestion import smpp, ss7
from ingestion.base import SourceHandlers

# Registered as ONE pair per source (see ingestion/base.py) - a source only
# appears here once it has a real clean(), not just a mapper. That's what
# previously let SS7 silently have a mapper with no cleaner and no
# enforcement catching the gap.
SOURCES: dict[str, SourceHandlers] = {
    "SMPP": SourceHandlers(clean=smpp.clean, map_to_canonical=smpp.map_to_canonical),
    "SS7": SourceHandlers(clean=ss7.clean, map_to_canonical=ss7.map_to_canonical),
}


def load_features_csv(path: Path) -> pd.DataFrame:
    """
    Reads a data/processed/<SOURCE>/features/<file>.csv the way it actually
    needs to be read: plain pd.read_csv brings `text` back as NaN for every
    text_decode_failed=True row, since CSV can't distinguish an empty string
    from a missing value on write. text_decode_failed is the ground truth
    for "should this be empty" here - so refill text from it rather than
    guessing from the NaN alone (a text column can be legitimately empty
    for other future reasons too; this only re-fills what THIS pipeline
    marked as a decode failure).
    """
    df = pd.read_csv(path, low_memory=False)
    if "text" in df.columns and "text_decode_failed" in df.columns:
        df.loc[df["text_decode_failed"], "text"] = df.loc[
            df["text_decode_failed"], "text"
        ].fillna("")
    return df


def ingest_file(csv_path: Path, source: str, out_dir: Path) -> dict:
    handlers = SOURCES[source]
    # low_memory=False: these files have empty columns (sar_ref, virtual_gt,
    # etc.) that pandas' chunked type-sniffing otherwise flags with a
    # DtypeWarning and can mis-infer across chunk boundaries.
    raw = pd.read_csv(csv_path, low_memory=False)

    cleaned = handlers.clean(raw)
    cleaned["record_id"] = csv_path.stem + "#" + cleaned["record_id"]

    features, label_source = handlers.map_to_canonical(cleaned)

    features_out = out_dir / source / "features" / csv_path.name
    labels_out = out_dir / source / "labels" / csv_path.name
    features_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(features_out, index=False)
    label_source.to_csv(labels_out, index=False)

    return {
        "file": csv_path.name,
        "raw_rows": len(raw),
        "kept_rows": len(cleaned),
        "rule_evaluated": int(label_source["rule_evaluated"].sum()),
        "rule_flagged": int(label_source["rule_flagged"].sum(skipna=True)),
    }


def ingest_source(raw_dir: Path, out_dir: Path, source: str) -> list[dict]:
    source_dir = raw_dir / source
    csv_files = sorted(source_dir.rglob("*.csv"))
    if not csv_files:
        print(f"  No CSVs found under {source_dir}")
        return []

    results = []
    for i, csv_path in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] {csv_path.relative_to(raw_dir)} ...", end=" ")
        try:
            result = ingest_file(csv_path, source, out_dir)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(
            f"{result['kept_rows']}/{result['raw_rows']} rows kept, "
            f"{result['rule_evaluated']} labelled (supervised), "
            f"{result['rule_flagged']} rule-flagged"
        )
        results.append(result)
    return results


def run_ingestion(
    raw_dir: Path, out_dir: Path, sources: list[str] | None = None
) -> pd.DataFrame:
    """
    Callable entry point (as opposed to main()'s CLI/argparse wrapper) so
    pipeline.py can chain this stage into the rest of the pipeline without
    shelling out. Returns the manifest as a DataFrame (empty if nothing was
    ingested) and also writes it to out_dir/ingestion_manifest.csv, same as
    the CLI path did.
    """
    sources = sources if sources is not None else list(SOURCES.keys())

    manifest_rows = []
    for source in sources:
        print(f"\n{source}:")
        results = ingest_source(raw_dir, out_dir, source)
        for r in results:
            r["source"] = source
        manifest_rows.extend(results)

    if not manifest_rows:
        print("\nNothing ingested.")
        return pd.DataFrame()

    manifest = pd.DataFrame(manifest_rows)[
        ["source", "file", "raw_rows", "kept_rows", "rule_evaluated", "rule_flagged"]
    ]
    manifest_path = out_dir / "ingestion_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\nTotals:")
    totals = manifest.groupby("source")[
        ["raw_rows", "kept_rows", "rule_evaluated", "rule_flagged"]
    ].sum()
    print(totals.to_string())
    print(f"\nManifest written to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument(
        "--source",
        type=str,
        choices=list(SOURCES.keys()),
        default=None,
        help=f"Only ingest one source. Default: all wired-in sources ({list(SOURCES.keys())}).",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source else None
    run_ingestion(Path(args.raw_dir), Path(args.out_dir), sources)


if __name__ == "__main__":
    main()
