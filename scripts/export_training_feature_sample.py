"""
One-off inspection script: exports a small, real sample of the exact
feature columns/values LightGBM (rule_pattern_score) and Isolation Forest
(anomaly_score) each actually train on - so you can eyeball whether
content-rule flags (has_url, has_phone_number, ...) are really in there,
without waiting on a full multi-million-row training run.

Reads only the first `--n_rows` rows of messages_with_behavioral.csv (fast
CSV read, no full-file scan), then builds each model's real pre-model
feature frame using the same functions train.py itself calls - not a
reimplementation. Isolation Forest's embedding dims are pulled for just
this sample via embeddings_id_map.parquet + a memory-mapped read of
embeddings.npy, so it doesn't load the full multi-GB embeddings array.

Run:
    python -m scripts.export_training_feature_sample --source SS7 --n_rows 1000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.anomaly.data import CONTENT_FLAG_COLS, build_combined_frame
from models.rule_pattern.data import _base_feature_frame


def _load_sample(source_dir: Path, n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "messages_with_behavioral.csv", nrows=n_rows, low_memory=False)
    df["source"] = df["source"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    df["message_key"] = df["source"] + "|" + df["record_id"]
    # This CSV predates a later addition to CONTENT_FLAG_PATTERNS - same
    # "absent -> 0, not a required column" convention as the real loaders
    # in models/rule_pattern/data.py and models/anomaly/data.py.
    for col in CONTENT_FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


def _attach_embeddings(df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    id_map = pd.read_parquet(source_dir / "embeddings_id_map.parquet")
    id_map = id_map.reset_index().rename(columns={"index": "_emb_row"})
    keys = id_map.set_index("message_key")["_emb_row"]
    rows = df["message_key"].map(keys)
    have_emb = rows.notna()
    df = df[have_emb].copy()
    rows = rows[have_emb].astype(int).to_numpy()

    embeddings = np.load(source_dir / "embeddings.npy", mmap_mode="r")
    emb_sample = np.asarray(embeddings[rows])
    emb_df = pd.DataFrame(emb_sample, columns=[f"emb_{i}" for i in range(emb_sample.shape[1])], index=df.index)
    return pd.concat([df, emb_df], axis=1)


def _attach_faiss(df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    near_dup = pd.read_parquet(source_dir / "faiss_output.parquet")
    return df.merge(near_dup, on="message_key", how="inner")


def run(source: str, data_dir: Path, n_rows: int, out_dir: Path) -> None:
    source_dir = data_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading first {n_rows} row(s) of {source_dir / 'messages_with_behavioral.csv'} ...")
    sample = _load_sample(source_dir, n_rows)

    # --- LightGBM (rule_pattern_score): base features only, matching the
    # default (no --with_tfidf/--with_embeddings) training path.
    lgbm_frame = _base_feature_frame(sample)
    lgbm_frame.insert(0, "rule_flagged", sample["rule_flagged"])
    lgbm_frame.insert(0, "rule_evaluated", sample["rule_evaluated"])
    lgbm_path = out_dir / f"feature_sample_lightgbm_{source}.csv"
    lgbm_frame.to_csv(lgbm_path, index=False)
    print(f"LightGBM: {lgbm_frame.shape[1]} feature column(s), {len(lgbm_frame)} row(s) -> {lgbm_path}")
    print(f"  columns: {list(lgbm_frame.columns)}")

    # --- Isolation Forest (anomaly_score): pre-preprocessor engineered
    # frame (log1p'd/bucketed/one-hot already applied, PCA not yet - PCA
    # needs a much bigger fit sample to mean anything).
    with_emb = _attach_embeddings(sample, source_dir)
    with_faiss = _attach_faiss(with_emb, source_dir)
    if len(with_faiss) == 0:
        print(f"Isolation Forest: 0 row(s) with both embedding + FAISS coverage in this sample - skipping export.")
        return
    combined, embedding_cols, other_cols = build_combined_frame(with_faiss)
    iso_path = out_dir / f"feature_sample_isolation_forest_{source}.csv"
    combined.to_csv(iso_path, index=False)
    print(f"Isolation Forest: {combined.shape[1]} feature column(s), {len(combined)} row(s) -> {iso_path}")
    print(f"  non-embedding columns: {other_cols}")
    print(f"  + {len(embedding_cols)} raw embedding dim(s) ({embedding_cols[0]}..{embedding_cols[-1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, choices=["SMPP", "SS7"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n_rows", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="data/processed/feature_samples")
    args = parser.parse_args()
    run(args.source, Path(args.data_dir), args.n_rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
