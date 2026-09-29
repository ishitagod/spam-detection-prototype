"""
Feature redundancy + relevance audit over the union of features both models
(rule_pattern LightGBM, anomaly Isolation Forest) actually consume.

  1. Spearman pairwise across the full feature set, |rho| >= --redundancy_thresh.
  2. Boundedness-artifact check for every flagged pair involving a bounded
     feature (ratios in [0,1], similarity scores, entropy): recompute on
     interior rows only and within message-count strata. A pair that only
     correlates because of mass points at 0/1 or a shared count denominator
     is NOT redundant.
  3. Feature-vs-label relevance (rule_flagged, rule_evaluated pool only):
     Spearman, univariate ROC-AUC, mutual information, flag lift. Rough
     signal only - low marginal association does not prove a feature useless.

Reuses the real builders (models/rule_pattern/data.py::_base_feature_frame,
models/anomaly/data.py::build_combined_frame) rather than reimplementing.
Content flags are recomputed from `text` with features/content_flags.py
because messages_with_behavioral.csv predates several CONTENT_FLAG_PATTERNS
entries (the loaders zero-fill those - reported below).
text_entropy is a CANDIDATE only (not a model feature, see
features/identity_baseline.py).

Run:  python -m scripts.analyze_feature_redundancy --out_dir <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import mutual_info_score, roc_auc_score

from features.content_flags import compute_content_flags
from features.identity_baseline import shannon_entropy
from models.anomaly.data import (
    BEHAVIORAL_COLS, CONTENT_FLAG_COLS, IMSI_DISTINCT_ORIG_COL, NEAR_DUP_COLS,
    PLATFORM_TOKEN_COL, SENDER_VELOCITY_ZSCORE_COL, build_combined_frame,
)
from models.rule_pattern.data import _base_feature_frame

RAW_COLS = (
    ["source", "record_id", "originator", "text", "dcs", "text_decode_failed",
     "rule_evaluated", "rule_flagged", PLATFORM_TOKEN_COL, IMSI_DISTINCT_ORIG_COL,
     SENDER_VELOCITY_ZSCORE_COL] + BEHAVIORAL_COLS
)

# Features bounded by construction - candidates for boundedness artifacts.
BOUNDED = {
    "sender_repeat_content_ratio_1hr", "sender_recipient_diversity_ratio_5min",
    "sender_recipient_diversity_ratio_1hr", "gated_diversity_5min", "gated_diversity_1hr",
    "near_dup_max_similarity_1hr", "near_dup_max_similarity_24hr", "text_entropy",
}
# Shared-denominator driver for ratio features (ratio = count/msgs).
STRATA_COL = "sender_msgs_last_1hr"


def load_sample(source, data_dir, p_uniform, p_eval, chunk=500_000, seed=0):
    rng = np.random.default_rng(seed)
    path = data_dir / source / "messages_with_behavioral.csv"
    parts = []
    for ch in pd.read_csv(path, usecols=lambda c: c in set(RAW_COLS), chunksize=chunk,
                          dtype={"record_id": str, "source": str, "originator": str}):
        if PLATFORM_TOKEN_COL in ch.columns:
            ch = ch[~ch[PLATFORM_TOKEN_COL].fillna(False).astype(bool)]
        u = rng.random(len(ch)) < p_uniform
        ev = ch["rule_evaluated"].astype(bool).to_numpy()
        e = ev & (rng.random(len(ch)) < p_eval)
        keep = u | e
        sub = ch[keep].copy()
        sub["in_uniform"] = u[keep]
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df["rule_evaluated"] = df["rule_evaluated"].astype(bool)
    df["text_decode_failed"] = df["text_decode_failed"].astype(bool)
    df["message_key"] = df["source"] + "|" + df["record_id"]
    if IMSI_DISTINCT_ORIG_COL not in df.columns:
        df[IMSI_DISTINCT_ORIG_COL] = np.nan
    flags = compute_content_flags(df["text"])
    df = pd.concat([df.drop(columns=[c for c in CONTENT_FLAG_COLS if c in df.columns]), flags], axis=1)
    nd = pd.read_parquet(data_dir / source / "faiss_output.parquet")
    return df.merge(nd, on="message_key", how="left")  # left: keep rows, NaN = no FAISS coverage


def build_features(df):
    base = _base_feature_frame(df)  # LightGBM's real input frame (raw behavioral)
    base = base.drop(columns=[c for c in base.columns if c.startswith("source_")])
    combined, _, _ = build_combined_frame(df.assign(**{c: df[c].fillna(0) for c in NEAR_DUP_COLS}))
    iso = pd.DataFrame(index=df.index)
    for c in NEAR_DUP_COLS:
        iso[c] = df[c]  # raw (NaN where no FAISS row), log1p is rank-invariant anyway
    for c in combined.columns:
        if c.endswith("_known") or c.startswith("sender_age_bucket_"):
            iso[c] = combined[c].to_numpy()
    iso["gated_diversity_5min"] = combined["sender_recipient_diversity_ratio_5min"].to_numpy()
    iso["gated_diversity_1hr"] = combined["sender_recipient_diversity_ratio_1hr"].to_numpy()
    X = pd.concat([base, iso], axis=1)
    X["text_entropy"] = df["text"].map(shannon_entropy)  # candidate, not in either model
    X["is_smpp"] = (df["source"] == "SMPP").astype(int)
    return X.astype("float64")


def redundant_pairs(corr, thresh):
    rows = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iat[i, j]
            if pd.notna(r) and abs(r) >= thresh:
                rows.append((cols[i], cols[j], r))
    return pd.DataFrame(rows, columns=["a", "b", "rho"]).sort_values("rho", key=abs, ascending=False)


def clusters(pairs, cols):
    idx = {c: i for i, c in enumerate(cols)}
    m = np.zeros((len(cols), len(cols)), dtype=int)
    for a, b in zip(pairs["a"], pairs["b"]):
        m[idx[a], idx[b]] = 1
    n, lab = connected_components(m, directed=False)
    out = {}
    for c, l in zip(cols, lab):
        out.setdefault(l, []).append(c)
    return [v for v in out.values() if len(v) > 1]


def spearman(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 50 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return np.nan
    return a[ok].rank().corr(b[ok].rank())


def boundedness_check(X, a, b, thresh):
    """Re-test a high-|rho| pair for mass-point / shared-denominator artifacts."""
    def at_bound(s):
        lo, hi = s.min(), s.max()
        return ((s == lo) | (s == hi)).mean()

    interior = pd.Series(True, index=X.index)
    for c in (a, b):
        if c in BOUNDED:
            interior &= (X[c] > X[c].min()) & (X[c] < X[c].max())
    r_int = spearman(X.loc[interior, a], X.loc[interior, b]) if interior.sum() > 200 else np.nan

    edges = [0, 1, 2, 5, 20, 100, np.inf]
    strata = pd.cut(X[STRATA_COL], edges, right=True, labels=False, include_lowest=True)
    rs, ws = [], []
    for s, g in X.groupby(strata):
        r = spearman(g[a], g[b])
        if pd.notna(r):
            rs.append(r); ws.append(len(g))
    r_strata = np.average(rs, weights=ws) if rs else np.nan
    holds = all(pd.notna(v) and abs(v) >= thresh for v in (r_int, r_strata))
    return dict(a=a, b=b, frac_at_bound_a=at_bound(X[a]), frac_at_bound_b=at_bound(X[b]),
                interior_rows=int(interior.sum()), rho_interior=r_int,
                rho_within_msg_strata=r_strata, holds_up=holds)


def label_relevance(X, y):
    rows = []
    for c in X.columns:
        s = X[c]
        ok = s.notna()
        auc = np.nan
        if ok.sum() > 100 and s[ok].nunique() > 1 and y[ok].nunique() > 1:
            auc = roc_auc_score(y[ok], s[ok])
        binned = (pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")
                  if s.nunique() > 10 else s.astype("Int64").astype("float"))
        mi = mutual_info_score(binned.fillna(-1), y)
        binary = set(s.dropna().unique()) <= {0.0, 1.0}
        lift = np.nan
        if binary and (s == 1).sum() >= 30:
            lift = y[s == 1].mean() / max(y[s == 0].mean(), 1e-9)
        rows.append(dict(feature=c, spearman_y=spearman(s, y.astype(float)),
                         roc_auc=auc, auc_dist_from_half=abs(auc - 0.5) if pd.notna(auc) else np.nan,
                         mutual_info=mi, pos_rate_if_1_over_if_0=lift,
                         nonnull_frac=ok.mean(), n_unique=s.nunique()))
    return pd.DataFrame(rows).sort_values("auc_dist_from_half", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--redundancy_thresh", type=float, default=0.9)
    ap.add_argument("--p_uniform_smpp", type=float, default=0.05)
    ap.add_argument("--p_uniform_ss7", type=float, default=0.10)
    ap.add_argument("--p_eval_smpp", type=float, default=1.0)
    ap.add_argument("--p_eval_ss7", type=float, default=0.15)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir)

    for source, pu, pe in (("SS7", a.p_uniform_ss7, a.p_eval_ss7), ("SMPP", a.p_uniform_smpp, a.p_eval_smpp)):
        print(f"\n===== {source} =====")
        df = load_sample(source, data_dir, pu, pe)
        print(f"sample rows {len(df):,}; uniform {df['in_uniform'].sum():,}; "
              f"rule_evaluated {df['rule_evaluated'].sum():,}; FAISS coverage {df[NEAR_DUP_COLS[0]].notna().mean():.1%}")
        X = build_features(df)
        X.to_parquet(out / f"features_{source}.parquet")
        meta = df[["rule_evaluated", "rule_flagged", "in_uniform", "originator"]]
        meta.to_parquet(out / f"meta_{source}.parquet")

        uni = X[df["in_uniform"].to_numpy()]
        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        print(f"constant/dead columns in sample: {const}")
        print(f"flag fire rates (uniform sample):\n{uni[CONTENT_FLAG_COLS].mean().round(4).to_string()}")

        corr = uni.corr(method="spearman")
        corr.to_csv(out / f"spearman_{source}.csv")
        pairs = redundant_pairs(corr, a.redundancy_thresh)
        pairs.to_csv(out / f"pairs_{source}.csv", index=False)
        print(f"\n{len(pairs)} pair(s) with |rho|>={a.redundancy_thresh} (uniform traffic sample, n={len(uni):,})")
        print(pairs.to_string(index=False))
        watch = redundant_pairs(corr, 0.8)
        watch = watch[watch["rho"].abs() < a.redundancy_thresh]
        print(f"\nwatch-list 0.8<=|rho|<{a.redundancy_thresh}:\n{watch.to_string(index=False)}")

        bchecks = [boundedness_check(uni, r.a, r.b, a.redundancy_thresh)
                   for r in pairs.itertuples() if r.a in BOUNDED or r.b in BOUNDED]
        if bchecks:
            bc = pd.DataFrame(bchecks)
            bc.to_csv(out / f"boundedness_{source}.csv", index=False)
            print("\nboundedness check (pairs involving a bounded feature):")
            print(bc.to_string(index=False))

        lab = X[df["rule_evaluated"].to_numpy()]
        y = df.loc[df["rule_evaluated"], "rule_flagged"].astype(float)
        print(f"\nlabelled pool: n={len(lab):,}, positive rate={y.mean():.4f}, classes={sorted(y.unique())}")
        if y.nunique() == 2:
            rel = label_relevance(lab.reset_index(drop=True), y.reset_index(drop=True))
            rel.to_csv(out / f"label_relevance_{source}.csv", index=False)
            print(rel.round(4).to_string(index=False))
        else:
            print("  single-class pool -> label relevance undefined for this source")


if __name__ == "__main__":
    main()
