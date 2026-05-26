"""Structural analysis: dataset shape, per-column statistics,
distribution distances and categorical-coverage between real and synthetic data."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from ._helpers import _iter_all, _iter_synthetic
from .utils import get_column_types


# ── Dataset structure summary ────────────────────────────────────────────────

def summarize_structure(df):
    """
    High-level structural summary of a DataFrame.

    Returns
    -------
    dict with keys: n_rows, n_cols, n_numeric, n_categorical,
                    n_missing_total, missing_rate (%), n_duplicate_rows,
                    duplicate_rate (%)
    """
    num_cols, cat_cols = get_column_types(df)
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_missing_total": n_missing,
        "missing_rate": (n_missing / (n_rows * n_cols) * 100) if df.size else 0.0,
        "n_duplicate_rows": n_dupes,
        "duplicate_rate": (n_dupes / n_rows * 100) if n_rows else 0.0,
    }


def run_structure_summary(train_datasets, methods, seeds):
    """
    Structure summary across all datasets (real + synthetic).

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, n_rows, n_cols,
        n_numeric, n_categorical, n_missing_total, missing_rate,
        n_duplicate_rows, duplicate_rate
    """
    results = []
    for method, llm, seed, df in _iter_all(train_datasets, methods, seeds):
        results.append({
            "method": method, "llm": llm, "seed": seed,
            **summarize_structure(df),
        })
    return pd.DataFrame(results)


# ── Per-column descriptive statistics ────────────────────────────────────────

def compute_column_summary(df):
    """
    Per-column descriptive statistics.

    Numerical columns: mean, std, min, q25, q50, q75, max, skew, kurt.
    Categorical columns: mode and mode frequency.
    Both: dtype, n_missing, n_unique.

    Returns
    -------
    pd.DataFrame indexed by column.
    """
    num_cols, _ = get_column_types(df)
    rows = []
    for col in df.columns:
        s = df[col]
        rec = {
            "column": col,
            "dtype": "numeric" if col in num_cols else "categorical",
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if col in num_cols:
            v = s.dropna()
            rec.update({
                "mean": float(v.mean()) if len(v) else np.nan,
                "std":  float(v.std())  if len(v) > 1 else np.nan,
                "min":  float(v.min())  if len(v) else np.nan,
                "q25":  float(v.quantile(0.25)) if len(v) else np.nan,
                "q50":  float(v.quantile(0.50)) if len(v) else np.nan,
                "q75":  float(v.quantile(0.75)) if len(v) else np.nan,
                "max":  float(v.max())  if len(v) else np.nan,
                "skew": float(v.skew()) if len(v) > 2 else np.nan,
                "kurt": float(v.kurt()) if len(v) > 3 else np.nan,
            })
        else:
            v = s.dropna().astype(str)
            mode_val = v.mode()
            if len(mode_val):
                rec["mode"] = mode_val.iloc[0]
                rec["mode_freq"] = float((v == mode_val.iloc[0]).mean())
            else:
                rec["mode"] = np.nan
                rec["mode_freq"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("column")


def compare_column_summaries(real_df, synthetic_df):
    """
    Side-by-side per-column statistics for real vs synthetic.

    Returns
    -------
    pd.DataFrame with a two-level column index (stat, source) where
    source ∈ {'real', 'synthetic'}.
    """
    real_sum = compute_column_summary(real_df)
    syn_sum = compute_column_summary(synthetic_df)
    common = [c for c in real_sum.index if c in syn_sum.index]
    real_sum = real_sum.loc[common]
    syn_sum = syn_sum.loc[common]
    out = pd.concat({"real": real_sum, "synthetic": syn_sum}, axis=1)
    out = out.swaplevel(axis=1).sort_index(axis=1)
    return out


# ── Distribution distances ───────────────────────────────────────────────────

def _jensen_shannon(p, q):
    """Jensen-Shannon divergence between two discrete distributions (log base 2)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def evaluate_distribution_distances(real_df, synthetic_df):
    """
    Per-column distribution distances between real and synthetic data.

    Numeric columns:
      - ks   : Kolmogorov-Smirnov statistic in [0, 1]
      - wass : Wasserstein distance, normalized by std(real) when std > 0
    Categorical columns:
      - tvd  : Total Variation Distance in [0, 1]
      - jsd  : Jensen-Shannon divergence (log2) in [0, 1]

    Returns
    -------
    dict with:
        per_column : pd.DataFrame[column, type, ks, wass, tvd, jsd]
        ks_mean, wass_mean, tvd_mean, jsd_mean : aggregated means (NaN-safe)
    """
    common = [c for c in real_df.columns if c in synthetic_df.columns]
    num_cols, _ = get_column_types(real_df[common])
    num_set = set(num_cols)

    rows = []
    for col in common:
        rec = {"column": col, "ks": np.nan, "wass": np.nan,
               "tvd": np.nan, "jsd": np.nan}
        if col in num_set:
            rec["type"] = "numeric"
            r = pd.to_numeric(real_df[col], errors="coerce").dropna().values
            s = pd.to_numeric(synthetic_df[col], errors="coerce").dropna().values
            if len(r) and len(s):
                rec["ks"] = float(ks_2samp(r, s).statistic)
                std = float(np.std(r))
                w = float(wasserstein_distance(r, s))
                rec["wass"] = w / std if std > 0 else w
        else:
            rec["type"] = "categorical"
            r_freq = real_df[col].astype(str).value_counts(normalize=True)
            s_freq = synthetic_df[col].astype(str).value_counts(normalize=True)
            cats = r_freq.index.union(s_freq.index)
            r_p = r_freq.reindex(cats, fill_value=0.0).values
            s_p = s_freq.reindex(cats, fill_value=0.0).values
            rec["tvd"] = float(0.5 * np.abs(r_p - s_p).sum())
            rec["jsd"] = float(_jensen_shannon(r_p, s_p))
        rows.append(rec)

    per_col = pd.DataFrame(rows)
    return {
        "per_column": per_col,
        "ks_mean":   float(per_col["ks"].mean(skipna=True)),
        "wass_mean": float(per_col["wass"].mean(skipna=True)),
        "tvd_mean":  float(per_col["tvd"].mean(skipna=True)),
        "jsd_mean":  float(per_col["jsd"].mean(skipna=True)),
    }


def run_distribution_distance_evaluation(train_datasets, methods, seeds):
    """
    Aggregated distribution distances per (method, llm, seed).

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed,
        ks_mean, wass_mean, tvd_mean, jsd_mean
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        d = evaluate_distribution_distances(real_df, syn_df)
        results.append({
            "method": method, "llm": llm, "seed": seed,
            "ks_mean":   d["ks_mean"],
            "wass_mean": d["wass_mean"],
            "tvd_mean":  d["tvd_mean"],
            "jsd_mean":  d["jsd_mean"],
        })
    return pd.DataFrame(results)


# ── Categorical coverage ─────────────────────────────────────────────────────

def evaluate_categorical_coverage(real_df, synthetic_df):
    """
    For each categorical column, measure how synthetic categories overlap real ones.

    Per column:
      - n_real_cats, n_syn_cats
      - n_new_in_syn      : categories present in synthetic but not in real
      - n_missing_in_syn  : real categories absent from synthetic
      - jaccard           : |R ∩ S| / |R ∪ S|
      - coverage          : |R ∩ S| / |R|  (fraction of real categories captured)

    Returns
    -------
    dict with:
        per_column : pd.DataFrame
        jaccard_mean, coverage_mean, new_cats_mean, missing_cats_mean
    """
    _, cat_cols = get_column_types(real_df)
    rows = []
    for col in cat_cols:
        if col not in synthetic_df.columns:
            continue
        r_cats = set(real_df[col].dropna().astype(str).unique())
        s_cats = set(synthetic_df[col].dropna().astype(str).unique())
        union = r_cats | s_cats
        inter = r_cats & s_cats
        rows.append({
            "column": col,
            "n_real_cats": len(r_cats),
            "n_syn_cats":  len(s_cats),
            "n_new_in_syn":     len(s_cats - r_cats),
            "n_missing_in_syn": len(r_cats - s_cats),
            "jaccard":  (len(inter) / len(union)) if union else 1.0,
            "coverage": (len(inter) / len(r_cats)) if r_cats else 1.0,
        })
    per_col = pd.DataFrame(rows)
    if per_col.empty:
        return {
            "per_column": per_col,
            "jaccard_mean":      np.nan,
            "coverage_mean":     np.nan,
            "new_cats_mean":     np.nan,
            "missing_cats_mean": np.nan,
        }
    return {
        "per_column": per_col,
        "jaccard_mean":      float(per_col["jaccard"].mean()),
        "coverage_mean":     float(per_col["coverage"].mean()),
        "new_cats_mean":     float(per_col["n_new_in_syn"].mean()),
        "missing_cats_mean": float(per_col["n_missing_in_syn"].mean()),
    }


def run_categorical_coverage_evaluation(train_datasets, methods, seeds):
    """
    Categorical-coverage summary across all synthetic datasets.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed,
        jaccard_mean, coverage_mean, new_cats_mean, missing_cats_mean
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        d = evaluate_categorical_coverage(real_df, syn_df)
        results.append({
            "method": method, "llm": llm, "seed": seed,
            "jaccard_mean":      d["jaccard_mean"],
            "coverage_mean":     d["coverage_mean"],
            "new_cats_mean":     d["new_cats_mean"],
            "missing_cats_mean": d["missing_cats_mean"],
        })
    return pd.DataFrame(results)
