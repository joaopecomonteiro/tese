"""Similarity metrics: discriminator, correlation, row-level matching."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ._helpers import _iter_synthetic
from .utils import get_column_types, preprocess_for_ml


# ── Discriminator ─────────────────────────────────────────────────────────────

def evaluate_discriminator(real_df, synthetic_df):
    """
    Train an XGBoost classifier to distinguish real (0) from synthetic (1).

    Returns
    -------
    float — AUC score (0.5 = indistinguishable, 1.0 = perfectly separable)
    """
    real_proc, encoders = preprocess_for_ml(real_df, fit=True)
    syn_proc, _ = preprocess_for_ml(synthetic_df, label_encoders=encoders, fit=False)

    real_proc["is_synthetic"] = 0
    syn_proc["is_synthetic"] = 1

    combined = pd.concat([real_proc, syn_proc], ignore_index=True)
    X = combined.drop(columns=["is_synthetic"])
    y = combined["is_synthetic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


def run_discriminator_evaluation(train_datasets, methods, seeds, test_datasets=None):
    """
    Run discriminator evaluation for all synthetic methods and seeds.

    When ``test_datasets`` is given, also computes the i.i.d. baseline
    (real_train vs real_test AUC) per seed under ``method='real_test'``. That
    baseline measures how distinguishable two real samples of the same
    distribution are; a synthetic AUC matching this baseline is i.i.d.-like,
    while AUC well above it indicates the synthetic deviates from the real
    distribution.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, disc_auc
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        auc = evaluate_discriminator(real_df, syn_df)
        results.append({"method": method, "llm": llm, "seed": seed, "disc_auc": auc})

    if test_datasets is not None:
        for seed in seeds:
            real_df = train_datasets["real"][seed]
            test_df = test_datasets[seed]
            auc = evaluate_discriminator(real_df, test_df)
            results.append({
                "method": "real_test", "llm": "-", "seed": seed, "disc_auc": auc,
            })

    return pd.DataFrame(results)


# ── Correlation ───────────────────────────────────────────────────────────────

def evaluate_correlation_difference(real_df, synthetic_df, method="pearson"):
    """
    Compute MAE between the correlation matrices of real and synthetic data
    (numerical columns only, upper triangle).

    Returns
    -------
    float — MAE
    """
    real_clean = real_df.dropna()
    syn_clean = synthetic_df.dropna()

    num_cols, _ = get_column_types(real_clean)

    real_corr = real_clean[num_cols].corr(method=method)
    syn_corr = syn_clean[num_cols].corr(method=method)

    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    real_vals = real_corr.where(mask).stack()
    syn_vals = syn_corr.where(mask).stack()

    return float(np.abs(real_vals - syn_vals).mean())


def run_correlation_evaluation(train_datasets, methods, seeds, method="pearson",
                                test_datasets=None):
    """
    Run correlation evaluation for all synthetic methods and seeds.

    When ``test_datasets`` is given, also computes the i.i.d. baseline
    (MAE between the correlation matrices of real_train and real_test) per seed
    under ``method='real_test'``. Two real samples of the same distribution
    differ slightly in their correlation matrices due to sampling noise; this
    baseline tells you the "irreducible" MAE under i.i.d. sampling. Synthetic
    MAE near this baseline indicates faithful preservation of correlation
    structure; MAE well above it indicates the generator distorts pairwise
    relationships.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, corr_mae
    """
    results = []
    for m, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        mae = evaluate_correlation_difference(real_df, syn_df, method=method)
        results.append({"method": m, "llm": llm, "seed": seed, "corr_mae": mae})

    if test_datasets is not None:
        for seed in seeds:
            real_df = train_datasets["real"][seed]
            test_df = test_datasets[seed]
            mae = evaluate_correlation_difference(real_df, test_df, method=method)
            results.append({
                "method": "real_test", "llm": "-", "seed": seed, "corr_mae": mae,
            })

    return pd.DataFrame(results)


# ── Exact row matching ────────────────────────────────────────────────────────

def calculate_percentage_equal_rows(df_real, df_synthetic):
    """
    Percentage of rows in df_real that have an exact match in df_synthetic.

    Returns
    -------
    float — percentage (0–100)
    """
    df_syn_aligned = df_synthetic[df_real.columns]
    syn_set = set(df_syn_aligned.itertuples(index=False, name=None))
    matches = sum(
        1 for row in df_real.itertuples(index=False, name=None) if row in syn_set
    )
    return (matches / len(df_real)) * 100


def run_percent_equal_rows_evaluation(train_datasets, methods, seeds):
    """
    Run exact-row-match evaluation for all synthetic methods and seeds.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, pct
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        pct = calculate_percentage_equal_rows(real_df, syn_df)
        results.append({"method": method, "llm": llm, "seed": seed, "pct": pct})
    return pd.DataFrame(results)


# ── Fuzzy / similar row matching ──────────────────────────────────────────────

def calculate_percentage_similar_rows(
    df_real,
    df_synthetic,
    thresholds=None,
    numeric_tol=0.0,
    chunk_size=256,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
):
    """
    For each synthetic row, find the best-matching real row and report
    the percentage of synthetic rows whose best match exceeds each threshold.

    This direction measures near-duplicate risk: how many synthetic records
    are close copies of real training records.

    Constant and near-constant columns are excluded to avoid trivial inflation.

    Parameters
    ----------
    df_real, df_synthetic : pd.DataFrame
    thresholds            : list[float]  - similarity thresholds (default 1.0...0.5)
    numeric_tol           : float        - tolerance for numerical column matching
    chunk_size            : int          - rows per chunk (tune for memory/speed)
    dominance_threshold   : float        - max fraction for a single value before
                                           the column is considered near-constant
    column_types          : 'all' | 'numeric' | 'categorical'
    continuous_cols       : list[str] | None - explicit list of continuous columns;
                                           when set, only these columns are treated as
                                           numeric (discrete numeric columns are
                                           excluded from the numeric set)

    Returns
    -------
    dict - {threshold: percentage, ...}
    """
    if thresholds is None:
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    df_syn = df_synthetic[df_real.columns].reset_index(drop=True)
    df_real = df_real.reset_index(drop=True)

    num_cols_all = df_real.select_dtypes(include=[np.number]).columns.tolist()

    constant_cols = [
        c for c in num_cols_all
        if (df_real[c].max() - df_real[c].min()) < 1e-9
    ]
    low_variance_cols = [
        c for c in df_real.columns
        if df_real[c].value_counts(normalize=True).iloc[0] >= dominance_threshold
    ]
    cols_to_exclude = set(constant_cols) | set(low_variance_cols)
    if cols_to_exclude:
        print(f"[WARNING] Ignoring low-variance/constant column(s): {cols_to_exclude}")

    num_cols = [c for c in num_cols_all if c not in cols_to_exclude]
    if continuous_cols is not None:
        num_cols = [c for c in num_cols if c in continuous_cols]
    cat_cols = [
        c for c in df_real.columns
        if c not in num_cols_all and c not in cols_to_exclude
    ]

    if column_types == "numeric":
        cat_cols = []
    elif column_types == "categorical":
        num_cols = []
    elif column_types != "all":
        raise ValueError(
            f"column_types must be 'all', 'numeric' or 'categorical', got '{column_types}'"
        )

    n_cols = len(cat_cols) + len(num_cols)
    n_cat = len(cat_cols)

    # Encode categorical columns
    cat_real = np.empty((len(df_real), len(cat_cols)), dtype=np.float32)
    cat_syn = np.empty((len(df_syn), len(cat_cols)), dtype=np.float32)
    for j, col in enumerate(cat_cols):
        combined = pd.Categorical(
            pd.concat([df_real[col], df_syn[col]], ignore_index=True)
        )
        codes = combined.codes.astype(np.float32)
        cat_real[:, j] = codes[: len(df_real)]
        cat_syn[:, j] = codes[len(df_real):]

    # Normalize numerical columns to [0, 1] using real data range
    num_real = np.empty((len(df_real), len(num_cols)), dtype=np.float32)
    num_syn = np.empty((len(df_syn), len(num_cols)), dtype=np.float32)
    for j, col in enumerate(num_cols):
        lo, hi = df_real[col].min(), df_real[col].max()
        col_range = hi - lo
        num_real[:, j] = ((df_real[col] - lo) / col_range).values.astype(np.float32)
        num_syn[:, j] = ((df_syn[col] - lo) / col_range).values.astype(np.float32)

    # Stack into single matrices: [cat | num]
    if n_cat and len(num_cols):
        real_arr = np.hstack([cat_real, num_real])
        syn_arr = np.hstack([cat_syn, num_syn])
    elif n_cat:
        real_arr, syn_arr = cat_real, cat_syn
    else:
        real_arr, syn_arr = num_real, num_syn

    # Exact-match pre-fill: check each synthetic row against the real set
    best_fracs = np.zeros(len(df_syn), dtype=np.float32)
    if 1.0 in thresholds:
        real_set = set(map(tuple, df_real.itertuples(index=False, name=None)))
        exact_flags = np.array(
            [row in real_set for row in df_syn.itertuples(index=False, name=None)]
        )
        best_fracs[exact_flags] = 1.0

    # Chunked vectorized best-match computation: iterate over synthetic rows
    indices = np.where(best_fracs < 1.0)[0]
    for start in range(0, len(indices), chunk_size):
        idx = indices[start: start + chunk_size]
        s = syn_arr[idx]
        diff = np.abs(s[:, np.newaxis, :] - real_arr[np.newaxis, :, :])

        cat_match = diff[:, :, :n_cat] == 0

        num_diff = diff[:, :, n_cat:]
        s_nan = np.isnan(s[:, np.newaxis, n_cat:])
        r_nan = np.isnan(real_arr[np.newaxis, :, n_cat:])
        num_match = (num_diff <= numeric_tol) | (s_nan & r_nan)

        match_frac = np.concatenate([cat_match, num_match], axis=2).sum(axis=2) / n_cols
        best_fracs[idx] = match_frac.max(axis=1)

    return {
        t: float((best_fracs >= t).sum() / len(df_syn) * 100) for t in thresholds
    }


def run_percentage_similar_rows_evaluation(
    train_datasets,
    methods,
    seeds,
    test_datasets=None,
    thresholds=None,
    numeric_tol=0.0,
    chunk_size=256,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
):
    """
    Run fuzzy row-similarity evaluation for all synthetic methods and seeds.

    When test_datasets is provided, also computes similarity of the real test
    set against the real training set (method='real_test', llm='-'), giving a
    natural baseline for how similar held-out real data is to training data.

    Parameters
    ----------
    test_datasets : dict | None — {seed: df} — held-out real data

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, pct_100, pct_90, …
    """
    if thresholds is None:
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    threshold_cols = {t: f"pct_{int(t * 100)}" for t in thresholds}
    results = []

    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        print(f"  Seed {seed} | {method} / {llm}")
        real_df = train_datasets["real"][seed]
        pct_dict = calculate_percentage_similar_rows(
            real_df,
            syn_df,
            thresholds=thresholds,
            numeric_tol=numeric_tol,
            chunk_size=chunk_size,
            dominance_threshold=dominance_threshold,
            column_types=column_types,
            continuous_cols=continuous_cols,
        )
        results.append({
            "method": method,
            "llm": llm,
            "seed": seed,
            **{threshold_cols[t]: pct_dict[t] for t in thresholds},
        })

    if test_datasets is not None:
        for seed in seeds:
            print(f"  Seed {seed} | real_test / -")
            real_df = train_datasets["real"][seed]
            test_df = test_datasets[seed]
            pct_dict = calculate_percentage_similar_rows(
                real_df,
                test_df,
                thresholds=thresholds,
                numeric_tol=numeric_tol,
                chunk_size=chunk_size,
                dominance_threshold=dominance_threshold,
                column_types=column_types,
                continuous_cols=continuous_cols,
            )
            results.append({
                "method": "real_test",
                "llm": "-",
                "seed": seed,
                **{threshold_cols[t]: pct_dict[t] for t in thresholds},
            })

    return pd.DataFrame(results)


# ── Synthetic-vs-train and synthetic-vs-test threshold curves ────────────────

def run_percentage_similar_rows_train_vs_test(
    train_datasets,
    test_datasets,
    methods,
    seeds,
    thresholds=None,
    numeric_tol=0.0,
    chunk_size=256,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
    include_iid_baseline=True,
):
    """
    Compute the percentage-similar-rows curve for each synthetic dataset twice:
    once against real_train and once against real_test.

    When ``include_iid_baseline=True`` (default), additionally computes
    real_test → real_train per seed and emits it as rows with
    ``method='real_test'``, ``reference='train'``. This is the i.i.d. baseline:
    what fraction of two real samples (train vs held-out test) cross each
    threshold. Synthetic methods exceeding this curve are closer to train than
    a fresh real sample of the same distribution would be — the cleanest
    memorisation definition when only LLMs are under analysis.

    Returns
    -------
    pd.DataFrame (long format) with columns:
        method, llm, seed, reference ('train'|'test'),
        and pct_100, pct_95, ... pct_50 (one per threshold).
    """
    if thresholds is None:
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    threshold_cols = {t: f"pct_{int(round(t * 100))}" for t in thresholds}

    rows = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        for ref_label, ref_df in (
            ("train", train_datasets["real"][seed]),
            ("test", test_datasets[seed]),
        ):
            print(f"  Seed {seed} | {method} / {llm} | reference={ref_label}")
            pct = calculate_percentage_similar_rows(
                ref_df, syn_df,
                thresholds=thresholds,
                numeric_tol=numeric_tol,
                chunk_size=chunk_size,
                dominance_threshold=dominance_threshold,
                column_types=column_types,
                continuous_cols=continuous_cols,
            )
            rows.append({
                "method": method, "llm": llm, "seed": seed,
                "reference": ref_label,
                **{threshold_cols[t]: pct[t] for t in thresholds},
            })

    if include_iid_baseline:
        for seed in seeds:
            print(f"  Seed {seed} | real_test → real_train (i.i.d. baseline)")
            pct = calculate_percentage_similar_rows(
                train_datasets["real"][seed], test_datasets[seed],
                thresholds=thresholds,
                numeric_tol=numeric_tol,
                chunk_size=chunk_size,
                dominance_threshold=dominance_threshold,
                column_types=column_types,
                continuous_cols=continuous_cols,
            )
            rows.append({
                "method": "real_test", "llm": "-", "seed": seed,
                "reference": "train",
                **{threshold_cols[t]: pct[t] for t in thresholds},
            })

    return pd.DataFrame(rows)


# ── Per-column decomposition of the match metric ──────────────────────────────

def _baseline_match_rates(df_real, df_syn, cat_cols, num_cols, numeric_tol):
    """
    Expected per-column match rate under independent sampling from real and
    synthetic marginals.

    Categorical: sum_v p_real(v) * p_syn(v).
    Numeric:     empirical rate from 5000 random pairs after [0,1] normalisation
                 with the real range (matches the metric's encoding).
    """
    rates = []
    for col in cat_cols:
        p_r = df_real[col].astype(str).value_counts(normalize=True)
        p_s = df_syn[col].astype(str).value_counts(normalize=True)
        idx = p_r.index.union(p_s.index)
        rates.append(float(
            (p_r.reindex(idx, fill_value=0) * p_s.reindex(idx, fill_value=0)).sum()
        ))
    rng = np.random.default_rng(0)
    n_pairs = 5000
    for col in num_cols:
        r = df_real[col].dropna().values
        s = df_syn[col].dropna().values
        if len(r) == 0 or len(s) == 0:
            rates.append(0.0)
            continue
        lo, hi = float(r.min()), float(r.max())
        rng_span = hi - lo if hi > lo else 1.0
        r_n = (r - lo) / rng_span
        s_n = (s - lo) / rng_span
        ri = rng.integers(0, len(r_n), n_pairs)
        si = rng.integers(0, len(s_n), n_pairs)
        rates.append(float((np.abs(r_n[ri] - s_n[si]) <= numeric_tol).mean()))
    return rates


def evaluate_match_column_contribution(
    df_real,
    df_synthetic,
    threshold=0.9,
    chunk_size=256,
    numeric_tol=0.0,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
):
    """
    Per-column decomposition of the row-similarity metric.

    For each synthetic row, find its single best-matching real row (same logic
    as ``calculate_percentage_similar_rows``); for rows whose best match-fraction
    is ≥ ``threshold``, record which columns matched in that pairing. Aggregated
    across those rows, the result tells which columns carry the high-similarity
    tier.

    A ``baseline_rate`` (chance match rate under independent marginal sampling)
    and a ``lift`` (= hit_rate / baseline_rate) are also returned. Low-entropy
    columns dominated by a mode value will have ``lift ≈ 1`` even when their
    ``hit_rate`` is high; high-cardinality columns with significant lift indicate
    genuine row-level memorisation.

    Returns
    -------
    dict
        n_matched_rows : int
        per_column     : pd.DataFrame[column, dtype, hit_rate, baseline_rate, lift]
        columns_used   : list[str]
    """
    df_syn = df_synthetic[df_real.columns].reset_index(drop=True)
    df_real = df_real.reset_index(drop=True)

    num_cols_all = df_real.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [
        c for c in num_cols_all
        if (df_real[c].max() - df_real[c].min()) < 1e-9
    ]
    low_variance_cols = [
        c for c in df_real.columns
        if df_real[c].value_counts(normalize=True).iloc[0] >= dominance_threshold
    ]
    cols_to_exclude = set(constant_cols) | set(low_variance_cols)
    if cols_to_exclude:
        print(f"[WARNING] Ignoring low-variance/constant column(s): {cols_to_exclude}")

    num_cols = [c for c in num_cols_all if c not in cols_to_exclude]
    if continuous_cols is not None:
        num_cols = [c for c in num_cols if c in continuous_cols]
    cat_cols = [
        c for c in df_real.columns
        if c not in num_cols_all and c not in cols_to_exclude
    ]

    if column_types == "numeric":
        cat_cols = []
    elif column_types == "categorical":
        num_cols = []
    elif column_types != "all":
        raise ValueError(
            f"column_types must be 'all', 'numeric' or 'categorical', got '{column_types}'"
        )

    cols_used = cat_cols + num_cols
    n_cols = len(cols_used)
    n_cat = len(cat_cols)
    if n_cols == 0:
        raise ValueError("No columns left to decompose after exclusions.")

    cat_real = np.empty((len(df_real), n_cat), dtype=np.float32)
    cat_syn = np.empty((len(df_syn), n_cat), dtype=np.float32)
    for j, col in enumerate(cat_cols):
        combined = pd.Categorical(
            pd.concat([df_real[col], df_syn[col]], ignore_index=True)
        )
        codes = combined.codes.astype(np.float32)
        cat_real[:, j] = codes[: len(df_real)]
        cat_syn[:, j] = codes[len(df_real):]

    num_real = np.empty((len(df_real), len(num_cols)), dtype=np.float32)
    num_syn = np.empty((len(df_syn), len(num_cols)), dtype=np.float32)
    for j, col in enumerate(num_cols):
        lo, hi = df_real[col].min(), df_real[col].max()
        col_range = hi - lo if hi > lo else 1.0
        num_real[:, j] = ((df_real[col] - lo) / col_range).values.astype(np.float32)
        num_syn[:, j] = ((df_syn[col] - lo) / col_range).values.astype(np.float32)

    if n_cat and len(num_cols):
        real_arr = np.hstack([cat_real, num_real])
        syn_arr = np.hstack([cat_syn, num_syn])
    elif n_cat:
        real_arr, syn_arr = cat_real, cat_syn
    else:
        real_arr, syn_arr = num_real, num_syn

    n_syn = len(df_syn)
    best_fracs = np.zeros(n_syn, dtype=np.float32)
    best_match_per_col = np.zeros((n_syn, n_cols), dtype=bool)

    for start in range(0, n_syn, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_syn))
        s = syn_arr[idx]
        diff = np.abs(s[:, np.newaxis, :] - real_arr[np.newaxis, :, :])
        cat_match = diff[:, :, :n_cat] == 0
        num_diff = diff[:, :, n_cat:]
        s_nan = np.isnan(s[:, np.newaxis, n_cat:])
        r_nan = np.isnan(real_arr[np.newaxis, :, n_cat:])
        num_match = (num_diff <= numeric_tol) | (s_nan & r_nan)
        all_match = np.concatenate([cat_match, num_match], axis=2)
        match_frac = all_match.sum(axis=2) / n_cols
        best_idx = match_frac.argmax(axis=1)
        rows = np.arange(len(idx))
        best_fracs[idx] = match_frac[rows, best_idx]
        best_match_per_col[idx] = all_match[rows, best_idx]

    matched_mask = best_fracs >= threshold
    n_matched = int(matched_mask.sum())

    baseline = _baseline_match_rates(df_real, df_syn, cat_cols, num_cols, numeric_tol)
    dtypes = ["categorical"] * n_cat + ["numeric"] * len(num_cols)

    if n_matched == 0:
        per_col = pd.DataFrame({
            "column": cols_used,
            "dtype": dtypes,
            "hit_rate": [0.0] * n_cols,
            "baseline_rate": baseline,
            "lift": [np.nan] * n_cols,
        })
        return {"n_matched_rows": 0, "per_column": per_col, "columns_used": cols_used}

    hit_rates = best_match_per_col[matched_mask].mean(axis=0)
    lifts = []
    for hr, bl in zip(hit_rates, baseline):
        if bl > 0:
            lifts.append(float(hr / bl))
        elif hr > 0:
            lifts.append(float("inf"))
        else:
            lifts.append(float("nan"))

    per_col = pd.DataFrame({
        "column": cols_used,
        "dtype": dtypes,
        "hit_rate": hit_rates.astype(float),
        "baseline_rate": baseline,
        "lift": lifts,
    })
    return {"n_matched_rows": n_matched, "per_column": per_col, "columns_used": cols_used}


def run_match_column_contribution_evaluation(
    train_datasets,
    methods,
    seeds,
    threshold=0.9,
    test_datasets=None,
    chunk_size=256,
    numeric_tol=0.0,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
):
    """
    Per-column match decomposition for every synthetic dataset.

    When ``test_datasets`` is given, also computes the decomposition for the real
    test set against the real train (method='real_test') as a control: that tells
    you how much each column matches "by chance" between two i.i.d. samples of
    the real distribution. Synthetic methods whose per-column hit-rates exceed
    that control (especially on high-cardinality columns) are over-fitting to
    train.

    Returns
    -------
    pd.DataFrame (long format) with columns:
        method, llm, seed, column, dtype, hit_rate, baseline_rate, lift,
        n_matched_rows
    """
    rows = []

    def _append(method, llm, seed, d):
        per = d["per_column"].copy()
        per["method"] = method
        per["llm"] = llm
        per["seed"] = seed
        per["n_matched_rows"] = d["n_matched_rows"]
        rows.append(per)

    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        print(f"  Seed {seed} | {method} / {llm}")
        d = evaluate_match_column_contribution(
            train_datasets["real"][seed], syn_df,
            threshold=threshold, chunk_size=chunk_size, numeric_tol=numeric_tol,
            dominance_threshold=dominance_threshold,
            column_types=column_types, continuous_cols=continuous_cols,
        )
        _append(method, llm, seed, d)

    if test_datasets is not None:
        for seed in seeds:
            print(f"  Seed {seed} | real_test / -")
            d = evaluate_match_column_contribution(
                train_datasets["real"][seed], test_datasets[seed],
                threshold=threshold, chunk_size=chunk_size, numeric_tol=numeric_tol,
                dominance_threshold=dominance_threshold,
                column_types=column_types, continuous_cols=continuous_cols,
            )
            _append("real_test", "-", seed, d)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[[
        "method", "llm", "seed", "column", "dtype",
        "hit_rate", "baseline_rate", "lift", "n_matched_rows",
    ]]


def check_match_sample_size(decomp_df, min_n=50):
    """
    Identify (method, llm, seed, reference) combinations whose ``n_matched_rows``
    is too small for the per-column hit_rate / lift analysis to be statistically
    reliable.

    With ``n_matched < min_n`` the hit_rate is dominated by binomial noise (e.g.
    n=10 → SE ≈ 0.16 at p=0.5), and at n=0 / n=1 it is degenerate. Datasets where
    the generator severely under-fits (e.g. cardiotocography with LLMs) collapse
    into this regime at typical thresholds, and lift values become uninterpretable.

    Parameters
    ----------
    decomp_df : pd.DataFrame
        Output of ``run_match_column_contribution_evaluation`` or
        ``run_match_column_contribution_train_vs_test``.
    min_n : int
        Minimum acceptable matched-row count per (method, llm, seed, reference).

    Returns
    -------
    pd.DataFrame[method, llm, seed, reference, n_matched_rows, ok]
        ``ok`` is True iff ``n_matched_rows >= min_n``.
    """
    keys = ['method', 'llm', 'seed', 'reference']
    keys = [k for k in keys if k in decomp_df.columns]
    sizes = decomp_df.groupby(keys)['n_matched_rows'].first().reset_index()
    sizes['ok'] = sizes['n_matched_rows'] >= min_n
    return sizes


def suggest_threshold_for_sample_size(curve_df, min_n=50, n_syn_hint=None):
    """
    Given a long-format curve from ``run_percentage_similar_rows_train_vs_test``,
    suggest the largest threshold *t* at which both reference='train' and
    reference='test' attain ``n_matched_rows >= min_n`` for every synthetic
    method/seed.

    Uses ``pct_X`` columns to estimate matched-row count via
    ``pct/100 * n_syn``, where ``n_syn`` is taken from ``n_syn_hint`` or
    estimated heuristically from any column.

    Returns
    -------
    float | None — the suggested threshold (as a [0,1] fraction). Returns None
    if no threshold in the curve achieves the requirement.
    """
    pct_cols = [c for c in curve_df.columns if c.startswith('pct_')]
    if not pct_cols:
        return None

    syn = curve_df[curve_df['method'] != 'real_test']
    if syn.empty:
        return None

    if n_syn_hint is None:
        # Conservative estimate: use 100 / 100 == 1.0 fraction = full set.
        # Without n_syn we can only use percentages directly with a hint.
        n_syn_hint = 1000  # a reasonable default for typical tabular benchmarks

    thresholds = sorted({int(c.split('_')[1]) / 100 for c in pct_cols}, reverse=True)
    for t in thresholds:
        col = f'pct_{int(round(t * 100))}'
        # Required pct so that pct/100 * n_syn >= min_n
        pct_required = (min_n / n_syn_hint) * 100
        if (syn[col] >= pct_required).all():
            return t
    return None


def run_match_column_contribution_train_vs_test(
    train_datasets,
    test_datasets,
    methods,
    seeds,
    threshold=0.9,
    chunk_size=256,
    numeric_tol=0.0,
    dominance_threshold=0.95,
    column_types="all",
    continuous_cols=None,
):
    """
    Run the per-column decomposition for every synthetic dataset twice:
    once using real_train as the reference and once using real_test. Used to
    disambiguate row-level memorisation (lift_train ≫ lift_test) from marginal
    coverage (lift_train ≈ lift_test).

    Returns
    -------
    pd.DataFrame (long format) with columns:
        method, llm, seed, reference ('train'|'test'), column, dtype,
        hit_rate, baseline_rate, lift, n_matched_rows
    """
    rows = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        for ref_label, ref_df in (
            ("train", train_datasets["real"][seed]),
            ("test", test_datasets[seed]),
        ):
            print(f"  Seed {seed} | {method} / {llm} | reference={ref_label}")
            d = evaluate_match_column_contribution(
                ref_df, syn_df,
                threshold=threshold, chunk_size=chunk_size, numeric_tol=numeric_tol,
                dominance_threshold=dominance_threshold,
                column_types=column_types, continuous_cols=continuous_cols,
            )
            per = d["per_column"].copy()
            per["method"] = method
            per["llm"] = llm
            per["seed"] = seed
            per["reference"] = ref_label
            per["n_matched_rows"] = d["n_matched_rows"]
            rows.append(per)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[[
        "method", "llm", "seed", "reference", "column", "dtype",
        "hit_rate", "baseline_rate", "lift", "n_matched_rows",
    ]]


# ── Value-set overlap (set-level memorisation diagnostic) ─────────────────────

def evaluate_value_overlap(real_df, synthetic_df, columns=None):
    """
    For each column, fraction of synthetic-unique values that also appear in
    ``real_df``. NaN is dropped before the comparison.

    Returns
    -------
    pd.DataFrame with columns:
        column, n_syn_unique, n_real_unique, n_intersection, overlap (= n_inter / n_syn)
    """
    if columns is None:
        columns = [c for c in real_df.columns if c in synthetic_df.columns]
    rows = []
    for col in columns:
        syn_set = set(synthetic_df[col].dropna().unique())
        real_set = set(real_df[col].dropna().unique())
        inter = syn_set & real_set
        rows.append({
            "column": col,
            "n_syn_unique": len(syn_set),
            "n_real_unique": len(real_set),
            "n_intersection": len(inter),
            "overlap": (len(inter) / len(syn_set)) if syn_set else float("nan"),
        })
    return pd.DataFrame(rows)


def run_value_overlap_evaluation(
    train_datasets,
    methods,
    seeds,
    test_datasets=None,
    columns=None,
):
    """
    Per-(method, llm, seed, column) value-overlap of synthetic with real_train.
    When ``test_datasets`` is given, also computes overlap with real_test under
    the same long format (reference column distinguishes the two).

    Returns
    -------
    pd.DataFrame with columns:
        method, llm, seed, reference ('train'|'test'), column,
        n_syn_unique, n_real_unique, n_intersection, overlap
    """
    rows = []
    refs = [("train", lambda s: train_datasets["real"][s])]
    if test_datasets is not None:
        refs.append(("test", lambda s: test_datasets[s]))

    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        for ref_label, ref_fn in refs:
            ref_df = ref_fn(seed)
            df = evaluate_value_overlap(ref_df, syn_df, columns=columns)
            df["method"] = method
            df["llm"] = llm
            df["seed"] = seed
            df["reference"] = ref_label
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[[
        "method", "llm", "seed", "reference", "column",
        "n_syn_unique", "n_real_unique", "n_intersection", "overlap",
    ]]
