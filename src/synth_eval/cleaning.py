"""Detect, suggest corrections for, and clean up out-of-vocabulary categorical
values in synthetic datasets (e.g. "admin. primary" → "admin.")."""

import difflib

import numpy as np
import pandas as pd

from ._helpers import _iter_synthetic
from .utils import get_column_types


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_invalid_categorical_values(real_df, synthetic_df, columns=None):
    """
    For each categorical column, return synthetic values not present in the real
    dataset together with their occurrence count.

    Parameters
    ----------
    real_df, synthetic_df : pd.DataFrame
    columns : list[str] | None
        Restrict to these columns. Defaults to all categorical columns of real_df.

    Returns
    -------
    pd.DataFrame with columns: column, value, count
    """
    if columns is None:
        _, cat_cols = get_column_types(real_df)
    else:
        cat_cols = columns

    rows = []
    for col in cat_cols:
        if col not in synthetic_df.columns:
            continue
        real_vals = set(real_df[col].dropna().astype(str).unique())
        syn_series = synthetic_df[col].dropna().astype(str)
        invalid = syn_series[~syn_series.isin(real_vals)]
        for v, c in invalid.value_counts().items():
            rows.append({"column": col, "value": v, "count": int(c)})
    return pd.DataFrame(rows, columns=["column", "value", "count"])


# ── Suggestion ────────────────────────────────────────────────────────────────

def _best_substring_match(invalid_value, real_values):
    """
    Return the longest real value that is a substring of ``invalid_value``
    (case-insensitive). Returns None if no real value is a substring or if the
    longest length is tied between multiple candidates.
    """
    needle = invalid_value.strip().lower()
    hits = [r for r in real_values if r.strip().lower() in needle]
    if not hits:
        return None
    max_len = max(len(r) for r in hits)
    longest = [r for r in hits if len(r) == max_len]
    if len(longest) != 1:
        return None
    return longest[0]


def suggest_corrections(
    real_df,
    synthetic_df,
    columns=None,
    fuzzy_cutoff=0.85,
):
    """
    For every invalid categorical value, suggest a correction drawn from the real
    column vocabulary. Strategies tried in order:

      1. ``contains`` — exactly one real value is a substring of the invalid
         value (longest, unambiguous; matches "admin. primary" → "admin.").
      2. ``fuzzy``    — closest real value via ``difflib.get_close_matches`` with
         similarity ratio ≥ ``fuzzy_cutoff``.
      3. ``None``     — no confident suggestion.

    Returns
    -------
    pd.DataFrame with columns: column, value, count, suggestion, strategy
    """
    invalid = detect_invalid_categorical_values(real_df, synthetic_df, columns)
    if invalid.empty:
        invalid["suggestion"] = pd.Series([], dtype="object")
        invalid["strategy"] = pd.Series([], dtype="object")
        return invalid

    real_vocab = {
        col: sorted(real_df[col].dropna().astype(str).unique(),
                    key=len, reverse=True)
        for col in invalid["column"].unique()
    }

    suggestions, strategies = [], []
    for _, row in invalid.iterrows():
        col, val = row["column"], row["value"]
        real_vals = real_vocab[col]

        match = _best_substring_match(val, real_vals)
        if match is not None:
            suggestions.append(match)
            strategies.append("contains")
            continue

        fuzzy = difflib.get_close_matches(val, real_vals, n=1, cutoff=fuzzy_cutoff)
        if fuzzy:
            suggestions.append(fuzzy[0])
            strategies.append("fuzzy")
            continue

        suggestions.append(None)
        strategies.append(None)

    out = invalid.copy()
    out["suggestion"] = suggestions
    out["strategy"] = strategies
    return out


# ── Application ───────────────────────────────────────────────────────────────

def apply_corrections(synthetic_df, corrections_df):
    """
    Apply suggested corrections (where ``suggestion`` is not None) to a copy
    of ``synthetic_df``.

    Returns
    -------
    (fixed_df, n_corrected, n_unfixed)
        fixed_df    : pd.DataFrame — copy with corrections applied
        n_corrected : int          — total cells modified
        n_unfixed   : int          — total cells with no confident suggestion
    """
    df = synthetic_df.copy()
    n_corrected = 0

    fixable = corrections_df.dropna(subset=["suggestion"])
    for _, row in fixable.iterrows():
        col = row["column"]
        if col not in df.columns:
            continue
        mask = df[col].astype(str) == row["value"]
        n_corrected += int(mask.sum())
        df.loc[mask, col] = row["suggestion"]

    n_unfixed = int(corrections_df.loc[
        corrections_df["suggestion"].isna(), "count"
    ].sum())
    return df, n_corrected, n_unfixed


# ── Whole-pipeline helpers ────────────────────────────────────────────────────

def run_invalid_values_evaluation(train_datasets, methods, seeds, fuzzy_cutoff=0.85):
    """
    Summarise invalid-value counts and fix-ability across all synthetic datasets.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed,
        n_invalid_total, n_fixable_contains, n_fixable_fuzzy, n_unfixable,
        n_invalid_unique
    """
    rows = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        sugg = suggest_corrections(real_df, syn_df, fuzzy_cutoff=fuzzy_cutoff)
        if sugg.empty:
            rows.append({
                "method": method, "llm": llm, "seed": seed,
                "n_invalid_total": 0, "n_invalid_unique": 0,
                "n_fixable_contains": 0, "n_fixable_fuzzy": 0, "n_unfixable": 0,
            })
            continue
        by_strategy = sugg.groupby(sugg["strategy"].fillna("none"))["count"].sum()
        rows.append({
            "method": method, "llm": llm, "seed": seed,
            "n_invalid_total":    int(sugg["count"].sum()),
            "n_invalid_unique":   int(len(sugg)),
            "n_fixable_contains": int(by_strategy.get("contains", 0)),
            "n_fixable_fuzzy":    int(by_strategy.get("fuzzy", 0)),
            "n_unfixable":        int(by_strategy.get("none", 0)),
        })
    return pd.DataFrame(rows)


def fix_invalid_values_in_train_datasets(
    train_datasets,
    methods,
    seeds,
    fuzzy_cutoff=0.85,
):
    """
    Apply suggested corrections in-place to every synthetic DataFrame in
    ``train_datasets``. The ``real`` entry is never modified.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, n_corrected, n_unfixed
    """
    report = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        sugg = suggest_corrections(real_df, syn_df, fuzzy_cutoff=fuzzy_cutoff)
        fixed_df, n_corrected, n_unfixed = apply_corrections(syn_df, sugg)
        if llm == "-":
            train_datasets[method][seed] = fixed_df
        else:
            train_datasets[method][llm][seed] = fixed_df
        report.append({
            "method": method, "llm": llm, "seed": seed,
            "n_corrected": n_corrected,
            "n_unfixed": n_unfixed,
        })
    return pd.DataFrame(report)
