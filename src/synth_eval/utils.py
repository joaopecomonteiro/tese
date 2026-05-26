"""Utility functions: column types, preprocessing, validity."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ._helpers import _iter_all


def get_column_types(df):
    """Return (numerical_cols, categorical_cols) for a DataFrame."""
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical, categorical


def preprocess_for_ml(df, label_encoders=None, fit=False):
    """
    Encode categorical columns with LabelEncoder for ML models.

    Parameters
    ----------
    df : pd.DataFrame
    label_encoders : dict, optional
        Pre-fitted encoders to apply (required when fit=False).
    fit : bool
        If True, fit new encoders on df and return them.

    Returns
    -------
    (df_encoded, label_encoders)
    """
    df = df.copy().dropna()
    _, cat_cols = get_column_types(df)

    if fit:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        return df, label_encoders
    else:
        for col in cat_cols:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        return df, label_encoders


def count_valid_rows(df):
    """
    Count rows with no NaN values.

    Returns
    -------
    dict with keys: total, valid, invalid, validity_rate (%)
    """
    total = len(df)
    valid = df.dropna().shape[0]
    invalid = total - valid
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "validity_rate": valid / total * 100,
    }


def run_validity_evaluation(train_datasets, methods, seeds):
    """
    Evaluate row validity (no-NaN rows) across all datasets.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, total, valid, invalid,
                                validity_rate, invalidity_rate
    """
    results = []
    for method, llm, seed, df in _iter_all(train_datasets, methods, seeds):
        v = count_valid_rows(df)
        results.append({
            "method": method,
            "llm": llm,
            "seed": seed,
            **v,
            "invalidity_rate": 100 - v["validity_rate"],
        })
    return pd.DataFrame(results)
