"""Fairness metrics: Discrimination Score."""

import pandas as pd

from ._helpers import _iter_all


def calculate_discrimination_score(df, sensitive_col, sensitive_condition, target, target_positive):
    """
    Discrimination Score: DS = P(y=1 | s=1) - P(y=1 | s=0)

    Parameters
    ----------
    df                  : pd.DataFrame
    sensitive_col       : str — name of the sensitive attribute column
    sensitive_condition : callable — returns True for the privileged group (s=1)
    target              : str — target column name
    target_positive     : value — positive outcome value (e.g. '>50K', 1, True)

    Returns
    -------
    dict with keys: ds, p_y1_s1, p_y1_s0, n_s1, n_s0
    """
    df_clean = df.dropna()

    s1_mask = df_clean[sensitive_col].apply(sensitive_condition)
    s1 = df_clean[s1_mask]
    s0 = df_clean[~s1_mask]

    p_y1_s1 = (s1[target] == target_positive).mean()
    p_y1_s0 = (s0[target] == target_positive).mean()

    return {
        "ds": float(p_y1_s1 - p_y1_s0),
        "p_y1_s1": float(p_y1_s1),
        "p_y1_s0": float(p_y1_s0),
        "n_s1": len(s1),
        "n_s0": len(s0),
    }


def run_fairness_evaluation(
    train_datasets, methods, seeds,
    sensitive_col, sensitive_condition, target, target_positive
):
    """
    Run fairness evaluation across all methods and seeds.

    Parameters
    ----------
    sensitive_col       : str
    sensitive_condition : callable — e.g. ``lambda x: x == 'Male'``
    target              : str
    target_positive     : value

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, ds, p_y1_s1, p_y1_s0, n_s1, n_s0
    """
    results = []
    for method, llm, seed, df in _iter_all(train_datasets, methods, seeds):
        score = calculate_discrimination_score(
            df, sensitive_col, sensitive_condition, target, target_positive
        )
        results.append({"method": method, "llm": llm, "seed": seed, **score})
    return pd.DataFrame(results)
