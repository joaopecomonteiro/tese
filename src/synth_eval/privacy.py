"""Privacy metrics: Distance to Closest Record (DCR) and Membership Inference Attack (MIA)."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from ._helpers import _iter_synthetic
from .utils import preprocess_for_ml


# ── DCR ───────────────────────────────────────────────────────────────────────

def evaluate_dcr(real_df, synthetic_df):
    """
    Compute Distance to Closest Record (DCR) from each synthetic record
    to the real training data.

    Columns are normalized with min-max scaling based on real data range.

    Returns
    -------
    dict with keys: dcr_mean, dcr_std, dcr_min, dcr_5th, dcr_median
    """
    real_enc, encoders = preprocess_for_ml(real_df.dropna(), fit=True)
    syn_enc, _ = preprocess_for_ml(synthetic_df.dropna(), label_encoders=encoders, fit=False)

    min_vals = real_enc.min()
    max_vals = real_enc.max()
    range_vals = (max_vals - min_vals).replace(0, 1)

    real_norm = (real_enc - min_vals) / range_vals
    syn_norm = (syn_enc - min_vals) / range_vals

    batch_size = 1000
    min_distances = []
    for i in range(0, len(syn_norm), batch_size):
        batch = syn_norm.iloc[i: i + batch_size]
        dists = cdist(batch, real_norm, metric="euclidean")
        min_distances.extend(dists.min(axis=1))

    min_distances = np.array(min_distances)
    return {
        "dcr_mean": float(np.mean(min_distances)),
        "dcr_std": float(np.std(min_distances)),
        "dcr_min": float(np.min(min_distances)),
        "dcr_5th": float(np.percentile(min_distances, 5)),
        "dcr_median": float(np.median(min_distances)),
    }


def run_dcr_evaluation(train_datasets, methods, seeds):
    """
    Run DCR evaluation for all synthetic methods and seeds.

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, dcr_median, dcr_mean,
                                dcr_std, dcr_min, dcr_5th
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        real_df = train_datasets["real"][seed]
        dcr = evaluate_dcr(real_df, syn_df)
        results.append({
            "method": method,
            "llm": llm,
            "seed": seed,
            "dcr_median": dcr["dcr_median"],
            "dcr_mean": dcr["dcr_mean"],
            "dcr_std": dcr["dcr_std"],
            "dcr_min": dcr["dcr_min"],
            "dcr_5th": dcr["dcr_5th"],
        })
    return pd.DataFrame(results)


# ── MIA ───────────────────────────────────────────────────────────────────────

def evaluate_mia(train_real_df, test_real_df, synthetic_df):
    """
    Distance-based Membership Inference Attack (MIA).

    Members (train_real_df) are expected to be closer to synthetic records
    than non-members (test_real_df).

    Returns
    -------
    float — AUC (0.5 = no privacy risk, > 0.5 = privacy risk)
    """
    train_enc, encoders = preprocess_for_ml(train_real_df, fit=True)
    test_enc, _ = preprocess_for_ml(test_real_df, label_encoders=encoders, fit=False)
    syn_enc, _ = preprocess_for_ml(synthetic_df, label_encoders=encoders, fit=False)

    min_vals = train_enc.min()
    max_vals = train_enc.max()
    range_vals = (max_vals - min_vals).replace(0, 1)

    train_norm = (train_enc - min_vals) / range_vals
    test_norm = (test_enc - min_vals) / range_vals
    syn_norm = (syn_enc - min_vals) / range_vals

    def _min_distances(real_norm, syn_norm, batch_size=1000):
        dists = []
        for i in range(0, len(real_norm), batch_size):
            batch = real_norm.iloc[i: i + batch_size]
            d = cdist(batch, syn_norm, metric="euclidean")
            dists.extend(d.min(axis=1))
        return np.array(dists)

    train_dists = _min_distances(train_norm, syn_norm)
    test_dists = _min_distances(test_norm, syn_norm)

    y_true = np.concatenate([np.ones(len(train_dists)), np.zeros(len(test_dists))])
    scores = np.concatenate([-train_dists, -test_dists])

    return float(roc_auc_score(y_true, scores))


def run_mia_evaluation(train_datasets, test_datasets, methods, seeds):
    """
    Run MIA evaluation for all synthetic methods and seeds.

    Parameters
    ----------
    test_datasets : dict — {seed: df}  — held-out real data (non-members)

    Returns
    -------
    pd.DataFrame with columns: method, llm, seed, mia_auc
    """
    results = []
    for method, llm, seed, syn_df in _iter_synthetic(train_datasets, methods, seeds):
        auc = evaluate_mia(train_datasets["real"][seed], test_datasets[seed], syn_df)
        results.append({"method": method, "llm": llm, "seed": int(seed), "mia_auc": auc})
    return pd.DataFrame(results)
