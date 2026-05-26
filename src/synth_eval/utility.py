"""Utility evaluation: Train on Synthetic, Test on Real (TSTR)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from ._helpers import _iter_synthetic
from .utils import preprocess_for_ml


def evaluate_utility(train_df, test_df, target, task="classification", drop_cols=None, dataset=None, sub=False):
    """
    Train an XGBoost model on train_df and evaluate on test_df.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    target   : str  — name of the target column
    task     : 'classification' | 'regression'

    Returns
    -------
    dict
        Classification: {f1, auc, acc}
        Regression:     {rmse, mae, r2}
    """



    train_proc, encoders = preprocess_for_ml(train_df, fit=True)
    test_proc, _ = preprocess_for_ml(test_df, label_encoders=encoders, fit=False)

    X_train = train_proc.drop(columns=[target])
    y_train = train_proc[target]
    X_test = test_proc.drop(columns=[target])
    y_test = test_proc[target]

    if drop_cols is not None:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)




    if task == "classification":
        # Filter synthetic train to only keep rows with classes present in the real test set
        valid_classes = set(y_test.unique())
        invalid_mask = ~y_train.isin(valid_classes)
        if invalid_mask.any():
            n_dropped = invalid_mask.sum()
            print(f'  [utility] A remover {n_dropped} linhas sintéticas com classes inválidas {y_train[invalid_mask].unique().tolist()}.')
            X_train = X_train.loc[~invalid_mask]
            y_train = y_train.loc[~invalid_mask]

        # XGBoost requires contiguous 0-based classes — remap using train classes only
        unique_train = np.sort(y_train.unique())
        if unique_train[0] != 0 or not np.array_equal(unique_train, np.arange(len(unique_train))):
            class_map = {c: i for i, c in enumerate(unique_train)}
            y_train = y_train.map(class_map)
            # Classes in test not seen in train → -1 (will always be predicted wrong)
            y_test = y_test.map(class_map).fillna(-1).astype(int)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        if dataset == 'cardiotocography':
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # Rows with y_test == -1 (class not seen in train) are correctly penalised
            # in acc/f1 (model never predicts -1). For roc_auc, restrict to known classes.
            known_mask = y_test != -1
            return {
                "f1":  f1_score(y_test, y_pred, average='weighted', labels=model.classes_, zero_division=0),
                "auc": roc_auc_score(y_test[known_mask], y_prob[known_mask], multi_class='ovr', average='weighted', labels=model.classes_),
                "acc": accuracy_score(y_test, y_pred),
            }


        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            return {
                "f1": f1_score(y_test, y_pred, pos_label=1),
                "auc": roc_auc_score(y_test, y_prob),
                "acc": accuracy_score(y_test, y_pred),
            }

    elif task == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

    else:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")


def run_utility_evaluation(train_datasets, test_datasets, methods, seeds, target,
                           task="classification", drop_cols=None, dataset=None):
    """
    TSTR evaluation: for each seed, train on synthetic (or real) data and
    test on the held-out real test set.

    Parameters
    ----------
    train_datasets : dict  — {method: {seed: df} or {llm: {seed: df}}}
    test_datasets  : dict  — {seed: df}
    methods        : list[str]
    seeds          : list
    target         : str   — target column name
    task           : 'classification' | 'regression'

    Returns
    -------
    pd.DataFrame
    """
    results = []

    # Real baseline
    if "real" in methods:
        for seed in seeds:
            metrics = evaluate_utility(
                train_datasets["real"][seed], test_datasets[seed], target, task, drop_cols, dataset
            )
            results.append({"method": "real", "llm": "-", "seed": int(seed), **metrics})

    # Synthetic methods
    for method, llm, seed, train_df in _iter_synthetic(train_datasets, methods, seeds):
        print(f"Method: {method}, LLM: {llm}, seed: {seed}\n")
        metrics = evaluate_utility(train_df, test_datasets[seed], target, task, drop_cols, dataset)
        results.append({"method": method, "llm": llm, "seed": int(seed), **metrics})

    return pd.DataFrame(results)
