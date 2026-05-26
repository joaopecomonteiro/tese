"""Internal helpers shared across modules."""

import pandas as pd


def _iter_synthetic(train_datasets, methods, seeds):
    """
    Yield (method, llm, seed, df) tuples for all non-real datasets.

    Automatically detects structure:
      - LLM frameworks: train_datasets[method][llm][seed] -> DataFrame
      - GANs / other:   train_datasets[method][seed]      -> DataFrame
    """
    for seed in seeds:
        for method in methods:
            if method == "real":
                continue
            if method not in train_datasets:
                continue
            first_val = next(iter(train_datasets[method].values()))
            if isinstance(first_val, dict):
                # LLM framework
                for llm in train_datasets[method]:
                    yield method, llm, seed, train_datasets[method][llm][seed]
            else:
                # GAN / other flat method
                yield method, "-", seed, train_datasets[method][seed]


def _iter_all(train_datasets, methods, seeds):
    """
    Like _iter_synthetic but also yields real data rows (llm='-').
    """
    for seed in seeds:
        for method in methods:
            if method not in train_datasets:
                continue
            if method == "real":
                yield method, "-", seed, train_datasets["real"][seed]
            else:
                first_val = next(iter(train_datasets[method].values()))
                if isinstance(first_val, dict):
                    for llm in train_datasets[method]:
                        yield method, llm, seed, train_datasets[method][llm][seed]
                else:
                    yield method, "-", seed, train_datasets[method][seed]
