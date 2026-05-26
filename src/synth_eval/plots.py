"""Visualization helpers for structural analysis."""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._helpers import _iter_synthetic
from .utils import get_column_types


def collect_datasets_by_seed(train_datasets, test_datasets, methods, seed):
    """
    Assemble a {label: DataFrame} dict for a given seed, suitable for the
    multi-dataset plotting helpers below.

    Output order: real_train, real_test (if provided), then each synthetic method
    in the order given by ``methods`` (LLM frameworks expanded as method/llm).
    """
    out = {}
    if "real" in methods and "real" in train_datasets:
        out["real_train"] = train_datasets["real"][seed]
    if test_datasets is not None and seed in test_datasets:
        out["real_test"] = test_datasets[seed]
    for method, llm, _, df in _iter_synthetic(train_datasets, methods, [seed]):
        label = method if llm == "-" else f"{method}/{llm}"
        out[label] = df
    return out


def plot_numeric_distributions(
    datasets,
    columns=None,
    ncols=3,
    bins=40,
    figsize_per_subplot=(5, 3),
    outlier_quantiles=(0.005, 0.995),
    reference="real_train",
    title=None,
):
    """
    For each numerical column, overlay step histograms of every dataset.

    Outliers are clipped by computing per-column quantiles on the reference
    dataset (``"real_train"`` if present, else the first dataset). Synthetic
    methods can produce extreme out-of-range values that would otherwise squash
    the visible mass of every curve. Values outside the quantile window are
    dropped from each histogram (densities renormalise within the window).
    Pass ``outlier_quantiles=None`` to disable clipping.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
    columns : list[str] | None
    ncols : int
    bins : int
    outlier_quantiles : tuple[float, float] | None
    reference : str — label whose quantiles define the x-range
    """
    if not datasets:
        raise ValueError("datasets is empty.")
    labels = list(datasets.keys())
    ref_label = reference if reference in datasets else labels[0]
    first_df = datasets[ref_label]
    if columns is None:
        num_cols, _ = get_column_types(first_df)
        columns = num_cols
    if not columns:
        raise ValueError("No numerical columns to plot.")

    n = len(columns)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, label in enumerate(labels)}

    for i, col in enumerate(columns):
        ax = axes[i // ncols, i % ncols]
        series = {
            label: pd.to_numeric(df[col], errors="coerce").dropna().values
            for label, df in datasets.items()
            if col in df.columns
        }
        non_empty = [v for v in series.values() if len(v)]
        if not non_empty:
            ax.set_title(col, fontsize=10)
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        if outlier_quantiles is not None and col in first_df.columns:
            ref_vals = pd.to_numeric(first_df[col], errors="coerce").dropna().values
            base = ref_vals if len(ref_vals) else np.concatenate(non_empty)
            q_lo, q_hi = np.quantile(base, outlier_quantiles)
            lo, hi = float(q_lo), float(q_hi)
            if hi <= lo:
                lo = float(min(v.min() for v in non_empty))
                hi = float(max(v.max() for v in non_empty))
        else:
            lo = float(min(v.min() for v in non_empty))
            hi = float(max(v.max() for v in non_empty))

        edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins

        for label, v in series.items():
            if not len(v):
                continue
            ax.hist(
                v, bins=edges, density=True, histtype="step",
                linewidth=1.4, label=label, color=colors[label],
            )
        if isinstance(edges, np.ndarray):
            ax.set_xlim(lo, hi)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_categorical_frequencies(
    datasets,
    columns=None,
    max_categories=15,
    reference="real_train",
    figsize_per_column=(8, 3),
    annotate=False,
    title=None,
):
    """
    For each categorical column, draw a heatmap of relative frequencies with one
    row per dataset and one column per category.

    Categories are selected from the most frequent ``max_categories`` values of the
    reference dataset (default ``"real_train"``); if that label is absent, the first
    dataset in ``datasets`` is used.
    """
    if not datasets:
        raise ValueError("datasets is empty.")
    labels = list(datasets.keys())
    ref_label = reference if reference in datasets else labels[0]
    ref_df = datasets[ref_label]

    if columns is None:
        _, cat_cols = get_column_types(ref_df)
        columns = cat_cols
    if not columns:
        raise ValueError("No categorical columns to plot.")

    n = len(columns)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_column[0], figsize_per_column[1] * n),
        squeeze=False,
    )

    for i, col in enumerate(columns):
        ax = axes[i, 0]
        ref_freq = ref_df[col].astype(str).value_counts(normalize=True)
        cats = ref_freq.head(max_categories).index.tolist()

        # Append any extra categories that show up in non-reference datasets
        # but were not in the reference top-N (helps spot fabricated values).
        extra = set()
        for label, df in datasets.items():
            if label == ref_label or col not in df.columns:
                continue
            extra.update(df[col].astype(str).unique())
        extra -= set(cats) | set(ref_freq.index)
        if extra:
            cats = cats + sorted(extra)[: max(0, max_categories - len(cats))]

        matrix = np.zeros((len(labels), len(cats)), dtype=np.float32)
        for r, label in enumerate(labels):
            df = datasets[label]
            if col not in df.columns:
                continue
            f = df[col].astype(str).value_counts(normalize=True)
            matrix[r] = f.reindex(cats, fill_value=0.0).values

        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

        if annotate:
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    ax.text(
                        c, r, f"{matrix[r, c]:.2f}",
                        ha="center", va="center",
                        color="white" if matrix[r, c] < 0.5 else "black",
                        fontsize=7,
                    )

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
    else:
        fig.tight_layout()
    return fig


def plot_similarity_threshold_curves(
    curve_df,
    ncols=3,
    figsize_per_subplot=(4, 3),
    title=None,
):
    """
    For each (method, llm), plot percentage-similar-rows as a function of the
    similarity threshold, with separate lines for ``reference='train'`` (solid)
    and ``reference='test'`` (dashed).

    Curves are means across seeds; faint bands span min/max across seeds.

    Parameters
    ----------
    curve_df : pd.DataFrame
        Output of ``run_percentage_similar_rows_train_vs_test``.
    """
    pct_cols = [c for c in curve_df.columns if c.startswith("pct_")]
    thresholds = sorted(int(c.split("_")[1]) / 100 for c in pct_cols)
    pct_cols_sorted = [f"pct_{int(round(t * 100))}" for t in thresholds]

    # Separate the i.i.d. baseline (real_test → real_train) so we can overlay
    # it on every subplot.
    baseline_mask = (curve_df["method"] == "real_test")
    baseline_sub = curve_df[baseline_mask]
    syn_df = curve_df[~baseline_mask]

    if not baseline_sub.empty:
        baseline_mean = baseline_sub[pct_cols_sorted].mean().values
        baseline_mn = baseline_sub[pct_cols_sorted].min().values
        baseline_mx = baseline_sub[pct_cols_sorted].max().values
    else:
        baseline_mean = baseline_mn = baseline_mx = None

    pairs = syn_df[["method", "llm"]].drop_duplicates().values.tolist()
    n = len(pairs)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
    )

    for i, (method, llm) in enumerate(pairs):
        ax = axes[i // ncols, i % ncols]
        sub = syn_df[(syn_df["method"] == method) & (syn_df["llm"] == llm)]

        for ref, style, color in (("train", "-", "C0"), ("test", "--", "C1")):
            ref_sub = sub[sub["reference"] == ref]
            if ref_sub.empty:
                continue
            mean = ref_sub[pct_cols_sorted].mean().values
            mn = ref_sub[pct_cols_sorted].min().values
            mx = ref_sub[pct_cols_sorted].max().values
            ax.plot(thresholds, mean, linestyle=style, color=color,
                    label=f"syn vs {ref}", linewidth=1.6)
            ax.fill_between(thresholds, mn, mx, color=color, alpha=0.15)

        if baseline_mean is not None:
            ax.plot(thresholds, baseline_mean, linestyle=":",
                    color="black", linewidth=1.4,
                    label="real_test vs train (i.i.d.)")
            ax.fill_between(thresholds, baseline_mn, baseline_mx,
                            color="black", alpha=0.08)

        label = method if llm == "-" else f"{method}/{llm}"
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("threshold t", fontsize=8)
        ax.set_ylabel("% syn rows ≥ t", fontsize=8)
        ax.set_xlim(min(thresholds), max(thresholds))
        ax.set_ylim(0, max(1, ax.get_ylim()[1]))
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_match_column_contribution(
    df_long,
    value="hit_rate",
    annotate=False,
    figsize_per_row=0.4,
    figsize_width=12,
    title=None,
):
    """
    Heatmap of per-column match rates across methods (averaged over seeds).

    Parameters
    ----------
    df_long : pd.DataFrame
        Output of ``run_match_column_contribution_evaluation``.
    value : 'hit_rate' | 'lift' | 'baseline_rate'
    annotate : bool — overlay numerical values on cells
    """
    if value not in df_long.columns:
        raise ValueError(f"Unknown value '{value}'. "
                         f"Choose one of: {list(df_long.columns)}.")

    # For 'lift' replace +inf with NaN to keep the colour scale sane.
    df = df_long.copy()
    if value == "lift":
        df[value] = df[value].replace([np.inf, -np.inf], np.nan)

    pivot = (df.groupby(["method", "llm", "column"])[value]
               .mean()
               .unstack("column"))
    pivot.index = [m if l == "-" else f"{m}/{l}" for m, l in pivot.index]

    n_rows = len(pivot)
    fig, ax = plt.subplots(
        figsize=(figsize_width, max(2.5, figsize_per_row * n_rows + 1.8))
    )

    if value == "lift":
        cmap, vmin, vmax = "magma", 0.0, None
    else:
        cmap, vmin, vmax = "viridis", 0.0, 1.0

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title or f"per-column {value} (mean across seeds)", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    if annotate:
        vals = pivot.values
        finite = vals[np.isfinite(vals)]
        mid = float(np.nanmedian(finite)) if finite.size else 0.5
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                v = vals[r, c]
                if not np.isfinite(v):
                    continue
                ax.text(
                    c, r, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < mid else "black",
                    fontsize=7,
                )

    fig.tight_layout()
    return fig


def plot_categorical_frequencies_bars(
    datasets,
    columns=None,
    max_categories=10,
    reference="real_train",
    figsize_per_column=(12, 3.5),
    title=None,
):
    """
    For each categorical column, draw a grouped bar chart of relative frequencies
    with one bar per dataset within each top-N category group.

    Categories are taken from the most frequent ``max_categories`` values of the
    reference dataset (default ``"real_train"``).
    """
    if not datasets:
        raise ValueError("datasets is empty.")
    labels = list(datasets.keys())
    ref_label = reference if reference in datasets else labels[0]
    ref_df = datasets[ref_label]

    if columns is None:
        _, cat_cols = get_column_types(ref_df)
        columns = cat_cols
    if not columns:
        raise ValueError("No categorical columns to plot.")

    n = len(columns)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_column[0], figsize_per_column[1] * n),
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, label in enumerate(labels)}

    for i, col in enumerate(columns):
        ax = axes[i, 0]
        ref_freq = ref_df[col].astype(str).value_counts(normalize=True)
        cats = ref_freq.head(max_categories).index.tolist()

        n_groups = len(cats)
        n_bars = len(labels)
        x = np.arange(n_groups)
        total_width = 0.85
        bar_w = total_width / max(n_bars, 1)

        for k, label in enumerate(labels):
            df = datasets[label]
            if col not in df.columns:
                continue
            f = df[col].astype(str).value_counts(normalize=True)
            vals = f.reindex(cats, fill_value=0.0).values
            offset = (k - (n_bars - 1) / 2) * bar_w
            ax.bar(x + offset, vals, width=bar_w, label=label, color=colors[label])

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("freq", fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, ncol=min(n_bars, 4), loc="upper right")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
    else:
        fig.tight_layout()
    return fig


def plot_column_distributions(
    real_df,
    synthetic_df,
    columns=None,
    ncols=3,
    max_categories=15,
    bins=30,
    figsize_per_subplot=(4, 2.8),
    title=None,
):
    """
    Grid of distribution overlays for each column.

    Numeric columns: histograms (real vs synthetic) on shared edges, density-normalized.
    Categorical columns: side-by-side bars of relative frequencies, restricted to the
    top ``max_categories`` most-frequent real categories.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if columns is None:
        columns = [c for c in real_df.columns if c in synthetic_df.columns]
    n = len(columns)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
    )

    num_cols, _ = get_column_types(real_df[columns])
    num_set = set(num_cols)

    for i, col in enumerate(columns):
        ax = axes[i // ncols, i % ncols]
        if col in num_set:
            r = pd.to_numeric(real_df[col], errors="coerce").dropna().values
            s = pd.to_numeric(synthetic_df[col], errors="coerce").dropna().values
            if len(r) and len(s):
                lo = float(min(r.min(), s.min()))
                hi = float(max(r.max(), s.max()))
                edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
                ax.hist(r, bins=edges, alpha=0.5, label="real", density=True, color="C0")
                ax.hist(s, bins=edges, alpha=0.5, label="synthetic", density=True, color="C1")
        else:
            r_freq = real_df[col].astype(str).value_counts(normalize=True)
            top = r_freq.head(max_categories).index.tolist()
            s_freq = synthetic_df[col].astype(str).value_counts(normalize=True)
            r_vals = r_freq.reindex(top, fill_value=0.0).values
            s_vals = s_freq.reindex(top, fill_value=0.0).values
            x = np.arange(len(top))
            w = 0.4
            ax.bar(x - w / 2, r_vals, width=w, label="real", color="C0")
            ax.bar(x + w / 2, s_vals, width=w, label="synthetic", color="C1")
            ax.set_xticks(x)
            ax.set_xticklabels(top, rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_correlation_heatmaps(real_df, synthetic_df, method="pearson", cmap="coolwarm"):
    """
    Three-panel heatmap: correlation matrix of real, of synthetic, and the absolute
    difference. Uses numerical columns only.
    """
    num_cols, _ = get_column_types(real_df)
    num_cols = [c for c in num_cols if c in synthetic_df.columns]
    if not num_cols:
        raise ValueError("No common numerical columns to plot.")

    real_corr = real_df[num_cols].corr(method=method)
    syn_corr = synthetic_df[num_cols].corr(method=method)
    diff = (real_corr - syn_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        (axes[0], real_corr, f"Real ({method})", -1, 1, cmap),
        (axes[1], syn_corr, f"Synthetic ({method})", -1, 1, cmap),
        (axes[2], diff, "|Real − Synthetic|", 0, 1, "Reds"),
    ]
    for ax, mat, t, vmin, vmax, c in panels:
        im = ax.imshow(mat.values, vmin=vmin, vmax=vmax, cmap=c, aspect="auto")
        ax.set_xticks(np.arange(len(num_cols)))
        ax.set_yticks(np.arange(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(num_cols, fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_distance_summary(distance_df, metrics=None, figsize_per_metric=(4, 4)):
    """
    Bar chart of mean distribution distances per method (averaged over seeds).

    Parameters
    ----------
    distance_df : pd.DataFrame
        Output of ``run_distribution_distance_evaluation`` with columns:
        method, llm, seed, ks_mean, wass_mean, tvd_mean, jsd_mean.
    metrics : list[str] | None
        Subset of metric columns to plot. Defaults to whichever of the four
        ``*_mean`` columns are present.
    """
    if metrics is None:
        metrics = [
            c for c in ["ks_mean", "wass_mean", "tvd_mean", "jsd_mean"]
            if c in distance_df.columns
        ]
    if not metrics:
        raise ValueError("No distance columns found in distance_df.")

    agg = distance_df.groupby(["method", "llm"])[metrics].mean().reset_index()
    agg["label"] = agg.apply(
        lambda r: r["method"] if r["llm"] == "-" else f'{r["method"]}/{r["llm"]}',
        axis=1,
    )

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(figsize_per_metric[0] * n_metrics, figsize_per_metric[1]),
        squeeze=False,
    )
    for ax, metric in zip(axes[0], metrics):
        ax.bar(agg["label"], agg[metric], color="C0")
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.set_ylabel("mean across seeds", fontsize=9)
    fig.tight_layout()
    return fig


def plot_categorical_coverage(coverage_df, figsize=(10, 4)):
    """
    Bar chart of categorical-coverage metrics per method (averaged over seeds).

    Two panels:
      - Jaccard / coverage means (higher is better).
      - Mean number of new and missing categories (lower is better).
    """
    metrics_top = [c for c in ["jaccard_mean", "coverage_mean"] if c in coverage_df.columns]
    metrics_bot = [c for c in ["new_cats_mean", "missing_cats_mean"] if c in coverage_df.columns]

    agg = coverage_df.groupby(["method", "llm"])[metrics_top + metrics_bot].mean().reset_index()
    agg["label"] = agg.apply(
        lambda r: r["method"] if r["llm"] == "-" else f'{r["method"]}/{r["llm"]}',
        axis=1,
    )
    x = np.arange(len(agg))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    width = 0.4

    for i, m in enumerate(metrics_top):
        axes[0].bar(x + (i - 0.5) * width, agg[m], width=width, label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Cobertura de categorias (↑ melhor)", fontsize=10)
    axes[0].legend(fontsize=8)

    for i, m in enumerate(metrics_bot):
        axes[1].bar(x + (i - 0.5) * width, agg[m], width=width, label=m)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg["label"], rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Categorias novas / em falta (↓ melhor)", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig
