"""
Exploratory Data Analysis (EDA) Module for Perfume Formula Analysis

Provides:
- Ingredient frequency analysis
- Usage statistics (mean, min, max, median ratio)
- Correlation heatmap between ingredients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def ingredient_frequency(df: pd.DataFrame,
                          formula_col: str = "formula_id",
                          ingredient_col: str = "ingredient") -> pd.DataFrame:
    """
    Count in how many formulas each ingredient appears and compute its usage rate.

    Returns a DataFrame sorted by frequency (descending).
    """
    total_formulas = df[formula_col].nunique()
    freq = (
        df.groupby(ingredient_col)[formula_col]
        .nunique()
        .rename("formula_count")
        .reset_index()
    )
    freq["usage_rate_pct"] = (freq["formula_count"] / total_formulas * 100).round(1)
    return freq.sort_values("formula_count", ascending=False).reset_index(drop=True)


def ingredient_usage_stats(df: pd.DataFrame,
                            ingredient_col: str = "ingredient",
                            ratio_col: str = "ratio_ppt") -> pd.DataFrame:
    """
    Compute descriptive statistics for each ingredient's usage ratio (ppt).
    Only includes rows where the ingredient is actually present (ratio > 0).
    """
    present = df[df[ratio_col] > 0]
    stats = (
        present.groupby(ingredient_col)[ratio_col]
        .agg(["mean", "median", "min", "max", "std", "count"])
        .rename(columns={
            "mean": "mean_ppt",
            "median": "median_ppt",
            "min": "min_ppt",
            "max": "max_ppt",
            "std": "std_ppt",
            "count": "n_formulas",
        })
        .round(2)
        .reset_index()
        .sort_values("n_formulas", ascending=False)
        .reset_index(drop=True)
    )
    return stats


def correlation_matrix(formula_matrix: pd.DataFrame,
                        method: str = "pearson") -> pd.DataFrame:
    """
    Compute pairwise correlation between ingredients across formulas.

    Parameters
    ----------
    formula_matrix : wide DataFrame (formulas x ingredients), values = ratio_ppt
    method         : 'pearson' | 'spearman' | 'kendall'
    """
    return formula_matrix.corr(method=method)


def plot_frequency_bar(freq_df: pd.DataFrame,
                        top_n: int = 20,
                        output_path: Optional[str] = None) -> None:
    """Bar chart of the top-N most-used ingredients by formula count."""
    top = freq_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["ingredient"][::-1], top["usage_rate_pct"][::-1], color="steelblue")
    ax.set_xlabel("Usage Rate (%)")
    ax.set_title(f"Top {top_n} Most Frequently Used Ingredients")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_usage_boxplot(df: pd.DataFrame,
                        top_n: int = 15,
                        ratio_col: str = "ratio_ppt",
                        ingredient_col: str = "ingredient",
                        output_path: Optional[str] = None) -> None:
    """Box plot of ratio distributions for the top-N ingredients."""
    freq = ingredient_frequency(df, ingredient_col=ingredient_col)
    top_ingredients = freq.head(top_n)["ingredient"].tolist()
    subset = df[df[ingredient_col].isin(top_ingredients) & (df[ratio_col] > 0)]

    order = (
        subset.groupby(ingredient_col)[ratio_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=subset, x=ingredient_col, y=ratio_col, order=order, ax=ax, palette="pastel")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Ratio (ppt)")
    ax.set_title(f"Usage Distribution of Top {top_n} Ingredients")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              top_n: int = 20,
                              output_path: Optional[str] = None) -> None:
    """
    Heatmap of the correlation matrix.
    Shows only the top-N ingredients by overall absolute correlation.
    """
    # Select top-N ingredients by mean absolute correlation with others
    if corr_matrix.shape[0] > top_n:
        scores = corr_matrix.abs().mean().sort_values(ascending=False)
        top_ingredients = scores.head(top_n).index.tolist()
        corr_matrix = corr_matrix.loc[top_ingredients, top_ingredients]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=corr_matrix.shape[0] <= 15,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Ingredient Correlation Heatmap")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def run_eda(df: pd.DataFrame,
            formula_matrix: pd.DataFrame,
            output_dir: Optional[str] = None) -> dict:
    """
    Run the full EDA pipeline and optionally save plots.

    Returns a dict with frequency, stats, and correlation results.
    """
    freq = ingredient_frequency(df)
    stats = ingredient_usage_stats(df)
    corr = correlation_matrix(formula_matrix)

    print("=== Ingredient Frequency (Top 10) ===")
    print(freq.head(10).to_string(index=False))
    print()
    print("=== Usage Statistics (Top 10 Ingredients) ===")
    print(stats.head(10).to_string(index=False))

    out = Path(output_dir) if output_dir else None

    plot_frequency_bar(
        freq,
        output_path=str(out / "frequency_bar.png") if out else None,
    )
    plot_usage_boxplot(
        df,
        output_path=str(out / "usage_boxplot.png") if out else None,
    )
    plot_correlation_heatmap(
        corr,
        output_path=str(out / "correlation_heatmap.png") if out else None,
    )

    return {"frequency": freq, "stats": stats, "correlation": corr}


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)
