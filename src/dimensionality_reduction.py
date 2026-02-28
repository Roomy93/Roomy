"""
Dimensionality Reduction & Visualization Module for Perfume Formula Analysis

Methods:
- PCA  (Principal Component Analysis)
- t-SNE
- UMAP  (requires the `umap-learn` package)

Each method reduces the high-dimensional formula matrix to 2D (or 3D)
so you can visually inspect clusters and outliers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(formula_matrix: pd.DataFrame,
            n_components: int = 2) -> tuple[pd.DataFrame, PCA]:
    """
    Fit PCA and return (embedding DataFrame, fitted PCA object).

    The embedding has columns ['PC1', 'PC2', ...] and is indexed by formula ID.
    """
    X = StandardScaler().fit_transform(formula_matrix.values)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    embedding = pd.DataFrame(coords, index=formula_matrix.index, columns=col_names)

    var_explained = pca.explained_variance_ratio_ * 100
    print("PCA – Variance explained per component:")
    for i, v in enumerate(var_explained, 1):
        print(f"  PC{i}: {v:.1f}%")
    print(f"  Total: {var_explained.sum():.1f}%")

    return embedding, pca


def plot_pca_variance(pca: PCA, output_path: Optional[str] = None) -> None:
    """Scree plot showing cumulative explained variance."""
    ratios = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(ratios) + 1), ratios, alpha=0.7, label="Individual")
    ax.plot(range(1, len(ratios) + 1), cumulative, "ro-", label="Cumulative")
    ax.axhline(90, color="gray", linestyle="--", alpha=0.6, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA – Explained Variance (Scree Plot)")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_pca_loadings(pca: PCA,
                      feature_names: list[str],
                      top_n: int = 10,
                      output_path: Optional[str] = None) -> None:
    """
    Plot the loadings of the top-N ingredients on PC1 and PC2,
    showing which ingredients drive each principal component.
    """
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=feature_names,
        columns=["PC1", "PC2"],
    )
    # Select ingredients with largest magnitude on either PC
    loadings["magnitude"] = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
    top = loadings.nlargest(top_n, "magnitude")

    fig, ax = plt.subplots(figsize=(8, 8))
    for ing, row in top.iterrows():
        ax.arrow(0, 0, row["PC1"], row["PC2"],
                 head_width=0.02, head_length=0.02, fc="steelblue", ec="steelblue")
        ax.text(row["PC1"] * 1.1, row["PC2"] * 1.1, ing, fontsize=8, ha="center")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("PC1 Loading")
    ax.set_ylabel("PC2 Loading")
    ax.set_title(f"PCA Loadings – Top {top_n} Ingredients")
    ax.set_aspect("equal")
    plt.tight_layout()
    _save_or_show(fig, output_path)


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(formula_matrix: pd.DataFrame,
             n_components: int = 2,
             perplexity: float = 5.0,
             random_state: int = 42) -> pd.DataFrame:
    """
    Fit t-SNE and return 2D embedding indexed by formula ID.

    Note: perplexity should be < n_samples.  With small datasets
    (< 30 formulas) set perplexity to n_samples / 3.
    """
    X = StandardScaler().fit_transform(formula_matrix.values)
    n = X.shape[0]
    perplexity = min(perplexity, n - 1)

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, max_iter=1000)
    coords = tsne.fit_transform(X)

    col_names = [f"tSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=formula_matrix.index, columns=col_names)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(formula_matrix: pd.DataFrame,
             n_components: int = 2,
             n_neighbors: int = 5,
             min_dist: float = 0.3,
             random_state: int = 42) -> pd.DataFrame:
    """
    Fit UMAP and return 2D embedding indexed by formula ID.
    Requires `umap-learn` package.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "UMAP is not installed. Run: pip install umap-learn"
        )

    X = StandardScaler().fit_transform(formula_matrix.values)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)
    col_names = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=formula_matrix.index, columns=col_names)


# ── Generic scatter plot ──────────────────────────────────────────────────────

def plot_embedding(embedding: pd.DataFrame,
                   labels: Optional[pd.Series] = None,
                   x_col: Optional[str] = None,
                   y_col: Optional[str] = None,
                   title: str = "Formula Embedding",
                   output_path: Optional[str] = None) -> None:
    """
    Scatter-plot a 2-column embedding.  Optionally color points by cluster label.

    Parameters
    ----------
    embedding  : DataFrame with at least 2 numeric columns
    labels     : optional Series of cluster labels (same index as embedding)
    x_col/y_col: column names to use as axes; defaults to first two columns
    """
    xcol = x_col or embedding.columns[0]
    ycol = y_col or embedding.columns[1]

    fig, ax = plt.subplots(figsize=(9, 7))

    if labels is not None:
        unique_labels = sorted(labels.unique())
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for lbl, color in zip(unique_labels, colors):
            mask = labels == lbl
            ax.scatter(
                embedding.loc[mask, xcol],
                embedding.loc[mask, ycol],
                label=f"Cluster {lbl}",
                color=color,
                s=100,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.legend(title="Cluster")
    else:
        ax.scatter(embedding[xcol], embedding[ycol], s=100, color="steelblue",
                   edgecolors="white", linewidths=0.5)

    # Annotate each point with the formula ID
    for formula_id, row in embedding.iterrows():
        ax.annotate(formula_id, (row[xcol], row[ycol]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, output_path)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_dimensionality_reduction(formula_matrix: pd.DataFrame,
                                  labels: Optional[pd.Series] = None,
                                  output_dir: Optional[str] = None) -> dict:
    """
    Run PCA, t-SNE, and UMAP; produce scatter plots.

    Returns dict with embeddings: 'pca', 'tsne', 'umap'.
    """
    out = Path(output_dir) if output_dir else None
    results = {}

    # PCA
    print("=== PCA ===")
    pca_embed, pca_model = run_pca(formula_matrix)
    results["pca"] = pca_embed

    plot_pca_variance(
        pca_model,
        output_path=str(out / "pca_scree.png") if out else None,
    )
    plot_embedding(
        pca_embed, labels=labels,
        title="PCA – Formula Map",
        output_path=str(out / "pca_scatter.png") if out else None,
    )
    plot_pca_loadings(
        pca_model, list(formula_matrix.columns),
        output_path=str(out / "pca_loadings.png") if out else None,
    )

    # t-SNE
    print("\n=== t-SNE ===")
    tsne_embed = run_tsne(formula_matrix)
    results["tsne"] = tsne_embed
    plot_embedding(
        tsne_embed, labels=labels,
        title="t-SNE – Formula Map",
        output_path=str(out / "tsne_scatter.png") if out else None,
    )

    # UMAP
    print("\n=== UMAP ===")
    try:
        umap_embed = run_umap(formula_matrix)
        results["umap"] = umap_embed
        plot_embedding(
            umap_embed, labels=labels,
            title="UMAP – Formula Map",
            output_path=str(out / "umap_scatter.png") if out else None,
        )
    except ImportError as e:
        print(f"Skipping UMAP: {e}")

    if out:
        for name, embed in results.items():
            embed.to_csv(out / f"{name}_embedding.csv")
        print(f"\nEmbeddings saved to {output_dir}")

    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)
