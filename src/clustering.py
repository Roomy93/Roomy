"""
Clustering Module for Perfume Formula Analysis

Supports:
- K-Means clustering (with elbow method for k selection)
- Agglomerative (Hierarchical) Clustering with dendrogram
- Similarity metrics: Cosine, Euclidean, Jaccard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import jaccard, pdist, squareform


# ── Distance / Similarity helpers ────────────────────────────────────────────

def compute_similarity_matrix(matrix: pd.DataFrame,
                               metric: str = "cosine") -> pd.DataFrame:
    """
    Compute a pairwise similarity matrix for formulas.

    Parameters
    ----------
    matrix : formula matrix (formulas x ingredients) — ratio or binary
    metric : 'cosine' | 'euclidean' | 'jaccard'

    Returns
    -------
    DataFrame with formula IDs as both index and columns.
    """
    ids = matrix.index.tolist()
    X = matrix.values

    if metric == "cosine":
        sim = cosine_similarity(X)
    elif metric == "euclidean":
        dist = euclidean_distances(X)
        # Convert distance → similarity (0–1)
        sim = 1 / (1 + dist)
    elif metric == "jaccard":
        X_bin = (X > 0).astype(float)
        dist = squareform(pdist(X_bin, metric="jaccard"))
        sim = 1 - dist
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'cosine', 'euclidean', or 'jaccard'.")

    return pd.DataFrame(sim, index=ids, columns=ids)


# ── K-Means ──────────────────────────────────────────────────────────────────

def elbow_analysis(matrix: pd.DataFrame,
                   k_range: range = range(2, 8),
                   output_path: Optional[str] = None) -> dict:
    """
    Run K-Means for each k in k_range and record inertia + silhouette score.
    Plots the elbow curve.
    """
    X = StandardScaler().fit_transform(matrix.values)
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels) if k > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "bo-")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(list(k_range), silhouettes, "rs-")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")

    plt.tight_layout()
    _save_or_show(fig, output_path)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"Best k by silhouette: {best_k}")
    return {"inertias": inertias, "silhouettes": silhouettes, "best_k": best_k}


def kmeans_clustering(matrix: pd.DataFrame,
                       n_clusters: int = 3,
                       random_state: int = 42) -> pd.Series:
    """
    Fit K-Means and return cluster labels indexed by formula ID.
    """
    X = StandardScaler().fit_transform(matrix.values)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return pd.Series(labels, index=matrix.index, name="cluster")


# ── Hierarchical clustering ───────────────────────────────────────────────────

def hierarchical_clustering(matrix: pd.DataFrame,
                             n_clusters: int = 3,
                             linkage_method: str = "ward") -> pd.Series:
    """
    Agglomerative hierarchical clustering.
    Returns cluster labels indexed by formula ID.
    """
    X = StandardScaler().fit_transform(matrix.values)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=matrix.index, name="cluster")


def plot_dendrogram(matrix: pd.DataFrame,
                    linkage_method: str = "ward",
                    output_path: Optional[str] = None) -> None:
    """Plot a hierarchical dendrogram of formulas."""
    X = StandardScaler().fit_transform(matrix.values)
    Z = linkage(X, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        labels=matrix.index.tolist(),
        orientation="top",
        leaf_rotation=45,
        ax=ax,
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method} linkage)")
    ax.set_xlabel("Formula ID")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    _save_or_show(fig, output_path)


# ── Cluster profile ───────────────────────────────────────────────────────────

def cluster_profiles(matrix: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Return mean ingredient ratios per cluster — useful for interpreting
    what characterizes each cluster (e.g., "Cluster 2 is wood-heavy").
    """
    combined = matrix.copy()
    combined["cluster"] = labels
    profile = combined.groupby("cluster").mean()
    return profile


def print_cluster_summary(matrix: pd.DataFrame, labels: pd.Series) -> None:
    """Print which formulas belong to each cluster and their key ingredients."""
    profiles = cluster_profiles(matrix, labels)
    for cluster_id in sorted(labels.unique()):
        members = labels[labels == cluster_id].index.tolist()
        top_ingredients = (
            profiles.loc[cluster_id]
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        print(f"Cluster {cluster_id}: {members}")
        print(f"  Top ingredients: {top_ingredients}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_clustering(formula_matrix: pd.DataFrame,
                   presence_matrix: pd.DataFrame,
                   n_clusters: int = 3,
                   output_dir: Optional[str] = None) -> dict:
    """
    Full clustering pipeline: elbow analysis → K-Means → hierarchical.
    """
    out = Path(output_dir) if output_dir else None

    print("=== Elbow & Silhouette Analysis ===")
    elbow = elbow_analysis(
        formula_matrix,
        output_path=str(out / "elbow.png") if out else None,
    )

    best_k = elbow["best_k"]
    print(f"\n=== K-Means Clustering (k={best_k}) ===")
    km_labels = kmeans_clustering(formula_matrix, n_clusters=best_k)
    print_cluster_summary(formula_matrix, km_labels)

    print(f"\n=== Hierarchical Clustering (k={n_clusters}) ===")
    hc_labels = hierarchical_clustering(formula_matrix, n_clusters=n_clusters)
    print_cluster_summary(formula_matrix, hc_labels)

    plot_dendrogram(
        formula_matrix,
        output_path=str(out / "dendrogram.png") if out else None,
    )

    print("\n=== Cosine Similarity Matrix ===")
    sim_matrix = compute_similarity_matrix(formula_matrix, metric="cosine")
    print(sim_matrix.round(2))

    if out:
        km_labels.to_csv(out / "kmeans_labels.csv", header=True)
        hc_labels.to_csv(out / "hierarchical_labels.csv", header=True)
        sim_matrix.to_csv(out / "cosine_similarity.csv")
        print(f"\nResults saved to {output_dir}")

    return {
        "kmeans_labels": km_labels,
        "hierarchical_labels": hc_labels,
        "similarity_matrix": sim_matrix,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)
