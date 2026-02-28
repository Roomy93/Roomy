"""
Network Analysis Module for Perfume Formula Analysis

Builds and analyzes ingredient co-occurrence networks:
- Nodes  = ingredients
- Edges  = co-occurrence in the same formula (weighted by frequency)

Key metrics:
- Degree centrality  : how many other ingredients an ingredient co-occurs with
- Betweenness centrality : how often an ingredient lies on the shortest path between others
- PageRank           : importance considering the importance of neighbors
- Community detection: groups of tightly co-occurring ingredients (accords)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from pathlib import Path
from typing import Optional
from itertools import combinations


# ── Graph construction ────────────────────────────────────────────────────────

def build_cooccurrence_graph(presence_matrix: pd.DataFrame,
                              min_cooccurrence: int = 2) -> nx.Graph:
    """
    Build a weighted ingredient co-occurrence graph from a binary presence matrix.

    Edge weight = number of formulas in which both ingredients appear together.
    Only edges with weight >= min_cooccurrence are retained.
    """
    G = nx.Graph()
    ingredients = presence_matrix.columns.tolist()
    G.add_nodes_from(ingredients)

    # Count co-occurrences for each pair
    for ing_a, ing_b in combinations(ingredients, 2):
        both_present = (
            (presence_matrix[ing_a] == 1) & (presence_matrix[ing_b] == 1)
        ).sum()
        if both_present >= min_cooccurrence:
            G.add_edge(ing_a, ing_b, weight=int(both_present))

    return G


def build_formula_ingredient_bipartite(presence_matrix: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph with formula nodes and ingredient nodes.
    Used to see which formulas share ingredients.
    """
    B = nx.Graph()
    formulas = presence_matrix.index.tolist()
    ingredients = presence_matrix.columns.tolist()

    B.add_nodes_from(formulas, bipartite=0)
    B.add_nodes_from(ingredients, bipartite=1)

    for formula in formulas:
        for ing in ingredients:
            if presence_matrix.loc[formula, ing] == 1:
                B.add_edge(formula, ing)

    return B


# ── Centrality metrics ────────────────────────────────────────────────────────

def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """
    Compute key centrality metrics for all nodes in the graph.

    Returns a DataFrame sorted by PageRank descending.
    """
    if len(G.nodes) == 0:
        return pd.DataFrame()

    degree_c = nx.degree_centrality(G)
    between_c = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight")

    df = pd.DataFrame({
        "ingredient": list(degree_c.keys()),
        "degree_centrality": list(degree_c.values()),
        "betweenness_centrality": [between_c[n] for n in degree_c],
        "pagerank": [pagerank[n] for n in degree_c],
        "degree": [G.degree(n) for n in degree_c],
    })
    return df.sort_values("pagerank", ascending=False).reset_index(drop=True).round(4)


# ── Community detection ───────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict[str, int]:
    """
    Detect communities using the Louvain method (greedy modularity).
    Returns a dict mapping ingredient -> community ID.
    """
    try:
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        raise ImportError("Requires networkx >= 2.4")

    communities = greedy_modularity_communities(G, weight="weight")
    community_map = {}
    for cid, members in enumerate(communities):
        for node in members:
            community_map[node] = cid
    return community_map


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_ingredient_network(G: nx.Graph,
                             centrality: Optional[pd.DataFrame] = None,
                             community_map: Optional[dict] = None,
                             title: str = "Ingredient Co-occurrence Network",
                             output_path: Optional[str] = None) -> None:
    """
    Draw the co-occurrence network.

    - Node size   ∝ PageRank (or degree if centrality not provided)
    - Node color  = community (if detected)
    - Edge width  ∝ co-occurrence weight
    """
    if len(G.nodes) == 0:
        print("Graph is empty — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5)

    # Node sizes
    if centrality is not None:
        pr = centrality.set_index("ingredient")["pagerank"]
        node_sizes = [pr.get(n, 0.01) * 8000 + 200 for n in G.nodes]
    else:
        deg = dict(G.degree())
        max_deg = max(deg.values()) if deg else 1
        node_sizes = [deg[n] / max_deg * 1500 + 200 for n in G.nodes]

    # Node colors
    if community_map:
        n_communities = len(set(community_map.values()))
        cmap = cm.tab10
        node_colors = [cmap(community_map.get(n, 0) / max(n_communities - 1, 1))
                       for n in G.nodes]
    else:
        node_colors = "steelblue"

    # Edge widths
    max_weight = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    edge_widths = [G[u][v]["weight"] / max_weight * 5 for u, v in G.edges]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           alpha=0.5, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_hub_ingredients(centrality: pd.DataFrame,
                          top_n: int = 10,
                          output_path: Optional[str] = None) -> None:
    """Bar chart of the top-N hub ingredients by PageRank."""
    top = centrality.head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, ["pagerank", "degree_centrality", "betweenness_centrality"]):
        bars = ax.barh(top["ingredient"][::-1], top[metric][::-1], color="coral")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(f"Top {top_n} by {metric.replace('_', ' ').title()}")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_network_analysis(presence_matrix: pd.DataFrame,
                          min_cooccurrence: int = 2,
                          output_dir: Optional[str] = None) -> dict:
    """
    Full network analysis pipeline.

    Returns dict with 'graph', 'centrality', 'communities'.
    """
    out = Path(output_dir) if output_dir else None

    print("=== Building Co-occurrence Graph ===")
    G = build_cooccurrence_graph(presence_matrix, min_cooccurrence)
    print(f"  Nodes (ingredients): {G.number_of_nodes()}")
    print(f"  Edges (co-occurrences): {G.number_of_edges()}")

    print("\n=== Centrality Metrics ===")
    centrality = compute_centrality(G)
    print(centrality.head(10).to_string(index=False))

    print("\n=== Community Detection ===")
    communities = {}
    if G.number_of_edges() > 0:
        communities = detect_communities(G)
        for cid in sorted(set(communities.values())):
            members = [n for n, c in communities.items() if c == cid]
            print(f"  Community {cid}: {members}")

    plot_ingredient_network(
        G, centrality=centrality, community_map=communities,
        output_path=str(out / "ingredient_network.png") if out else None,
    )
    plot_hub_ingredients(
        centrality,
        output_path=str(out / "hub_ingredients.png") if out else None,
    )

    if out:
        out.mkdir(parents=True, exist_ok=True)
        centrality.to_csv(out / "centrality.csv", index=False)
        pd.DataFrame(
            [{"ingredient": k, "community": v} for k, v in communities.items()]
        ).to_csv(out / "communities.csv", index=False)
        nx.write_graphml(G, str(out / "ingredient_network.graphml"))
        print(f"\nResults saved to {output_dir}")

    return {"graph": G, "centrality": centrality, "communities": communities}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)
