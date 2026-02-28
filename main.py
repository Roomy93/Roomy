#!/usr/bin/env python3
"""
Perfume Formula Analysis — Main Entry Point

Usage
-----
# Run all analyses on the sample dataset
python main.py

# Specify a custom CSV file and output directory
python main.py --csv data/my_formulas.csv --output output/my_run

# Run only specific analyses
python main.py --analyses eda clustering

Available analyses:
  eda          Exploratory Data Analysis (frequency, stats, correlation heatmap)
  association  Association rule mining (frequent ingredient groups)
  clustering   K-Means + Hierarchical clustering + dendrogram
  reduction    Dimensionality reduction (PCA, t-SNE, UMAP)
  network      Ingredient co-occurrence network analysis
  all          Run all of the above (default)
"""

import argparse
import sys
from pathlib import Path

from src.preprocessing import preprocess
from src.eda import run_eda
from src.association_rules import run_association_analysis
from src.clustering import run_clustering, kmeans_clustering
from src.dimensionality_reduction import run_dimensionality_reduction
from src.network_analysis import run_network_analysis


AVAILABLE_ANALYSES = ["eda", "association", "clustering", "reduction", "network"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perfume Formula Pattern Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        default="data/sample_formulas.csv",
        help="Path to the formula CSV file (default: data/sample_formulas.csv)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for saving plots and results (default: output/)",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["all"],
        choices=AVAILABLE_ANALYSES + ["all"],
        help="Which analyses to run (default: all)",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.3,
        help="Minimum support for association rules (default: 0.3)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for association rules (default: 0.6)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters for K-Means / hierarchical (default: 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display plots interactively instead of saving to files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = None if args.no_save else args.output
    analyses = AVAILABLE_ANALYSES if "all" in args.analyses else args.analyses

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 – Preprocessing")
    print("=" * 60)
    df, formula_matrix, presence_matrix = preprocess(csv_path)
    print(f"  Formulas loaded  : {formula_matrix.shape[0]}")
    print(f"  Unique ingredients: {formula_matrix.shape[1]}")

    results: dict = {
        "df": df,
        "formula_matrix": formula_matrix,
        "presence_matrix": presence_matrix,
    }

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if "eda" in analyses:
        print("\n" + "=" * 60)
        print("STEP 2 – Exploratory Data Analysis")
        print("=" * 60)
        eda_out = f"{output_dir}/eda" if output_dir else None
        eda_results = run_eda(df, formula_matrix, output_dir=eda_out)
        results["eda"] = eda_results

    # ── 3. Association rules ──────────────────────────────────────────────────
    if "association" in analyses:
        print("\n" + "=" * 60)
        print("STEP 3 – Association Rule Mining")
        print("=" * 60)
        assoc_out = f"{output_dir}/association" if output_dir else None
        assoc_results = run_association_analysis(
            presence_matrix,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            output_dir=assoc_out,
        )
        results["association"] = assoc_results

    # ── 4. Clustering ─────────────────────────────────────────────────────────
    if "clustering" in analyses:
        print("\n" + "=" * 60)
        print("STEP 4 – Clustering")
        print("=" * 60)
        cluster_out = f"{output_dir}/clustering" if output_dir else None
        cluster_results = run_clustering(
            formula_matrix,
            presence_matrix,
            n_clusters=args.n_clusters,
            output_dir=cluster_out,
        )
        results["clustering"] = cluster_results

    # ── 5. Dimensionality reduction ────────────────────────────────────────────
    if "reduction" in analyses:
        print("\n" + "=" * 60)
        print("STEP 5 – Dimensionality Reduction")
        print("=" * 60)
        # Use K-Means labels for coloring if clustering was run
        labels = results.get("clustering", {}).get("kmeans_labels")
        if labels is None:
            labels = kmeans_clustering(formula_matrix, n_clusters=args.n_clusters)

        reduction_out = f"{output_dir}/reduction" if output_dir else None
        reduction_results = run_dimensionality_reduction(
            formula_matrix,
            labels=labels,
            output_dir=reduction_out,
        )
        results["reduction"] = reduction_results

    # ── 6. Network analysis ───────────────────────────────────────────────────
    if "network" in analyses:
        print("\n" + "=" * 60)
        print("STEP 6 – Network Analysis")
        print("=" * 60)
        network_out = f"{output_dir}/network" if output_dir else None
        network_results = run_network_analysis(
            presence_matrix,
            output_dir=network_out,
        )
        results["network"] = network_results

    print("\n" + "=" * 60)
    print("Analysis complete.")
    if output_dir:
        print(f"All outputs saved under: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
