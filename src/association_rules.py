"""
Association Rule Mining Module for Perfume Formula Analysis

Finds ingredient co-occurrence patterns using the FP-Growth algorithm
(via mlxtend).  Surfaces "accords" — groups of ingredients that
frequently appear together.

Key metrics
-----------
support    : fraction of formulas that contain the itemset
confidence : P(consequent | antecedent)
lift       : how much more likely the rule fires vs. random chance
"""

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pathlib import Path
from typing import Optional


def find_frequent_itemsets(presence_matrix: pd.DataFrame,
                            min_support: float = 0.3) -> pd.DataFrame:
    """
    Mine frequent itemsets from a binary presence matrix.

    Parameters
    ----------
    presence_matrix : binary DataFrame (formulas x ingredients)
    min_support     : minimum fraction of formulas (0–1)

    Returns
    -------
    DataFrame with columns: support, itemsets
    """
    itemsets = fpgrowth(
        presence_matrix.astype(bool),
        min_support=min_support,
        use_colnames=True,
    )
    itemsets["itemset_size"] = itemsets["itemsets"].apply(len)
    return itemsets.sort_values(["itemset_size", "support"], ascending=[False, False])


def mine_association_rules(presence_matrix: pd.DataFrame,
                            min_support: float = 0.3,
                            min_confidence: float = 0.6,
                            min_lift: float = 1.0) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    Returns a DataFrame with antecedents, consequents, support,
    confidence, lift, and leverage — sorted by lift descending.
    """
    itemsets = find_frequent_itemsets(presence_matrix, min_support)

    if itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift]

    # Pretty-print itemsets as sorted lists
    rules["antecedents"] = rules["antecedents"].apply(lambda x: sorted(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: sorted(x))

    cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage"]
    return rules[cols].sort_values("lift", ascending=False).reset_index(drop=True)


def find_accords(presence_matrix: pd.DataFrame,
                 min_support: float = 0.3,
                 min_size: int = 2) -> pd.DataFrame:
    """
    Extract co-occurring ingredient groups ("accords") —
    frequent itemsets of size >= min_size.
    """
    itemsets = find_frequent_itemsets(presence_matrix, min_support)
    accords = itemsets[itemsets["itemset_size"] >= min_size].copy()
    accords["ingredients"] = accords["itemsets"].apply(lambda x: sorted(x))
    accords = accords.drop(columns=["itemsets"])
    return accords.reset_index(drop=True)


def print_rules_summary(rules: pd.DataFrame, top_n: int = 15) -> None:
    """Print a human-readable summary of the top association rules."""
    if rules.empty:
        print("No rules to display.")
        return

    print(f"=== Top {min(top_n, len(rules))} Association Rules (by lift) ===")
    for _, row in rules.head(top_n).iterrows():
        ant = " + ".join(row["antecedents"])
        con = " + ".join(row["consequents"])
        print(
            f"  [{ant}]  →  [{con}]"
            f"  | support={row['support']:.2f}"
            f"  conf={row['confidence']:.2f}"
            f"  lift={row['lift']:.2f}"
        )


def run_association_analysis(presence_matrix: pd.DataFrame,
                              min_support: float = 0.3,
                              min_confidence: float = 0.6,
                              output_dir: Optional[str] = None) -> dict:
    """
    Full association rule mining pipeline.

    Returns dict with 'itemsets', 'rules', and 'accords'.
    Optionally saves CSVs to output_dir.
    """
    itemsets = find_frequent_itemsets(presence_matrix, min_support)
    rules = mine_association_rules(presence_matrix, min_support, min_confidence)
    accords = find_accords(presence_matrix, min_support)

    print_rules_summary(rules)

    print(f"\n=== Frequent Ingredient Groups (Accords, size ≥ 2) ===")
    for _, row in accords.head(10).iterrows():
        print(f"  {row['ingredients']}  support={row['support']:.2f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        itemsets.to_csv(out / "frequent_itemsets.csv", index=False)
        if not rules.empty:
            rules.to_csv(out / "association_rules.csv", index=False)
        accords.to_csv(out / "accords.csv", index=False)
        print(f"\nResults saved to {output_dir}")

    return {"itemsets": itemsets, "rules": rules, "accords": accords}
