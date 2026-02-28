"""
Data Preprocessing Module for Perfume Formula Analysis

Handles:
- Ingredient name standardization
- Ratio normalization (to 100% or 1000ppt basis)
- Dilution correction (convert to pure ingredient basis)
"""

import pandas as pd
import numpy as np
from typing import Optional


# Synonym map: maps alternative names -> canonical name
INGREDIENT_SYNONYMS: dict[str, str] = {
    "Tetramethyl acetyloctahydronaphthalenes": "Iso E Super",
    "TMAONH": "Iso E Super",
    "Karanal": "Iso E Super",
    "Ambroxan": "Ambroxan",
    "Ambroxide": "Ambroxan",
    "Ambrofix": "Ambroxan",
    "PEA": "Phenylethyl Alcohol",
    "Phenylethyl alcohol": "Phenylethyl Alcohol",
    "2-Phenylethanol": "Phenylethyl Alcohol",
    "Habanolide": "Musks (Habanolide)",
    "Exaltolide": "Musks (Habanolide)",
    "Hedione HC": "Hedione",
    "Methyl dihydrojasmonate": "Hedione",
}


def standardize_ingredient_names(df: pd.DataFrame, column: str = "ingredient") -> pd.DataFrame:
    """
    Standardize ingredient names using the synonym map.
    Strips whitespace and applies case-insensitive matching.
    """
    df = df.copy()
    df[column] = df[column].str.strip()

    # Build a lowercased lookup for case-insensitive matching
    synonym_lower = {k.lower(): v for k, v in INGREDIENT_SYNONYMS.items()}

    def _standardize(name: str) -> str:
        return synonym_lower.get(name.lower(), name)

    df[column] = df[column].apply(_standardize)
    return df


def apply_dilution_correction(df: pd.DataFrame,
                               amount_col: str = "amount_g",
                               dilution_col: str = "dilution_pct") -> pd.DataFrame:
    """
    Convert ingredient amounts to pure (100%) basis.

    pure_amount = amount * (dilution_pct / 100)
    """
    df = df.copy()
    df["pure_amount_g"] = df[amount_col] * (df[dilution_col] / 100.0)
    return df


def normalize_formula_ratios(df: pd.DataFrame,
                              formula_col: str = "formula_id",
                              amount_col: str = "pure_amount_g") -> pd.DataFrame:
    """
    Normalize each formula so all ingredient amounts sum to 1000 ppt (parts per thousand).
    This allows fair comparison across formulas of different total sizes.
    """
    df = df.copy()
    formula_totals = df.groupby(formula_col)[amount_col].transform("sum")
    df["ratio_ppt"] = (df[amount_col] / formula_totals) * 1000.0
    return df


def build_formula_matrix(df: pd.DataFrame,
                          formula_col: str = "formula_id",
                          ingredient_col: str = "ingredient",
                          value_col: str = "ratio_ppt",
                          fill_value: float = 0.0) -> pd.DataFrame:
    """
    Pivot the long-format dataframe into a wide formula matrix.

    Rows    = formulas
    Columns = ingredients
    Values  = ratio_ppt (0 if ingredient not in formula)
    """
    matrix = df.pivot_table(
        index=formula_col,
        columns=ingredient_col,
        values=value_col,
        aggfunc="sum",
        fill_value=fill_value,
    )
    matrix.columns.name = None
    return matrix


def build_presence_matrix(formula_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a ratio matrix to a binary presence/absence matrix (0 or 1).
    Used for Jaccard similarity and association rule mining.
    """
    return (formula_matrix > 0).astype(int)


def preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df_clean      : cleaned long-format dataframe with pure_amount_g and ratio_ppt
    formula_matrix: wide ratio matrix (formulas x ingredients)
    presence_matrix: wide binary matrix  (formulas x ingredients)
    """
    df = pd.read_csv(csv_path)

    df = standardize_ingredient_names(df)
    df = apply_dilution_correction(df)
    df = normalize_formula_ratios(df)

    formula_matrix = build_formula_matrix(df)
    presence_matrix = build_presence_matrix(formula_matrix)

    return df, formula_matrix, presence_matrix


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_formulas.csv"
    df_clean, fm, pm = preprocess(path)
    print("Cleaned data shape:", df_clean.shape)
    print("Formula matrix:", fm.shape)
    print("\nFirst 5 formulas, first 6 ingredients:")
    print(fm.iloc[:5, :6].round(1))
