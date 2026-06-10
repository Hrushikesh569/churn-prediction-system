"""Stratified train / validation / test splitting.

Splits data 70 / 15 / 15 using two sequential ``train_test_split`` calls
with stratification on the target column to preserve class ratios.
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train / val / test sets.

    Two-step strategy:
    1. Split into *train* (70 %) vs *temp* (30 %).
    2. Split *temp* into *val* (50 %) and *test* (50 %)
       → each 15 % of the original data.

    Args:
        config: Must contain::

            data:
              target_column: "Churn"
            split:
              train_ratio: 0.70
              val_ratio: 0.15
              test_ratio: 0.15
              random_state: 42

    Returns:
        Tuple of ``(train_df, val_df, test_df)`` with original indices
        preserved.
    """
    target_col: str = config["data"]["target_column"]
    random_state: int = config["split"]["random_state"]

    val_ratio: float = config["split"]["val_ratio"]
    test_ratio: float = config["split"]["test_ratio"]
    temp_ratio: float = val_ratio + test_ratio          # 0.30

    # Step 1 — train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df[target_col],
    )

    # Step 2 — val vs test (50 / 50 of the 30 %)
    relative_test = test_ratio / temp_ratio              # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_state,
        stratify=temp_df[target_col],
    )

    # Log split details
    total = len(df)
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        churn_rate = split_df[target_col].mean()
        pct = len(split_df) / total * 100
        logger.info(
            "%s: %d rows (%.1f%%), churn rate: %.3f",
            name, len(split_df), pct, churn_rate,
        )

    return train_df, val_df, test_df
