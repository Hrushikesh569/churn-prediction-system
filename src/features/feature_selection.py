"""Feature selection using mutual information.

Provides an informational ranking of features by their mutual-information
score with the target.  Does **not** drop features automatically.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


def rank_features_by_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by mutual information with the target variable.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Binary target vector.
        feature_names: Names corresponding to the columns of *X*.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns ``['feature', 'mi_score']`` sorted in
        descending order.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    mi_df = (
        pd.DataFrame({"feature": feature_names, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top 10 features by mutual information:\n%s", mi_df.head(10).to_string())
    return mi_df
