"""Feature engineering module.

Creates six domain-specific features from the cleaned (but un-encoded)
DataFrame.  Must be called **before** one-hot encoding and scaling.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

SERVICE_COLUMNS: List[str] = [
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create six engineered features from raw (cleaned) data.

    New columns:
    1. ``avg_monthly_charges`` — TotalCharges / (tenure + 1)
    2. ``is_short_tenure``     — 1 if tenure < 12
    3. ``is_long_tenure``      — 1 if tenure > 48
    4. ``service_count``       — count of active services among 5 service columns
    5. ``high_charger``        — 1 if MonthlyCharges > 70
    6. ``payment_risk``        — 1 if PaymentMethod == 'Electronic check'

    Args:
        df: Cleaned DataFrame with original column values still present.

    Returns:
        DataFrame with the six new columns appended.
    """
    df = df.copy()

    # 1. Average monthly charges
    df["avg_monthly_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    logger.debug("Created avg_monthly_charges")

    # 2. Short tenure indicator
    df["is_short_tenure"] = (df["tenure"] < 12).astype(int)
    logger.debug("Created is_short_tenure")

    # 3. Long tenure indicator
    df["is_long_tenure"] = (df["tenure"] > 48).astype(int)
    logger.debug("Created is_long_tenure")

    # 4. Service count (vectorised)
    #    InternetService counts if value != "No" (i.e. DSL or Fiber optic)
    #    All others count if value == "Yes"
    df["service_count"] = (
        (df["InternetService"] != "No").astype(int)
        + (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    logger.debug("Created service_count")

    # 5. High charger
    df["high_charger"] = (df["MonthlyCharges"] > 70).astype(int)
    logger.debug("Created high_charger")

    # 6. Payment risk
    df["payment_risk"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    logger.debug("Created payment_risk")

    logger.info(
        "Created 6 engineered features.  Total columns: %d", len(df.columns)
    )
    return df


def get_engineered_feature_names() -> List[str]:
    """Return the names of the six engineered feature columns.

    Returns:
        Ordered list of column names.
    """
    return [
        "avg_monthly_charges",
        "is_short_tenure",
        "is_long_tenure",
        "service_count",
        "high_charger",
        "payment_risk",
    ]
