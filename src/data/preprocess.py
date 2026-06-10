"""Data preprocessing module.

Handles cleaning, type conversion, missing-value imputation,
outlier capping, and a reusable sklearn ``ColumnTransformer`` pipeline
(StandardScaler + OneHotEncoder).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------- column groups (after feature engineering) ------------------

NUMERIC_FEATURES: List[str] = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_charges", "service_count",
]

BINARY_ENGINEERED: List[str] = [
    "SeniorCitizen",
    "is_short_tenure", "is_long_tenure",
    "high_charger", "payment_risk",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


# ======================================================================
# Raw-data cleaning
# ======================================================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: fix types, handle blanks, encode target.

    Operations performed (in order):
    1. Drop ``customerID`` column.
    2. Convert ``TotalCharges`` to numeric (blanks / spaces → 0).
    3. Encode ``Churn`` as 0 / 1.
    4. Impute any remaining missing values (median for numeric, mode for
       categorical).

    Args:
        df: Raw DataFrame straight from the CSV.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # 1. Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column")

    # 2. Fix TotalCharges
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_fixed = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    logger.info("TotalCharges: converted to numeric, filled %d blanks/NaN with 0", n_fixed)

    # 3. Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        logger.info("Encoded Churn: %s", df["Churn"].value_counts().to_dict())

    # 4. Impute remaining missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        logger.warning("Remaining missing values:\n%s", missing[missing > 0])
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ("float64", "int64"):
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    logger.debug("Imputed %s with median=%s", col, fill_val)
                else:
                    fill_val = df[col].mode().iloc[0]
                    df[col] = df[col].fillna(fill_val)
                    logger.debug("Imputed %s with mode=%s", col, fill_val)
        logger.info("Imputed all remaining missing values")

    return df


# ======================================================================
# Outlier handling (IQR → 99th-percentile cap)
# ======================================================================

def detect_and_cap_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Detect and cap outliers using the IQR method at the 99th percentile.

    Thresholds are computed from *reference_df* (the training set) so they
    can later be re-applied to validation / test sets without leakage.

    Args:
        df: DataFrame to process.
        numeric_cols: Columns to check for outliers.
        reference_df: DataFrame used to compute thresholds.
            If ``None``, ``df`` itself is used (only appropriate for the
            training set).

    Returns:
        Tuple of ``(processed_df, cap_values)`` where *cap_values* maps
        column name → cap threshold.
    """
    df = df.copy()
    ref = reference_df if reference_df is not None else df
    cap_values: Dict[str, float] = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        cap_val = float(ref[col].quantile(0.99))
        cap_values[col] = cap_val
        n_capped = int((df[col] > cap_val).sum())
        df[col] = df[col].clip(upper=cap_val)
        if n_capped > 0:
            logger.info("Capped %d outliers in '%s' at %.2f", n_capped, col, cap_val)

    return df, cap_values


def apply_outlier_caps(
    df: pd.DataFrame,
    cap_values: Dict[str, float],
) -> pd.DataFrame:
    """Apply pre-computed outlier caps (from training set) to a DataFrame.

    Args:
        df: DataFrame to process.
        cap_values: Mapping of column name → cap value.

    Returns:
        DataFrame with outliers capped.
    """
    df = df.copy()
    for col, cap_val in cap_values.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap_val)
    return df


# ======================================================================
# Sklearn preprocessing pipeline (ColumnTransformer)
# ======================================================================

def build_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    binary_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a reusable ``ColumnTransformer`` pipeline.

    * Numeric features → ``StandardScaler``
    * Categorical features → ``OneHotEncoder(drop='first')``
    * Binary features → passthrough

    Args:
        numeric_features: Override list of numeric column names.
        categorical_features: Override list of categorical column names.
        binary_features: Override list of binary (0/1) column names.

    Returns:
        Un-fitted ``ColumnTransformer``.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if binary_features is None:
        binary_features = BINARY_ENGINEERED

    transformers = [
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        ),
        ("bin", "passthrough", binary_features),
    ]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    logger.info(
        "Built preprocessing pipeline: %d numeric, %d categorical, %d binary features",
        len(numeric_features),
        len(categorical_features),
        len(binary_features),
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract feature names from a *fitted* ``ColumnTransformer``.

    Args:
        preprocessor: Fitted ``ColumnTransformer``.

    Returns:
        Ordered list of output feature names.
    """
    return list(preprocessor.get_feature_names_out())


# ======================================================================
# Persistence helpers
# ======================================================================

def save_pipeline(pipeline: ColumnTransformer, path: str) -> None:
    """Serialise a fitted pipeline to disk with ``joblib``.

    Args:
        pipeline: Fitted ``ColumnTransformer``.
        path: Destination file path.
    """
    joblib.dump(pipeline, path)
    logger.info("Saved preprocessing pipeline to %s", path)


def load_pipeline(path: str) -> ColumnTransformer:
    """Load a previously saved pipeline from disk.

    Args:
        path: Path to the joblib file.

    Returns:
        Fitted ``ColumnTransformer``.
    """
    pipeline: ColumnTransformer = joblib.load(path)
    logger.info("Loaded preprocessing pipeline from %s", path)
    return pipeline
