"""Data loading module for the churn prediction system.

Downloads the Telco Customer Churn dataset from GitHub, caches it locally,
and validates the schema against expected columns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: List[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset.

    Tries the local cache first, then downloads from the primary URL
    (with fallback).  The downloaded file is cached for subsequent runs.

    Args:
        config: Configuration dictionary.  Must contain::

            data:
              urls:
                primary: <url>
                fallback: <url>   # optional
              cache_path: <relative path>

    Returns:
        Raw ``DataFrame`` with all original columns.

    Raises:
        ValueError: If the downloaded data does not match the expected schema.
        RuntimeError: If the data cannot be fetched from any URL.
    """
    cache_path = Path(config["data"]["cache_path"])

    # ---- Try loading from local cache --------------------------------
    if cache_path.exists():
        logger.info("Loading cached dataset from %s", cache_path)
        df = pd.read_csv(cache_path)
        _validate_schema(df)
        logger.info("Loaded %d rows from cache", len(df))
        return df

    # ---- Download from remote URLs -----------------------------------
    urls: List[str] = [config["data"]["urls"]["primary"]]
    fallback = config["data"]["urls"].get("fallback")
    if fallback:
        urls.append(fallback)

    for url in urls:
        try:
            logger.info("Downloading dataset from %s", url)
            df = pd.read_csv(url)
            _validate_schema(df)

            # Cache locally
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info("Downloaded and cached %d rows to %s", len(df), cache_path)
            return df
        except Exception as exc:
            logger.warning("Failed to download from %s: %s", url, exc)
            continue

    raise RuntimeError(
        "Failed to download dataset from all URLs. "
        "Please download manually and place at: " + str(cache_path)
    )


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame contains the expected columns.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.debug("Schema validation passed — %d columns present", len(df.columns))
