"""Model evaluation with comprehensive metrics and bootstrap CIs.

Computes ROC-AUC, Accuracy, Precision, Recall, F1, Specificity, and
Log Loss for any sklearn-compatible classifier.  Also provides
bootstrap 95 % confidence intervals and a multi-model comparison CSV.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "test",
) -> Dict[str, float]:
    """Compute all evaluation metrics for a fitted model.

    Args:
        model: Fitted estimator with ``predict`` and ``predict_proba``.
        X: Feature matrix.
        y: True binary labels.
        dataset_name: Label used for logging (e.g. ``"train"``, ``"test"``).

    Returns:
        Dictionary mapping metric name → value.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "log_loss": float(log_loss(y, y_proba)),
    }

    logger.info("--- %s metrics ---", dataset_name)
    for name, value in metrics.items():
        logger.info("  %-12s  %.4f", name, value)

    return metrics


def bootstrap_confidence_intervals(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 95 % confidence intervals via bootstrap resampling.

    Args:
        model: Fitted estimator.
        X: Feature matrix (typically the test set).
        y: True labels.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Seed for reproducibility.

    Returns:
        Nested dict: ``metric_name → {'mean', 'lower', 'upper'}``.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y)

    collectors: Dict[str, list] = {
        "roc_auc": [], "accuracy": [], "precision": [],
        "recall": [], "f1": [], "specificity": [], "log_loss": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        X_b, y_b = X[idx], y[idx]

        # Need both classes in the sample
        if len(np.unique(y_b)) < 2:
            continue

        try:
            y_pred = model.predict(X_b)
            y_proba = model.predict_proba(X_b)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_b, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            collectors["roc_auc"].append(roc_auc_score(y_b, y_proba))
            collectors["accuracy"].append(accuracy_score(y_b, y_pred))
            collectors["precision"].append(precision_score(y_b, y_pred, zero_division=0))
            collectors["recall"].append(recall_score(y_b, y_pred, zero_division=0))
            collectors["f1"].append(f1_score(y_b, y_pred, zero_division=0))
            collectors["specificity"].append(spec)
            collectors["log_loss"].append(log_loss(y_b, y_proba))
        except Exception:
            continue

    ci: Dict[str, Dict[str, float]] = {}
    for metric_name, values in collectors.items():
        arr = np.array(values)
        ci[metric_name] = {
            "mean": float(arr.mean()),
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
        }

    logger.info("95%% Confidence Intervals (%d bootstrap iterations):", n_bootstrap)
    for name, vals in ci.items():
        logger.info(
            "  %-12s  %.4f  [%.4f, %.4f]", name, vals["mean"], vals["lower"], vals["upper"]
        )

    return ci


def generate_comparison_report(
    results: Dict[str, Dict[str, float]],
    output_path: str,
) -> pd.DataFrame:
    """Generate a CSV comparison table of all models' test metrics.

    Args:
        results: ``{model_name: {metric: value, ...}, ...}``
        output_path: Destination CSV path.

    Returns:
        Comparison ``DataFrame`` (models as rows, metrics as columns).
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df.round(4)
    df.to_csv(output_path)
    logger.info("Saved metrics comparison report to %s", output_path)
    return df
