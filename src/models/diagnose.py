"""Overfitting / underfitting diagnostic module.

Compares train, validation, and test metrics to detect model-capacity
problems and emits actionable recommendations.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def check_overfitting(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    threshold: float = 0.05,
) -> str:
    """Check for overfitting by comparing train / val / test accuracy.

    Flags overfitting when train accuracy exceeds validation accuracy
    by more than *threshold* (default 5 %).

    Args:
        train_metrics: Metrics computed on training set.
        val_metrics: Metrics computed on validation set.
        test_metrics: Metrics computed on test set.
        threshold: Maximum acceptable train–val accuracy gap.

    Returns:
        Human-readable diagnostic report string.
    """
    lines = [
        "=" * 60,
        "OVERFITTING DIAGNOSTIC REPORT",
        "=" * 60,
    ]

    train_acc = train_metrics["accuracy"]
    val_acc = val_metrics["accuracy"]
    test_acc = test_metrics["accuracy"]
    gap_val = train_acc - val_acc
    gap_test = train_acc - test_acc

    lines.append(f"Train Accuracy:      {train_acc:.4f}")
    lines.append(f"Validation Accuracy: {val_acc:.4f}")
    lines.append(f"Test Accuracy:       {test_acc:.4f}")
    lines.append(f"Train-Val Gap:       {gap_val:.4f}")
    lines.append(f"Train-Test Gap:      {gap_test:.4f}")

    if gap_val > threshold:
        lines += [
            "",
            "⚠️  OVERFITTING DETECTED!",
            f"   Train accuracy exceeds validation by {gap_val * 100:.1f}%",
            "   Suggested fixes:",
            "   1. Increase regularisation (reg_alpha, reg_lambda, lower C)",
            "   2. Reduce model complexity (lower max_depth, fewer estimators)",
            "   3. Add more training data or apply data augmentation",
            "   4. Reduce the number of features (feature selection)",
        ]
    else:
        lines += [
            "",
            "✅ No significant overfitting detected.",
        ]

    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report


def check_underfitting(
    metrics: Dict[str, float],
    threshold: float = 0.65,
) -> str:
    """Check for underfitting by examining overall performance.

    Flags underfitting when accuracy falls below *threshold* (default 65 %).

    Args:
        metrics: Metrics dict (typically from the validation set).
        threshold: Minimum acceptable accuracy.

    Returns:
        Human-readable diagnostic report string.
    """
    lines = [
        "=" * 60,
        "UNDERFITTING DIAGNOSTIC REPORT",
        "=" * 60,
    ]

    accuracy = metrics["accuracy"]
    roc_auc = metrics["roc_auc"]

    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append(f"ROC-AUC:  {roc_auc:.4f}")

    if accuracy < threshold:
        lines += [
            "",
            "⚠️  UNDERFITTING DETECTED!",
            f"   Accuracy ({accuracy:.4f}) is below threshold ({threshold})",
            "   Suggested fixes:",
            "   1. Add more features or polynomial / interaction terms",
            "   2. Increase model complexity (higher max_depth, more estimators)",
            "   3. Reduce regularisation (higher C, lower reg_alpha / reg_lambda)",
            "   4. Use a more powerful model (ensemble / VotingClassifier)",
        ]
    else:
        lines += [
            "",
            "✅ No underfitting detected.",
        ]

    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report
