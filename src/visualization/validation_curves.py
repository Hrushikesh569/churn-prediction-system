"""Validation-curve visualisation.

Plots the effect of varying a single hyperparameter on train and
validation ROC-AUC to identify the optimal complexity trade-off.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, validation_curve
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def plot_validation_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    output_dir: str = "outputs/plots",
) -> List[str]:
    """Generate validation-curve plots for ``max_depth`` and ``n_estimators``.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Configuration with ``validation_curves`` ranges and ``cv``
            settings.
        output_dir: Directory to save the PNGs.

    Returns:
        List of paths to the saved plots.
    """
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_folds"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_paths: List[str] = []

    # ---- max_depth -------------------------------------------------------
    param_range = config["validation_curves"]["max_depth"]
    base = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    train_scores, val_scores = validation_curve(
        base, X_train, y_train,
        param_name="max_depth",
        param_range=param_range,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    path = _plot_single(
        param_range, train_scores, val_scores,
        "max_depth", "Max Depth", output_dir,
    )
    output_paths.append(path)

    # ---- n_estimators ----------------------------------------------------
    param_range = config["validation_curves"]["n_estimators"]
    base = XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    train_scores, val_scores = validation_curve(
        base, X_train, y_train,
        param_name="n_estimators",
        param_range=param_range,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    path = _plot_single(
        param_range, train_scores, val_scores,
        "n_estimators", "Number of Estimators", output_dir,
    )
    output_paths.append(path)

    return output_paths


# ---- internal helper -----------------------------------------------------

def _plot_single(
    param_range: List[Any],
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    param_name: str,
    param_label: str,
    output_dir: str,
) -> str:
    """Render a single validation-curve plot.

    Args:
        param_range: Hyperparameter values on the x-axis.
        train_scores: 2-D array of train scores.
        val_scores: 2-D array of validation scores.
        param_name: Parameter name (used in filename).
        param_label: Human-readable label (used in title).
        output_dir: Save directory.

    Returns:
        Path to the saved PNG.
    """
    t_mean = train_scores.mean(axis=1)
    t_std = train_scores.std(axis=1)
    v_mean = val_scores.mean(axis=1)
    v_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(param_range, t_mean - t_std, t_mean + t_std, alpha=0.15, color="#1f77b4")
    ax.fill_between(param_range, v_mean - v_std, v_mean + v_std, alpha=0.15, color="#ff7f0e")
    ax.plot(param_range, t_mean, "o-", color="#1f77b4", label="Training Score")
    ax.plot(param_range, v_mean, "o-", color="#ff7f0e", label="Validation Score")

    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_title(f"Validation Curve — {param_label}", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    path = str(Path(output_dir) / f"validation_curves_{param_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved validation curve (%s) to %s", param_name, path)
    return path
