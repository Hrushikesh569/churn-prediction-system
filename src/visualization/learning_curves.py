"""Learning curve visualisation.

Plots training-set size vs model score to diagnose overfitting
(diverging curves) or underfitting (low, converging curves).
"""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve

logger = logging.getLogger(__name__)


def plot_learning_curves(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    model_name: str = "Model",
    output_dir: str = "outputs/plots",
) -> str:
    """Generate and save a learning-curve plot.

    Args:
        model: Un-fitted sklearn-compatible estimator (will be cloned
            internally by ``learning_curve``).
        X_train: Training features.
        y_train: Training labels.
        config: Configuration (``cv`` section used for fold count).
        model_name: Title label for the plot.
        output_dir: Directory to save the PNG.

    Returns:
        Absolute path to the saved plot.
    """
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_folds"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=config["split"]["random_state"],
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color="#1f77b4",
    )
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std,
        alpha=0.15, color="#ff7f0e",
    )
    ax.plot(train_sizes, train_mean, "o-", color="#1f77b4", label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="#ff7f0e", label="Validation Score")

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_title(f"Learning Curve — {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "learning_curves.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved learning curves to %s", output_path)
    return output_path
