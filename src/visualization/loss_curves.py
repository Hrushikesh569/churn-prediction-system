"""XGBoost loss-curve visualisation.

Plots per-boosting-round train and validation log-loss and AUC
from the ``evals_result`` dict to diagnose overfitting in XGBoost.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_loss_curves(
    evals_result: Dict[str, Any],
    output_dir: str = "outputs/plots",
) -> str:
    """Generate and save XGBoost epoch-level loss & AUC curves.

    Expects ``evals_result`` to contain keys ``'validation_0'`` (train)
    and ``'validation_1'`` (val), each with ``'logloss'`` and ``'auc'``.

    Args:
        evals_result: Return value of ``model.evals_result()``.
        output_dir: Directory to save the PNG.

    Returns:
        Absolute path to the saved plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Log Loss --------------------------------------------------------
    train_ll = evals_result["validation_0"]["logloss"]
    val_ll = evals_result["validation_1"]["logloss"]
    epochs = range(1, len(train_ll) + 1)

    axes[0].plot(epochs, train_ll, label="Train", color="#1f77b4", alpha=0.85)
    axes[0].plot(epochs, val_ll, label="Validation", color="#ff7f0e", alpha=0.85)
    axes[0].set_xlabel("Boosting Round (Epoch)", fontsize=12)
    axes[0].set_ylabel("Log Loss", fontsize=12)
    axes[0].set_title("XGBoost — Loss Curves", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ---- AUC -------------------------------------------------------------
    train_auc = evals_result["validation_0"]["auc"]
    val_auc = evals_result["validation_1"]["auc"]

    axes[1].plot(epochs, train_auc, label="Train", color="#1f77b4", alpha=0.85)
    axes[1].plot(epochs, val_auc, label="Validation", color="#ff7f0e", alpha=0.85)
    axes[1].set_xlabel("Boosting Round (Epoch)", fontsize=12)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_title("XGBoost — AUC Curves", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "loss_curves.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved loss curves to %s", output_path)
    return output_path
