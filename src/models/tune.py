"""Hyperparameter tuning with Optuna.

Uses Optuna with median pruning to search the XGBoost hyperparameter
space over 50 trials, optimising validation ROC-AUC.

MLflow autolog is enabled at import time.
"""

import logging
from typing import Any, Dict

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---- MLflow autolog (requested by spec) ----------------------------
mlflow.xgboost.autolog(log_models=False, silent=True)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna.

    Runs *n_trials* trials with early stopping and median-based pruning.
    Each trial trains an XGBClassifier and evaluates ROC-AUC on the
    validation set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix (for objective evaluation).
        y_val: Validation labels.
        config: Configuration dict with ``tuning`` and ``models.xgboost``
            sections.

    Returns:
        Dictionary of the best hyperparameters found.
    """
    search_space = config["tuning"]["search_space"]
    n_trials: int = config["tuning"]["n_trials"]
    xgb_cfg = config["models"]["xgboost"]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Disable autolog during tuning (avoids 50+ noisy runs)
    mlflow.xgboost.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximise validation ROC-AUC."""
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", search_space["n_estimators"]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", search_space["max_depth"]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", search_space["learning_rate"]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", search_space["subsample"]
            ),
            "reg_alpha": trial.suggest_categorical(
                "reg_alpha", search_space["reg_alpha"]
            ),
            "reg_lambda": trial.suggest_categorical(
                "reg_lambda", search_space["reg_lambda"]
            ),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=xgb_cfg["random_state"],
            eval_metric="logloss",
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            verbosity=0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        return auc

    # --- Run optimisation ------------------------------------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params: Dict[str, Any] = study.best_params
    best_value: float = study.best_value

    logger.info("Optuna best trial:  AUC = %.4f", best_value)
    logger.info("Best params: %s", best_params)

    # Re-enable autolog
    mlflow.xgboost.autolog(log_models=False, silent=True)
    mlflow.sklearn.autolog(log_models=False, silent=True)

    return best_params
