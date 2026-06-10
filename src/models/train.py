"""Model training module.

Trains Logistic Regression, Random Forest, XGBoost, and a Voting
Classifier.  Supports class-weight balancing and SMOTE resampling.
Includes cross-validation helpers.

MLflow autolog is enabled at import time so that every ``model.fit()``
call within an active MLflow run is automatically instrumented.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---- MLflow autolog (requested by spec) ----------------------------
mlflow.sklearn.autolog(log_models=False, silent=True)
mlflow.xgboost.autolog(log_models=False, silent=True)


# =====================================================================
# Logistic Regression
# =====================================================================

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    class_weight: Optional[str] = "balanced",
) -> LogisticRegression:
    """Train Logistic Regression, selecting the best C via CV.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Configuration dict (``models.logistic_regression`` section).
        class_weight: ``'balanced'`` or ``None`` (for SMOTE data).

    Returns:
        Fitted ``LogisticRegression`` with the best regularisation C.
    """
    lr_cfg = config["models"]["logistic_regression"]
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_folds"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    best_c: float = 1.0
    best_score: float = 0.0

    # Disable autolog during internal CV grid search
    mlflow.sklearn.autolog(disable=True)

    for c_val in lr_cfg["C_values"]:
        model = LogisticRegression(
            C=c_val,
            class_weight=class_weight,
            max_iter=lr_cfg["max_iter"],
            random_state=lr_cfg["random_state"],
            solver="lbfgs",
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_score = float(scores.mean())
        logger.info(
            "LR  C=%-5s  CV AUC = %.4f (±%.4f)", c_val, mean_score, scores.std()
        )
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val

    # Re-enable autolog
    mlflow.sklearn.autolog(log_models=False, silent=True)

    logger.info("Best LR  C=%s  CV AUC=%.4f", best_c, best_score)

    model = LogisticRegression(
        C=best_c,
        class_weight=class_weight,
        max_iter=lr_cfg["max_iter"],
        random_state=lr_cfg["random_state"],
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


# =====================================================================
# Random Forest
# =====================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    class_weight: Optional[str] = "balanced",
) -> RandomForestClassifier:
    """Train a Random Forest with controlled complexity.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Configuration dict (``models.random_forest`` section).
        class_weight: ``'balanced'`` or ``None``.

    Returns:
        Fitted ``RandomForestClassifier``.
    """
    rf_cfg = config["models"]["random_forest"]

    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=class_weight,
        random_state=rf_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Trained Random Forest (n_estimators=%d)", rf_cfg["n_estimators"])
    return model


# =====================================================================
# XGBoost
# =====================================================================

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost with early stopping and per-round logging.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Configuration dict (``models.xgboost`` section).
        params: Optional override parameters (e.g. from Optuna tuning).

    Returns:
        Tuple of ``(fitted_model, evals_result)`` where *evals_result*
        contains per-round train/val logloss and AUC.
    """
    xgb_cfg = config["models"]["xgboost"]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model_params: Dict[str, Any] = {
        "n_estimators": xgb_cfg["n_estimators"],
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "subsample": xgb_cfg["subsample"],
        "reg_alpha": xgb_cfg["reg_alpha"],
        "reg_lambda": xgb_cfg["reg_lambda"],
        "scale_pos_weight": scale_pos_weight,
        "random_state": xgb_cfg["random_state"],
        "eval_metric": ["logloss", "auc"],
        "early_stopping_rounds": xgb_cfg["early_stopping_rounds"],
        "verbosity": 0,
    }

    if params:
        model_params.update(params)

    model = XGBClassifier(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    evals_result: Dict[str, Any] = model.evals_result()
    best_iter = model.best_iteration
    logger.info(
        "XGBoost trained: best_iteration=%d  val_auc=%.4f  val_logloss=%.4f",
        best_iter,
        evals_result["validation_1"]["auc"][best_iter],
        evals_result["validation_1"]["logloss"][best_iter],
    )

    return model, evals_result


# =====================================================================
# Voting Classifier
# =====================================================================

def train_voting_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    xgb_best_params: Optional[Dict[str, Any]] = None,
    xgb_best_iteration: Optional[int] = None,
) -> VotingClassifier:
    """Train a soft-voting ensemble of LR + RF + XGBoost.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Full configuration dict.
        xgb_best_params: Tuned XGBoost hyperparameters.
        xgb_best_iteration: Optimal boosting round count from early stopping.

    Returns:
        Fitted ``VotingClassifier``.
    """
    lr_cfg = config["models"]["logistic_regression"]
    rf_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=lr_cfg["max_iter"],
        random_state=lr_cfg["random_state"],
        solver="lbfgs",
    )

    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight="balanced",
        random_state=rf_cfg["random_state"],
        n_jobs=-1,
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    xgb_params: Dict[str, Any] = {
        "n_estimators": xgb_best_iteration or xgb_cfg["n_estimators"],
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "subsample": xgb_cfg["subsample"],
        "reg_alpha": xgb_cfg["reg_alpha"],
        "reg_lambda": xgb_cfg["reg_lambda"],
        "scale_pos_weight": n_neg / max(n_pos, 1),
        "random_state": xgb_cfg["random_state"],
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if xgb_best_params:
        xgb_params.update(xgb_best_params)
        if xgb_best_iteration:
            xgb_params["n_estimators"] = xgb_best_iteration

    xgb = XGBClassifier(**xgb_params)

    # Disable autolog during VotingClassifier fit (avoids nested logging)
    mlflow.sklearn.autolog(disable=True)
    mlflow.xgboost.autolog(disable=True)

    voting = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft",
    )
    voting.fit(X_train, y_train)

    # Re-enable autolog
    mlflow.sklearn.autolog(log_models=False, silent=True)
    mlflow.xgboost.autolog(log_models=False, silent=True)

    logger.info("Trained VotingClassifier (LR + RF + XGB, soft voting)")
    return voting


# =====================================================================
# SMOTE helper
# =====================================================================

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to the training data only.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(X_resampled, y_resampled)``.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    counts = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info(
        "SMOTE: %d → %d samples.  Class distribution: %s",
        len(y_train), len(y_res), counts,
    )
    return X_res, y_res


# =====================================================================
# Cross-validation helper
# =====================================================================

def cross_validate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    model_name: str = "Model",
) -> np.ndarray:
    """Run stratified K-fold cross-validation and return AUC scores.

    For ``XGBClassifier`` models with ``early_stopping_rounds`` set,
    the parameter is removed from the clone (sklearn CV does not pass
    ``eval_set``).

    Args:
        model: Any sklearn-compatible estimator.
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Configuration dict with ``cv`` section.
        model_name: Label used for logging.

    Returns:
        Array of per-fold ROC-AUC scores.
    """
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_folds"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    cv_model = clone(model)
    # XGBClassifier with early_stopping_rounds needs eval_set — remove it
    if hasattr(cv_model, "early_stopping_rounds") and cv_model.early_stopping_rounds is not None:
        cv_model.set_params(early_stopping_rounds=None)

    # Disable autolog during CV
    mlflow.sklearn.autolog(disable=True)
    mlflow.xgboost.autolog(disable=True)

    scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc")

    # Re-enable autolog
    mlflow.sklearn.autolog(log_models=False, silent=True)
    mlflow.xgboost.autolog(log_models=False, silent=True)

    logger.info(
        "%s  5-fold CV AUC: %.4f (±%.4f)  [%s]",
        model_name,
        scores.mean(),
        scores.std(),
        ", ".join(f"{s:.4f}" for s in scores),
    )
    return scores
