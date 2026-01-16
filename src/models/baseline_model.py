"""Baseline model using Logistic Regression for churn prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.utils.logging import logger
from src.utils.metrics import calculate_classification_metrics


def create_logistic_regression(
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Create a configured Logistic Regression classifier.
    
    Args:
        C: Inverse of regularization strength
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        solver: Algorithm for optimization
        class_weight: Class weighting strategy
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        
    Returns:
        Configured LogisticRegression model
    """
    # Adjust solver for L1 penalty
    if penalty == "l1" and solver == "lbfgs":
        solver = "saga"
    elif penalty == "elasticnet":
        solver = "saga"
        
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    log_to_mlflow: bool = True,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Train the baseline Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: Model parameters
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    params = params or {}
    
    logger.info("Training baseline Logistic Regression model...")
    
    model = create_logistic_regression(**params)
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
    
    metrics = {"train_" + k: v for k, v in train_metrics.items()}
    
    # Calculate validation metrics if provided
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = calculate_classification_metrics(y_val, y_val_pred, y_val_proba)
        metrics.update({"val_" + k: v for k, v in val_metrics.items()})
        
    logger.info(f"Baseline model - Train ROC-AUC: {metrics['train_roc_auc']:.4f}")
    if "val_roc_auc" in metrics:
        logger.info(f"Baseline model - Val ROC-AUC: {metrics['val_roc_auc']:.4f}")
        
    # Log to MLflow
    if log_to_mlflow:
        _log_baseline_to_mlflow(model, params, metrics)
        
    return model, metrics


def cross_validate_baseline(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Cross-validate the baseline model with stratified k-fold.
    
    Args:
        X: Features
        y: Labels
        params: Model parameters
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (model trained on full data, CV results dict)
    """
    params = params or {}
    
    logger.info(f"Running {n_splits}-fold stratified cross-validation for baseline model...")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics: List[Dict[str, float]] = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        model = create_logistic_regression(**params)
        model.fit(X_fold_train, y_fold_train)
        
        y_pred = model.predict(X_fold_val)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        
        metrics = calculate_classification_metrics(y_fold_val, y_pred, y_proba)
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold + 1} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
    # Aggregate metrics
    cv_results = _aggregate_cv_metrics(fold_metrics)
    
    # Train final model on all data
    final_model = create_logistic_regression(**params)
    final_model.fit(X, y)
    
    logger.info(f"CV Results - Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f} (+/- {cv_results['std_roc_auc']:.4f})")
    
    return final_model, cv_results


def _aggregate_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate metrics across CV folds.
    
    Args:
        fold_metrics: List of metrics dicts from each fold
        
    Returns:
        Aggregated metrics with mean and std
    """
    aggregated = {}
    metric_names = fold_metrics[0].keys()
    
    for metric in metric_names:
        values = [m[metric] for m in fold_metrics]
        aggregated[f"mean_{metric}"] = np.mean(values)
        aggregated[f"std_{metric}"] = np.std(values)
        aggregated[f"all_{metric}"] = values
        
    return aggregated


def _log_baseline_to_mlflow(
    model: LogisticRegression,
    params: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    """Log baseline model to MLflow.
    
    Args:
        model: Trained model
        params: Model parameters
        metrics: Calculated metrics
    """
    try:
        mlflow.log_params({f"baseline_{k}": v for k, v in params.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in metrics.items() 
                          if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(model, "baseline_model")
        logger.info("Logged baseline model to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def save_baseline_model(model: LogisticRegression, path: Path) -> None:
    """Save the trained baseline model.
    
    Args:
        model: Trained model
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved baseline model to {path}")


def load_baseline_model(path: Path) -> LogisticRegression:
    """Load a saved baseline model.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(path)
