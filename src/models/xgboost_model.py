"""XGBoost model for churn prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from src.utils.logging import logger
from src.utils.metrics import calculate_classification_metrics


def create_xgboost_classifier(
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    eval_metric: str = "auc",
    early_stopping_rounds: Optional[int] = None,
    **kwargs
) -> xgb.XGBClassifier:
    """Create a configured XGBoost classifier.
    
    Args:
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        n_estimators: Number of boosting rounds
        min_child_weight: Minimum sum of instance weight in a child
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        gamma: Minimum loss reduction required for split
        reg_alpha: L1 regularization term
        reg_lambda: L2 regularization term
        scale_pos_weight: Balance of positive and negative weights
        random_state: Random seed
        eval_metric: Evaluation metric for early stopping
        early_stopping_rounds: Rounds for early stopping (None to disable)
        
    Returns:
        Configured XGBClassifier
    """
    return xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        **kwargs
    )



def calculate_scale_pos_weight(y: np.ndarray) -> float:
    """Calculate scale_pos_weight for class imbalance.
    
    Args:
        y: Target labels
        
    Returns:
        Calculated scale_pos_weight value
    """
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    return neg_count / pos_count if pos_count > 0 else 1.0


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    auto_scale_pos_weight: bool = True,
    early_stopping: bool = True,
    log_to_mlflow: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """Train the XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional, required for early stopping)
        y_val: Validation labels (optional, required for early stopping)
        params: Model parameters
        auto_scale_pos_weight: Automatically calculate class weight
        early_stopping: Whether to use early stopping
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    params = params or {}
    
    logger.info("Training XGBoost model...")
    
    # Calculate class weight if needed
    if auto_scale_pos_weight and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = calculate_scale_pos_weight(y_train)
        logger.info(f"Calculated scale_pos_weight: {params['scale_pos_weight']:.3f}")
        
    # Configure early stopping
    if early_stopping and X_val is not None:
        params["early_stopping_rounds"] = params.get("early_stopping_rounds", 50)
        
    model = create_xgboost_classifier(**params)
    
    # Fit with or without validation set
    if X_val is not None and y_val is not None and early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logger.info(f"Best iteration: {model.best_iteration}")
    else:
        # Disable early stopping if no validation set
        model.set_params(early_stopping_rounds=None)
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
        
    logger.info(f"XGBoost - Train ROC-AUC: {metrics['train_roc_auc']:.4f}")
    if "val_roc_auc" in metrics:
        logger.info(f"XGBoost - Val ROC-AUC: {metrics['val_roc_auc']:.4f}")
        
    # Log to MLflow
    if log_to_mlflow:
        _log_xgboost_to_mlflow(model, params, metrics)
        
    return model, metrics


def cross_validate_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    auto_scale_pos_weight: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Cross-validate XGBoost with stratified k-fold.
    
    Args:
        X: Features
        y: Labels
        params: Model parameters
        n_splits: Number of CV folds
        random_state: Random seed
        auto_scale_pos_weight: Automatically calculate class weight
        
    Returns:
        Tuple of (model trained on full data, CV results dict)
    """
    params = params or {}
    
    logger.info(f"Running {n_splits}-fold stratified cross-validation for XGBoost...")
    
    # Calculate class weight once
    if auto_scale_pos_weight and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = calculate_scale_pos_weight(y)
        
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics: List[Dict[str, float]] = []
    feature_importances: List[np.ndarray] = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        model = create_xgboost_classifier(**params, early_stopping_rounds=50)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_fold_val)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        
        metrics = calculate_classification_metrics(y_fold_val, y_pred, y_proba)
        fold_metrics.append(metrics)
        feature_importances.append(model.feature_importances_)
        
        logger.info(f"Fold {fold + 1} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
    # Aggregate metrics
    cv_results = _aggregate_cv_metrics(fold_metrics)
    cv_results["feature_importances"] = np.mean(feature_importances, axis=0)
    
    # Train final model on all data (no early stopping)
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_model = create_xgboost_classifier(**final_params)
    final_model.fit(X, y)
    
    logger.info(f"CV Results - Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f} (+/- {cv_results['std_roc_auc']:.4f})")
    
    return final_model, cv_results


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Get feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'weight', 'cover')
        
    Returns:
        DataFrame with feature importances sorted descending
    """
    if importance_type == "gain":
        importances = model.feature_importances_
    else:
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)
        importances = np.array([
            importance_dict.get(f"f{i}", 0) 
            for i in range(len(feature_names))
        ])
        
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _aggregate_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate metrics across CV folds."""
    aggregated = {}
    metric_names = fold_metrics[0].keys()
    
    for metric in metric_names:
        values = [m[metric] for m in fold_metrics]
        aggregated[f"mean_{metric}"] = np.mean(values)
        aggregated[f"std_{metric}"] = np.std(values)
        aggregated[f"all_{metric}"] = values
        
    return aggregated


def _log_xgboost_to_mlflow(
    model: xgb.XGBClassifier,
    params: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    """Log XGBoost model to MLflow."""
    try:
        mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
        mlflow.log_metrics({f"xgb_{k}": v for k, v in metrics.items()
                          if isinstance(v, (int, float))})
        mlflow.xgboost.log_model(model, "xgboost_model")
        logger.info("Logged XGBoost model to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def save_xgboost_model(model: xgb.XGBClassifier, path: Path) -> None:
    """Save the trained XGBoost model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved XGBoost model to {path}")


def load_xgboost_model(path: Path) -> xgb.XGBClassifier:
    """Load a saved XGBoost model."""
    return joblib.load(path)
