"""Hyperparameter tuning using Optuna for model optimization."""

from typing import Any, Callable, Dict, Optional, Tuple

import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from src.models.baseline_model import create_logistic_regression
from src.models.xgboost_model import calculate_scale_pos_weight, create_xgboost_classifier
from src.utils.logging import logger
from src.utils.metrics import calculate_classification_metrics


def create_baseline_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for Logistic Regression.
    
    Args:
        X: Features
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        # Define search space
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        
        # Create model
        params = {
            "C": C,
            "penalty": penalty,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = create_logistic_regression(**params)
            model.fit(X_train, y_train)
            
            y_proba = model.predict_proba(X_val)[:, 1]
            metrics = calculate_classification_metrics(
                y_val, model.predict(X_val), y_proba
            )
            scores.append(metrics["roc_auc"])
            
        return np.mean(scores)
    
    return objective


def create_xgboost_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for XGBoost.
    
    Args:
        X: Features
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        Objective function for Optuna
    """
    scale_pos_weight = calculate_scale_pos_weight(y)
    
    def objective(trial: optuna.Trial) -> float:
        # Define search space
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = create_xgboost_classifier(**params, early_stopping_rounds=30)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_proba = model.predict_proba(X_val)[:, 1]
            metrics = calculate_classification_metrics(
                y_val, model.predict(X_val), y_proba
            )
            scores.append(metrics["roc_auc"])
            
        return np.mean(scores)
    
    return objective


def tune_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    log_to_mlflow: bool = True,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Tune Logistic Regression hyperparameters.
    
    Args:
        X: Features
        y: Labels
        n_trials: Number of Optuna trials
        n_splits: Number of CV folds
        random_state: Random seed
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Tuple of (best parameters, Optuna study)
    """
    logger.info(f"Tuning baseline model with {n_trials} trials...")
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    objective = create_baseline_objective(X, y, n_splits, random_state)
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = random_state
    
    logger.info(f"Best baseline ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best baseline params: {best_params}")
    
    if log_to_mlflow:
        try:
            mlflow.log_params({f"tune_baseline_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("tune_baseline_best_roc_auc", study.best_value)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    return best_params, study


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    log_to_mlflow: bool = True,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Tune XGBoost hyperparameters.
    
    Args:
        X: Features
        y: Labels
        n_trials: Number of Optuna trials
        n_splits: Number of CV folds
        random_state: Random seed
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Tuple of (best parameters, Optuna study)
    """
    logger.info(f"Tuning XGBoost model with {n_trials} trials...")
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    objective = create_xgboost_objective(X, y, n_splits, random_state)
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params["scale_pos_weight"] = calculate_scale_pos_weight(y)
    best_params["random_state"] = random_state
    
    logger.info(f"Best XGBoost ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best XGBoost params: {best_params}")
    
    if log_to_mlflow:
        try:
            mlflow.log_params({f"tune_xgb_{k}": v for k, v in best_params.items() 
                             if not isinstance(v, (list, dict))})
            mlflow.log_metric("tune_xgb_best_roc_auc", study.best_value)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    return best_params, study


def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    n_trials: int = 10,
) -> Dict[str, Any]:
    """Quick hyperparameter tuning for testing.
    
    Args:
        X: Features
        y: Labels
        model_type: 'baseline' or 'xgboost'
        n_trials: Number of trials (default 10 for quick testing)
        
    Returns:
        Best parameters
    """
    if model_type == "baseline":
        params, _ = tune_baseline(X, y, n_trials=n_trials, log_to_mlflow=False)
    else:
        params, _ = tune_xgboost(X, y, n_trials=n_trials, log_to_mlflow=False)
    return params
