"""MLflow experiment tracking with DagsHub integration."""

import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import logger


# DagsHub/MLflow configuration
DAGSHUB_REPO_OWNER = "aadilsal"
DAGSHUB_REPO_NAME = "Churn-Predictor"
DAGSHUB_MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

# Experiment naming conventions
EXPERIMENT_NAMES = {
    "training": "churn-model-training",
    "tuning": "hyperparameter-tuning",
    "evaluation": "model-evaluation",
    "survival": "survival-analysis",
}


def init_tracking(
    experiment_name: Optional[str] = None,
    use_dagshub: bool = True,
) -> str:
    """Initialize MLflow tracking with DagsHub.
    
    Args:
        experiment_name: Name of experiment (uses default if None)
        use_dagshub: Whether to use DagsHub as tracking server
        
    Returns:
        Tracking URI being used
    """
    if use_dagshub:
        try:
            import dagshub
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            tracking_uri = DAGSHUB_MLFLOW_URI
            logger.info(f"Initialized DagsHub MLflow tracking: {tracking_uri}")
        except ImportError:
            logger.warning("dagshub not installed, falling back to local tracking")
            tracking_uri = "file:./mlruns"
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as e:
            logger.warning(f"DagsHub init failed: {e}, falling back to local")
            tracking_uri = "file:./mlruns"
            mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Using local MLflow tracking: {tracking_uri}")
        
    # Set experiment
    experiment_name = experiment_name or EXPERIMENT_NAMES["training"]
    mlflow.set_experiment(experiment_name)
    logger.info(f"Set experiment: {experiment_name}")
    
    return tracking_uri


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> mlflow.ActiveRun:
    """Start a new MLflow run.
    
    Args:
        run_name: Name for the run (auto-generated if None)
        tags: Additional tags for the run
        nested: Whether this is a nested run
        
    Returns:
        Active MLflow run context
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    default_tags = {
        "version": "1.0.0",
        "framework": "sklearn+xgboost",
        "stage": "development",
    }
    
    if tags:
        default_tags.update(tags)
        
    return mlflow.start_run(run_name=run_name, tags=default_tags, nested=nested)


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to current run.
    
    Args:
        params: Dictionary of parameters
    """
    # Flatten nested dicts and convert to strings
    flat_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_params[f"{key}_{sub_key}"] = str(sub_value)
        else:
            flat_params[key] = str(value) if not isinstance(value, (int, float, str, bool)) else value
            
    mlflow.log_params(flat_params)
    logger.debug(f"Logged {len(flat_params)} parameters")


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log metrics to current run.
    
    Args:
        metrics: Dictionary of metrics
        step: Step number for time series metrics
        prefix: Prefix to add to metric names
    """
    prefixed_metrics = {
        f"{prefix}{k}" if prefix else k: v 
        for k, v in metrics.items() 
        if isinstance(v, (int, float))
    }
    
    mlflow.log_metrics(prefixed_metrics, step=step)
    logger.debug(f"Logged {len(prefixed_metrics)} metrics")


def log_artifact(
    local_path: Union[str, Path],
    artifact_path: Optional[str] = None,
) -> None:
    """Log an artifact file.
    
    Args:
        local_path: Path to local file
        artifact_path: Subdirectory in artifact storage
    """
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    logger.debug(f"Logged artifact: {local_path}")


def log_artifacts(
    local_dir: Union[str, Path],
    artifact_path: Optional[str] = None,
) -> None:
    """Log all files in a directory.
    
    Args:
        local_dir: Path to local directory
        artifact_path: Subdirectory in artifact storage
    """
    mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
    logger.debug(f"Logged artifacts from: {local_dir}")


def log_model(
    model: Any,
    artifact_path: str,
    model_type: str = "sklearn",
    **kwargs,
) -> None:
    """Log a model to MLflow.
    
    Args:
        model: Model object
        artifact_path: Path in artifacts
        model_type: Type of model ('sklearn', 'xgboost', etc.)
        **kwargs: Additional arguments for model logging
    """
    if model_type == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path, **kwargs)
    else:
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        
    logger.info(f"Logged {model_type} model to {artifact_path}")


def log_feature_info(
    feature_names: List[str],
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
) -> None:
    """Log feature information as parameters.
    
    Args:
        feature_names: List of all feature names
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
    """
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_param("feature_names", ",".join(feature_names[:20]))  # Truncate for display
    
    if categorical_features:
        mlflow.log_param("n_categorical", len(categorical_features))
    if numerical_features:
        mlflow.log_param("n_numerical", len(numerical_features))


def log_data_info(
    n_samples: int,
    n_train: int,
    n_val: int,
    n_test: int,
    class_balance: float,
) -> None:
    """Log dataset information.
    
    Args:
        n_samples: Total samples
        n_train: Training samples
        n_val: Validation samples
        n_test: Test samples
        class_balance: Proportion of positive class
    """
    mlflow.log_params({
        "n_samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "class_balance": class_balance,
    })


def get_run_info() -> Dict[str, Any]:
    """Get current run information.
    
    Returns:
        Dictionary with run info
    """
    run = mlflow.active_run()
    if run is None:
        return {"active": False}
        
    return {
        "active": True,
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "artifact_uri": run.info.artifact_uri,
    }


def track_experiment(
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    use_dagshub: bool = True,
) -> Callable:
    """Decorator to track a function as an MLflow experiment.
    
    Args:
        experiment_name: Experiment name
        run_name: Run name
        use_dagshub: Use DagsHub tracking
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            init_tracking(experiment_name, use_dagshub)
            
            with start_run(run_name=run_name):
                result = func(*args, **kwargs)
                
            return result
        return wrapper
    return decorator


def get_best_run(
    experiment_name: str,
    metric: str = "val_roc_auc",
    ascending: bool = False,
) -> Optional[Dict[str, Any]]:
    """Get the best run from an experiment.
    
    Args:
        experiment_name: Name of experiment
        metric: Metric to optimize
        ascending: Whether lower is better
        
    Returns:
        Best run info or None
    """
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    
    if not runs:
        return None
        
    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifact_uri": best_run.info.artifact_uri,
    }
