"""End-to-end training pipeline for churn prediction models."""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import clean_telco_dataset, split_features_target
from src.models.baseline_model import cross_validate_baseline, train_baseline_model
from src.models.evaluation import (
    create_evaluation_plots,
    evaluate_model,
    find_optimal_threshold,
    save_evaluation_results,
)
from src.models.feature_engineering import (
    FeatureEngineer,
    create_train_test_split,
    prepare_features,
)
from src.models.hyperparameter_tuning import tune_baseline, tune_xgboost
from src.models.model_comparison import (
    compare_models,
    create_performance_report,
    select_best_model,
)
from src.models.xgboost_model import cross_validate_xgboost, train_xgboost_model, get_feature_importance
from src.utils.logging import logger


def setup_mlflow(experiment_name: str = "churn_prediction") -> None:
    """Setup MLflow tracking.
    
    Args:
        experiment_name: Name of MLflow experiment
    """
    # Use local file storage as fallback
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {e}. Using local storage.")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)


def load_and_prepare_data(
    data_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load and prepare data for training.
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        Train, validation, and test splits
    """
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create stratified splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
        X, y, test_size=0.2, val_size=0.2
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_tuning_trials: int = 50,
    quick_test: bool = False,
    skip_tuning: bool = False,
) -> Dict[str, Any]:
    """Run the complete training pipeline.
    
    Args:
        data_path: Path to processed data
        output_dir: Directory for output artifacts
        n_tuning_trials: Number of hyperparameter tuning trials
        quick_test: Run with minimal settings for testing
        skip_tuning: Skip hyperparameter tuning (use defaults)
        
    Returns:
        Dictionary with pipeline results
    """
    # Set defaults
    data_path = data_path or Path("data/processed/telco_churn_processed.csv")
    output_dir = output_dir or Path("models")
    reports_dir = Path("reports")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if quick_test:
        n_tuning_trials = 5
        logger.info("Running in quick test mode")
        
    # Setup MLflow
    setup_mlflow()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "cv_results": {},
        "evaluation": {},
    }
    
    with mlflow.start_run(run_name=f"churn_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(data_path)
        
        # Feature engineering
        logger.info("Running feature engineering...")
        feature_engineer = FeatureEngineer()
        
        X_train_transformed = feature_engineer.fit_transform(X_train)
        X_val_transformed = feature_engineer.transform(X_val)
        X_test_transformed = feature_engineer.transform(X_test)
        
        feature_names = feature_engineer.get_feature_names()
        logger.info(f"Transformed to {len(feature_names)} features")
        
        # Save feature engineering artifacts
        feature_engineer.save(output_dir)
        
        # Combine train and val for CV
        X_trainval = np.vstack([X_train_transformed, X_val_transformed])
        y_trainval = np.concatenate([y_train.values, y_val.values])
        
        # ============================================
        # BASELINE MODEL
        # ============================================
        logger.info("=" * 50)
        logger.info("Training Baseline Model (Logistic Regression)")
        logger.info("=" * 50)
        
        if not skip_tuning:
            baseline_params, _ = tune_baseline(
                X_trainval, y_trainval,
                n_trials=n_tuning_trials,
            )
        else:
            baseline_params = {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "random_state": 42}
            
        # Cross-validation
        baseline_model, baseline_cv = cross_validate_baseline(
            X_trainval, y_trainval, params=baseline_params
        )
        results["cv_results"]["baseline"] = baseline_cv
        
        # Train final baseline on train set, evaluate on validation
        baseline_model, baseline_train_metrics = train_baseline_model(
            X_train_transformed, y_train.values,
            X_val_transformed, y_val.values,
            params=baseline_params,
        )
        
        # Evaluate on test set
        y_test_pred_baseline = baseline_model.predict(X_test_transformed)
        y_test_proba_baseline = baseline_model.predict_proba(X_test_transformed)[:, 1]
        baseline_test_metrics = evaluate_model(
            y_test.values, y_test_pred_baseline, y_test_proba_baseline, "Baseline"
        )
        results["evaluation"]["baseline"] = baseline_test_metrics
        
        # Save baseline plots
        create_evaluation_plots(
            y_test.values, y_test_pred_baseline, y_test_proba_baseline,
            "Baseline (Logistic Regression)",
            save_path=reports_dir / "baseline_evaluation.png"
        )
        
        # ============================================
        # XGBOOST MODEL
        # ============================================
        logger.info("=" * 50)
        logger.info("Training XGBoost Model")
        logger.info("=" * 50)
        
        if not skip_tuning:
            xgb_params, _ = tune_xgboost(
                X_trainval, y_trainval,
                n_trials=n_tuning_trials,
            )
        else:
            xgb_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "scale_pos_weight": (y_trainval == 0).sum() / (y_trainval == 1).sum(),
                "random_state": 42,
            }
            
        # Cross-validation
        xgb_model, xgb_cv = cross_validate_xgboost(
            X_trainval, y_trainval, params=xgb_params
        )
        results["cv_results"]["xgboost"] = xgb_cv
        
        # Train final XGBoost
        xgb_model, xgb_train_metrics = train_xgboost_model(
            X_train_transformed, y_train.values,
            X_val_transformed, y_val.values,
            params=xgb_params,
        )
        
        # Evaluate on test set
        y_test_pred_xgb = xgb_model.predict(X_test_transformed)
        y_test_proba_xgb = xgb_model.predict_proba(X_test_transformed)[:, 1]
        xgb_test_metrics = evaluate_model(
            y_test.values, y_test_pred_xgb, y_test_proba_xgb, "XGBoost"
        )
        results["evaluation"]["xgboost"] = xgb_test_metrics
        
        # Save XGBoost plots
        create_evaluation_plots(
            y_test.values, y_test_pred_xgb, y_test_proba_xgb,
            "XGBoost",
            save_path=reports_dir / "xgboost_evaluation.png"
        )
        
        # Feature importance
        importance_df = get_feature_importance(xgb_model, feature_names)
        importance_df.to_csv(reports_dir / "feature_importance.csv", index=False)
        logger.info(f"Top 10 features:\n{importance_df.head(10)}")
        
        # ============================================
        # MODEL COMPARISON & SELECTION
        # ============================================
        logger.info("=" * 50)
        logger.info("Model Comparison & Selection")
        logger.info("=" * 50)
        
        model_results = {
            "baseline": baseline_test_metrics,
            "xgboost": xgb_test_metrics,
        }
        
        comparison_df = compare_models(model_results)
        logger.info(f"\nModel Comparison:\n{comparison_df}")
        
        best_model_name, justification = select_best_model(
            comparison_df,
            cv_results=results["cv_results"]
        )
        
        results["selected_model"] = best_model_name
        results["selection_justification"] = justification
        
        # ============================================
        # THRESHOLD OPTIMIZATION
        # ============================================
        logger.info("=" * 50)
        logger.info("Threshold Optimization")
        logger.info("=" * 50)
        
        # Use best model's predictions
        if best_model_name == "xgboost":
            y_proba_best = y_test_proba_xgb
            final_model = xgb_model
        else:
            y_proba_best = y_test_proba_baseline
            final_model = baseline_model
            
        # Find optimal thresholds with different methods
        for method in ["f1", "youden", "cost"]:
            threshold, threshold_metrics = find_optimal_threshold(
                y_test.values, y_proba_best, method=method
            )
            results[f"threshold_{method}"] = {
                "value": threshold,
                "metrics": threshold_metrics
            }
            
        # Use F1-optimized threshold as default
        optimal_threshold = results["threshold_f1"]["value"]
        
        # ============================================
        # SAVE FINAL MODEL & ARTIFACTS
        # ============================================
        logger.info("=" * 50)
        logger.info("Saving Final Model Artifacts")
        logger.info("=" * 50)
        
        # Save final model
        joblib.dump(final_model, output_dir / "final_model.joblib")
        logger.info(f"Saved final model ({best_model_name}) to {output_dir / 'final_model.joblib'}")
        
        # Save threshold configuration
        threshold_config = {
            "optimal_threshold": optimal_threshold,
            "selected_model": best_model_name,
            "threshold_methods": {
                method: {
                    "threshold": results[f"threshold_{method}"]["value"],
                    "f1_score": results[f"threshold_{method}"]["metrics"].get("f1_score"),
                }
                for method in ["f1", "youden", "cost"]
            }
        }
        with open(output_dir / "threshold.json", "w") as f:
            json.dump(threshold_config, f, indent=2)
            
        # Save evaluation results
        save_evaluation_results(
            results["evaluation"][best_model_name],
            output_dir / "evaluation_results.json",
            best_model_name
        )
        
        # Generate performance report
        create_performance_report(
            model_results,
            results["cv_results"],
            best_model_name,
            justification,
            reports_dir / "model_performance_report.md"
        )
        
        # Log final metrics to MLflow
        mlflow.log_metrics({
            f"final_{k}": v for k, v in results["evaluation"][best_model_name].items()
            if isinstance(v, (int, float))
        })
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("optimal_threshold", optimal_threshold)
        
        # Log artifacts
        mlflow.log_artifact(str(output_dir / "final_model.joblib"))
        mlflow.log_artifact(str(output_dir / "threshold.json"))
        
    logger.info("=" * 50)
    logger.info("Pipeline Complete!")
    logger.info(f"Selected Model: {best_model_name}")
    logger.info(f"Test ROC-AUC: {results['evaluation'][best_model_name]['roc_auc']:.4f}")
    logger.info(f"Optimal Threshold: {optimal_threshold:.3f}")
    logger.info("=" * 50)
    
    return results


def main():
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument(
        "--data", type=str, default="data/processed/telco_churn_processed.csv",
        help="Path to processed data file"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of hyperparameter tuning trials"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run with minimal settings for testing"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip hyperparameter tuning (use defaults)"
    )
    
    args = parser.parse_args()
    
    results = run_pipeline(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        n_tuning_trials=args.trials,
        quick_test=args.quick_test,
        skip_tuning=args.skip_tuning,
    )
    
    return results


if __name__ == "__main__":
    main()
