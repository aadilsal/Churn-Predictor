"""Automated MLOps pipeline for end-to-end training."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.mlops.tracking import (
    init_tracking,
    log_artifact,
    log_data_info,
    log_feature_info,
    log_metrics,
    log_model,
    log_params,
    start_run,
)
from src.mlops.registry import ModelRegistry
from src.utils.logging import logger


class MLOpsPipeline:
    """End-to-end MLOps training pipeline."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        use_dagshub: bool = True,
    ):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to training config YAML
            use_dagshub: Whether to use DagsHub for tracking
        """
        self.config_path = config_path or Path("config/training_config.yaml")
        self.use_dagshub = use_dagshub
        self.config = self._load_config()
        self.registry = ModelRegistry()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "path": "data/processed/telco_churn_processed.csv",
                "target": "Churn",
                "test_size": 0.2,
                "val_size": 0.2,
            },
            "model": {
                "type": "xgboost",
                "params": {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 200,
                },
            },
            "training": {
                "n_tuning_trials": 50,
                "random_seed": 42,
                "early_stopping_rounds": 50,
            },
            "registration": {
                "model_name": "churn-predictor",
                "auto_register": True,
                "min_roc_auc": 0.8,
            },
        }
        
    def _compute_data_version(self, df: pd.DataFrame) -> str:
        """Compute a hash-based version for the dataset."""
        content = df.to_csv(index=False)
        return hashlib.md5(content.encode()).hexdigest()[:8]
        
    def run(
        self,
        experiment_name: str = "churn-model-training",
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full training pipeline.
        
        Args:
            experiment_name: MLflow experiment name
            run_name: Optional run name
            
        Returns:
            Pipeline results
        """
        from src.data.preprocessing import clean_telco_dataset
        from src.models.baseline_model import cross_validate_baseline, train_baseline_model
        from src.models.evaluation import evaluate_model, find_optimal_threshold
        from src.models.feature_engineering import (
            FeatureEngineer,
            create_train_test_split,
            prepare_features,
        )
        from src.models.hyperparameter_tuning import tune_xgboost
        from src.models.xgboost_model import cross_validate_xgboost, train_xgboost_model
        
        logger.info("=" * 60)
        logger.info("Starting MLOps Training Pipeline")
        logger.info("=" * 60)
        
        # Initialize tracking
        init_tracking(experiment_name, use_dagshub=self.use_dagshub)
        
        run_name = run_name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            results = {"run_id": run_id}
            
            # ========================================
            # STEP 1: Load and Version Data
            # ========================================
            logger.info("Step 1: Loading and versioning data...")
            
            data_path = Path(self.config["data"]["path"])
            df = pd.read_csv(data_path)
            
            data_version = self._compute_data_version(df)
            log_params({"data_version": data_version, "data_path": str(data_path)})
            
            # ========================================
            # STEP 2: Prepare Features
            # ========================================
            logger.info("Step 2: Preparing features...")
            
            X, y = prepare_features(df, target_col=self.config["data"]["target"])
            
            X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
                X, y,
                test_size=self.config["data"]["test_size"],
                val_size=self.config["data"]["val_size"],
                random_state=self.config["training"]["random_seed"],
            )
            
            log_data_info(
                n_samples=len(df),
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
                class_balance=float(y.mean()),
            )
            
            # Feature engineering
            feature_engineer = FeatureEngineer()
            X_train_t = feature_engineer.fit_transform(X_train)
            X_val_t = feature_engineer.transform(X_val)
            X_test_t = feature_engineer.transform(X_test)
            
            feature_names = feature_engineer.get_feature_names()
            feature_version = hashlib.md5(",".join(feature_names).encode()).hexdigest()[:8]
            
            log_feature_info(
                feature_names=feature_names,
                categorical_features=feature_engineer.categorical_cols,
                numerical_features=feature_engineer.numerical_cols,
            )
            log_params({"feature_version": feature_version})
            
            # ========================================
            # STEP 3: Train Model
            # ========================================
            logger.info("Step 3: Training model...")
            
            model_type = self.config["model"]["type"]
            log_params({"model_type": model_type})
            log_params(self.config["model"]["params"])
            
            X_trainval = np.vstack([X_train_t, X_val_t])
            y_trainval = np.concatenate([y_train.values, y_val.values])
            
            if model_type == "xgboost":
                # Hyperparameter tuning
                n_trials = self.config["training"]["n_tuning_trials"]
                if n_trials > 0:
                    best_params, _ = tune_xgboost(
                        X_trainval, y_trainval,
                        n_trials=n_trials,
                        log_to_mlflow=False,
                    )
                else:
                    best_params = self.config["model"]["params"]
                    
                log_params({"tuned_params": best_params})
                
                # Cross-validation
                model, cv_results = cross_validate_xgboost(
                    X_trainval, y_trainval,
                    params=best_params,
                )
                
                log_metrics({
                    "cv_roc_auc_mean": cv_results["mean_roc_auc"],
                    "cv_roc_auc_std": cv_results["std_roc_auc"],
                    "cv_f1_mean": cv_results["mean_f1_score"],
                })
                
                # Train final model
                model, train_metrics = train_xgboost_model(
                    X_train_t, y_train.values,
                    X_val_t, y_val.values,
                    params=best_params,
                    log_to_mlflow=False,
                )
            else:
                # Baseline model
                model, cv_results = cross_validate_baseline(X_trainval, y_trainval)
                model, train_metrics = train_baseline_model(
                    X_train_t, y_train.values,
                    X_val_t, y_val.values,
                    log_to_mlflow=False,
                )
                
            log_metrics(train_metrics)
            
            # ========================================
            # STEP 4: Evaluate on Test Set
            # ========================================
            logger.info("Step 4: Evaluating on test set...")
            
            y_test_pred = model.predict(X_test_t)
            y_test_proba = model.predict_proba(X_test_t)[:, 1]
            
            test_metrics = evaluate_model(
                y_test.values, y_test_pred, y_test_proba,
                model_name=model_type,
            )
            
            log_metrics(test_metrics, prefix="test_")
            
            # Threshold optimization
            optimal_threshold, _ = find_optimal_threshold(y_test.values, y_test_proba)
            log_params({"optimal_threshold": optimal_threshold})
            
            results["metrics"] = test_metrics
            results["threshold"] = optimal_threshold
            
            # ========================================
            # STEP 5: Save and Log Artifacts
            # ========================================
            logger.info("Step 5: Saving artifacts...")
            
            # Save feature engineer
            artifact_dir = Path("models")
            artifact_dir.mkdir(exist_ok=True)
            feature_engineer.save(artifact_dir)
            
            # Log model
            log_model(model, "model", model_type=model_type)
            
            # Log artifacts
            log_artifact(artifact_dir / "feature_preprocessor.joblib", "preprocessing")
            log_artifact(artifact_dir / "feature_names.json", "preprocessing")
            
            # ========================================
            # STEP 6: Conditional Registration
            # ========================================
            logger.info("Step 6: Checking registration criteria...")
            
            register_config = self.config["registration"]
            should_register = (
                register_config["auto_register"] and
                test_metrics["roc_auc"] >= register_config["min_roc_auc"]
            )
            
            if should_register:
                logger.info("Model meets criteria, registering...")
                
                version = self.registry.register_with_metadata(
                    run_id=run_id,
                    model_path="model",
                    model_name=register_config["model_name"],
                    model_type=model_type,
                    metrics=test_metrics,
                    dataset_version=data_version,
                    feature_version=feature_version,
                    training_config=self.config,
                )
                
                results["registered"] = True
                results["model_version"] = version
                results["model_name"] = register_config["model_name"]
                
                log_params({"registered_version": version})
            else:
                logger.info(f"Model did not meet criteria (ROC-AUC: {test_metrics['roc_auc']:.4f} < {register_config['min_roc_auc']})")
                results["registered"] = False
                
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Registered: {results.get('registered', False)}")
        logger.info("=" * 60)
        
        return results
        
    def run_quick(self) -> Dict[str, Any]:
        """Run pipeline with minimal tuning for testing."""
        # Override config for quick test
        self.config["training"]["n_tuning_trials"] = 0
        return self.run(run_name="quick_test")


def main():
    """CLI entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MLOps training pipeline")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--experiment", type=str, default="churn-model-training")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--local", action="store_true", help="Use local MLflow only")
    
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline(
        config_path=Path(args.config),
        use_dagshub=not args.local,
    )
    
    if args.quick:
        results = pipeline.run_quick()
    else:
        results = pipeline.run(experiment_name=args.experiment, run_name=args.run_name)
        
    print(json.dumps(results, indent=2, default=str))
    

if __name__ == "__main__":
    main()
