"""MLflow Model Registry for versioning and stage management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import logger


class ModelRegistry:
    """Model registry for versioning and stage promotion."""
    
    # Stage definitions
    STAGES = ["None", "Staging", "Production", "Archived"]
    
    def __init__(self):
        """Initialize model registry client."""
        self.client = MlflowClient()
        
    def register_model(
        self,
        run_id: str,
        model_path: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a model from a run to the registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_path: Path to model artifact within run
            model_name: Name for the registered model
            description: Model description
            tags: Additional tags
            
        Returns:
            Model version number
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        
        logger.info(f"Registered model '{model_name}' version {version}")
        
        # Update description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description,
            )
            
        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)
                
        return version
        
    def register_with_metadata(
        self,
        run_id: str,
        model_path: str,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        dataset_version: str,
        feature_version: str,
        training_config: Dict[str, Any],
    ) -> str:
        """Register model with full lineage metadata.
        
        Args:
            run_id: MLflow run ID
            model_path: Model artifact path
            model_name: Registry model name
            model_type: Type of model (e.g., 'xgboost', 'logistic_regression')
            metrics: Performance metrics
            dataset_version: Version identifier for training data
            feature_version: Version identifier for feature set
            training_config: Training configuration used
            
        Returns:
            Model version number
        """
        # Create description with key metrics
        description = (
            f"Model Type: {model_type}\n"
            f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}\n"
            f"PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}\n"
            f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Dataset: {dataset_version}\n"
            f"Features: {feature_version}"
        )
        
        # Comprehensive tags
        tags = {
            "model_type": model_type,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "roc_auc": str(metrics.get("roc_auc", "")),
            "pr_auc": str(metrics.get("pr_auc", "")),
            "training_date": datetime.now().isoformat(),
        }
        
        # Add config as tag (truncated if too long)
        config_str = json.dumps(training_config)
        if len(config_str) < 500:
            tags["training_config"] = config_str
            
        return self.register_model(
            run_id=run_id,
            model_path=model_path,
            model_name=model_name,
            description=description,
            tags=tags,
        )
        
    def promote_to_staging(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """Promote a model version to Staging.
        
        Args:
            model_name: Registered model name
            version: Version to promote
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        logger.info(f"Promoted {model_name} v{version} to Staging")
        
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_current: bool = True,
    ) -> None:
        """Promote a model version to Production.
        
        Args:
            model_name: Registered model name
            version: Version to promote
            archive_current: Whether to archive current production model
        """
        if archive_current:
            # Archive current production models
            current_prod = self.get_production_model(model_name)
            if current_prod:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_prod["version"],
                    stage="Archived",
                )
                logger.info(f"Archived previous production model v{current_prod['version']}")
                
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        logger.info(f"Promoted {model_name} v{version} to Production")
        
    def get_production_model(
        self,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get current production model info.
        
        Args:
            model_name: Registered model name
            
        Returns:
            Production model info or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                v = versions[0]
                return {
                    "version": v.version,
                    "run_id": v.run_id,
                    "description": v.description,
                    "tags": v.tags,
                }
        except Exception as e:
            logger.warning(f"Error getting production model: {e}")
            
        return None
        
    def get_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all versions of a model.
        
        Args:
            model_name: Registered model name
            stages: Filter by stages
            
        Returns:
            List of model version info
        """
        try:
            if stages:
                versions = self.client.get_latest_versions(model_name, stages=stages)
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "description": v.description,
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in versions
            ]
        except Exception as e:
            logger.warning(f"Error getting model versions: {e}")
            return []
            
    def load_production_model(
        self,
        model_name: str,
    ) -> Any:
        """Load the production model.
        
        Args:
            model_name: Registered model name
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)
        
    def load_model_version(
        self,
        model_name: str,
        version: str,
    ) -> Any:
        """Load a specific model version.
        
        Args:
            model_name: Registered model name
            version: Version number
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
        
    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            model_name: Registered model name
            version_a: First version
            version_b: Second version
            
        Returns:
            Comparison results
        """
        versions = self.get_model_versions(model_name)
        
        v_a = next((v for v in versions if v["version"] == version_a), None)
        v_b = next((v for v in versions if v["version"] == version_b), None)
        
        if not v_a or not v_b:
            return {"error": "One or both versions not found"}
            
        # Get run metrics
        run_a = self.client.get_run(v_a["run_id"])
        run_b = self.client.get_run(v_b["run_id"])
        
        return {
            "version_a": {
                "version": version_a,
                "metrics": run_a.data.metrics,
            },
            "version_b": {
                "version": version_b,
                "metrics": run_b.data.metrics,
            },
            "comparison": {
                metric: run_b.data.metrics.get(metric, 0) - run_a.data.metrics.get(metric, 0)
                for metric in run_a.data.metrics.keys()
            },
        }
        
    def get_model_lineage(
        self,
        model_name: str,
        version: str,
    ) -> Dict[str, Any]:
        """Get complete lineage for a model version.
        
        Args:
            model_name: Registered model name
            version: Version number
            
        Returns:
            Lineage information
        """
        versions = self.get_model_versions(model_name)
        v = next((v for v in versions if v["version"] == version), None)
        
        if not v:
            return {"error": "Version not found"}
            
        run = self.client.get_run(v["run_id"])
        
        return {
            "model_name": model_name,
            "version": version,
            "run_id": v["run_id"],
            "stage": v.get("stage", "None"),
            "created": v.get("creation_timestamp"),
            "parameters": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
            "artifact_uri": run.info.artifact_uri,
        }
