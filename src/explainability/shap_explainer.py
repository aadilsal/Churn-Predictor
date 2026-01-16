"""SHAP-based model explainability for churn prediction."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logging import logger


class ShapExplainer:
    """SHAP-based explainer for churn prediction models.
    
    Provides both global and local (individual) explanations
    with business-friendly output formats.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
    ):
        """Initialize the SHAP explainer.
        
        Args:
            model: Trained model with predict_proba method
            feature_names: List of feature names
            background_data: Background data for SHAP (optional)
            n_background_samples: Number of background samples to use
        """
        self.model = model
        self.feature_names = feature_names
        self.n_background_samples = n_background_samples
        self._explainer: Optional[shap.Explainer] = None
        self._background_data = background_data
        self._global_shap_values: Optional[np.ndarray] = None
        
    def _create_explainer(self, X: np.ndarray) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model type.
        
        Args:
            X: Data to use for background (if not provided in __init__)
            
        Returns:
            SHAP Explainer instance
        """
        # Use provided background or sample from X
        if self._background_data is not None:
            background = self._background_data
        else:
            n_samples = min(self.n_background_samples, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            background = X[indices]
            
        # Try TreeExplainer first (faster for tree-based models)
        model_type = type(self.model).__name__
        
        if "XGB" in model_type or "LightGBM" in model_type or "Random" in model_type:
            try:
                explainer = shap.TreeExplainer(self.model)
                logger.info(f"Using TreeExplainer for {model_type}")
                return explainer
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {e}, falling back to KernelExplainer")
                
        # Fallback to KernelExplainer
        def predict_proba_positive(X):
            return self.model.predict_proba(X)[:, 1]
            
        explainer = shap.KernelExplainer(predict_proba_positive, background)
        logger.info(f"Using KernelExplainer for {model_type}")
        return explainer
        
    def compute_shap_values(
        self,
        X: np.ndarray,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """Compute SHAP values for given data.
        
        Args:
            X: Feature data
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array
        """
        if self._explainer is None:
            self._explainer = self._create_explainer(X)
            
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        # Compute SHAP values
        shap_values = self._explainer.shap_values(X, check_additivity=check_additivity)
        
        # Handle multi-output (binary classification returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
            
        return shap_values
        
    def get_global_importance(
        self,
        X: np.ndarray,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Get global feature importance based on mean |SHAP|.
        
        Args:
            X: Feature data to compute SHAP values on
            normalize: Whether to normalize importance to sum to 1
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self.compute_shap_values(X)
        self._global_shap_values = shap_values
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        if normalize:
            importance = importance / importance.sum()
            
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "mean_shap": shap_values.mean(axis=0),  # Direction of impact
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
        
    def explain_individual(
        self,
        X: np.ndarray,
        index: int = 0,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Generate explanation for an individual prediction.
        
        Args:
            X: Feature data (single sample or array)
            index: Index of sample to explain (if X has multiple rows)
            top_n: Number of top features to include
            
        Returns:
            Dictionary with explanation details
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        sample = X[index:index+1]
        shap_values = self.compute_shap_values(sample)[0]
        
        # Get base value (expected value)
        if hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5
            
        # Get predicted probability
        pred_proba = self.model.predict_proba(sample)[0, 1]
        
        # Sort features by absolute impact
        feature_impacts = list(zip(self.feature_names, shap_values, sample[0]))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build explanation
        top_drivers = []
        for name, shap_val, feat_val in feature_impacts[:top_n]:
            direction = "increases" if shap_val > 0 else "decreases"
            top_drivers.append({
                "feature": name,
                "value": float(feat_val),
                "shap_value": float(shap_val),
                "direction": direction,
                "impact": "risk" if shap_val > 0 else "protection",
            })
            
        return {
            "prediction_probability": float(pred_proba),
            "base_probability": float(base_value),
            "risk_level": self._get_risk_level(pred_proba),
            "top_drivers": top_drivers,
            "all_shap_values": dict(zip(self.feature_names, shap_values.tolist())),
        }
        
    def _get_risk_level(self, probability: float) -> str:
        """Categorize probability into risk level."""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
            
    def plot_summary(
        self,
        X: np.ndarray,
        plot_type: str = "bar",
        max_display: int = 15,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create SHAP summary plot.
        
        Args:
            X: Feature data
            plot_type: 'bar', 'beeswarm', or 'dot'
            max_display: Maximum features to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self._global_shap_values is None:
            shap_values = self.compute_shap_values(X)
        else:
            shap_values = self._global_shap_values
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == "bar":
            shap.summary_plot(
                shap_values, X,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values, X,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {save_path}")
            
        return plt.gcf()
        
    def plot_waterfall(
        self,
        X: np.ndarray,
        index: int = 0,
        max_display: int = 10,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create SHAP waterfall plot for individual prediction.
        
        Args:
            X: Feature data
            index: Sample index to explain
            max_display: Maximum features to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        shap_values = self.compute_shap_values(X[index:index+1])
        
        # Get base value
        if hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5
            
        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X[index],
            feature_names=self.feature_names
        )
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title("Individual Prediction Explanation", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved waterfall plot to {save_path}")
            
        return fig
        
    def validate_stability(
        self,
        X: np.ndarray,
        n_samples: int = 3,
        sample_size: int = 500,
    ) -> Dict[str, Any]:
        """Validate SHAP explanation stability across samples.
        
        Args:
            X: Full dataset
            n_samples: Number of random samples to compare
            sample_size: Size of each sample
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info("Validating SHAP explanation stability...")
        
        importance_rankings = []
        
        for i in range(n_samples):
            indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            sample = X[indices]
            
            importance_df = self.get_global_importance(sample, normalize=True)
            importance_rankings.append(importance_df.set_index("feature")["importance"])
            
        # Calculate correlation between rankings
        combined = pd.DataFrame(importance_rankings).T
        correlations = combined.corr().values
        
        # Get average importance
        mean_importance = combined.mean(axis=1)
        std_importance = combined.std(axis=1)
        
        stability_score = correlations[np.triu_indices(n_samples, k=1)].mean()
        
        return {
            "stability_score": float(stability_score),
            "interpretation": "stable" if stability_score > 0.9 else "moderate" if stability_score > 0.7 else "unstable",
            "mean_importance": mean_importance.to_dict(),
            "std_importance": std_importance.to_dict(),
        }
        
    def save(self, path: Path) -> None:
        """Save explainer artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save feature names
        with open(path / "shap_feature_names.json", "w") as f:
            json.dump({"feature_names": self.feature_names}, f, indent=2)
            
        logger.info(f"Saved SHAP explainer artifacts to {path}")
        
    @classmethod
    def from_saved_model(
        cls,
        model_path: Path,
        feature_names_path: Path,
    ) -> "ShapExplainer":
        """Load explainer from saved model artifacts."""
        model = joblib.load(model_path)
        
        with open(feature_names_path, "r") as f:
            data = json.load(f)
            feature_names = data.get("feature_names", data.get("feature_names"))
            
        return cls(model=model, feature_names=feature_names)
