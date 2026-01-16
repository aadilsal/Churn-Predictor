"""Unified explanation service for API and dashboard integration."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.explainability.business_insights import (
    BusinessInsightGenerator,
    create_customer_risk_profile,
)
from src.explainability.cohort_analysis import CohortAnalyzer
from src.explainability.scenario_analysis import ScenarioAnalyzer
from src.explainability.shap_explainer import ShapExplainer
from src.utils.logging import logger


class ExplanationService:
    """Unified service for model explanations.
    
    Provides programmatic interfaces for dashboards and APIs
    with structured JSON outputs.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
    ):
        """Initialize the explanation service.
        
        Args:
            model: Trained model with predict_proba
            feature_names: List of feature names
            background_data: Background data for SHAP (optional)
        """
        self.model = model
        self.feature_names = feature_names
        
        # Initialize components
        self.shap_explainer = ShapExplainer(
            model=model,
            feature_names=feature_names,
            background_data=background_data,
        )
        self.insight_generator = BusinessInsightGenerator(feature_names)
        self.cohort_analyzer = CohortAnalyzer(feature_names)
        self.scenario_analyzer = ScenarioAnalyzer(model, feature_names)
        
        self._global_importance: Optional[pd.DataFrame] = None
        
    @classmethod
    def from_artifacts(
        cls,
        model_path: Path,
        feature_names_path: Path,
    ) -> "ExplanationService":
        """Load explanation service from saved artifacts.
        
        Args:
            model_path: Path to saved model
            feature_names_path: Path to feature names JSON
            
        Returns:
            Initialized ExplanationService
        """
        model = joblib.load(model_path)
        
        with open(feature_names_path, "r") as f:
            data = json.load(f)
            feature_names = data.get("feature_names", [])
            
        return cls(model=model, feature_names=feature_names)
        
    def compute_global_explanations(
        self,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute global model explanations.
        
        Args:
            X: Feature data
            
        Returns:
            Global explanation results
        """
        logger.info("Computing global explanations...")
        
        # Get global importance
        importance_df = self.shap_explainer.get_global_importance(X)
        self._global_importance = importance_df
        
        # Validate stability
        stability = self.shap_explainer.validate_stability(X)
        
        # Convert to JSON-serializable format
        top_features = []
        for _, row in importance_df.head(15).iterrows():
            top_features.append({
                "feature": row["feature"],
                "business_name": self.insight_generator.get_business_name(row["feature"]),
                "importance": float(row["importance"]),
                "direction": "risk_driver" if row["mean_shap"] > 0 else "protective",
            })
            
        return {
            "timestamp": datetime.now().isoformat(),
            "samples_analyzed": len(X),
            "top_features": top_features,
            "stability": {
                "score": stability["stability_score"],
                "interpretation": stability["interpretation"],
            },
        }
        
    def explain_customer(
        self,
        X: np.ndarray,
        customer_id: Optional[str] = None,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Generate complete explanation for a customer.
        
        Args:
            X: Feature data (single customer or batch)
            customer_id: Optional customer identifier
            index: Index if X contains multiple customers
            
        Returns:
            Complete customer explanation with risk profile
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Get SHAP explanation
        shap_explanation = self.shap_explainer.explain_individual(X, index=index)
        
        # Generate business explanation
        business_explanation = self.insight_generator.generate_customer_explanation(
            shap_explanation,
            include_recommendations=True,
        )
        
        # Create risk profile
        profile = create_customer_risk_profile(
            customer_id=customer_id or f"customer_{index}",
            churn_probability=shap_explanation["prediction_probability"],
            shap_explanation=shap_explanation,
            business_explanation=business_explanation,
        )
        
        return profile
        
    def explain_batch(
        self,
        X: np.ndarray,
        customer_ids: Optional[List[str]] = None,
        top_n_per_customer: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate explanations for multiple customers.
        
        Args:
            X: Feature data for all customers
            customer_ids: Optional list of customer identifiers
            top_n_per_customer: Top features per customer
            
        Returns:
            List of customer explanations
        """
        logger.info(f"Generating explanations for {len(X)} customers...")
        
        explanations = []
        
        for i in range(len(X)):
            customer_id = customer_ids[i] if customer_ids else None
            exp = self.explain_customer(X, customer_id=customer_id, index=i)
            explanations.append(exp)
            
        return explanations
        
    def analyze_cohorts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segment_by: str = "Contract_Two year",
    ) -> Dict[str, Any]:
        """Analyze churn patterns by customer cohorts.
        
        Args:
            X: Feature data
            y: Labels or probabilities
            segment_by: Feature to segment by
            
        Returns:
            Cohort analysis results
        """
        # Compute SHAP values if not already done
        shap_values = self.shap_explainer.compute_shap_values(X)
        
        # Analyze by segment
        segment_results = self.cohort_analyzer.analyze_by_segment(
            X, y, shap_values, segment_by
        )
        
        # Compare cohorts
        comparison = self.cohort_analyzer.compare_cohorts(segment_results)
        
        return comparison
        
    def run_scenario(
        self,
        X: np.ndarray,
        feature: str,
        new_value: float,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Run what-if scenario analysis.
        
        Args:
            X: Feature data
            feature: Feature to modify
            new_value: New value for the feature
            index: Customer index
            
        Returns:
            Scenario analysis results
        """
        return self.scenario_analyzer.analyze_feature_change(
            X, feature, new_value, index
        )
        
    def find_best_interventions(
        self,
        X: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Find most effective interventions for high-risk customers.
        
        Args:
            X: Feature data
            y_proba: Predicted probabilities
            
        Returns:
            Intervention recommendations
        """
        # Features that can reasonably be changed
        intervention_features = [
            f for f in self.feature_names 
            if any(kw in f for kw in ["Contract", "PaymentMethod", "Paperless",
                                       "OnlineSecurity", "TechSupport", "OnlineBackup",
                                       "DeviceProtection", "StreamingTV", "StreamingMovies"])
        ]
        
        return self.scenario_analyzer.find_intervention_impact(
            X, y_proba, intervention_features
        )
        
    def generate_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation report.
        
        Args:
            X: Feature data
            y: True labels or probabilities
            output_path: Path to save report (optional)
            
        Returns:
            Complete explanation report
        """
        logger.info("Generating comprehensive explanation report...")
        
        # Global explanations
        global_exp = self.compute_global_explanations(X)
        
        # Sample customer explanations
        sample_size = min(100, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        sample_explanations = [
            self.explain_customer(X, index=i) for i in sample_indices[:10]
        ]
        
        # Get predicted probabilities
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Intervention analysis
        interventions = self.find_best_interventions(X, y_proba)
        
        # Generate insight report
        if self._global_importance is not None:
            insight_report = self.insight_generator.generate_insight_report(
                self._global_importance,
                sample_explanations,
            )
        else:
            insight_report = {}
            
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_summary": {
                "total_customers": len(X),
                "churn_rate": float(y.mean()) if y.max() <= 1 else None,
                "avg_churn_probability": float(y_proba.mean()),
            },
            "global_explanations": global_exp,
            "sample_explanations": sample_explanations,
            "intervention_analysis": interventions,
            "business_insights": insight_report,
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Saved explanation report to {output_path}")
            
        return report
        
    def to_json(self, data: Any) -> str:
        """Convert explanation data to JSON string.
        
        Args:
            data: Explanation data
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=2, default=str)
        
    def get_api_response(
        self,
        X: np.ndarray,
        customer_id: Optional[str] = None,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Get API-ready response for a customer explanation.
        
        Args:
            X: Feature data
            customer_id: Customer identifier
            index: Customer index
            
        Returns:
            API response with status and data
        """
        try:
            explanation = self.explain_customer(X, customer_id, index)
            return {
                "status": "success",
                "data": explanation,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def run_explanation_pipeline(
    model_path: Path = Path("models/final_model.joblib"),
    feature_names_path: Path = Path("models/feature_names.json"),
    data_path: Path = Path("data/processed/telco_churn_processed.csv"),
    output_dir: Path = Path("reports"),
) -> Dict[str, Any]:
    """Run complete explanation pipeline.
    
    Args:
        model_path: Path to trained model
        feature_names_path: Path to feature names
        data_path: Path to processed data
        output_dir: Output directory for reports
        
    Returns:
        Pipeline results
    """
    from src.models.feature_engineering import FeatureEngineer, prepare_features
    
    logger.info("Starting explanation pipeline...")
    
    # Load model and feature names
    service = ExplanationService.from_artifacts(model_path, feature_names_path)
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    X, y = prepare_features(df)
    
    # Load feature engineer and transform
    feature_engineer = FeatureEngineer.load(Path("models"))
    X_transformed = feature_engineer.transform(X)
    
    # Generate SHAP plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    service.shap_explainer.plot_summary(
        X_transformed, plot_type="bar",
        save_path=output_dir / "shap_summary_bar.png"
    )
    
    service.shap_explainer.plot_summary(
        X_transformed, plot_type="dot",
        save_path=output_dir / "shap_summary_beeswarm.png"
    )
    
    # Generate sample waterfall plots
    for i in range(3):
        service.shap_explainer.plot_waterfall(
            X_transformed, index=i,
            save_path=output_dir / f"shap_waterfall_customer_{i}.png"
        )
        
    # Generate full report
    report = service.generate_report(
        X_transformed, y.values,
        output_path=output_dir / "explanation_report.json"
    )
    
    logger.info("Explanation pipeline complete!")
    
    return report
