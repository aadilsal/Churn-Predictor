"""Scenario (what-if) analysis for churn counterfactuals."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import logger


class ScenarioAnalyzer:
    """Analyze what-if scenarios for churn intervention planning."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_preprocessor: Optional[Any] = None,
    ):
        """Initialize scenario analyzer.
        
        Args:
            model: Trained model with predict_proba
            feature_names: List of feature names in model input order
            feature_preprocessor: Optional preprocessor for raw features
        """
        self.model = model
        self.feature_names = feature_names
        self.preprocessor = feature_preprocessor
        
    def analyze_feature_change(
        self,
        X: np.ndarray,
        feature_name: str,
        new_value: float,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Analyze impact of changing a single feature.
        
        Args:
            X: Original feature data
            feature_name: Name of feature to change
            new_value: New value for the feature
            index: Sample index to analyze
            
        Returns:
            Scenario analysis results
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
            
        feat_idx = self.feature_names.index(feature_name)
        
        # Get original prediction
        sample = X[index:index+1].copy()
        original_prob = self.model.predict_proba(sample)[0, 1]
        original_value = sample[0, feat_idx]
        
        # Modify feature
        modified = sample.copy()
        modified[0, feat_idx] = new_value
        new_prob = self.model.predict_proba(modified)[0, 1]
        
        # Calculate impact
        prob_change = new_prob - original_prob
        
        return {
            "feature": feature_name,
            "original_value": float(original_value),
            "new_value": float(new_value),
            "original_probability": float(original_prob),
            "new_probability": float(new_prob),
            "probability_change": float(prob_change),
            "change_percent": float(prob_change * 100),
            "impact": "reduces risk" if prob_change < 0 else "increases risk",
            "effective": abs(prob_change) > 0.05,
        }
        
    def analyze_contract_upgrade(
        self,
        X: np.ndarray,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Analyze impact of upgrading from month-to-month to annual contract.
        
        Args:
            X: Original feature data
            index: Sample index
            
        Returns:
            Contract upgrade scenario results
        """
        results = {
            "scenario": "Contract Upgrade Analysis",
            "upgrades": [],
        }
        
        # Find contract-related features
        contract_features = [f for f in self.feature_names if "Contract" in f]
        
        sample = X[index:index+1].copy()
        original_prob = self.model.predict_proba(sample)[0, 1]
        
        for contract_feat in contract_features:
            feat_idx = self.feature_names.index(contract_feat)
            
            # Simulate having this contract type (set to 1)
            modified = sample.copy()
            
            # Reset all contract features to 0, then set target to 1
            for cf in contract_features:
                cf_idx = self.feature_names.index(cf)
                modified[0, cf_idx] = 0
            modified[0, feat_idx] = 1
            
            new_prob = self.model.predict_proba(modified)[0, 1]
            
            results["upgrades"].append({
                "contract_type": contract_feat.replace("Contract_", ""),
                "probability": float(new_prob),
                "change_from_current": float(new_prob - original_prob),
            })
            
        results["original_probability"] = float(original_prob)
        results["upgrades"].sort(key=lambda x: x["probability"])
        results["best_option"] = results["upgrades"][0] if results["upgrades"] else None
        
        return results
        
    def batch_scenario_analysis(
        self,
        X: np.ndarray,
        scenarios: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run multiple scenarios across all customers.
        
        Args:
            X: Feature data for all customers
            scenarios: List of {feature, new_value} dicts
            
        Returns:
            Aggregated scenario results
        """
        results = []
        
        for scenario in scenarios:
            feature = scenario["feature"]
            new_value = scenario["new_value"]
            
            if feature not in self.feature_names:
                logger.warning(f"Skipping unknown feature: {feature}")
                continue
                
            feat_idx = self.feature_names.index(feature)
            
            # Original predictions
            original_probs = self.model.predict_proba(X)[:, 1]
            
            # Modified predictions
            X_modified = X.copy()
            X_modified[:, feat_idx] = new_value
            new_probs = self.model.predict_proba(X_modified)[:, 1]
            
            # Calculate impacts
            changes = new_probs - original_probs
            
            results.append({
                "scenario": f"{feature} → {new_value}",
                "feature": feature,
                "new_value": new_value,
                "avg_probability_change": float(changes.mean()),
                "customers_helped": int((changes < -0.05).sum()),
                "customers_hurt": int((changes > 0.05).sum()),
                "total_risk_reduction": float(-changes.sum()),
            })
            
        return sorted(results, key=lambda x: x["total_risk_reduction"], reverse=True)
        
    def find_intervention_impact(
        self,
        X: np.ndarray,
        y_proba: np.ndarray,
        intervention_features: List[str],
        high_risk_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Find most impactful interventions for high-risk customers.
        
        Args:
            X: Feature data
            y_proba: Predicted probabilities
            intervention_features: Features that can be changed
            high_risk_threshold: Threshold for high-risk classification
            
        Returns:
            Intervention impact analysis
        """
        high_risk_mask = y_proba >= high_risk_threshold
        high_risk_X = X[high_risk_mask]
        high_risk_probs = y_proba[high_risk_mask]
        
        if len(high_risk_X) == 0:
            return {"message": "No high-risk customers found"}
            
        intervention_results = []
        
        for feature in intervention_features:
            if feature not in self.feature_names:
                continue
                
            feat_idx = self.feature_names.index(feature)
            
            # Try setting feature to both 0 and 1 to find best intervention
            for target_value in [0, 1]:
                X_modified = high_risk_X.copy()
                X_modified[:, feat_idx] = target_value
                new_probs = self.model.predict_proba(X_modified)[:, 1]
                
                changes = new_probs - high_risk_probs
                avg_reduction = -changes.mean()
                
                if avg_reduction > 0.01:  # Only include if helpful
                    intervention_results.append({
                        "feature": feature,
                        "target_value": target_value,
                        "avg_risk_reduction": float(avg_reduction),
                        "customers_below_threshold": int((new_probs < high_risk_threshold).sum()),
                        "pct_rescued": float((new_probs < high_risk_threshold).mean() * 100),
                    })
                    
        intervention_results.sort(key=lambda x: x["avg_risk_reduction"], reverse=True)
        
        return {
            "high_risk_customers": int(high_risk_mask.sum()),
            "avg_risk_probability": float(high_risk_probs.mean()),
            "top_interventions": intervention_results[:10],
            "recommendation": self._generate_intervention_recommendation(intervention_results),
        }
        
    def _generate_intervention_recommendation(
        self,
        interventions: List[Dict],
    ) -> str:
        """Generate intervention recommendation text."""
        if not interventions:
            return "No effective interventions found for current high-risk customers."
            
        top = interventions[0]
        return (
            f"Top recommendation: Modify '{top['feature']}' to {top['target_value']} "
            f"for average {top['avg_risk_reduction']*100:.1f}% risk reduction. "
            f"Could rescue {top['pct_rescued']:.0f}% of high-risk customers."
        )
        
    def generate_what_if_report(
        self,
        X: np.ndarray,
        sample_indices: List[int],
    ) -> Dict[str, Any]:
        """Generate what-if report for sample customers.
        
        Args:
            X: Feature data
            sample_indices: Indices of customers to analyze
            
        Returns:
            What-if report
        """
        customer_scenarios = []
        
        for idx in sample_indices:
            original_prob = self.model.predict_proba(X[idx:idx+1])[0, 1]
            
            # Analyze contract upgrade
            contract_impact = self.analyze_contract_upgrade(X, idx)
            
            customer_scenarios.append({
                "sample_index": idx,
                "original_probability": float(original_prob),
                "contract_scenarios": contract_impact["upgrades"],
                "best_contract": contract_impact.get("best_option"),
            })
            
        return {
            "customers_analyzed": len(sample_indices),
            "scenarios": customer_scenarios,
            "summary": self._summarize_scenarios(customer_scenarios),
        }
        
    def _summarize_scenarios(
        self,
        scenarios: List[Dict],
    ) -> List[str]:
        """Summarize scenario analysis findings."""
        summaries = []
        
        # Count how many would benefit from contract upgrade
        contract_helps = sum(
            1 for s in scenarios 
            if s.get("best_contract") and s["best_contract"]["change_from_current"] < -0.05
        )
        
        if contract_helps > 0:
            pct = contract_helps / len(scenarios) * 100
            summaries.append(
                f"{pct:.0f}% of analyzed customers would significantly benefit "
                "from a contract upgrade intervention."
            )
            
        return summaries
