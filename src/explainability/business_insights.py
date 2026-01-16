"""Business-friendly insight generation from model explanations."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import logger


# Feature to business language mapping
FEATURE_BUSINESS_NAMES = {
    "Contract_One year": "One-Year Contract",
    "Contract_Two year": "Two-Year Contract",
    "Contract_Month-to-month": "Month-to-Month Contract",
    "InternetService_Fiber optic": "Fiber Optic Internet",
    "InternetService_DSL": "DSL Internet",
    "InternetService_No": "No Internet Service",
    "PaymentMethod_Electronic check": "Electronic Check Payment",
    "PaymentMethod_Mailed check": "Mailed Check Payment",
    "PaymentMethod_Bank transfer (automatic)": "Automatic Bank Transfer",
    "PaymentMethod_Credit card (automatic)": "Automatic Credit Card",
    "tenure": "Customer Tenure (months)",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "PhoneService_Yes": "Phone Service",
    "MultipleLines_Yes": "Multiple Phone Lines",
    "OnlineSecurity_Yes": "Online Security Service",
    "OnlineBackup_Yes": "Online Backup Service",
    "DeviceProtection_Yes": "Device Protection",
    "TechSupport_Yes": "Tech Support Service",
    "StreamingTV_Yes": "Streaming TV",
    "StreamingMovies_Yes": "Streaming Movies",
    "PaperlessBilling_Yes": "Paperless Billing",
    "SeniorCitizen_Yes": "Senior Citizen",
    "Partner_Yes": "Has Partner",
    "Dependents_Yes": "Has Dependents",
    "gender_Male": "Male Customer",
}

# Intervention recommendations based on churn drivers
INTERVENTION_RECOMMENDATIONS = {
    "Contract_Month-to-month": {
        "risk": True,
        "recommendation": "Offer incentive to upgrade to annual contract",
        "action": "Present 1-year or 2-year contract options with discount",
        "priority": "HIGH",
    },
    "Contract_Two year": {
        "risk": False,
        "recommendation": "Customer locked in - focus on satisfaction",
        "action": "Ensure positive experience for renewal",
        "priority": "LOW",
    },
    "Contract_One year": {
        "risk": False,
        "recommendation": "Monitor for contract renewal opportunity",
        "action": "Proactive outreach before contract ends",
        "priority": "MEDIUM",
    },
    "InternetService_Fiber optic": {
        "risk": True,
        "recommendation": "Review pricing competitiveness for fiber plans",
        "action": "Consider loyalty discount or bundle offer",
        "priority": "HIGH",
    },
    "tenure": {
        "risk": True,  # Low tenure is risk
        "recommendation": "New customer requires onboarding attention",
        "action": "Engage with welcome program and satisfaction check",
        "priority": "HIGH",
    },
    "MonthlyCharges": {
        "risk": True,  # High charges can be risk
        "recommendation": "Review if charges align with perceived value",
        "action": "Offer plan optimization or value-add services",
        "priority": "MEDIUM",
    },
    "PaymentMethod_Electronic check": {
        "risk": True,
        "recommendation": "Encourage automatic payment method",
        "action": "Offer incentive for switching to auto-pay",
        "priority": "MEDIUM",
    },
    "OnlineSecurity_Yes": {
        "risk": False,
        "recommendation": "Customer uses security service - good engagement",
        "action": "Cross-sell complementary services",
        "priority": "LOW",
    },
    "TechSupport_Yes": {
        "risk": False,
        "recommendation": "Customer values support services",
        "action": "Ensure support quality remains high",
        "priority": "LOW",
    },
}


class BusinessInsightGenerator:
    """Generate business-friendly insights from SHAP explanations."""
    
    def __init__(
        self,
        feature_names: List[str],
        custom_feature_names: Optional[Dict[str, str]] = None,
        custom_interventions: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize the insight generator.
        
        Args:
            feature_names: List of model feature names
            custom_feature_names: Override default feature name mappings
            custom_interventions: Override default intervention recommendations
        """
        self.feature_names = feature_names
        self.feature_name_map = {**FEATURE_BUSINESS_NAMES}
        if custom_feature_names:
            self.feature_name_map.update(custom_feature_names)
        self.interventions = {**INTERVENTION_RECOMMENDATIONS}
        if custom_interventions:
            self.interventions.update(custom_interventions)
            
    def get_business_name(self, feature: str) -> str:
        """Get business-friendly name for a feature."""
        return self.feature_name_map.get(feature, feature.replace("_", " ").title())
        
    def generate_customer_explanation(
        self,
        shap_explanation: Dict[str, Any],
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """Generate business-friendly explanation for a customer.
        
        Args:
            shap_explanation: Output from ShapExplainer.explain_individual()
            include_recommendations: Whether to include action recommendations
            
        Returns:
            Business-friendly explanation dictionary
        """
        probability = shap_explanation["prediction_probability"]
        risk_level = shap_explanation["risk_level"]
        top_drivers = shap_explanation["top_drivers"]
        
        # Generate narrative explanation
        risk_factors = []
        protective_factors = []
        recommendations = []
        
        for driver in top_drivers:
            feature = driver["feature"]
            business_name = self.get_business_name(feature)
            
            if driver["impact"] == "risk":
                risk_factors.append({
                    "factor": business_name,
                    "original_feature": feature,
                    "contribution": abs(driver["shap_value"]),
                })
            else:
                protective_factors.append({
                    "factor": business_name,
                    "original_feature": feature,
                    "contribution": abs(driver["shap_value"]),
                })
                
            # Get intervention if available
            if include_recommendations and feature in self.interventions:
                intervention = self.interventions[feature]
                if (intervention["risk"] and driver["impact"] == "risk") or \
                   (not intervention["risk"] and driver["impact"] == "protection"):
                    recommendations.append({
                        "feature": business_name,
                        "recommendation": intervention["recommendation"],
                        "action": intervention["action"],
                        "priority": intervention["priority"],
                    })
                    
        # Generate summary narrative
        narrative = self._generate_narrative(probability, risk_level, risk_factors, protective_factors)
        
        return {
            "churn_probability": probability,
            "churn_probability_percent": f"{probability * 100:.1f}%",
            "risk_level": risk_level,
            "narrative_summary": narrative,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommended_actions": recommendations,
        }
        
    def _generate_narrative(
        self,
        probability: float,
        risk_level: str,
        risk_factors: List[Dict],
        protective_factors: List[Dict],
    ) -> str:
        """Generate human-readable narrative summary."""
        if risk_level == "HIGH":
            intro = f"This customer has a HIGH risk of churning ({probability*100:.0f}% probability)."
        elif risk_level == "MEDIUM":
            intro = f"This customer has a MODERATE risk of churning ({probability*100:.0f}% probability)."
        else:
            intro = f"This customer has a LOW risk of churning ({probability*100:.0f}% probability)."
            
        risk_text = ""
        if risk_factors:
            top_risks = [f["factor"] for f in risk_factors[:3]]
            risk_text = f" Key risk factors include: {', '.join(top_risks)}."
            
        protection_text = ""
        if protective_factors:
            top_protection = [f["factor"] for f in protective_factors[:2]]
            protection_text = f" Factors reducing risk: {', '.join(top_protection)}."
            
        return intro + risk_text + protection_text
        
    def generate_top_interventions(
        self,
        global_importance: pd.DataFrame,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate top intervention recommendations based on global importance.
        
        Args:
            global_importance: DataFrame with feature importance from SHAP
            top_n: Number of top interventions to return
            
        Returns:
            List of prioritized interventions
        """
        interventions = []
        
        for _, row in global_importance.head(10).iterrows():
            feature = row["feature"]
            importance = row["importance"]
            mean_shap = row.get("mean_shap", 0)
            
            if feature in self.interventions:
                intervention = self.interventions[feature].copy()
                intervention["feature"] = feature
                intervention["business_name"] = self.get_business_name(feature)
                intervention["importance"] = importance
                intervention["direction"] = "risk_driver" if mean_shap > 0 else "protective"
                interventions.append(intervention)
                
        # Sort by importance and priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        interventions.sort(key=lambda x: (priority_order.get(x["priority"], 3), -x["importance"]))
        
        return interventions[:top_n]
        
    def generate_insight_report(
        self,
        global_importance: pd.DataFrame,
        sample_explanations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comprehensive insight report.
        
        Args:
            global_importance: Global feature importance
            sample_explanations: List of individual customer explanations
            
        Returns:
            Comprehensive insight report
        """
        # Top global drivers
        top_drivers = []
        for _, row in global_importance.head(10).iterrows():
            top_drivers.append({
                "feature": row["feature"],
                "business_name": self.get_business_name(row["feature"]),
                "importance": row["importance"],
                "direction": "increases risk" if row.get("mean_shap", 0) > 0 else "decreases risk",
            })
            
        # Top interventions
        top_interventions = self.generate_top_interventions(global_importance)
        
        # Risk distribution from samples
        risk_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for exp in sample_explanations:
            level = exp.get("risk_level", "MEDIUM")
            risk_distribution[level] = risk_distribution.get(level, 0) + 1
            
        return {
            "summary": {
                "total_customers_analyzed": len(sample_explanations),
                "risk_distribution": risk_distribution,
            },
            "top_global_drivers": top_drivers,
            "recommended_interventions": top_interventions,
            "key_findings": self._generate_key_findings(top_drivers, risk_distribution),
        }
        
    def _generate_key_findings(
        self,
        top_drivers: List[Dict],
        risk_distribution: Dict[str, int],
    ) -> List[str]:
        """Generate key business findings."""
        findings = []
        
        # Finding about top driver
        if top_drivers:
            top = top_drivers[0]
            findings.append(
                f"'{top['business_name']}' is the strongest predictor of churn, "
                f"accounting for {top['importance']*100:.1f}% of model importance."
            )
            
        # Finding about contract
        contract_drivers = [d for d in top_drivers if "Contract" in d["feature"]]
        if contract_drivers:
            findings.append(
                "Contract type is a critical factor. Month-to-month customers are at highest risk."
            )
            
        # Finding about risk distribution
        total = sum(risk_distribution.values())
        if total > 0:
            high_pct = risk_distribution.get("HIGH", 0) / total * 100
            if high_pct > 20:
                findings.append(
                    f"WARNING: {high_pct:.0f}% of customers are at HIGH churn risk. "
                    "Immediate intervention recommended."
                )
                
        return findings


def create_customer_risk_profile(
    customer_id: str,
    churn_probability: float,
    shap_explanation: Dict[str, Any],
    business_explanation: Dict[str, Any],
) -> Dict[str, Any]:
    """Create complete customer risk profile.
    
    Args:
        customer_id: Customer identifier
        churn_probability: Model prediction
        shap_explanation: Raw SHAP explanation
        business_explanation: Business-friendly explanation
        
    Returns:
        Complete customer risk profile
    """
    return {
        "customer_id": customer_id,
        "churn_probability": churn_probability,
        "risk_level": business_explanation["risk_level"],
        "summary": business_explanation["narrative_summary"],
        "risk_factors": business_explanation["risk_factors"],
        "protective_factors": business_explanation["protective_factors"],
        "recommended_actions": business_explanation["recommended_actions"],
        "technical_details": {
            "base_probability": shap_explanation.get("base_probability"),
            "shap_values": shap_explanation.get("all_shap_values"),
        },
    }
