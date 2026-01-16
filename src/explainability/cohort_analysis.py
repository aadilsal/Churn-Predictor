"""Cohort-level analysis of churn drivers."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import logger


class CohortAnalyzer:
    """Analyze churn patterns across customer cohorts."""
    
    def __init__(
        self,
        feature_names: List[str],
    ):
        """Initialize cohort analyzer.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
    def analyze_by_segment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shap_values: np.ndarray,
        segment_feature: str,
        segment_values: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze churn drivers by customer segment.
        
        Args:
            X: Feature data
            y: True labels
            shap_values: SHAP values for all samples
            segment_feature: Feature to segment by
            segment_values: Specific values to analyze (optional)
            
        Returns:
            Segment analysis results
        """
        # Find feature index
        if segment_feature not in self.feature_names:
            logger.warning(f"Feature {segment_feature} not found")
            return {}
            
        feat_idx = self.feature_names.index(segment_feature)
        
        # Get unique segment values
        unique_values = np.unique(X[:, feat_idx])
        if segment_values:
            unique_values = [v for v in unique_values if v in segment_values]
            
        results = {}
        
        for value in unique_values:
            mask = X[:, feat_idx] == value
            segment_X = X[mask]
            segment_y = y[mask]
            segment_shap = shap_values[mask]
            
            # Calculate segment statistics
            churn_rate = segment_y.mean()
            
            # Top drivers for this segment
            mean_abs_shap = np.abs(segment_shap).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
            
            top_drivers = []
            for idx in top_indices:
                top_drivers.append({
                    "feature": self.feature_names[idx],
                    "importance": float(mean_abs_shap[idx]),
                    "mean_impact": float(segment_shap[:, idx].mean()),
                })
                
            segment_name = f"{segment_feature}={value}"
            results[segment_name] = {
                "segment_size": int(mask.sum()),
                "churn_rate": float(churn_rate),
                "top_drivers": top_drivers,
            }
            
        return results
        
    def compare_cohorts(
        self,
        cohort_results: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Compare churn patterns across cohorts.
        
        Args:
            cohort_results: Results from analyze_by_segment
            
        Returns:
            Cohort comparison with insights
        """
        cohort_names = list(cohort_results.keys())
        
        # Build comparison table
        comparison = []
        for name, data in cohort_results.items():
            comparison.append({
                "cohort": name,
                "size": data["segment_size"],
                "churn_rate": data["churn_rate"],
                "top_driver": data["top_drivers"][0]["feature"] if data["top_drivers"] else None,
            })
            
        df = pd.DataFrame(comparison)
        
        # Find highest and lowest risk cohorts
        highest_risk = df.loc[df["churn_rate"].idxmax()]
        lowest_risk = df.loc[df["churn_rate"].idxmin()]
        
        # Identify common vs divergent drivers
        all_top_drivers = {}
        for name, data in cohort_results.items():
            for driver in data["top_drivers"][:3]:
                feat = driver["feature"]
                if feat not in all_top_drivers:
                    all_top_drivers[feat] = []
                all_top_drivers[feat].append(name)
                
        common_drivers = [f for f, cohorts in all_top_drivers.items() 
                         if len(cohorts) == len(cohort_names)]
        
        return {
            "comparison_table": df.to_dict(orient="records"),
            "highest_risk_cohort": {
                "name": highest_risk["cohort"],
                "churn_rate": highest_risk["churn_rate"],
            },
            "lowest_risk_cohort": {
                "name": lowest_risk["cohort"],
                "churn_rate": lowest_risk["churn_rate"],
            },
            "common_drivers": common_drivers,
            "insights": self._generate_cohort_insights(df, common_drivers),
        }
        
    def _generate_cohort_insights(
        self,
        comparison_df: pd.DataFrame,
        common_drivers: List[str],
    ) -> List[str]:
        """Generate insights from cohort comparison."""
        insights = []
        
        # Churn rate spread
        rate_spread = comparison_df["churn_rate"].max() - comparison_df["churn_rate"].min()
        if rate_spread > 0.2:
            insights.append(
                f"Large variation in churn rates across cohorts ({rate_spread*100:.0f}pp spread). "
                "Targeted interventions by segment recommended."
            )
            
        # Common drivers
        if common_drivers:
            insights.append(
                f"Universal churn drivers across all cohorts: {', '.join(common_drivers[:3])}. "
                "Address these for broad impact."
            )
            
        return insights
        
    def identify_risk_patterns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shap_values: np.ndarray,
        risk_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Identify common patterns among high-risk customers.
        
        Args:
            X: Feature data
            y: True labels or predicted probabilities
            shap_values: SHAP values
            risk_threshold: Threshold for high-risk classification
            
        Returns:
            Risk pattern analysis
        """
        high_risk_mask = y >= risk_threshold
        low_risk_mask = y < risk_threshold
        
        high_risk_shap = shap_values[high_risk_mask]
        low_risk_shap = shap_values[low_risk_mask]
        
        # Compare mean SHAP values
        high_risk_mean = high_risk_shap.mean(axis=0)
        low_risk_mean = low_risk_shap.mean(axis=0)
        
        differential = high_risk_mean - low_risk_mean
        
        # Find features that differentiate high from low risk
        diff_ranking = np.argsort(np.abs(differential))[::-1]
        
        differentiating_features = []
        for idx in diff_ranking[:10]:
            differentiating_features.append({
                "feature": self.feature_names[idx],
                "differential": float(differential[idx]),
                "high_risk_impact": float(high_risk_mean[idx]),
                "low_risk_impact": float(low_risk_mean[idx]),
                "interpretation": "more positive in high-risk" if differential[idx] > 0 
                                 else "more negative in high-risk",
            })
            
        return {
            "high_risk_count": int(high_risk_mask.sum()),
            "low_risk_count": int(low_risk_mask.sum()),
            "differentiating_features": differentiating_features,
            "patterns": self._summarize_patterns(differentiating_features),
        }
        
    def _summarize_patterns(
        self,
        differentiating_features: List[Dict],
    ) -> List[str]:
        """Summarize risk patterns in plain language."""
        patterns = []
        
        for feat in differentiating_features[:5]:
            name = feat["feature"]
            if feat["differential"] > 0.05:
                patterns.append(f"High-risk customers show stronger '{name}' as a churn driver")
            elif feat["differential"] < -0.05:
                patterns.append(f"Low-risk customers benefit more from '{name}' as protection")
                
        return patterns
        
    def analyze_tenure_cohorts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shap_values: np.ndarray,
        tenure_feature: str = "tenure",
        bins: List[int] = [0, 6, 12, 24, 48, 100],
        labels: List[str] = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
    ) -> Dict[str, Any]:
        """Analyze churn patterns by customer tenure.
        
        Args:
            X: Feature data
            y: Labels
            shap_values: SHAP values
            tenure_feature: Name of tenure feature
            bins: Tenure bucket boundaries
            labels: Labels for buckets
            
        Returns:
            Tenure cohort analysis
        """
        if tenure_feature not in self.feature_names:
            logger.warning(f"Tenure feature {tenure_feature} not found")
            return {}
            
        tenure_idx = self.feature_names.index(tenure_feature)
        tenure_values = X[:, tenure_idx]
        
        # For normalized tenure, denormalize approximately
        if tenure_values.max() < 10:  # Likely normalized
            tenure_values = tenure_values * 72  # Approximate max tenure
            
        cohort_results = {}
        
        for i in range(len(bins) - 1):
            mask = (tenure_values >= bins[i]) & (tenure_values < bins[i+1])
            if mask.sum() == 0:
                continue
                
            label = labels[i]
            segment_y = y[mask]
            segment_shap = shap_values[mask]
            
            mean_abs_shap = np.abs(segment_shap).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
            
            top_drivers = []
            for idx in top_indices:
                top_drivers.append({
                    "feature": self.feature_names[idx],
                    "importance": float(mean_abs_shap[idx]),
                })
                
            cohort_results[label] = {
                "segment_size": int(mask.sum()),
                "churn_rate": float(segment_y.mean()),
                "top_drivers": top_drivers,
            }
            
        return self.compare_cohorts(cohort_results)
