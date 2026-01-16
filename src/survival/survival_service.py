"""Unified survival analysis service with intervention timing recommendations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.survival.data_preparation import (
    add_tenure_buckets,
    create_survival_dataset,
    get_survival_features,
    get_survival_summary,
    prepare_survival_data,
)
from src.survival.kaplan_meier import KaplanMeierAnalyzer
from src.survival.cox_model import CoxSurvivalModel
from src.utils.logging import logger


class SurvivalService:
    """Unified service for survival analysis and intervention timing."""
    
    def __init__(self):
        """Initialize survival service."""
        self.km_analyzer = KaplanMeierAnalyzer()
        self.cox_model = CoxSurvivalModel()
        self._fitted = False
        
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "tenure",
        event_col: str = "Churn",
    ) -> "SurvivalService":
        """Fit survival models.
        
        Args:
            df: Raw data with duration and event columns
            duration_col: Duration column name
            event_col: Event column name
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting survival analysis models...")
        
        # Prepare data
        survival_df = prepare_survival_data(df, duration_col, event_col)
        survival_df = add_tenure_buckets(survival_df)
        
        durations = survival_df["duration"].values
        events = survival_df["event"].values
        
        # Fit overall KM curve
        self.km_analyzer.fit(durations, events, label="Overall")
        
        # Fit KM curves by contract type
        if "Contract" in df.columns:
            contract_labels = df["Contract"].values
            self.km_analyzer.fit_by_segment(durations, events, contract_labels)
            
        # Prepare features for Cox model
        features_df, feature_names = get_survival_features(survival_df)
        cox_df = features_df.copy()
        cox_df["duration"] = survival_df["duration"]
        cox_df["event"] = survival_df["event"]
        
        # Fit Cox model
        self.cox_model.fit(cox_df, duration_col="duration", event_col="event")
        
        self._fitted = True
        logger.info("Survival models fitted successfully")
        
        return self
        
    def get_intervention_timing(
        self,
        segment: str = "Overall",
    ) -> Dict[str, Any]:
        """Get recommended intervention timing for a segment.
        
        Args:
            segment: Segment to analyze
            
        Returns:
            Intervention timing recommendations
        """
        timeline = self.km_analyzer.get_survival_timeline(segment)
        
        # Find key milestones
        milestones = {
            "90pct_survival": None,
            "75pct_survival": None,
            "50pct_survival": None,
            "25pct_survival": None,
        }
        
        for _, row in timeline.iterrows():
            prob = row["survival_probability"]
            time = row["time"]
            
            if milestones["90pct_survival"] is None and prob <= 0.90:
                milestones["90pct_survival"] = time
            if milestones["75pct_survival"] is None and prob <= 0.75:
                milestones["75pct_survival"] = time
            if milestones["50pct_survival"] is None and prob <= 0.50:
                milestones["50pct_survival"] = time
            if milestones["25pct_survival"] is None and prob <= 0.25:
                milestones["25pct_survival"] = time
                
        # Determine intervention windows
        recommendations = []
        
        early_window = milestones["90pct_survival"]
        if early_window:
            recommendations.append({
                "timing": "early",
                "trigger_month": int(early_window),
                "action": "Proactive engagement survey",
                "urgency": "LOW",
                "description": f"Begin engagement at month {int(early_window)} before 10% churn threshold",
            })
            
        medium_window = milestones["75pct_survival"]
        if medium_window:
            recommendations.append({
                "timing": "medium",
                "trigger_month": int(medium_window),
                "action": "Retention offer or contract discussion",
                "urgency": "MEDIUM",
                "description": f"Present retention offer by month {int(medium_window)} (25% attrition point)",
            })
            
        late_window = milestones["50pct_survival"]
        if late_window:
            recommendations.append({
                "timing": "late",
                "trigger_month": int(late_window),
                "action": "Escalated retention with significant incentive",
                "urgency": "HIGH",
                "description": f"High-value intervention needed by month {int(late_window)} (50% attrition)",
            })
            
        return {
            "segment": segment,
            "milestones": milestones,
            "recommendations": recommendations,
            "summary": self._generate_timing_summary(milestones, recommendations),
        }
        
    def _generate_timing_summary(
        self,
        milestones: Dict,
        recommendations: List[Dict],
    ) -> str:
        """Generate timing summary text."""
        if milestones["50pct_survival"]:
            return (
                f"Median customer lifetime is {milestones['50pct_survival']:.0f} months. "
                f"Begin proactive engagement at month {milestones['90pct_survival'] or 1:.0f}."
            )
        return "Insufficient data to determine milestone timing."
        
    def get_customer_risk_timeline(
        self,
        customer_features: pd.DataFrame,
        customer_id: Optional[str] = None,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Get detailed risk timeline for a customer.
        
        Args:
            customer_features: Feature data (survival-preprocessed)
            customer_id: Optional customer ID
            index: Customer index in dataframe
            
        Returns:
            Customer risk timeline
        """
        timeline = self.cox_model.get_individual_timeline(customer_features, index)
        
        # Add intervention recommendations
        survival_probs = timeline["survival_probabilities"]
        times = timeline["times"]
        
        interventions = []
        for i, (t, s) in enumerate(zip(times, survival_probs)):
            if i > 0 and survival_probs[i-1] >= 0.9 > s:
                interventions.append({
                    "time": t,
                    "trigger": "90% threshold crossed",
                    "action": "Begin engagement",
                })
            if i > 0 and survival_probs[i-1] >= 0.7 > s:
                interventions.append({
                    "time": t,
                    "trigger": "70% threshold crossed",
                    "action": "Present retention offer",
                })
            if i > 0 and survival_probs[i-1] >= 0.5 > s:
                interventions.append({
                    "time": t,
                    "trigger": "50% threshold crossed",
                    "action": "Escalated intervention",
                })
                
        return {
            "customer_id": customer_id or f"customer_{index}",
            "timeline": timeline,
            "recommended_interventions": interventions,
            "current_risk_level": timeline["risk_level_at_12mo"],
        }
        
    def compare_to_churn_probability(
        self,
        churn_probabilities: np.ndarray,
        median_survival_times: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare classification churn probability with survival time.
        
        Args:
            churn_probabilities: From classification model
            median_survival_times: From Cox model
            
        Returns:
            Comparison analysis
        """
        # Segment by churn probability
        high_prob = churn_probabilities >= 0.7
        medium_prob = (churn_probabilities >= 0.3) & (churn_probabilities < 0.7)
        low_prob = churn_probabilities < 0.3
        
        comparison = {
            "high_churn_prob": {
                "count": int(high_prob.sum()),
                "avg_median_survival": float(np.nanmean(median_survival_times[high_prob])),
            },
            "medium_churn_prob": {
                "count": int(medium_prob.sum()),
                "avg_median_survival": float(np.nanmean(median_survival_times[medium_prob])),
            },
            "low_churn_prob": {
                "count": int(low_prob.sum()),
                "avg_median_survival": float(np.nanmean(median_survival_times[low_prob])),
            },
        }
        
        # Correlation
        valid_mask = ~np.isinf(median_survival_times) & ~np.isnan(median_survival_times)
        if valid_mask.sum() > 10:
            correlation = np.corrcoef(
                churn_probabilities[valid_mask],
                median_survival_times[valid_mask]
            )[0, 1]
        else:
            correlation = None
            
        comparison["correlation"] = float(correlation) if correlation else None
        comparison["interpretation"] = self._interpret_correlation(correlation)
        
        return comparison
        
    def _interpret_correlation(self, correlation: Optional[float]) -> str:
        """Interpret correlation between churn prob and survival time."""
        if correlation is None:
            return "Insufficient data for correlation analysis"
        if correlation < -0.5:
            return "Strong negative correlation: high churn probability customers have shorter survival times (expected)"
        elif correlation < -0.2:
            return "Moderate negative correlation: churn probability and survival time are meaningfully related"
        else:
            return "Weak correlation: survival analysis provides complementary insights to churn probability"
            
    def generate_report(
        self,
        df: pd.DataFrame,
        output_dir: Path = Path("reports"),
    ) -> Dict[str, Any]:
        """Generate comprehensive survival analysis report.
        
        Args:
            df: Raw data
            output_dir: Output directory for plots and report
            
        Returns:
            Complete report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        survival_df = prepare_survival_data(df)
        summary = get_survival_summary(survival_df)
        
        # KM insights
        km_insights = self.km_analyzer.generate_business_insights()
        km_summary = self.km_analyzer.get_segment_summary()
        
        # Cox hazard ratios
        hazard_interpretations = self.cox_model.interpret_hazard_ratios(top_n=10)
        
        # Intervention timing
        intervention_timing = {}
        for segment in self.km_analyzer.fitted_curves.keys():
            intervention_timing[segment] = self.get_intervention_timing(segment)
            
        # Generate plots
        self.km_analyzer.plot_survival_curve(
            save_path=output_dir / "km_survival_curves.png"
        )
        self.cox_model.plot_hazard_ratios(
            save_path=output_dir / "cox_hazard_ratios.png"
        )
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_summary": summary,
            "kaplan_meier": {
                "segment_summary": km_summary.to_dict(orient="records"),
                "insights": km_insights,
            },
            "cox_model": {
                "concordance_index": float(self.cox_model.cph.concordance_index_),
                "hazard_ratios": hazard_interpretations,
            },
            "intervention_timing": intervention_timing,
            "key_findings": self._generate_key_findings(
                km_summary, hazard_interpretations, intervention_timing
            ),
        }
        
        # Save report
        with open(output_dir / "survival_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Saved survival analysis report to {output_dir}")
        
        return report
        
    def _generate_key_findings(
        self,
        km_summary: pd.DataFrame,
        hazard_ratios: List[Dict],
        intervention_timing: Dict,
    ) -> List[str]:
        """Generate key business findings."""
        findings = []
        
        # Median survival
        if "Overall" in intervention_timing:
            overall = intervention_timing["Overall"]
            if overall["milestones"]["50pct_survival"]:
                findings.append(
                    f"Median customer lifetime is {overall['milestones']['50pct_survival']:.0f} months"
                )
                
        # Segment differences
        if len(km_summary) > 1:
            best = km_summary.iloc[0]
            worst = km_summary.iloc[-1]
            diff = best["median_survival"] - worst["median_survival"]
            if diff > 6:
                findings.append(
                    f"'{best['segment']}' customers survive {diff:.0f} months longer than '{worst['segment']}'"
                )
                
        # Top hazard
        if hazard_ratios:
            top_risk = [h for h in hazard_ratios if h["direction"] == "risk_factor"]
            if top_risk:
                findings.append(
                    f"Highest risk factor: {top_risk[0]['feature']} "
                    f"(HR={top_risk[0]['hazard_ratio']:.2f})"
                )
                
        return findings


def run_survival_pipeline(
    data_path: Path = Path("data/processed/telco_churn_processed.csv"),
    output_dir: Path = Path("reports"),
) -> Dict[str, Any]:
    """Run complete survival analysis pipeline.
    
    Args:
        data_path: Path to processed data
        output_dir: Output directory
        
    Returns:
        Pipeline results
    """
    logger.info("Starting survival analysis pipeline...")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Initialize and fit service
    service = SurvivalService()
    service.fit(df)
    
    # Generate report
    report = service.generate_report(df, output_dir)
    
    logger.info("Survival analysis pipeline complete!")
    
    return report
