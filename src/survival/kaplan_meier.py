"""Kaplan-Meier survival analysis for churn time dynamics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from src.utils.logging import logger


class KaplanMeierAnalyzer:
    """Kaplan-Meier survival curve analysis for customer churn."""
    
    def __init__(self):
        """Initialize the KM analyzer."""
        self.kmf = KaplanMeierFitter()
        self.fitted_curves: Dict[str, KaplanMeierFitter] = {}
        
    def fit(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        label: str = "Overall",
    ) -> "KaplanMeierAnalyzer":
        """Fit Kaplan-Meier curve.
        
        Args:
            durations: Time durations
            events: Event indicators (1=churned, 0=censored)
            label: Label for the curve
            
        Returns:
            Self for method chaining
        """
        self.kmf.fit(durations, events, label=label)
        self.fitted_curves[label] = self.kmf
        
        logger.info(f"Fitted KM curve for '{label}': "
                   f"median survival = {self.kmf.median_survival_time_:.1f} months")
        
        return self
        
    def fit_by_segment(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        segment_labels: np.ndarray,
    ) -> Dict[str, KaplanMeierFitter]:
        """Fit separate KM curves for each segment.
        
        Args:
            durations: Time durations
            events: Event indicators
            segment_labels: Segment identifiers for each observation
            
        Returns:
            Dictionary of fitted KM models by segment
        """
        unique_segments = np.unique(segment_labels)
        
        for segment in unique_segments:
            mask = segment_labels == segment
            kmf = KaplanMeierFitter()
            kmf.fit(durations[mask], events[mask], label=str(segment))
            self.fitted_curves[str(segment)] = kmf
            
            logger.info(f"Segment '{segment}': "
                       f"n={mask.sum()}, "
                       f"median survival = {kmf.median_survival_time_:.1f}")
            
        return self.fitted_curves
        
    def get_survival_probability(
        self,
        time_point: float,
        label: str = "Overall",
    ) -> float:
        """Get survival probability at specific time.
        
        Args:
            time_point: Time to evaluate
            label: Curve label
            
        Returns:
            Survival probability
        """
        if label not in self.fitted_curves:
            raise ValueError(f"No curve fitted for '{label}'")
            
        kmf = self.fitted_curves[label]
        return float(kmf.predict(time_point))
        
    def get_survival_timeline(
        self,
        label: str = "Overall",
    ) -> pd.DataFrame:
        """Get full survival timeline.
        
        Args:
            label: Curve label
            
        Returns:
            DataFrame with time, survival probability, and confidence intervals
        """
        if label not in self.fitted_curves:
            raise ValueError(f"No curve fitted for '{label}'")
            
        kmf = self.fitted_curves[label]
        
        return pd.DataFrame({
            "time": kmf.survival_function_.index,
            "survival_probability": kmf.survival_function_.values.flatten(),
            "ci_lower": kmf.confidence_interval_survival_function_.iloc[:, 0].values,
            "ci_upper": kmf.confidence_interval_survival_function_.iloc[:, 1].values,
        })
        
    def identify_risk_inflection_points(
        self,
        label: str = "Overall",
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Identify time points where churn risk accelerates.
        
        Args:
            label: Curve label
            threshold: Minimum drop in survival probability to flag
            
        Returns:
            List of inflection points with details
        """
        timeline = self.get_survival_timeline(label)
        
        # Calculate survival drops between consecutive time points
        timeline["survival_drop"] = -timeline["survival_probability"].diff()
        
        inflection_points = []
        for _, row in timeline.iterrows():
            if row["survival_drop"] > threshold:
                inflection_points.append({
                    "time": row["time"],
                    "survival_probability": row["survival_probability"],
                    "drop_magnitude": row["survival_drop"],
                    "interpretation": f"Sharp {row['survival_drop']*100:.1f}% churn surge at month {row['time']:.0f}",
                })
                
        return inflection_points
        
    def compare_segments(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        segment_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare survival between segments using log-rank test.
        
        Args:
            durations: Time durations
            events: Event indicators
            segment_labels: Segment identifiers
            
        Returns:
            Comparison results with statistical test
        """
        unique_segments = np.unique(segment_labels)
        
        if len(unique_segments) == 2:
            # Two-sample log-rank test
            mask1 = segment_labels == unique_segments[0]
            mask2 = segment_labels == unique_segments[1]
            
            result = logrank_test(
                durations[mask1], durations[mask2],
                events[mask1], events[mask2]
            )
            
            return {
                "test": "log-rank",
                "segments": unique_segments.tolist(),
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "significant": result.p_value < 0.05,
                "interpretation": self._interpret_logrank(result.p_value, unique_segments),
            }
        else:
            # Multivariate log-rank test
            result = multivariate_logrank_test(durations, segment_labels, events)
            
            return {
                "test": "multivariate log-rank",
                "segments": unique_segments.tolist(),
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "significant": result.p_value < 0.05,
                "interpretation": self._interpret_logrank(result.p_value, unique_segments),
            }
            
    def _interpret_logrank(
        self,
        p_value: float,
        segments: np.ndarray,
    ) -> str:
        """Interpret log-rank test result."""
        if p_value < 0.001:
            return f"Highly significant difference in survival between segments (p<0.001)"
        elif p_value < 0.05:
            return f"Significant difference in survival between segments (p={p_value:.3f})"
        else:
            return f"No significant difference in survival between segments (p={p_value:.3f})"
            
    def plot_survival_curve(
        self,
        labels: Optional[List[str]] = None,
        title: str = "Kaplan-Meier Survival Curves",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot survival curves.
        
        Args:
            labels: Curves to plot (None = all)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = labels or list(self.fitted_curves.keys())
        
        for label in labels:
            if label in self.fitted_curves:
                self.fitted_curves[label].plot_survival_function(ax=ax)
                
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        # Add median survival line
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved KM plot to {save_path}")
            
        return fig
        
    def get_segment_summary(self) -> pd.DataFrame:
        """Get summary of all fitted segments."""
        rows = []
        for label, kmf in self.fitted_curves.items():
            rows.append({
                "segment": label,
                "median_survival": kmf.median_survival_time_,
                "survival_at_6mo": float(kmf.predict(6)),
                "survival_at_12mo": float(kmf.predict(12)),
                "survival_at_24mo": float(kmf.predict(24)),
            })
            
        return pd.DataFrame(rows).sort_values("median_survival", ascending=False)
        
    def generate_business_insights(self) -> List[str]:
        """Generate business-friendly insights from KM analysis."""
        insights = []
        
        summary = self.get_segment_summary()
        
        if len(summary) > 1:
            best = summary.iloc[0]
            worst = summary.iloc[-1]
            
            insights.append(
                f"'{best['segment']}' customers have the longest survival "
                f"(median: {best['median_survival']:.0f} months)"
            )
            insights.append(
                f"'{worst['segment']}' customers churn fastest "
                f"(median: {worst['median_survival']:.0f} months)"
            )
            
            # Early churn risk
            high_early_churn = summary[summary["survival_at_6mo"] < 0.7]
            if len(high_early_churn) > 0:
                segments = high_early_churn["segment"].tolist()
                insights.append(
                    f"HIGH RISK in first 6 months: {', '.join(segments)}"
                )
                
        return insights
