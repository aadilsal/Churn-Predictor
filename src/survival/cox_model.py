"""Cox Proportional Hazards model for time-to-churn analysis."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.utils.logging import logger


class CoxSurvivalModel:
    """Cox Proportional Hazards model for churn survival analysis."""
    
    def __init__(
        self,
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
    ):
        """Initialize Cox PH model.
        
        Args:
            penalizer: Regularization strength (helps with convergence)
            l1_ratio: Mix of L1/L2 regularization (0=L2, 1=L1)
        """
        self.cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self._is_fitted = False
        self.feature_names: List[str] = []
        
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "CoxSurvivalModel":
        """Fit Cox PH model.
        
        Args:
            df: DataFrame with features, duration, and event columns
            duration_col: Name of duration column
            event_col: Name of event column
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Cox PH model...")
        
        # Store feature names (exclude duration and event)
        self.feature_names = [c for c in df.columns if c not in [duration_col, event_col]]
        
        self.cph.fit(df, duration_col=duration_col, event_col=event_col)
        self._is_fitted = True
        
        # Log summary
        self._log_model_summary()
        
        return self
        
    def _log_model_summary(self) -> None:
        """Log model fitting summary."""
        n_features = len(self.feature_names)
        c_index = self.cph.concordance_index_
        
        logger.info(f"Cox PH model fitted with {n_features} features")
        logger.info(f"Concordance index: {c_index:.4f}")
        
        # Top hazard ratios
        summary = self.cph.summary
        top_hazards = summary.nlargest(5, "exp(coef)")
        
        logger.info("Top 5 hazard ratios:")
        for idx, row in top_hazards.iterrows():
            logger.info(f"  {idx}: HR={row['exp(coef)']:.3f}")
            
    def check_proportional_hazards(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> Dict[str, Any]:
        """Test proportional hazards assumption.
        
        Args:
            df: Original training data
            duration_col: Duration column
            event_col: Event column
            
        Returns:
            PH test results
        """
        logger.info("Testing proportional hazards assumption...")
        
        try:
            test_results = self.cph.check_assumptions(df, show_plots=False)
            
            # Parse results
            violations = []
            for feature, result in test_results.items():
                if result["p"] < 0.05:
                    violations.append({
                        "feature": feature,
                        "p_value": result["p"],
                        "test_statistic": result["test_statistic"],
                    })
                    
            return {
                "assumption_valid": len(violations) == 0,
                "violations": violations,
                "interpretation": self._interpret_ph_violations(violations),
            }
        except Exception as e:
            logger.warning(f"PH test failed: {e}")
            return {
                "assumption_valid": None,
                "error": str(e),
                "interpretation": "Unable to validate proportional hazards assumption",
            }
            
    def _interpret_ph_violations(self, violations: List[Dict]) -> str:
        """Interpret PH assumption violations."""
        if not violations:
            return "Proportional hazards assumption is satisfied for all features"
        else:
            features = [v["feature"] for v in violations]
            return (
                f"PH assumption violated for: {', '.join(features)}. "
                "Consider stratification or time-varying covariates."
            )
            
    def get_hazard_ratios(self) -> pd.DataFrame:
        """Get hazard ratios with confidence intervals.
        
        Returns:
            DataFrame with HR, CI, and p-values
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        summary = self.cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
        summary.columns = ["hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]
        
        return summary.sort_values("hazard_ratio", ascending=False)
        
    def interpret_hazard_ratios(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Interpret hazard ratios in business language.
        
        Args:
            top_n: Number of top features to interpret
            
        Returns:
            List of interpreted hazard ratios
        """
        hr_df = self.get_hazard_ratios()
        
        interpretations = []
        for feature, row in hr_df.head(top_n).iterrows():
            hr = row["hazard_ratio"]
            
            if hr > 1:
                effect = f"increases churn risk by {(hr-1)*100:.0f}%"
                direction = "risk_factor"
            else:
                effect = f"decreases churn risk by {(1-hr)*100:.0f}%"
                direction = "protective"
                
            interpretations.append({
                "feature": feature,
                "hazard_ratio": float(hr),
                "ci_lower": float(row["hr_ci_lower"]),
                "ci_upper": float(row["hr_ci_upper"]),
                "p_value": float(row["p_value"]),
                "significant": row["p_value"] < 0.05,
                "effect": effect,
                "direction": direction,
            })
            
        return interpretations
        
    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Predict survival function for customers.
        
        Args:
            X: Feature data (must have same columns as training)
            times: Time points to predict (None = auto)
            
        Returns:
            DataFrame with survival probabilities over time
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        survival = self.cph.predict_survival_function(X, times=times)
        return survival
        
    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median survival time for customers.
        
        Args:
            X: Feature data
            
        Returns:
            Array of median survival times
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        return self.cph.predict_median(X).values
        
    def predict_hazard(
        self,
        X: pd.DataFrame,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Predict cumulative hazard for customers.
        
        Args:
            X: Feature data
            times: Time points
            
        Returns:
            Cumulative hazard DataFrame
        """
        return self.cph.predict_cumulative_hazard(X, times=times)
        
    def get_individual_timeline(
        self,
        X: pd.DataFrame,
        index: int = 0,
        times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Get detailed survival timeline for individual.
        
        Args:
            X: Feature data
            index: Customer index
            times: Time points
            
        Returns:
            Individual timeline data
        """
        if times is None:
            times = np.arange(1, 73, 1)  # Monthly for 6 years
            
        customer_X = X.iloc[[index]]
        
        survival = self.predict_survival_function(customer_X, times)
        median = self.predict_median_survival(customer_X)[0]
        
        # Find risk windows
        risk_windows = []
        survival_values = survival.values.flatten()
        
        for t, s in zip(times, survival_values):
            if 0.4 <= s <= 0.6:
                risk_windows.append(t)
                
        return {
            "times": times.tolist(),
            "survival_probabilities": survival_values.tolist(),
            "median_survival": float(median) if not np.isinf(median) else None,
            "high_risk_window": [min(risk_windows), max(risk_windows)] if risk_windows else None,
            "risk_level_at_12mo": self._get_risk_level(survival_values[11] if len(survival_values) > 11 else 1.0),
        }
        
    def _get_risk_level(self, survival_prob: float) -> str:
        """Categorize survival probability."""
        if survival_prob < 0.3:
            return "CRITICAL"
        elif survival_prob < 0.5:
            return "HIGH"
        elif survival_prob < 0.7:
            return "MEDIUM"
        else:
            return "LOW"
            
    def plot_hazard_ratios(
        self,
        top_n: int = 15,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot hazard ratios forest plot.
        
        Args:
            top_n: Number of features to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hr_df = self.get_hazard_ratios().head(top_n)
        
        y_pos = np.arange(len(hr_df))
        
        # Plot horizontal bars
        ax.barh(y_pos, hr_df["hazard_ratio"] - 1, align="center", 
                color=["red" if hr > 1 else "green" for hr in hr_df["hazard_ratio"]], alpha=0.6)
        
        # Add reference line at HR=1
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        
        # Add error bars for CI
        xerr = np.array([
            hr_df["hazard_ratio"] - hr_df["hr_ci_lower"],
            hr_df["hr_ci_upper"] - hr_df["hazard_ratio"]
        ])
        ax.errorbar(hr_df["hazard_ratio"] - 1, y_pos, xerr=xerr, fmt="none", color="black", capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hr_df.index)
        ax.set_xlabel("Hazard Ratio (centered at 1)")
        ax.set_title("Cox PH Hazard Ratios")
        ax.grid(True, alpha=0.3, axis="x")
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved hazard ratio plot to {save_path}")
            
        return fig
        
    def plot_survival_timeline(
        self,
        X: pd.DataFrame,
        indices: List[int] = [0, 1, 2],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot survival timelines for multiple customers.
        
        Args:
            X: Feature data
            indices: Customer indices to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = np.arange(1, 73, 1)
        
        for i, idx in enumerate(indices):
            timeline = self.get_individual_timeline(X, idx, times)
            ax.plot(times, timeline["survival_probabilities"], 
                   label=f"Customer {idx}", linewidth=2)
            
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% survival")
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Individual Customer Survival Timelines")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved survival timeline to {save_path}")
            
        return fig
        
    def save(self, path: Path) -> None:
        """Save fitted model."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self.cph, f)
            
        logger.info(f"Saved Cox model to {path}")
        
    @classmethod
    def load(cls, path: Path) -> "CoxSurvivalModel":
        """Load fitted model."""
        import pickle
        
        with open(path, "rb") as f:
            cph = pickle.load(f)
            
        instance = cls()
        instance.cph = cph
        instance._is_fitted = True
        
        return instance
