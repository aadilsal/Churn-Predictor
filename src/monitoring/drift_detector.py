"""Drift detection using Evidently for data and prediction monitoring."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.logging import logger

# Try to import evidently
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Some drift detection features unavailable.")


# Drift thresholds with rationale
DRIFT_THRESHOLDS = {
    "dataset_drift": {
        "threshold": 0.5,  # 50% of features drifted
        "rationale": "If more than half of features have drifted, model reliability is questionable",
    },
    "feature_drift": {
        "threshold": 0.05,  # p-value threshold
        "rationale": "Statistical significance at 5% level indicates meaningful distribution shift",
    },
    "prediction_drift": {
        "mean_shift": 0.10,  # 10% shift in mean probability
        "std_shift": 0.15,  # 15% change in std
        "rationale": "Significant changes in prediction distribution signal model behavior change",
    },
}


class DriftDetector:
    """Detect data and prediction drift using Evidently."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
    ):
        """Initialize drift detector.
        
        Args:
            reference_data: Reference (training) data
            feature_columns: Columns to monitor for drift
            categorical_columns: Categorical feature columns
            numerical_columns: Numerical feature columns
        """
        self.reference_data = reference_data
        
        # Auto-detect column types if not provided
        if feature_columns is None:
            feature_columns = [c for c in reference_data.columns 
                             if c not in ["customerID", "Churn"]]
        self.feature_columns = feature_columns
        
        if categorical_columns is None:
            categorical_columns = reference_data[feature_columns].select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        self.categorical_columns = categorical_columns
        
        if numerical_columns is None:
            numerical_columns = reference_data[feature_columns].select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
        self.numerical_columns = numerical_columns
        
        # Column mapping for Evidently
        self.column_mapping = None
        if EVIDENTLY_AVAILABLE:
            self.column_mapping = ColumnMapping(
                numerical_features=self.numerical_columns,
                categorical_features=self.categorical_columns,
            )
            
        self._drift_history: List[Dict] = []
        
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Detect data drift between reference and current data.
        
        Args:
            current_data: Current inference data
            save_report: Whether to save HTML report
            report_path: Path for HTML report
            
        Returns:
            Drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            return self._fallback_drift_detection(current_data)
            
        logger.info("Running Evidently data drift detection...")
        
        # Prepare data
        ref = self.reference_data[self.feature_columns].copy()
        cur = current_data[self.feature_columns].copy()
        
        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        report.run(reference_data=ref, current_data=cur, column_mapping=self.column_mapping)
        
        # Extract results
        results = report.as_dict()
        
        # Parse drift results
        dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
        drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]
        
        # Feature-level drift
        feature_drift = {}
        drift_table = results["metrics"][1]["result"]["drift_by_columns"]
        for feature, info in drift_table.items():
            feature_drift[feature] = {
                "drifted": info.get("drift_detected", False),
                "drift_score": info.get("drift_score", 0),
                "stattest": info.get("stattest_name", "unknown"),
            }
            
        drifted_features = [f for f, v in feature_drift.items() if v["drifted"]]
        
        drift_result = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": dataset_drift,
            "drift_share": drift_share,
            "n_drifted_features": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_drift": feature_drift,
            "threshold_exceeded": drift_share > DRIFT_THRESHOLDS["dataset_drift"]["threshold"],
            "samples_reference": len(ref),
            "samples_current": len(cur),
        }
        
        # Save report
        if save_report:
            report_path = report_path or Path("reports/drift_report.html")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(report_path))
            drift_result["report_path"] = str(report_path)
            logger.info(f"Saved drift report to {report_path}")
            
        # Store in history
        self._drift_history.append(drift_result)
        
        return drift_result
        
    def _fallback_drift_detection(
        self,
        current_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Fallback drift detection without Evidently.
        
        Uses simple statistical tests.
        """
        logger.info("Using fallback drift detection (Evidently not available)")
        
        ref = self.reference_data[self.feature_columns]
        cur = current_data[self.feature_columns]
        
        feature_drift = {}
        drifted_count = 0
        
        for col in self.numerical_columns:
            if col in cur.columns:
                # Simple mean/std comparison
                ref_mean, ref_std = ref[col].mean(), ref[col].std()
                cur_mean, cur_std = cur[col].mean(), cur[col].std()
                
                mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
                drifted = mean_shift > 2  # Simple 2-sigma rule
                
                feature_drift[col] = {
                    "drifted": drifted,
                    "drift_score": mean_shift,
                    "stattest": "mean_shift",
                }
                if drifted:
                    drifted_count += 1
                    
        drift_share = drifted_count / len(self.numerical_columns) if self.numerical_columns else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": drift_share > 0.5,
            "drift_share": drift_share,
            "n_drifted_features": drifted_count,
            "drifted_features": [f for f, v in feature_drift.items() if v["drifted"]],
            "feature_drift": feature_drift,
            "threshold_exceeded": drift_share > DRIFT_THRESHOLDS["dataset_drift"]["threshold"],
            "method": "fallback",
        }
        
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Detect drift in prediction distributions.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
            
        Returns:
            Prediction drift results
        """
        ref_mean = reference_predictions.mean()
        cur_mean = current_predictions.mean()
        
        ref_std = reference_predictions.std()
        cur_std = current_predictions.std()
        
        mean_shift = abs(cur_mean - ref_mean)
        std_shift = abs(cur_std - ref_std)
        
        # Check thresholds
        mean_drifted = mean_shift > DRIFT_THRESHOLDS["prediction_drift"]["mean_shift"]
        std_drifted = std_shift > DRIFT_THRESHOLDS["prediction_drift"]["std_shift"]
        
        # Risk category distribution
        ref_high = (reference_predictions >= 0.7).mean()
        cur_high = (current_predictions >= 0.7).mean()
        high_risk_shift = abs(cur_high - ref_high)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "reference_mean": float(ref_mean),
            "current_mean": float(cur_mean),
            "mean_shift": float(mean_shift),
            "mean_drifted": mean_drifted,
            "reference_std": float(ref_std),
            "current_std": float(cur_std),
            "std_shift": float(std_shift),
            "std_drifted": std_drifted,
            "high_risk_share_reference": float(ref_high),
            "high_risk_share_current": float(cur_high),
            "high_risk_shift": float(high_risk_shift),
            "prediction_drift_detected": mean_drifted or std_drifted,
            "business_impact": self._assess_business_impact(mean_shift, high_risk_shift),
        }
        
    def _assess_business_impact(
        self,
        mean_shift: float,
        high_risk_shift: float,
    ) -> str:
        """Assess business impact of prediction drift."""
        if mean_shift > 0.2 or high_risk_shift > 0.15:
            return "HIGH - Significant change in churn predictions may affect retention strategy"
        elif mean_shift > 0.1 or high_risk_shift > 0.08:
            return "MEDIUM - Moderate shift in predictions, monitor closely"
        else:
            return "LOW - Minor prediction variations within normal range"
            
    def get_drift_history(self) -> List[Dict]:
        """Get history of drift detection runs."""
        return self._drift_history
        
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift history."""
        if not self._drift_history:
            return {"message": "No drift history available"}
            
        recent = self._drift_history[-1]
        
        drift_events = sum(1 for d in self._drift_history if d.get("dataset_drift_detected", False))
        
        return {
            "total_checks": len(self._drift_history),
            "drift_events": drift_events,
            "drift_rate": drift_events / len(self._drift_history),
            "last_check": recent["timestamp"],
            "last_drift_detected": recent.get("dataset_drift_detected", False),
        }
