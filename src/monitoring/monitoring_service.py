"""Unified monitoring service orchestrating all monitoring components."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.monitoring.data_pipeline import MonitoringDataPipeline
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.alerts import AlertManager, AlertType, AlertSeverity
from src.monitoring.retraining import RetrainingTrigger
from src.utils.logging import logger


class MonitoringService:
    """Unified monitoring service for the churn prediction system."""
    
    def __init__(
        self,
        reference_data_path: Path = Path("data/processed/telco_churn_processed.csv"),
        monitoring_dir: Path = Path("data/monitoring"),
        baseline_metrics: Optional[Dict[str, float]] = None,
    ):
        """Initialize monitoring service.
        
        Args:
            reference_data_path: Path to reference data
            monitoring_dir: Directory for monitoring data
            baseline_metrics: Baseline model metrics
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_pipeline = MonitoringDataPipeline(
            storage_dir=monitoring_dir,
            reference_path=reference_data_path,
        )
        
        reference_data = self.data_pipeline.load_reference_data()
        self.drift_detector = DriftDetector(reference_data=reference_data)
        
        self.performance_monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics or {"roc_auc": 0.84, "brier_score": 0.16},
            storage_path=monitoring_dir / "performance_log.json",
        )
        
        self.alert_manager = AlertManager(
            storage_path=monitoring_dir / "alerts.json",
        )
        
        self.retraining_trigger = RetrainingTrigger(
            storage_path=monitoring_dir / "retraining_log.json",
        )
        
        self._consecutive_alerts = 0
        
    def run_drift_check(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive drift detection.
        
        Args:
            current_data: Current inference data
            save_report: Whether to save HTML report
            
        Returns:
            Drift check results
        """
        logger.info("Running drift check...")
        
        # Data drift
        data_drift = self.drift_detector.detect_data_drift(
            current_data,
            save_report=save_report,
            report_path=self.monitoring_dir / "reports" / f"drift_{datetime.now().strftime('%Y%m%d')}.html",
        )
        
        # Alert if needed
        if data_drift["threshold_exceeded"]:
            self.alert_manager.alert_data_drift(
                drift_share=data_drift["drift_share"],
                drifted_features=data_drift["drifted_features"],
            )
            
        return data_drift
        
    def run_prediction_drift_check(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Check for prediction distribution drift.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
            
        Returns:
            Prediction drift results
        """
        logger.info("Running prediction drift check...")
        
        pred_drift = self.drift_detector.detect_prediction_drift(
            reference_predictions,
            current_predictions,
        )
        
        # Alert if needed
        if pred_drift["prediction_drift_detected"]:
            self.alert_manager.alert_prediction_drift(
                mean_shift=pred_drift["mean_shift"],
                high_risk_shift=pred_drift["high_risk_shift"],
            )
            
        return pred_drift
        
    def run_performance_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        window_name: str = "current",
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            window_name: Evaluation window name
            
        Returns:
            Evaluation results
        """
        logger.info("Running performance evaluation...")
        
        result = self.performance_monitor.evaluate(
            y_true, y_pred, y_proba, window_name
        )
        
        # Track consecutive alerts
        if result["alerts"]:
            self._consecutive_alerts += 1
        else:
            self._consecutive_alerts = 0
            
        # Create alerts
        for alert in result["alerts"]:
            if alert["type"] == "performance_drop":
                self.alert_manager.alert_performance_drop(
                    metric=alert["metric"],
                    current_value=alert["value"],
                    threshold=alert["threshold"],
                )
                
        return result
        
    def check_retraining_needed(
        self,
        drift_share: Optional[float] = None,
        current_roc_auc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if retraining is needed.
        
        Args:
            drift_share: Current drift share
            current_roc_auc: Current ROC-AUC
            
        Returns:
            Retraining trigger evaluation
        """
        logger.info("Evaluating retraining trigger...")
        
        result = self.retraining_trigger.evaluate_trigger(
            drift_share=drift_share,
            current_roc_auc=current_roc_auc,
            baseline_roc_auc=self.performance_monitor.baseline_metrics.get("roc_auc"),
            consecutive_alerts=self._consecutive_alerts,
        )
        
        if result["should_retrain"]:
            self.alert_manager.alert_retraining_recommended(
                reason="Multiple trigger conditions met",
                metrics={"drift_share": drift_share, "roc_auc": current_roc_auc},
            )
            
        return result
        
    def run_full_monitoring_cycle(
        self,
        current_data: pd.DataFrame,
        predictions: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run complete monitoring cycle.
        
        Args:
            current_data: Current inference data
            predictions: Current predictions
            y_true: Ground truth if available
            
        Returns:
            Complete monitoring results
        """
        logger.info("=" * 50)
        logger.info("Starting full monitoring cycle")
        logger.info("=" * 50)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(current_data),
        }
        
        # 1. Data drift
        results["data_drift"] = self.run_drift_check(current_data)
        
        # 2. Prediction drift
        ref_data = self.data_pipeline.load_reference_data()
        # Simulate reference predictions (in practice, store these)
        ref_predictions = np.random.beta(2, 5, size=len(ref_data))  # Placeholder
        results["prediction_drift"] = self.run_prediction_drift_check(
            ref_predictions, predictions
        )
        
        # 3. Performance (if labels available)
        if y_true is not None:
            y_pred = (predictions >= 0.5).astype(int)
            results["performance"] = self.run_performance_evaluation(
                y_true, y_pred, predictions
            )
        else:
            results["performance"] = {"status": "labels_unavailable"}
            
        # 4. Retraining check
        results["retraining"] = self.check_retraining_needed(
            drift_share=results["data_drift"].get("drift_share"),
            current_roc_auc=results.get("performance", {}).get("metrics", {}).get("roc_auc"),
        )
        
        # Summary
        results["summary"] = {
            "data_drift_detected": results["data_drift"].get("dataset_drift_detected", False),
            "prediction_drift_detected": results["prediction_drift"].get("prediction_drift_detected", False),
            "retraining_recommended": results["retraining"].get("should_retrain", False),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
        }
        
        logger.info(f"Monitoring complete. Data drift: {results['summary']['data_drift_detected']}, "
                   f"Retraining recommended: {results['summary']['retraining_recommended']}")
        
        return results
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard.
        
        Returns:
            Dashboard data
        """
        return {
            "data_summary": self.data_pipeline.get_data_summary(),
            "drift_summary": self.drift_detector.get_drift_summary(),
            "performance_summary": self.performance_monitor.get_summary(),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "retraining_summary": self.retraining_trigger.get_summary(),
        }


def run_monitoring_demo():
    """Run a demonstration of the monitoring system."""
    import numpy as np
    
    logger.info("Running monitoring demo...")
    
    # Initialize service
    service = MonitoringService()
    
    # Load reference data as "current" for demo
    current_data = service.data_pipeline.load_reference_data()
    
    # Generate mock predictions
    predictions = np.random.beta(2, 5, size=len(current_data))
    
    # Run monitoring
    results = service.run_full_monitoring_cycle(
        current_data=current_data,
        predictions=predictions,
    )
    
    print("\n" + "=" * 50)
    print("MONITORING RESULTS")
    print("=" * 50)
    print(f"Data drift detected: {results['summary']['data_drift_detected']}")
    print(f"Prediction drift detected: {results['summary']['prediction_drift_detected']}")
    print(f"Retraining recommended: {results['summary']['retraining_recommended']}")
    print(f"Active alerts: {results['summary']['active_alerts']}")
    
    return results


if __name__ == "__main__":
    run_monitoring_demo()
