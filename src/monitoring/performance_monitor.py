"""Performance monitoring for model accuracy tracking over time."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from src.utils.logging import logger


class PerformanceMonitor:
    """Monitor model performance when ground truth becomes available."""
    
    # Performance thresholds
    THRESHOLDS = {
        "roc_auc_min": 0.75,  # Below this triggers alert
        "accuracy_min": 0.70,
        "calibration_max": 0.15,  # Brier score
        "decay_threshold": 0.05,  # 5% drop from baseline triggers alert
    }
    
    def __init__(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
        storage_path: Path = Path("data/monitoring/performance_log.json"),
    ):
        """Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline metrics from initial training
            storage_path: Path to store performance history
        """
        self.baseline_metrics = baseline_metrics or {}
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._performance_history: List[Dict] = []
        self._load_history()
        
    def _load_history(self) -> None:
        """Load performance history from storage."""
        if self.storage_path.exists():
            import json
            with open(self.storage_path, "r") as f:
                self._performance_history = json.load(f)
                
    def _save_history(self) -> None:
        """Save performance history to storage."""
        import json
        with open(self.storage_path, "w") as f:
            json.dump(self._performance_history, f, indent=2)
            
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        window_name: str = "current",
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            window_name: Name for this evaluation window
            
        Returns:
            Performance metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else None,
            "brier_score": brier_score_loss(y_true, y_proba),
        }
        
        # Check against thresholds
        alerts = []
        if metrics["roc_auc"] and metrics["roc_auc"] < self.THRESHOLDS["roc_auc_min"]:
            alerts.append({
                "type": "performance_drop",
                "metric": "roc_auc",
                "value": metrics["roc_auc"],
                "threshold": self.THRESHOLDS["roc_auc_min"],
            })
            
        if metrics["brier_score"] > self.THRESHOLDS["calibration_max"]:
            alerts.append({
                "type": "calibration_degradation",
                "metric": "brier_score",
                "value": metrics["brier_score"],
                "threshold": self.THRESHOLDS["calibration_max"],
            })
            
        # Check for decay from baseline
        decay_alerts = self._check_decay(metrics)
        alerts.extend(decay_alerts)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "window": window_name,
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "metrics": metrics,
            "alerts": alerts,
            "status": "degraded" if alerts else "healthy",
        }
        
        # Store in history
        self._performance_history.append(result)
        self._save_history()
        
        logger.info(f"Performance evaluation: ROC-AUC={metrics['roc_auc']:.4f}, Status={result['status']}")
        
        return result
        
    def _check_decay(self, current_metrics: Dict[str, float]) -> List[Dict]:
        """Check for performance decay from baseline."""
        alerts = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric not in current_metrics:
                continue
                
            current_value = current_metrics[metric]
            if current_value is None:
                continue
                
            # For metrics where higher is better
            if metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                decay = baseline_value - current_value
                if decay > self.THRESHOLDS["decay_threshold"]:
                    alerts.append({
                        "type": "performance_decay",
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "decay": decay,
                    })
                    
            # For metrics where lower is better (brier score)
            elif metric == "brier_score":
                increase = current_value - baseline_value
                if increase > self.THRESHOLDS["decay_threshold"]:
                    alerts.append({
                        "type": "calibration_decay",
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "increase": increase,
                    })
                    
        return alerts
        
    def get_performance_trend(
        self,
        metric: str = "roc_auc",
        n_windows: int = 10,
    ) -> Dict[str, Any]:
        """Get performance trend over recent windows.
        
        Args:
            metric: Metric to track
            n_windows: Number of recent windows
            
        Returns:
            Trend analysis
        """
        if not self._performance_history:
            return {"message": "No performance history available"}
            
        recent = self._performance_history[-n_windows:]
        
        values = [
            w["metrics"].get(metric) 
            for w in recent 
            if w["metrics"].get(metric) is not None
        ]
        
        if not values:
            return {"message": f"No {metric} data available"}
            
        trend = {
            "metric": metric,
            "n_windows": len(values),
            "current": values[-1],
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
            "trend_direction": "stable",
        }
        
        # Simple trend detection
        if len(values) >= 3:
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            diff = second_half - first_half
            
            if diff < -0.02:
                trend["trend_direction"] = "declining"
            elif diff > 0.02:
                trend["trend_direction"] = "improving"
                
        return trend
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance monitoring."""
        if not self._performance_history:
            return {"message": "No performance history"}
            
        recent = self._performance_history[-1]
        total_alerts = sum(len(w["alerts"]) for w in self._performance_history)
        
        return {
            "total_evaluations": len(self._performance_history),
            "total_alerts": total_alerts,
            "last_evaluation": recent["timestamp"],
            "last_status": recent["status"],
            "last_roc_auc": recent["metrics"].get("roc_auc"),
        }
