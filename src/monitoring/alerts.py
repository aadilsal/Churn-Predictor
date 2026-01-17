"""Alerting system for monitoring events."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging import logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DROP = "performance_drop"
    CALIBRATION_DEGRADATION = "calibration_degradation"
    RETRAINING_RECOMMENDED = "retraining_recommended"
    SYSTEM_ERROR = "system_error"


class Alert:
    """Individual alert object."""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggested_actions: Optional[List[str]] = None,
    ):
        self.id = f"{alert_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.suggested_actions = suggested_actions or []
        self.timestamp = datetime.now()
        self.acknowledged = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """Manage monitoring alerts and notifications."""
    
    def __init__(
        self,
        storage_path: Path = Path("data/monitoring/alerts.json"),
        notification_handlers: Optional[List[Callable[[Alert], None]]] = None,
    ):
        """Initialize alert manager.
        
        Args:
            storage_path: Path to store alert history
            notification_handlers: Optional handlers for notifications
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.notification_handlers = notification_handlers or []
        self._alerts: List[Alert] = []
        self._load_alerts()
        
    def _load_alerts(self) -> None:
        """Load alerts from storage."""
        if self.storage_path.exists():
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                # Just store as dicts, don't reconstruct Alert objects
                self._alert_history = data
        else:
            self._alert_history = []
            
    def _save_alerts(self) -> None:
        """Save alerts to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(self._alert_history, f, indent=2)
            
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggested_actions: Optional[List[str]] = None,
    ) -> Alert:
        """Create and register a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            details: Additional details
            suggested_actions: Suggested remediation actions
            
        Returns:
            Created alert
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
            suggested_actions=suggested_actions,
        )
        
        self._alerts.append(alert)
        self._alert_history.append(alert.to_dict())
        self._save_alerts()
        
        # Log the alert
        log_msg = f"[{severity.value.upper()}] {alert_type.value}: {message}"
        if severity == AlertSeverity.CRITICAL:
            logger.error(log_msg)
        elif severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
            
        # Trigger notification handlers
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
                
        return alert
        
    def alert_data_drift(
        self,
        drift_share: float,
        drifted_features: List[str],
        threshold: float = 0.5,
    ) -> Optional[Alert]:
        """Create alert for data drift if threshold exceeded.
        
        Args:
            drift_share: Proportion of features drifted
            drifted_features: List of drifted feature names
            threshold: Threshold for alerting
            
        Returns:
            Alert if created, None otherwise
        """
        if drift_share <= threshold:
            return None
            
        severity = AlertSeverity.CRITICAL if drift_share > 0.7 else AlertSeverity.WARNING
        
        return self.create_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=severity,
            message=f"Data drift detected: {drift_share*100:.1f}% of features have drifted",
            details={
                "drift_share": drift_share,
                "drifted_features": drifted_features[:10],  # Limit to first 10
                "n_drifted": len(drifted_features),
            },
            suggested_actions=[
                "Review drifted features for data quality issues",
                "Check for upstream data pipeline changes",
                "Consider retraining model with recent data",
            ],
        )
        
    def alert_prediction_drift(
        self,
        mean_shift: float,
        high_risk_shift: float,
    ) -> Optional[Alert]:
        """Create alert for prediction drift.
        
        Args:
            mean_shift: Shift in mean prediction
            high_risk_shift: Shift in high-risk proportion
            
        Returns:
            Alert if created, None otherwise
        """
        if mean_shift < 0.1 and high_risk_shift < 0.1:
            return None
            
        severity = AlertSeverity.CRITICAL if mean_shift > 0.2 or high_risk_shift > 0.15 else AlertSeverity.WARNING
        
        return self.create_alert(
            alert_type=AlertType.PREDICTION_DRIFT,
            severity=severity,
            message=f"Prediction distribution has shifted significantly",
            details={
                "mean_shift": mean_shift,
                "high_risk_shift": high_risk_shift,
            },
            suggested_actions=[
                "Verify input data quality",
                "Check for feature distribution changes",
                "Review business impact of prediction changes",
                "Consider model recalibration",
            ],
        )
        
    def alert_performance_drop(
        self,
        metric: str,
        current_value: float,
        threshold: float,
        baseline: Optional[float] = None,
    ) -> Alert:
        """Create alert for performance drop.
        
        Args:
            metric: Metric that dropped
            current_value: Current metric value
            threshold: Threshold that was breached
            baseline: Baseline value for comparison
            
        Returns:
            Created alert
        """
        severity = AlertSeverity.CRITICAL
        
        details = {
            "metric": metric,
            "current_value": current_value,
            "threshold": threshold,
        }
        if baseline:
            details["baseline"] = baseline
            details["decay"] = baseline - current_value
            
        return self.create_alert(
            alert_type=AlertType.PERFORMANCE_DROP,
            severity=severity,
            message=f"Model {metric} dropped to {current_value:.4f} (below threshold {threshold})",
            details=details,
            suggested_actions=[
                "Investigate data quality issues",
                "Check for concept drift",
                "Trigger model retraining",
                "Review model for business suitability",
            ],
        )
        
    def alert_retraining_recommended(
        self,
        reason: str,
        metrics: Dict[str, float],
    ) -> Alert:
        """Create alert recommending retraining.
        
        Args:
            reason: Reason for recommendation
            metrics: Current metrics
            
        Returns:
            Created alert
        """
        return self.create_alert(
            alert_type=AlertType.RETRAINING_RECOMMENDED,
            severity=AlertSeverity.WARNING,
            message=f"Model retraining recommended: {reason}",
            details={"current_metrics": metrics},
            suggested_actions=[
                "Prepare updated training dataset",
                "Run retraining pipeline",
                "Validate new model before deployment",
            ],
        )
        
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Dict]:
        """Get unacknowledged alerts.
        
        Args:
            severity: Filter by severity
            
        Returns:
            List of active alerts
        """
        active = [a for a in self._alert_history if not a.get("acknowledged", False)]
        
        if severity:
            active = [a for a in active if a["severity"] == severity.value]
            
        return active
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.
        
        Args:
            alert_id: ID of alert to acknowledge
            
        Returns:
            True if found and acknowledged
        """
        for alert in self._alert_history:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                self._save_alerts()
                return True
        return False
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert history."""
        if not self._alert_history:
            return {"message": "No alerts recorded"}
            
        by_type = {}
        by_severity = {"info": 0, "warning": 0, "critical": 0}
        
        for alert in self._alert_history:
            alert_type = alert["type"]
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            by_severity[alert["severity"]] = by_severity.get(alert["severity"], 0) + 1
            
        active = [a for a in self._alert_history if not a.get("acknowledged", False)]
        
        return {
            "total_alerts": len(self._alert_history),
            "active_alerts": len(active),
            "by_type": by_type,
            "by_severity": by_severity,
            "most_recent": self._alert_history[-1] if self._alert_history else None,
        }
