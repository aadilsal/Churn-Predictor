"""Automated retraining trigger logic."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging import logger


class RetrainingTrigger:
    """Automated retraining trigger with safeguards."""
    
    # Retraining conditions
    CONDITIONS = {
        "max_drift_share": 0.6,  # 60% features drifted
        "performance_decay": 0.08,  # 8% drop from baseline
        "min_roc_auc": 0.72,  # Absolute minimum
        "consecutive_alerts": 3,  # N consecutive performance alerts
    }
    
    # Safeguards
    SAFEGUARDS = {
        "min_days_between_retraining": 7,
        "require_approval": True,
        "max_retraining_per_month": 4,
    }
    
    def __init__(
        self,
        storage_path: Path = Path("data/monitoring/retraining_log.json"),
        pipeline_callback: Optional[Callable[[], Dict]] = None,
    ):
        """Initialize retraining trigger.
        
        Args:
            storage_path: Path to store retraining history
            pipeline_callback: Optional callback to trigger retraining
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.pipeline_callback = pipeline_callback
        self._retraining_history: List[Dict] = []
        self._pending_triggers: List[Dict] = []
        self._load_history()
        
    def _load_history(self) -> None:
        """Load retraining history."""
        if self.storage_path.exists():
            import json
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self._retraining_history = data.get("history", [])
                self._pending_triggers = data.get("pending", [])
                
    def _save_history(self) -> None:
        """Save retraining history."""
        import json
        with open(self.storage_path, "w") as f:
            json.dump({
                "history": self._retraining_history,
                "pending": self._pending_triggers,
            }, f, indent=2)
            
    def evaluate_trigger(
        self,
        drift_share: Optional[float] = None,
        current_roc_auc: Optional[float] = None,
        baseline_roc_auc: Optional[float] = None,
        consecutive_alerts: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate whether retraining should be triggered.
        
        Args:
            drift_share: Current drift share
            current_roc_auc: Current ROC-AUC
            baseline_roc_auc: Baseline ROC-AUC
            consecutive_alerts: Number of consecutive alerts
            
        Returns:
            Trigger evaluation result
        """
        triggers = []
        
        # Check drift condition
        if drift_share and drift_share > self.CONDITIONS["max_drift_share"]:
            triggers.append({
                "condition": "data_drift",
                "value": drift_share,
                "threshold": self.CONDITIONS["max_drift_share"],
            })
            
        # Check performance decay
        if current_roc_auc and baseline_roc_auc:
            decay = baseline_roc_auc - current_roc_auc
            if decay > self.CONDITIONS["performance_decay"]:
                triggers.append({
                    "condition": "performance_decay",
                    "value": decay,
                    "threshold": self.CONDITIONS["performance_decay"],
                })
                
        # Check absolute minimum
        if current_roc_auc and current_roc_auc < self.CONDITIONS["min_roc_auc"]:
            triggers.append({
                "condition": "below_minimum",
                "value": current_roc_auc,
                "threshold": self.CONDITIONS["min_roc_auc"],
            })
            
        # Check consecutive alerts
        if consecutive_alerts >= self.CONDITIONS["consecutive_alerts"]:
            triggers.append({
                "condition": "consecutive_alerts",
                "value": consecutive_alerts,
                "threshold": self.CONDITIONS["consecutive_alerts"],
            })
            
        should_retrain = len(triggers) > 0
        
        # Check safeguards
        safeguard_check = self._check_safeguards()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": should_retrain and safeguard_check["passed"],
            "triggers": triggers,
            "safeguards": safeguard_check,
            "requires_approval": self.SAFEGUARDS["require_approval"],
        }
        
        if should_retrain and not safeguard_check["passed"]:
            result["blocked_reason"] = safeguard_check["reason"]
            
        # Store pending trigger
        if should_retrain:
            self._pending_triggers.append(result)
            self._save_history()
            
        return result
        
    def _check_safeguards(self) -> Dict[str, Any]:
        """Check retraining safeguards."""
        import pandas as pd
        
        # Check minimum days between retraining
        if self._retraining_history:
            last_retrain = pd.to_datetime(self._retraining_history[-1]["timestamp"])
            days_since = (datetime.now() - last_retrain).days
            
            if days_since < self.SAFEGUARDS["min_days_between_retraining"]:
                return {
                    "passed": False,
                    "reason": f"Only {days_since} days since last retraining (min: {self.SAFEGUARDS['min_days_between_retraining']})",
                }
                
        # Check max retraining per month
        month_start = datetime.now().replace(day=1)
        month_retrains = sum(
            1 for r in self._retraining_history
            if pd.to_datetime(r["timestamp"]) >= month_start
        )
        
        if month_retrains >= self.SAFEGUARDS["max_retraining_per_month"]:
            return {
                "passed": False,
                "reason": f"Already {month_retrains} retrains this month (max: {self.SAFEGUARDS['max_retraining_per_month']})",
            }
            
        return {"passed": True, "reason": None}
        
    def approve_retraining(
        self,
        trigger_id: Optional[str] = None,
        approver: str = "system",
    ) -> Dict[str, Any]:
        """Approve and execute retraining.
        
        Args:
            trigger_id: ID of pending trigger to approve
            approver: Name of approver
            
        Returns:
            Retraining result
        """
        if not self._pending_triggers:
            return {"error": "No pending retraining triggers"}
            
        # Get trigger to approve
        if trigger_id:
            trigger = next((t for t in self._pending_triggers if t.get("id") == trigger_id), None)
            if not trigger:
                return {"error": f"Trigger {trigger_id} not found"}
        else:
            trigger = self._pending_triggers[-1]
            
        # Execute retraining
        result = {
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "approver": approver,
            "status": "initiated",
        }
        
        if self.pipeline_callback:
            try:
                logger.info("Executing retraining pipeline...")
                pipeline_result = self.pipeline_callback()
                result["pipeline_result"] = pipeline_result
                result["status"] = "completed"
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                logger.error(f"Retraining failed: {e}")
        else:
            result["status"] = "pending_execution"
            result["message"] = "No pipeline callback configured"
            
        # Record in history
        self._retraining_history.append(result)
        self._pending_triggers = [t for t in self._pending_triggers if t != trigger]
        self._save_history()
        
        return result
        
    def get_retraining_history(self) -> List[Dict]:
        """Get retraining history."""
        return self._retraining_history
        
    def get_pending_triggers(self) -> List[Dict]:
        """Get pending retraining triggers."""
        return self._pending_triggers
        
    def get_summary(self) -> Dict[str, Any]:
        """Get retraining summary."""
        return {
            "total_retrains": len(self._retraining_history),
            "pending_triggers": len(self._pending_triggers),
            "last_retrain": self._retraining_history[-1]["timestamp"] if self._retraining_history else None,
            "conditions": self.CONDITIONS,
            "safeguards": self.SAFEGUARDS,
        }
