"""Monitoring data pipeline for storing reference and inference data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import logger


class MonitoringDataPipeline:
    """Pipeline for managing monitoring data storage and retrieval."""
    
    def __init__(
        self,
        storage_dir: Path = Path("data/monitoring"),
        reference_path: Optional[Path] = None,
    ):
        """Initialize the monitoring pipeline.
        
        Args:
            storage_dir: Directory for storing monitoring data
            reference_path: Path to reference (training) data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_path = reference_path or Path("data/processed/telco_churn_processed.csv")
        self._reference_data: Optional[pd.DataFrame] = None
        
        # Inference log storage
        self.inference_log_path = self.storage_dir / "inference_log.csv"
        self.predictions_log_path = self.storage_dir / "predictions_log.csv"
        
    def load_reference_data(self) -> pd.DataFrame:
        """Load reference (training) data."""
        if self._reference_data is None:
            self._reference_data = pd.read_csv(self.reference_path)
            logger.info(f"Loaded reference data: {len(self._reference_data)} records")
        return self._reference_data
        
    def log_inference(
        self,
        features: Dict[str, Any],
        prediction: float,
        risk_level: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a single inference request.
        
        Args:
            features: Input features
            prediction: Churn probability
            risk_level: Risk classification
            timestamp: Inference timestamp
        """
        timestamp = timestamp or datetime.now()
        
        record = {
            "timestamp": timestamp.isoformat(),
            "prediction": prediction,
            "risk_level": risk_level,
            **features,
        }
        
        df = pd.DataFrame([record])
        
        # Append to log
        if self.inference_log_path.exists():
            df.to_csv(self.inference_log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.inference_log_path, index=False)
            
    def log_batch_inference(
        self,
        features_list: List[Dict[str, Any]],
        predictions: List[float],
        risk_levels: List[str],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log batch inference requests.
        
        Args:
            features_list: List of feature dictionaries
            predictions: List of predictions
            risk_levels: List of risk levels
            timestamp: Batch timestamp
        """
        timestamp = timestamp or datetime.now()
        
        records = []
        for features, pred, risk in zip(features_list, predictions, risk_levels):
            records.append({
                "timestamp": timestamp.isoformat(),
                "prediction": pred,
                "risk_level": risk,
                **features,
            })
            
        df = pd.DataFrame(records)
        
        if self.inference_log_path.exists():
            df.to_csv(self.inference_log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.inference_log_path, index=False)
            
        logger.info(f"Logged {len(records)} inference records")
        
    def get_inference_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get logged inference data.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            limit: Maximum records to return
            
        Returns:
            DataFrame of inference logs
        """
        if not self.inference_log_path.exists():
            return pd.DataFrame()
            
        df = pd.read_csv(self.inference_log_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] <= end_date]
        if limit:
            df = df.tail(limit)
            
        return df
        
    def log_ground_truth(
        self,
        customer_ids: List[str],
        actual_churn: List[int],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log ground truth labels when they become available.
        
        Args:
            customer_ids: Customer identifiers
            actual_churn: Actual churn outcomes (0/1)
            timestamp: When labels were received
        """
        timestamp = timestamp or datetime.now()
        
        records = [
            {
                "timestamp": timestamp.isoformat(),
                "customer_id": cid,
                "actual_churn": churn,
            }
            for cid, churn in zip(customer_ids, actual_churn)
        ]
        
        df = pd.DataFrame(records)
        labels_path = self.storage_dir / "ground_truth_log.csv"
        
        if labels_path.exists():
            df.to_csv(labels_path, mode="a", header=False, index=False)
        else:
            df.to_csv(labels_path, index=False)
            
        logger.info(f"Logged {len(records)} ground truth labels")
        
    def get_windowed_data(
        self,
        window_days: int = 7,
    ) -> pd.DataFrame:
        """Get inference data from the last N days.
        
        Args:
            window_days: Number of days to include
            
        Returns:
            DataFrame of recent inference data
        """
        cutoff = datetime.now() - pd.Timedelta(days=window_days)
        return self.get_inference_data(start_date=cutoff)
        
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of monitoring data."""
        reference = self.load_reference_data()
        inference = self.get_inference_data()
        
        return {
            "reference_samples": len(reference),
            "inference_samples": len(inference),
            "inference_start": inference["timestamp"].min().isoformat() if len(inference) > 0 else None,
            "inference_end": inference["timestamp"].max().isoformat() if len(inference) > 0 else None,
            "avg_prediction": float(inference["prediction"].mean()) if len(inference) > 0 else None,
        }
