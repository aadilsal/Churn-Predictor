"""Survival data preparation for time-to-churn analysis.

ASSUMPTIONS:
1. Tenure represents time since customer acquisition
2. Churn=1 indicates the event occurred (customer churned)
3. Churn=0 indicates censored observation (still active or lost to follow-up)
4. Time origin is customer acquisition date
5. All customers are observed for their entire tenure period
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import logger


def prepare_survival_data(
    df: pd.DataFrame,
    duration_col: str = "tenure",
    event_col: str = "Churn",
    min_duration: int = 1,
) -> pd.DataFrame:
    """Prepare data for survival analysis.
    
    Args:
        df: Input dataframe with customer data
        duration_col: Column representing time duration (tenure)
        event_col: Column indicating event occurrence (churn)
        min_duration: Minimum duration to include (handles edge cases)
        
    Returns:
        DataFrame with survival-ready columns
    """
    logger.info("Preparing survival data...")
    
    survival_df = df.copy()
    
    # Ensure duration is numeric and positive
    survival_df["duration"] = pd.to_numeric(survival_df[duration_col], errors="coerce")
    survival_df["duration"] = survival_df["duration"].clip(lower=min_duration)
    
    # Ensure event indicator is binary
    survival_df["event"] = survival_df[event_col].astype(int)
    
    # Calculate observation properties
    n_events = survival_df["event"].sum()
    n_censored = len(survival_df) - n_events
    censoring_rate = n_censored / len(survival_df)
    
    logger.info(f"Survival data prepared: {len(survival_df)} observations")
    logger.info(f"Events (churned): {n_events}, Censored: {n_censored}")
    logger.info(f"Censoring rate: {censoring_rate:.1%}")
    
    # Validate assumptions
    _validate_survival_assumptions(survival_df)
    
    return survival_df


def _validate_survival_assumptions(df: pd.DataFrame) -> None:
    """Validate survival analysis assumptions."""
    # Check for negative durations
    if (df["duration"] <= 0).any():
        logger.warning("Found non-positive durations - may violate survival assumptions")
        
    # Check censoring balance
    event_rate = df["event"].mean()
    if event_rate < 0.1:
        logger.warning(f"Low event rate ({event_rate:.1%}) - may limit model power")
    if event_rate > 0.9:
        logger.warning(f"High event rate ({event_rate:.1%}) - limited censored observations")


def add_tenure_buckets(
    df: pd.DataFrame,
    duration_col: str = "duration",
    bins: List[int] = [0, 6, 12, 24, 48, 100],
    labels: List[str] = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
) -> pd.DataFrame:
    """Add tenure bucket column for stratified analysis.
    
    Args:
        df: Survival dataframe
        duration_col: Duration column name
        bins: Bucket boundaries
        labels: Bucket labels
        
    Returns:
        DataFrame with tenure_bucket column
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df[duration_col],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return df


def get_survival_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract features suitable for Cox PH model.
    
    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude
        
    Returns:
        Tuple of (feature dataframe, feature names)
    """
    exclude_cols = exclude_cols or [
        "customerID", "Churn", "tenure", "duration", "event", "tenure_bucket"
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle categorical variables
    features_df = df[feature_cols].copy()
    
    # One-hot encode categoricals
    categorical_cols = features_df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
        
    feature_names = features_df.columns.tolist()
    
    logger.info(f"Extracted {len(feature_names)} survival features")
    
    return features_df, feature_names


def create_survival_dataset(
    df: pd.DataFrame,
    duration_col: str = "tenure",
    event_col: str = "Churn",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create complete survival dataset.
    
    Args:
        df: Raw input dataframe
        duration_col: Duration column
        event_col: Event column
        
    Returns:
        Tuple of (features_df, durations, events)
    """
    # Prepare survival data
    survival_df = prepare_survival_data(df, duration_col, event_col)
    
    # Add tenure buckets
    survival_df = add_tenure_buckets(survival_df)
    
    # Extract features
    features_df, feature_names = get_survival_features(survival_df)
    
    durations = survival_df["duration"].values
    events = survival_df["event"].values
    
    return features_df, durations, events


def get_survival_summary(df: pd.DataFrame) -> Dict:
    """Get summary statistics for survival data."""
    return {
        "total_observations": len(df),
        "events": int(df["event"].sum()),
        "censored": int((1 - df["event"]).sum()),
        "event_rate": float(df["event"].mean()),
        "median_duration": float(df["duration"].median()),
        "mean_duration": float(df["duration"].mean()),
        "max_duration": float(df["duration"].max()),
    }
