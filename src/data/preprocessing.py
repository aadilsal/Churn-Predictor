"""Data preprocessing utilities."""

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logging import logger


def clean_telco_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco Customer Churn dataset.

    Args:
        df: Raw dataframe

    Returns:
        Cleaned dataframe
    """
    logger.info("Starting data cleaning...")
    df_clean = df.copy()

    # 1. Handle TotalCharges (known issue: empty strings for new customers)
    logger.info("Cleaning TotalCharges column...")
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # For customers with 0 tenure, TotalCharges should be 0
    mask_zero_tenure = df_clean["tenure"] == 0
    df_clean.loc[mask_zero_tenure, "TotalCharges"] = 0.0

    # For others with missing TotalCharges, impute with MonthlyCharges * tenure
    mask_missing = df_clean["TotalCharges"].isna()
    df_clean.loc[mask_missing, "TotalCharges"] = (
        df_clean.loc[mask_missing, "MonthlyCharges"] * df_clean.loc[mask_missing, "tenure"]
    )

    logger.info(f"Imputed {mask_missing.sum()} missing TotalCharges values")

    # 2. Convert SeniorCitizen to Yes/No for consistency
    df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 3. Standardize "No internet service" and "No phone service" to "No"
    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in internet_cols:
        df_clean[col] = df_clean[col].replace("No internet service", "No")

    df_clean["MultipleLines"] = df_clean["MultipleLines"].replace("No phone service", "No")

    # 4. Convert target variable to binary
    df_clean["Churn"] = (df_clean["Churn"] == "Yes").astype(int)

    logger.info("Data cleaning completed")
    return df_clean


def validate_data_quality(df: pd.DataFrame) -> dict:
    """Perform comprehensive data quality checks.

    Args:
        df: Dataframe to validate

    Returns:
        Dictionary with quality metrics
    """
    logger.info("Performing data quality checks...")

    quality_report = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "missing_values": {},
        "duplicate_records": df.duplicated().sum(),
        "data_types": df.dtypes.to_dict(),
    }

    # Check missing values
    missing = df.isnull().sum()
    quality_report["missing_values"] = {
        col: int(count) for col, count in missing.items() if count > 0
    }

    # Check for outliers in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            outliers[col] = int(outlier_count)

    quality_report["outliers"] = outliers

    # Check categorical value distributions
    categorical_cols = df.select_dtypes(include=["object"]).columns
    value_counts = {}

    for col in categorical_cols:
        value_counts[col] = df[col].value_counts().to_dict()

    quality_report["categorical_distributions"] = value_counts

    logger.info("Data quality checks completed")
    return quality_report


def split_features_target(
    df: pd.DataFrame, target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.

    Args:
        df: Input dataframe
        target_col: Name of target column

    Returns:
        Tuple of (features, target)
    """
    # Exclude customerID and target
    feature_cols = [col for col in df.columns if col not in ["customerID", target_col]]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(f"Split data into {X.shape[1]} features and target variable")
    return X, y
