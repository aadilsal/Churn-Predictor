"""Unit tests for data preprocessing functions.

Tests src/data/preprocessing.py:
- clean_telco_dataset
- validate_data_quality  
- split_features_target
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_telco_dataset,
    split_features_target,
    validate_data_quality,
)


@pytest.mark.unit
class TestCleanTelcoDataset:
    """Tests for clean_telco_dataset function."""

    def test_handles_empty_total_charges(self, sample_raw_dataframe):
        """Test that empty TotalCharges strings are handled."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        # Should not have any NaN in TotalCharges
        assert result["TotalCharges"].isna().sum() == 0
        
        # Zero tenure customer should have TotalCharges = 0
        zero_tenure_row = result[result["tenure"] == 0]
        assert len(zero_tenure_row) == 1
        assert zero_tenure_row["TotalCharges"].values[0] == 0.0

    def test_converts_senior_citizen_to_yes_no(self, sample_raw_dataframe):
        """Test SeniorCitizen is converted from 0/1 to Yes/No."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        # Should only have Yes/No values
        assert set(result["SeniorCitizen"].unique()) == {"Yes", "No"}

    def test_standardizes_no_internet_service(self, sample_raw_dataframe):
        """Test 'No internet service' is replaced with 'No'."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        internet_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        
        for col in internet_cols:
            assert "No internet service" not in result[col].values

    def test_standardizes_no_phone_service(self, sample_raw_dataframe):
        """Test 'No phone service' is replaced with 'No'."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        assert "No phone service" not in result["MultipleLines"].values

    def test_converts_churn_to_binary(self, sample_raw_dataframe):
        """Test Churn column is converted to 0/1."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        assert result["Churn"].dtype in [np.int32, np.int64]
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_preserves_all_rows(self, sample_raw_dataframe):
        """Test that cleaning preserves all rows."""
        result = clean_telco_dataset(sample_raw_dataframe)
        
        assert len(result) == len(sample_raw_dataframe)

    def test_imputes_missing_total_charges_correctly(self):
        """Test TotalCharges imputation for non-zero tenure."""
        df = pd.DataFrame({
            "customerID": ["C001"],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [10],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["Yes"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [100.0],
            "TotalCharges": [""],  # Empty string
            "Churn": ["No"],
        })
        
        result = clean_telco_dataset(df)
        
        # Should impute as MonthlyCharges * tenure = 100 * 10 = 1000
        assert result["TotalCharges"].values[0] == 1000.0


@pytest.mark.unit
class TestValidateDataQuality:
    """Tests for validate_data_quality function."""

    def test_returns_correct_structure(self, sample_cleaned_dataframe):
        """Test that quality report has expected keys."""
        result = validate_data_quality(sample_cleaned_dataframe)
        
        expected_keys = [
            "total_records",
            "total_features",
            "missing_values",
            "duplicate_records",
            "data_types",
            "outliers",
            "categorical_distributions",
        ]
        
        for key in expected_keys:
            assert key in result

    def test_correct_record_count(self, sample_cleaned_dataframe):
        """Test that total_records matches dataframe length."""
        result = validate_data_quality(sample_cleaned_dataframe)
        
        assert result["total_records"] == len(sample_cleaned_dataframe)

    def test_correct_feature_count(self, sample_cleaned_dataframe):
        """Test that total_features matches column count."""
        result = validate_data_quality(sample_cleaned_dataframe)
        
        assert result["total_features"] == len(sample_cleaned_dataframe.columns)

    def test_detects_no_duplicates_in_sample(self, sample_cleaned_dataframe):
        """Test duplicate detection on sample data."""
        result = validate_data_quality(sample_cleaned_dataframe)
        
        assert result["duplicate_records"] == 0

    def test_detects_duplicates(self):
        """Test that duplicates are detected."""
        df = pd.DataFrame({
            "col1": [1, 1, 2],
            "col2": ["a", "a", "b"],
        })
        
        result = validate_data_quality(df)
        
        assert result["duplicate_records"] == 1

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        
        result = validate_data_quality(df)
        
        assert result["total_records"] == 0
        assert result["total_features"] == 0

    def test_detects_missing_values(self):
        """Test that missing values are detected."""
        df = pd.DataFrame({
            "col1": [1, 2, np.nan],
            "col2": ["a", None, "c"],
        })
        
        result = validate_data_quality(df)
        
        assert "col1" in result["missing_values"]
        assert result["missing_values"]["col1"] == 1


@pytest.mark.unit
class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""

    def test_splits_correctly(self, sample_cleaned_dataframe):
        """Test basic split functionality."""
        X, y = split_features_target(sample_cleaned_dataframe)
        
        # Target should not be in features
        assert "Churn" not in X.columns
        
        # customerID should be excluded
        assert "customerID" not in X.columns
        
        # Should have correct length
        assert len(X) == len(sample_cleaned_dataframe)
        assert len(y) == len(sample_cleaned_dataframe)

    def test_target_column_parameter(self, sample_cleaned_dataframe):
        """Test custom target column name."""
        df = sample_cleaned_dataframe.copy()
        df["CustomTarget"] = df["Churn"]
        
        X, y = split_features_target(df, target_col="CustomTarget")
        
        assert "CustomTarget" not in X.columns
        assert len(y) == len(df)

    def test_returns_series_for_target(self, sample_cleaned_dataframe):
        """Test that target is returned as Series."""
        X, y = split_features_target(sample_cleaned_dataframe)
        
        assert isinstance(y, pd.Series)

    def test_returns_dataframe_for_features(self, sample_cleaned_dataframe):
        """Test that features is returned as DataFrame."""
        X, y = split_features_target(sample_cleaned_dataframe)
        
        assert isinstance(X, pd.DataFrame)

    def test_missing_target_column_raises_error(self, sample_cleaned_dataframe):
        """Test that missing target column raises appropriate error."""
        df = sample_cleaned_dataframe.drop(columns=["Churn"])
        
        with pytest.raises(KeyError):
            split_features_target(df)
