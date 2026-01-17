"""Unit tests for feature engineering module.

Tests src/models/feature_engineering.py:
- FeatureEngineer class
- create_train_test_split function
- prepare_features function
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.feature_engineering import (
    FeatureEngineer,
    create_train_test_split,
    prepare_features,
)


@pytest.mark.unit
class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        fe = FeatureEngineer()
        
        assert fe.categorical_cols is None
        assert fe.numerical_cols is None
        assert fe.random_state == 42
        assert fe.preprocessor is None

    def test_init_with_custom_columns(self):
        """Test initialization with custom column lists."""
        cat_cols = ["Contract", "PaymentMethod"]
        num_cols = ["tenure", "MonthlyCharges"]
        
        fe = FeatureEngineer(
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
            random_state=123,
        )
        
        assert fe.categorical_cols == cat_cols
        assert fe.numerical_cols == num_cols
        assert fe.random_state == 123

    def test_fit_identifies_columns(self, sample_cleaned_dataframe):
        """Test that fit correctly identifies column types."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        fe.fit(df)
        
        assert fe.categorical_cols is not None
        assert fe.numerical_cols is not None
        assert len(fe.categorical_cols) > 0
        assert len(fe.numerical_cols) > 0

    def test_transform_returns_numpy_array(self, sample_cleaned_dataframe):
        """Test that transform returns numpy array."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)
        
        assert isinstance(result, np.ndarray)

    def test_fit_transform_consistency(self, sample_cleaned_dataframe):
        """Test that fit_transform equals fit + transform."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe1 = FeatureEngineer(random_state=42)
        result1 = fe1.fit_transform(df)
        
        fe2 = FeatureEngineer(random_state=42)
        fe2.fit(df)
        result2 = fe2.transform(df)
        
        np.testing.assert_array_almost_equal(result1, result2)

    def test_transform_before_fit_raises_error(self, sample_cleaned_dataframe):
        """Test that transform before fit raises error."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        
        with pytest.raises(RuntimeError):
            fe.transform(df)

    def test_get_feature_names(self, sample_cleaned_dataframe):
        """Test that feature names are retrievable after fit."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        fe.fit(df)
        
        feature_names = fe.get_feature_names()
        
        assert feature_names is not None
        assert len(feature_names) > 0

    def test_save_and_load(self, sample_cleaned_dataframe):
        """Test save and load functionality."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        fe.fit(df)
        original_result = fe.transform(df)
        
        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            fe.save(save_path)
            
            # Check files exist (actual filenames from implementation)
            assert (save_path / "feature_preprocessor.joblib").exists()
            assert (save_path / "feature_names.json").exists()
            
            # Load and compare
            loaded_fe = FeatureEngineer.load(save_path)
            loaded_result = loaded_fe.transform(df)
            
            np.testing.assert_array_almost_equal(original_result, loaded_result)

    def test_output_shape_consistency(self, sample_cleaned_dataframe):
        """Test that output shape is consistent."""
        df = sample_cleaned_dataframe.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        
        # Should have same number of rows
        assert result.shape[0] == len(df)
        
        # Number of columns should match feature names
        assert result.shape[1] == len(fe.get_feature_names())


@pytest.mark.unit
class TestCreateTrainTestSplit:
    """Tests for create_train_test_split function."""

    def test_returns_correct_tuple_size(self, sample_cleaned_dataframe):
        """Test that function returns 6 elements."""
        # Need larger dataset for stratified split to work
        df_large = pd.concat([sample_cleaned_dataframe] * 10).reset_index(drop=True)
        X = df_large.drop(columns=["customerID", "Churn"])
        y = df_large["Churn"]
        
        result = create_train_test_split(X, y)
        
        assert len(result) == 6

    def test_split_sizes(self, sample_cleaned_dataframe):
        """Test that split sizes are approximately correct."""
        # Need more data for meaningful split
        df_large = pd.concat([sample_cleaned_dataframe] * 20).reset_index(drop=True)
        X = df_large.drop(columns=["customerID", "Churn"])
        y = df_large["Churn"]
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
            X, y, test_size=0.2, val_size=0.2
        )
        
        total = len(X)
        
        # Test size should be ~20%
        assert 0.15 <= len(X_test) / total <= 0.25
        
        # Validation should be ~20% of remaining
        remaining = total - len(X_test)
        assert 0.15 <= len(X_val) / remaining <= 0.25

    def test_reproducibility(self, sample_cleaned_dataframe):
        """Test that same random_state gives same split."""
        df_large = pd.concat([sample_cleaned_dataframe] * 10).reset_index(drop=True)
        X = df_large.drop(columns=["customerID", "Churn"])
        y = df_large["Churn"]
        
        result1 = create_train_test_split(X, y, random_state=42)
        result2 = create_train_test_split(X, y, random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(result1[0], result2[0])
        pd.testing.assert_series_equal(result1[3], result2[3])

    def test_stratification(self, sample_cleaned_dataframe):
        """Test that stratification preserves class ratios."""
        # Create larger dataset with known class ratio
        df_large = pd.concat([sample_cleaned_dataframe] * 20).reset_index(drop=True)
        X = df_large.drop(columns=["customerID", "Churn"])
        y = df_large["Churn"]
        
        original_ratio = y.mean()
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
        
        # All splits should have similar ratio (within tolerance)
        assert abs(y_train.mean() - original_ratio) < 0.1
        assert abs(y_val.mean() - original_ratio) < 0.1
        assert abs(y_test.mean() - original_ratio) < 0.1


@pytest.mark.unit
class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_excludes_target_column(self, sample_cleaned_dataframe):
        """Test that target column is excluded from features."""
        X, y = prepare_features(sample_cleaned_dataframe, target_col="Churn")
        
        assert "Churn" not in X.columns

    def test_excludes_customer_id_by_default(self, sample_cleaned_dataframe):
        """Test that customerID is excluded by default."""
        X, y = prepare_features(sample_cleaned_dataframe)
        
        assert "customerID" not in X.columns

    def test_custom_exclude_columns(self, sample_cleaned_dataframe):
        """Test custom column exclusion."""
        X, y = prepare_features(
            sample_cleaned_dataframe,
            exclude_cols=["customerID", "gender"]
        )
        
        assert "customerID" not in X.columns
        assert "gender" not in X.columns

    def test_returns_correct_types(self, sample_cleaned_dataframe):
        """Test return types are correct."""
        X, y = prepare_features(sample_cleaned_dataframe)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_values_correct(self, sample_cleaned_dataframe):
        """Test that target values match original."""
        X, y = prepare_features(sample_cleaned_dataframe)
        
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            sample_cleaned_dataframe["Churn"].reset_index(drop=True)
        )
