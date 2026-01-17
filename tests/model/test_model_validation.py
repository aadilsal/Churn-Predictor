"""Model validation tests.

Tests model loading, prediction stability, output correctness,
and performance regression detection.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.integration
class TestModelLoading:
    """Tests for model loading functionality."""

    def test_model_file_exists(self, model_path):
        """Test that model file exists."""
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_preprocessor_file_exists(self, preprocessor_path):
        """Test that preprocessor file exists."""
        assert preprocessor_path.exists(), f"Preprocessor not found at {preprocessor_path}"

    def test_feature_names_file_exists(self, feature_names_path):
        """Test that feature names file exists."""
        assert feature_names_path.exists(), f"Feature names not found at {feature_names_path}"

    def test_model_loads_successfully(self, model_path):
        """Test that model loads without errors."""
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_preprocessor_loads_successfully(self, preprocessor_path):
        """Test that preprocessor loads without errors."""
        preprocessor = joblib.load(preprocessor_path)
        assert preprocessor is not None
        assert hasattr(preprocessor, "transform")

    def test_feature_names_loads_successfully(self, feature_names_path):
        """Test that feature names load correctly."""
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


@pytest.mark.integration
class TestModelPredictions:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def loaded_model(self, model_path):
        """Load the model for testing."""
        return joblib.load(model_path)

    @pytest.fixture
    def loaded_preprocessor(self, preprocessor_path):
        """Load the preprocessor for testing."""
        return joblib.load(preprocessor_path)

    @pytest.fixture
    def feature_names(self, feature_names_path):
        """Load feature names."""
        with open(feature_names_path, "r") as f:
            return json.load(f)

    def test_prediction_output_shape(self, loaded_model, feature_names):
        """Test that prediction output has correct shape."""
        # Create dummy input with correct number of features
        n_samples = 5
        X = np.random.randn(n_samples, len(feature_names))
        
        predictions = loaded_model.predict(X)
        
        assert predictions.shape == (n_samples,)

    def test_probability_output_shape(self, loaded_model, feature_names):
        """Test that probability output has correct shape."""
        n_samples = 5
        X = np.random.randn(n_samples, len(feature_names))
        
        probabilities = loaded_model.predict_proba(X)
        
        # Should have 2 columns (binary classification)
        assert probabilities.shape == (n_samples, 2)

    def test_probabilities_in_valid_range(self, loaded_model, feature_names):
        """Test that probabilities are between 0 and 1."""
        n_samples = 10
        X = np.random.randn(n_samples, len(feature_names))
        
        probabilities = loaded_model.predict_proba(X)
        
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)

    def test_probabilities_sum_to_one(self, loaded_model, feature_names):
        """Test that class probabilities sum to 1."""
        n_samples = 10
        X = np.random.randn(n_samples, len(feature_names))
        
        probabilities = loaded_model.predict_proba(X)
        row_sums = probabilities.sum(axis=1)
        
        np.testing.assert_array_almost_equal(row_sums, np.ones(n_samples))

    def test_binary_predictions(self, loaded_model, feature_names):
        """Test that predictions are binary (0 or 1)."""
        n_samples = 10
        X = np.random.randn(n_samples, len(feature_names))
        
        predictions = loaded_model.predict(X)
        
        assert set(predictions).issubset({0, 1})


@pytest.mark.integration
class TestPredictionStability:
    """Tests for prediction stability."""

    @pytest.fixture
    def loaded_model(self, model_path):
        """Load the model for testing."""
        return joblib.load(model_path)

    @pytest.fixture
    def feature_names(self, feature_names_path):
        """Load feature names."""
        with open(feature_names_path, "r") as f:
            return json.load(f)

    def test_same_input_same_output(self, loaded_model, feature_names):
        """Test that same input produces same output."""
        np.random.seed(42)
        X = np.random.randn(5, len(feature_names))
        
        predictions1 = loaded_model.predict(X)
        predictions2 = loaded_model.predict(X)
        
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_probability_reproducibility(self, loaded_model, feature_names):
        """Test that probabilities are reproducible."""
        np.random.seed(42)
        X = np.random.randn(5, len(feature_names))
        
        proba1 = loaded_model.predict_proba(X)
        proba2 = loaded_model.predict_proba(X)
        
        np.testing.assert_array_equal(proba1, proba2)


@pytest.mark.integration
@pytest.mark.slow
class TestModelPerformance:
    """Tests for model performance on test data."""

    @pytest.fixture
    def test_data(self):
        """Load test data for performance evaluation."""
        data_path = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"
        if not data_path.exists():
            pytest.skip("Test data not available")
        
        df = pd.read_csv(data_path)
        return df

    @pytest.fixture
    def loaded_model(self, model_path):
        """Load the model for testing."""
        return joblib.load(model_path)

    @pytest.fixture
    def loaded_preprocessor(self, preprocessor_path):
        """Load the preprocessor for testing."""
        return joblib.load(preprocessor_path)

    def test_minimum_accuracy(self, test_data, loaded_model, loaded_preprocessor):
        """Test that model meets minimum accuracy threshold."""
        # Prepare features
        feature_cols = [c for c in test_data.columns if c not in ["customerID", "Churn"]]
        X = test_data[feature_cols]
        y = test_data["Churn"]
        
        # Transform and predict
        X_transformed = loaded_preprocessor.transform(X)
        predictions = loaded_model.predict(X_transformed)
        
        accuracy = (predictions == y).mean()
        
        # Minimum accuracy threshold
        assert accuracy >= 0.75, f"Accuracy {accuracy:.2%} below minimum threshold of 75%"

    def test_minimum_auc(self, test_data, loaded_model, loaded_preprocessor):
        """Test that model meets minimum AUC threshold."""
        from sklearn.metrics import roc_auc_score
        
        feature_cols = [c for c in test_data.columns if c not in ["customerID", "Churn"]]
        X = test_data[feature_cols]
        y = test_data["Churn"]
        
        X_transformed = loaded_preprocessor.transform(X)
        probabilities = loaded_model.predict_proba(X_transformed)[:, 1]
        
        auc = roc_auc_score(y, probabilities)
        
        # Minimum AUC threshold
        assert auc >= 0.80, f"AUC {auc:.3f} below minimum threshold of 0.80"
