"""Integration tests for data and model pipelines.

Tests component interactions:
- Data ingestion → preprocessing → model prediction
- Feature pipeline → model inference
- End-to-end data flow
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Tests for data pipeline integration."""

    @pytest.fixture
    def raw_sample_data(self):
        """Create raw sample data for testing."""
        return pd.DataFrame({
            "customerID": ["TEST-001", "TEST-002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 1],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No internet service"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [85.50, 45.00],
            "TotalCharges": ["1026.0", "45.0"],
            "Churn": ["Yes", "No"],
        })

    def test_preprocessing_creates_valid_data(self, raw_sample_data):
        """Test that preprocessing produces valid data structure."""
        from src.data.preprocessing import clean_telco_dataset
        
        cleaned = clean_telco_dataset(raw_sample_data)
        
        # Should have same number of rows
        assert len(cleaned) == len(raw_sample_data)
        
        # TotalCharges should be numeric
        assert cleaned["TotalCharges"].dtype in [np.float64, np.float32]
        
        # Churn should be binary
        assert cleaned["Churn"].dtype in [np.int32, np.int64]

    def test_feature_engineering_accepts_cleaned_data(self, raw_sample_data):
        """Test that feature engineering works on cleaned data."""
        from src.data.preprocessing import clean_telco_dataset
        from src.models.feature_engineering import FeatureEngineer
        
        cleaned = clean_telco_dataset(raw_sample_data)
        features = cleaned.drop(columns=["customerID", "Churn"])
        
        fe = FeatureEngineer()
        transformed = fe.fit_transform(features)
        
        assert transformed is not None
        assert transformed.shape[0] == len(features)

    def test_full_pipeline_preprocessing_to_prediction(self, raw_sample_data, model_path, preprocessor_path):
        """Test complete pipeline from raw data to predictions."""
        from src.data.preprocessing import clean_telco_dataset
        
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Model artifacts not available")
        
        # Step 1: Clean data
        cleaned = clean_telco_dataset(raw_sample_data)
        
        # Step 2: Prepare features
        features = cleaned.drop(columns=["customerID", "Churn"])
        
        # Step 3: Load preprocessor and transform
        preprocessor = joblib.load(preprocessor_path)
        transformed = preprocessor.transform(features)
        
        # Step 4: Load model and predict
        model = joblib.load(model_path)
        predictions = model.predict_proba(transformed)
        
        # Assertions
        assert predictions.shape[0] == len(raw_sample_data)
        assert predictions.shape[1] == 2
        assert np.all(predictions >= 0) and np.all(predictions <= 1)


@pytest.mark.integration
class TestModelPreprocessorIntegration:
    """Tests for model and preprocessor integration."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return pd.DataFrame({
            "gender": ["Male", "Female"],
            "SeniorCitizen": ["No", "Yes"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 1],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [85.50, 45.00],
            "TotalCharges": [1026.0, 45.0],
        })

    def test_preprocessor_output_matches_model_input(
        self, sample_features, model_path, preprocessor_path, feature_names_path
    ):
        """Test that preprocessor output matches model expected input."""
        if not all(p.exists() for p in [model_path, preprocessor_path, feature_names_path]):
            pytest.skip("Model artifacts not available")
        
        preprocessor = joblib.load(preprocessor_path)
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        
        transformed = preprocessor.transform(sample_features)
        
        # Number of features should match
        assert transformed.shape[1] == len(feature_names)

    def test_model_accepts_preprocessor_output(
        self, sample_features, model_path, preprocessor_path
    ):
        """Test that model can process preprocessor output."""
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Model artifacts not available")
        
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        
        transformed = preprocessor.transform(sample_features)
        
        # Should not raise error
        predictions = model.predict(transformed)
        probabilities = model.predict_proba(transformed)
        
        assert len(predictions) == len(sample_features)
        assert len(probabilities) == len(sample_features)


@pytest.mark.integration
class TestAPIModelIntegration:
    """Tests for API and model integration."""

    def test_api_uses_correct_model(self, api_client, sample_customer_data, model_path, preprocessor_path):
        """Test that API predictions match direct model predictions."""
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Model artifacts not available")
        
        # Get API prediction
        response = api_client.post("/predict", json=sample_customer_data)
        api_prob = response.json()["churn_probability"]
        
        # Get direct model prediction
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Convert customer data to DataFrame
        customer_df = pd.DataFrame([{
            k: v for k, v in sample_customer_data.items() 
            if k != "customerID"
        }])
        
        transformed = preprocessor.transform(customer_df)
        direct_prob = model.predict_proba(transformed)[0, 1]
        
        # Should be approximately equal (allow small floating point differences)
        assert abs(api_prob - direct_prob) < 0.01

    def test_batch_predictions_consistency(self, api_client, batch_customers, model_path, preprocessor_path):
        """Test that batch predictions equal individual predictions."""
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Model artifacts not available")
        
        # Get batch predictions
        batch_response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        batch_probs = [p["churn_probability"] for p in batch_response.json()["predictions"]]
        
        # Get individual predictions
        individual_probs = []
        for customer in batch_customers:
            response = api_client.post("/predict", json=customer)
            individual_probs.append(response.json()["churn_probability"])
        
        # Should be equal
        for batch_prob, individual_prob in zip(batch_probs, individual_probs):
            assert abs(batch_prob - individual_prob) < 0.001
