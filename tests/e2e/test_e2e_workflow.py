"""End-to-end tests for complete user workflows.

Simulates real user interactions from data input to result consumption.
"""

import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.e2e
class TestFullPredictionWorkflow:
    """E2E tests for complete prediction workflow."""

    def test_single_customer_prediction_workflow(self, api_client, sample_customer_data):
        """Test complete workflow: input → predict → explain → consume."""
        # Step 1: Make prediction
        predict_response = api_client.post("/predict", json=sample_customer_data)
        assert predict_response.status_code == 200
        
        prediction = predict_response.json()
        assert "churn_probability" in prediction
        assert "risk_level" in prediction
        assert "summary" in prediction
        
        # Step 2: Get explanation
        explain_response = api_client.post("/explain", json=sample_customer_data)
        assert explain_response.status_code == 200
        
        explanation = explain_response.json()
        assert "risk_factors" in explanation
        assert "protective_factors" in explanation
        assert "recommended_actions" in explanation
        
        # Step 3: Verify consistency
        assert abs(prediction["churn_probability"] - explanation["churn_probability"]) < 0.001

    def test_batch_processing_workflow(self, api_client, batch_customers):
        """Test complete batch processing workflow."""
        # Step 1: Submit batch
        batch_response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        assert batch_response.status_code == 200
        
        result = batch_response.json()
        
        # Step 2: Verify summary statistics
        summary = result["summary"]
        assert summary["total_customers"] == len(batch_customers)
        assert "high_risk_count" in summary
        assert "avg_churn_probability" in summary
        
        # Step 3: Verify individual predictions
        predictions = result["predictions"]
        assert len(predictions) == len(batch_customers)
        
        for pred in predictions:
            assert "churn_probability" in pred
            assert 0 <= pred["churn_probability"] <= 1
            assert "risk_level" in pred

    def test_high_risk_customer_workflow(self, api_client, sample_high_risk_customer):
        """Test workflow for high-risk customer with recommendations."""
        # Predict
        response = api_client.post("/predict", json=sample_high_risk_customer)
        prediction = response.json()
        
        # High-risk customers should get high probability
        assert prediction["churn_probability"] >= 0.5
        
        # Get explanation with recommendations
        explain_response = api_client.post("/explain", json=sample_high_risk_customer)
        explanation = explain_response.json()
        
        # Should have actionable recommendations
        assert len(explanation["recommended_actions"]) > 0
        
        # Risk factors should be identified
        assert len(explanation["risk_factors"]) > 0


@pytest.mark.e2e
class TestDataToInsightsWorkflow:
    """E2E tests for data processing to insights workflow."""

    def test_raw_data_to_predictions(self):
        """Test processing raw data file to predictions."""
        from src.data.preprocessing import clean_telco_dataset
        import joblib
        
        # Check if we have test data
        data_path = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"
        model_path = PROJECT_ROOT / "models" / "final_model.joblib"
        preprocessor_path = PROJECT_ROOT / "models" / "feature_preprocessor.joblib"
        
        if not all(p.exists() for p in [data_path, model_path, preprocessor_path]):
            pytest.skip("Required data/model files not available")
        
        # Load raw data (use processed as proxy)
        df = pd.read_csv(data_path)
        sample = df.head(10)
        
        # Prepare features
        feature_cols = [c for c in sample.columns if c not in ["customerID", "Churn"]]
        X = sample[feature_cols]
        
        # Load model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Transform and predict
        X_transformed = preprocessor.transform(X)
        predictions = model.predict_proba(X_transformed)[:, 1]
        
        # Verify predictions
        assert len(predictions) == len(sample)
        assert all(0 <= p <= 1 for p in predictions)


@pytest.mark.e2e
class TestErrorRecoveryWorkflow:
    """E2E tests for error handling and recovery."""

    def test_invalid_input_provides_useful_error(self, api_client):
        """Test that invalid inputs return actionable error messages."""
        invalid_data = {"tenure": "invalid"}
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422
        error = response.json()
        
        # Should have error detail
        assert "detail" in error

    def test_partial_batch_failure_handling(self, api_client, sample_customer_data):
        """Test handling when some batch items are valid and others aren't."""
        # Create batch with mix of valid and edge case data
        customers = [
            sample_customer_data,
            {**sample_customer_data, "customerID": "EDGE-CASE"},
        ]
        
        response = api_client.post(
            "/predict/batch",
            json={"customers": customers}
        )
        
        # Should still return 200 for valid customers
        assert response.status_code == 200
        
        result = response.json()
        assert len(result["predictions"]) >= 1


@pytest.mark.e2e
class TestConsistencyWorkflow:
    """E2E tests for result consistency."""

    def test_repeated_predictions_are_stable(self, api_client, sample_customer_data):
        """Test that repeated predictions give same results."""
        results = []
        
        for _ in range(5):
            response = api_client.post("/predict", json=sample_customer_data)
            results.append(response.json()["churn_probability"])
        
        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_prediction_explanation_consistency(self, api_client, sample_customer_data):
        """Test that prediction and explanation are consistent."""
        predict_response = api_client.post("/predict", json=sample_customer_data)
        explain_response = api_client.post("/explain", json=sample_customer_data)
        
        pred = predict_response.json()
        expl = explain_response.json()
        
        # Probabilities should match
        assert pred["churn_probability"] == expl["churn_probability"]
        
        # Risk levels should match
        assert pred["risk_level"] == expl["risk_level"]
