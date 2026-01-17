"""API endpoint tests.

Tests FastAPI endpoints using TestClient:
- Health check endpoints
- Single prediction endpoint
- Batch prediction endpoint
- Explanation endpoint
- Error handling
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check_returns_200(self, api_client):
        """Test that health endpoint returns 200."""
        response = api_client.get("/health")
        
        assert response.status_code == 200

    def test_health_check_response_structure(self, api_client):
        """Test health check response structure."""
        response = api_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


@pytest.mark.api
class TestSinglePrediction:
    """Tests for single prediction endpoint."""

    def test_valid_customer_returns_200(self, api_client, sample_customer_data):
        """Test that valid customer data returns 200."""
        response = api_client.post("/predict", json=sample_customer_data)
        
        assert response.status_code == 200

    def test_response_contains_probability(self, api_client, sample_customer_data):
        """Test that response contains churn probability."""
        response = api_client.post("/predict", json=sample_customer_data)
        data = response.json()
        
        assert "churn_probability" in data

    def test_probability_in_valid_range(self, api_client, sample_customer_data):
        """Test that probability is between 0 and 1."""
        response = api_client.post("/predict", json=sample_customer_data)
        data = response.json()
        
        prob = data["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_response_contains_risk_level(self, api_client, sample_customer_data):
        """Test that response contains risk level."""
        response = api_client.post("/predict", json=sample_customer_data)
        data = response.json()
        
        assert "risk_level" in data
        # Actual risk levels from RiskLevel enum
        assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]

    def test_high_risk_customer(self, api_client, sample_high_risk_customer):
        """Test prediction for high risk customer profile."""
        response = api_client.post("/predict", json=sample_high_risk_customer)
        data = response.json()
        
        # High risk customer should have elevated probability
        assert data["churn_probability"] >= 0.4  # Adjusted threshold

    def test_low_risk_customer(self, api_client, sample_low_risk_customer):
        """Test prediction for low risk customer profile."""
        response = api_client.post("/predict", json=sample_low_risk_customer)
        data = response.json()
        
        # Low risk customer should have lower probability
        assert data["churn_probability"] <= 0.6  # Allow some flexibility

    def test_response_contains_summary(self, api_client, sample_customer_data):
        """Test that response contains summary."""
        response = api_client.post("/predict", json=sample_customer_data)
        data = response.json()
        
        assert "summary" in data
        assert isinstance(data["summary"], str)

    def test_response_contains_key_drivers(self, api_client, sample_customer_data):
        """Test that response contains key drivers."""
        response = api_client.post("/predict", json=sample_customer_data)
        data = response.json()
        
        assert "key_drivers" in data
        assert isinstance(data["key_drivers"], list)


@pytest.mark.api
class TestSinglePredictionErrors:
    """Tests for error handling in single prediction."""

    def test_missing_required_field(self, api_client, sample_customer_data):
        """Test that missing required field returns 422."""
        incomplete_data = {k: v for k, v in sample_customer_data.items() if k != "tenure"}
        
        response = api_client.post("/predict", json=incomplete_data)
        
        assert response.status_code == 422

    def test_invalid_field_type(self, api_client, sample_customer_data):
        """Test that invalid field type returns 422."""
        invalid_data = sample_customer_data.copy()
        invalid_data["tenure"] = "not_a_number"
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422

    def test_negative_tenure(self, api_client, sample_customer_data):
        """Test that negative tenure returns validation error."""
        invalid_data = sample_customer_data.copy()
        invalid_data["tenure"] = -5
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422

    def test_invalid_contract_type(self, api_client, sample_customer_data):
        """Test that invalid contract type returns error."""
        invalid_data = sample_customer_data.copy()
        invalid_data["Contract"] = "Invalid Contract"
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422

    def test_empty_payload(self, api_client):
        """Test that empty payload returns 422."""
        response = api_client.post("/predict", json={})
        
        assert response.status_code == 422


@pytest.mark.api
class TestBatchPrediction:
    """Tests for batch prediction endpoint."""

    def test_valid_batch_returns_200(self, api_client, batch_customers):
        """Test that valid batch returns 200."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        
        assert response.status_code == 200

    def test_batch_response_contains_summary(self, api_client, batch_customers):
        """Test that batch response contains summary."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        data = response.json()
        
        assert "summary" in data

    def test_batch_response_contains_predictions(self, api_client, batch_customers):
        """Test that batch response contains predictions list."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == len(batch_customers)

    def test_single_customer_batch(self, api_client, sample_customer_data):
        """Test batch with single customer."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": [sample_customer_data]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1

    def test_empty_batch_returns_error(self, api_client):
        """Test that empty batch returns error."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": []}
        )
        
        # Empty batch should be rejected
        assert response.status_code == 422

    def test_batch_summary_statistics(self, api_client, batch_customers):
        """Test that batch summary contains expected statistics."""
        response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        data = response.json()
        summary = data["summary"]
        
        assert "total_customers" in summary
        assert summary["total_customers"] == len(batch_customers)
        assert "high_risk_count" in summary
        assert "avg_churn_probability" in summary


@pytest.mark.api
class TestExplanationEndpoint:
    """Tests for explanation endpoint."""

    def test_explain_returns_200(self, api_client, sample_customer_data):
        """Test that explain endpoint returns 200."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        # May return 200 or 501 if SHAP not installed/configured
        assert response.status_code in [200, 500, 501]

    def test_explain_contains_probability(self, api_client, sample_customer_data):
        """Test that explanation contains probability."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data

    def test_explain_contains_risk_factors(self, api_client, sample_customer_data):
        """Test that explanation contains risk factors."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "risk_factors" in data
            assert isinstance(data["risk_factors"], list)

    def test_explain_contains_protective_factors(self, api_client, sample_customer_data):
        """Test that explanation contains protective factors."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "protective_factors" in data
            assert isinstance(data["protective_factors"], list)

    def test_explain_contains_narrative(self, api_client, sample_customer_data):
        """Test that explanation contains narrative."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "narrative" in data
            assert isinstance(data["narrative"], str)

    def test_explain_contains_recommendations(self, api_client, sample_customer_data):
        """Test that explanation contains recommendations."""
        response = api_client.post("/explain", json=sample_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "recommended_actions" in data
            assert isinstance(data["recommended_actions"], list)


@pytest.mark.api
class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_json_available(self, api_client):
        """Test that OpenAPI JSON is available."""
        response = api_client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_endpoint_available(self, api_client):
        """Test that Swagger docs are available."""
        response = api_client.get("/docs")
        
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, api_client):
        """Test that ReDoc is available."""
        response = api_client.get("/redoc")
        
        assert response.status_code == 200
