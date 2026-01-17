"""Security and robustness tests.

Tests for input validation, edge cases, and security vulnerabilities.
"""

import json
import pytest


@pytest.mark.security
class TestInputValidation:
    """Tests for input validation security."""

    def test_sql_injection_in_customer_id(self, api_client, sample_customer_data):
        """Test that SQL injection attempts are handled safely."""
        malicious_data = sample_customer_data.copy()
        malicious_data["customerID"] = "'; DROP TABLE customers; --"
        
        response = api_client.post("/predict", json=malicious_data)
        
        # Should either succeed (treating it as normal string) or return validation error
        # Should NOT cause server error
        assert response.status_code in [200, 422]

    def test_xss_in_string_fields(self, api_client, sample_customer_data):
        """Test that XSS attempts are handled safely."""
        malicious_data = sample_customer_data.copy()
        malicious_data["customerID"] = "<script>alert('xss')</script>"
        
        response = api_client.post("/predict", json=malicious_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            # If accepted, should not reflect raw script in response
            assert "<script>" not in response.text

    def test_very_long_customer_id(self, api_client, sample_customer_data):
        """Test handling of very long customer ID."""
        malicious_data = sample_customer_data.copy()
        malicious_data["customerID"] = "A" * 10000  # Very long string
        
        response = api_client.post("/predict", json=malicious_data)
        
        # Should handle gracefully (accept or reject, not crash)
        assert response.status_code in [200, 400, 422]

    def test_unicode_in_fields(self, api_client, sample_customer_data):
        """Test Unicode handling in string fields."""
        unicode_data = sample_customer_data.copy()
        unicode_data["customerID"] = "客户-001-🎉"
        
        response = api_client.post("/predict", json=unicode_data)
        
        # Should handle Unicode gracefully
        assert response.status_code in [200, 422]

    def test_null_bytes_in_strings(self, api_client, sample_customer_data):
        """Test null byte injection handling."""
        malicious_data = sample_customer_data.copy()
        malicious_data["customerID"] = "customer\x00id"
        
        response = api_client.post("/predict", json=malicious_data)
        
        # Should not cause server error
        assert response.status_code != 500


@pytest.mark.security
class TestEdgeCaseInputs:
    """Tests for edge case inputs."""

    def test_negative_tenure(self, api_client, sample_customer_data):
        """Test that negative tenure is rejected."""
        invalid_data = sample_customer_data.copy()
        invalid_data["tenure"] = -10
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422

    def test_extremely_large_tenure(self, api_client, sample_customer_data):
        """Test extremely large tenure value."""
        edge_data = sample_customer_data.copy()
        edge_data["tenure"] = 999999
        
        response = api_client.post("/predict", json=edge_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 422]

    def test_zero_monthly_charges(self, api_client, sample_customer_data):
        """Test zero monthly charges."""
        edge_data = sample_customer_data.copy()
        edge_data["MonthlyCharges"] = 0.0
        edge_data["TotalCharges"] = 0.0
        
        response = api_client.post("/predict", json=edge_data)
        
        assert response.status_code == 200

    def test_negative_charges(self, api_client, sample_customer_data):
        """Test that negative charges are rejected."""
        invalid_data = sample_customer_data.copy()
        invalid_data["MonthlyCharges"] = -50.0
        
        response = api_client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422

    def test_very_large_charges(self, api_client, sample_customer_data):
        """Test very large monetary values."""
        edge_data = sample_customer_data.copy()
        edge_data["MonthlyCharges"] = 999999.99
        edge_data["TotalCharges"] = 999999999.99
        
        response = api_client.post("/predict", json=edge_data)
        
        # Should handle or reject, not crash
        assert response.status_code in [200, 422]

    def test_float_overflow(self, api_client, sample_customer_data):
        """Test float overflow values."""
        edge_data = sample_customer_data.copy()
        edge_data["MonthlyCharges"] = 1e308  # Near max float
        
        response = api_client.post("/predict", json=edge_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


@pytest.mark.security
class TestMalformedPayloads:
    """Tests for malformed payload handling."""

    def test_empty_json_object(self, api_client):
        """Test empty JSON object."""
        response = api_client.post("/predict", json={})
        
        assert response.status_code == 422

    def test_null_json(self, api_client):
        """Test null JSON payload."""
        response = api_client.post(
            "/predict",
            content="null",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]

    def test_array_instead_of_object(self, api_client, sample_customer_data):
        """Test array instead of object for single prediction."""
        response = api_client.post("/predict", json=[sample_customer_data])
        
        assert response.status_code == 422

    def test_invalid_json_syntax(self, api_client):
        """Test invalid JSON syntax."""
        response = api_client.post(
            "/predict",
            content="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_missing_content_type(self, api_client, sample_customer_data):
        """Test request without content type."""
        response = api_client.post(
            "/predict",
            content=json.dumps(sample_customer_data)
        )
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 400, 415, 422]

    def test_wrong_http_method(self, api_client, sample_customer_data):
        """Test wrong HTTP method."""
        response = api_client.get("/predict")
        
        assert response.status_code == 405  # Method Not Allowed


@pytest.mark.security
class TestBatchSecurityLimits:
    """Tests for batch size limits and DoS protection."""

    def test_batch_size_limit(self, api_client, sample_customer_data):
        """Test that batch size limits are enforced."""
        # Try to send 2000 customers (over 1000 limit)
        customers = [sample_customer_data.copy() for _ in range(2000)]
        for i, c in enumerate(customers):
            c["customerID"] = f"BATCH-{i}"
        
        response = api_client.post(
            "/predict/batch",
            json={"customers": customers}
        )
        
        # Should be rejected due to size limit
        assert response.status_code == 422

    def test_deeply_nested_payload(self, api_client, sample_customer_data):
        """Test handling of deeply nested JSON."""
        # This tests parser limits
        nested = sample_customer_data.copy()
        nested["nested"] = {"level": sample_customer_data.copy()}
        
        response = api_client.post("/predict", json=nested)
        
        # Should handle (extra fields ignored or rejected)
        assert response.status_code in [200, 422]


@pytest.mark.security
class TestErrorLeakage:
    """Tests for information leakage in errors."""

    def test_error_does_not_leak_internal_paths(self, api_client):
        """Test that errors don't leak internal file paths."""
        response = api_client.post("/predict", json={})
        
        error_text = response.text.lower()
        
        # Should not leak internal paths
        assert "c:\\" not in error_text
        assert "d:\\" not in error_text
        assert "/home/" not in error_text
        assert "/usr/" not in error_text

    def test_error_does_not_leak_stack_traces(self, api_client):
        """Test that errors don't leak full stack traces."""
        response = api_client.post("/predict", json={"invalid": "data"})
        
        error_text = response.text.lower()
        
        # Should not contain stack trace indicators
        assert "traceback" not in error_text
        assert "file \"" not in error_text

    def test_error_does_not_leak_model_info(self, api_client):
        """Test that errors don't leak model implementation details."""
        response = api_client.post("/predict", json={})
        
        error_text = response.text.lower()
        
        # Should not leak model details
        assert "xgboost" not in error_text
        assert "sklearn" not in error_text
        assert "joblib" not in error_text
