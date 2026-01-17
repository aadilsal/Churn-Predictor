"""Performance tests and benchmarks.

Tests for latency, throughput, and scalability.
Uses pytest-benchmark for Python performance tests.
"""

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.performance
class TestPredictionLatency:
    """Tests for prediction latency."""

    @pytest.fixture
    def model(self):
        """Load model for benchmarking."""
        model_path = PROJECT_ROOT / "models" / "final_model.joblib"
        if not model_path.exists():
            pytest.skip("Model not available")
        return joblib.load(model_path)

    @pytest.fixture
    def preprocessor(self):
        """Load preprocessor for benchmarking."""
        path = PROJECT_ROOT / "models" / "feature_preprocessor.joblib"
        if not path.exists():
            pytest.skip("Preprocessor not available")
        return joblib.load(path)

    @pytest.fixture
    def sample_input(self):
        """Create sample input for benchmarking."""
        return pd.DataFrame({
            "gender": ["Male"],
            "SeniorCitizen": ["No"],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
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
            "MonthlyCharges": [85.50],
            "TotalCharges": [1026.0],
        })

    def test_single_prediction_latency(self, model, preprocessor, sample_input, benchmark=None):
        """Benchmark single prediction latency."""
        X = preprocessor.transform(sample_input)
        
        if benchmark:
            result = benchmark(model.predict_proba, X)
        else:
            # Manual timing if benchmark not available
            times = []
            for _ in range(100):
                start = time.perf_counter()
                model.predict_proba(X)
                times.append(time.perf_counter() - start)
            
            avg_time_ms = np.mean(times) * 1000
            
            # Should complete in under 100ms
            assert avg_time_ms < 100, f"Prediction took {avg_time_ms:.2f}ms, exceeds 100ms target"

    def test_preprocessing_latency(self, preprocessor, sample_input, benchmark=None):
        """Benchmark preprocessing latency."""
        if benchmark:
            benchmark(preprocessor.transform, sample_input)
        else:
            times = []
            for _ in range(100):
                start = time.perf_counter()
                preprocessor.transform(sample_input)
                times.append(time.perf_counter() - start)
            
            avg_time_ms = np.mean(times) * 1000
            
            # Preprocessing should be fast
            assert avg_time_ms < 50, f"Preprocessing took {avg_time_ms:.2f}ms"

    def test_batch_prediction_scaling(self, model, preprocessor, sample_input):
        """Test that batch predictions scale linearly."""
        # Create batches of different sizes
        batch_sizes = [1, 10, 50, 100]
        times_per_sample = []
        
        for size in batch_sizes:
            batch = pd.concat([sample_input] * size, ignore_index=True)
            X = preprocessor.transform(batch)
            
            # Time prediction
            start = time.perf_counter()
            model.predict_proba(X)
            elapsed = time.perf_counter() - start
            
            times_per_sample.append(elapsed / size)
        
        # Time per sample should not increase dramatically with batch size
        # Allow up to 5x variation
        assert max(times_per_sample) / min(times_per_sample) < 5


@pytest.mark.performance
class TestAPILatency:
    """Tests for API endpoint latency."""

    def test_single_prediction_api_latency(self, api_client, sample_customer_data):
        """Test API single prediction latency."""
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            response = api_client.post("/predict", json=sample_customer_data)
            elapsed = (time.perf_counter() - start) * 1000
            
            assert response.status_code == 200
            times.append(elapsed)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        # Average should be under 200ms, p95 under 500ms
        assert avg_time < 200, f"Average latency {avg_time:.0f}ms exceeds 200ms"
        assert p95_time < 500, f"P95 latency {p95_time:.0f}ms exceeds 500ms"

    def test_batch_prediction_api_latency(self, api_client, batch_customers):
        """Test API batch prediction latency for small batch."""
        start = time.perf_counter()
        response = api_client.post(
            "/predict/batch",
            json={"customers": batch_customers}
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        
        # Small batch (3 customers) should complete quickly
        assert elapsed < 1000, f"Batch prediction took {elapsed:.0f}ms"

    def test_large_batch_prediction_latency(self, api_client, sample_customer_data):
        """Test API batch prediction latency for larger batch."""
        # Create batch of 100 customers
        customers = [sample_customer_data.copy() for _ in range(100)]
        for i, c in enumerate(customers):
            c["customerID"] = f"BATCH-{i:03d}"
        
        start = time.perf_counter()
        response = api_client.post(
            "/predict/batch",
            json={"customers": customers}
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        
        # 100 customers should complete in under 5 seconds
        assert elapsed < 5000, f"Batch of 100 took {elapsed:.0f}ms, exceeds 5s target"


@pytest.mark.performance
class TestThroughput:
    """Tests for prediction throughput."""

    def test_sustained_throughput(self, api_client, sample_customer_data):
        """Test sustained prediction throughput."""
        duration = 5  # seconds
        requests = 0
        errors = 0
        
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            response = api_client.post("/predict", json=sample_customer_data)
            if response.status_code == 200:
                requests += 1
            else:
                errors += 1
        
        elapsed = time.perf_counter() - start
        throughput = requests / elapsed
        error_rate = errors / (requests + errors) if (requests + errors) > 0 else 0
        
        # Should achieve at least 10 requests per second
        assert throughput >= 10, f"Throughput {throughput:.1f} req/s below target"
        
        # Error rate should be minimal
        assert error_rate < 0.01, f"Error rate {error_rate:.1%} too high"

    def test_health_endpoint_throughput(self, api_client):
        """Test health endpoint throughput."""
        num_requests = 100
        
        start = time.perf_counter()
        for _ in range(num_requests):
            response = api_client.get("/health")
            assert response.status_code == 200
        elapsed = time.perf_counter() - start
        
        throughput = num_requests / elapsed
        
        # Health endpoint should be very fast
        assert throughput >= 100, f"Health throughput {throughput:.1f} req/s"


@pytest.mark.performance
@pytest.mark.slow
class TestStressConditions:
    """Tests under stress conditions."""

    def test_many_sequential_requests(self, api_client, sample_customer_data):
        """Test many sequential requests don't degrade."""
        num_requests = 200
        latencies = []
        
        for _ in range(num_requests):
            start = time.perf_counter()
            response = api_client.post("/predict", json=sample_customer_data)
            latencies.append((time.perf_counter() - start) * 1000)
            
            assert response.status_code == 200
        
        # Compare first 50 and last 50 latencies
        first_avg = np.mean(latencies[:50])
        last_avg = np.mean(latencies[-50:])
        
        # Latency shouldn't degrade more than 50%
        assert last_avg < first_avg * 1.5, "Latency degradation detected"
