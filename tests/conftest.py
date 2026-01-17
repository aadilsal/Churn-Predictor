"""Pytest configuration and shared fixtures for churn predictor tests."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# SAMPLE DATA FIXTURES
# ============================================


@pytest.fixture
def sample_customer_data() -> Dict[str, Any]:
    """Valid customer data matching API schema."""
    return {
        "customerID": "TEST-001",
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.0,
    }


@pytest.fixture
def sample_high_risk_customer() -> Dict[str, Any]:
    """Customer with high churn risk profile."""
    return {
        "customerID": "HIGH-RISK-001",
        "gender": "Female",
        "SeniorCitizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.00,
        "TotalCharges": 95.00,
    }


@pytest.fixture
def sample_low_risk_customer() -> Dict[str, Any]:
    """Customer with low churn risk profile."""
    return {
        "customerID": "LOW-RISK-001",
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.00,
        "TotalCharges": 3900.00,
    }


@pytest.fixture
def batch_customers(
    sample_customer_data, sample_high_risk_customer, sample_low_risk_customer
) -> List[Dict[str, Any]]:
    """Batch of multiple customer records."""
    return [sample_customer_data, sample_high_risk_customer, sample_low_risk_customer]


@pytest.fixture
def invalid_customer_missing_fields() -> Dict[str, Any]:
    """Customer data with missing required fields."""
    return {
        "customerID": "INVALID-001",
        "gender": "Male",
        # Missing most required fields
    }


@pytest.fixture
def invalid_customer_wrong_types() -> Dict[str, Any]:
    """Customer data with wrong field types."""
    return {
        "customerID": "INVALID-002",
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "twelve",  # Should be int
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "high",  # Should be float
        "TotalCharges": None,  # Should be float
    }


@pytest.fixture
def edge_case_customer() -> Dict[str, Any]:
    """Customer with edge case values."""
    return {
        "customerID": "EDGE-001",
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,  # New customer
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0,
    }


# ============================================
# DATAFRAME FIXTURES
# ============================================


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Raw dataframe mimicking telco dataset structure."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004", "C005"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes", "No"],
            "tenure": [12, 1, 48, 0, 72],
            "PhoneService": ["Yes", "Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes", "No phone service", "Yes"],
            "InternetService": ["Fiber optic", "DSL", "No", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No internet service", "No internet service", "Yes", "Yes"],
            "OnlineBackup": ["No", "No internet service", "No internet service", "Yes", "Yes"],
            "DeviceProtection": ["No", "No", "No internet service", "No", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes", "Yes"],
            "StreamingTV": ["Yes", "No", "No internet service", "No", "Yes"],
            "StreamingMovies": ["Yes", "No", "No internet service", "Yes", "Yes"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [85.5, 45.0, 25.0, 55.0, 105.0],
            "TotalCharges": ["1026.0", "45.0", "1200.0", "", "7560.0"],  # Note: string with empty
            "Churn": ["Yes", "Yes", "No", "No", "No"],
        }
    )


@pytest.fixture
def sample_cleaned_dataframe(sample_raw_dataframe) -> pd.DataFrame:
    """Pre-cleaned dataframe for testing."""
    df = sample_raw_dataframe.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ============================================
# MODEL FIXTURES
# ============================================


@pytest.fixture
def mock_model_predictions() -> np.ndarray:
    """Mock model probability predictions."""
    return np.array([0.75, 0.85, 0.15, 0.25, 0.10])


@pytest.fixture
def mock_labels() -> np.ndarray:
    """Mock true labels."""
    return np.array([1, 1, 0, 0, 0])


@pytest.fixture
def model_path() -> Path:
    """Path to the trained model."""
    return PROJECT_ROOT / "models" / "final_model.joblib"


@pytest.fixture
def preprocessor_path() -> Path:
    """Path to the feature preprocessor."""
    return PROJECT_ROOT / "models" / "feature_preprocessor.joblib"


@pytest.fixture
def feature_names_path() -> Path:
    """Path to feature names file."""
    return PROJECT_ROOT / "models" / "feature_names.json"


# ============================================
# API TEST FIXTURES
# ============================================


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)


# ============================================
# CLEANUP FIXTURES
# ============================================


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup any test files created during tests."""
    yield
    # Cleanup logic after tests
    test_output_dir = PROJECT_ROOT / "tests" / "output"
    if test_output_dir.exists():
        import shutil
        shutil.rmtree(test_output_dir, ignore_errors=True)


# ============================================
# PYTEST CONFIGURATION HOOKS
# ============================================


def pytest_configure(config):
    """Configure custom pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip slow tests unless explicitly requested
    if not config.getoption("--slow", default=False):
        skip_slow = pytest.mark.skip(reason="Slow test - use --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
