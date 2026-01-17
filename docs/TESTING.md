# Testing Strategy Documentation

Comprehensive testing documentation for the Customer Churn Intelligence Platform.

## Overview

This project uses a **layered testing strategy** to ensure reliability, security, and performance:

| Layer | Purpose | Tools |
|-------|---------|-------|
| Unit Tests | Logic correctness | pytest, pytest-mock |
| Integration | Component interaction | pytest |
| API Tests | Endpoint validation | pytest, TestClient |
| E2E Tests | Full workflows | pytest |
| Performance | Latency/throughput | pytest-benchmark, locust |
| Security | Robustness | pytest |

---

## Quick Start

### Run All Tests
```bash
cd d:\Projects\churn_predictor
pytest tests/ -v
```

### Run by Category
```bash
# Unit tests only (fast)
pytest tests/unit/ -v -m unit

# Integration tests
pytest tests/integration/ -v -m integration

# API tests
pytest tests/api/ -v -m api

# E2E tests
pytest tests/e2e/ -v -m e2e

# Security tests
pytest tests/security/ -v -m security
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html --cov-fail-under=80
# Open htmlcov/index.html for detailed report
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

| File | Tests | Coverage |
|------|-------|----------|
| `test_preprocessing.py` | Data cleaning, validation, splitting | `src/data/preprocessing.py` |
| `test_feature_engineering.py` | Feature transformer, train/test split | `src/models/feature_engineering.py` |
| `test_metrics.py` | Classification metrics, calibration, business | `src/utils/metrics.py` |
| `test_evaluation.py` | Model evaluation, threshold optimization | `src/models/evaluation.py` |

### Model Validation (`tests/model/`)

| File | Tests |
|------|-------|
| `test_model_validation.py` | Model loading, prediction output, stability, accuracy thresholds |

**Performance thresholds:**
- Minimum accuracy: 75%
- Minimum AUC: 0.80

### API Tests (`tests/api/`)

| File | Endpoints |
|------|-----------|
| `test_api_endpoints.py` | `/health`, `/predict`, `/predict/batch`, `/explain` |

**Tested scenarios:**
- Valid inputs
- Missing/invalid fields
- Type validation errors
- Batch size limits

### Integration Tests (`tests/integration/`)

Tests for component interaction:
- Data ingestion → preprocessing → prediction
- Feature pipeline → model inference
- API → model → explainer chain

### E2E Tests (`tests/e2e/`)

Full workflow simulations:
- Single customer prediction workflow
- Batch processing workflow
- High-risk customer with recommendations

### Performance Tests (`tests/performance/`)

**Latency targets:**
- Single prediction: < 100ms
- API single prediction: < 200ms (avg), < 500ms (p95)
- Batch 100 customers: < 5 seconds

**Load testing:**
```bash
cd tests/performance
locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 for UI
```

### Security Tests (`tests/security/`)

- Input validation (SQL injection, XSS)
- Edge cases (extreme values, null bytes)
- Malformed payloads
- Error information leakage

---

## Coverage Requirements

| Metric | Threshold |
|--------|-----------|
| Line coverage | ≥ 80% |
| Branch coverage | ≥ 70% |

Coverage reports are generated in `htmlcov/` directory.

---

## Test Fixtures

Shared fixtures in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `sample_customer_data` | Valid customer record |
| `sample_high_risk_customer` | High churn risk profile |
| `sample_low_risk_customer` | Low churn risk profile |
| `batch_customers` | Multiple customer records |
| `sample_raw_dataframe` | Raw telco dataset format |
| `sample_cleaned_dataframe` | Preprocessed dataframe |
| `api_client` | FastAPI TestClient |
| `model_path`, `preprocessor_path` | Paths to model artifacts |

---

## Adding New Tests

1. **Create test file** in appropriate directory (`tests/unit/`, `tests/api/`, etc.)
2. **Add markers** using `@pytest.mark.<category>`
3. **Use fixtures** from `conftest.py` for common data
4. **Follow naming**: `test_<function>_<scenario>`

Example:
```python
@pytest.mark.unit
class TestNewFeature:
    def test_new_feature_handles_edge_case(self, sample_customer_data):
        result = new_feature(sample_customer_data)
        assert result is not None
```

---

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: coverage.xml
```

---

## Postman Collection

Import `tests/postman/churn_api_collection.json` into Postman for manual API testing.

Includes:
- All endpoints with example requests
- Automated test scripts
- Environment variable support

---

## Troubleshooting

### Tests fail with import errors
```bash
# Ensure project root is in PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;d:\Projects\churn_predictor
```

### Model tests skip
Ensure model artifacts exist in `models/` directory.

### Performance tests slow
Use `--slow` flag to include slow tests:
```bash
pytest tests/performance/ --slow
```
