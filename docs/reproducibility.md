# Model Reproducibility Guide

This document describes how to reproduce any model version in the Churn Predictor system.

## Reproducibility Framework

### What is Tracked

Every model run captures:

| Component | Location | Version ID |
|-----------|----------|------------|
| Code | GitHub commit | Git SHA |
| Data | `data/processed/` | MD5 hash |
| Features | Feature names | MD5 hash |
| Config | `config/training_config.yaml` | In params |
| Random seed | Training config | 42 (default) |
| Model | MLflow artifacts | Run ID |

### Environment Setup

```bash
# Clone repository
git clone https://github.com/aadilsal/Churn-Predictor.git
cd Churn-Predictor

# Create environment
conda create -n churn python=3.10 -y
conda activate churn

# Install dependencies (exact versions)
pip install -r requirements.txt
```

## Reproducing a Model

### Method 1: From MLflow Run ID

```python
import mlflow
from src.mlops.tracking import init_tracking

# Initialize tracking
init_tracking(use_dagshub=True)

# Load specific run
run_id = "YOUR_RUN_ID_HERE"
run = mlflow.get_run(run_id)

# View parameters used
print(run.data.params)

# Load the model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

### Method 2: From Model Registry Version

```python
from src.mlops.registry import ModelRegistry

registry = ModelRegistry()

# Get lineage for version
lineage = registry.get_model_lineage("churn-predictor", version="1")
print(lineage)

# Load the model
model = registry.load_model_version("churn-predictor", "1")
```

### Method 3: Retrain from Config

```bash
# Run pipeline with same config
python -m src.mlops.pipeline --config config/training_config.yaml

# Or programmatically
python -c "
from src.mlops.pipeline import MLOpsPipeline
pipeline = MLOpsPipeline()
results = pipeline.run()
"
```

## Lineage Information

Each registered model contains:

```json
{
  "model_name": "churn-predictor",
  "version": "1",
  "run_id": "abc123...",
  "dataset_version": "3f7a8b2c",
  "feature_version": "9d2e1f4a",
  "parameters": {
    "model_type": "xgboost",
    "max_depth": "6",
    "learning_rate": "0.1"
  },
  "metrics": {
    "test_roc_auc": 0.8444,
    "test_pr_auc": 0.6529
  }
}
```

## Verifying Reproducibility

```python
# Compare two runs with same config
from src.mlops.registry import ModelRegistry

registry = ModelRegistry()
comparison = registry.compare_versions("churn-predictor", "1", "2")
print(comparison)
```

## Controlled Variables

| Variable | Control Method |
|----------|----------------|
| Random seed | `random_seed: 42` in config |
| Data split | Seed-controlled stratified split |
| Model init | Seed parameter in model |
| CV folds | Seed-controlled StratifiedKFold |

## Known Limitations

1. **XGBoost threading**: Minor variations possible with different thread counts
2. **NumPy/SciPy versions**: Float precision may vary slightly
3. **GPU vs CPU**: Results may differ if using GPU training

## Quick Reproduction Test

```bash
# Run quick test to verify setup
python -m src.mlops.pipeline --quick --local

# Verify metrics match expected
python -c "
from src.mlops.tracking import get_best_run
best = get_best_run('churn-model-training')
print(f'Best ROC-AUC: {best[\"metrics\"].get(\"test_roc_auc\", \"N/A\")}')
"
```
