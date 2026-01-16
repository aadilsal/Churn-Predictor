# Quick Start Guide

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- Docker and Docker Compose (optional, for containerized deployment)

---

## Local Setup (Recommended for Development)

### Step 1: Clone Repository

```bash
cd d:\Projects\churn_predictor
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Environment File

```bash
# Copy example environment file
copy .env.example .env

# Edit .env with your settings (optional)
```

### Step 5: Run Data Ingestion Pipeline

```bash
# Download and process data
python src/data/ingestion.py
```

**Expected Output:**
```
================================================================================
STARTING DATA INGESTION PIPELINE
================================================================================
Downloading dataset from https://raw.githubusercontent.com/...
Dataset downloaded successfully to data/raw/telco_churn.csv
Dataset shape: (7043, 21)
...
DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

### Step 6: Explore Data with Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Open: notebooks/01_eda.ipynb
```

---

## Docker Setup (Recommended for Production)

### Step 1: Build Containers

```bash
docker-compose build
```

### Step 2: Start Services

```bash
# Start MLflow and application containers
docker-compose up -d
```

### Step 3: Verify Services

```bash
# Check running containers
docker-compose ps

# Access MLflow UI
# Open browser: http://localhost:5000
```

### Step 4: Run Data Pipeline in Container

```bash
# Execute ingestion pipeline
docker-compose exec app python src/data/ingestion.py
```

### Step 5: Stop Services

```bash
docker-compose down
```

---

## Verify Installation

### Check Python Environment

```bash
python --version
# Should show: Python 3.10.x or higher
```

### Check Installed Packages

```bash
pip list | findstr "pandas scikit-learn mlflow"
```

### Run Quick Test

```python
# test_setup.py
from src.utils.config import config
from src.utils.logging import logger

logger.info("Setup test successful!")
print(f"Data path: {config.data_raw_path}")
print(f"MLflow URI: {config.mlflow_tracking_uri}")
```

```bash
python test_setup.py
```

---

## Directory Structure After Setup

```
churn_predictor/
├── venv/                          # Virtual environment (if using local setup)
├── data/
│   ├── raw/
│   │   └── telco_churn.csv       # Downloaded dataset
│   └── processed/
│       ├── telco_churn_processed.csv
│       ├── data_quality_report.json
│       └── data_quality_summary.txt
├── logs/
│   └── churn_predictor.log       # Application logs
├── mlruns/                        # MLflow tracking (created on first use)
└── notebooks/                     # Jupyter notebooks
```

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in the project root directory
cd d:\Projects\churn_predictor

# Activate virtual environment
.\venv\Scripts\activate
```

### Issue: Dataset download fails

**Solution:**
```bash
# Manually download dataset
curl -o data/raw/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

### Issue: Permission denied errors

**Solution:**
```bash
# Create directories manually
mkdir data\raw data\processed logs
```

### Issue: Docker containers won't start

**Solution:**
```bash
# Check Docker is running
docker --version

# View logs
docker-compose logs

# Rebuild containers
docker-compose build --no-cache
```

---

## Next Steps

After successful setup:

1. ✅ **Explore Data**: Open `notebooks/01_eda.ipynb`
2. ✅ **Review Documentation**: Read `docs/module_01_data_foundation.md`
3. ✅ **Check Data Quality**: View `data/processed/data_quality_summary.txt`
4. ⏭️ **Module 2**: Feature Engineering (coming next)

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Logs**: Check `logs/churn_predictor.log`
- **Issues**: Review error messages in terminal

---

## Development Workflow

```bash
# 1. Activate environment
.\venv\Scripts\activate

# 2. Make changes to code

# 3. Run tests (when available)
pytest tests/

# 4. Run linting
black src/
flake8 src/

# 5. Commit changes
git add .
git commit -m "Description of changes"
```

---

**You're all set! 🚀**
