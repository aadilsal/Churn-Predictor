# Module 1: Data Foundation - Documentation

## Overview

This module establishes the data foundation for the Customer Churn Intelligence Platform. It implements a production-grade data pipeline with validation, quality checks, and versioning.

## Components

### 1. Data Download (`src/data/download_data.py`)

**Purpose:** Automated dataset acquisition with verification

**Features:**
- Downloads Telco Customer Churn dataset from IBM GitHub
- Calculates MD5 hash for versioning
- Validates CSV format
- Prevents redundant downloads

**Usage:**
```python
from src.data.download_data import download_telco_churn_dataset

# Download dataset
dataset_path = download_telco_churn_dataset()
```

**Command Line:**
```bash
python src/data/download_data.py
```

---

### 2. Data Validation (`src/data/validation.py`)

**Purpose:** Schema enforcement using Pydantic

**Features:**
- Type validation for all columns
- Enum constraints for categorical variables
- Range validation for numerical fields
- Custom validators for edge cases (e.g., TotalCharges)

**Key Schemas:**
- `TelcoCustomerRecord`: Individual customer validation
- `DataQualityReport`: Quality metrics structure

**Example:**
```python
from src.data.validation import TelcoCustomerRecord

# Validate a single record
record = TelcoCustomerRecord(**customer_data)
```

---

### 3. Data Preprocessing (`src/data/preprocessing.py`)

**Purpose:** Data cleaning and transformation

**Key Functions:**

#### `clean_telco_dataset(df)`
Performs the following transformations:
1. Converts `TotalCharges` to numeric (handles empty strings)
2. Imputes missing `TotalCharges` using `MonthlyCharges * tenure`
3. Standardizes `SeniorCitizen` to Yes/No format
4. Normalizes "No internet service" → "No"
5. Normalizes "No phone service" → "No"
6. Converts `Churn` to binary (0/1)

#### `validate_data_quality(df)`
Generates comprehensive quality report:
- Missing value counts
- Duplicate detection
- Outlier identification (3*IQR method)
- Categorical value distributions

#### `split_features_target(df)`
Separates features from target variable

**Usage:**
```python
from src.data.preprocessing import clean_telco_dataset, validate_data_quality

# Clean data
df_clean = clean_telco_dataset(df_raw)

# Check quality
quality_report = validate_data_quality(df_clean)
```

---

### 4. Data Ingestion Pipeline (`src/data/ingestion.py`)

**Purpose:** End-to-end data pipeline orchestration

**Pipeline Steps:**
1. Download raw data (if needed)
2. Load raw data
3. Validate raw data quality
4. Clean data
5. Validate cleaned data quality
6. Save processed data
7. Generate quality reports

**Outputs:**
- `data/processed/telco_churn_processed.csv` - Cleaned dataset
- `data/processed/data_quality_report.json` - Machine-readable report
- `data/processed/data_quality_summary.txt` - Human-readable summary

**Usage:**
```python
from src.data.ingestion import ingest_data, load_processed_data

# Run full pipeline
df = ingest_data()

# Or load existing processed data
df = load_processed_data()
```

**Command Line:**
```bash
python src/data/ingestion.py
```

---

## Data Quality Report

After running the ingestion pipeline, you'll find:

### JSON Report (`data_quality_report.json`)
```json
{
  "timestamp": "2026-01-14T22:45:00",
  "raw_data_quality": {
    "total_records": 7043,
    "missing_values": {"TotalCharges": 11},
    ...
  },
  "cleaned_data_quality": {
    "total_records": 7043,
    "missing_values": {},
    ...
  }
}
```

### Text Summary (`data_quality_summary.txt`)
Human-readable summary with:
- Record counts
- Missing value percentages
- Data quality improvements

---

## Dataset Characteristics

### Source
- **Name:** Telco Customer Churn
- **Source:** IBM GitHub Repository
- **Records:** 7,043 customers
- **Features:** 21 columns

### Target Variable
- **Column:** `Churn`
- **Type:** Binary (0 = Retained, 1 = Churned)
- **Distribution:** ~27% churn rate (class imbalance)

### Feature Categories

#### Demographics (4 features)
- `gender`: Male/Female
- `SeniorCitizen`: Yes/No
- `Partner`: Yes/No
- `Dependents`: Yes/No

#### Account Information (2 features)
- `tenure`: Months with company (0-72)
- `PhoneService`: Yes/No

#### Services (9 features)
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No
- `OnlineBackup`: Yes/No
- `DeviceProtection`: Yes/No
- `TechSupport`: Yes/No
- `StreamingTV`: Yes/No
- `StreamingMovies`: Yes/No
- `MultipleLines`: Yes/No

#### Billing (4 features)
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: 4 categories
- `MonthlyCharges`: Continuous ($18-$119)
- `TotalCharges`: Continuous ($18-$8,684)

---

## Key Findings from EDA

### 🎯 Top Churn Drivers

1. **Contract Type**
   - Month-to-month: 42.7% churn
   - One year: 11.3% churn
   - Two year: 2.8% churn

2. **Tenure**
   - Avg tenure (churned): 17.9 months
   - Avg tenure (retained): 37.6 months
   - High risk in first 12 months

3. **Internet Service**
   - Fiber optic: 41.9% churn
   - DSL: 18.9% churn
   - No internet: 7.4% churn

4. **Value-Added Services**
   - No online security: 41.8% churn
   - No tech support: 41.7% churn
   - With services: <20% churn

5. **Payment Method**
   - Electronic check: 45.3% churn
   - Automatic payments: <20% churn

6. **Demographics**
   - Senior citizens: 41.7% churn
   - No partner: 33.0% churn
   - No dependents: 31.3% churn

---

## Data Limitations & Risks

### ⚠️ Known Issues

1. **Class Imbalance**
   - 73% retained vs 27% churned
   - Will require balancing strategies in modeling

2. **Missing TotalCharges**
   - 11 records with empty strings
   - Resolved via imputation (MonthlyCharges * tenure)

3. **Temporal Limitations**
   - No timestamp data
   - Cannot model time-to-churn directly
   - Will use tenure as proxy

4. **Feature Interactions**
   - "No internet service" appears in multiple columns
   - Standardized to "No" for consistency

---

## Readiness Assessment

### ✅ Ready for Modeling

**Strengths:**
- Clean, validated data
- Clear churn signals
- Multiple predictive features
- Well-understood business context

**Next Steps:**
1. Feature engineering (Module 2)
2. Model training (Module 3)
3. Explainability (Module 4)

---

## Running the Pipeline

### Option 1: Python Script
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run ingestion pipeline
python src/data/ingestion.py
```

### Option 2: Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_eda.ipynb
```

### Option 3: Docker
```bash
# Build and start containers
docker-compose up -d

# Run ingestion in container
docker-compose exec app python src/data/ingestion.py
```

---

## File Outputs

After running Module 1, you should have:

```
data/
├── raw/
│   └── telco_churn.csv                    # Original dataset
└── processed/
    ├── telco_churn_processed.csv          # Cleaned dataset
    ├── data_quality_report.json           # Quality metrics (JSON)
    └── data_quality_summary.txt           # Quality summary (text)

logs/
└── churn_predictor.log                    # Pipeline logs
```

---

## Troubleshooting

### Issue: Dataset download fails
**Solution:** Check internet connection or manually download from:
```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

### Issue: Import errors
**Solution:** Ensure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

### Issue: Permission errors
**Solution:** Ensure write permissions for `data/` and `logs/` directories

---

## Next Module

**Module 2: Feature Engineering**
- Behavioral features
- Temporal features
- Feature pipelines
- Feature validation
