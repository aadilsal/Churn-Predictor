# Customer Churn Intelligence Platform

> **A production-grade ML system for predicting, explaining, and preventing customer churn**

## 🎯 Problem Statement

Customer churn directly impacts revenue and growth. This platform helps businesses:

- **Identify** which customers will churn (with calibrated probabilities)
- **Understand** WHY each customer is at risk (explainable AI)
- **Predict** WHEN churn is likely to occur (time-to-churn estimates)
- **Monitor** model reliability and data drift continuously
- **Act** on insights through business-friendly dashboards

This is **NOT** a Kaggle notebook project — it's an enterprise-ready system designed for real business deployment.

---

## 🏗️ System Architecture

```
churn_predictor/
├── config/                     # Configuration files
├── data/
│   ├── raw/                   # Immutable source data
│   ├── processed/             # Cleaned, validated data
│   └── features/              # Engineered feature sets
├── notebooks/                 # Exploratory analysis
├── src/
│   ├── data/                  # Data ingestion & validation
│   ├── features/              # Feature engineering
│   ├── models/                # ML models (churn, time-to-churn, explainability)
│   ├── inference/             # Batch and online scoring
│   ├── monitoring/            # Drift detection & alerts
│   └── utils/                 # Shared utilities
├── dashboards/                # Streamlit applications
├── mlops/                     # Training pipelines & experiments
├── tests/                     # Unit & integration tests
├── docker/                    # Containerization
└── docs/                      # Documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd churn_predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
python src/data/download_data.py
```

5. **Run data validation**
```bash
python src/data/ingestion.py
```

### Docker Deployment

1. **Build and start services**
```bash
docker-compose up -d
```

2. **Access services**
- MLflow UI: http://localhost:5000
- Streamlit Dashboard: http://localhost:8501

3. **Stop services**
```bash
docker-compose down
```

---

## 📊 Module Structure

### **Module 1: Data Foundation** ✅ (Current)
- Data acquisition and versioning
- Validation schemas
- Quality checks
- Business-driven EDA

### **Module 2: Feature Engineering** (Upcoming)
- Behavioral features
- Temporal features
- Feature pipelines

### **Module 3: Churn Classification** (Upcoming)
- Binary churn prediction
- Probability calibration
- Model evaluation

### **Module 4: Time-to-Churn Modeling** (Upcoming)
- Survival analysis
- Time-horizon predictions

### **Module 5: Explainability Engine** (Upcoming)
- SHAP values
- Business translations

### **Module 6: Monitoring & Drift Detection** (Upcoming)
- Data drift detection
- Model performance tracking

### **Module 7: Dashboards** (Upcoming)
- Executive dashboard
- Operational dashboard
- Technical dashboard

### **Module 8: MLOps** (Upcoming)
- Training pipelines
- Model registry
- Deployment automation

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-learn, XGBoost, Lifelines |
| **Experiment Tracking** | MLflow |
| **Dashboard** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Data Validation** | Pydantic, Great Expectations |
| **Explainability** | SHAP |

---

## 📈 Key Features

### For Business Stakeholders
✅ Identify top at-risk customers  
✅ Understand churn drivers in plain language  
✅ Track churn trends over time  
✅ Know when predictions are unreliable  

### For ML Engineers
✅ Clean, modular codebase  
✅ Reproducible experiments (MLflow)  
✅ Automated monitoring  
✅ Containerized deployment  

### For Data Scientists
✅ Reusable feature pipelines  
✅ Model versioning  
✅ Explainable predictions  
✅ One-command retraining  

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ --cov=src/

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

---

## 📚 Documentation

- [Implementation Plan](docs/implementation_plan.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Model Documentation](docs/models.md)
- [API Reference](docs/api.md)

---

## 🤝 Contributing

This is a production ML system. All contributions must:
- Include unit tests
- Pass linting (black, flake8, mypy)
- Update documentation
- Follow the existing architecture

---

## 📝 License

[Add your license here]

---

## 🏆 Success Metrics

**Business Impact:**
- Identify 80%+ of churners in top 20% risk scores
- Reduce churn rate by 10%+ through targeted interventions

**Technical Excellence:**
- AUC-ROC > 0.75
- Calibration error < 0.05
- 95%+ test coverage
- Zero-downtime deployments

---

**Built with production ML engineering principles, not Kaggle shortcuts.**
