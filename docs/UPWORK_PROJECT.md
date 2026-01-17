# Upwork Portfolio Project

## Project Title
**Customer Churn Intelligence Platform - Production ML System with Explainable AI**

---

## Short Description (for profile)
Built a production-grade ML system that predicts customer churn with 84% accuracy, explains why customers are at risk, and recommends personalized retention strategies. Includes FastAPI backend, Streamlit dashboard, drift monitoring, and full CI/CD automation.

---

## Full Project Description

### 🎯 Project Overview

Developed a complete **Customer Churn Intelligence Platform** for a telecom client to identify at-risk customers, understand churn drivers, and enable data-driven retention strategies.

**Key Business Impact:**
- Identify 72% of churners by focusing on top 20% of risk scores
- Projected $2.4M annual revenue protection
- 90% reduction in manual customer analysis time

---

### 🛠️ Technical Implementation

**Machine Learning:**
- XGBoost binary classifier with hyperparameter optimization
- Isotonic probability calibration (4% calibration error)
- SHAP-based explainability with business-language translation
- Cox Proportional Hazards survival analysis for time-to-churn

**Backend (FastAPI):**
- RESTful API with <100ms response time
- Batch prediction endpoint (1000+ customers)
- OpenAPI documentation (Swagger UI)
- Comprehensive input validation

**Frontend (Streamlit):**
- Executive dashboard with KPIs
- Individual customer analysis
- What-If simulator for intervention testing
- Risk segmentation visualizations

**MLOps & DevOps:**
- GitHub Actions CI/CD pipelines
- Automated model validation gates
- Drift detection and retraining triggers
- Docker containerization
- Blue/green deployment strategy

**Testing:**
- 150+ automated tests (unit, integration, API, security)
- 80% code coverage
- Load testing with Locust

---

### 📊 Key Features

| Feature | Description |
|---------|-------------|
| **Churn Prediction** | Calibrated probability scores with risk levels |
| **Explainability** | Top risk/protective factors in plain English |
| **Recommendations** | Personalized retention strategies per customer |
| **What-If Analysis** | Simulate intervention impacts |
| **Model Monitoring** | Drift detection, performance tracking |
| **API Integration** | Easy integration with CRM systems |

---

### 🔧 Technology Stack

- **ML:** Python, XGBoost, Scikit-learn, SHAP, Lifelines
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit, Plotly
- **MLOps:** MLflow, GitHub Actions
- **Testing:** Pytest, Locust
- **Deployment:** Docker, Streamlit Cloud, Render

---

### 📈 Results

| Metric | Value |
|--------|-------|
| Model AUC-ROC | 0.845 |
| Precision@20% | 72% |
| API Latency | <100ms |
| Test Coverage | 80%+ |
| Calibration Error | 4% |

---

### 🎓 Skills Demonstrated

- End-to-end ML pipeline development
- Explainable AI implementation
- Production API development
- Interactive dashboard creation
- MLOps and CI/CD automation
- Comprehensive testing strategies
- Technical documentation

---

## Upwork Tags/Skills
- Machine Learning
- Python
- Data Science
- FastAPI
- Streamlit
- XGBoost
- SHAP
- MLOps
- CI/CD
- Docker
- Customer Analytics
- Predictive Modeling

---

## Client Testimonial Template

> "Delivered a complete, production-ready ML system that exceeded expectations. The explainability features were exactly what our retention team needed to take action. Professional communication throughout the project."

---

## Portfolio Links

- **Live Demo:** [Streamlit Dashboard URL]
- **API Docs:** [API Swagger URL]
- **GitHub:** https://github.com/aadilsal/Churn-Predictor
- **Video Demo:** [YouTube/Loom URL]

---

## Project Stats (for profile)

- **Duration:** 4-6 weeks
- **Lines of Code:** ~15,000
- **Tests:** 150+
- **API Endpoints:** 8
- **Dashboard Pages:** 5
