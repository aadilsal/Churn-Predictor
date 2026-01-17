# System Architecture

This document describes the architecture of the Customer Churn Intelligence Platform, including system design, data flow, and component interactions.

---

## High-Level System Design

```mermaid
flowchart TB
    subgraph Data["Data Layer"]
        RAW[("Raw Data")]
        PROC[("Processed Data")]
        FEAT[("Features")]
    end
    
    subgraph ML["ML Layer"]
        TRAIN["Training Pipeline"]
        MODEL[("Model Registry")]
        PRED["Prediction Service"]
    end
    
    subgraph EXPLAIN["Explainability"]
        SHAP["SHAP Explainer"]
        BUSINESS["Business Insights"]
    end
    
    subgraph MONITOR["Monitoring"]
        DRIFT["Drift Detection"]
        PERF["Performance Tracking"]
        ALERT["Alerting"]
    end
    
    subgraph UI["User Interfaces"]
        API["FastAPI"]
        DASH["Streamlit Dashboard"]
    end
    
    RAW --> PROC
    PROC --> FEAT
    FEAT --> TRAIN
    TRAIN --> MODEL
    MODEL --> PRED
    PRED --> SHAP
    SHAP --> BUSINESS
    
    PRED --> API
    BUSINESS --> API
    API --> DASH
    
    PRED --> DRIFT
    DRIFT --> ALERT
    PERF --> ALERT
```

---

## Component Overview

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| **Data Ingestion** | Python, Pandas | Load, validate, clean raw data |
| **Feature Engineering** | Scikit-learn | Transform features for ML models |
| **Model Training** | XGBoost, MLflow | Train and version models |
| **Prediction Service** | FastAPI | Real-time and batch inference |
| **Explainability** | SHAP | Generate prediction explanations |
| **Monitoring** | Custom | Track drift and performance |
| **Dashboard** | Streamlit | Interactive visualizations |

---

## Data Flow Pipeline

```mermaid
flowchart LR
    subgraph Ingestion
        A[Raw CSV/API] --> B[Validation]
        B --> C[Cleaning]
        C --> D[Quality Checks]
    end
    
    subgraph Features
        D --> E[Encoding]
        E --> F[Scaling]
        F --> G[Feature Store]
    end
    
    subgraph Prediction
        G --> H[Preprocessor]
        H --> I[XGBoost Model]
        I --> J[Probability]
        J --> K[Risk Level]
    end
    
    subgraph Explanation
        J --> L[SHAP Values]
        L --> M[Risk Factors]
        M --> N[Recommendations]
    end
    
    style Ingestion fill:#e1f5fe
    style Features fill:#e8f5e9
    style Prediction fill:#fff3e0
    style Explanation fill:#fce4ec
```

### Data Flow Stages

1. **Ingestion**: Raw data loaded, validated against Pydantic schemas, cleaned
2. **Features**: Categorical encoding (OneHot), numerical scaling (Standard)
3. **Prediction**: Model inference with probability calibration
4. **Explanation**: SHAP values converted to business insights

---

## Model Lifecycle

```mermaid
flowchart TB
    subgraph Development
        A[Data Collection] --> B[EDA & Analysis]
        B --> C[Feature Engineering]
        C --> D[Model Training]
        D --> E[Evaluation]
    end
    
    subgraph Deployment
        E -->|Meets Criteria| F[Model Registry]
        F --> G[Production Deployment]
        G --> H[Inference Service]
    end
    
    subgraph Monitoring
        H --> I[Prediction Logging]
        I --> J[Drift Detection]
        J --> K{Drift Detected?}
    end
    
    subgraph Retraining
        K -->|Yes| L[Trigger Retrain]
        L --> A
        K -->|No| M[Continue Serving]
    end
    
    style Development fill:#e3f2fd
    style Deployment fill:#e8f5e9
    style Monitoring fill:#fff8e1
    style Retraining fill:#ffebee
```

### Lifecycle Stages

| Stage | Description |
|-------|-------------|
| **Development** | Data analysis, feature creation, model training |
| **Deployment** | Validated models pushed to registry and production |
| **Monitoring** | Continuous drift and performance tracking |
| **Retraining** | Automatic trigger when drift exceeds thresholds |

---

## API Architecture

```mermaid
flowchart LR
    CLIENT[Client App] --> LB[Load Balancer]
    
    subgraph API["FastAPI Service"]
        LB --> AUTH[Authentication]
        AUTH --> VALID[Validation]
        VALID --> ROUTE[Router]
        
        ROUTE --> SINGLE["/predict"]
        ROUTE --> BATCH["/predict/batch"]
        ROUTE --> EXPLAIN["/explain"]
        ROUTE --> HEALTH["/health"]
    end
    
    subgraph Backend
        SINGLE --> PREPROC[Preprocessor]
        BATCH --> PREPROC
        EXPLAIN --> PREPROC
        PREPROC --> MODEL[Model]
        MODEL --> SHAP[SHAP]
    end
    
    subgraph Response
        MODEL --> JSON[JSON Response]
        SHAP --> JSON
    end
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and model status |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions (up to 1000) |
| `/explain` | POST | SHAP-based explanation |
| `/docs` | GET | Swagger UI |

---

## Monitoring Architecture

```mermaid
flowchart TB
    subgraph Inputs
        PRED[Predictions] --> LOG[Prediction Logs]
        TRUTH[Ground Truth] --> LOG
    end
    
    subgraph Detection
        LOG --> DATA_DRIFT[Data Drift Check]
        LOG --> PRED_DRIFT[Prediction Drift Check]
        LOG --> PERF_CHECK[Performance Check]
    end
    
    subgraph Thresholds
        DATA_DRIFT --> T1{PSI > 0.1?}
        PRED_DRIFT --> T2{KS > 0.1?}
        PERF_CHECK --> T3{AUC < 0.75?}
    end
    
    subgraph Actions
        T1 -->|Yes| ALERT1[Data Drift Alert]
        T2 -->|Yes| ALERT2[Prediction Drift Alert]
        T3 -->|Yes| ALERT3[Performance Alert]
        ALERT1 --> RETRAIN[Trigger Retraining]
        ALERT2 --> RETRAIN
        ALERT3 --> RETRAIN
    end
```

### Monitoring Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI (Population Stability Index) | > 0.1 | Data drift alert |
| KS Statistic | > 0.1 | Prediction drift alert |
| AUC-ROC | < 0.75 | Performance degradation alert |
| Calibration Error | > 0.1 | Probability reliability alert |

---

## Deployment Architecture

```mermaid
flowchart TB
    subgraph Development
        CODE[Source Code] --> TEST[Testing]
        TEST --> BUILD[Docker Build]
    end
    
    subgraph Registry
        BUILD --> IMG[Container Registry]
        MODELS[Model Artifacts] --> REG[Model Registry]
    end
    
    subgraph Production
        IMG --> API_CONT[API Container]
        IMG --> DASH_CONT[Dashboard Container]
        REG --> API_CONT
    end
    
    subgraph Infrastructure
        API_CONT --> NGINX[Reverse Proxy]
        DASH_CONT --> NGINX
        NGINX --> LB[Load Balancer]
    end
```

---

## File Structure

```
churn_predictor/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI application
│   │   └── schemas.py        # Pydantic models
│   ├── data/
│   │   ├── ingestion.py      # Data loading
│   │   ├── preprocessing.py  # Data cleaning
│   │   └── validation.py     # Schema validation
│   ├── models/
│   │   ├── feature_engineering.py
│   │   ├── train_pipeline.py
│   │   └── evaluation.py
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── business_insights.py
│   └── monitoring/
│       └── monitoring_service.py
├── dashboard/
│   ├── app.py                # Main Streamlit app
│   └── pages/                # Dashboard pages
├── models/                   # Saved model artifacts
├── tests/                    # Test suite
└── docs/                     # Documentation
```
