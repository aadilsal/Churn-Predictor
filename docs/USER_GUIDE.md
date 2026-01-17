# User Guide

A comprehensive guide for using the Customer Churn Intelligence Platform.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard Navigation](#dashboard-navigation)
3. [Making Predictions](#making-predictions)
4. [Understanding Explanations](#understanding-explanations)
5. [Monitoring Model Health](#monitoring-model-health)
6. [Business Workflows](#business-workflows)

---

## Getting Started

### Accessing the Platform

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | http://localhost:8501 | Interactive analysis |
| API Docs | http://localhost:8000/docs | API reference |
| Health Check | http://localhost:8000/health | System status |

### Starting Services

```bash
# Start API (Terminal 1)
uvicorn src.api.main:app --reload --port 8000

# Start Dashboard (Terminal 2)
streamlit run dashboard/app.py
```

---

## Dashboard Navigation

### Overview Page
The main dashboard provides:
- **Churn Rate Trend**: Historical churn patterns
- **Risk Distribution**: Customer risk breakdown
- **Key Metrics**: Total customers, at-risk count, predicted churners

### Risk Analysis Page
Analyze customer risk segments:
- Filter by risk level (High, Medium, Low)
- View aggregated statistics per segment
- Identify common churn drivers

### Customer View Page
Individual customer analysis:
1. Search for a customer by ID
2. View prediction and risk level
3. See risk and protective factors
4. Get tailored recommendations

### Model Health Page
Monitor system performance:
- Current model accuracy
- Drift detection status
- Prediction distribution
- Calibration quality

### What-If Simulator
Test intervention scenarios:
1. Select a customer
2. Modify attributes (contract, tenure, services)
3. See how changes affect churn probability
4. Identify most impactful interventions

---

## Making Predictions

### Via Dashboard

1. Navigate to **Customer View**
2. Enter customer data or upload CSV
3. Click **Predict**
4. View results with risk level and recommendations

### Via API (Single Customer)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "C001",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.00,
    "TotalCharges": 1320.00
  }'
```

### Via API (Batch)

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'
```

---

## Understanding Explanations

### Risk Levels

| Level | Probability | Action Priority |
|-------|-------------|-----------------|
| **Low** | < 40% | Standard monitoring |
| **Medium** | 40-70% | Proactive engagement |
| **High** | > 70% | Immediate intervention |

### Risk Factors

Risk factors increase churn probability:
- **Month-to-Month Contract**: Highest impact
- **Low Tenure**: New customers at risk
- **Fiber Optic + No Security**: Service gaps
- **Electronic Check Payment**: Payment friction

### Protective Factors

Protective factors decrease churn probability:
- **Long-Term Contract**: Strong commitment
- **Multiple Services**: Higher engagement
- **Automatic Payment**: Reduced friction
- **High Tenure**: Established loyalty

### Recommendations

Each prediction includes actionable recommendations:
- **HIGH Priority**: Address immediately (contract offers)
- **MEDIUM Priority**: Near-term engagement (service bundles)
- **LOW Priority**: Background optimization (payment methods)

---

## Monitoring Model Health

### Dashboard Metrics

| Metric | Healthy Range | Meaning |
|--------|---------------|---------|
| AUC-ROC | > 0.80 | Model discrimination |
| Calibration Error | < 0.05 | Probability accuracy |
| Data Drift (PSI) | < 0.1 | Feature stability |
| Prediction Drift | < 0.1 | Output stability |

### Health Check API

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-01-17T23:30:00"
}
```

### Interpreting Drift Alerts

| Alert | Cause | Action |
|-------|-------|--------|
| Data Drift | Customer population changed | Review feature distributions |
| Prediction Drift | Model behavior shifted | Evaluate recent predictions |
| Performance Drop | Accuracy decreased | Consider retraining |

---

## Business Workflows

### Weekly Churn Review

1. **Open Dashboard** → Overview page
2. **Review** high-risk customer count
3. **Export** high-risk customer list
4. **Assign** customers to retention team
5. **Track** intervention outcomes

### Customer Retention Campaign

1. **Filter** customers by risk level > 60%
2. **Segment** by primary risk factor
3. **Design** targeted offers per segment
4. **Execute** campaign
5. **Measure** churn rate changes

### New Customer Onboarding

1. **Score** new customer at signup
2. **Identify** early risk indicators
3. **Trigger** proactive engagement for high-risk
4. **Monitor** 30/60/90 day retention

### Monthly Model Review

1. **Check** Model Health dashboard
2. **Review** drift metrics
3. **Validate** prediction accuracy on recent data
4. **Decide** if retraining needed
5. **Document** findings

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Dashboard blank | Clear browser cache, restart Streamlit |
| API timeout | Check model is loaded at `/health` |
| Predictions missing | Verify all required fields provided |
| Drift alerts | Review recent data quality |

### Getting Help

- **API Errors**: Check response `detail` field
- **Dashboard Issues**: Check terminal for Streamlit errors
- **Model Questions**: Review `docs/data_dictionary.md`
