# Data Dictionary

## Dataset Overview

- **Name:** Telco Customer Churn
- **Source:** IBM GitHub Repository
- **Total Records:** 7,043
- **Total Features:** 21
- **Target Variable:** Churn (binary)

---

## Feature Catalog

### Identifier

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `customerID` | String | Unique customer identifier | "7590-VHVEG" | Not used in modeling |

---

### Demographics (4 features)

| Column | Type | Description | Values | Business Meaning |
|--------|------|-------------|--------|------------------|
| `gender` | Categorical | Customer gender | Male, Female | Demographic segmentation |
| `SeniorCitizen` | Binary | Senior citizen status | Yes, No | Age-based risk factor |
| `Partner` | Binary | Has partner | Yes, No | Household stability indicator |
| `Dependents` | Binary | Has dependents | Yes, No | Family commitment indicator |

**Churn Insights:**
- Senior citizens: 41.7% churn rate
- No partner: 33.0% churn rate
- No dependents: 31.3% churn rate

---

### Account Information (2 features)

| Column | Type | Range | Description | Business Meaning |
|--------|------|-------|-------------|------------------|
| `tenure` | Integer | 0-72 | Months with company | Customer lifetime, loyalty indicator |
| `PhoneService` | Binary | Yes, No | Has phone service | Service adoption |

**Churn Insights:**
- Avg tenure (churned): 17.9 months
- Avg tenure (retained): 37.6 months
- First 12 months are highest risk

---

### Internet Services (7 features)

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `InternetService` | Categorical | DSL, Fiber optic, No | Type of internet service |
| `OnlineSecurity` | Binary | Yes, No | Has online security add-on |
| `OnlineBackup` | Binary | Yes, No | Has online backup service |
| `DeviceProtection` | Binary | Yes, No | Has device protection plan |
| `TechSupport` | Binary | Yes, No | Has tech support service |
| `StreamingTV` | Binary | Yes, No | Has TV streaming service |
| `StreamingMovies` | Binary | Yes, No | Has movie streaming service |

**Churn Insights:**
- Fiber optic: 41.9% churn (highest)
- No online security: 41.8% churn
- No tech support: 41.7% churn
- Value-added services reduce churn significantly

---

### Phone Services (1 feature)

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `MultipleLines` | Binary | Yes, No | Has multiple phone lines |

---

### Billing & Contract (4 features)

| Column | Type | Values/Range | Description | Business Meaning |
|--------|------|--------------|-------------|------------------|
| `Contract` | Categorical | Month-to-month, One year, Two year | Contract term | Commitment level |
| `PaperlessBilling` | Binary | Yes, No | Uses paperless billing | Digital engagement |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Payment method | Payment reliability |
| `MonthlyCharges` | Float | $18.25 - $118.75 | Monthly bill amount | Revenue per customer |
| `TotalCharges` | Float | $18.80 - $8,684.80 | Total amount charged | Customer lifetime value |

**Churn Insights:**
- Month-to-month: 42.7% churn
- One year: 11.3% churn
- Two year: 2.8% churn
- Electronic check: 45.3% churn
- Automatic payment: <20% churn

---

### Target Variable

| Column | Type | Values | Description | Distribution |
|--------|------|--------|-------------|--------------|
| `Churn` | Binary | 0 (No), 1 (Yes) | Customer churned | 73% retained, 27% churned |

---

## Data Preprocessing Notes

### TotalCharges
- **Issue:** 11 records had empty strings
- **Resolution:** Imputed using `MonthlyCharges * tenure`
- **Validation:** Zero tenure customers set to $0

### SeniorCitizen
- **Original:** 0/1 integer
- **Transformed:** Yes/No string for consistency

### Service Columns
- **Original:** "No internet service" or "No phone service"
- **Transformed:** Standardized to "No"
- **Rationale:** Simplifies modeling, maintains business meaning

### Churn
- **Original:** "Yes"/"No" string
- **Transformed:** 1/0 binary integer
- **Purpose:** Model-ready format

---

## Feature Engineering Opportunities

### Behavioral Features (Module 2)
- `service_count`: Number of services subscribed
- `has_premium_services`: Online security OR tech support
- `is_streaming_customer`: StreamingTV OR StreamingMovies
- `payment_reliability`: Automatic payment methods

### Temporal Features (Module 2)
- `tenure_group`: 0-12, 13-24, 25-48, 49+ months
- `is_new_customer`: tenure < 12 months
- `customer_lifetime_value`: TotalCharges / tenure

### Derived Features (Module 2)
- `monthly_to_total_ratio`: MonthlyCharges / TotalCharges
- `service_to_charge_ratio`: service_count / MonthlyCharges
- `contract_value`: Contract type encoded by churn risk

---

## Missing Values

### After Preprocessing
- ✅ **Zero missing values**
- All columns have complete data

### Original Issues (Resolved)
- `TotalCharges`: 11 missing (0.16%) → Imputed

---

## Outliers

### Identified (3*IQR method)
- `MonthlyCharges`: 0 outliers
- `TotalCharges`: 0 outliers
- `tenure`: 0 outliers

**Conclusion:** No extreme outliers detected

---

## Categorical Value Distributions

### Contract
- Month-to-month: 55.0%
- Two year: 24.1%
- One year: 20.9%

### InternetService
- Fiber optic: 43.9%
- DSL: 34.4%
- No: 21.7%

### PaymentMethod
- Electronic check: 33.6%
- Mailed check: 22.9%
- Bank transfer: 21.9%
- Credit card: 21.6%

---

## Data Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Records | 7,043 | ✅ |
| Missing Values | 0 | ✅ |
| Duplicate Records | 0 | ✅ |
| Data Type Errors | 0 | ✅ |
| Outliers | 0 | ✅ |
| Class Balance | 73/27 | ⚠️ Imbalanced |

---

## Modeling Considerations

### Strengths
- ✅ Clean, complete data
- ✅ Multiple predictive features
- ✅ Clear business interpretation
- ✅ Sufficient sample size

### Challenges
- ⚠️ Class imbalance (27% churn)
- ⚠️ No temporal data for time-to-churn
- ⚠️ Categorical features require encoding

### Recommendations
1. Use SMOTE or class weights for imbalance
2. Use tenure as proxy for time-to-churn
3. Apply target encoding for high-cardinality categoricals
4. Monitor for overfitting on small churn class

---

## Business Glossary

| Term | Definition |
|------|------------|
| **Churn** | Customer cancels service |
| **Tenure** | Length of customer relationship |
| **Contract** | Service agreement term |
| **Monthly Charges** | Recurring monthly bill |
| **Total Charges** | Cumulative revenue from customer |
| **Value-Added Services** | Security, backup, support services |
| **Automatic Payment** | Bank transfer or credit card |

---

**Last Updated:** 2026-01-14
**Module:** 1 - Data Foundation
