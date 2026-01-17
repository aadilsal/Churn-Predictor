# Video Demo Script

**Duration:** 5-8 minutes  
**Target Audience:** Technical decision-makers, data science leaders, potential clients  
**Tone:** Professional, confident, focused on business value

---

## Opening (0:00 - 0:30)

**[Screen: Title card with project name]**

> "Customer churn costs businesses 5 to 25 times more than customer acquisition. Today, I'll show you a production-grade ML system that not only predicts which customers will churn—but explains *why* and recommends *what to do about it*."

**[Screen: Dashboard overview]**

> "This isn't a Kaggle notebook. This is an enterprise-ready churn intelligence platform designed for real business deployment."

---

## The Problem (0:30 - 1:00)

**[Screen: Business context slide]**

> "Most churn prediction projects stop at 'here's a probability score.' But business stakeholders need more:
> - *Who* is at risk?
> - *Why* are they at risk?
> - *When* will they churn?
> - *What* should we do about it?
> 
> This system answers all of those questions."

---

## Live Dashboard Demo (1:00 - 3:30)

### Overview Page (1:00 - 1:45)

**[Screen: Dashboard main page]**

> "Here's the executive dashboard. At a glance, you see:
> - Current churn rate and trend
> - High-risk customer count
> - Revenue at risk
> 
> This isn't just analytics—it's actionable intelligence."

**[Click through risk distribution chart]**

> "We segment customers into Low, Medium, and High risk categories. Right now, we have X customers in the high-risk zone requiring immediate attention."

### Customer Analysis (1:45 - 2:30)

**[Screen: Customer detail view]**

> "Let's look at an individual customer. I'll search for customer ID..."

**[Enter customer ID]**

> "Here's the prediction: 72% churn probability, High risk level."

**[Scroll to explanation section]**

> "But here's where it gets interesting. The system shows:
> - **Risk factors**: Month-to-month contract, low tenure, electronic check payment
> - **Protective factors**: None for this customer
> - **Recommendations**: Offer annual contract with discount—this is the #1 lever"

### What-If Simulator (2:30 - 3:30)

**[Screen: What-if analysis tool]**

> "What if we could offer this customer an annual contract? Let's simulate that."

**[Change contract type]**

> "Watch the probability drop from 72% to 45%—a 27-point reduction from one intervention. That's the power of explainable AI driving business decisions."

---

## API Demonstration (3:30 - 4:30)

**[Screen: API documentation (Swagger UI)]**

> "For integration into your systems, everything is available via REST API."

**[Show /predict endpoint]**

> "The predict endpoint takes customer data and returns:
> - Calibrated probability
> - Risk level classification
> - Key churn drivers
> - Recommended actions"

**[Execute API call]**

> "Response time: under 100 milliseconds. Batch endpoint handles 1000 customers simultaneously."

**[Show /explain endpoint]**

> "For deeper analysis, the explain endpoint provides SHAP-based interpretation—the same explainability you saw in the dashboard, now programmable."

---

## Monitoring & Reliability (4:30 - 5:30)

**[Screen: Model health dashboard]**

> "Production ML systems need monitoring. This platform includes:
> - **Drift detection**: Alerts when customer patterns change
> - **Performance tracking**: Continuous accuracy monitoring
> - **Calibration checks**: Ensuring probabilities remain trustworthy"

**[Show metrics]**

> "Current model performance: 84% AUC-ROC, 4% calibration error. The system flags when these degrade and triggers retraining workflows."

---

## Technical Highlights (5:30 - 6:30)

**[Screen: Architecture diagram]**

> "Under the hood:
> - **XGBoost** for classification with probability calibration
> - **SHAP** for explainability—translated into business language
> - **FastAPI** backend with sub-100ms latency
> - **Streamlit** dashboards for stakeholder interaction
> - **150+ automated tests** with 80% coverage
> - **MLflow** for experiment tracking and model versioning"

**[Screen: Test results]**

> "This is production code, not research code. Every component is tested, documented, and deployment-ready."

---

## Business Impact (6:30 - 7:30)

**[Screen: ROI calculations]**

> "Let's talk numbers:
> - Identify 72% of churners in the top 20% of predictions
> - $36,000 saved per 1,000 customers at $1,000 average value
> - 90% reduction in manual analysis time"

**[Screen: Use case summary]**

> "This system is ideal for:
> - Telecom providers
> - SaaS companies
> - Subscription businesses
> - Financial services"

---

## Closing (7:30 - 8:00)

**[Screen: Summary slide]**

> "To recap—this isn't just a model. It's a complete churn intelligence platform:
> 
> ✅ Prediction with calibrated probabilities  
> ✅ Explainability in business terms  
> ✅ Actionable recommendations  
> ✅ Production-ready infrastructure  
> ✅ Monitoring and drift detection  
> 
> Built with production ML engineering principles, not Kaggle shortcuts."

**[Screen: Contact/Next steps]**

> "Ready to reduce churn and protect revenue? Let's talk."

---

## Production Notes

### B-Roll Suggestions
- Dashboard interactions (clicking, filtering)
- API calls executing
- Code editor showing test suite
- Architecture diagrams

### Screen Recording Tips
- Use 1920x1080 resolution
- Enable dark mode for better contrast
- Pre-load all pages to avoid waiting
- Have sample customer IDs ready
- Test API calls before recording

### Equipment
- Screen recording: OBS Studio or Loom
- Microphone: USB condenser for clarity
- Edit: DaVinci Resolve or Premiere Pro
