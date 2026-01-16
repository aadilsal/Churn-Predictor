# Model Selection Justification

This document provides business-focused justification for the selected churn prediction model.

## Executive Summary

*To be populated after training pipeline execution.*

## Model Candidates

### 1. Logistic Regression (Baseline)

**Pros:**
- Highly interpretable - coefficients directly show feature impact
- Well-calibrated probability outputs
- Fast training and inference
- Strong regularization prevents overfitting

**Cons:**
- Cannot capture non-linear relationships
- May underperform on complex interaction patterns

### 2. XGBoost

**Pros:**
- Handles non-linear relationships and feature interactions
- Built-in handling of class imbalance
- Feature importance for business insights
- Generally higher predictive performance

**Cons:**
- Less interpretable (black box)
- More hyperparameters to tune
- Marginally slower inference

## Selection Criteria

Models were evaluated on:

1. **Discrimination** (40%): Ability to separate churners from non-churners
   - ROC-AUC, PR-AUC

2. **Calibration** (20%): Accuracy of probability estimates
   - Brier Score, Expected Calibration Error

3. **Business Relevance** (25%): Performance on high-risk customers
   - Precision@10%, Precision@20%, Recall

4. **Stability** (15%): Consistency across cross-validation folds
   - Standard deviation of metrics

## Final Selection

*Model name and rationale to be filled after training.*

---

## Threshold Selection

**Chosen Threshold:** *To be determined*

**Rationale:**
- Balances precision and recall for retention campaigns
- Optimizes F1-score for practical deployment
- Considers business cost of false positives vs false negatives

## Known Risks & Assumptions

1. **Data Drift**: Model assumes future customers behave similarly to training data
2. **Class Imbalance**: ~26% churn rate in training data
3. **Feature Availability**: Assumes all features available at inference time
4. **Temporal Validity**: No temporal leakage validation performed

## Readiness Assessment

| Factor | Status | Notes |
|--------|--------|-------|
| Model trained | ⏳ | Run train_pipeline.py |
| Evaluation complete | ⏳ | Pending |
| Threshold optimized | ⏳ | Pending |
| Artifacts exported | ⏳ | Pending |
| Ready for explainability | ⏳ | Pending model selection |

## Next Steps

1. Proceed to Module 4: Model Explainability (SHAP analysis)
2. Implement monitoring for production deployment
3. Establish retraining schedule

---

*Document auto-generated. Updated after each training run.*
