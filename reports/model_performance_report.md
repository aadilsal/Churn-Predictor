# Model Performance Report

## Summary

**Selected Model:** xgboost

## Model Comparison

| model    |   roc_auc |   pr_auc |   f1_score |   precision |   recall |   brier_score |   expected_calibration_error |
|:---------|----------:|---------:|-----------:|------------:|---------:|--------------:|-----------------------------:|
| baseline |  0.842869 | 0.634926 |   0.613445 |    0.50519  | 0.780749 |      0.167464 |                     0.146963 |
| xgboost  |  0.844433 | 0.652873 |   0.625    |    0.517544 | 0.78877  |      0.162314 |                     0.150299 |

## Cross-Validation Stability

- **baseline**: ROC-AUC = 0.8463 ± 0.0071
- **xgboost**: ROC-AUC = 0.8476 ± 0.0090

## Selection Rationale

- Score difference vs baseline: 0.9000
