"""Custom evaluation metrics for churn prediction."""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # Specificity
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def precision_at_k(
    y_true: np.ndarray, y_pred_proba: np.ndarray, k: float = 0.1
) -> float:
    """Calculate precision at top k% of predictions.

    This is critical for churn prediction where we want to identify
    the highest-risk customers accurately.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        k: Percentage of top predictions to consider (0-1)

    Returns:
        Precision at k
    """
    n = len(y_true)
    n_top_k = max(1, int(n * k))

    # Get indices of top k predictions
    top_k_indices = np.argsort(y_pred_proba)[-n_top_k:]

    # Calculate precision on top k
    y_true_top_k = y_true[top_k_indices]
    precision = np.sum(y_true_top_k) / n_top_k

    return precision


def calculate_calibration_error(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate Expected Calibration Error (ECE).

    A well-calibrated model should have predicted probabilities
    that match actual outcomes.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (ECE, bin_accuracies, bin_confidences)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)

    return ece, np.array(bin_accuracies), np.array(bin_confidences)


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    revenue_per_customer: float = 1000.0,
    intervention_cost: float = 50.0,
    intervention_success_rate: float = 0.3,
) -> Dict[str, float]:
    """Calculate business-oriented metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        revenue_per_customer: Average annual revenue per customer
        intervention_cost: Cost of retention intervention
        intervention_success_rate: Success rate of interventions

    Returns:
        Dictionary of business metrics
    """
    # Total revenue at risk
    total_churners = np.sum(y_true)
    revenue_at_risk = total_churners * revenue_per_customer

    # If we intervene on top 20% highest risk
    top_20_pct = precision_at_k(y_true, y_pred_proba, k=0.2)
    n_interventions = int(len(y_true) * 0.2)

    # Expected customers saved
    customers_saved = n_interventions * top_20_pct * intervention_success_rate

    # ROI calculation
    revenue_saved = customers_saved * revenue_per_customer
    intervention_costs = n_interventions * intervention_cost
    net_benefit = revenue_saved - intervention_costs
    roi = (net_benefit / intervention_costs * 100) if intervention_costs > 0 else 0

    return {
        "revenue_at_risk": revenue_at_risk,
        "customers_saved": customers_saved,
        "revenue_saved": revenue_saved,
        "intervention_costs": intervention_costs,
        "net_benefit": net_benefit,
        "roi_percentage": roi,
    }
