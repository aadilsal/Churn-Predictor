"""Comprehensive model evaluation framework for churn prediction."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging import logger
from src.utils.metrics import (
    calculate_calibration_error,
    calculate_classification_metrics,
    precision_at_k,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name for logging
        
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Base classification metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    
    # Additional discrimination metrics
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    
    # Precision at different k values
    for k in [0.05, 0.1, 0.2, 0.3]:
        metrics[f"precision_at_{int(k*100)}"] = precision_at_k(y_true, y_proba, k=k)
        
    # Probability calibration
    metrics["brier_score"] = brier_score_loss(y_true, y_proba)
    ece, _, _ = calculate_calibration_error(y_true, y_proba)
    metrics["expected_calibration_error"] = ece
    
    # Log key metrics
    logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}, "
                f"PR-AUC: {metrics['pr_auc']:.4f}, "
                f"Brier: {metrics['brier_score']:.4f}")
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Tuple[float, Dict[str, float]]:
    """Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        method: Optimization method ('f1', 'youden', 'cost')
        cost_fp: Cost of false positive (for cost method)
        cost_fn: Cost of false negative (for cost method)
        
    Returns:
        Tuple of (optimal threshold, metrics at threshold)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = -np.inf
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if method == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif method == "youden":
            # Youden's J statistic: Sensitivity + Specificity - 1
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif method == "cost":
            # Minimize total cost (negate for maximization)
            total_cost = cost_fp * fp + cost_fn * fn
            score = -total_cost
        else:
            raise ValueError(f"Unknown method: {method}")
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
            
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    metrics = calculate_classification_metrics(y_true, y_pred_optimal, y_proba)
    metrics["threshold"] = best_threshold
    metrics["optimization_method"] = method
    
    logger.info(f"Optimal threshold ({method}): {best_threshold:.3f}")
    
    return best_threshold, metrics


def analyze_threshold_tradeoffs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Analyze metrics across different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        DataFrame with metrics at each threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "flagged_rate": (tp + fp) / len(y_true),
        })
        
    return pd.DataFrame(results)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Model name for legend
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Model name for legend
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    
    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap_score:.3f})")
    
    # Baseline (proportion of positives)
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot probability calibration curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Model name for legend
        n_bins: Number of calibration bins
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    
    ax.plot(prob_pred, prob_true, "s-", linewidth=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    cm = confusion_matrix(y_true, y_pred)
    
    im = ax.imshow(cm, cmap="Blues")
    
    labels = ["No Churn", "Churn"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=14)
            
    plt.colorbar(im, ax=ax)
    
    return ax


def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comprehensive evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Model name for titles
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} Evaluation", fontsize=14)
    
    plot_roc_curve(y_true, y_proba, model_name, axes[0, 0])
    plot_precision_recall_curve(y_true, y_proba, model_name, axes[0, 1])
    plot_calibration_curve(y_true, y_proba, model_name, ax=axes[1, 0])
    plot_confusion_matrix(y_true, y_pred, axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved evaluation plots to {save_path}")
        
    return fig


def save_evaluation_results(
    metrics: Dict[str, Any],
    path: Path,
    model_name: str = "model",
) -> None:
    """Save evaluation results to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary
        path: Path to save results
        model_name: Model name
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v
            
    with open(path, "w") as f:
        json.dump({model_name: clean_metrics}, f, indent=2)
        
    logger.info(f"Saved evaluation results to {path}")
